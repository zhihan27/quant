"""个股风格因子观察与模型血缘核对，不用当前快照回填训练历史。"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from config import settings
from services.style_factors import RAW_COLUMNS, SIZE_COLUMNS, VALUE_COLUMNS, align_history, calculate_style_values

LOGGER = logging.getLogger("quant.stock_style")
FACTOR_LABELS = {
    "log_market_cap_total": "总市值对数",
    "log_market_cap_float": "流通市值对数",
    "earnings_yield": "盈利收益率 E/P",
    "book_to_price": "账面市值比 B/P",
    "sales_to_price": "销售市值比 S/P",
}


def _number(value: Any) -> float | None:
    """过滤接口占位符、布尔值与无穷值，缺失不得变成零。"""
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _is_verified(value: Any) -> bool:
    """兼容 pandas / NumPy 布尔标量，并将缺失值保守解释为未核验。"""
    return not pd.isna(value) and bool(value)


def stock_model_features(
    models: dict[int, list[dict[str, Any]]], context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """逐周期核对训练列与本股有效预测，旧模型含某列不代表本股当前可用。"""
    trained = (context or {}).get("models") or {}
    rows = []
    for horizon, results in models.items():
        candidates = trained.get(horizon, trained.get(str(horizon), []))
        by_name = {item.get("model"): item for item in candidates}
        for item in results:
            original = by_name.get(item.get("model"), item)
            features = list(original.get("features_used") or [])
            actual = [f"xs_{name}" for name in FACTOR_LABELS if f"xs_{name}" in features]
            # 必须已有同股同日、同价格血缘核验后的有效预测，不能只检查训练开关。
            has_signal = any(
                _number(item.get(column)) is not None
                for column in ("predicted_excess_return", "prediction_zscore", "prediction_percentile")
            )
            effective = bool(item.get("validated")) and has_signal
            effective_columns = (
                [name for name in actual if name in (item.get("features_used") or [])]
                if effective else []
            )
            rows.append({
                "model": item.get("model"), "horizon": int(horizon),
                "features_used": features, "style_features": actual,
                "effective_features": effective_columns, "prediction_available": effective,
                "status": item.get("status") or "状态未提供",
            })
    return rows


def fundamental_status(models: dict[int, list[dict[str, Any]]]) -> str:
    """根据本股实际模型结果说明生效范围，避免固定文案否认已经启用的特征。"""
    rows = stock_model_features(models)
    effective = {column for row in rows for column in row["effective_features"]}
    if effective:
        labels = [label for name, label in FACTOR_LABELS.items() if f"xs_{name}" in effective]
        return "量价走势分只含量价因子；本股有效截面模型已使用：" + "、".join(labels) + "。"
    return "量价走势分只含量价因子；本股当前没有使用市值、估值特征的有效模型预测，具体原因见风格因子状态。"


def _missing_reason(name: str, row: pd.Series, *, historical: bool) -> str:
    """按血缘、市值与估值分母逐层解释缺失，保留真实亏损信息。"""
    if historical and pd.isna(row.get("style_source")):
        return "无同股同日且当日已可用的历史版本"
    if historical and not _is_verified(row.get("style_size_pit")):
        return "历史股本或市值点时血缘未核验"
    if historical and name in VALUE_COLUMNS and not _is_verified(row.get("style_value_pit")):
        return "历史财报披露或修订版本未核验"
    cap_name = "market_cap_float" if name == "log_market_cap_float" else "market_cap_total"
    cap = _number(row.get(f"style_{cap_name}"))
    if cap is None or cap <= settings.STYLE_MINIMUM_MARKET_CAP:
        return ("流通市值" if cap_name == "market_cap_float" else "总市值") + "缺失或低于有效门槛"
    if name == "earnings_yield":
        return "TTM 利润缺失，且 PE(TTM) 缺失或非正；不对非正 PE 求倒数"
    if name == "book_to_price":
        return "净资产异常或非正，或缺少净资产与有效 PB"
    if name == "sales_to_price":
        return "TTM 营收异常或非正，或缺少营收与有效 PS(TTM)"
    return "原始值无效"


def build_stock_style_analysis(
    code: str, reference_date: str, quote: dict[str, Any],
    models: dict[int, list[dict[str, Any]]], *, history: pd.DataFrame | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """生成五项风格明细及训练状态；仅读取传入数据，不联网、不修改生产配置。"""
    code = str(code).zfill(6)
    snapshot = pd.DataFrame({f"style_{column}": [_number(quote.get(column))] for column in RAW_COLUMNS})
    current_values = calculate_style_values(snapshot, require_pit=False).iloc[0]
    quote_matches = str(quote.get("code", "")).zfill(6) == code
    current_error = None
    if not quote_matches:
        current_error = "快照股票代码不匹配"
    elif quote.get("valuation_unit") != "ratio" or quote.get("market_cap_unit") != "CNY":
        current_error = "估值或市值单位未核验"
    if current_error:
        current_values[:] = np.nan
    aligned = align_history(pd.DataFrame({"code": [code], "date": [reference_date]}), None)
    history_error = None
    if history is not None and not history.empty:
        try:
            aligned = align_history(aligned[["code", "date"]], history)
        except (KeyError, TypeError, ValueError) as exc:
            history_error = f"历史估值校验失败：{exc}"
            LOGGER.warning("个股历史估值校验失败 code=%s detail=%s", code, exc)
    historical = aligned.iloc[0]
    historical_values = calculate_style_values(aligned).iloc[0]
    model_rows = stock_model_features(models, context)
    factors = []
    for name, label in FACTOR_LABELS.items():
        current = _number(current_values[name])
        past = _number(historical_values[name])
        column = f"xs_{name}"
        factors.append({
            "factor": name, "label": label, "current_value": current, "historical_value": past,
            "current_status": current_error or (
                "当前快照观察值" if current is not None
                else _missing_reason(name, snapshot.iloc[0], historical=False)
            ),
            "historical_status": history_error or (
                "已核验点时原始值" if past is not None
                else _missing_reason(name, historical, historical=True)
            ),
            "trained_models": [
                f"{row['model']} / {row['horizon']}日"
                for row in model_rows if column in row["style_features"]
            ],
            "effective_models": [
                f"{row['model']} / {row['horizon']}日"
                for row in model_rows if column in row["effective_features"]
            ],
        })
    report = (context or {}).get("style_factors") or {}
    requested = report.get("requested") or {"size": None, "value": None}
    observed = {column: _number(quote.get(column)) for column in RAW_COLUMNS}
    if current_error:
        observed = {column: None for column in RAW_COLUMNS}
    metadata = {}
    for name in ("source", "available_at", "collected_at", "revision_id", "quality", "unit"):
        value = historical.get(f"style_{name}")
        metadata[name] = None if pd.isna(value) else str(value)
    return {
        "code": code, "reference_date": reference_date, "factors": factors, "raw": observed,
        "quote_source": quote.get("source"), "quote_time": quote.get("quote_time"),
        "collected_at": quote.get("fetched_at"), "quote_status": quote.get("data_status"),
        "quote_valid": bool(quote.get("quote_valid")), "history_metadata": metadata,
        "models": model_rows, "requested": requested,
        "training_status": report.get("status") or "未加载风格特征训练摘要；实际入模以逐模型特征名单为准",
        "fundamental_status": fundamental_status(models),
    }
