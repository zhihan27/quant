"""股票池范围处理与候选横截面排序。"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd

from config import settings
from services.market_data import MarketDataError, MarketDataService
from services.rank_model import SCORE_KIND_RANK
from services.universe import filter_universe, load_reference

LOGGER = logging.getLogger("quant.scanning")


def load_universe_reference(data_service: MarketDataService, progress: Any = None) -> tuple[Any, dict[str, Any]]:
    """读取退市与 ST 名单；接口不可用时降级为证券简称启发式，并如实说明降级原因。"""
    try:
        return load_reference(getattr(data_service, "panels", None), progress=progress)
    except (OSError, ValueError, ImportError) as exc:
        LOGGER.warning("股票池状态名单不可用 detail=%s", exc)
        return None, {"status": f"股票池状态名单不可用，改用证券简称启发式：{type(exc).__name__}"}


def _apply_universe_filter(
    frame: pd.DataFrame, reference: Any, meta: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """统一入口：所有扫描模式（含用户自定义代码）都必须先过股票池状态过滤。"""
    kept, report = filter_universe(frame, reference)
    merged = {**meta, "universe_filter": report}
    if report.get("excluded"):
        merged["warning"] = "；".join(filter(None, [meta.get("warning"), report["status"]]))
    return kept, merged, report


def candidate_codes(
    mode: str, raw_codes: str, limit: int, data_service: MarketDataService, progress: Any = None
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if mode == "用户自定义":
        codes = list(dict.fromkeys(re.findall(r"(?<!\d)\d{6}(?!\d)", raw_codes)))
        frame = pd.DataFrame({"code": codes, "name": codes})
        reference, reference_meta = load_universe_reference(data_service, progress)
        # 名单信息单独保存，不参与报价来源、时间和有效状态的判断。
        frame, meta, report = _apply_universe_filter(frame, reference, {
            "source": "用户输入", "status": "用户输入", "fetched_at": None, "complete": True,
            "universe_reference": reference_meta,
        })
        if report.get("excluded"):
            meta["error"] = f"输入的代码中有 {report['excluded']} 只被股票池规则排除：{report['status']}"
        return frame.to_dict("records"), meta
    try:
        scope = {"沪深主板": "main", "创业板": "chinext", "科创板": "star", "北交所": "bj"}.get(mode, "all")
        universe, meta = data_service.get_universe(scope=scope)
    except MarketDataError as exc:
        return [], {"status": "不可用", "error": f"股票列表获取失败：{exc}。可切换为用户自定义代码。"}
    if universe.empty or not meta.get("complete"):
        return [], {"status": "不可用", "error": "股票列表为空。"}
    universe["code"] = universe["code"].astype(str).str.zfill(6)
    universe["name"] = universe["name"].astype(str)
    chinext_prefixes = settings.BOARD_PREFIXES["chinext"]
    star_prefixes = settings.BOARD_PREFIXES["star"]
    if mode == "沪深主板":
        universe = universe.loc[
            universe["code"].str.match(r"^(00[0-3]|60[0-5])")
            & ~universe["code"].str.startswith((*chinext_prefixes, *star_prefixes))
        ]
    elif mode == "创业板":
        universe = universe.loc[universe["code"].str.startswith(chinext_prefixes)]
    elif mode == "科创板":
        universe = universe.loc[universe["code"].str.startswith(star_prefixes)]
    elif mode == "北交所":
        universe = universe.loc[universe["code"].str.startswith(settings.BOARD_PREFIXES["bj"])]
    reference, reference_meta = load_universe_reference(data_service, progress)
    universe, meta, _ = _apply_universe_filter(universe, reference, {**meta, "universe_reference": reference_meta})
    meta["scope_valid_symbols"] = int(len(universe))
    if universe.empty:
        meta["error"] = "股票池经退市与 ST 过滤后为空。"
        return [], meta
    if "amount_yuan" in universe:
        universe = universe.sort_values("amount_yuan", ascending=False, na_position="last")
    rows = universe.to_dict("records")
    for row in rows:
        row["quote"] = data_service.quote_from_universe_row(row, meta)
    meta = {**meta, "scope_valid_symbols": len(rows), "display_limit": int(limit)}
    return rows, meta


def rank_candidates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按交易日汇总因子与已验证模型百分位，返回可追溯的候选排序。"""
    valid_records = [item for item in records if not item.get("error")]
    if not valid_records:
        return records
    factor_columns = ["momentum_20", "momentum_60", "trend_strength", "relative_strength_20", "volatility_20"]
    for item in valid_records:
        item["rank_parts"] = {}
    dates = pd.Series({index: str(item.get("analysis", {}).get("prediction_reference_date", "未提供")) for index, item in enumerate(valid_records)})
    for column in factor_columns:
        values = pd.Series({index: item["features"].get(column) for index, item in enumerate(valid_records)}, dtype=float)
        valid_values = values.dropna()
        if valid_values.empty:
            continue
        percentile = valid_values.groupby(dates).rank(pct=True)
        percentile = percentile.where(valid_values.groupby(dates).transform("count").ge(5))
        if column == "volatility_20":
            percentile = 1 - percentile
        for index, score in percentile.items():
            if pd.notna(score):
                valid_records[index]["rank_parts"][column] = float(score)
    for item in valid_records:
        parts = list(item["rank_parts"].values())
        item["cross_section_score"] = float(np.mean(parts) * 100) if parts else None
        features = item["features"]
        reasons = []
        if features.get("momentum_20") is not None and features["momentum_20"] > 0:
            reasons.append("20 日动量为正")
        if features.get("trend_strength") is not None and features["trend_strength"] > 0:
            reasons.append("20/60 日均线趋势向上")
        item["reason"] = "；".join(reasons) if reasons else "由候选池因子分数排序，趋势条件未全部满足"
        risk_parts = ["超额为正不代表股票绝对上涨；当前股票池回溯存在幸存者偏差"]
        if item.get("entry_trade_status") and item["entry_trade_status"] != "可正常评估":
            risk_parts.append(item["entry_trade_status"])
        item["risk"] = "；".join(risk_parts)
    for model_name, rank_key in (("LightGBM", "lightgbm_rank"), ("XGBoost", "xgboost_rank")):
        ranked = []
        for item in valid_records:
            model = next((value for value in item["analysis"]["models"].get(20, []) if value["model"] == model_name), {})
            if model.get("validated") and model.get("target_kind") == "excess_return_vs_csi300" and model.get("prediction_zscore") is not None:
                ranked.append((item, model["prediction_zscore"]))
        for item, _ in ranked:
            model = next(value for value in item["analysis"]["models"][20] if value["model"] == model_name)
            item[rank_key] = model.get("model_rank")
    # 总分采用固定权重；没有通过验证的模型时仅展示因子分，不混入模型推荐。
    for item in valid_records:
        item["factor_cross_section_score"] = item["cross_section_score"]
        models = item["analysis"]["models"].get(20, [])
        validated = [
            model for model in models
            if model.get("validated") and model.get("prediction_zscore") is not None
            and model.get("target_kind") == "excess_return_vs_csi300"
            and model.get("prediction_percentile") is not None and np.isfinite(model["prediction_zscore"])
        ]
        item["ranking_eligible"] = bool(validated) and item["factor_cross_section_score"] is not None
        model_scores = [100 * model["prediction_percentile"] for model in validated]
        item["model_score"] = float(np.mean(model_scores)) if model_scores else None
        item["cross_section_score"] = (
            0.4 * item["factor_cross_section_score"] + 0.6 * item["model_score"]
            if item["ranking_eligible"] else None
        )
        if not item["ranking_eligible"]:
            item["reason"] = "模型未通过验证，仅展示因子观察结果"
    return sorted(
        records,
        key=lambda item: (bool(item.get("ranking_eligible")), item.get("cross_section_score") or -1),
        reverse=True,
    )


def validated_return(model: dict[str, Any]) -> float | str:
    """仅显示通过样本外验证的预测，否则返回拒绝原因。

    排序目标的输出是无量纲的相对排序分，不能当作预期超额收益展示，这里统一换算为同日截面
    百分位；只有回归目标才给出有量纲的预期超额收益。
    """
    if not model.get("validated"):
        return model.get("status", "模型未训练")
    if model.get("score_kind") == SCORE_KIND_RANK:
        percentile = model.get("prediction_percentile")
        return percentile if percentile is not None else "排序分缺少同日截面百分位"
    if model.get("predicted_excess_return") is not None:
        return model["predicted_excess_return"]
    return model.get("status", "模型未训练")

