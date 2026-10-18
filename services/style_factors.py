"""可追溯的历史市值、估值版本与显式风格特征门控。"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from config import settings
from services.feature_engineering import FEATURE_COLUMNS, normalize_cross_section
from services.panel_store import PanelStore

LOGGER = logging.getLogger("quant.style_factors")
SIZE_COLUMNS = ("log_market_cap_total", "log_market_cap_float")
VALUE_COLUMNS = ("earnings_yield", "book_to_price", "sales_to_price")
RAW_COLUMNS = ("market_cap_total", "market_cap_float", "pe_ttm", "pb", "ps_ttm",
               "net_profit_ttm", "book_equity", "revenue_ttm")
METADATA_COLUMNS = ("source", "collected_at", "available_at", "revision_id", "unit",
                    "quality", "size_pit", "value_pit")


def processing_config(
    objective: str | None = None, training_window: int | None = None, *,
    size: bool | None = None, value: bool | None = None,
    neutralization_enabled: bool = False, base_feature_columns: list[str] | None = None,
    extra_factor_families: list[str] | None = None,
) -> dict[str, Any]:
    """返回影响入模与独立验收有效性的全部处理配置。"""
    return {
        "protocol": settings.STYLE_FEATURE_PROTOCOL,
        "configuration_version": 2,
        "minimum_coverage": float(settings.STYLE_MINIMUM_COVERAGE),
        "winsor_quantiles": list(settings.STYLE_WINSOR_QUANTILES),
        "minimum_market_cap": float(settings.STYLE_MINIMUM_MARKET_CAP),
        "minimum_book_equity": float(settings.STYLE_MINIMUM_BOOK_EQUITY),
        "minimum_revenue": float(settings.STYLE_MINIMUM_REVENUE),
        "base_features_enabled": bool(settings.ENABLE_BASE_FEATURES),
        "base_feature_definitions": list(FEATURE_COLUMNS),
        "base_feature_columns": list(base_feature_columns or []),
        "size_features_enabled": bool(settings.ENABLE_SIZE_FEATURES if size is None else size),
        "value_features_enabled": bool(settings.ENABLE_VALUE_FEATURES if value is None else value),
        "neutralization_enabled": bool(neutralization_enabled),
        "neutralization_minimum_stocks": int(settings.NEUTRALIZATION_MINIMUM_STOCKS),
        "neutralization_minimum_industry_stocks": int(settings.NEUTRALIZATION_MINIMUM_INDUSTRY_STOCKS),
        "extra_factor_minimum_coverage": float(settings.EXTRA_FACTOR_MINIMUM_COVERAGE),
        "extra_factor_families": sorted(str(name) for name in (extra_factor_families or [])),
        "training_window": int(settings.TRAINING_WINDOW_DAYS if training_window is None else training_window),
        "objective": objective or settings.DEFAULT_OBJECTIVE_MODE,
        "label_upper_quantile": float(settings.LABEL_WINSOR_UPPER_QUANTILE),
        "regression_loss": settings.REGRESSION_LOSS,
        "oos_retrain_days": int(settings.OOS_RETRAIN_DAYS),
        "rank_grade_count": int(settings.RANK_GRADE_COUNT),
        "ndcg_top_fraction": float(settings.NDCG_TOP_FRACTION),
        "costs": settings.DEFAULT_COSTS,
    }


def configuration_id(
    objective: str | None = None, training_window: int | None = None, **feature_options: Any,
) -> str:
    """为处理配置生成稳定指纹，规则变更后旧验收自动失效。"""
    payload = json.dumps(processing_config(objective, training_window, **feature_options), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def configuration_id_from_config(configuration: dict[str, Any]) -> str:
    """保留本轮显式选项和实际列，使用当前处理规则重新核验指纹。"""
    return configuration_id(
        configuration.get("objective"), configuration.get("training_window"),
        size=configuration.get("size_features_enabled"), value=configuration.get("value_features_enabled"),
        neutralization_enabled=configuration.get("neutralization_enabled", False),
        base_feature_columns=configuration.get("base_feature_columns"),
        extra_factor_families=configuration.get("extra_factor_families"),
    )


def normalize_history(frame: pd.DataFrame) -> pd.DataFrame:
    """校验规范历史长表；金额统一元、股本统一股，不接受无血缘记录。"""
    required = {"code", "date", *METADATA_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("历史估值缺少字段：" + "、".join(sorted(missing)))
    result = frame.copy()
    result["code"] = result["code"].astype(str).str.strip().str.zfill(6)
    if not result["code"].str.fullmatch(r"[0-9]{6}").all():
        raise ValueError("历史估值包含非法股票代码。")
    for column in ("date", "available_at", "collected_at"):
        values = pd.to_datetime(result[column], errors="raise")
        if isinstance(values.dtype, pd.DatetimeTZDtype):
            values = values.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
        result[column] = values
        if result[column].isna().any():
            raise ValueError(f"历史估值 {column} 不能为空。")
    result["date"] = result["date"].dt.normalize()
    if not result["unit"].eq("CNY/share").all():
        raise ValueError("规范历史估值单位必须为 CNY/share（金额元、股本股）。")
    for column in ("source", "revision_id", "quality"):
        if result[column].isna().any() or result[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"历史估值 {column} 不能为空。")
    for column in ("size_pit", "value_pit"):
        if not result[column].isin([True, False]).all():
            raise ValueError(f"历史估值 {column} 必须为布尔值。")
    for column in RAW_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce") if column in result else np.nan
    result = result.replace([np.inf, -np.inf], np.nan)
    if result.duplicated(["code", "date", "available_at", "revision_id"]).any():
        raise ValueError("历史估值存在重复版本。")
    return result.sort_values(["date", "code", "available_at", "revision_id"]).reset_index(drop=True)


def save_history(store: PanelStore, frame: pd.DataFrame) -> dict[str, Any]:
    """追加保存历史版本；同一版本重采不覆盖原始记录，修订不得倒灌。"""
    normalized = normalize_history(frame)
    count = 0
    added = 0
    for code, group in normalized.groupby("code"):
        try:
            existing = store.read_partition("valuation", code)
            previous_rows = len(existing) if existing is not None and not existing.empty else 0
            if existing is not None and not existing.empty:
                # 日指标接口不提供历史修订发布时间；重采出现新版本时只能从采集时点可用。
                previous = existing.loc[existing["source"].eq("Tushare daily_basic")]
                known = set(zip(previous["date"], previous["revision_id"]))
                changed = group["source"].eq("Tushare daily_basic") & group["date"].isin(previous["date"])
                changed &= pd.Series([(date, revision) not in known for date, revision in zip(group["date"], group["revision_id"])], index=group.index)
                if changed.any():
                    group = group.copy()
                    group.loc[changed, "available_at"] = group.loc[changed, ["available_at", "collected_at"]].max(axis=1)
                    group.loc[changed, "quality"] += "；修订可用日按重采时间，禁止倒灌"
                    LOGGER.warning("历史日市值重采发现修订，已隔离过去可用日 code=%s rows=%d", code, int(changed.sum()))
                group = pd.concat([existing, group], ignore_index=True)
                # 来源版本号不可用于悄悄覆盖已经保存的数值。
                group = group.drop_duplicates(["code", "date", "source", "revision_id"], keep="first")
            store.write_partition("valuation", code, group)
            count += len(group)
            added += max(len(group) - previous_rows, 0)
        except (OSError, ValueError, ImportError):
            LOGGER.exception("历史估值版本落盘失败 code=%s", code)
            raise
    return {"codes": int(normalized["code"].nunique()), "stored_rows": count, "new_rows": added}


def load_history(store: PanelStore, codes: list[str]) -> pd.DataFrame:
    """仅读取请求股票的历史估值，损坏分区由存储层记录并显式缺席。"""
    frames = [store.read_partition("valuation", code) for code in codes]
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def align_history(panel: pd.DataFrame, history: pd.DataFrame | None) -> pd.DataFrame:
    """只匹配同股同日、当日晚间已可用的版本；不会用当前快照填充历史。"""
    result = panel.copy()
    if history is None or history.empty:
        for column in (*RAW_COLUMNS, *METADATA_COLUMNS):
            result[f"style_{column}"] = np.nan
        return result
    facts = normalize_history(history)
    facts = facts.loc[facts["available_at"] < facts["date"] + pd.Timedelta(days=1)]
    facts = facts.sort_values(["available_at", "revision_id"]).drop_duplicates(["code", "date"], keep="last")
    selected = facts[["code", "date", *RAW_COLUMNS, *METADATA_COLUMNS]].rename(
        columns={column: f"style_{column}" for column in (*RAW_COLUMNS, *METADATA_COLUMNS)}
    )
    for column in ("industry", "industry_pit"):
        if column in facts:
            selected[f"style_{column}"] = facts[column].to_numpy()
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    result["code"] = result["code"].astype(str).str.zfill(6)
    return result.merge(selected, on=["code", "date"], how="left", validate="many_to_one")


def approval_status(size: bool, value: bool, config_id: str | None = None) -> dict[str, Any]:
    """生产只接受当前版本、规则与特征组合一致的独立验收结果。"""
    variant = "size_value" if size and value else ("size" if size else "value")
    try:
        report = json.loads(settings.STYLE_APPROVAL_PATH.read_text(encoding="utf-8"))
        approved = bool(
            report.get("passed") is True and report.get("variant") == variant
            and report.get("protocol") == settings.STYLE_FEATURE_PROTOCOL
            and report.get("model_version") == settings.MODEL_VERSION
            and report.get("configuration_id") == (config_id or configuration_id(size=size, value=value))
        )
    except (OSError, ValueError, TypeError):
        approved = False
    return {"approved": approved, "variant": variant,
            "reason": "独立验收通过" if approved else "没有匹配当前配置的独立验收，新增特征不进入生产模型"}


def production_report_is_current(report: dict[str, Any] | None) -> bool:
    """读取缓存时重新检查规则与验收，失败后旧风格权重和旧排名同样失效。"""
    if not report:
        return True
    configuration = report.get("config") or {}
    expected = configuration_id_from_config(configuration)
    if report.get("protocol") != settings.STYLE_FEATURE_PROTOCOL or report.get("configuration_id") != expected:
        return False
    if not report.get("accepted_columns"):
        return True
    requested = report.get("requested") or {}
    if (bool(requested.get("size")), bool(requested.get("value"))) != (
        configuration.get("size_features_enabled"), configuration.get("value_features_enabled"),
    ):
        return False
    return approval_status(bool(requested.get("size")), bool(requested.get("value")), expected)["approved"]


def calculate_style_values(frame: pd.DataFrame, *, require_pit: bool = True) -> pd.DataFrame:
    """计算未标准化风格值；当前快照可观察，但历史计算必须核验点时血缘。"""
    result = pd.DataFrame(index=frame.index)
    cap = frame["style_market_cap_total"]
    total_valid = cap.gt(settings.STYLE_MINIMUM_MARKET_CAP)
    size_pit = (
        frame["style_size_pit"].eq(True)
        if require_pit else pd.Series(True, index=frame.index)
    )
    value_pit = (
        frame["style_value_pit"].eq(True) & size_pit
        if require_pit else size_pit
    )
    for column, raw in zip(SIZE_COLUMNS, ("market_cap_total", "market_cap_float")):
        values = frame[f"style_{raw}"]
        result[column] = np.log(values.where(size_pit & values.gt(settings.STYLE_MINIMUM_MARKET_CAP)))
    # 亏损只通过已披露利润表达，不能把非正 PE 的倒数当作可靠盈利收益率。
    profit = frame["style_net_profit_ttm"]
    pe = frame["style_pe_ttm"]
    earnings = (1 / pe.where(pe > 0)).where(
        profit.isna(), profit / cap.where(total_valid)
    )
    result["earnings_yield"] = earnings.where(value_pit & total_valid)
    for column, raw, ratio, minimum in (
        ("book_to_price", "book_equity", "pb", settings.STYLE_MINIMUM_BOOK_EQUITY),
        ("sales_to_price", "revenue_ttm", "ps_ttm", settings.STYLE_MINIMUM_REVENUE),
    ):
        numerator = frame[f"style_{raw}"]
        multiple = frame[f"style_{ratio}"]
        values = (1 / multiple.where(multiple > 0)).where(
            numerator.isna(), numerator / cap.where(total_valid)
        )
        valid = numerator.isna() | numerator.gt(minimum)
        result[column] = values.where(value_pit & total_valid & valid)
    return result.replace([np.inf, -np.inf], np.nan)


def attach_style_factors(
    panel: pd.DataFrame, history: pd.DataFrame | None, *, size: bool | None = None,
    value: bool | None = None, research: bool = False, calibration_end: Any = None,
    objective: str | None = None, training_window: int | None = None,
    neutralization_enabled: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """生成截面风格特征，按校准期覆盖门控；显式特征不对自身中性化。"""
    size = settings.ENABLE_SIZE_FEATURES if size is None else bool(size)
    value = settings.ENABLE_VALUE_FEATURES if value is None else bool(value)
    config = processing_config(objective, training_window, size=size, value=value,
                               neutralization_enabled=neutralization_enabled,
                               base_feature_columns=panel.attrs.get("model_feature_columns", []),
                               extra_factor_families=panel.attrs.get("extra_factor_families", []))
    report = {"protocol": settings.STYLE_FEATURE_PROTOCOL, "config": config,
              "configuration_id": configuration_id_from_config(config), "requested": {"size": size, "value": value},
              "features": {}, "accepted_columns": [], "model_feature_columns": [],
              "sources": [], "status": "显式市值与估值特征均关闭"}
    if not size and not value:
        return panel, report
    approval = approval_status(size, value, report["configuration_id"])
    report["acceptance"] = approval
    if not research and not approval["approved"]:
        report["status"] = approval["reason"]
        return panel, report
    result = align_history(panel, history)
    cap = result["style_market_cap_total"]
    size_pit = result["style_size_pit"].eq(True)
    value_pit = result["style_value_pit"].eq(True) & size_pit
    values = calculate_style_values(result)
    result[list(values.columns)] = values
    columns = [*(SIZE_COLUMNS if size else ()), *(VALUE_COLUMNS if value else ())]
    low, high = settings.STYLE_WINSOR_QUANTILES
    if not 0 <= low < high <= 1 or not 0 < settings.STYLE_MINIMUM_COVERAGE <= 1:
        raise ValueError("风格特征截尾分位或覆盖率门槛无效。")
    for column in columns:
        values = result[column].replace([np.inf, -np.inf], np.nan)
        groups = values.groupby(result["date"])
        result[column] = values.clip(groups.transform("quantile", low), groups.transform("quantile", high))
    result = normalize_cross_section(result, columns, minimum_stocks=5, minimum_coverage=settings.STYLE_MINIMUM_COVERAGE)
    # 只在最早可训练的 80 个截面选列，留出期或未来日期不能改变特征名单。
    calibration = result.loc[result["date"] < pd.Timestamp(calibration_end)] if calibration_end is not None else result
    base = [column for column in panel.attrs.get("model_feature_columns", []) if column in calibration]
    if base:
        eligible = calibration[base].notna().any(axis=1)
        dates = sorted(calibration.loc[eligible, "date"].unique())[:80]
    else:
        dates = sorted(calibration["date"].unique())[:80]
    sample = calibration.loc[calibration["date"].isin(dates)]
    for column in columns:
        transformed = f"xs_{column}"
        coverage = float(sample[transformed].notna().mean()) if len(sample) else 0.0
        accepted = coverage >= settings.STYLE_MINIMUM_COVERAGE
        report["features"][column] = {"calibration_coverage": coverage,
            "missing_ratio": float(result[column].isna().mean()), "accepted": accepted,
            "missing_rows": int(result[column].isna().sum()),
            "reason": "校准期覆盖达标" if accepted else "可靠点时数据或校准期覆盖率不足"}
        raw = result["style_market_cap_float"] if column == "log_market_cap_float" else cap
        report["features"][column]["quality_reasons"] = {
            "missing_or_nonpositive_cap": int((raw.isna() | raw.le(settings.STYLE_MINIMUM_MARKET_CAP)).sum()),
            "unverified_point_in_time": int((~(size_pit if column in SIZE_COLUMNS else value_pit)).sum()),
            "undefined_or_invalid_value": int(result[column].isna().sum()),
        }
        if accepted:
            report["accepted_columns"].append(column)
            report["model_feature_columns"].append(transformed)
    report["sources"] = sorted(result["style_source"].dropna().astype(str).unique())
    report["calibration_end"] = str(max(dates)) if dates else None
    report["status"] = "已通过覆盖门控：" + "、".join(report["accepted_columns"]) if report["accepted_columns"] else "新增特征全部被排除：没有可靠历史或覆盖不足"
    report["exposure_warning"] = "显式风格特征保留市值和价值暴露，整体预测不保证市值或行业中性"
    return result, report
