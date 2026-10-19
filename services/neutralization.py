"""市值与行业中性化：每日截面对暴露做多元线性回归，取残差作为纯净 Alpha。

中性化只使用同一交易日的截面信息，不引入跨日或未来数据。同一交易日的全部因子使用相同的
有效样本（暴露与因子均非缺失），避免不同因子的回归样本不一致而失去可比性。
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd

from config import settings

LOGGER = logging.getLogger("quant.neutralization")
NEUTRAL_PREFIX = "neutral_"


def _standardize(values: pd.Series) -> pd.Series:
    deviation = float(values.std(ddof=0))
    if not np.isfinite(deviation) or deviation <= 1e-12:
        return pd.Series(0.0, index=values.index)
    return (values - float(values.mean())) / deviation


def _design_matrix(
    exposure_frame: pd.DataFrame,
    exposure_columns: Sequence[str],
    industry_column: str | None,
    minimum_industry_stocks: int,
) -> tuple[np.ndarray, list[str]]:
    """构造含截距、标准化暴露与行业哑变量的设计矩阵；哑变量丢弃首列避免共线。"""
    columns = [_standardize(pd.to_numeric(exposure_frame[column], errors="coerce")) for column in exposure_columns]
    names = [f"exposure:{column}" for column in exposure_columns]
    if industry_column and industry_column in exposure_frame:
        industry = exposure_frame[industry_column].astype(object)
        counts = industry.value_counts()
        kept = [value for value in counts.index[counts >= minimum_industry_stocks] if isinstance(value, str) and value]
        if len(kept) > 1:
            kept = sorted(kept)
            # 丢弃第一个行业作为基准组，其余行业进入哑变量，避免与截距完全共线。
            for value in kept[1:]:
                columns.append(industry.eq(value).astype(float))
                names.append(f"industry:{value}")
    matrix = np.column_stack([np.ones(len(exposure_frame))] + [series.to_numpy(dtype=float) for series in columns])
    return matrix, names


def neutralize_cross_section(
    panel: pd.DataFrame,
    factor_columns: Sequence[str],
    exposure_columns: Sequence[str] = ("log_market_cap",),
    industry_column: str | None = "industry",
    minimum_stocks: int | None = None,
    minimum_industry_stocks: int | None = None,
    output_prefix: str = NEUTRAL_PREFIX,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """对每日截面因子回归掉市值与行业暴露，返回残差列与可追溯报告。"""
    minimum_stocks = int(minimum_stocks if minimum_stocks is not None else settings.NEUTRALIZATION_MINIMUM_STOCKS)
    minimum_industry_stocks = int(
        minimum_industry_stocks
        if minimum_industry_stocks is not None
        else settings.NEUTRALIZATION_MINIMUM_INDUSTRY_STOCKS
    )
    factors = [column for column in factor_columns if column in panel]
    exposures = [column for column in exposure_columns if column in panel]
    result = panel.copy()
    for column in factors:
        result[f"{output_prefix}{column}"] = np.nan
    report: dict[str, Any] = {
        "status": "未执行中性化",
        "factors": factors,
        "exposures": exposures,
        "industry_column": industry_column if industry_column in panel else None,
        "dates": 0,
        "skipped_dates": 0,
        "factor_missing_dates": 0,
        "thin_cross_section_dates": 0,
        "exposure_missing_dates": 0,
        "industry_missing_dates": 0,
        "industry_missing_rows": 0,
        "removed_variance": {},
        "minimum_stocks": minimum_stocks,
    }
    if result.empty or not factors or not exposures:
        report["status"] = "缺少因子或暴露列，未执行中性化"
        return result, report

    industry_values = result[industry_column] if industry_column and industry_column in result else None
    removed: dict[str, list[float]] = {column: [] for column in factors}
    design_names: list[str] = []
    processed = 0
    skipped = 0
    thin_cross_section = 0
    factor_missing = 0
    exposure_missing = 0
    industry_missing = 0
    industry_missing_rows = 0
    for date, index in result.groupby("date").groups.items():
        block = result.loc[index]
        exposures_valid = block[exposures].notna().all(axis=1)
        factors_valid = block[factors].notna().all(axis=1)
        factor_exposure_valid = exposures_valid & factors_valid
        valid = factor_exposure_valid.copy()
        if industry_values is not None:
            industry_valid = block[industry_column].notna()
            industry_missing_rows += int((factor_exposure_valid & ~industry_valid).sum())
            valid &= industry_valid
        sample = block.loc[valid]
        if len(sample) < minimum_stocks:
            skipped += 1
            if len(block) < minimum_stocks:
                thin_cross_section += 1
            elif int(exposures_valid.sum()) < minimum_stocks:
                exposure_missing += 1
            elif int(factor_exposure_valid.sum()) < minimum_stocks:
                factor_missing += 1
            elif industry_values is not None:
                industry_missing += 1
            continue
        matrix, design_names = _design_matrix(sample, exposures, industry_column, minimum_industry_stocks)
        if matrix.shape[1] >= len(sample):
            # 行业哑变量过多会让回归退化为插值，宁可放弃该日也不给出虚假残差。
            skipped += 1
            continue
        target = sample[factors].to_numpy(dtype=float)
        try:
            coefficients, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
        except np.linalg.LinAlgError:
            LOGGER.warning("中性化回归失败 date=%s，该交易日不生成残差", date)
            skipped += 1
            continue
        if rank < matrix.shape[1]:
            # 秩亏说明暴露矩阵仍然共线，残差不可信，直接跳过该交易日。
            skipped += 1
            continue
        residual = target - matrix @ coefficients
        for position, column in enumerate(factors):
            centred = target[:, position] - float(np.mean(target[:, position]))
            total = float(np.sum(centred**2))
            residual_sum = float(np.sum(residual[:, position] ** 2))
            removed[column].append(0.0 if total <= 1e-18 else max(0.0, min(1.0, 1 - residual_sum / total)))
        result.loc[sample.index, [f"{output_prefix}{column}" for column in factors]] = residual
        processed += 1

    report["dates"] = processed
    report["skipped_dates"] = skipped
    report["thin_cross_section_dates"] = thin_cross_section
    report["factor_missing_dates"] = factor_missing
    report["exposure_missing_dates"] = exposure_missing
    report["industry_missing_dates"] = industry_missing
    report["industry_missing_rows"] = industry_missing_rows
    report["design_terms"] = len(design_names)
    report["removed_variance"] = {column: float(np.mean(values)) if values else None for column, values in removed.items()}
    if processed == 0:
        reasons = []
        if thin_cross_section:
            reasons.append(f"{thin_cross_section} 个交易日的原始截面不足最小股票数")
        if exposure_missing:
            reasons.append(f"{exposure_missing} 个交易日的暴露样本不足")
        if factor_missing:
            reasons.append(f"{factor_missing} 个交易日的因子共同样本不足")
        if industry_missing:
            reasons.append(f"{industry_missing} 个交易日的行业分类缺失")
        report["status"] = (
            f"{'；'.join(reasons)}，未生成中性化因子"
            if reasons else "没有满足最小截面要求的交易日，未生成中性化因子"
        )
    elif skipped:
        reasons = []
        if thin_cross_section:
            reasons.append(f"原始截面不足 {thin_cross_section} 日")
        if exposure_missing:
            reasons.append(f"暴露样本不足 {exposure_missing} 日")
        if factor_missing:
            reasons.append(f"因子共同样本不足 {factor_missing} 日")
        if industry_missing:
            reasons.append(f"行业分类不足 {industry_missing} 日")
        detail = "；".join(reasons) or f"截面或设计矩阵条件不满足 {skipped} 日"
        report["status"] = f"已中性化 {processed} 个交易日，跳过原因：{detail}"
    else:
        report["status"] = f"已中性化 {processed} 个交易日的市值与行业暴露"
    report["note"] = "残差保留相对强弱信息；市值取流通市值近似历史值，行业使用当前分类，均非严格点时数据。"
    return result, report


def neutralized_feature_columns(panel: pd.DataFrame, factor_columns: Sequence[str], output_prefix: str = NEUTRAL_PREFIX) -> list[str]:
    """返回面板中实际可用于建模的中性化因子列，缺失时为空列表。"""
    return [f"{output_prefix}{column}" for column in factor_columns if f"{output_prefix}{column}" in panel]


def attach_market_cap_exposure(
    panel: pd.DataFrame,
    exposures: pd.DataFrame | None,
    share_column: str = "market_cap_float",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """只用带生效日期的暴露记录构造历史市值与行业分类。"""
    result = panel.copy()
    report = {
        "status": "未接入点时市值与行业暴露", "codes_with_exposure": 0,
        "codes_total": int(result["code"].nunique()) if "code" in result else 0,
        "point_in_time": False, "industry_available": False,
        "market_cap_coverage": 0.0, "industry_coverage": 0.0,
    }
    result["log_market_cap"] = np.nan
    result["industry"] = None
    if exposures is None or exposures.empty or "code" not in result:
        return result, report
    reference = exposures.copy()
    if "date" not in reference:
        report.update(
            status="暴露缺少生效日期，拒绝把当前市值或行业回填到历史截面",
            limitation="需提供带有效日期的历史流通市值、价格与行业分类。",
        )
        return result, report
    reference["date"] = pd.to_datetime(reference["date"], errors="coerce")
    reference["code"] = reference["code"].astype(str).str.zfill(6)
    required = {"date", "code", share_column, "price"}
    missing = required - set(reference.columns)
    if missing:
        report.update(
            status=f"点时暴露缺少字段：{', '.join(sorted(missing))}",
            limitation="需要生效日期、流通市值和对应价格；未使用非点时值替代。",
        )
        return result, report
    reference = reference.dropna(subset=["date"]).sort_values(["date", "code"])
    reference = reference.drop_duplicates(["date", "code"], keep="last")
    source = result[["date", "code", "close"]].copy()
    source["date"] = pd.to_datetime(source["date"], errors="coerce")
    source["code"] = source["code"].astype(str).str.zfill(6)
    source["_row"] = result.index
    source = source.sort_values(["date", "code"])
    matched = pd.merge_asof(
        source,
        reference.sort_values(["date", "code"]),
        on="date",
        by="code",
        direction="backward",
    ).set_index("_row").reindex(result.index)
    prices = pd.to_numeric(matched["price"], errors="coerce")
    caps = pd.to_numeric(matched[share_column], errors="coerce")
    shares = (caps / prices.where(prices > 0)).replace([np.inf, -np.inf], np.nan)
    historical_cap = shares * pd.to_numeric(result["close"], errors="coerce")
    result["log_market_cap"] = np.log(historical_cap.where(historical_cap > 0))
    industry_field_available = "industry" in matched
    industry_available = False
    industry_rows = 0
    if industry_field_available:
        result["industry"] = matched["industry"]
    available = int(result["log_market_cap"].notna().sum())
    market_cap_coverage = available / len(result) if len(result) else 0.0
    if industry_field_available:
        industry_rows = int((result["industry"].notna() & result["log_market_cap"].notna()).sum())
    industry_coverage = industry_rows / len(result) if len(result) else 0.0
    industry_available = industry_rows > 0
    report.update(
        codes_with_exposure=int(result.loc[result["log_market_cap"].notna(), "code"].nunique()),
        rows_with_market_cap=available,
        rows_with_industry=industry_rows,
        point_in_time=available > 0,
        industry_available=industry_available,
        market_cap_coverage=market_cap_coverage,
        industry_coverage=industry_coverage,
        status=(
            f"已按生效日期匹配暴露；市值覆盖 {market_cap_coverage:.1%}，行业覆盖 {industry_coverage:.1%}"
            if available and industry_available
            else "历史行业分类无有效值，未生成中性化输入"
            if available
            else "暴露数据缺少可用市值，未生成中性化输入"
        ),
        limitation="市值按暴露日期的流通股数近似；行业分类使用该日期及此前最近记录。",
    )
    return result, report
