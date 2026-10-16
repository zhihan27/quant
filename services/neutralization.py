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
    factor_missing = 0
    for date, index in result.groupby("date").groups.items():
        block = result.loc[index]
        exposures_valid = block[exposures].notna().all(axis=1)
        factors_valid = block[factors].notna().all(axis=1)
        valid = exposures_valid & factors_valid
        if industry_values is not None:
            valid &= block[industry_column].notna()
        sample = block.loc[valid]
        if len(sample) < minimum_stocks:
            skipped += 1
            if int(exposures_valid.sum()) >= minimum_stocks:
                # 暴露本身够用，只是共同样本被子集缺失的因子拖垮，需要与截面不足区分开。
                factor_missing += 1
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
    report["factor_missing_dates"] = factor_missing
    report["design_terms"] = len(design_names)
    report["removed_variance"] = {column: float(np.mean(values)) if values else None for column, values in removed.items()}
    if processed == 0:
        report["status"] = (
            f"{factor_missing} 个交易日的因子存在整体缺失，共同样本不足，未生成中性化因子"
            if factor_missing
            else "没有满足最小截面要求的交易日，未生成中性化因子"
        )
    elif skipped:
        reason = "截面不足或因子缺失" if factor_missing else "截面不足"
        report["status"] = f"已中性化 {processed} 个交易日，{skipped} 个交易日{reason}被跳过"
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
    """按流通股本与历史收盘价近似历史市值，构造中性化所需的市值与行业暴露。

    公开行情源只提供当前市值，这里用「当前流通市值 ÷ 当前价 × 历史收盘价」近似历史市值，
    并明确标注该近似忽略股本变动，属于非严格点时数据。
    """
    result = panel.copy()
    report = {"status": "未接入市值与行业暴露", "codes_with_exposure": 0, "codes_total": int(result["code"].nunique()) if "code" in result else 0}
    result["log_market_cap"] = np.nan
    result["industry"] = None
    if exposures is None or exposures.empty or "code" not in result:
        return result, report
    reference = exposures.copy()
    reference["code"] = reference["code"].astype(str).str.zfill(6)
    reference = reference.drop_duplicates("code", keep="last").set_index("code")
    exposure_codes = set(reference.index)
    share_value = None
    if share_column in reference and "price" in reference:
        prices = pd.to_numeric(reference["price"], errors="coerce")
        caps = pd.to_numeric(reference[share_column], errors="coerce")
        share_value = (caps / prices.where(prices > 0)).replace([np.inf, -np.inf], np.nan)
    has_exposure = result["code"].isin(exposure_codes) & result["close"].gt(0)
    if share_value is not None:
        shares = result["code"].map(share_value)
        result.loc[has_exposure, "log_market_cap"] = np.log((shares * result.loc[has_exposure, "close"]).where(lambda value: value > 0))
    if "industry" in reference:
        result.loc[has_exposure, "industry"] = result.loc[has_exposure, "code"].map(reference["industry"])
    available = int(result["log_market_cap"].notna().sum())
    report.update(
        codes_with_exposure=int(result.loc[has_exposure, "code"].nunique()),
        rows_with_market_cap=available,
        status=f"已匹配 {int(result.loc[has_exposure, 'code'].nunique())} 只股票的市值暴露"
        if available
        else "暴露数据缺少可用市值，未生成中性化输入",
        limitation="历史市值为流通股本×历史收盘价的近似，未使用严格点时股本与历史行业分类。",
    )
    return result, report
