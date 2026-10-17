"""按交易日截面处理回归训练标签，保留原始标签用于样本外评估。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config import settings


def winsorize_training_labels(
    labels: pd.Series, dates: pd.Series, upper_quantile: float | None = None,
    minimum_stocks: int = 5,
) -> tuple[pd.Series, dict[str, Any]]:
    """将同一日期训练标签的上尾压至指定分位数，返回新标签与处理摘要。

    labels 与 dates 必须索引一致；调用方须先清除跨入测试期的标签。
    不修改输入，不截下尾，样本不足的截面保留原值，缺失值仍为缺失。
    """
    quantile = float(settings.LABEL_WINSOR_UPPER_QUANTILE if upper_quantile is None else upper_quantile)
    if not np.isfinite(quantile) or not 0.5 < quantile <= 1.0:
        raise ValueError("标签上侧截尾分位数必须大于 0.5 且不超过 1.0。")
    if minimum_stocks < 2:
        raise ValueError("截尾截面至少需要两只股票。")
    if not labels.index.equals(dates.index):
        raise ValueError("训练标签与截面日期的索引必须一致。")
    values = pd.to_numeric(labels, errors="coerce").replace([np.inf, -np.inf], np.nan)
    date_keys = pd.to_datetime(dates, errors="raise")
    groups = values.groupby(date_keys)
    counts = groups.transform("count")
    valid = values.notna() & date_keys.notna()
    eligible = valid & counts.ge(minimum_stocks)
    result = values.copy()
    clipped = pd.Series(False, index=values.index)
    if quantile < 1.0:
        # 仅在已成熟的训练样本内部逐日计算，不能把未来日期或测试标签并入阈值。
        bounds = groups.transform("quantile", q=quantile)
        clipped = eligible & values.gt(bounds)
        result.loc[clipped] = bounds.loc[clipped]
    report = {
        "method": "cross_section_upper_winsorization",
        "enabled": quantile < 1.0,
        "upper_quantile": quantile,
        "minimum_stocks": int(minimum_stocks),
        "samples": int(valid.sum()),
        "eligible_samples": int(eligible.sum()),
        "clipped_samples": int(clipped.sum()),
        "clipped_fraction": float(clipped.sum() / valid.sum()) if valid.any() else None,
        "skipped_dates": int(date_keys.loc[valid & ~eligible].nunique()),
        "evaluation_labels": "raw_excess_return",
    }
    return result, report
