"""筹码分布集中度因子：日线级筹码堆积模拟。

按每日高、低、收盘价与换手率，在价格轴上以三角加权分布模拟当日新增筹码，并用换手率对历史
筹码做衰减，滚动推算个股的筹码成本分布。输出两个横截面因子：

- ``chip_cost_range_width``：90% 成本区间宽度，即 5% 与 95% 分位成本价之差相对现价的比值。
  数值越小说明筹码越集中，越大说明成本越发散。
- ``chip_winner_rate``：获利盘比例，成本低于现价的筹码占比，衡量上方套牢盘与下方获利盘的
  力量对比。

无未来函数约束：第 t 日的分布只累积第 t 日及以前成交的筹码；价格轴采用扩张窗口边界，
不使用未来价格区间决定分箱宽度，避免用未来极值反向影响历史分位。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from config import settings

LOGGER = logging.getLogger("quant.chip")

CHIP_FACTOR_COLUMNS = ["chip_cost_range_width", "chip_winner_rate"]
# 换手率在免费行情源里以百分比给出（0.5 表示 0.5%）；低于该中位数时按小数口径处理。
_TURNOVER_PERCENT_MEDIAN = 0.05


def turnover_fraction(turnover_rate: Any) -> pd.Series:
    """把换手率统一为 0~1 的比例；无法判定口径或缺失时返回缺失值，不猜测。"""
    values = pd.to_numeric(turnover_rate, errors="coerce")
    positive = values.where(values > 0)
    median = positive.median()
    if pd.notna(median) and float(median) > _TURNOVER_PERCENT_MEDIAN:
        values = values / 100.0
    return values.clip(lower=0.0, upper=1.0)


def _average_price(frame: pd.DataFrame) -> pd.Series:
    """当日成交均价优先取成交额/成交量，缺失时退回高低收三价均值。"""
    if "amount" in frame and "volume" in frame:
        amount = pd.to_numeric(frame["amount"], errors="coerce")
        volume = pd.to_numeric(frame["volume"], errors="coerce")
        vwap = (amount / volume.where(volume > 0)).replace([np.inf, -np.inf], np.nan)
    else:
        vwap = pd.Series(np.nan, index=frame.index, dtype=float)
    fallback = pd.concat(
        [frame["high"], frame["low"], frame["close"]], axis=1
    ).mean(axis=1)
    return vwap.where(vwap > 0, fallback)


def _triangle_weights(centers: np.ndarray, low: float, high: float, peak: float, mass: float) -> np.ndarray:
    """在 [low, high] 上按峰值 peak 生成三角分布并归一到 mass；退化区间落在最近一档。"""
    weights = np.zeros_like(centers)
    if not (mass > 0):
        return weights
    span = high - low
    if not np.isfinite(span) or span <= 0:
        index = int(np.clip(np.searchsorted(centers, peak), 0, len(centers) - 1))
        weights[index] = mass
        return weights
    peak = float(np.clip(peak, low, high))
    inside = (centers >= low) & (centers <= high)
    left = centers <= peak
    right = centers >= peak
    with np.errstate(divide="ignore", invalid="ignore"):
        rising = np.where(inside & left, (centers - low) / max(peak - low, 1e-12), 0.0)
        falling = np.where(inside & right, (high - centers) / max(high - peak, 1e-12), 0.0)
    weights = np.maximum(rising, falling)
    total = float(weights.sum())
    if total <= 1e-18:
        index = int(np.clip(np.searchsorted(centers, peak), 0, len(centers) - 1))
        weights = np.zeros_like(centers)
        weights[index] = mass
        return weights
    return weights / total * mass


def _rebuild_grid(centers: np.ndarray, weights: np.ndarray, low: float, high: float, bins: int) -> tuple[np.ndarray, np.ndarray]:
    """价格区间扩张时按插值重分箱；重分箱后按总量守恒还原权重，避免反复插值丢质量。"""
    new_centers = np.linspace(low, high, bins)
    total_before = float(weights.sum())
    if total_before <= 1e-18:
        return new_centers, np.zeros(bins, dtype=float)
    rebinned = np.interp(new_centers, centers, weights, left=0.0, right=0.0)
    total_after = float(rebinned.sum())
    if total_after <= 1e-18:
        return new_centers, np.zeros(bins, dtype=float)
    return new_centers, rebinned * (total_before / total_after)


def _quantile_price(centers: np.ndarray, cumulative: np.ndarray, target: float) -> float:
    """在累计分布上取分位价格；累计分布已归一到 1。"""
    index = int(np.searchsorted(cumulative, target, side="left"))
    return float(centers[min(index, len(centers) - 1)])


def chip_distribution(
    history: pd.DataFrame,
    bins: int | None = None,
    decay: float | None = None,
    coverage: float | None = None,
    minimum_bars: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """滚动模拟单只股票的筹码分布。

    返回逐日因子、最后一日完整筹码分布与可追溯报告。历史长度不足时因子列为缺失，不以短样本
    冒充稳态筹码。
    """
    required = {"date", "high", "low", "close"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"筹码分布缺少字段：{', '.join(sorted(missing))}")
    bins = int(bins or settings.CHIP_PRICE_BINS)
    decay = float(settings.CHIP_DECAY_FACTOR if decay is None else decay)
    coverage = float(settings.CHIP_COVERAGE_RATIO if coverage is None else coverage)
    minimum_bars = int(settings.CHIP_MINIMUM_BARS if minimum_bars is None else minimum_bars)
    if bins < 8:
        raise ValueError("筹码分布价格分箱至少为 8。")
    if not 0 < coverage < 1:
        raise ValueError("成本区间覆盖率必须在 0 与 1 之间。")

    frame = history.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame["date"].duplicated().any():
        raise ValueError("筹码分布不接受重复交易日。")
    highs = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
    averages = _average_price(frame).to_numpy(dtype=float)
    usable = frame["is_usable"].astype(bool).to_numpy() if "is_usable" in frame else np.ones(len(frame), dtype=bool)
    turnover = turnover_fraction(frame["turnover_rate"]).to_numpy(dtype=float) if "turnover_rate" in frame else np.full(len(frame), np.nan)
    # 停牌与复权异常 K 线不产生新筹码，也不衰减历史筹码：当日实际无成交。
    valid = usable & np.isfinite(highs) & np.isfinite(lows) & np.isfinite(closes) & (closes > 0) & (highs >= lows)

    centers: np.ndarray | None = None
    weights: np.ndarray | None = None
    low_bound = high_bound = float("nan")
    widths = np.full(len(frame), np.nan, dtype=float)
    winners = np.full(len(frame), np.nan, dtype=float)
    accumulated = 0

    for position in range(len(frame)):
        if not valid[position]:
            continue
        day_low, day_high, day_close = float(lows[position]), float(highs[position]), float(closes[position])
        if centers is None:
            low_bound, high_bound = day_low, day_high
            if high_bound <= low_bound:
                high_bound = low_bound + max(abs(low_bound) * 1e-4, 0.01)
            centers = np.linspace(low_bound, high_bound, bins)
            weights = np.zeros(bins, dtype=float)
        elif day_low < low_bound or day_high > high_bound:
            low_bound = min(low_bound, day_low)
            high_bound = max(high_bound, day_high)
            centers, weights = _rebuild_grid(centers, weights, low_bound, high_bound, bins)

        rate = turnover[position]
        rate = 0.0 if not np.isfinite(rate) else float(rate)
        rate = float(np.clip(rate * decay, 0.0, 1.0))
        # 先按换手衰减存量筹码，再按当日换手补入新筹码，总权重保持在 1 附近。
        weights *= (1.0 - rate)
        peak = averages[position] if np.isfinite(averages[position]) else (day_high + day_low + day_close) / 3
        weights += _triangle_weights(centers, day_low, day_high, float(peak), rate)
        total = float(weights.sum())
        accumulated += 1
        if total <= 1e-18 or accumulated < minimum_bars:
            continue
        normalized = weights / total
        # 累计分布理论上到 1 封顶，但浮点累加会溢出到 1.0000000000000007；
        # 获利盘比例是「比例」，必须在 [0, 1] 内，否则会污染后续的截面标准化。
        cumulative = np.minimum(np.cumsum(normalized), 1.0)
        lower_price = _quantile_price(centers, cumulative, (1 - coverage) / 2)
        upper_price = _quantile_price(centers, cumulative, 1 - (1 - coverage) / 2)
        widths[position] = (upper_price - lower_price) / day_close
        below = int(np.searchsorted(centers, day_close, side="right"))
        winners[position] = float(cumulative[min(below, len(centers) - 1)])

    result = frame[["date"]].copy()
    result["code"] = str(history["code"].iloc[0]).zfill(6) if "code" in history and len(history) else None
    result["chip_cost_range_width"] = widths
    result["chip_winner_rate"] = winners
    final = pd.DataFrame(columns=["price", "weight"])
    if centers is not None and weights is not None and float(weights.sum()) > 1e-18:
        final = pd.DataFrame({"price": centers, "weight": weights / float(weights.sum())})
    report = {
        "bins": bins,
        "decay": decay,
        "coverage": coverage,
        "minimum_bars": minimum_bars,
        "bars": int(len(frame)),
        "accumulated_bars": int(accumulated),
        "factor_bars": int(np.isfinite(widths).sum()),
        "turnover_source": "换手率（缺失时该日不新增筹码）" if "turnover_rate" in frame else "缺少换手率字段，未产生因子",
        "status": (
            f"筹码分布已计算 {int(np.isfinite(widths).sum())} 个交易日"
            if np.isfinite(widths).sum()
            else f"有效历史不足 {minimum_bars} 个交易日，未产生筹码因子"
        ),
        "rule": f"三角加权堆积 + 换手衰减，取 {coverage:.0%} 筹码覆盖区间宽度与现价位置计算获利盘比例。",
    }
    return result, final, report


def attach_chip_factors(
    panel: pd.DataFrame, histories: dict[str, pd.DataFrame] | None = None, progress: Any = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """把逐股筹码因子并入日截面面板；缺失股票留空，不用池均值填充。"""
    if panel is None or panel.empty or "code" not in panel:
        return panel, {"status": "面板为空，未计算筹码因子"}
    result = panel.copy()
    for column in CHIP_FACTOR_COLUMNS:
        if column not in result:
            result[column] = np.nan
    histories = histories or {}
    computed = 0
    failures: list[str] = []
    for index, (code, history) in enumerate(histories.items()):
        if progress:
            progress(f"正在计算 {code} 的筹码分布（{index + 1}/{len(histories)}）")
        try:
            factors, _, _ = chip_distribution(history)
        except (ValueError, KeyError, TypeError) as exc:
            failures.append(f"{code}：{type(exc).__name__}")
            continue
        if factors.empty:
            continue
        keys = pd.to_datetime(factors["date"])
        values = factors.set_index(keys)[CHIP_FACTOR_COLUMNS]
        mask = result["code"].astype(str).str.zfill(6).eq(str(code).zfill(6))
        if not mask.any():
            continue
        aligned = values.reindex(pd.to_datetime(result.loc[mask, "date"]))
        for column in CHIP_FACTOR_COLUMNS:
            result.loc[mask, column] = aligned[column].to_numpy()
        computed += 1
    coverage = float(result[CHIP_FACTOR_COLUMNS].notna().all(axis=1).mean()) if len(result) else 0.0
    report = {
        "codes_requested": int(len(histories)),
        "codes_computed": int(computed),
        "rows_with_factors": int(result[CHIP_FACTOR_COLUMNS].notna().all(axis=1).sum()),
        "coverage": coverage,
        "failures": failures[:20],
        "status": f"筹码因子已覆盖 {computed} 只股票，面板覆盖率 {coverage:.0%}" if computed else "未产生筹码因子",
        "columns": CHIP_FACTOR_COLUMNS,
    }
    return result, report
