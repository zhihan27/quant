"""生成无未来数据的中低频行情与量价因子。"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "momentum_5",
    "momentum_20",
    "momentum_60",
    "ma20_gap",
    "ma60_gap",
    "trend_strength",
    "volatility_20",
    "atr_pct",
    "volume_change_20",
    "turnover_20",
    "price_volume_corr_20",
    "relative_strength_20",
    "rsi_14",
    # 另类特征：短期反转捕捉 A 股散户主导的均值回归，Amihud 与量价背离描述流动性与量价结构。
    "reversal_1",
    "reversal_3",
    "amihud_20",
    "volume_price_divergence_60",
]


def build_features(
    history: pd.DataFrame, benchmark: pd.DataFrame | None = None
) -> pd.DataFrame:
    """从历史 OHLCV 生成每行仅依赖当日及以前数据的因子。"""
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"历史行情缺少字段：{', '.join(sorted(missing))}")

    frame = history.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame["date"].duplicated().any():
        raise ValueError("历史行情包含重复日期，无法计算可靠因子。")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    # 停牌与复权异常 K 线不参与因子计算，避免用重复价格制造虚假动量与低波动。
    # 原始收盘价保留给标签与参考价使用，停牌期间的复牌收益仍然有效。
    usable = frame["is_usable"].astype(bool) if "is_usable" in frame else pd.Series(True, index=frame.index)
    close = pd.to_numeric(frame["close"], errors="coerce").where(usable)
    volume = pd.to_numeric(frame["volume"], errors="coerce").where(usable)
    high = pd.to_numeric(frame["high"], errors="coerce").where(usable)
    low = pd.to_numeric(frame["low"], errors="coerce").where(usable)
    amount = pd.to_numeric(frame["amount"], errors="coerce").where(usable) if "amount" in frame else None
    daily_return = close.pct_change(fill_method=None)
    ma20 = close.rolling(20, min_periods=20).mean()
    ma60 = close.rolling(60, min_periods=60).mean()
    true_range = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    average_gain = daily_return.clip(lower=0).rolling(14, min_periods=14).mean()
    average_loss = (-daily_return.clip(upper=0)).rolling(14, min_periods=14).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)

    frame["momentum_5"] = close.pct_change(5)
    frame["momentum_20"] = close.pct_change(20)
    frame["momentum_60"] = close.pct_change(60)
    frame["ma20_gap"] = close / ma20 - 1
    frame["ma60_gap"] = close / ma60 - 1
    frame["trend_strength"] = ma20 / ma60 - 1
    frame["volatility_20"] = daily_return.rolling(20, min_periods=20).std() * np.sqrt(252)
    frame["atr_pct"] = true_range.rolling(14, min_periods=14).mean() / close
    frame["volume_change_20"] = volume.pct_change(20)
    frame["volume_ma20"] = volume.rolling(20, min_periods=20).mean()
    frame["turnover_20"] = (
        pd.to_numeric(frame["turnover_rate"], errors="coerce").rolling(20, min_periods=5).mean()
        if "turnover_rate" in frame
        else np.nan
    )
    frame["price_volume_corr_20"] = daily_return.rolling(20, min_periods=20).corr(
        volume.pct_change()
    )
    frame["rsi_14"] = 100 - 100 / (1 + relative_strength)
    frame.loc[(average_loss == 0) & (average_gain > 0), "rsi_14"] = 100

    # 短期反转：过去 1、3 个交易日下跌的股票取正分，刻画 A 股散户主导的均值回归。
    frame["reversal_1"] = -close.pct_change(1)
    frame["reversal_3"] = -close.pct_change(3)
    # Amihud 非流动性：单位成交额推动的价格变动，数值越大越难成交。
    if amount is not None:
        daily_amount = amount.where(amount > 0)
        frame["amihud_20"] = (daily_return.abs() / daily_amount).rolling(20, min_periods=15).mean() * 1e8
    else:
        frame["amihud_20"] = np.nan
    # 量价背离：价格相对自身 60 日区间越高而成交量越低，背离越明显。
    price_window = close.rolling(60, min_periods=60)
    volume_window = volume.rolling(60, min_periods=60)
    price_z = (close - price_window.mean()) / price_window.std().where(lambda value: value > 1e-12)
    volume_z = (volume - volume_window.mean()) / volume_window.std().where(lambda value: value > 1e-12)
    frame["volume_price_divergence_60"] = price_z - volume_z

    frame["relative_strength_20"] = np.nan
    if benchmark is not None and not benchmark.empty:
        reference = benchmark[["date", "close"]].copy().sort_values("date")
        reference["date"] = pd.to_datetime(reference["date"], errors="raise")
        reference = reference.drop_duplicates("date", keep="last")
        reference["benchmark_return_20"] = reference["close"].pct_change(20)
        frame = frame.merge(
            reference[["date", "benchmark_return_20"]], on="date", how="left"
        )
        frame["relative_strength_20"] = frame["momentum_20"] - frame["benchmark_return_20"]
        frame = frame.drop(columns=["benchmark_return_20"])

    return frame.replace([np.inf, -np.inf], np.nan)


def add_forward_returns(
    features: pd.DataFrame,
    horizons: tuple[int, ...] = (5, 20, 60),
    benchmark: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """以基准交易日历生成未来超额收益标签，缺失端点不填充。"""
    frame = features.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if any(horizon not in (5, 20, 60) for horizon in horizons):
        raise ValueError("标签周期仅支持5、20、60个交易日。")
    if benchmark is None or benchmark.empty:
        for horizon in horizons:
            frame[f"target_{horizon}"] = np.nan
            frame[f"label_end_{horizon}"] = pd.NaT
        return frame
    reference = benchmark[["date", "close"]].copy()
    reference["date"] = pd.to_datetime(reference["date"])
    reference = reference.sort_values("date").set_index("date")
    if reference.index.duplicated().any() or frame["date"].duplicated().any():
        raise ValueError("标签计算不接受重复交易日。")
    reference["close"] = pd.to_numeric(reference["close"], errors="coerce")
    stock = frame.set_index("date")["close"].reindex(reference.index)
    for horizon in horizons:
        absolute = stock.shift(-horizon) / stock - 1
        benchmark_return = reference["close"].shift(-horizon) / reference["close"] - 1
        valid = stock.gt(0) & stock.shift(-horizon).gt(0) & reference["close"].gt(0) & reference["close"].shift(-horizon).gt(0)
        valid &= np.isfinite(absolute) & np.isfinite(benchmark_return)
        frame[f"forward_return_{horizon}"] = frame["date"].map(absolute.where(valid))
        frame[f"target_{horizon}"] = frame["date"].map((absolute - benchmark_return).where(valid))
        endpoints = pd.Series(reference.index, index=reference.index).shift(-horizon)
        frame[f"label_end_{horizon}"] = frame["date"].map(endpoints)
        frame.loc[frame[f"target_{horizon}"].isna(), f"label_end_{horizon}"] = pd.NaT
    return frame


def normalize_cross_section(
    panel: pd.DataFrame, columns: list[str], minimum_stocks: int = 5, minimum_coverage: float = 0.8
) -> pd.DataFrame:
    """只使用同一交易日的股票截面计算Z分数，禁止跨日去均值。"""
    if panel.duplicated(["date", "code"]).any():
        raise ValueError("截面数据中股票与交易日重复。")
    frame = panel.copy()
    size = frame.groupby("date")["code"].transform("nunique")
    required = max(minimum_stocks, int(np.ceil(frame["code"].nunique() * minimum_coverage)))
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        groups = values.groupby(frame["date"])
        mean = groups.transform("mean")
        deviation = groups.transform("std", ddof=0)
        score = (values - mean) / deviation.where(deviation > 1e-12)
        score = score.mask((deviation <= 1e-12) & values.notna(), 0.0)
        enough = size.ge(required) & groups.transform("count").ge(required)
        frame[f"xs_{column}"] = score.where(enough)
    return frame

