"""构造短线日频点时特征与明确标记的价格代理标签。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config.short_term import SHORT_TERM_FEATURES, SHORT_TERM_HORIZONS, SHORT_TERM_RESEARCH_COSTS

REQUIRED_PRICE_COLUMNS = ("date", "open", "high", "low", "close", "volume", "amount")


def _clean_history(code: str, history: pd.DataFrame, as_of: Any | None = None) -> pd.DataFrame:
    """校验日线列、去重并截断到决策日，避免读取未来行。"""
    missing = [column for column in REQUIRED_PRICE_COLUMNS if column not in history.columns]
    if missing:
        raise ValueError(f"{code} 日线缺少必要字段：{'、'.join(missing)}。")
    frame = history.loc[:, REQUIRED_PRICE_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    for column in REQUIRED_PRICE_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "open", "high", "low", "close"])
    frame = frame.loc[(frame[["open", "high", "low", "close"]] > 0).all(axis=1)]
    valid_ohlc = (
        frame["high"].ge(frame["low"])
        & frame["open"].between(frame["low"], frame["high"])
        & frame["close"].between(frame["low"], frame["high"])
    )
    frame = frame.loc[valid_ohlc]
    frame = frame.sort_values("date", kind="stable").drop_duplicates("date", keep="last")
    if as_of is not None:
        cutoff = pd.Timestamp(as_of).normalize()
        frame = frame.loc[frame["date"] <= cutoff]
    frame["code"] = str(code).zfill(6)
    return frame.reset_index(drop=True)


def build_short_features(
    history_by_code: dict[str, pd.DataFrame], as_of: Any | None = None, minimum_history: int = 20
) -> pd.DataFrame:
    """生成截面特征；每只股票先独立计算，再拼接同日截面。"""
    if int(minimum_history) < 20:
        raise ValueError("特征计算至少需要 20 根历史日线。")
    frames = []
    for code, history in history_by_code.items():
        bars = _clean_history(code, history, as_of=as_of)
        if len(bars) < int(minimum_history):
            continue
        close = bars["close"]
        previous_close = close.shift(1)
        returns = close.pct_change(fill_method=None)
        true_range = pd.concat(
            [
                bars["high"] - bars["low"],
                (bars["high"] - previous_close).abs(),
                (bars["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        rolling_volume_20 = bars["volume"].rolling(20, min_periods=20).mean()
        rolling_amount_20 = bars["amount"].rolling(20, min_periods=20).mean()
        feature_frame = pd.DataFrame(
            {
                "code": bars["code"],
                "date": bars["date"],
                "return_1d": returns,
                "return_3d": close.pct_change(3, fill_method=None),
                "return_5d": close.pct_change(5, fill_method=None),
                "return_10d": close.pct_change(10, fill_method=None),
                "return_20d": close.pct_change(20, fill_method=None),
                "overnight_return": bars["open"] / previous_close - 1.0,
                "intraday_return": close / bars["open"] - 1.0,
                "close_sma5_gap": close / close.rolling(5, min_periods=5).mean() - 1.0,
                "close_sma10_gap": close / close.rolling(10, min_periods=10).mean() - 1.0,
                "close_sma20_gap": close / close.rolling(20, min_periods=20).mean() - 1.0,
                "volatility_5d": returns.rolling(5, min_periods=5).std(ddof=0),
                "volatility_20d": returns.rolling(20, min_periods=20).std(ddof=0),
                "downside_volatility_20d": returns.clip(upper=0).rolling(20, min_periods=20).std(ddof=0),
                "atr14_pct": true_range.rolling(14, min_periods=14).mean() / close,
                "range_location_20d": (close - bars["low"].rolling(20, min_periods=20).min())
                / (bars["high"].rolling(20, min_periods=20).max() - bars["low"].rolling(20, min_periods=20).min()).replace(0, np.nan),
                "volume_ratio_5_20": bars["volume"].rolling(5, min_periods=5).mean() / rolling_volume_20.replace(0, np.nan),
                "amount_ratio_5_20": bars["amount"].rolling(5, min_periods=5).mean() / rolling_amount_20.replace(0, np.nan),
                "amihud_20d": (returns.abs() / bars["amount"].where(bars["amount"] > 0)).rolling(20, min_periods=20).mean(),
                "open": bars["open"],
            }
        )
        frames.append(feature_frame)
    if not frames:
        return pd.DataFrame(columns=["code", "date", *SHORT_TERM_FEATURES])
    result = pd.concat(frames, ignore_index=True)
    for column in [*SHORT_TERM_FEATURES, "open"]:
        result[column] = pd.to_numeric(result[column], errors="coerce").astype("float32")
    result["signal_time"] = result["date"] + pd.Timedelta(hours=15, minutes=5)
    return result.sort_values(["date", "code"], kind="stable").reset_index(drop=True)


def build_short_labels(
    features: pd.DataFrame, horizon: int = 3, costs: dict[str, float] | None = None
) -> pd.DataFrame:
    """按次日开盘入场、入场后 H 个交易间隔开盘退出计算净收益代理。"""
    if int(horizon) not in SHORT_TERM_HORIZONS:
        raise ValueError(f"短线研究周期仅支持 {SHORT_TERM_HORIZONS}。")
    rates = dict(SHORT_TERM_RESEARCH_COSTS if costs is None else costs)
    required_rates = ("commission_rate", "stamp_duty_rate", "slippage_rate")
    if any(key not in rates for key in required_rates):
        raise ValueError("成本配置必须包含佣金、印花税和滑点比例。")
    if any(not np.isfinite(float(rates[key])) or float(rates[key]) < 0 for key in required_rates):
        raise ValueError("成本比例必须是有限的非负数。")
    required = {"code", "date", "open"}
    if not required.issubset(features.columns):
        raise ValueError("标签数据必须包含 code、date 和 open。")
    result = features.loc[:, [column for column in features.columns if column in {"code", "date", "signal_time", "open"}]].copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result["entry_date"] = result.groupby("code", sort=False)["date"].shift(-1)
    result["entry_open"] = result.groupby("code", sort=False)["open"].shift(-1)
    exit_offset = int(horizon) + 1
    result["label_end_date"] = result.groupby("code", sort=False)["date"].shift(-exit_offset)
    result["exit_open"] = result.groupby("code", sort=False)["open"].shift(-exit_offset)
    buy_cost = float(rates["commission_rate"]) + float(rates["slippage_rate"])
    sell_cost = buy_cost + float(rates["stamp_duty_rate"])
    result["target_net_return"] = (result["exit_open"] * (1.0 - sell_cost)) / (result["entry_open"] * (1.0 + buy_cost)) - 1.0
    result["label_status"] = np.where(result["target_net_return"].notna(), "price_proxy_matured", "pending")
    result["label_basis"] = "qfq_open_price_proxy_no_execution_state"
    result["horizon"] = int(horizon)
    return result.sort_values(["date", "code"], kind="stable").reset_index(drop=True)
