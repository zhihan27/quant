"""行情异常清洗：前复权防跳空、长期停牌剔除与一字涨跌停流动性锁定。

清洗只标注与剔除可信度不足的 K 线，不改写原始价格；所有判定依据仅来自当日及以前数据。
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Any

import numpy as np
import pandas as pd

from config import settings

FLAG_COLUMNS = ("is_suspended", "is_locked_up", "is_locked_down", "is_adjustment_gap", "is_long_suspension", "is_usable")


def price_limit_rate(code: str | None, date: Any = None, is_st: bool = False) -> float:
    """按板块与交易日期返回单日涨跌幅上限比例。"""
    if code and code.startswith(("4", "8", "92")):
        return 0.30
    if code and code.startswith(("688", "689")):
        return 0.20
    if code and code.startswith(("300", "301")):
        return 0.10 if date is not None and pd.Timestamp(date) < pd.Timestamp("2020-08-24") else 0.20
    return 0.05 if is_st else 0.10


def open_price_fillable(bar: pd.Series, direction: str, previous_close: float, limit_rate: float) -> bool:
    """仅用开盘价、已知昨收及停牌状态作保守成交判断，封板开盘不假设排队成功。"""
    price = bar.get("open")
    volume = bar.get("volume")
    if pd.isna(price) or not np.isfinite(price) or price <= 0:
        return False
    if bool(bar.get("is_suspended", False)) or pd.isna(volume) or volume <= 0:
        return False
    if not np.isfinite(previous_close) or previous_close <= 0:
        return False
    if bool(bar.get("no_price_limit", False)):
        return True
    factor = 1 + limit_rate if direction == "buy" else 1 - limit_rate
    limit = float(Decimal(str(previous_close * factor)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    return price < limit - 0.0049 if direction == "buy" else price > limit + 0.0049


def _limit_series(codes: pd.Series, dates: pd.Series, is_st: bool) -> pd.Series:
    """逐行计算涨跌幅上限，创业板 2020-08-24 前后的制度差异按日期区分。"""
    rate = pd.Series(0.10 if not is_st else 0.05, index=codes.index, dtype=float)
    rate = rate.mask(codes.str.startswith(("300", "301")), 0.20)
    early_chinext = codes.str.startswith(("300", "301")) & (dates < pd.Timestamp("2020-08-24"))
    rate = rate.mask(early_chinext, 0.10)
    rate = rate.mask(codes.str.startswith(("688", "689")), 0.20)
    rate = rate.mask(codes.str.startswith(("4", "8", "92")), 0.30)
    return rate


def clean_daily_history(
    history: pd.DataFrame,
    code: str | None = None,
    is_st: bool = False,
    adjusted: bool = False,
    long_suspension_days: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """标注停牌、一字涨跌停与前复权异常跳空，返回清洗后的行情及可追溯报告。

    只有 `is_usable` 为真的 K 线才允许进入因子、标签与成交模拟；被标注的行保留原值供页面核对。
    """
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"历史行情缺少字段：{', '.join(sorted(missing))}")
    frame = history.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame["date"].duplicated().any():
        raise ValueError("历史行情包含重复日期，无法执行清洗。")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    codes = pd.Series(str(code or ""), index=frame.index, dtype=object)
    limits = _limit_series(codes, frame["date"], is_st)
    previous_close = frame["close"].shift(1)
    returns = frame["close"] / previous_close.where(previous_close > 0) - 1

    # 停牌：无成交量或缺失价格；停牌当日仍保留用于持仓估值。
    suspended = frame["volume"].isna() | frame["volume"].le(0) | frame[["open", "high", "low", "close"]].isna().any(axis=1)

    # 一字板：全天无价格振幅且已触及涨跌停，属于流动性锁定状态，不能假设成交。
    amplitude = frame["high"] - frame["low"]
    flat = amplitude.le(np.maximum(0.01, frame["close"].abs() * 0.001))
    locked_up = flat & returns.ge(limits * 0.98) & previous_close.gt(0)
    locked_down = flat & returns.le(-limits * 0.98) & previous_close.gt(0)

    suspension_days = int(long_suspension_days if long_suspension_days is not None else settings.LONG_SUSPENSION_DAYS)
    runs = (suspended != suspended.shift()).cumsum()
    run_length = suspended.groupby(runs).transform("size")
    long_suspension = suspended & run_length.ge(suspension_days)

    # 前复权序列不应出现超过涨跌幅制度的价格跳空；出现即视为复权口径不一致或源数据异常。
    # 新股上市前 5 个交易日与恢复上市无涨跌幅限制，无法用该规则判断，明确跳过。
    limit_free = pd.Series(frame.index < 6, index=frame.index)
    gap = returns.abs().gt(limits * settings.ADJUSTMENT_GAP_TOLERANCE + 0.01)
    adjustment_gap = gap & previous_close.gt(0) & ~suspended & ~limit_free
    if not adjusted:
        # 不复权序列的除权跳空属于真实价格行为，只作标记提示，不作为脏数据剔除。
        adjustment_gap = adjustment_gap & pd.Series(False, index=frame.index)

    frame["previous_close"] = previous_close
    frame["return_pct"] = returns
    frame["limit_rate"] = limits
    frame["is_suspended"] = suspended.fillna(False).astype(bool)
    frame["is_locked_up"] = locked_up.fillna(False).astype(bool)
    frame["is_locked_down"] = locked_down.fillna(False).astype(bool)
    frame["is_adjustment_gap"] = adjustment_gap.fillna(False).astype(bool)
    frame["is_long_suspension"] = long_suspension.fillna(False).astype(bool)
    frame["is_usable"] = ~(frame["is_suspended"] | frame["is_adjustment_gap"])
    report = {
        "code": str(code) if code else None,
        "adjustment": "前复权" if adjusted else "不复权",
        "bars": int(len(frame)),
        "suspended_bars": int(frame["is_suspended"].sum()),
        "long_suspension_bars": int(frame["is_long_suspension"].sum()),
        "locked_up_bars": int(frame["is_locked_up"].sum()),
        "locked_down_bars": int(frame["is_locked_down"].sum()),
        "adjustment_gap_bars": int(frame["is_adjustment_gap"].sum()),
        "usable_bars": int(frame["is_usable"].sum()),
        "gap_rule": "跨日涨跌幅超过该板块上限的 1.2 倍并超过 1%，按复权口径异常处理；上市前 6 根 K 线不参与判定",
        "long_suspension_rule": f"连续停牌达到 {suspension_days} 个交易日即标记，长期停牌股票不进入当日截面",
    }
    return frame, report


def tradable_mask(frame: pd.DataFrame, direction: str) -> pd.Series:
    """按成交方向给出可成交标记，停牌与反向一字板不计为可成交。"""
    if direction not in ("buy", "sell"):
        raise ValueError("成交方向仅支持 buy 或 sell。")
    blocked = frame.get("is_locked_up" if direction == "buy" else "is_locked_down", pd.Series(False, index=frame.index))
    usable = frame.get("is_usable", pd.Series(True, index=frame.index))
    return usable.fillna(False).astype(bool) & ~blocked.fillna(False).astype(bool)
