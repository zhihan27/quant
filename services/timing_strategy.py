"""由分析页、扫描和历史回测共同使用的交易规则。"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


def _finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def build_trade_plan(
    current_price: float | None,
    atr_pct: float | None,
    ma20: float | None,
    ma60: float | None,
    relative_signal: float | None,
    model_ready: bool,
    as_of: str | None = None,
    data_status: str | None = None,
    volume: float | None = None,
    average_volume: float | None = None,
    history_ready: bool = True,
    evaluation_time: str | None = None,
) -> dict[str, object]:
    """根据收盘价、因子和已验证的相对信号生成计划，返回价格及有效标志。"""
    if not _finite(current_price) or not _finite(atr_pct):
        return {"status": "价格或 ATR 不是有限数值", "signal_active": False}
    if current_price <= 0 or not 0 < atr_pct < 0.5:
        return {"status": "价格或 ATR 超出有效范围", "signal_active": False}

    atr = current_price * atr_pct
    trend_ok = (
        _finite(ma20) and _finite(ma60)
        and ma20 > ma60 > 0 and current_price > ma20
    )
    volume_ok = (
        _finite(volume) and _finite(average_volume)
        and average_volume > 0 and volume >= average_volume
    )
    forecast_ok = _finite(relative_signal) and relative_signal > 0
    data_fresh = bool(data_status) and not any(
        marker in data_status for marker in ("过期", "非实时", "未知", "失败")
    )
    now = datetime.fromisoformat(evaluation_time) if evaluation_time else datetime.now(ZoneInfo("Asia/Shanghai"))
    if now.tzinfo is None:
        now = now.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    try:
        generated = datetime.fromisoformat(as_of) if as_of else now
        if generated.tzinfo is None:
            generated = generated.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
        valid_until = generated + timedelta(days=7)
        time_ok = generated <= now <= valid_until
    except (TypeError, ValueError):
        generated = now
        valid_until = now
        time_ok = False

    active = bool(model_ready and forecast_ok and trend_ok and volume_ok and data_fresh and history_ready and time_ok)
    if not history_ready:
        state = "历史行情未更新或不完整，不生成有效信号"
    elif not data_fresh or not time_ok:
        state = "规则化参考（行情过期、非实时或信号已失效，仅供复核）"
    elif not model_ready:
        state = "规则化参考（模型尚未通过样本外验证）"
    elif not forecast_ok:
        state = "模型预测不支持买入"
    elif not trend_ok or not volume_ok:
        state = "趋势或成交量条件未满足"
    else:
        state = "模型与规则均支持（研究参考）"
    return {
        "status": state,
        "signal_active": active,
        # 完整收盘信号供次日人工复核，行情实时性仍由即时信号单独约束。
        "close_signal_active": bool(model_ready and forecast_ok and trend_ok and volume_ok and history_ready and time_ok),
        "reference_price": float(current_price),
        "buy_zone": (round(current_price - 0.25 * atr, 2), round(current_price + 0.25 * atr, 2)),
        "entry_trigger": "模型通过基线、Rank IC及分组检验，预期超额收益为正且位于同日截面均值上方；收盘价高于20日均线、20日均线高于60日均线且量能达标；次日开盘仅在买入区间内评估",
        "initial_stop": round(current_price - 2 * atr, 2),
        "take_profit": round(current_price + 3 * atr, 2),
        "trailing_exit": "收盘价跌破20日均线，或收盘价从持仓最高收盘价回撤2倍ATR，下一交易日开盘评估退出",
        "holding_period": "最多20个交易日；每日复核",
        "generated_at": generated.isoformat(timespec="seconds"),
        "valid_until": valid_until.isoformat(timespec="seconds"),
        "invalidation": "参考条件、历史数据或行情失效；开盘价超出买入区间；无法成交时不假设成交",
        "calculation": "参考价为最新完整日线收盘价；买入区间±0.25ATR，止损−2ATR，止盈+3ATR",
        "risk_note": "参考价格不等同实际成交价；涨跌停、停牌和账户权限可能导致无法交易。",
        "trend_ok": bool(trend_ok),
        "volume_ok": bool(volume_ok),
        "model_relative_signal_positive": bool(forecast_ok) if model_ready else None,
    }


def plan_from_features(
    row: pd.Series,
    reference_price: float,
    relative_signal: float | None,
    model_ready: bool,
    **metadata: Any,
) -> dict[str, object]:
    """将因子换算至参考价格基准，并调用统一交易计划规则。"""
    ma20_gap = row.get("ma20_gap")
    ma60_gap = row.get("ma60_gap")
    return build_trade_plan(
        current_price=reference_price,
        atr_pct=row.get("atr_pct"),
        ma20=reference_price / (1 + ma20_gap) if _finite(ma20_gap) and ma20_gap > -1 else None,
        ma60=reference_price / (1 + ma60_gap) if _finite(ma60_gap) and ma60_gap > -1 else None,
        relative_signal=relative_signal,
        model_ready=model_ready,
        volume=row.get("volume"),
        average_volume=row.get("volume_ma20"),
        **metadata,
    )
