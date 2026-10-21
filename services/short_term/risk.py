"""独立于分钟模型的短线持仓基础风险检查。"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from services.intraday_data import quote_is_fresh

SHANGHAI = ZoneInfo("Asia/Shanghai")


def evaluate_position_risks(
    positions: list[dict[str, Any]], quotes: dict[str, dict[str, Any]], now: datetime | None = None
) -> list[dict[str, Any]]:
    """只根据新鲜报价和事前设置的批次规则生成提醒，不模拟成交。"""
    now = now or datetime.now(SHANGHAI)
    events = []
    for position in positions:
        code = str(position["code"])
        quote = quotes.get(code, {})
        if not quote_is_fresh(quote, now) or quote.get("price") is None:
            events.append(
                {
                    "position_id": int(position["id"]),
                    "code": code,
                    "type": "quote_stale",
                    "status": "coverage_gap",
                    "rule": "行情新鲜度",
                    "message": "缺少有效新鲜行情，当前持仓无法完成价格风控检查。",
                    "quote_time": quote.get("quote_time"),
                }
            )
            continue
        price = float(quote["price"])
        if price <= 0:
            events.append(
                {
                    "position_id": int(position["id"]),
                    "code": code,
                    "type": "quote_invalid",
                    "status": "coverage_gap",
                    "rule": "行情有效性",
                    "message": "最新行情价格无效，当前持仓无法完成价格风控检查。",
                    "quote_time": quote.get("quote_time"),
                }
            )
            continue
        rule = None
        stop = position.get("stop_loss_price")
        take = position.get("take_profit_price")
        trailing_pct = position.get("trailing_stop_pct")
        high_watermark = max(float(position.get("high_watermark_price") or position["average_buy_price"]), price)
        trailing_stop = high_watermark * (1.0 - float(trailing_pct) / 100.0) if trailing_pct is not None else None
        invalidation = position.get("strategy_invalidation_price")
        planned_exit = position.get("planned_exit_date")
        if planned_exit and str(planned_exit)[:10] <= now.date().isoformat():
            rule = "计划退出日"
        elif invalidation is not None and price <= float(invalidation):
            rule = "策略失效价"
        elif stop is not None and price <= float(stop):
            rule = "止损价"
        elif trailing_stop is not None and price <= trailing_stop:
            rule = "移动止盈"
        elif take is not None and price >= float(take):
            rule = "止盈价"
        if rule:
            available = int(position.get("available_shares", 0))
            events.append(
                {
                    "position_id": int(position["id"]),
                    "code": code,
                    "type": "exit_trigger",
                    "status": "exit_reminder" if available > 0 else "blocked_unavailable",
                    "rule": rule,
                    "message": "退出条件已触发；请核对可卖数量并人工确认成交。" if available > 0 else "退出条件已触发，但该批次没有已核对可卖数量。",
                    "reference_price": price,
                    "available_shares": available,
                    "high_watermark_price": high_watermark,
                    "trailing_stop_price": trailing_stop,
                    "quote_time": quote.get("quote_time"),
                }
            )
    return events


def evaluate_intent_triggers(
    intents: list[dict[str, Any]], quotes: dict[str, dict[str, Any]], now: datetime | None = None
) -> list[dict[str, Any]]:
    """检查买卖任务的限价方向和有效期，只生成待人工确认的提醒。"""
    now = now or datetime.now(SHANGHAI)
    events = []
    for intent in intents:
        if intent.get("status") != "pending":
            continue
        intent_id = int(intent["id"])
        details = intent.get("details") or {}
        valid_until = intent.get("valid_until")
        if valid_until is not None:
            expiry = valid_until if isinstance(valid_until, datetime) else datetime.fromisoformat(str(valid_until))
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=SHANGHAI)
            if now > expiry:
                events.append(
                    {
                        "intent_id": intent_id,
                        "code": intent["code"],
                        "type": "intent_expired",
                        "status": "expired",
                        "rule": "任务有效期",
                        "message": "任务有效期已结束，未将提醒视作订单或成交。",
                    }
                )
                continue
        limit_price = intent.get("limit_price")
        if limit_price is None:
            continue
        quote = quotes.get(str(intent["code"]), {})
        if not quote_is_fresh(quote, now) or quote.get("price") is None:
            events.append(
                {
                    "intent_id": intent_id,
                    "code": intent["code"],
                    "type": "intent_quote_stale",
                    "status": "coverage_gap",
                    "rule": "价格条件监控",
                    "message": "缺少有效新鲜行情，无法检查任务价格条件。",
                    "quote_time": quote.get("quote_time"),
                }
            )
            continue
        price = float(quote["price"])
        direction = details.get("price_direction") or ("lte" if intent["side"] == "BUY" else "gte")
        matched = price <= float(limit_price) if direction == "lte" else price >= float(limit_price)
        if not matched:
            continue
        has_quantity = intent.get("intended_shares") is not None
        events.append(
            {
                "intent_id": intent_id,
                "code": intent["code"],
                "type": "intent_trigger",
                "status": "triggered" if has_quantity else "triggered_no_quantity",
                "rule": intent["trigger_rule"],
                "message": "价格条件已触发；请核对账户与盘口并人工决定是否执行。",
                "side": intent["side"],
                "reference_price": price,
                "limit_price": float(limit_price),
                "intended_shares": intent.get("intended_shares"),
                "quote_time": quote.get("quote_time"),
            }
        )
    return events


def evaluate_minute_entry_confirmations(
    intents: list[dict[str, Any]],
    minute_bars: dict[str, pd.DataFrame],
    quotes: dict[str, dict[str, Any]],
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """只用 as-of 时点前已完成的一分钟线检查预登记入场条件。"""
    now = now or datetime.now(SHANGHAI)
    now_ts = pd.Timestamp(now)
    now_ts = now_ts.tz_localize(SHANGHAI) if now_ts.tzinfo is None else now_ts.tz_convert(SHANGHAI)
    cutoff = now_ts.floor("min")
    events = []
    for intent in intents:
        details = intent.get("details") or {}
        if intent.get("side") != "BUY" or not details.get("m3_enabled"):
            continue
        intent_id = int(intent["id"])
        code = str(intent["code"])
        event = {"intent_id": intent_id, "code": code, "type": "m3_entry_asof", "as_of": now_ts.isoformat()}
        frame = minute_bars.get(code)
        if frame is None or frame.empty:
            events.append({**event, "status": "coverage_gap", "reason": "没有分钟线"})
            continue
        required = {"date", "open", "close", "volume", "amount"}
        if not required.issubset(frame.columns):
            events.append({**event, "status": "coverage_gap", "reason": "分钟线缺少必要字段"})
            continue
        clean = frame.copy()
        clean["date"] = pd.to_datetime(clean["date"], errors="coerce", utc=True).dt.tz_convert(SHANGHAI)
        for column in ("open", "close", "volume", "amount"):
            clean[column] = pd.to_numeric(clean[column], errors="coerce")
        clean = clean.dropna(subset=["date", "open", "close", "volume", "amount"])
        clean = clean.loc[(clean["date"] < cutoff) & (clean["volume"] > 0)].sort_values("date")
        current_day = clean.loc[clean["date"].dt.date == now_ts.date()].copy()
        session_bars = current_day.copy()
        created_at = intent.get("created_at")
        if created_at is not None and not current_day.empty:
            created_ts = pd.Timestamp(created_at)
            created_ts = created_ts.tz_localize(SHANGHAI) if created_ts.tzinfo is None else created_ts.tz_convert(SHANGHAI)
            if created_ts.date() == now_ts.date():
                current_day = current_day.loc[current_day["date"] > created_ts]
        if current_day.empty:
            events.append({**event, "status": "coverage_gap", "reason": "当前交易日尚无已完成分钟线"})
            continue
        latest = current_day["date"].iloc[-1]
        age_seconds = float((now_ts - latest).total_seconds())
        if age_seconds < 0 or age_seconds > 125:
            events.append({**event, "status": "coverage_gap", "reason": "最新分钟线超过125秒", "latest_bar": latest.isoformat()})
            continue
        recent = current_day.tail(5)
        gaps = recent["date"].diff().dropna().dt.total_seconds()
        if len(recent) < 5 or not gaps.eq(60).all():
            events.append({**event, "status": "coverage_gap", "reason": "最近5根分钟线不完整或不连续", "latest_bar": latest.isoformat()})
            continue
        returns = session_bars["close"].pct_change(fill_method=None).dropna().tail(20)
        if len(returns) < 20:
            events.append({**event, "status": "coverage_gap", "reason": "实现波动率至少需要21根已完成分钟线", "latest_bar": latest.isoformat()})
            continue
        slots = set(recent["date"].dt.strftime("%H:%M"))
        historical = clean.loc[clean["date"].dt.date < now_ts.date()].copy()
        historical["slot"] = historical["date"].dt.strftime("%H:%M")
        historical = historical.loc[historical["slot"].isin(slots)]
        slot_counts = historical.groupby(historical["date"].dt.date)["slot"].nunique()
        eligible_days = slot_counts[slot_counts == len(slots)].index
        baseline = historical.loc[historical["date"].dt.date.isin(eligible_days)].groupby(
            historical["date"].dt.date
        )["volume"].sum()
        if len(baseline) < 3:
            events.append({**event, "status": "coverage_gap", "reason": "同一时段历史成交量基线不足3个交易日", "latest_bar": latest.isoformat()})
            continue
        session_volume = float(session_bars["volume"].sum())
        session_shares = float((session_bars["volume"] * 100.0).sum())
        session_amount = float(session_bars["amount"].sum())
        if session_volume <= 0 or session_shares <= 0 or session_amount <= 0:
            events.append({**event, "status": "coverage_gap", "reason": "分钟成交量或成交额无效"})
            continue
        metrics = {
            "return_5m": float(recent["close"].iloc[-1] / recent["open"].iloc[0] - 1.0),
            "volume_ratio_same_time": float(recent["volume"].sum() / baseline.median()),
            "vwap_deviation": float(current_day["close"].iloc[-1] / (session_amount / session_shares) - 1.0),
            "realized_volatility_20m": float(np.std(returns.to_numpy(dtype=float), ddof=0)),
            "latest_bar": latest.isoformat(),
            "decision_time": now_ts.isoformat(),
            "earliest_action_time": (latest + pd.Timedelta(minutes=1)).isoformat(),
            "spread_status": "not_available_from_current_quote_contract",
        }
        thresholds = (
            details.get("m3_min_5m_return"),
            details.get("m3_min_volume_ratio"),
            details.get("m3_max_abs_vwap_deviation"),
            details.get("m3_max_realized_volatility"),
            details.get("m3_max_spread_bps"),
            details.get("m3_max_round_trip_cost_bps"),
        )
        if any(value is None for value in thresholds) or any(float(thresholds[index]) <= 0 for index in (1, 2, 3, 4, 5)):
            events.append({**event, "status": "not_configured", "reason": "分钟收益、量比、VWAP偏离、波动率、价差及总成本门槛必须预先设置", **metrics})
            continue
        quote = quotes.get(code, {})
        if not quote_is_fresh(quote, now):
            events.append({**event, "status": "coverage_gap", "reason": "缺少有效实时价格确认", **metrics})
            continue
        bid = quote.get("best_bid")
        ask = quote.get("best_ask")
        if bid is None or ask is None or float(bid) <= 0 or float(ask) < float(bid):
            events.append({**event, "status": "coverage_gap", "reason": "报价源没有有效买一卖一，无法计算价差", **metrics})
            continue
        mid = (float(bid) + float(ask)) / 2.0
        spread_bps = (float(ask) - float(bid)) / mid * 10000.0
        cost_profile = details.get("cost_profile_snapshot") or {}
        notional = float(intent.get("limit_price") or 0) * int(intent.get("intended_shares") or 0)
        if notional <= 0 or any(cost_profile.get(field) is None for field in ("commission_rate", "minimum_commission", "stamp_duty_rate", "slippage_rate")):
            events.append({**event, "status": "coverage_gap", "reason": "缺少计划金额或已冻结的账户费率", **metrics})
            continue
        commission = max(notional * float(cost_profile["commission_rate"]), float(cost_profile["minimum_commission"]))
        round_trip_cost_bps = (
            2.0 * commission / notional
            + float(cost_profile["stamp_duty_rate"])
            + 2.0 * float(cost_profile["slippage_rate"])
            + 2.0 * spread_bps / 10000.0
        ) * 10000.0
        metrics["spread_bps"] = spread_bps
        metrics["estimated_round_trip_cost_bps"] = round_trip_cost_bps
        checks = {
            "return_5m": metrics["return_5m"] >= float(thresholds[0]),
            "volume_ratio_same_time": metrics["volume_ratio_same_time"] >= float(thresholds[1]),
            "vwap_deviation": abs(metrics["vwap_deviation"]) <= float(thresholds[2]),
            "realized_volatility_20m": metrics["realized_volatility_20m"] <= float(thresholds[3]),
            "spread_bps": spread_bps <= float(thresholds[4]),
            "estimated_round_trip_cost_bps": round_trip_cost_bps <= float(thresholds[5]),
        }
        status = "confirmed" if all(checks.values()) else "not_confirmed"
        events.append({**event, "status": status, "checks": checks, **metrics})
    return events
