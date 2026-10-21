"""基于原始日线与点时主数据进行保守的短线撮合回放。"""

from __future__ import annotations

import math
from bisect import bisect_right
from datetime import datetime, time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

SHANGHAI = ZoneInfo("Asia/Shanghai")
BAR_COLUMNS = ("code", "date", "open", "high", "low", "close", "volume")
UNIVERSE_COLUMNS = ("code", "date", "available_at", "is_member")
STATUS_COLUMNS = (
    "code", "date", "available_at", "is_listed", "is_st",
    "is_suspended", "is_limit_up", "is_limit_down",
)
TERM_FIELDS = (
    "order_shares", "lot_size", "volume_unit", "max_participation_rate",
    "entry_limit_offset_bps", "exit_limit_offset_bps", "entry_valid_sessions",
    "exit_grace_sessions", "commission_rate", "minimum_commission",
    "stamp_duty_rate", "slippage_bps", "allow_st",
)


def _as_shanghai(value: Any) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    return parsed.tz_localize(SHANGHAI) if parsed.tzinfo is None else parsed.tz_convert(SHANGHAI)


def _normalize_codes(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    result = frame.copy()
    result["code"] = result["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    if result[["code", "date"]].isna().any().any():
        raise ValueError(f"{name} 含无效股票代码或日期")
    if not result["code"].str.fullmatch(r"\d{6}").all():
        raise ValueError(f"{name} 含非六位证券代码")
    return result


def _boolean_column(frame: pd.DataFrame, column: str, name: str) -> pd.Series:
    """只接受明确布尔值，避免 bool(\"false\") 被误判为真。"""
    def parse(value: Any) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        raise ValueError(f"{name}.{column} 含非布尔状态值")

    return frame[column].map(parse)


def _validate_terms(terms: dict[str, Any] | None) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(terms, dict):
        return None, ["缺少冻结的 execution_terms"]
    missing = [field for field in TERM_FIELDS if terms.get(field) is None]
    if missing:
        return None, [f"execution_terms 缺少字段：{', '.join(missing)}"]
    try:
        normalized = dict(terms)
        for field in ("order_shares", "lot_size", "entry_valid_sessions", "exit_grace_sessions"):
            normalized[field] = int(terms[field])
            if float(terms[field]) != normalized[field]:
                raise ValueError(field)
        for field in (
            "max_participation_rate", "entry_limit_offset_bps", "exit_limit_offset_bps",
            "commission_rate", "minimum_commission", "stamp_duty_rate", "slippage_bps",
        ):
            normalized[field] = float(terms[field])
    except (TypeError, ValueError, OverflowError):
        return None, ["execution_terms 含无效数值"]
    if normalized["order_shares"] <= 0 or normalized["lot_size"] <= 0:
        return None, ["order_shares 与 lot_size 必须为正整数"]
    if normalized["order_shares"] % normalized["lot_size"]:
        return None, ["order_shares 必须是 lot_size 的整数倍"]
    if normalized["entry_valid_sessions"] < 1 or normalized["exit_grace_sessions"] < 0:
        return None, ["入场有效期至少为 1 个交易日，退出宽限期不得为负数"]
    if normalized["volume_unit"] not in ("shares", "lots"):
        return None, ["volume_unit 必须明确为 shares 或 lots"]
    if not isinstance(normalized["allow_st"], bool):
        return None, ["allow_st 必须是冻结的布尔值"]
    if not 0 < normalized["max_participation_rate"] <= 1:
        return None, ["max_participation_rate 必须在 0 到 1 之间"]
    numeric_fields = (
        "entry_limit_offset_bps", "exit_limit_offset_bps", "commission_rate",
        "minimum_commission", "stamp_duty_rate", "slippage_bps",
    )
    if any(not math.isfinite(normalized[field]) or normalized[field] < 0 for field in numeric_fields):
        return None, ["费用、滑点和限价偏移必须为有限非负数"]
    return normalized, []


def _pit_index(frame: pd.DataFrame) -> dict[tuple[str, pd.Timestamp], tuple[list[pd.Timestamp], list[dict[str, Any]]]]:
    indexed = {}
    for (code, session), group in frame.groupby(["code", "date"], sort=False):
        ordered = group.sort_values("available_at")
        indexed[(code, pd.Timestamp(session))] = (
            ordered["available_at"].tolist(),
            ordered.to_dict("records"),
        )
    return indexed


def _pit_row(
    index: dict[tuple[str, pd.Timestamp], tuple[list[pd.Timestamp], list[dict[str, Any]]]],
    code: str,
    session: pd.Timestamp,
    cutoff: pd.Timestamp,
) -> dict[str, Any] | None:
    records = index.get((code, session))
    if records is None:
        return None
    position = bisect_right(records[0], cutoff) - 1
    return records[1][position] if position >= 0 else None


def _fee(notional: float, terms: dict[str, Any], side: str) -> float:
    commission = max(notional * terms["commission_rate"], terms["minimum_commission"])
    stamp = notional * terms["stamp_duty_rate"] if side == "SELL" else 0.0
    return commission + stamp


def _daily_capacity(bar: dict[str, Any], terms: dict[str, Any]) -> int:
    volume = float(bar["volume"])
    if terms["volume_unit"] == "lots":
        volume *= terms["lot_size"]
    capacity = int(math.floor(volume * terms["max_participation_rate"] / terms["lot_size"])) * terms["lot_size"]
    return max(capacity, 0)


def run_execution_replay(
    signals: pd.DataFrame,
    daily_bars: pd.DataFrame | None,
    universe_history: pd.DataFrame | None,
    security_status_history: pd.DataFrame | None,
    terms: dict[str, Any] | None,
    horizon: int,
    top_k: int = 20,
) -> dict[str, Any]:
    """回放样本外信号；日线触价近似不会被描述为真实队列成交。"""
    blockers: list[str] = []
    normalized_terms, term_errors = _validate_terms(terms)
    blockers.extend(term_errors)
    if not isinstance(signals, pd.DataFrame) or signals.empty:
        blockers.append("没有可回放的样本外信号")
    elif not {"code", "date", "ranking_score", "reference_price"}.issubset(signals.columns):
        blockers.append("样本外信号必须包含 code、date、ranking_score、reference_price")
    if not isinstance(daily_bars, pd.DataFrame) or daily_bars.empty:
        blockers.append("缺少不复权原始日线")
    elif not set(BAR_COLUMNS).issubset(daily_bars.columns):
        blockers.append("原始日线缺少 OHLCV 字段")
    if not isinstance(universe_history, pd.DataFrame) or universe_history.empty:
        blockers.append("缺少带 available_at 的历史股票池快照")
    elif not set(UNIVERSE_COLUMNS).issubset(universe_history.columns):
        blockers.append("历史股票池必须包含 code、date、available_at、is_member")
    if not isinstance(security_status_history, pd.DataFrame) or security_status_history.empty:
        blockers.append("缺少带 available_at 的历史证券状态")
    elif not set(STATUS_COLUMNS).issubset(security_status_history.columns):
        blockers.append("历史证券状态缺少上市、ST、停牌或涨跌停字段")
    if blockers:
        return {
            "status": "blocked_missing_inputs",
            "promotion_evidence": False,
            "blockers": list(dict.fromkeys(blockers)),
            "fills": 0,
            "trades": [],
        }
    if int(horizon) not in (1, 3, 5) or int(top_k) < 1:
        raise ValueError("horizon 必须为 1、3、5，且 top_k 必须为正数")

    try:
        bar_frame = _normalize_codes(daily_bars, "原始日线")
        universe = _normalize_codes(universe_history, "历史股票池")
        status = _normalize_codes(security_status_history, "历史证券状态")
        signal_frame = signals.copy()
        signal_frame["code"] = signal_frame["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
        signal_frame["date"] = pd.to_datetime(signal_frame["date"], errors="coerce").dt.normalize()
        signal_frame["ranking_score"] = pd.to_numeric(signal_frame["ranking_score"], errors="coerce")
        if signal_frame[["code", "date", "ranking_score"]].isna().any().any():
            raise ValueError("样本外信号含无效股票代码、日期或排序分数")
        signal_times = signal_frame["signal_time"] if "signal_time" in signal_frame else signal_frame["date"] + pd.Timedelta(hours=15, minutes=5)
        signal_frame["_signal_time"] = signal_times.map(_as_shanghai)
        for frame in (universe, status):
            frame["available_at"] = pd.to_datetime(frame["available_at"], errors="coerce", utc=True).dt.tz_convert(SHANGHAI)
            if frame["available_at"].isna().any():
                raise ValueError("点时主数据含无效 available_at")
        universe["is_member"] = _boolean_column(universe, "is_member", "历史股票池")
        for column in STATUS_COLUMNS[3:]:
            status[column] = _boolean_column(status, column, "历史证券状态")
        universe_index = _pit_index(universe)
        status_index = _pit_index(status)
        for column in ("open", "high", "low", "close", "volume"):
            bar_frame[column] = pd.to_numeric(bar_frame[column], errors="coerce")
        bar_frame = bar_frame.dropna(subset=["open", "high", "low", "close", "volume"])
        if bar_frame.duplicated(["code", "date"]).any():
            raise ValueError("原始日线存在重复股票日期")
    except (TypeError, ValueError, KeyError) as exc:
        return {
            "status": "blocked_invalid_inputs",
            "promotion_evidence": False,
            "blockers": [f"回放输入无法解析：{type(exc).__name__}: {exc}"],
            "fills": 0,
            "trades": [],
        }

    all_dates = sorted(pd.Timestamp(value) for value in bar_frame["date"].drop_duplicates())
    date_index = {value: index for index, value in enumerate(all_dates)}
    bars_by_code = {code: group.set_index("date", drop=False) for code, group in bar_frame.groupby("code", sort=False)}
    selected_signals: list[dict[str, Any]] = []
    coverage_blockers: list[str] = []
    for (signal_date, _), group in signal_frame.groupby(["date", "_signal_time"], sort=True):
        eligible = []
        candidates = group.replace([np.inf, -np.inf], np.nan).dropna(subset=["ranking_score"])
        for row in candidates.to_dict("records"):
            membership = _pit_row(universe_index, row["code"], pd.Timestamp(signal_date), row["_signal_time"])
            if membership is None:
                coverage_blockers.append(f"{row['code']} {pd.Timestamp(signal_date).date()} 缺点时股票池记录")
            elif bool(membership["is_member"]):
                eligible.append(row)
        selected_signals.extend(sorted(eligible, key=lambda row: row["ranking_score"], reverse=True)[:int(top_k)])

    trades = []
    for signal in selected_signals:
        code = signal["code"]
        signal_date = pd.Timestamp(signal["date"])
        signal_index = date_index.get(signal_date)
        if signal_index is None or code not in bars_by_code:
            coverage_blockers.append(f"{code} {signal_date.date()} 缺少可对齐的原始日线")
            continue
        stock_bars = bars_by_code[code]
        entry_lots: list[dict[str, Any]] = []
        remaining_to_buy = int(normalized_terms["order_shares"])
        reasons: list[str] = []
        entry_end = min(signal_index + int(normalized_terms["entry_valid_sessions"]), len(all_dates) - 1)
        for session_index in range(signal_index + 1, entry_end + 1):
            session = all_dates[session_index]
            market_open = pd.Timestamp(datetime.combine(session.date(), time(9, 30), SHANGHAI))
            state = _pit_row(status_index, code, session, market_open)
            if state is None:
                reasons.append(f"{session.date()} 缺点时证券状态")
                coverage_blockers.append(f"{code} {session.date()} 缺点时证券状态")
                break
            if not bool(state["is_listed"]) or bool(state["is_suspended"]) or (bool(state["is_st"]) and not normalized_terms["allow_st"]):
                reasons.append(f"{session.date()} 不可买入：上市、停牌或 ST 条件不满足")
                continue
            if bool(state["is_limit_up"]):
                reasons.append(f"{session.date()} 涨停锁定，按未成交处理")
                continue
            if session not in stock_bars.index:
                reasons.append(f"{session.date()} 缺少可成交日线")
                coverage_blockers.append(f"{code} {session.date()} 缺可成交日线")
                continue
            bar = stock_bars.loc[session].to_dict()
            if float(bar["volume"]) <= 0 or float(bar["open"]) <= 0:
                reasons.append(f"{session.date()} 无有效成交量")
                continue
            reference_price = float(signal["reference_price"])
            if not math.isfinite(reference_price) or reference_price <= 0:
                coverage_blockers.append(f"{code} {signal_date.date()} 缺少有效的信号日原始收盘价")
                break
            limit_price = reference_price * (1.0 + normalized_terms["entry_limit_offset_bps"] / 10000.0)
            if float(bar["low"]) > limit_price:
                continue
            execution_price = min(float(bar["open"]), limit_price) * (1.0 + normalized_terms["slippage_bps"] / 10000.0)
            if execution_price > limit_price:
                continue
            capacity = _daily_capacity(bar, normalized_terms)
            filled = min(remaining_to_buy, capacity)
            filled = filled // normalized_terms["lot_size"] * normalized_terms["lot_size"]
            if filled <= 0:
                continue
            notional = filled * execution_price
            buy_cost = notional + _fee(notional, normalized_terms, "BUY")
            entry_lots.append(
                {
                    "entry_index": session_index,
                    "shares": filled,
                    "remaining": filled,
                    "cost": buy_cost,
                    "remaining_cost": buy_cost,
                }
            )
            remaining_to_buy -= filled
            if remaining_to_buy == 0:
                break

        if not entry_lots:
            trades.append({
                "code": code,
                "signal_date": signal_date.date().isoformat(),
                "status": "unfilled",
                "reason": "; ".join(reasons) or "限价未触及或参与率容量不足",
                "filled_shares": 0,
                "unfilled_shares": int(normalized_terms["order_shares"]),
            })
            continue

        total_buy_cost = float(sum(lot["cost"] for lot in entry_lots))
        total_sell_proceeds = 0.0
        sell_fills = 0
        blocked_exit = False
        for lot in entry_lots:
            scheduled_exit = lot["entry_index"] + int(horizon)
            last_exit = min(scheduled_exit + int(normalized_terms["exit_grace_sessions"]), len(all_dates) - 1)
            for session_index in range(max(scheduled_exit, lot["entry_index"] + 1), last_exit + 1):
                if lot["remaining"] == 0:
                    break
                session = all_dates[session_index]
                market_open = pd.Timestamp(datetime.combine(session.date(), time(9, 30), SHANGHAI))
                state = _pit_row(status_index, code, session, market_open)
                if state is None:
                    reasons.append(f"{session.date()} 缺点时证券状态")
                    coverage_blockers.append(f"{code} {session.date()} 缺点时证券状态")
                    blocked_exit = True
                    break
                if not bool(state["is_listed"]) or bool(state["is_suspended"]):
                    reasons.append(f"{session.date()} 不可卖出：退市或停牌")
                    continue
                if bool(state["is_limit_down"]):
                    reasons.append(f"{session.date()} 跌停锁定，延后至宽限期")
                    continue
                if session not in stock_bars.index:
                    reasons.append(f"{session.date()} 缺少退出日线")
                    coverage_blockers.append(f"{code} {session.date()} 缺退出日线")
                    continue
                bar = stock_bars.loc[session].to_dict()
                if float(bar["volume"]) <= 0 or float(bar["open"]) <= 0:
                    continue
                previous_date = all_dates[session_index - 1]
                if previous_date not in stock_bars.index:
                    reasons.append(f"{session.date()} 缺少事前可用的前收盘价")
                    coverage_blockers.append(f"{code} {session.date()} 缺前收盘价")
                    blocked_exit = True
                    break
                previous_close = float(stock_bars.loc[previous_date]["close"])
                if not math.isfinite(previous_close) or previous_close <= 0:
                    reasons.append(f"{session.date()} 前收盘价无效")
                    coverage_blockers.append(f"{code} {session.date()} 前收盘价无效")
                    blocked_exit = True
                    break
                limit_price = previous_close * (1.0 - normalized_terms["exit_limit_offset_bps"] / 10000.0)
                if float(bar["high"]) < limit_price:
                    continue
                execution_price = max(float(bar["open"]), limit_price) * (1.0 - normalized_terms["slippage_bps"] / 10000.0)
                if execution_price < limit_price:
                    continue
                capacity = _daily_capacity(bar, normalized_terms)
                filled = min(int(lot["remaining"]), capacity)
                filled = filled // normalized_terms["lot_size"] * normalized_terms["lot_size"]
                if filled <= 0:
                    continue
                notional = filled * execution_price
                total_sell_proceeds += notional - _fee(notional, normalized_terms, "SELL")
                lot["remaining_cost"] = max(
                    lot["remaining_cost"] - lot["cost"] * filled / lot["shares"],
                    0.0,
                )
                lot["remaining"] -= filled
                sell_fills += 1

        bought_shares = int(sum(lot["shares"] for lot in entry_lots))
        remaining_shares = int(sum(lot["remaining"] for lot in entry_lots))
        closed = remaining_shares == 0 and not blocked_exit
        trade_status = "unclosed" if not closed else ("closed_partial_entry" if remaining_to_buy else "closed")
        remaining_cost_basis = float(sum(lot["remaining_cost"] for lot in entry_lots))
        mark_date = None
        mark_price = None
        marked_market_value = None
        unrealized_pnl = None
        unrealized_return = None
        if remaining_shares > 0 and not stock_bars.empty:
            last_bar = stock_bars.sort_index(kind="stable").iloc[-1]
            candidate_mark = pd.to_numeric(pd.Series([last_bar.get("close")]), errors="coerce").iloc[0]
            if pd.notna(candidate_mark) and float(candidate_mark) > 0:
                mark_date = pd.Timestamp(last_bar["date"]).date().isoformat()
                mark_price = float(candidate_mark)
                marked_market_value = remaining_shares * mark_price
                unrealized_pnl = marked_market_value - remaining_cost_basis
                unrealized_return = unrealized_pnl / remaining_cost_basis if remaining_cost_basis > 0 else None

        trades.append({
            "code": code,
            "signal_date": signal_date.date().isoformat(),
            "status": trade_status,
            "score": float(signal["ranking_score"]),
            "filled_shares": bought_shares,
            "unfilled_shares": remaining_to_buy,
            "entry_completion_status": "filled" if remaining_to_buy == 0 else "partially_filled",
            "remaining_sellable_shares": remaining_shares,
            "remaining_cost_basis_including_buy_fees": remaining_cost_basis,
            "mark_date": mark_date,
            "mark_close_price": mark_price,
            "open_market_value_before_exit_cost": marked_market_value,
            "unrealized_pnl_before_exit_cost": unrealized_pnl,
            "unrealized_return_before_exit_cost": unrealized_return,
            "entry_fill_count": len(entry_lots),
            "exit_fill_count": sell_fills,
            "buy_cost_including_minimum_commission": total_buy_cost,
            "realized_net_proceeds": total_sell_proceeds if closed else None,
            "net_return": total_sell_proceeds / total_buy_cost - 1.0 if closed and total_buy_cost > 0 else None,
            "reason": "; ".join(reasons) or None,
            "execution_basis": "daily_bar_touch_proxy_not_order_queue_truth",
            "commission_basis": "minimum_commission_applied_per_daily_fill",
        })

    closed_returns = [
        row["net_return"] for row in trades
        if row.get("status") in ("closed", "closed_partial_entry") and row.get("net_return") is not None
    ]
    open_trades = [row for row in trades if row.get("status") == "unclosed"]
    marked_open_trades = [row for row in open_trades if row.get("open_market_value_before_exit_cost") is not None]
    open_market_value = float(sum(row["open_market_value_before_exit_cost"] for row in marked_open_trades))
    open_cost_basis = float(sum(row["remaining_cost_basis_including_buy_fees"] for row in marked_open_trades))
    open_unrealized_pnl = float(sum(row["unrealized_pnl_before_exit_cost"] for row in marked_open_trades))
    return {
        "status": "partial_coverage" if coverage_blockers else "daily_bar_proxy_not_queue_truth",
        "promotion_evidence": False,
        "label_basis": "unadjusted_daily_ohlcv_with_point_in_time_universe_and_security_status",
        "execution_basis": "daily_bar_touch_proxy_not_order_queue_truth",
        "horizon": int(horizon),
        "top_k": int(top_k),
        "signal_count": int(len(selected_signals)),
        "trade_count": int(len(trades)),
        "closed_trade_count": int(sum(row.get("status") in ("closed", "closed_partial_entry") for row in trades)),
        "partial_entry_trade_count": int(sum(row.get("entry_completion_status") == "partially_filled" for row in trades)),
        "unfilled_trade_count": int(sum(row.get("status") == "unfilled" for row in trades)),
        "unclosed_trade_count": int(sum(row.get("status") == "unclosed" for row in trades)),
        "entry_fill_count": int(sum(row.get("entry_fill_count", 0) for row in trades)),
        "exit_fill_count": int(sum(row.get("exit_fill_count", 0) for row in trades)),
        "mean_closed_net_return": float(np.mean(closed_returns)) if closed_returns else None,
        "portfolio_risk": {
            "status": "gross_exposure_only_nav_missing",
            "open_position_count": len(open_trades),
            "marked_open_position_count": len(marked_open_trades),
            "gross_open_market_value_before_exit_cost": open_market_value,
            "remaining_open_cost_basis": open_cost_basis,
            "unrealized_pnl_before_exit_cost": open_unrealized_pnl,
            "max_drawdown": None,
            "blockers": [
                "完整组合收益、回撤与资金占用需要冻结的初始净值、账户资金约束及并发订单分配规则。",
                "未接入可核验的历史公司行为数据，未平仓估值不含分红、送转或税费调整。",
            ],
        },
        "corporate_actions": {
            "status": "not_available",
            "promotion_evidence": False,
            "blockers": ["未找到可核验并按可得时间记录的历史分红、送转及除权数据源。"],
        },
        "coverage_blockers": list(dict.fromkeys(coverage_blockers))[:100],
        "trades": trades,
    }
