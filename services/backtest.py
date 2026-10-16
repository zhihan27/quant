"""按日记账、保留未平仓头寸的模型及规则样本外回测。"""

from __future__ import annotations

import math
import uuid
from typing import Any

import numpy as np
import pandas as pd

from config import settings
from services.data_cleaning import open_price_fillable, price_limit_rate
from services.feature_engineering import build_features
from services.stock_scoring import walk_forward_predictions
from services.timing_strategy import plan_from_features


def _limit_rate(code: str | None, date: Any = None, is_st: bool = False) -> float:
    """保留原有调用名，涨跌幅制度统一由清洗模块给出，避免两处规则漂移。"""
    return price_limit_rate(code, date, is_st)


def _is_locked(bar: pd.Series, direction: str, limit_rate: float) -> bool:
    """识别全天封板，仅用于行情诊断，不用于推断开盘成交。"""
    volume = bar.get("volume")
    if pd.isna(volume) or float(volume) <= 0:
        return True
    pct = bar.get("pct_change")
    if pd.isna(pct):
        return False
    size = float(bar["high"] - bar["low"])
    return size <= max(0.01, float(bar["close"]) * 0.001) and (
        (direction == "buy" and float(pct) >= limit_rate * 98)
        or (direction == "sell" and float(pct) <= -limit_rate * 98)
    )


def _open_fillable(bar: pd.Series, direction: str, previous_close: float, limit_rate: float) -> bool:
    """仅用开盘价、已知昨收及停牌状态作保守成交判断，封板开盘不假设排队成功。"""
    return open_price_fillable(bar, direction, previous_close, limit_rate)


def _metrics(
    trades: list[dict[str, Any]],
    test_bars: int,
    equity_path: list[dict[str, Any]],
    initial_equity: float = 1.0,
    initial_gross: float = 1.0,
) -> dict[str, Any]:
    """用含空仓及未平仓市值的逐日净值计算风险，交易胜率仅统计已平仓交易。"""
    equity = np.asarray([initial_equity] + [row["equity"] for row in equity_path], dtype=float)
    daily_returns = equity[1:] / equity[:-1] - 1
    peak = np.maximum.accumulate(equity)
    returns = np.asarray([trade["net_return"] for trade in trades], dtype=float)
    total_return = float(equity[-1] / initial_equity - 1)
    deviation = float(np.std(daily_returns, ddof=1)) if len(daily_returns) > 1 else 0.0
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    return {
        "trades": len(trades),
        "total_return": total_return,
        "annualized_return": float((1 + total_return) ** (252 / max(test_bars, 1)) - 1),
        "maximum_drawdown": float(np.min(equity / peak - 1)),
        "sharpe_ratio": float(daily_returns.mean() / deviation * math.sqrt(252)) if deviation > 1e-12 else None,
        "win_rate": float(np.mean(returns > 0)) if len(returns) else None,
        "profit_loss_ratio": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else None,
        "annualized_turnover": float(sum(row["turnover"] for row in equity_path) * 252 / max(test_bars, 1)),
        "gross_return": float(equity_path[-1]["gross_equity"] / initial_gross - 1) if equity_path else 0.0,
    }


def run_backtest(
    history: pd.DataFrame,
    code: str | None = None,
    benchmark: pd.DataFrame | None = None,
    commission_rate: float = settings.DEFAULT_COSTS["commission_rate"],
    stamp_duty_rate: float = settings.DEFAULT_COSTS["stamp_duty_rate"],
    slippage_rate: float = settings.DEFAULT_COSTS["slippage_rate"],
    holding_days: int = 20,
    is_st: bool = False,
    strategy: str = "model",
    adjusted_history: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """输入不复权成交行情及可选复权因子行情，返回逐日净值、成交、未平仓和滚动验证。"""
    if history.empty or len(history) < 180:
        return {"status": f"样本不足：需要至少180根日线，当前{len(history)}", "metrics": None}
    if strategy not in ("model", "trend") or not 1 <= holding_days <= 60:
        raise ValueError("策略须为 model/trend，持仓期限须为1至60个交易日。")
    rates = [commission_rate, stamp_duty_rate, slippage_rate]
    if not all(np.isfinite(rate) and 0 <= rate < 1 for rate in rates) or sum(rates) >= 1:
        raise ValueError("交易费用必须是有效的非负比例，费用合计须小于1。")
    bars = history.copy()
    bars["date"] = pd.to_datetime(bars["date"], errors="raise")
    bars = bars.sort_values("date").reset_index(drop=True)
    if bars["date"].duplicated().any() or not np.isfinite(bars["close"]).all() or (bars["close"] <= 0).any():
        raise ValueError("历史行情存在重复日期或无效收盘价。")
    active = bars["volume"].gt(0)
    candles = bars.loc[active, ["open", "high", "low", "close"]]
    if not np.isfinite(candles.to_numpy()).all() or (candles <= 0).any(axis=None):
        raise ValueError("有成交的日线包含无效价格。")
    if (
        candles["high"].lt(candles[["open", "close"]].max(axis=1)).any()
        or candles["low"].gt(candles[["open", "close"]].min(axis=1)).any()
    ):
        raise ValueError("日线开收盘价超出最高最低价范围。")
    factor_bars = bars
    if adjusted_history is not None:
        factor_bars = adjusted_history.copy()
        factor_bars["date"] = pd.to_datetime(factor_bars["date"], errors="raise")
        if factor_bars["date"].duplicated().any():
            raise ValueError("复权历史日期重复。")
        factor_bars = factor_bars.set_index("date").reindex(bars["date"]).reset_index()
        if factor_bars["close"].isna().any() or (factor_bars["close"] <= 0).any():
            raise ValueError("复权与不复权日线日期不一致，无法回测。")
    features = build_features(factor_bars, benchmark)
    count = len(bars)
    warmup = max(320 if strategy == "model" else 80, count // 2)
    if count - warmup < 30:
        return {"status": "样本不足：模型训练后需保留至少30根测试日线", "metrics": None}
    test_size = (count - warmup) // 3
    if strategy == "model":
        predictions, training_log = walk_forward_predictions(features, warmup, code=code)
    else:
        predictions = pd.Series(0.01, index=features.index)
        training_log = []

    cash = gross_cash = 1.0
    shares = gross_shares = 0.0
    position = None
    pending_entry = None
    pending_exit = None
    trades = []
    equity_path = []
    unfilled_entries = unfilled_exits = 0
    buy_cost = commission_rate + slippage_rate
    sell_cost = commission_rate + stamp_duty_rate + slippage_rate
    for index in range(warmup, count):
        bar = bars.iloc[index]
        row = features.iloc[index]
        previous_close = float(bars.iloc[index - 1]["close"])
        daily_turnover = 0.0
        prior_equity = equity_path[-1]["equity"] if equity_path else 1.0
        limit_rate = _limit_rate(code, bar["date"], is_st)
        # 复权比例只用于公司行为的权益近似，不把复权价当作可成交价。
        adjustment = 1.0
        if adjusted_history is not None:
            raw_ratio = float(bar["close"]) / previous_close
            adjusted_ratio = float(factor_bars.iloc[index]["close"]) / float(factor_bars.iloc[index - 1]["close"])
            candidate = adjusted_ratio / raw_ratio
            if abs(candidate - 1) > 0.001:
                adjustment = candidate
                previous_close /= adjustment
                if position is not None:
                    shares *= adjustment
                    gross_shares *= adjustment
                    position["corporate_multiplier"] *= adjustment
                    for key in ("stop", "target", "peak"):
                        position[key] /= adjustment
        exit_price = None
        exit_reason = None
        if position is not None and index > position["entry_index"]:
            can_sell = _open_fillable(bar, "sell", previous_close, limit_rate)
            if pending_exit or index - position["entry_index"] >= holding_days:
                if can_sell:
                    exit_price = float(bar["open"])
                    exit_reason = pending_exit or "持有期结束"
                else:
                    pending_exit = pending_exit or "持有期结束"
                    unfilled_exits += 1
            elif can_sell:
                # 开盘跳空以开盘价成交；盘中触价必须位于当日价格范围内。
                opening = float(bar["open"])
                low, high = float(bar["low"]), float(bar["high"])
                if opening <= position["stop"]:
                    exit_price, exit_reason = opening, "ATR初始止损"
                elif opening >= position["target"]:
                    exit_price, exit_reason = opening, "ATR止盈"
                elif low <= position["stop"] <= high:
                    exit_price, exit_reason = position["stop"], "ATR初始止损"
                elif low <= position["target"] <= high:
                    exit_price, exit_reason = position["target"], "ATR止盈"
            elif float(bar["close"]) <= position["stop"]:
                pending_exit = "止损退出待成交"
                unfilled_exits += 1
        if exit_price is not None:
            daily_turnover += shares * exit_price / prior_equity
            cash = shares * exit_price * (1 - sell_cost)
            gross_cash = gross_shares * exit_price
            multiplier = position["corporate_multiplier"]
            gross_return = exit_price * multiplier / position["entry_price"] - 1
            net_return = exit_price * multiplier * (1 - sell_cost) / (position["entry_price"] * (1 + buy_cost)) - 1
            trades.append({
                "entry_date": position["entry_date"],
                "exit_date": bar["date"].strftime("%Y-%m-%d"),
                "entry_price": position["entry_price"],
                "exit_price": exit_price,
                "gross_return": gross_return,
                "net_return": net_return,
                "holding_days": index - position["entry_index"],
                "exit_reason": exit_reason,
                "fold": min((index - warmup) // test_size + 1, 3),
            })
            shares = gross_shares = 0.0
            position = None
            pending_exit = None
        if position is None and pending_entry is not None and exit_price is None:
            lower, upper = pending_entry["buy_zone"]
            opening = float(bar["open"])
            if adjustment != 1 or not _open_fillable(bar, "buy", previous_close, limit_rate) or not lower <= opening <= upper:
                unfilled_entries += 1
            else:
                shares = cash / (opening * (1 + buy_cost))
                gross_shares = gross_cash / opening
                daily_turnover += shares * opening / prior_equity
                cash = gross_cash = 0.0
                position = {
                    "entry_index": index,
                    "entry_date": bar["date"].strftime("%Y-%m-%d"),
                    "entry_price": opening,
                    "stop": pending_entry["initial_stop"],
                    "target": pending_entry["take_profit"],
                    "peak": opening,
                    "corporate_multiplier": 1.0,
                }
            pending_entry = None
        close = float(bar["close"])
        equity = cash + shares * close
        equity_path.append({
            "date": bar["date"].strftime("%Y-%m-%d"),
            "equity": equity,
            "gross_equity": gross_cash + gross_shares * close,
            "cash": cash,
            "position_value": shares * close,
            "turnover": daily_turnover,
        })
        if position is not None:
            position["peak"] = max(position["peak"], close)
            gap = row.get("ma20_gap")
            atr_pct = row.get("atr_pct")
            if pd.notna(gap) and gap < 0:
                pending_exit = "收盘跌破20日均线"
            elif pd.notna(atr_pct) and position["peak"] - close >= 2 * close * atr_pct:
                pending_exit = "最高收盘价回撤2倍ATR"
        else:
            timestamp = pd.Timestamp(bar["date"]).replace(hour=15).isoformat()
            prediction = predictions.iloc[index]
            plan = plan_from_features(
                row, close, float(prediction) if pd.notna(prediction) else None,
                model_ready=pd.notna(prediction), as_of=timestamp,
                evaluation_time=timestamp, data_status="历史完整收盘（回测）",
            )
            pending_entry = plan if plan.get("signal_active") else None

    folds = []
    for fold in range(3):
        start = fold * test_size
        end = len(equity_path) if fold == 2 else (fold + 1) * test_size
        path = equity_path[start:end]
        fold_trades = [trade for trade in trades if path[0]["date"] <= trade["exit_date"] <= path[-1]["date"]]
        metrics = _metrics(
            fold_trades, len(path), path,
            equity_path[start - 1]["equity"] if start else 1.0,
            equity_path[start - 1]["gross_equity"] if start else 1.0,
        )
        folds.append({**metrics, "fold": fold + 1, "test_start": path[0]["date"], "test_end": path[-1]["date"]})
    metrics = _metrics(trades, len(equity_path), equity_path)
    benchmark_return = None
    if benchmark is not None and not benchmark.empty:
        reference = benchmark.copy()
        reference["date"] = pd.to_datetime(reference["date"])
        reference = reference.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
        first = bars.iloc[warmup - 1]["date"]
        last = bars.iloc[-1]["date"]
        if first in reference.index and last in reference.index:
            benchmark_return = float(reference.loc[last, "close"] / reference.loc[first, "close"] - 1)
    open_position = None
    if position is not None:
        open_position = {
            **position, "shares": shares, "market_value": shares * float(bars.iloc[-1]["close"]),
            "unrealized_return": float(bars.iloc[-1]["close"]) * position["corporate_multiplier"] / (position["entry_price"] * (1 + buy_cost)) - 1,
            "pending_exit": pending_exit,
        }
    status = "实验性个股模型与统一交易规则样本外回测" if strategy == "model" else "固定规则基线回测（不代表模型选股表现）"
    if strategy == "model" and predictions.iloc[warmup:].isna().all():
        # 没有合格预测时，空仓零收益不能被解释为模型已完成有效验证。
        status = "未生成通过验证的模型信号；本次仅为空仓记账，请查看训练记录"
    return {
        "status": status,
        "prediction_target": "超额收益模型的同日截面Z分数" if strategy == "model" else "固定趋势规则",
        "strategy": strategy,
        "split": {"warmup_bars": warmup, "fold_test_bars": test_size, "folds": 3, "method": "每20日仅使用此前历史训练验证；持仓跨分段连续计账"},
        "training_log": training_log,
        "costs": {"commission_rate_each_side": commission_rate, "stamp_duty_sell_side": stamp_duty_rate, "slippage_each_side": slippage_rate},
        "constraints": [
            "T+1；开盘涨停买单和跌停卖单不假设成交；停牌保留持仓并每日估值",
            "入场沿用页面的模型、均线、量能和买入区间；止盈止损同日触发时止损优先",
            "未平仓计入净值；公司行为按复权比例近似计权益，未精确模拟分红税和配股",
            "费率按比例近似，未模拟最低佣金、整手、盘口排队和历史股票池组合",
        ],
        "benchmark_buy_hold_return": benchmark_return,
        "metrics": metrics, "folds": folds, "trades": trades,
        "daily_equity": equity_path, "open_position": open_position,
        "unfilled_entries": unfilled_entries, "unfilled_exits": unfilled_exits,
        "model_version": settings.MODEL_VERSION,
    }


def save_backtest_run(parameters: dict[str, Any], result: dict[str, Any]) -> str:
    """将回测参数、逐日净值和模型版本保存到 PostgreSQL，返回运行编号。"""
    from services.database import QuantRepository

    run_id = str(uuid.uuid4())
    QuantRepository().save_backtest(run_id, parameters, result)
    return run_id
