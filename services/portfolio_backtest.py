"""多因子组合构建与含交易摩擦的组合回测。

选股口径是每日截面排序：融合多个模型在同一交易日内的相对得分，取前 N 名等权持有，
再对单股仓位施加上限。撮合严格模拟 A 股 T+1、开盘封板不可成交、停牌不交易，
并逐笔计入佣金、印花税与滑点。
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd

from config import settings
from services.backtest import _metrics, cost_rates
from services.data_cleaning import clean_daily_history
from services.ml_pipeline import MODEL_NAMES, fit_panel_model, predict_panel
from services.strategy_scoring import fuse_prediction_percentiles

LOGGER = logging.getLogger("quant.portfolio")
PRICE_COLUMNS = (
    "date", "code", "open", "close", "previous_close", "limit_rate",
    "is_usable", "is_suspended", "is_locked_up", "is_locked_down",
)
CONSTRAINTS = (
    "T+1：当日开盘买入的股票当日起不可卖出，最早于下一个交易日开盘卖出",
    "开盘一字涨停不假设买入成交，开盘一字跌停不假设卖出成交，停牌当日不交易并按最近收盘价估值",
    "卖出收印花税，买卖双边各收佣金与滑点；费率按比例近似，未模拟最低佣金",
    "默认标准化权益允许小数股；传入逐股 lot_sizes 后才执行买入整手及卖出整手约束",
    "单股仓位上限按调仓前权益计算，超出上限的资金保留为现金，不放大杠杆",
    "组合净值仅代表本次采集股票池内按模型排序的结果，未考虑冲击成本与申购赎回",
)


def _round_half_up(values: Any) -> np.ndarray:
    """复现 Decimal ROUND_HALF_UP 的正数取整，用于向量化计算涨跌停价。"""
    return np.floor(np.asarray(values, dtype=float) * 100 + 0.5) / 100


def prepare_price_panel(
    histories: dict[str, pd.DataFrame], is_st_codes: dict[str, bool] | None = None, adjusted: bool = True
) -> pd.DataFrame:
    """把逐股日线整理为组合回测长表；缺少清洗标记的行情先执行一次清洗流水线。"""
    is_st_codes = is_st_codes or {}
    frames = []
    for code, history in histories.items():
        bars = history
        if "is_usable" not in bars or "previous_close" not in bars:
            bars, _ = clean_daily_history(bars, code=str(code), is_st=bool(is_st_codes.get(str(code))), adjusted=adjusted)
        bars = bars.copy()
        bars["code"] = str(code)
        frames.append(bars[list(PRICE_COLUMNS)])
    if not frames:
        return pd.DataFrame(columns=list(PRICE_COLUMNS))
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    return panel.sort_values(["date", "code"]).reset_index(drop=True)


def _fillable(opens: pd.DataFrame, previous_close: pd.DataFrame, limit_rate: pd.DataFrame, usable: pd.DataFrame, direction: str) -> pd.DataFrame:
    """向量化复现开盘可成交判断：开盘即封板、停牌或价格缺失时不假设成交。"""
    factor = 1 + limit_rate if direction == "buy" else 1 - limit_rate
    limit = pd.DataFrame(_round_half_up(previous_close * factor), index=opens.index, columns=opens.columns)
    allowed = opens.lt(limit - 0.0049) if direction == "buy" else opens.gt(limit + 0.0049)
    return allowed.fillna(False) & usable.fillna(False) & opens.gt(0).fillna(False)


def _price_matrices(prices: pd.DataFrame, codes: Sequence[str]) -> dict[str, Any]:
    """把长表转成以交易日为行、股票为列的矩阵，停牌日估值沿用最近收盘价。"""
    subset = prices.loc[prices["code"].isin(codes)]
    dates = sorted(subset["date"].unique())
    columns = sorted(set(codes))

    def matrix(column: str) -> pd.DataFrame:
        # 统一重排到相同的行列坐标，停牌或缺失的格子保留为缺失值。
        return subset.pivot_table(index="date", columns="code", values=column, aggfunc="last").reindex(index=dates, columns=columns)

    opens, closes = matrix("open"), matrix("close")
    return {
        "dates": dates,
        "open": opens,
        "close": closes,
        "valuation": closes.ffill(),
        "buyable": _fillable(opens, matrix("previous_close"), matrix("limit_rate"), matrix("is_usable"), "buy"),
        "sellable": _fillable(opens, matrix("previous_close"), matrix("limit_rate"), matrix("is_usable"), "sell"),
    }


def portfolio_weights(
    scores: pd.DataFrame, top_n: int | None = None, max_weight: float | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """每日截面按综合打分取前 N 名等权持有，单股权重不超过上限，超出部分保留现金。"""
    top_n = int(settings.PORTFOLIO_TOP_N if top_n is None else top_n)
    max_weight = float(settings.MAX_POSITION_WEIGHT if max_weight is None else max_weight)
    if top_n < 1:
        raise ValueError("组合持股数量必须为正整数。")
    if not 0 < max_weight <= 1:
        raise ValueError("单股仓位上限必须在 0 与 1 之间。")
    required = {"date", "code", "score"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"打分表缺少字段：{', '.join(sorted(missing))}")
    weight = min(1.0 / top_n, max_weight)
    clean = scores.dropna(subset=["score"]).copy()
    clean = clean.sort_values(["date", "score"], ascending=[True, False], kind="stable")
    clean["rank"] = clean.groupby("date").cumcount() + 1
    selected = clean.loc[clean["rank"] <= top_n].copy()
    selected["target_weight"] = weight
    per_date = selected.groupby("date")["code"].size()
    invested = float(min(1.0, weight * top_n))
    report = {
        "top_n": top_n,
        "max_weight": max_weight,
        "per_stock_weight": weight,
        "cap_binding": bool(1.0 / top_n > max_weight),
        "invested_ratio": invested,
        "cash_ratio": float(1 - invested),
        "dates": int(per_date.size),
        "average_holdings": float(per_date.mean()) if per_date.size else 0.0,
        "status": (
            f"每日取截面综合打分前 {top_n} 名等权持有，单股 {weight:.2%}，上限 {max_weight:.2%}"
            if not 1.0 / top_n > max_weight
            else f"每日取截面综合打分前 {top_n} 名，单股上限 {max_weight:.2%} 生效，其余资金保留为现金"
        ),
    }
    columns = ["date", "code", "score", "rank", "target_weight"]
    return selected[columns].reset_index(drop=True), report


def run_portfolio_backtest(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    commission_rate: float = settings.DEFAULT_COSTS["commission_rate"],
    stamp_duty_rate: float = settings.DEFAULT_COSTS["stamp_duty_rate"],
    slippage_rate: float = settings.DEFAULT_COSTS["slippage_rate"],
    rebalance_every: int = 1,
    initial_equity: float = 1.0,
    lot_sizes: dict[str, int] | None = None,
) -> dict[str, Any]:
    """在信号日的下一个交易日开盘调仓，逐日估值并计入全部交易摩擦。"""
    if rebalance_every < 1:
        raise ValueError("调仓间隔必须为不小于 1 的交易日数。")
    rates = [commission_rate, stamp_duty_rate, slippage_rate]
    if not all(np.isfinite(rate) and 0 <= rate < 1 for rate in rates) or sum(rates) >= 1:
        raise ValueError("交易费用必须是有效的非负比例，费用合计须小于 1。")
    if not np.isfinite(initial_equity) or initial_equity <= 0:
        raise ValueError("初始权益必须为正数。")
    required = {"date", "code", "target_weight"}
    missing = required - set(weights.columns)
    if missing:
        raise ValueError(f"目标权重表缺少字段：{', '.join(sorted(missing))}")
    if weights.empty:
        return {"status": "没有可用目标权重，未产生组合交易", "metrics": None, "trades": [], "daily_equity": []}
    target = weights.pivot_table(index="date", columns="code", values="target_weight", aggfunc="last")
    lot_size_by_code: dict[str, int] | None = None
    if lot_sizes is not None:
        supplied_lot_sizes = {str(code): value for code, value in lot_sizes.items()}
        target_codes = {str(code) for code in target.columns}
        missing_lot_sizes = sorted(target_codes - set(supplied_lot_sizes))
        if missing_lot_sizes:
            raise ValueError(f"整手回测缺少股票交易单位：{', '.join(missing_lot_sizes)}")
        lot_size_by_code = {}
        for code in target_codes:
            try:
                raw_size = float(supplied_lot_sizes[code])
            except (TypeError, ValueError):
                raise ValueError(f"股票 {code} 的交易单位必须是正整数。") from None
            if not np.isfinite(raw_size) or raw_size < 1 or not raw_size.is_integer():
                raise ValueError(f"股票 {code} 的交易单位必须是正整数。")
            lot_size_by_code[code] = int(raw_size)
    matrices = _price_matrices(prices, sorted(target.columns))
    dates = matrices["dates"]
    if not dates:
        return {"status": "目标权重对应的股票缺少行情，未产生组合交易", "metrics": None, "trades": [], "daily_equity": []}

    cash = gross_cash = float(initial_equity)
    holdings: dict[str, dict[str, Any]] = {}
    trades: list[dict[str, Any]] = []
    equity_path: list[dict[str, Any]] = []
    unfilled_buys = unfilled_sells = 0
    partial_buys = 0
    buy_cost, sell_cost = cost_rates(commission_rate, stamp_duty_rate, slippage_rate)

    for index, date in enumerate(dates):
        daily_turnover = 0.0
        prior_equity = equity_path[-1]["equity"] if equity_path else float(initial_equity)
        if index and index % rebalance_every == 0:
            signal_date = dates[index - 1]
            if signal_date in target.index:
                desired = target.loc[signal_date].dropna()
                desired = desired[desired > 0].sort_values(ascending=False)
                opens = matrices["open"].loc[date]
                previous_close = matrices["valuation"].loc[dates[index - 1]]
                current_values: dict[str, float] = {}
                execution_equity = cash
                for code, item in holdings.items():
                    price = opens.get(code)
                    if price is None or not np.isfinite(price) or price <= 0:
                        price = previous_close.get(code)
                    if price is not None and np.isfinite(price) and price > 0:
                        current_values[code] = item["shares"] * float(price)
                        execution_equity += current_values[code]
                target_budget = execution_equity
                # 反推扣除本次买卖费用后的权益，令可成交持仓按该权益满足目标权重。
                for _ in range(20):
                    sell_notional = 0.0
                    for code, item in holdings.items():
                        price = opens.get(code)
                        can_sell = (
                            item["entry_date"] < date
                            and price is not None
                            and np.isfinite(price)
                            and price > 0
                            and bool(matrices["sellable"].loc[date].get(code, False))
                        )
                        if can_sell:
                            target_value = float(desired.get(code, 0.0)) * target_budget
                            sell_notional += max(0.0, current_values.get(code, 0.0) - target_value)
                    buy_notional = 0.0
                    for code, weight in desired.items():
                        price = opens.get(code)
                        can_buy = (
                            price is not None
                            and np.isfinite(price)
                            and price > 0
                            and bool(matrices["buyable"].loc[date].get(code, False))
                        )
                        if can_buy:
                            target_value = float(weight) * target_budget
                            buy_notional += max(0.0, target_value - current_values.get(code, 0.0))
                    adjusted_budget = max(0.0, execution_equity - sell_cost * sell_notional - buy_cost * buy_notional)
                    if abs(adjusted_budget - target_budget) <= 1e-12:
                        target_budget = adjusted_budget
                        break
                    target_budget = adjusted_budget
                # 先卖出超出目标股数的部分；跌停、停牌或 T+1 限制会阻止本次减仓。
                for code in list(holdings):
                    item = holdings[code]
                    if item["entry_date"] >= date:
                        continue
                    open_price = opens.get(code)
                    if open_price is None or not np.isfinite(open_price) or open_price <= 0:
                        if code not in desired.index:
                            unfilled_sells += 1
                        continue
                    target_weight = float(desired.get(code, 0.0))
                    target_shares = target_weight * target_budget / float(open_price)
                    current_shares = float(item["shares"])
                    shares_to_sell = min(current_shares, max(0.0, current_shares - target_shares))
                    lot_size = lot_size_by_code.get(str(code)) if lot_size_by_code is not None else None
                    if lot_size is not None:
                        if current_shares - shares_to_sell <= 1e-9:
                            shares_to_sell = current_shares
                        else:
                            # 部分卖出向上取整到整手，优先守住目标仓位上限；全量退出允许卖完剩余零股。
                            lots_to_sell = int(np.ceil(shares_to_sell / lot_size - 1e-12))
                            shares_to_sell = float(min(current_shares, lots_to_sell * lot_size))
                    if shares_to_sell <= 1e-12:
                        continue
                    if not bool(matrices["sellable"].loc[date].get(code, False)):
                        unfilled_sells += 1
                        continue
                    price = float(open_price)
                    gross_shares_sold = item["gross_shares"] * shares_to_sell / current_shares
                    cash += shares_to_sell * price * (1 - sell_cost)
                    gross_cash += gross_shares_sold * price
                    daily_turnover += shares_to_sell * price / prior_equity
                    is_partial_exit = shares_to_sell < current_shares - 1e-12
                    trades.append({
                        "entry_date": item["entry_date"].strftime("%Y-%m-%d"),
                        "exit_date": date.strftime("%Y-%m-%d"),
                        "code": code,
                        "entry_price": item["entry_price"],
                        "exit_price": price,
                        "gross_return": float(price / item["entry_price"] - 1),
                        "net_return": float(price * (1 - sell_cost) / (item["entry_price"] * (1 + buy_cost)) - 1),
                        "holding_days": int((date - item["entry_date"]).days),
                        "shares": shares_to_sell,
                        "partial_exit": is_partial_exit,
                        "exit_reason": "目标权重再平衡" if code in desired.index else "调出组合",
                    })
                    item["shares"] -= shares_to_sell
                    item["gross_shares"] -= gross_shares_sold
                    if item["shares"] <= 1e-12:
                        del holdings[code]
                # 再按打分顺序补足目标股数；涨停或停牌时保留现金。
                for code, weight in desired.items():
                    open_price = opens.get(code)
                    if open_price is None or not np.isfinite(open_price) or open_price <= 0:
                        unfilled_buys += 1
                        continue
                    price = float(open_price)
                    current = holdings.get(code)
                    current_shares = float(current["shares"]) if current else 0.0
                    target_shares = float(weight) * target_budget / price
                    shares_needed = max(0.0, target_shares - current_shares)
                    if shares_needed <= 1e-12:
                        continue
                    if not bool(matrices["buyable"].loc[date].get(code, False)):
                        unfilled_buys += 1
                        continue
                    lot_size = lot_size_by_code.get(str(code)) if lot_size_by_code is not None else None
                    if lot_size is not None:
                        # 买入只取完整交易单位，资金不足时保留现金，不以零股补单。
                        target_lots = int(np.floor(shares_needed / lot_size + 1e-12))
                        affordable_shares = cash / (price * (1 + buy_cost))
                        affordable_lots = int(np.floor(affordable_shares / lot_size))
                        shares = float(max(0, min(target_lots, affordable_lots)) * lot_size)
                        allocation = shares * price * (1 + buy_cost)
                    else:
                        allocation = min(shares_needed * price * (1 + buy_cost), cash)
                        shares = allocation / (price * (1 + buy_cost)) if allocation > 0 else 0.0
                    if allocation <= 0 or shares <= 1e-12:
                        unfilled_buys += 1
                        continue
                    if shares < shares_needed - 1e-12:
                        partial_buys += 1
                    gross_shares = allocation / price
                    cash = max(0.0, cash - allocation)
                    gross_cash -= gross_shares * price
                    daily_turnover += shares * price / prior_equity
                    if current:
                        current["entry_price"] = (
                            current_shares * current["entry_price"] + shares * price
                        ) / (current_shares + shares)
                        current["shares"] += shares
                        current["gross_shares"] += gross_shares
                    else:
                        holdings[code] = {
                            "shares": shares, "gross_shares": gross_shares,
                            "entry_date": date, "entry_price": price,
                        }
        close = matrices["valuation"].loc[date]
        position_value = gross_value = 0.0
        for code, item in holdings.items():
            price = close.get(code)
            if price is None or not np.isfinite(price):
                continue
            position_value += item["shares"] * float(price)
            gross_value += item["gross_shares"] * float(price)
        equity_path.append({
            "date": date.strftime("%Y-%m-%d"),
            "equity": cash + position_value,
            "gross_equity": gross_cash + gross_value,
            "cash": cash,
            "position_value": position_value,
            "holdings": len(holdings),
            "turnover": daily_turnover,
        })

    # equity_path 的首日是信号收盘后的初始现金状态；有效收益区间从其后交易日开始。
    metrics = _metrics(trades, max(len(equity_path) - 1, 1), equity_path)
    beginning, ending = matrices["valuation"].iloc[0], matrices["valuation"].iloc[-1]
    comparable = beginning.gt(0).fillna(False) & ending.notna()
    universe_return = float((ending[comparable] / beginning[comparable] - 1).mean()) if comparable.any() else None
    open_positions = []
    for code, item in holdings.items():
        price = matrices["valuation"].iloc[-1].get(code)
        if price is None or not np.isfinite(price):
            continue
        open_positions.append({
            "code": code,
            "entry_date": item["entry_date"].strftime("%Y-%m-%d"),
            "entry_price": item["entry_price"],
            "shares": item["shares"],
            "market_value": item["shares"] * float(price),
            "unrealized_return": float(price * (1 - sell_cost) / (item["entry_price"] * (1 + buy_cost)) - 1),
        })
    return {
        "status": "多模型截面排序组合研究回测（含 T+1 与比例交易摩擦）",
        "metrics": metrics,
        "trades": trades,
        "daily_equity": equity_path,
        "open_positions": sorted(open_positions, key=lambda item: item["market_value"], reverse=True),
        "unfilled_buys": unfilled_buys,
        "unfilled_sells": unfilled_sells,
        "partial_buys": partial_buys,
        "lot_constraints_applied": lot_size_by_code is not None,
        "share_lot_sizes": lot_size_by_code,
        "fractional_shares_allowed": lot_size_by_code is None,
        "initial_equity": float(initial_equity),
        "capital_unit_note": "资金与行情价格使用同一货币单位；默认权益 1.0 仅适合标准化研究，整手回测应传入实际资金量。",
        "universe_equal_weight_return": universe_return,
        "costs": {
            "commission_rate_each_side": commission_rate,
            "stamp_duty_sell_side": stamp_duty_rate,
            "slippage_each_side": slippage_rate,
        },
        "constraints": list(CONSTRAINTS),
        "model_version": settings.MODEL_VERSION,
    }


def walk_forward_scores(
    panel: pd.DataFrame,
    start: int,
    horizon: int = 20,
    retrain_every: int = 60,
    objective: str | None = None,
    window: int | None = None,
    model_names: Sequence[str] = MODEL_NAMES,
    progress: Any = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """按区段滚动重训并融合模型截面百分位，统一每日选股打分。"""
    dates = np.sort(panel["date"].unique())
    columns = ["date", "code", "score", *model_names]
    if start >= len(dates) or start < 0:
        return pd.DataFrame(columns=columns), [{"status": "起始位置超过可用交易日，未产生组合打分"}]
    frames = []
    training_log = []
    for block_start in range(start, len(dates), retrain_every):
        cutoff = pd.Timestamp(dates[block_start])
        block_end = min(block_start + retrain_every, len(dates))
        live = panel.loc[panel["date"].between(cutoff, pd.Timestamp(dates[block_end - 1]))]
        block_frames: dict[str, pd.DataFrame] = {}
        log_models = []
        for name in model_names:
            if progress:
                progress(f"滚动重训 {name} · {horizon}日截面排序模型（截至 {cutoff.date()}）")
            model = fit_panel_model(panel, name, horizon, cutoff=cutoff, objective=objective, window=window)
            predicted = predict_panel(model, live)
            # 记录实际检验数值：只写“未通过验证”无法判断是哪条门槛没过，也无法复核。
            metrics = model.get("oos_metrics") or {}
            log_models.append({
                "model": name, "objective": model.get("objective"), "score_kind": model.get("score_kind"),
                "status": model["status"], "training_dates": model.get("training_dates"),
                "rank_ic": metrics.get("rank_ic"), "ndcg_ratio": metrics.get("ndcg_ratio"),
                "mae": metrics.get("mae"), "validation_rule": metrics.get("validation_rule"),
            })
            if not predicted.empty:
                block_frames[name] = predicted[["date", "code", "prediction_percentile"]]
        if block_frames:
            frames.append(fuse_prediction_percentiles(block_frames, model_names).reindex(columns=columns))
        training_log.append({
            "prediction_start": str(cutoff.date()),
            "trained_through": str(panel.loc[panel["date"] < cutoff, "date"].max().date()),
            "models": log_models,
        })
    if not frames:
        return pd.DataFrame(columns=columns), training_log
    scores = pd.concat(frames, ignore_index=True).dropna(subset=["score"])
    return scores.sort_values(["date", "code"]).reset_index(drop=True), training_log


def trim_universe(
    histories: dict[str, pd.DataFrame], limit: int | None = None, as_of_date: Any = None
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """超出上限时按成交额中位数保留流动性最好的股票，并明确报告被剔除的数量。"""
    limit = int(settings.PORTFOLIO_BACKTEST_MAX_STOCKS if limit is None else limit)
    cutoff = pd.Timestamp(as_of_date) if as_of_date is not None else None
    report = {
        "limit": limit, "requested": len(histories), "kept": len(histories), "trimmed": 0,
        "selection_as_of": str(cutoff.date()) if cutoff is not None else None,
        "selection_policy": "按该日期可见历史流动性冻结股票池" if cutoff is not None else "按传入历史范围筛选股票池",
    }
    if limit <= 0 or len(histories) <= limit:
        return histories, report
    liquidity = {}
    for code, history in histories.items():
        dated_history = history
        if cutoff is not None:
            if "date" not in history:
                dated_history = history.iloc[0:0]
            else:
                dates = pd.to_datetime(history["date"], errors="coerce")
                dated_history = history.loc[dates.le(cutoff)]
        amount = pd.to_numeric(dated_history["amount"], errors="coerce") if "amount" in dated_history else None
        if amount is None or amount.notna().sum() == 0:
            close = pd.to_numeric(dated_history["close"], errors="coerce") if "close" in dated_history else pd.Series(dtype=float)
            volume = pd.to_numeric(dated_history["volume"], errors="coerce") if "volume" in dated_history else pd.Series(dtype=float)
            amount = close * volume
        median = amount.tail(60).median()
        liquidity[code] = float(median) if median is not None and np.isfinite(median) else 0.0
    ordered = sorted(histories, key=lambda code: liquidity.get(code, 0.0), reverse=True)[:limit]
    report.update(kept=len(ordered), trimmed=len(histories) - len(ordered),
                  status=f"股票池超过 {limit} 只，按近 60 日成交额中位数保留前 {limit} 只")
    return {code: histories[code] for code in ordered}, report


def run_portfolio_strategy(
    histories: dict[str, pd.DataFrame],
    panel: pd.DataFrame,
    benchmark: pd.DataFrame | None = None,
    start: int = 320,
    horizon: int = 20,
    retrain_every: int = 60,
    top_n: int | None = None,
    max_weight: float | None = None,
    rebalance_every: int = 1,
    objective: str | None = None,
    window: int | None = None,
    universe_limit: int | None = None,
    adjusted: bool = True,
    is_st_codes: dict[str, bool] | None = None,
    progress: Any = None,
    initial_equity: float = 1.0,
    lot_sizes: dict[str, int] | None = None,
) -> dict[str, Any]:
    """滚动重训打分、构建前 N 名组合并做含摩擦回测的完整流程。"""
    if panel is None or panel.empty:
        return {"status": "缺少历史截面，无法运行组合回测", "metrics": None}
    if lot_sizes is not None and adjusted:
        raise ValueError("启用整手约束时必须使用未复权成交价格，请将 adjusted 设为 False。")
    panel_dates = pd.DatetimeIndex(pd.to_datetime(panel["date"], errors="coerce").dropna().unique()).sort_values()
    universe_as_of = panel_dates[start] if 0 <= start < len(panel_dates) else None
    histories, universe_report = trim_universe(histories, universe_limit, as_of_date=universe_as_of)
    universe_report["historical_membership_verified"] = False
    universe_report["warning"] = (
        "回测仅覆盖输入历史行情与训练截面的代码集合；未接入历史成分变更和退市数据，"
        "结果可能存在幸存者偏差。"
    )
    if universe_report["trimmed"]:
        # 打分必须与回测使用同一股票池，否则排序会选中没有行情、根本无法成交的股票。
        kept = set(histories)
        attributes = dict(panel.attrs)
        panel = panel.loc[panel["code"].isin(kept)].reset_index(drop=True)
        panel.attrs.update(attributes)
        universe_report["status"] = f"{universe_report['status']}，打分与回测均限定在该股票池内"
    if panel.empty:
        return {"status": "流动性筛选后股票池为空，未产生组合", "metrics": None, "universe": universe_report}
    scores, training_log = walk_forward_scores(
        panel, start, horizon=horizon, retrain_every=retrain_every,
        objective=objective, window=window, progress=progress,
    )
    if scores.empty:
        return {"status": "滚动重训未生成任何通过验证的截面打分，未产生组合", "metrics": None,
                "training_log": training_log, "universe": universe_report}
    prices = prepare_price_panel(histories, is_st_codes=is_st_codes, adjusted=adjusted)
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    price_dates = pd.DatetimeIndex(sorted(prices["date"].dropna().unique()))
    executable_dates = set(price_dates[:-1])
    scores = scores.loc[pd.to_datetime(scores["date"]).isin(executable_dates)].copy()
    if scores.empty:
        return {"status": "没有同时具备下一交易日行情的信号，未产生组合", "metrics": None,
                "training_log": training_log, "universe": universe_report}
    score_start = pd.Timestamp(scores["date"].min())
    score_end = pd.Timestamp(scores["date"].max())
    execution_end = price_dates[price_dates > score_end][0]
    prices = prices.loc[prices["date"].between(score_start, execution_end)].copy()
    weights, weight_report = portfolio_weights(scores, top_n=top_n, max_weight=max_weight)
    result = run_portfolio_backtest(
        prices, weights, rebalance_every=rebalance_every, initial_equity=initial_equity,
        lot_sizes=lot_sizes,
    )
    if result.get("metrics") is None:
        return {**result, "training_log": training_log, "universe": universe_report, "portfolio": weight_report}
    benchmark_return = None
    benchmark_start = None
    benchmark_range = None
    first_execution = price_dates[price_dates > score_start][0]
    ending = pd.Timestamp(result["daily_equity"][-1]["date"])
    if benchmark is not None and not benchmark.empty:
        reference = benchmark.copy()
        reference["date"] = pd.to_datetime(reference["date"])
        reference = reference.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
        if "open" in reference and first_execution in reference.index and ending in reference.index:
            benchmark_return = float(reference.loc[ending, "close"] / reference.loc[first_execution, "open"] - 1)
            benchmark_start = str(first_execution.date())
            benchmark_range = {
                "start": benchmark_start, "start_price": "open",
                "end": str(ending.date()), "end_price": "close",
            }
    result.update(
        training_log=training_log,
        universe=universe_report,
        portfolio=weight_report,
        score_range={"start": str(score_start.date()), "end": str(score_end.date()),
                     "dates": int(scores["date"].nunique()), "scored_rows": int(len(scores))},
        performance_range={
            "signal_date": str(score_start.date()),
            "start": str(first_execution.date()), "start_price": "open",
            "start_state": "信号日收盘后持有现金，首笔订单在下一交易日开盘执行",
            "end": str(ending.date()), "end_price": "close",
            "dates": max(len(result["daily_equity"]) - 1, 0),
            "equity_points": len(result["daily_equity"]),
            "annualization_intervals": max(len(result["daily_equity"]) - 1, 1),
        },
        rebalance_every=rebalance_every,
        benchmark_buy_hold_return=benchmark_return,
        benchmark_start=benchmark_start,
        benchmark_range=benchmark_range,
        horizon=horizon,
        research_only=True,
        validation_eligible=False,
        validation_limitation="历史成分与退市股票数据尚未验证；该结果不得作为历史全市场策略验收。",
        strategy_rules_aligned=False,
        strategy_rules_limitation="本回测共用模型百分位排序，但未复现人工任务单的入场过滤、趋势退出与固定止损规则。",
        share_lot_constraints_applied=lot_sizes is not None,
        fractional_shares_allowed=lot_sizes is None,
        initial_equity=float(initial_equity),
        capital_unit_note="资金与行情价格使用同一货币单位；默认权益 1.0 仅适合标准化研究，整手回测应传入实际资金量。",
    )
    return result
