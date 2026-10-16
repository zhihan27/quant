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

LOGGER = logging.getLogger("quant.portfolio")
PRICE_COLUMNS = (
    "date", "code", "open", "close", "previous_close", "limit_rate",
    "is_usable", "is_suspended", "is_locked_up", "is_locked_down",
)
CONSTRAINTS = (
    "T+1：当日开盘买入的股票当日起不可卖出，最早于下一个交易日开盘卖出",
    "开盘一字涨停不假设买入成交，开盘一字跌停不假设卖出成交，停牌当日不交易并按最近收盘价估值",
    "卖出收 0.05% 印花税，买卖双边各收佣金与滑点；费率按比例近似，未模拟最低佣金与整手",
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
    matrices = _price_matrices(prices, sorted(target.columns))
    dates = matrices["dates"]
    if not dates:
        return {"status": "目标权重对应的股票缺少行情，未产生组合交易", "metrics": None, "trades": [], "daily_equity": []}

    cash = gross_cash = float(initial_equity)
    holdings: dict[str, dict[str, Any]] = {}
    trades: list[dict[str, Any]] = []
    equity_path: list[dict[str, Any]] = []
    unfilled_buys = unfilled_sells = 0
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
                # 仓位上限以调仓前权益为基准，避免用当日尚未实现的收益放大仓位。
                reference = cash
                for code, item in holdings.items():
                    price = previous_close.get(code)
                    if price is not None and np.isfinite(price):
                        reference += item["shares"] * float(price)
                # 先卖出不在目标名单内的持仓；跌停或停牌则顺延到下一个可成交日。
                for code in list(holdings):
                    item = holdings[code]
                    if code in desired.index or item["entry_date"] >= date:
                        continue
                    if not bool(matrices["sellable"].loc[date].get(code, False)):
                        unfilled_sells += 1
                        continue
                    price = float(opens[code])
                    cash += item["shares"] * price * (1 - sell_cost)
                    gross_cash += item["gross_shares"] * price
                    daily_turnover += item["shares"] * price / prior_equity
                    trades.append({
                        "entry_date": item["entry_date"].strftime("%Y-%m-%d"),
                        "exit_date": date.strftime("%Y-%m-%d"),
                        "code": code,
                        "entry_price": item["entry_price"],
                        "exit_price": price,
                        "gross_return": float(price / item["entry_price"] - 1),
                        "net_return": float(price * (1 - sell_cost) / (item["entry_price"] * (1 + buy_cost)) - 1),
                        "holding_days": int((date - item["entry_date"]).days),
                        "exit_reason": "调出组合",
                    })
                    del holdings[code]
                # 再按打分顺序买入新入选股票；涨停或停牌则放弃本次买入并保留现金。
                for code, weight in desired.items():
                    if code in holdings:
                        continue
                    allocation = min(float(weight) * reference, cash)
                    if allocation <= 0:
                        continue
                    if not bool(matrices["buyable"].loc[date].get(code, False)):
                        unfilled_buys += 1
                        continue
                    price = float(opens[code])
                    if not np.isfinite(price) or price <= 0:
                        unfilled_buys += 1
                        continue
                    shares = allocation / (price * (1 + buy_cost))
                    gross_shares = allocation / price
                    cash = max(0.0, cash - shares * price * (1 + buy_cost))
                    gross_cash -= gross_shares * price
                    daily_turnover += shares * price / prior_equity
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

    metrics = _metrics(trades, len(equity_path), equity_path)
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
        "status": "多模型截面排序组合回测（含 T+1 与全部交易摩擦）",
        "metrics": metrics,
        "trades": trades,
        "daily_equity": equity_path,
        "open_positions": sorted(open_positions, key=lambda item: item["market_value"], reverse=True),
        "unfilled_buys": unfilled_buys,
        "unfilled_sells": unfilled_sells,
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
    """按区段滚动重训并对整个截面打分，融合多个模型的截面 Z 分数作为每日选股打分。"""
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
        block_frames = []
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
                block_frames.append(
                    predicted[["date", "code", "prediction_zscore"]].rename(columns={"prediction_zscore": name})
                )
        if block_frames:
            merged = block_frames[0]
            for frame in block_frames[1:]:
                merged = merged.merge(frame, on=["date", "code"], how="outer")
            available = [name for name in model_names if name in merged]
            # 各模型的截面Z分数等权平均；单一模型可用时即为该模型得分。
            merged["score"] = merged[available].mean(axis=1, skipna=True)
            frames.append(merged[columns])
        training_log.append({
            "prediction_start": str(cutoff.date()),
            "trained_through": str(panel.loc[panel["date"] < cutoff, "date"].max().date()),
            "models": log_models,
        })
    if not frames:
        return pd.DataFrame(columns=columns), training_log
    scores = pd.concat(frames, ignore_index=True).dropna(subset=["score"])
    return scores.sort_values(["date", "code"]).reset_index(drop=True), training_log


def trim_universe(histories: dict[str, pd.DataFrame], limit: int | None = None) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """超出上限时按成交额中位数保留流动性最好的股票，并明确报告被剔除的数量。"""
    limit = int(settings.PORTFOLIO_BACKTEST_MAX_STOCKS if limit is None else limit)
    report = {"limit": limit, "requested": len(histories), "kept": len(histories), "trimmed": 0}
    if limit <= 0 or len(histories) <= limit:
        return histories, report
    liquidity = {}
    for code, history in histories.items():
        amount = pd.to_numeric(history["amount"], errors="coerce") if "amount" in history else None
        if amount is None or amount.notna().sum() == 0:
            close = pd.to_numeric(history["close"], errors="coerce")
            volume = pd.to_numeric(history["volume"], errors="coerce")
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
) -> dict[str, Any]:
    """滚动重训打分、构建前 N 名组合并做含摩擦回测的完整流程。"""
    if panel is None or panel.empty:
        return {"status": "缺少历史截面，无法运行组合回测", "metrics": None}
    histories, universe_report = trim_universe(histories, universe_limit)
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
    weights, weight_report = portfolio_weights(scores, top_n=top_n, max_weight=max_weight)
    prices = prepare_price_panel(histories, is_st_codes=is_st_codes, adjusted=adjusted)
    result = run_portfolio_backtest(prices, weights, rebalance_every=rebalance_every)
    if result.get("metrics") is None:
        return {**result, "training_log": training_log, "universe": universe_report, "portfolio": weight_report}
    benchmark_return = None
    if benchmark is not None and not benchmark.empty:
        reference = benchmark.copy()
        reference["date"] = pd.to_datetime(reference["date"])
        reference = reference.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
        beginning, ending = pd.Timestamp(scores["date"].min()), pd.Timestamp(scores["date"].max())
        if beginning in reference.index and ending in reference.index:
            benchmark_return = float(reference.loc[ending, "close"] / reference.loc[beginning, "close"] - 1)
    result.update(
        training_log=training_log,
        universe=universe_report,
        portfolio=weight_report,
        score_range={"start": str(pd.Timestamp(scores["date"].min()).date()), "end": str(pd.Timestamp(scores["date"].max()).date()),
                     "dates": int(scores["date"].nunique()), "scored_rows": int(len(scores))},
        rebalance_every=rebalance_every,
        benchmark_buy_hold_return=benchmark_return,
        horizon=horizon,
    )
    return result
