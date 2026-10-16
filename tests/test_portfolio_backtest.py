"""验证组合权重上限、T+1、一字板不可成交与交易摩擦计入。"""

import unittest

import numpy as np
import pandas as pd

from config import settings
from services.data_cleaning import open_price_fillable
from services.feature_engineering import FEATURE_COLUMNS
from services.ml_pipeline import build_market_panel
from services.portfolio_backtest import (
    _price_matrices, portfolio_weights, prepare_price_panel, run_portfolio_backtest,
    run_portfolio_strategy, trim_universe,
)
from services.rank_model import SCORE_KIND_RANK

CODES = ("600000", "600001", "600002")
LIMIT_RATE = 0.10


def _bars(days=12, codes=CODES, start="2024-01-01", suspended_at=None, locked_up_at=None):
    """构造等速上涨的确定性行情，可指定停牌与开盘一字涨停的股票与位置。"""
    dates = pd.bdate_range(start, periods=days)
    rows = []
    for code in codes:
        for position, date in enumerate(dates):
            price = 10.0 + position * 0.1
            previous = 10.0 + (position - 1) * 0.1 if position else np.nan
            opening = price
            suspended = suspended_at is not None and suspended_at.get(code) == position
            locked = locked_up_at is not None and locked_up_at.get(code) == position and position
            if locked:
                previous_close = float(previous)
                opening = round(previous_close * (1 + LIMIT_RATE), 2)
                price = opening
            rows.append({
                "date": date, "code": code,
                "open": np.nan if suspended else opening,
                "high": np.nan if suspended else price * 1.02,
                "low": np.nan if suspended else price * 0.98,
                "close": np.nan if suspended else price,
                "volume": 0.0 if suspended else 1_000_000.0,
                "previous_close": np.nan if suspended else previous,
                "limit_rate": LIMIT_RATE,
                "is_usable": not suspended,
                "is_suspended": suspended,
                "is_locked_up": bool(locked),
                "is_locked_down": False,
            })
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _weights(mapping):
    rows = [{"date": date, "code": code, "score": 1.0, "rank": index + 1, "target_weight": weight}
            for date, codes in mapping.items() for index, (code, weight) in enumerate(codes.items())]
    return pd.DataFrame(rows)


class FillableMatrixTests(unittest.TestCase):
    """矩阵化的成交判断必须与逐行规则完全一致。"""

    def test_matrix_matches_single_bar_rule(self):
        bars = _bars(days=3, codes=("600000",))
        bars.loc[bars.index[1], "open"] = 11.0  # 昨收10.0，开盘价恰为涨停价
        bars.loc[bars.index[2], "open"] = 10.99
        matrices = _price_matrices(bars, ("600000",))
        for direction, column in (("buy", "buyable"), ("sell", "sellable")):
            for _, bar in bars.iterrows():
                expected = open_price_fillable(bar, direction, bar["previous_close"], LIMIT_RATE)
                actual = bool(matrices[column].loc[bar["date"], bar["code"]])
                self.assertEqual(actual, expected, f"{direction} {bar['date']}")

    def test_missing_price_or_volume_blocks_trade(self):
        bars = _bars(days=3, codes=("600000",), suspended_at={"600000": 1})
        buyable = _price_matrices(bars, ("600000",))["buyable"]
        # 停牌当日不可买入，复牌后恢复；首个交易日缺少昨收，保守视为不可成交。
        self.assertFalse(bool(buyable.loc[bars["date"].iloc[1], "600000"]))
        self.assertTrue(bool(buyable.loc[bars["date"].iloc[2], "600000"]))
        self.assertFalse(bool(buyable.loc[bars["date"].iloc[0], "600000"]))


class PortfolioWeightTests(unittest.TestCase):
    """每日取前N名，单股仓位不得超过上限。"""

    def scores(self):
        dates = pd.bdate_range("2024-01-01", periods=2)
        rows = [{"date": date, "code": f"60000{index}", "score": float(10 - index)} for date in dates for index in range(6)]
        return pd.DataFrame(rows)

    def test_top_n_selection_and_equal_weight(self):
        weights, report = portfolio_weights(self.scores(), top_n=3, max_weight=0.5)
        self.assertEqual(len(weights), 6)
        self.assertTrue((weights["rank"] <= 3).all())
        self.assertTrue(np.allclose(weights["target_weight"], 1 / 3))
        self.assertFalse(report["cap_binding"])
        self.assertAlmostEqual(report["invested_ratio"], 1.0)
        best = weights.loc[weights["date"].eq(weights["date"].min())].sort_values("rank")
        self.assertEqual(best.iloc[0]["score"], 10.0)

    def test_position_cap_keeps_remainder_in_cash(self):
        weights, report = portfolio_weights(self.scores(), top_n=2, max_weight=0.05)
        self.assertTrue(report["cap_binding"])
        self.assertTrue(np.allclose(weights["target_weight"], 0.05))
        self.assertAlmostEqual(report["invested_ratio"], 0.10)
        self.assertAlmostEqual(report["cash_ratio"], 0.90)

    def test_invalid_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            portfolio_weights(self.scores(), top_n=0)
        with self.assertRaises(ValueError):
            portfolio_weights(self.scores(), max_weight=1.5)
        with self.assertRaises(ValueError):
            portfolio_weights(pd.DataFrame({"date": [], "code": []}))


class PortfolioExecutionTests(unittest.TestCase):
    """信号在次日开盘执行，摩擦成本进入净值。"""

    def test_signal_executes_on_next_open(self):
        bars = _bars(days=8)
        dates = sorted(bars["date"].unique())
        weights = _weights({dates[0]: {"600000": 0.5}})
        result = run_portfolio_backtest(bars, weights)
        self.assertIsNotNone(result["metrics"])
        self.assertEqual(len(result["open_positions"]), 1)
        self.assertEqual(result["open_positions"][0]["entry_date"], str(pd.Timestamp(dates[1]).date()))
        self.assertAlmostEqual(result["open_positions"][0]["entry_price"], float(bars.loc[bars["date"].eq(dates[1]) & bars["code"].eq("600000"), "open"].iloc[0]))
        # 50% 仓位下单股，其余保留现金。
        first_day = result["daily_equity"][1]
        self.assertGreater(first_day["cash"], 0)
        self.assertGreater(first_day["position_value"], 0)

    def test_cap_limits_single_position(self):
        bars = _bars(days=8)
        dates = sorted(bars["date"].unique())
        weights = _weights({dates[0]: {"600000": settings.MAX_POSITION_WEIGHT}})
        result = run_portfolio_backtest(bars, weights)
        entry = result["daily_equity"][1]
        self.assertLessEqual(entry["position_value"] / entry["equity"], settings.MAX_POSITION_WEIGHT + 1e-6)

    def test_locked_up_open_cannot_be_bought(self):
        bars = _bars(days=8, locked_up_at={"600000": 1})
        dates = sorted(bars["date"].unique())
        weights = _weights({dates[0]: {"600000": 0.5}})
        result = run_portfolio_backtest(bars, weights)
        self.assertEqual(result["unfilled_buys"], 1)
        self.assertEqual(result["open_positions"], [])
        self.assertAlmostEqual(result["daily_equity"][-1]["equity"], 1.0)

    def test_suspended_stock_is_never_traded_and_is_valued_at_last_close(self):
        bars = _bars(days=8, suspended_at={"600000": 3})
        dates = sorted(bars["date"].unique())
        weights = _weights({dates[0]: {"600000": 0.5}})
        result = run_portfolio_backtest(bars, weights)
        self.assertEqual(len(result["open_positions"]), 1)
        self.assertEqual(result["unfilled_buys"], 0)
        suspended_day = result["daily_equity"][3]
        self.assertGreater(suspended_day["position_value"], 0)

    def test_costs_reduce_equity_below_gross(self):
        bars = _bars(days=8)
        dates = sorted(bars["date"].unique())
        weights = _weights({dates[0]: {"600000": 0.5}})
        result = run_portfolio_backtest(bars, weights)
        final = result["daily_equity"][-1]
        self.assertLess(final["equity"], final["gross_equity"])
        self.assertGreater(result["metrics"]["gross_return"], result["metrics"]["total_return"])

    def test_exit_records_round_trip_cost(self):
        bars = _bars(days=8)
        dates = sorted(bars["date"].unique())
        # 目标名单在第三个信号日换成另一只股票，原持仓在下一个交易日开盘卖出。
        weights = _weights({dates[0]: {"600000": 0.5}, dates[3]: {"600001": 0.5}})
        result = run_portfolio_backtest(bars, weights)
        self.assertEqual(len(result["trades"]), 1)
        trade = result["trades"][0]
        self.assertEqual(trade["exit_date"], str(pd.Timestamp(dates[4]).date()))
        self.assertGreater(trade["gross_return"], trade["net_return"])
        expected = (1 + trade["gross_return"]) * (1 - 0.0005 - 0.0003 - 0.001) / (1 + 0.0003 + 0.001) - 1
        self.assertAlmostEqual(trade["net_return"], expected, places=10)
        self.assertGreater(trade["holding_days"], 0)

    def test_last_date_signal_cannot_change_any_trade(self):
        bars = _bars(days=8)
        dates = sorted(bars["date"].unique())
        # 最后一个交易日的信号没有下一个交易日可供执行，因此不能改变任何已发生的成交。
        base = _weights({dates[0]: {"600000": 0.5}})
        altered = pd.concat([base, _weights({dates[-1]: {"600002": 0.5}})], ignore_index=True)
        first = run_portfolio_backtest(bars, base)
        second = run_portfolio_backtest(bars, altered)
        self.assertEqual(first["trades"], second["trades"])
        self.assertEqual(first["daily_equity"], second["daily_equity"])

    def test_invalid_parameters_are_rejected(self):
        bars = _bars(days=4)
        dates = sorted(bars["date"].unique())
        weights = _weights({dates[0]: {"600000": 0.5}})
        with self.assertRaises(ValueError):
            run_portfolio_backtest(bars, weights, rebalance_every=0)
        with self.assertRaises(ValueError):
            run_portfolio_backtest(bars, weights, stamp_duty_rate=-0.1)
        with self.assertRaises(ValueError):
            run_portfolio_backtest(bars, weights, initial_equity=0)
        with self.assertRaises(ValueError):
            run_portfolio_backtest(bars, pd.DataFrame({"date": dates[:1], "code": ["600000"]}))


class PricePanelTests(unittest.TestCase):
    """未清洗的行情必须先走清洗流水线再进入回测。"""

    def test_raw_history_is_cleaned_before_use(self):
        dates = pd.bdate_range("2024-01-01", periods=5)
        raw = pd.DataFrame({"date": dates, "open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0, "volume": 1000.0})
        raw.loc[3, "volume"] = 0.0
        raw.loc[[3], ["open", "high", "low", "close"]] = np.nan
        panel = prepare_price_panel({"600000": raw})
        self.assertIn("is_usable", panel.columns)
        self.assertFalse(bool(panel.loc[panel["date"].eq(dates[3]), "is_usable"].iloc[0]))
        self.assertTrue(bool(panel.loc[panel["date"].eq(dates[0]), "is_usable"].iloc[0]))

    def test_trim_universe_keeps_most_liquid_and_reports(self):
        histories = {}
        for index in range(4):
            dates = pd.bdate_range("2024-01-01", periods=5)
            histories[f"60000{index}"] = pd.DataFrame({
                "date": dates, "close": 10.0, "volume": 1000.0 * (index + 1),
                "is_usable": True, "previous_close": 10.0, "limit_rate": 0.10,
                "is_suspended": False, "is_locked_up": False, "is_locked_down": False,
                "open": 10.0,
            })
        kept, report = trim_universe(histories, limit=2)
        self.assertEqual(len(kept), 2)
        self.assertEqual(report["trimmed"], 2)
        self.assertEqual(sorted(kept), ["600002", "600003"])
        _, untouched = trim_universe(histories, limit=10)
        self.assertEqual(untouched["trimmed"], 0)
        self.assertEqual(untouched["kept"], 4)


def _strategy_inputs(stocks=40, days=620, signal=True):
    """构造质量分层的确定性行情：质量越高的股票后续超额收益越高，成交额随序号递增。

    `signal=False` 时去掉质量分层，只留下固定种子的个股噪声，用来构造“能训练但通不过验证”的截面。
    """
    dates = pd.bdate_range("2018-01-01", periods=days)
    histories = {}
    for stock in range(stocks):
        if signal:
            quality = (stock - (stocks - 1) / 2) / max(stocks / 2, 1)
            returns = 0.0006 + quality * 0.0012 + 0.004 * np.sin(np.arange(days) * 0.7 + stock * 1.4)
        else:
            returns = 0.0002 + np.random.default_rng(1000 + stock).normal(0, 0.012, days)
        close = 20.0 * np.cumprod(1 + returns)
        volume = (1_000_000 + stock * 200_000) * (1 + 0.2 * np.sin(np.arange(days) * 0.35 + stock))
        histories[f"{600000 + stock}"] = pd.DataFrame({
            "date": dates, "open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
            "volume": volume, "amount": volume * close, "turnover_rate": 1.0 + stock * 0.1,
        })
    benchmark = pd.DataFrame({"date": dates, "close": 3000.0 * np.cumprod(1 + 0.0006 + 0.003 * np.sin(np.arange(days) * 0.7))})
    return histories, benchmark


class PortfolioStrategyTests(unittest.TestCase):
    """组合流程必须让打分与回测落在同一股票池内。"""

    def test_trimmed_universe_is_also_the_scoring_universe(self):
        histories, benchmark = _strategy_inputs()
        panel = build_market_panel(histories, benchmark)
        result = run_portfolio_strategy(
            histories, panel, benchmark=benchmark, start=250, horizon=20, retrain_every=200,
            top_n=2, max_weight=0.5, universe_limit=6, objective="lambdarank", window=400,
        )
        if result.get("metrics") is None:
            self.skipTest(f"模型未通过验证，无法检验股票池一致性：{result.get('status')}")
        self.assertEqual(result["universe"]["kept"], 6)
        self.assertIn("打分与回测均限定在该股票池内", result["universe"]["status"])
        # 成交额随序号递增，因此保留的是序号最大的 6 只。
        expected = {str(600000 + stock) for stock in range(34, 40)}
        traded = {position["code"] for position in result["open_positions"]}
        traded.update(trade["code"] for trade in result["trades"])
        self.assertTrue(traded, "组合回测应至少产生一笔成交")
        self.assertTrue(traded <= expected, f"成交了股票池外的股票：{traded - expected}")
        for row in result["daily_equity"]:
            self.assertLessEqual(row["holdings"], 6)

    def test_empty_panel_is_reported_not_crashed(self):
        result = run_portfolio_strategy({}, pd.DataFrame(), universe_limit=5)
        self.assertIsNone(result["metrics"])
        self.assertIn("缺少历史截面", result["status"])

    def test_training_log_records_why_a_block_scored_nothing(self):
        """被门槛拦住的区段必须留下可复核的检验数值，而不是只有一句“未通过验证”。"""
        histories, benchmark = _strategy_inputs(stocks=30, days=620, signal=False)
        panel = build_market_panel(histories, benchmark)
        result = run_portfolio_strategy(
            histories, panel, benchmark=benchmark, start=250, horizon=20, retrain_every=200,
            top_n=3, max_weight=0.5, objective="lambdarank", window=0,
        )
        blocks = result["training_log"]
        self.assertTrue(blocks, "滚动重训必须留下记录")
        self.assertIsNone(result["metrics"], f"噪声截面不应产生组合：{result['status']}")
        rejected = 0
        for block in blocks:
            self.assertIn("prediction_start", block)
            self.assertLess(block["trained_through"], block["prediction_start"])
            self.assertEqual([item["model"] for item in block["models"]], ["LightGBM", "XGBoost"])
            for item in block["models"]:
                self.assertEqual(item["score_kind"], SCORE_KIND_RANK)
                self.assertTrue(item["status"])
                if item["status"] != "未通过样本外基线、Rank IC或分组收益检验":
                    continue
                rejected += 1
                # 通过与否由这些数值决定，它们必须随记录一起留存。
                self.assertIsNotNone(item["rank_ic"])
                self.assertIsNotNone(item["ndcg_ratio"])
                self.assertIsNotNone(item["validation_rule"])
                self.assertIsNone(item["mae"], "排序目标不应给出无量纲分数上的绝对误差")
        self.assertTrue(rejected, "本用例必须真正走到被门槛拦下的分支")


if __name__ == "__main__":
    unittest.main()
