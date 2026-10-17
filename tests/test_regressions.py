"""复核套利、成交、逐日净值、模型排序及数据缓存的关键边界。"""

import logging
import os
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from config import settings
from services.backtest import _open_fillable, _limit_rate, run_backtest
from services.market_data import MarketDataError, MarketDataService, _market_id, history_is_current
from services.scanning import rank_candidates
from services.statistical_arbitrage import analyze_pair
from services.stock_scoring import analyze_stock, walk_forward_predictions
from services.timing_strategy import build_trade_plan


def _bars(count=180):
    return pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=count),
        "open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0,
        "volume": 1000.0, "pct_change": 0.0,
    })


def _factors(frame, benchmark=None, atr=0.02):
    result = frame.copy()
    result["ma20_gap"] = 0.01
    result["ma60_gap"] = 0.05
    result["atr_pct"] = atr
    result["volume_ma20"] = 2000.0
    result.loc[90, "volume_ma20"] = 500.0
    return result


def _run(frame, atr=0.02, **kwargs):
    with patch("services.backtest.build_features", side_effect=lambda data, benchmark=None: _factors(data, benchmark, atr)):
        return run_backtest(
            frame, code="600000", strategy="trend", holding_days=5,
            commission_rate=0, stamp_duty_rate=0, slippage_rate=0, **kwargs,
        )


class StrategyRegressionTests(unittest.TestCase):
    """验证交易条件、成交先后和持仓净值，不依赖外部行情。"""

    def test_invalid_prices_never_create_active_signals(self):
        for value in (float("nan"), float("inf"), -1, 0):
            plan = build_trade_plan(value, 0.02, 9, 8, 0.1, True, data_status="盘中行情")
            self.assertFalse(plan["signal_active"])
            self.assertNotIn("buy_zone", plan)

    def test_all_entry_conditions_are_required(self):
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        arguments = dict(current_price=10, atr_pct=0.02, ma20=9.5, ma60=9,
                         relative_signal=0.1, model_ready=True, as_of=stamp,
                         data_status="盘中行情", volume=1000, average_volume=900)
        self.assertTrue(build_trade_plan(**arguments)["signal_active"])
        for change in ({"model_ready": False}, {"relative_signal": -0.1},
                       {"ma20": 8}, {"volume": 500}, {"history_ready": False},
                       {"data_status": "未知"}, {"as_of": "2000-01-01T10:00:00+08:00"}):
            self.assertFalse(build_trade_plan(**{**arguments, **change})["signal_active"])

    def test_opening_limit_fill_does_not_use_later_prices(self):
        bar = pd.Series({"open": 11, "high": 11, "low": 11, "close": 11, "volume": 1000}, dtype=float)
        changed = bar.copy()
        changed["low"], changed["close"] = 10.5, 10.6
        self.assertFalse(_open_fillable(bar, "buy", 10, 0.1))
        self.assertFalse(_open_fillable(changed, "buy", 10, 0.1))

    def test_unfilled_position_remains_in_equity(self):
        bars = _bars()
        bars.loc[92, ["open", "high", "low", "close", "pct_change"]] = [9, 9, 9, 9, -10]
        bars.loc[93, ["open", "high", "low", "close", "pct_change"]] = [8.1, 8.1, 8.1, 8.1, -10]
        bars.loc[94:, ["open", "high", "low", "close", "volume"]] = [8.1, 8.1, 8.1, 8.1, 0]
        result = _run(bars)
        self.assertIsNotNone(result["open_position"])
        self.assertAlmostEqual(result["metrics"]["total_return"], -0.19)
        self.assertAlmostEqual(result["metrics"]["maximum_drawdown"], -0.19)
        self.assertEqual(result["metrics"]["trades"], 0)

    def test_suspended_position_carries_across_folds(self):
        bars = _bars()
        bars.loc[92:119, ["open", "high", "low", "close", "volume"]] = [9, 9, 9, 9, 0]
        result = _run(bars)
        self.assertEqual(len(result["trades"]), 1)
        self.assertEqual(result["trades"][0]["exit_date"], str(bars.loc[120, "date"].date()))
        self.assertAlmostEqual(result["daily_equity"][29]["position_value"], 0.9)

    def test_daily_drawdown_includes_recovered_losses(self):
        bars = _bars()
        bars.loc[92, ["open", "high", "low", "close"]] = [10, 10, 7, 7]
        bars.loc[93, ["open", "high", "low", "close"]] = [7, 10, 7, 10]
        result = _run(bars, atr=0.25)
        self.assertAlmostEqual(result["metrics"]["maximum_drawdown"], -0.3)
        self.assertAlmostEqual(result["metrics"]["total_return"], 0)
        self.assertEqual(len(result["daily_equity"]), 90)

    def test_gap_take_profit_executes_at_open(self):
        bars = _bars()
        bars.loc[92, ["open", "high", "low", "close"]] = [11, 11.2, 11, 11.1]
        result = _run(bars)
        self.assertEqual(result["trades"][0]["exit_price"], 11)

    def test_entry_must_be_inside_reference_zone(self):
        bars = _bars()
        bars.loc[91, "open"] = 10.5
        bars.loc[91, "high"] = 10.6
        result = _run(bars)
        self.assertEqual(result["trades"], [])
        self.assertEqual(result["unfilled_entries"], 1)

    def test_model_backtest_cannot_trade_without_validated_predictions(self):
        bars = _bars(420)
        no_predictions = pd.Series(np.nan, index=bars.index)
        with patch("services.backtest.walk_forward_predictions", return_value=(no_predictions, [])) as fit:
            result = run_backtest(bars)
        fit.assert_called_once()
        self.assertEqual(result["strategy"], "model")
        self.assertEqual(result["trades"], [])
        self.assertIn("未生成通过验证的模型信号", result["status"])
        self.assertEqual(result["metrics"]["total_return"], 0)

    def test_benchmark_and_equity_start_at_same_previous_close(self):
        bars = _bars()
        benchmark = bars[["date", "close"]].copy()
        benchmark.loc[90:, "close"] = 11
        result = _run(bars, benchmark=benchmark)
        self.assertAlmostEqual(result["benchmark_buy_hold_return"], 0.1)

    def test_negative_cost_is_rejected(self):
        with self.assertRaises(ValueError):
            run_backtest(_bars(), commission_rate=-0.01)

    def test_invalid_candle_cannot_create_impossible_fill(self):
        bars = _bars()
        bars.loc[91, "open"] = 20
        with self.assertRaises(ValueError):
            _run(bars)

    def test_split_does_not_create_artificial_position_loss(self):
        adjusted = _bars()
        raw = adjusted.copy()
        raw.loc[92:, ["open", "high", "low", "close"]] *= 0.5
        result = _run(raw, adjusted_history=adjusted)
        self.assertAlmostEqual(result["metrics"]["total_return"], 0)
        self.assertAlmostEqual(result["metrics"]["maximum_drawdown"], 0)

    def test_forecast_price_uses_complete_bar_not_intraday_quote(self):
        result = analyze_stock(_bars(), current_price=20, horizons=(20,), market_context={})
        self.assertEqual(result["prediction_reference_price"], 10)
        self.assertTrue(all(item["predicted_price"] is None for item in result["models"][20]))

    def test_beijing_market_route_and_price_limit(self):
        self.assertEqual(_market_id("920001"), 0)
        self.assertEqual(_limit_rate("920001"), 0.3)
        self.assertEqual(_limit_rate("300001", "2020-01-01"), 0.1)

    def test_model_return_changes_composite_ranking(self):
        def record(code, factor, prediction):
            return {"code": code, "features": {
                "momentum_20": factor, "momentum_60": factor,
                "trend_strength": factor, "relative_strength_20": factor, "volatility_20": 0.1,
            }, "analysis": {"models": {20: [
                {"model": "LightGBM", "validated": True, "target_kind": "excess_return_vs_csi300", "prediction_zscore": prediction, "prediction_percentile": (prediction + 1) / 2},
                {"model": "XGBoost", "validated": True, "target_kind": "excess_return_vs_csi300", "prediction_zscore": prediction, "prediction_percentile": (prediction + 1) / 2},
            ]}}}
        extras = [record(str(index), 0, -1) for index in range(3)]
        first = rank_candidates([record("甲", 2, -0.5), record("乙", 1, 0.5), *extras])
        second = rank_candidates([record("甲", 2, 0.5), record("乙", 1, -0.5), *extras])
        self.assertEqual(first[0]["code"], "乙")
        self.assertEqual(second[0]["code"], "甲")
        invalid = record("丙", 3, 0.9)
        for model in invalid["analysis"]["models"][20]:
            model["validated"] = False
        self.assertIsNone(rank_candidates([invalid])[0]["cross_section_score"])

    def test_walk_forward_models_only_receive_past_data(self):
        features = _bars(400)
        with patch("services.ml_pipeline.load_context", return_value=(None, None)):
            predictions, log = walk_forward_predictions(features, 320, code="600000")
        self.assertTrue(predictions.isna().all())
        self.assertIn("缺少", log[0]["status"])


class PairRegressionTests(unittest.TestCase):
    """验证双腿计账方向、执行时点及未来数据隔离。"""

    def setUp(self):
        random = np.random.default_rng(42)
        count = 600
        b = 2 + np.cumsum(random.normal(0, 0.012, count))
        error = np.zeros(count)
        for index in range(1, count):
            error[index] = 0.65 * error[index - 1] + random.normal(0, 0.002)
        dates = pd.bdate_range("2020-01-01", periods=count)
        self.a = pd.DataFrame({"date": dates, "close": np.exp(0.4 + b + error)})
        self.b = pd.DataFrame({"date": dates, "close": np.exp(b)})

    def analyze(self, a=None):
        a = self.a if a is None else a
        def coint(first, second):
            probability = 0.9 if np.std(first - second) > 0.03 else 0.001
            return 0, probability, None
        with patch("statsmodels.tsa.stattools.coint", side_effect=coint), patch(
            "services.statistical_arbitrage._kalman_hedge_ratio",
            return_value=(np.ones(600), np.full(600, 0.4)),
        ):
            return analyze_pair(a, self.b, "甲", "乙", commission_rate=0, slippage_rate=0, stamp_duty_rate=0)

    def test_pair_profit_matches_executed_long_short_weights(self):
        result = self.analyze()
        self.assertTrue(result["training_stable"])
        a = self.a.set_index("date")["close"].pct_change()
        b = self.b.set_index("date")["close"].pct_change()
        active = [row for row in result["backtest_path"] if row["position"]]
        self.assertTrue(active)
        for row in active:
            date = pd.Timestamp(row["date"])
            self.assertAlmostEqual(row["net_return"], row["weight_a"] * a.loc[date] + row["weight_b"] * b.loc[date])
            self.assertAlmostEqual(abs(row["weight_a"]) + abs(row["weight_b"]), 1)
            self.assertEqual(np.sign(row["weight_a"]), row["position"])
            self.assertLess(row["signal_date"], row["date"])

    def test_changing_future_does_not_remove_past_trades(self):
        original = self.analyze()
        changed = self.a.copy()
        changed.loc[540:, "close"] *= np.exp(np.linspace(0, 0.2, 60))
        altered = self.analyze(changed)
        first_path = [row for row in original["backtest_path"] if row["date"] < str(self.a.loc[540, "date"].date())]
        second_path = [row for row in altered["backtest_path"] if row["date"] < str(self.a.loc[540, "date"].date())]
        self.assertEqual(first_path, second_path)
        self.assertTrue(altered["training_stable"])
        self.assertFalse(altered["current_stable"])


class CacheRegressionTests(unittest.TestCase):
    """验证前复权替换、定期刷新、空返回和中文子进程响应。"""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.service = MarketDataService(self.directory / "test.db")

    def tearDown(self):
        self.service.session.close()
        self.temporary.cleanup()

    def response(self, dates, price=10):
        return {"data": {"code": "600000", "klines": [
            f"{date}," + ",".join(map(str, [price, price, price * 1.01, price * 0.99, 1000, 100000, 2, 0, 0, 1])) for date in dates
        ]}}

    def test_adjusted_refresh_replaces_entire_series(self):
        dates = pd.bdate_range("2024-01-01", periods=15).strftime("%Y-%m-%d").tolist()
        with patch.object(self.service, "_get_json", side_effect=[self.response(dates), self.response(dates, 5)]) as request:
            self.service.get_history("600000", count=15)
            history, _ = self.service.get_history("600000", count=15, force_refresh=True)
        self.assertTrue(history["close"].eq(5).all())
        self.assertNotIn("beg", request.call_args_list[-1].args[1])

    def test_empty_response_keeps_old_cache_explicitly_stale(self):
        dates = ["2024-01-01", "2024-01-02"]
        with patch.object(self.service, "_get_json", return_value=self.response(dates)):
            self.service.get_history("600000", count=5)
        with patch.object(self.service, "_get_json", return_value=self.response([])), patch.object(
            self.service, "_load_sina_history", side_effect=MarketDataError("备用源失败")
        ) as fallback:
            _, metadata = self.service.get_history("600000", count=5, force_refresh=True)
        self.assertIn("过期", metadata["status"])
        fallback.assert_called_once()

    def test_weekly_full_refresh_clock_survives_incremental_updates(self):
        class TestClock(datetime):
            value = datetime(2024, 1, 1, 16, tzinfo=ZoneInfo("Asia/Shanghai"))

            @classmethod
            def now(cls, tz=None):
                return cls.value.astimezone(tz) if tz else cls.value.replace(tzinfo=None)

        requests = []
        def load(url, params, code=None):
            requests.append("beg" not in params)
            return self.response([TestClock.value.date().isoformat()])
        with patch("services.market_data.datetime", TestClock), patch.object(self.service, "_get_json", side_effect=load):
            for day in range(10):
                TestClock.value = datetime(2024, 1, 1, 16, tzinfo=ZoneInfo("Asia/Shanghai")) + timedelta(days=day)
                self.service.get_history("600000", count=15, adjusted=False, force_refresh=True)
        self.assertEqual([index for index, full in enumerate(requests) if full], [0, 7])

    def test_real_child_process_returns_chinese_as_utf8(self):
        module = self.directory / "akshare.py"
        module.write_text(
            'import pandas as pd\ndef stock_info_a_code_name():\n    return pd.DataFrame([{"code":"600000","name":"中文股票"}])\n',
            encoding="utf-8",
        )
        with patch.dict(os.environ, {"PYTHONPATH": str(self.directory)}):
            result = self.service._load_akshare_code_table()
        self.assertEqual(result.iloc[0]["name"], "中文股票")

    def test_scan_expired_snapshot_fetches_current_pool(self):
        candidate = {"code": "600000", "price": 10, "volume_lots": 1000}
        old = (datetime.now(ZoneInfo("Asia/Shanghai")) - timedelta(seconds=901)).isoformat()
        fresh = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        with patch.object(self.service, "get_universe", return_value=(
            pd.DataFrame([{**candidate, "price": 11}]), {"fetched_at": fresh, "is_stale": False}
        )) as refresh:
            quote = self.service.get_scan_quote(candidate, {"fetched_at": old, "scope": "main"})
        self.assertEqual(quote["price"], 11)
        self.assertTrue(quote["quote_valid"])
        refresh.assert_called_once_with(scope="main")

    def test_close_quote_cannot_activate_yesterday_history(self):
        quote = {"quote_valid": True, "quote_time": "2026-10-15T15:01:00+08:00"}
        history = pd.DataFrame({"date": [pd.Timestamp("2026-10-14")]})
        self.assertFalse(history_is_current(history, quote, {"status": "接口获取"}))

    def test_history_without_source_time_requires_benchmark_date(self):
        """下载时间不代表交易时间；缺少交易日锚点时不能放行历史。"""
        quote = {
            "quote_valid": True, "quote_time": None,
            "fetched_at": "2026-10-17T21:01:08+08:00",
        }
        history = pd.DataFrame({"date": [pd.Timestamp("2026-10-16")]})
        meta = {"status": "本地缓存"}
        self.assertTrue(history_is_current(history, quote, meta, "2026-10-16"))
        self.assertFalse(history_is_current(history, quote, meta))
        self.assertFalse(history_is_current(history, quote, meta, "2026-10-19"))
        self.assertFalse(history_is_current(history, quote, {"status": "过期缓存"}, "2026-10-16"))
        self.assertFalse(history_is_current(history, {**quote, "quote_valid": False}, meta, "2026-10-16"))

    def test_actual_quote_time_still_rejects_history_behind_quote(self):
        """即使基准较旧，真实收盘报价已更新时也不能放行旧日线。"""
        quote = {"quote_valid": True, "quote_time": "2026-10-19T15:01:00+08:00"}
        history = pd.DataFrame({"date": [pd.Timestamp("2026-10-16")]})
        self.assertFalse(history_is_current(history, quote, {"status": "接口获取"}, "2026-10-16"))
