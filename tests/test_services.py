"""验证因子、信号、回测约束及统计模块的基础行为。"""

import unittest
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import requests

from services.backtest import _is_locked, run_backtest
from services.feature_engineering import build_features
from services.stock_scoring import analyze_stock
from services.statistical_arbitrage import _kalman_hedge_ratio
from services.timing_strategy import build_trade_plan
from services.scanning import candidate_codes
from services.market_data import MarketDataError, MarketDataService


def _empty_reference() -> pd.DataFrame:
    """不含任何退市/ST 记录的状态名单，用来把「过滤」与「取名单」这两件事分开测。"""
    return pd.DataFrame(columns=["code", "name", "stock_status", "status_source", "delist_date"])


def _history(count: int = 420) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=count)
    trend = np.arange(count, dtype=float) * 0.025
    close = 10 + trend + np.sin(np.arange(count) / 9) * 0.25
    return pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.998,
            "high": close * 1.015,
            "low": close * 0.985,
            "close": close,
            "volume": 100000 + np.arange(count) * 100,
            "amount": close * (100000 + np.arange(count) * 100),
            "pct_change": pd.Series(close).pct_change().fillna(0) * 100,
            "turnover_rate": 1.0,
        }
    )


class QuantServiceTests(unittest.TestCase):
    def test_features_use_only_history_until_current_bar(self) -> None:
        history = _history()
        baseline = build_features(history)
        changed = history.copy()
        changed.loc[len(changed) - 1, "close"] *= 3
        altered = build_features(changed)
        self.assertAlmostEqual(baseline.loc[len(baseline) - 2, "momentum_20"], altered.loc[len(altered) - 2, "momentum_20"])
        self.assertNotAlmostEqual(baseline.iloc[-1]["momentum_20"], altered.iloc[-1]["momentum_20"])

    def test_trade_plan_uses_fixed_atr_rules(self) -> None:
        plan = build_trade_plan(10.0, 0.02, 10.2, 10.0, None, False, "2026-01-01T12:00:00+08:00")
        self.assertEqual(plan["buy_zone"], (9.95, 10.05))
        self.assertEqual(plan["initial_stop"], 9.6)
        self.assertEqual(plan["take_profit"], 10.6)
        self.assertIn("规则化参考", plan["status"])

    def test_stale_quote_disables_active_trade_signal(self) -> None:
        plan = build_trade_plan(
            10.0,
            0.02,
            10.2,
            10.0,
            None,
            False,
            "2026-01-01T12:00:00+08:00",
            data_status="行情源失败，使用过期缓存（非实时）",
        )
        self.assertFalse(plan["signal_active"])
        self.assertIn("非实时", plan["status"])

    def test_model_reports_insufficient_training_samples(self) -> None:
        result = analyze_stock(_history(180), 14.0, horizons=(60,), code="600000")
        statuses = [item["status"] for item in result["models"][60]]
        self.assertTrue(all("截面" in status for status in statuses))

    def test_model_without_market_panel_never_returns_absolute_forecast(self) -> None:
        result = analyze_stock(_history(), 14.0, horizons=(20,), code="600000", market_context={})
        for model in result["models"][20]:
            self.assertFalse(model["validated"])
            self.assertIsNone(model["predicted_return"])
            self.assertIsNone(model["predicted_price"])

    def test_backtest_requires_minimum_history(self) -> None:
        result = run_backtest(_history(100), code="600000")
        self.assertIn("样本不足", result["status"])
        self.assertIsNone(result["metrics"])

    def test_backtest_runs_three_oos_folds_with_costs(self) -> None:
        result = run_backtest(_history(420), code="600000", strategy="trend")
        self.assertEqual(len(result["folds"]), 3)
        self.assertGreater(result["metrics"]["trades"], 0)
        self.assertLess(result["metrics"]["total_return"], result["metrics"]["gross_return"])
        self.assertTrue(all(trade["entry_date"] < trade["exit_date"] for trade in result["trades"]))

    def test_custom_stock_list_parses_unique_six_digit_codes(self) -> None:
        # 用空闲名单隔离缓存/联网，这两条断言只关心代码解析与来源说明。
        with patch("services.scanning.load_reference", return_value=(_empty_reference(), {"status": "名单已就绪"})):
            candidates, meta = candidate_codes("用户自定义", "600000, 000001 600000", 10, None)
        self.assertEqual([item["code"] for item in candidates], ["600000", "000001"])
        self.assertEqual(meta["status"], "用户输入")

    def test_market_candidate_list_is_not_truncated_to_display_limit(self) -> None:
        class FakeMarketData:
            def get_universe(self, scope: str) -> tuple[pd.DataFrame, dict[str, object]]:
                rows = [
                    {"code": f"600{index:03d}", "name": f"普通公司{index}",
                     "price": 10.0, "volume_lots": 1000.0, "amount_yuan": float(index + 1)}
                    for index in range(105)
                ]
                return pd.DataFrame(rows), {
                    "complete": True, "source_unique_symbols": 105,
                    "valid_symbols": 105, "fetched_at": "2026-10-15T15:00:00+08:00",
                    "source": "test", "is_stale": False,
                }

            def quote_from_universe_row(self, row: dict[str, object], meta: dict[str, object]) -> dict[str, object]:
                return {"price": row["price"], "quote_valid": True, "data_status": "test"}

        # 本测试只验证显示上限不截断候选列表，因此用空闲名单把退市/ST 过滤隔离掉。
        with patch("services.scanning.load_reference", return_value=(_empty_reference(), {"status": "名单已就绪"})):
            candidates, _ = candidate_codes("沪深京全部 A 股", "", 8, FakeMarketData())
        self.assertEqual(len(candidates), 105)

    def test_eastmoney_universe_fetches_every_page_and_rejects_partial_page(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            service = MarketDataService(db_path=Path(temporary_directory) / "test.db")
            requested_pages: list[int] = []

            def full_page(url: str, params: dict[str, object], code: str | None = None) -> dict[str, object]:
                page = int(params["pn"])
                requested_pages.append(page)
                first_code = (page - 1) * 100 + 1
                last_code = min(page * 100, 101)
                records = [
                    {"f12": f"600{index:03d}", "f14": f"公司{index}",
                     "f2": 10, "f5": 100, "f6": 1000}
                    for index in range(first_code, last_code + 1)
                ]
                return {"rc": 0, "data": {"total": 101, "diff": records}}

            with patch.object(service, "_get_json", side_effect=full_page):
                rows, total = service._fetch_eastmoney_universe("main")
            self.assertEqual((len(rows), total), (101, 101))
            self.assertEqual(requested_pages, [1, 2])

            def partial_page(url: str, params: dict[str, object], code: str | None = None) -> dict[str, object]:
                if int(params["pn"]) == 1:
                    return full_page(url, params, code)
                return {"rc": 0, "data": {"total": 101, "diff": []}}

            with patch.object(service, "_get_json", side_effect=partial_page):
                with self.assertRaises(MarketDataError) as captured:
                    service._fetch_eastmoney_universe("main")
            self.assertIn("分页中断", str(captured.exception))

    def test_universe_filter_excludes_special_invalid_and_halted_rows(self) -> None:
        frame = MarketDataService._map_universe_rows([
            {"code": "600000", "name": "正常公司", "price": 10, "volume_lots": 100},
            {"code": "600001", "name": "ST 公司", "price": 10, "volume_lots": 100},
            {"code": "600002", "name": "*ST 公司", "price": 10, "volume_lots": 100},
            {"code": "600003", "name": "退市整理", "price": 10, "volume_lots": 100},
            {"code": "600004", "name": "停牌公司", "price": 10, "volume_lots": 0},
            {"code": "900001", "name": "B 股", "price": 10, "volume_lots": 100},
        ])
        valid, excluded = MarketDataService._filter_universe_rows(frame)
        self.assertEqual(valid["code"].tolist(), ["600000"])
        self.assertEqual(excluded, 5)

    def test_universe_filter_accepts_valid_beijing_codes(self) -> None:
        frame = MarketDataService._map_universe_rows([
            {"code": "920001", "name": "北交所普通公司", "price": 10, "volume_lots": 100},
        ])
        valid, excluded = MarketDataService._filter_universe_rows(frame)
        self.assertEqual(valid["code"].tolist(), ["920001"])
        self.assertEqual(excluded, 0)

    def test_universe_cache_rejects_first_page_sized_snapshot(self) -> None:
        rows = [{"code": f"600{index:03d}", "name": f"公司{index}"} for index in range(100)]
        payload = {
            "rows": rows,
            "meta": {"complete": True, "valid_symbols": 100, "source_unique_symbols": 100},
        }
        self.assertFalse(MarketDataService._valid_universe_cache(payload, "all"))
        self.assertFalse(MarketDataService._valid_universe_cache(payload, "main"))

    def test_universe_failure_reports_reason_without_first_page_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            service = MarketDataService(db_path=Path(temporary_directory) / "pool_failure.db")
            with patch.object(
                service, "_fetch_eastmoney_universe", side_effect=MarketDataError("ProxyError: disconnected")
            ), patch.object(
                service, "_load_akshare_spot", side_effect=MarketDataError("backup list incomplete: 100 rows")
            ):
                with self.assertRaises(MarketDataError) as captured:
                    service.get_universe("all", force_refresh=True)

            message = str(captured.exception)
            self.assertIn("未使用首屏或部分结果", message)
            self.assertIn("ProxyError", message)
            self.assertIn("100 rows", message)

    def test_history_refresh_requests_only_incremental_bars(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            service = MarketDataService(db_path=Path(temporary_directory) / "history.db")
            initial_dates = pd.bdate_range("2026-01-05", periods=3)
            next_dates = pd.bdate_range(initial_dates[-1] + pd.Timedelta(days=1), periods=2)

            def response(dates: pd.DatetimeIndex) -> dict[str, object]:
                return {
                    "rc": 0,
                    "data": {
                        "code": "600000",
                        "klines": [
                            f"{date:%Y-%m-%d},10,10.1,10.2,9.9,1000,10000,3,1,0.1,1"
                            for date in dates
                        ],
                    },
                }

            request_params: list[dict[str, object]] = []
            responses = [response(initial_dates), response(next_dates)]

            def fake_request(url: str, params: dict[str, object], code: str | None = None) -> dict[str, object]:
                request_params.append(dict(params))
                return responses[len(request_params) - 1]

            with patch("services.market_data.settings.HISTORY_CACHE_SECONDS", 0), patch(
                "services.market_data.settings.FULL_HISTORY_REFRESH_SECONDS", 1_000_000
            ), patch("services.market_data.settings.REQUEST_INTERVAL_SECONDS", 0):
                with patch.object(service, "_get_json", side_effect=fake_request):
                    service.get_history("600000", count=5, adjusted=False)
                    history, meta = service.get_history("600000", count=5, adjusted=False)

            self.assertEqual(len(history), 5)
            self.assertIn("beg", request_params[1])
            self.assertEqual(meta["latest_bar_date"], next_dates[-1].strftime("%Y-%m-%d"))

    def test_history_summary_reports_per_stock_request_success_rate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            service = MarketDataService(db_path=Path(temporary_directory) / "history_stats.db")
            service._record_history_status(
                "600000", "daily", "前复权", "失败", None, 0,
                "东方财富公开日线接口", "ProxyError: proxy disconnected", 2,
            )
            service._record_request_event(
                "push2his.eastmoney.com/api/qt/stock/kline/get", "600000", 120.0,
                None, 0, True, None,
            )
            service._record_request_event(
                "push2his.eastmoney.com/api/qt/stock/kline/get", "600000", 350.0,
                "ProxyError", 1, False, "proxy disconnected",
            )

            summary = service.history_fetch_summary()

            self.assertEqual(summary["failure_count"], 1)
            self.assertEqual(summary["by_stock"][0]["success_rate_pct"], 50.0)
            self.assertIn("ProxyError", summary["by_stock"][0]["failure_reason"])

    def test_sina_history_normalizes_volume_and_turnover_units(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            service = MarketDataService(db_path=Path(temporary_directory) / "sina_units.db")
            output = [{
                "date": "2026-10-14", "open": 10, "high": 11, "low": 9,
                "close": 10.5, "volume": 25000, "amount": 262500,
                "turnover": 0.004,
            }]
            completed = Mock(stdout=pd.DataFrame(output).to_json(orient="records"))
            with patch("services.market_data.settings.REQUEST_INTERVAL_SECONDS", 0), patch(
                "services.market_data.subprocess.run", return_value=completed
            ):
                rows = service._load_sina_history("600000", True, "20261001", "daily")

            self.assertEqual(rows[0]["volume"], 250)
            self.assertAlmostEqual(rows[0]["turnover_rate"], 0.4)

    def test_request_retries_and_persists_attempt_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database = Path(temporary_directory) / "requests.db"
            service = MarketDataService(db_path=database)
            failed_response = Mock(status_code=502)
            failed_response.raise_for_status.side_effect = requests.HTTPError("502 Bad Gateway", response=failed_response)
            success_response = Mock(status_code=200)
            success_response.raise_for_status.return_value = None
            with patch("services.market_data.settings.REQUEST_INTERVAL_SECONDS", 0), patch(
                "services.market_data.settings.RETRY_BACKOFF_SECONDS", 0
            ), patch.object(service.session, "get", side_effect=[failed_response, success_response]) as request:
                result = service._request_response("https://example.test/quote", {}, code="600000")
            self.assertIs(result, success_response)
            self.assertEqual(request.call_count, 2)
            with closing(sqlite3.connect(database)) as connection:
                events = connection.execute(
                    "SELECT stock_code, elapsed_ms, exception_type, retry_count, success FROM request_events ORDER BY event_id"
                ).fetchall()
            self.assertEqual(events[0][0], "600000")
            self.assertGreaterEqual(events[0][1], 0)
            self.assertEqual(events[0][2], "HTTPError")
            self.assertEqual([row[4] for row in events], [0, 1])

    def test_locked_limit_up_is_not_buyable(self) -> None:
        bar = pd.Series({"volume": 1000, "pct_change": 9.9, "high": 11.0, "low": 11.0, "close": 11.0})
        self.assertTrue(_is_locked(bar, "buy", 0.10))
        self.assertFalse(_is_locked(bar, "sell", 0.10))

    def test_kalman_hedge_ratio_is_finite(self) -> None:
        independent = np.linspace(2.0, 3.0, 200)
        dependent = 1.4 * independent + np.sin(np.arange(200) / 8) * 0.01
        beta, intercept = _kalman_hedge_ratio(dependent, independent)
        self.assertTrue(np.isfinite(beta).all())
        self.assertTrue(np.isfinite(intercept).all())
        self.assertGreater(beta[-1], 0.5)


if __name__ == "__main__":
    unittest.main()
