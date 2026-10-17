"""验证因子、信号、回测约束及统计模块的基础行为。"""

import json
import threading
import time
import unittest
import sqlite3
import tempfile
from contextlib import closing
from datetime import datetime, time as datetime_time, timedelta
from pathlib import Path
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from config import settings
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


SHANGHAI = ZoneInfo("Asia/Shanghai")


def _daily_rows(dates: list[str]) -> list[dict[str, object]]:
    return [
        {
            "date": date, "open": 10.0, "close": 10.1, "high": 10.2, "low": 9.9,
            "volume": 1000.0, "amount": 10000.0,
        }
        for date in dates
    ]


def _seed_history_cache(
    service: MarketDataService, code: str, rows: list[dict[str, object]], fetched_at: str, count: int = 5
) -> None:
    """直接写入带指定抓取时间的历史缓存，用于验证交易日复用与收盘前后边界。"""
    key = f"history:v2:{code}:daily:{count}:1:1"
    payload = {"rows": rows, "source": "测试源", "last_full_refresh_at": fetched_at}
    with service._connect() as connection:
        connection.execute(
            "INSERT INTO cache_items(cache_key, payload, fetched_at) VALUES (?, ?, ?) "
            "ON CONFLICT(cache_key) DO UPDATE SET payload=excluded.payload, fetched_at=excluded.fetched_at",
            (key, json.dumps(payload, ensure_ascii=False), fetched_at),
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

    def test_board_prefixes_are_all_valid_codes(self) -> None:
        """settings.BOARD_PREFIXES 里的每个号段都必须能过代码有效性白名单。

        两处分开维护：号段加了、白名单忘了，整批股票会被静默剔出股票池，
        既不报错也不提示，只能靠这条断言把它变成一次显式失败。
        """
        rows = []
        for board, prefixes in settings.BOARD_PREFIXES.items():
            for prefix in prefixes:
                width = 6 - len(prefix)
                self.assertGreater(width, 0, f"{board} 号段 {prefix} 不可能是六位代码前缀")
                # 名称取中性字面量：含 "st" 或 "退" 会被 ST/退市过滤器拦下，干扰本用例的判定对象。
                rows.append({"code": prefix + "1" * width, "name": f"样本{prefix}"})
        frame = pd.DataFrame(rows)
        kept, _ = MarketDataService._filter_universe_rows(frame)
        self.assertEqual(sorted(kept["code"]), sorted(row["code"] for row in rows))

    def test_universe_reference_metadata_does_not_replace_quote_metadata(self) -> None:
        """名单缓存时间和过期状态不得覆盖股票池行情的来源与时效。"""
        service = Mock()
        quote_meta = {
            "complete": True, "source": "完整行情接口", "is_stale": False,
            "fetched_at": "2026-10-17T21:01:08+08:00", "status": "接口获取",
        }
        service.get_universe.return_value = (pd.DataFrame([{
            "code": "600000", "name": "普通公司", "price": 10.0,
        }]), quote_meta)
        reference_meta = {
            "source": "状态名单", "fetched_at": "2026-10-17T16:08:06+08:00",
            "status": "名单缓存", "is_stale": True,
        }
        with patch("services.scanning.load_reference", return_value=(_empty_reference(), reference_meta)):
            _, meta = candidate_codes("沪深京全部 A 股", "", 20, service)
        for key, value in quote_meta.items():
            self.assertEqual(meta[key], value)
        self.assertEqual(meta["universe_reference"], reference_meta)
        self.assertEqual(service.quote_from_universe_row.call_args.args[1]["fetched_at"], quote_meta["fetched_at"])

    def test_chinext_scope_keeps_new_302_segment(self) -> None:
        """302 号段漏登记时，这批股票会从创业板扫描里静默消失而不报错。"""
        class FakeMarketData:
            def get_universe(self, scope: str) -> tuple[pd.DataFrame, dict[str, object]]:
                rows = [
                    {"code": code, "name": name, "price": 10.0, "volume_lots": 1000.0, "amount_yuan": amount}
                    for code, name, amount in (
                        ("300001", "老创业板", 3.0), ("301001", "次新创业板", 2.0),
                        ("302132", "新号段创业板", 1.0), ("600000", "沪市主板", 4.0),
                    )
                ]
                return pd.DataFrame(rows), {
                    "complete": True, "source_unique_symbols": 4,
                    "valid_symbols": 4, "fetched_at": "2026-10-15T15:00:00+08:00",
                    "source": "test", "is_stale": False,
                }

            def quote_from_universe_row(self, row: dict[str, object], meta: dict[str, object]) -> dict[str, object]:
                return {"price": row["price"], "quote_valid": True, "data_status": "test"}

        with patch("services.scanning.load_reference", return_value=(_empty_reference(), {"status": "名单已就绪"})):
            candidates, _ = candidate_codes("创业板", "", 10, FakeMarketData())
        self.assertEqual([item["code"] for item in candidates], ["300001", "301001", "302132"])

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

    def test_history_cache_is_reused_when_no_new_trading_day(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            service = MarketDataService(db_path=Path(temporary_directory) / "reuse.db")
            day = (datetime.now(SHANGHAI) - timedelta(days=1)).date()
            rows = _daily_rows([(day - timedelta(days=1)).isoformat(), day.isoformat()])
            fetched_at = datetime.combine(day, datetime_time(16, 0), tzinfo=SHANGHAI).isoformat(timespec="seconds")
            _seed_history_cache(service, "600000", rows, fetched_at)
            with patch("services.market_data.settings.REQUEST_INTERVAL_SECONDS", 0), patch.object(
                service, "_get_json", side_effect=AssertionError("没有新交易日时不应重新请求前复权历史")
            ):
                history, meta = service.get_history(
                    "600000", count=5, adjusted=True, expected_latest_trading_date=day.isoformat()
                )
            self.assertEqual(meta["status"], "本地缓存")
            self.assertEqual(meta["latest_bar_date"], day.isoformat())
            self.assertEqual(len(history), 2)

    def test_history_cache_is_refetched_when_a_new_trading_day_appears(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            service = MarketDataService(db_path=Path(temporary_directory) / "new_day.db")
            day = (datetime.now(SHANGHAI) - timedelta(days=2)).date()
            rows = _daily_rows([(day - timedelta(days=1)).isoformat(), day.isoformat()])
            fetched_at = datetime.combine(day, datetime_time(16, 0), tzinfo=SHANGHAI).isoformat(timespec="seconds")
            _seed_history_cache(service, "600000", rows, fetched_at)
            next_day = day + timedelta(days=1)
            response = {
                "rc": 0,
                "data": {
                    "code": "600000",
                    "klines": [
                        f"{stamp.strftime('%Y-%m-%d')},10,10.1,10.2,9.9,1000,10000,3,1,0.1,1"
                        for stamp in pd.bdate_range(day.isoformat(), next_day.isoformat())
                    ],
                },
            }
            with patch("services.market_data.settings.REQUEST_INTERVAL_SECONDS", 0), patch.object(
                service, "_get_json", return_value=response
            ) as request:
                history, meta = service.get_history(
                    "600000", count=5, adjusted=True, expected_latest_trading_date=next_day.isoformat()
                )
            self.assertTrue(request.called, "出现新交易日必须整段重取前复权历史")
            self.assertEqual(meta["latest_bar_date"], next_day.isoformat())

    def test_history_cache_fetched_before_close_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            service = MarketDataService(db_path=Path(temporary_directory) / "before_close.db")
            day = (datetime.now(SHANGHAI) - timedelta(days=1)).date()
            rows = _daily_rows([(day - timedelta(days=1)).isoformat(), day.isoformat()])
            fetched_at = datetime.combine(day, datetime_time(10, 0), tzinfo=SHANGHAI).isoformat(timespec="seconds")
            _seed_history_cache(service, "600000", rows, fetched_at)
            response = {
                "rc": 0,
                "data": {
                    "code": "600000",
                    "klines": [
                        f"{stamp.strftime('%Y-%m-%d')},10,10.1,10.2,9.9,1000,10000,3,1,0.1,1"
                        for stamp in pd.bdate_range(day.isoformat(), day.isoformat())
                    ],
                },
            }
            with patch("services.market_data.settings.REQUEST_INTERVAL_SECONDS", 0), patch.object(
                service, "_get_json", return_value=response
            ) as request:
                service.get_history(
                    "600000", count=5, adjusted=True, expected_latest_trading_date=day.isoformat()
                )
            self.assertTrue(request.called, "盘中抓取的缓存可能含半根 K 线，收盘后必须重取")

    def test_force_refresh_bypasses_trading_day_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            service = MarketDataService(db_path=Path(temporary_directory) / "force.db")
            day = (datetime.now(SHANGHAI) - timedelta(days=1)).date()
            rows = _daily_rows([(day - timedelta(days=1)).isoformat(), day.isoformat()])
            fetched_at = datetime.combine(day, datetime_time(16, 0), tzinfo=SHANGHAI).isoformat(timespec="seconds")
            _seed_history_cache(service, "600000", rows, fetched_at)
            response = {
                "rc": 0,
                "data": {
                    "code": "600000",
                    "klines": [
                        f"{stamp.strftime('%Y-%m-%d')},10,10.1,10.2,9.9,1000,10000,3,1,0.1,1"
                        for stamp in pd.bdate_range(day.isoformat(), day.isoformat())
                    ],
                },
            }
            with patch("services.market_data.settings.REQUEST_INTERVAL_SECONDS", 0), patch.object(
                service, "_get_json", return_value=response
            ) as request:
                service.get_history(
                    "600000", count=5, adjusted=True,
                    expected_latest_trading_date=day.isoformat(), force_refresh=True,
                )
            self.assertTrue(request.called, "force_refresh 永远不能被缓存复用绕过")

    def test_source_breaker_skips_network_after_repeated_network_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            service = MarketDataService(db_path=Path(temporary_directory) / "breaker.db")
            with patch("services.market_data.settings.REQUEST_INTERVAL_SECONDS", 0), patch(
                "services.market_data.settings.REQUEST_RETRIES", 0
            ), patch.object(
                service.session, "get", side_effect=requests.ConnectionError("连接失败")
            ) as request:
                for _ in range(3):
                    with self.assertRaises(requests.ConnectionError):
                        service._request_response(settings.EASTMONEY_KLINE_URL, {}, code="600000")
                with self.assertRaises(MarketDataError):
                    service._request_response(settings.EASTMONEY_KLINE_URL, {}, code="600000")
            self.assertEqual(request.call_count, 3, "熔断打开后不应再发网络请求")
            snapshot = service.source_health_snapshot()
            self.assertEqual(snapshot["eastmoney.kline"]["status"], "open")

    def test_universe_snapshot_is_shared_copied_and_single_flight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            service = MarketDataService(db_path=Path(temporary_directory) / "universe.db")
            frame = pd.DataFrame({"code": ["600000", "000001"], "name": ["浦发银行", "平安银行"]})
            meta = {
                "status": "完整股票池获取成功", "complete": True,
                "fetched_at": datetime.now(SHANGHAI).isoformat(timespec="seconds"),
            }
            calls: list[str] = []

            def slow_load(scope: str, force_refresh: bool):
                calls.append(scope)
                time.sleep(0.05)
                return frame.copy(), dict(meta)

            with patch.object(service, "_load_universe", side_effect=slow_load):
                first, first_meta = service.get_universe("all")
                first["code"] = "999999"
                first["extra"] = True
                second, second_meta = service.get_universe("all")
                results: list[pd.DataFrame] = []

                def read() -> None:
                    results.append(service.get_universe("all")[0])

                threads = [threading.Thread(target=read) for _ in range(4)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

            self.assertEqual(calls, ["all"], "同一轮扫描应复用内存快照，只真正刷新一次")
            self.assertEqual(list(second["code"]), ["600000", "000001"])
            self.assertNotIn("extra", second.columns, "调用方对返回值的修改不能污染共享快照")
            self.assertIsNot(first_meta, second_meta)
            self.assertTrue(all(list(item["code"]) == ["600000", "000001"] for item in results))

    def test_universe_failure_uses_short_negative_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            service = MarketDataService(db_path=Path(temporary_directory) / "universe_fail.db")
            with patch.object(
                service, "_load_universe", side_effect=MarketDataError("股票池源不可用")
            ) as load:
                with self.assertRaises(MarketDataError):
                    service.get_universe("all")
                with self.assertRaises(MarketDataError) as blocked:
                    service.get_universe("all")
            self.assertEqual(load.call_count, 1, "失败后的短时间内不应重复拉取整份股票池")
            self.assertIn("不重复请求", str(blocked.exception))

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
