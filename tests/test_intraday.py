"""复核分钟数据边界、持仓成本、T+1和卡尔曼提醒的失效条件。"""

import unittest
from datetime import datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from services.database import validate_position
from services.intraday_data import normalize_minutes, quote_is_fresh
from services.intraday_strategy import analyze_intraday_pair, analyze_intraday_stock, validate_monitor_config
from services.market_data import MarketDataError
from services.portfolio import summarize_positions

SHANGHAI = ZoneInfo("Asia/Shanghai")


def _minutes(count=65):
    return pd.DataFrame({
        "date": pd.date_range("2026-10-15 09:31", periods=count, freq="min", tz=SHANGHAI),
        "open": 10.0, "high": 10.1, "low": 9.9,
        "close": 10 + np.sin(np.arange(count) / 3) * 0.01,
        "volume": 100.0, "amount": 100000.0,
    })


class MinuteBoundaryTests(unittest.TestCase):
    """复核当前分钟和无效分钟线不会进入策略。"""

    def test_current_minute_is_excluded(self):
        now = datetime(2026, 10, 15, 10, 30, 30, tzinfo=SHANGHAI)
        frame = normalize_minutes(_minutes(), now)
        self.assertEqual(frame.iloc[-1]["date"].strftime("%H:%M"), "10:29")

    def test_invalid_candle_is_rejected(self):
        frame = _minutes()
        frame.loc[20, "close"] = 11
        with self.assertRaises(MarketDataError):
            normalize_minutes(frame, datetime(2026, 10, 15, 11, 0, tzinfo=SHANGHAI))

    def test_missing_amount_does_not_create_fake_amount(self):
        frame = normalize_minutes(_minutes().drop(columns="amount"), datetime(2026, 10, 15, 11, 0, tzinfo=SHANGHAI))
        self.assertTrue(frame["amount"].isna().all())

    def test_collecting_an_old_quote_does_not_make_it_fresh(self):
        now = datetime(2026, 10, 15, 10, 40, tzinfo=SHANGHAI)
        quote = {"price": 10, "quote_valid": True, "fetched_at": now.isoformat(), "quote_time": (now - timedelta(seconds=21)).isoformat()}
        self.assertFalse(quote_is_fresh(quote, now))
        quote["quote_time"] = now.isoformat()
        self.assertTrue(quote_is_fresh(quote, now))
        self.assertFalse(quote_is_fresh(quote, now.replace(hour=12)))


class PortfolioTests(unittest.TestCase):
    """复核分批持仓、费用和可卖数量。"""

    def test_cost_and_t_plus_one(self):
        now = datetime(2026, 10, 15, 10, 40, tzinfo=SHANGHAI)
        rows = [
            {"code": "600000", "name": "测试", "shares": 100, "available_shares": 100,
             "buy_price": Decimal("10"), "fees": Decimal("5"), "buy_date": now.date()},
            {"code": "600000", "name": "测试", "shares": 200, "available_shares": 150,
             "buy_price": Decimal("11"), "fees": Decimal("5"), "buy_date": now.date() - timedelta(days=1)},
        ]
        summary = summarize_positions(rows, {"600000": {"price": 12}}, now)[0]
        self.assertEqual(summary["shares"], 300)
        self.assertEqual(summary["available_shares"], 150)
        self.assertEqual(summary["cost"], 3210)
        self.assertEqual(summary["pnl"], 390)
        self.assertAlmostEqual(summary["average_cost"], 10.7)

    def test_position_validation_rejects_fractional_and_excess_shares(self):
        values = {"code": "600000", "buy_date": "2024-01-01", "buy_price": 10, "shares": 100, "available_shares": 100}
        for change in ({"shares": 1.5}, {"available_shares": 101}, {"buy_price": float("nan")}, {"shares": 0}, {"code": "6;DROP"}):
            with self.assertRaises(ValueError):
                validate_position({**values, **change})


class IntradaySignalTests(unittest.TestCase):
    """复核分钟策略不绕过行情时效、持仓可卖和参数限制。"""

    def test_configuration_rejects_busy_polling_and_same_pair(self):
        for values in ({"poll_seconds": 0}, {"entry_z": float("nan")}, {"pair_a": "600000", "pair_b": "600000"}):
            with self.assertRaises(ValueError):
                validate_monitor_config(values)

    def test_stale_quote_cannot_activate_stop(self):
        now = datetime(2026, 10, 15, 10, 36, tzinfo=SHANGHAI)
        quote = {"price": 9, "quote_valid": True, "quote_time": (now - timedelta(minutes=1)).isoformat()}
        result = analyze_intraday_stock("600000", _minutes(), quote, {"shares": 100, "available_shares": 100, "average_cost": 10}, now=now)
        self.assertFalse(result["active"])
        self.assertIn("失效", result["reason"])

    def test_unsellable_position_cannot_activate_sell(self):
        now = datetime(2026, 10, 15, 10, 36, tzinfo=SHANGHAI)
        quote = {"price": 9, "quote_valid": True, "quote_time": now.isoformat()}
        result = analyze_intraday_stock("600000", _minutes(), quote, {"shares": 100, "available_shares": 0, "average_cost": 10}, now=now)
        self.assertFalse(result["active"])
        self.assertIn("不可卖", result["action"])

    def test_sellable_stop_produces_reminder(self):
        now = datetime(2026, 10, 15, 10, 36, tzinfo=SHANGHAI)
        quote = {"price": 9, "quote_valid": True, "quote_time": now.isoformat()}
        result = analyze_intraday_stock("600000", _minutes(), quote, {"shares": 100, "available_shares": 100, "average_cost": 10}, now=now)
        self.assertTrue(result["active"])
        self.assertEqual(result["action"], "卖出提醒")

    def test_no_history_returns_waiting(self):
        result = analyze_intraday_stock("600000", pd.DataFrame(), {})
        self.assertFalse(result["active"])
        self.assertIsNone(result["bar_time"])

    def test_future_minutes_do_not_change_signal(self):
        now = datetime(2026, 10, 15, 10, 36, tzinfo=SHANGHAI)
        frame = _minutes(75)
        quote = {"price": 10, "quote_valid": True, "quote_time": now.isoformat()}
        original = analyze_intraday_stock("600000", frame, quote, now=now)
        frame.loc[65:, "close"] = 100
        changed = analyze_intraday_stock("600000", frame, quote, now=now)
        self.assertEqual(original["z_score"], changed["z_score"])
        self.assertEqual(original["bar_time"], changed["bar_time"])

    def test_pair_does_not_bypass_borrow_confirmation(self):
        now = datetime(2026, 10, 15, 10, 36, tzinfo=SHANGHAI)
        prior_dates = pd.date_range("2026-10-14 09:31", periods=120, freq="min", tz=SHANGHAI).append(
            pd.date_range("2026-10-14 13:01", periods=120, freq="min", tz=SHANGHAI)
        )
        dates = prior_dates.append(pd.DatetimeIndex(_minutes()["date"]))
        random = np.random.default_rng(2026)
        b = 10 + np.cumsum(random.normal(0, 0.01, len(dates)))
        a = 1.8 * b + 0.4 + random.normal(0, 0.0005, len(dates))
        a[-1] += 0.2
        first, second = pd.DataFrame({"date": dates, "close": a}), pd.DataFrame({"date": dates, "close": b})
        quotes = {code: {"price": float(prices[-1]), "quote_valid": True, "quote_time": now.isoformat()} for code, prices in [("600000", a), ("000001", b)]}
        result = analyze_intraday_pair("600000", "000001", first, second, quotes, now=now)
        self.assertEqual(result["training_bars"], 240)
        self.assertFalse(result["active"])
        self.assertIn("借券", result["reason"])
