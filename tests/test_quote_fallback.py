"""验证过期报价不会阻断备用源和完整股票池快照回退。"""

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

from services.market_data import MarketDataError, MarketDataService


class QuoteFallbackTests(unittest.TestCase):
    """覆盖单股缓存过期、备用源过期和自定义扫描的恢复路径。"""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.service = MarketDataService(Path(self.temporary.name) / "quotes.db")
        self.payload = {
            "f57": "301363", "f58": "美好医疗", "f43": 1490, "f60": 1527,
            "f59": 2, "f47": 94499.63, "f86": 1792136145,
        }
        self.meta = {
            "fetched_at": "2026-10-16T14:47:47+08:00",
            "status": "行情源失败，使用过期缓存（非实时）",
            "warning": "主源连接断开",
        }

    def tearDown(self):
        self.service.network.close()
        self.service.session.close()
        self.temporary.cleanup()

    def test_stale_primary_quote_tries_available_backup(self):
        backup = {"code": "301363", "price": 14.9, "quote_valid": True}
        with patch.object(self.service, "_request_with_cache", return_value=(self.payload, self.meta)), patch.object(
            self.service, "_get_sina_quote", return_value=backup
        ) as sina, patch.object(self.service, "_close_quote_fallback") as close:
            self.assertIs(self.service.get_quote("301363"), backup)
        sina.assert_called_once()
        close.assert_not_called()

    def test_stale_backup_continues_to_daily_close(self):
        backup = {"quote_valid": False, "data_status": "新浪行情快照过期"}
        close_quote = {"price": 14.9, "quote_valid": True}
        with patch.object(self.service, "_request_with_cache", side_effect=MarketDataError("主源连接断开")), patch.object(
            self.service, "_get_sina_quote", return_value=backup
        ), patch.object(self.service, "_close_quote_fallback", return_value=close_quote) as close:
            self.assertIs(self.service.get_quote("301363"), close_quote)
        close.assert_called_once()

    def test_all_sources_failed_keeps_primary_explicitly_invalid(self):
        with patch.object(self.service, "_request_with_cache", return_value=(self.payload, self.meta)), patch.object(
            self.service, "_get_sina_quote", side_effect=MarketDataError("新浪失败")
        ), patch.object(self.service, "_close_quote_fallback", side_effect=MarketDataError("日线失败")):
            quote = self.service.get_quote("301363")
        self.assertEqual(quote["price"], 14.9)
        self.assertFalse(quote["quote_valid"])
        self.assertIn("过期", quote["data_status"])
        self.assertIn("新浪失败", quote["warning"])

    def test_custom_scan_reuses_current_universe_after_invalid_quote(self):
        candidate = {"code": "301363", "name": "美好医疗"}
        current = pd.DataFrame([{**candidate, "price": 14.9, "volume_lots": 94499.63}])
        metadata = {
            "fetched_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "source": "新浪完整股票池", "is_stale": False,
        }
        with patch.object(self.service, "get_quote", return_value={"quote_valid": False}), patch.object(
            self.service, "get_universe", return_value=(current, metadata)
        ) as refresh:
            quote = self.service.get_scan_quote(candidate, {})
        self.assertTrue(quote["quote_valid"])
        self.assertEqual(quote["price"], 14.9)
        self.assertEqual(quote["source"], "新浪完整股票池")
        refresh.assert_called_once_with(scope="all")


if __name__ == "__main__":
    unittest.main()
