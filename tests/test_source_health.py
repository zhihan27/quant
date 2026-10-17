"""数据源熔断器：连续失败开断、半开单探针、成功复位与失败口径。"""

import unittest

import requests
import threading
from unittest.mock import Mock, patch

from config import settings
from services.market_data import MarketDataService
from services.source_health import SourceHealth, is_network_failure


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


class SourceHealthTests(unittest.TestCase):
    def setUp(self):
        self.clock = _Clock()
        self.health = SourceHealth(threshold=3, cooldown_seconds=300, clock=self.clock)

    def test_opens_after_consecutive_failures_and_blocks(self):
        self.assertTrue(self.health.allow("eastmoney.kline"))
        for _ in range(3):
            self.health.record_failure("eastmoney.kline")
        self.assertFalse(self.health.allow("eastmoney.kline"))
        snapshot = self.health.snapshot()
        self.assertEqual(snapshot["eastmoney.kline"]["status"], "open")
        self.assertGreater(snapshot["eastmoney.kline"]["retry_after_seconds"], 0)

    def test_success_resets_consecutive_failure_count(self):
        self.health.record_failure("tencent.kline")
        self.health.record_failure("tencent.kline")
        self.health.record_success("tencent.kline")
        self.health.record_failure("tencent.kline")
        self.health.record_failure("tencent.kline")
        self.assertTrue(self.health.allow("tencent.kline"), "成功应清空连续失败计数，而不是只减一")

    def test_half_open_allows_only_one_probe(self):
        for _ in range(3):
            self.health.record_failure("sina.history")
        self.clock.now += 300
        self.assertEqual(self.health.snapshot()["sina.history"]["status"], "half_open")
        self.assertTrue(self.health.allow("sina.history"), "冷却结束应放行一个探针")
        self.assertFalse(self.health.allow("sina.history"), "探测期间其余线程必须快速失败")

    def test_probe_success_closes_and_probe_failure_reopens(self):
        for _ in range(3):
            self.health.record_failure("eastmoney.quote")
        self.clock.now += 300
        self.assertTrue(self.health.allow("eastmoney.quote"))
        self.health.record_success("eastmoney.quote")
        self.assertTrue(self.health.allow("eastmoney.quote"))

        for _ in range(3):
            self.health.record_failure("eastmoney.quote")
        self.clock.now += 300
        self.assertTrue(self.health.allow("eastmoney.quote"))
        self.health.record_failure("eastmoney.quote")
        self.assertFalse(self.health.allow("eastmoney.quote"), "探针失败应立即重新打开")

    def test_sources_are_independent(self):
        for _ in range(3):
            self.health.record_failure("eastmoney.kline")
        self.assertFalse(self.health.allow("eastmoney.kline"))
        self.assertTrue(self.health.allow("tencent.kline"))

    def test_non_network_failure_releases_half_open_probe(self):
        service = MarketDataService.__new__(MarketDataService)
        service._health = self.health
        service._request_gate = threading.BoundedSemaphore(1)
        service._pace_request = Mock()
        service._record_request_event = Mock()
        service._last_retries = {}
        service.logger = Mock()
        service.network = Mock()
        key = service._source_key(settings.EASTMONEY_QUOTE_URL)
        for _ in range(3):
            self.health.record_failure(key)
        self.clock.now += 300
        response = requests.Response()
        response.status_code = 404
        service.network.get.return_value = response
        with self.assertRaises(requests.HTTPError):
            service._request_response(settings.EASTMONEY_QUOTE_URL, {}, "600000")
        service.network.get.return_value = Mock()
        service._request_response(settings.EASTMONEY_QUOTE_URL, {}, "000001")
        self.assertEqual(service.network.get.call_count, 2)
        self.assertEqual(self.health.snapshot()[key]["status"], "closed")

    def test_cancelled_request_releases_half_open_probe(self):
        service = MarketDataService.__new__(MarketDataService)
        service._health = self.health
        service._request_gate = threading.BoundedSemaphore(1)
        service._pace_request = Mock(side_effect=KeyboardInterrupt())
        key = service._source_key(settings.EASTMONEY_QUOTE_URL)
        for _ in range(3):
            self.health.record_failure(key)
        self.clock.now += 300
        with self.assertRaises(KeyboardInterrupt):
            service._request_response(settings.EASTMONEY_QUOTE_URL, {}, "600000")
        self.assertTrue(self.health.allow(key))

    def test_sina_business_failure_releases_half_open_probe(self):
        service = MarketDataService.__new__(MarketDataService)
        service._health = self.health
        service._request_gate = threading.BoundedSemaphore(1)
        service._pace_request = Mock()
        service._record_request_event = Mock()
        service.logger = Mock()
        key = "sina.history"
        for _ in range(3):
            self.health.record_failure(key)
        self.clock.now += 300
        with patch.object(settings, "REQUEST_RETRIES", 0), patch(
            "services.market_data.subprocess.run", return_value=Mock(stdout="[]"),
        ):
            with self.assertRaisesRegex(RuntimeError, "未返回日线"):
                service._load_sina_history("600000", True, "19900101", "daily")
        self.assertTrue(self.health.allow(key))


class NetworkFailureClassificationTests(unittest.TestCase):
    def test_connection_errors_and_timeouts_count(self):
        self.assertTrue(is_network_failure(requests.ConnectionError("断开")))
        self.assertTrue(is_network_failure(requests.Timeout("超时")))

    def test_rate_limit_and_server_errors_count(self):
        for status in (403, 429, 500, 503):
            response = requests.Response()
            response.status_code = status
            self.assertTrue(is_network_failure(requests.HTTPError("失败", response=response)), status)

    def test_client_errors_and_business_errors_do_not_count(self):
        response = requests.Response()
        response.status_code = 404
        self.assertFalse(is_network_failure(requests.HTTPError("未找到", response=response)))
        self.assertFalse(is_network_failure(ValueError("字段缺失")))


if __name__ == "__main__":
    unittest.main()
