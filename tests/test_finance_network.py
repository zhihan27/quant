"""复核行情代理故障恢复及限定域名、限定错误类型的回退边界。"""

import os
import threading
import unittest
from unittest.mock import Mock, patch

import requests

from config import settings
from services.finance_network import FinanceNetwork, brief_network_error
from services.market_data import MarketDataError, MarketDataService


class FinanceNetworkTests(unittest.TestCase):
    """验证故障恢复不影响系统代理及其他网络请求。"""

    def setUp(self):
        self.mode = patch.dict(os.environ, {"QUANT_NETWORK_MODE": "auto"})
        self.mode.start()
        self.session = requests.Session()
        self.network = FinanceNetwork(self.session)
        self.response = Mock(spec=requests.Response)
        self.response.status_code = 200

    def tearDown(self):
        self.network.close()
        self.session.close()
        self.mode.stop()

    def test_failed_proxy_recovers_and_remembers_direct_route(self):
        before = {key: value for key, value in os.environ.items() if "proxy" in key.lower()}
        with patch.object(self.session, "get", side_effect=requests.exceptions.ProxyError("测试代理不可达")) as proxy, patch.object(
            self.network.direct_session, "get", return_value=self.response
        ) as direct:
            self.assertIs(self.network.get("https://82.push2.eastmoney.com/api/qt/clist/get", timeout=(3, 5)), self.response)
            self.network.get("https://82.push2.eastmoney.com/api/qt/clist/get", timeout=(3, 5))
        self.assertEqual(proxy.call_count, 1)
        self.assertEqual(direct.call_count, 2)
        self.assertTrue(self.session.trust_env)
        self.assertFalse(self.network.direct_session.trust_env)
        self.assertEqual(before, {key: value for key, value in os.environ.items() if "proxy" in key.lower()})

    def test_unknown_host_does_not_bypass_proxy(self):
        for host in ("private.example", "eastmoney.com.private.example"):
            with patch.object(self.session, "get", side_effect=requests.exceptions.ProxyError("测试代理不可达")), patch.object(
                self.network.direct_session, "get"
            ) as direct:
                with self.assertRaises(requests.exceptions.ProxyError):
                    self.network.get(f"https://{host}/data")
                direct.assert_not_called()

    def test_proxy_only_mode_does_not_bypass_proxy(self):
        self.network.mode = "proxy"
        with patch.object(self.session, "get", side_effect=requests.exceptions.ProxyError("测试代理不可达")), patch.object(
            self.network.direct_session, "get"
        ) as direct:
            with self.assertRaises(requests.exceptions.ProxyError):
                self.network.get("https://hq.sinajs.cn/list=sh600000")
            direct.assert_not_called()

    def test_http_error_does_not_trigger_direct_fallback(self):
        with patch.object(self.session, "get", side_effect=requests.exceptions.HTTPError("403")), patch.object(
            self.network.direct_session, "get"
        ) as direct:
            with self.assertRaises(requests.exceptions.HTTPError):
                self.network.get("https://82.push2.eastmoney.com/data")
            direct.assert_not_called()

    def test_failed_direct_route_is_not_remembered(self):
        with patch.object(self.session, "get", side_effect=requests.exceptions.ProxyError("测试代理不可达")), patch.object(
            self.network.direct_session, "get", side_effect=requests.exceptions.ConnectTimeout("直连超时")
        ):
            with self.assertRaises(requests.exceptions.ConnectTimeout):
                self.network.get("https://82.push2.eastmoney.com/data")
        self.assertEqual(self.network.status()["direct_fallback_hosts"], [])

    def test_worker_threads_use_isolated_sessions_and_close_releases_them(self):
        snapshots = {}
        barrier = threading.Barrier(3)
        release = threading.Event()

        def worker(index):
            barrier.wait(5)
            snapshots[index] = self.network._sessions()
            for session in snapshots[index]:
                session.close = Mock()
            barrier.wait(5)
            release.wait(5)

        threads = [threading.Thread(target=worker, args=(index,)) for index in range(2)]
        for thread in threads:
            thread.start()
        # 两个线程仍在运行时先检查隔离性，再让它们退出并检查自动释放。
        barrier.wait(5)
        barrier.wait(5)

        self.assertEqual(len(snapshots), 2, "工作线程必须各拿到一套 Session")
        self.assertIs(self.network._sessions()[0], self.session, "主线程继续沿用构造时传入的 Session")
        proxies = [proxy for proxy, _ in snapshots.values()]
        self.assertEqual(len({id(proxy) for proxy in proxies}), 2, "不同线程不能共享同一个 Session")
        self.assertNotIn(self.session, proxies)
        for proxy, direct in snapshots.values():
            self.assertIsNot(direct, self.network.direct_session)
        release.set()
        for thread in threads:
            thread.join(5)
        self.assertEqual(len(self.network._worker_sessions), 0, "退出线程不能遗留连接池强引用")
        self.network.close()
        for proxy, direct in snapshots.values():
            proxy.close.assert_called_once()
            direct.close.assert_called_once()

    def test_close_releases_live_worker_sessions_only_once(self):
        ready = threading.Event()
        release = threading.Event()
        sessions = []

        def worker():
            pair = self.network._sessions()
            for session in pair:
                session.close = Mock()
            sessions.extend(pair)
            ready.set()
            release.wait(5)

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(ready.wait(5))
        try:
            self.network.close()
        finally:
            release.set()
            thread.join(5)
        for session in sessions:
            session.close.assert_called_once()

    def test_traceback_is_not_used_as_page_message(self):
        detail = "Traceback (most recent call last):\n" + "调用栈\n" * 100 + "requests.exceptions.ProxyError: Unable to connect to proxy"
        message = brief_network_error(detail)
        self.assertIn("ProxyError", message)
        self.assertNotIn("Traceback", message)
        self.assertLess(len(message), 280)


class SinaUniverseTests(unittest.TestCase):
    """复核备用源按总数完整分页、单位转换及缺页拒绝，避免退回首屏。"""

    def setUp(self):
        self.service = MarketDataService.__new__(MarketDataService)
        self.service.logger = Mock()
        self.service._cache_result = Mock(return_value=None)
        self.service._write_cache = Mock()
        self.minimum = patch.object(settings, "MINIMUM_EXPECTED_A_SHARE_UNIVERSE", 100)
        self.minimum.start()

    def tearDown(self):
        self.minimum.stop()

    @staticmethod
    def page(page):
        return [
            {"code": "920001" if index == 0 else f"600{index:03d}", "name": f"测试股票{index}",
             "trade": "10", "volume": 1000, "amount": 10000}
            for index in range((page - 1) * 80, min(page * 80, 161))
        ]

    def response(self, url, params, *arguments):
        return Mock(json=Mock(return_value="161" if "StockCount" in url else self.page(params["page"])))

    def test_all_pages_and_share_volume_conversion(self):
        self.service._request_response = Mock(side_effect=self.response)
        rows = self.service._fetch_sina_universe()
        self.assertEqual(len(rows), 161)
        self.assertEqual(rows[0]["volume_lots"], 10)
        self.assertEqual([call.args[1]["page"] for call in self.service._request_response.call_args_list[1:]], [1, 2, 3])
        self.service._write_cache.assert_called_once()

    def test_missing_last_page_is_never_cached(self):
        def response(url, params, *arguments):
            if params.get("page") == 3:
                return Mock(json=Mock(return_value=[]))
            return self.response(url, params, *arguments)
        self.service._request_response = Mock(side_effect=response)
        with self.assertRaises(MarketDataError):
            self.service._fetch_sina_universe()
        self.service._write_cache.assert_not_called()

    def test_repeated_page_is_never_cached(self):
        def response(url, params, *arguments):
            if params.get("page") == 2:
                return Mock(json=Mock(return_value=self.page(1)))
            return self.response(url, params, *arguments)
        self.service._request_response = Mock(side_effect=response)
        with self.assertRaises(MarketDataError):
            self.service._fetch_sina_universe()
        self.service._write_cache.assert_not_called()

    def test_sina_success_does_not_launch_timeout_limited_subprocess(self):
        self.service._fetch_sina_universe = Mock(return_value=self.page(1))
        with patch("services.market_data.subprocess.run") as subprocess:
            rows, source = self.service._load_akshare_spot("all")
        self.assertEqual(len(rows), 80)
        self.assertIn("新浪", source)
        subprocess.assert_not_called()
