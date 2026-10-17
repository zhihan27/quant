"""验证无界面扫描编排：不依赖 Streamlit、输入校验、以及组合回测与逐股历史的先后次序。"""

import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import patch

import pandas as pd

from config import settings
from services.scan_pipeline import (
    FULL_MARKET_POOL_MODE,
    POOL_MODES,
    ScanOptions,
    _collect_candidates,
    run_scan_pipeline,
)


def _history(end: str = "2026-10-15", bars: int = 400) -> pd.DataFrame:
    """造一段足够长的已完成日线，末根为 end，用来通过历史样本长度与新鲜度检查。"""
    dates = pd.bdate_range(end=end, periods=bars)
    return pd.DataFrame(
        {
            "date": [stamp.date().isoformat() for stamp in dates],
            "open": [10.0] * bars,
            "close": [10.0] * bars,
            "high": [10.5] * bars,
            "low": [9.5] * bars,
            "volume": [1_000_000] * bars,
            "amount": [10_000_000] * bars,
        }
    )


class _FakeService:
    """只实现扫描编排真正调用到的行情接口，其余一律不去联网。"""

    def __init__(self) -> None:
        self.frame = _history()
        self.portfolio_histories: dict[str, pd.DataFrame] = {}

    def get_scan_quote(self, candidate, source_meta):
        return {
            "name": candidate["name"], "price": 10.0, "volume_lots": 1000, "change_pct": 1.0,
            "quote_valid": True, "quote_time": "2026-10-15 15:00:00", "data_status": "实时",
        }

    def get_history(self, code, count=1200, adjusted=True, market_id=None, expected_latest_trading_date=None):
        return self.frame.copy(), {"status": "实时"}


class StreamlitFreeTests(unittest.TestCase):
    """定时任务在没有网页的进程里跑同一套编排，模块不能把 Streamlit 捎带进来。"""

    def test_importing_pipeline_does_not_load_streamlit(self):
        # 必须开子进程：unittest discover 会在同一进程里先跑别的用例，那时 streamlit 早已入 sys.modules。
        code = "import sys; import services.scan_pipeline; sys.exit(1 if 'streamlit' in sys.modules else 0)"
        completed = subprocess.run(
            [sys.executable, "-c", code], cwd=settings.PROJECT_ROOT, capture_output=True, text=True
        )
        self.assertEqual(completed.returncode, 0, f"services.scan_pipeline 把 Streamlit 引了进来：{completed.stderr}")


class ScanOptionsTests(unittest.TestCase):
    """services/scanning.py 用 .get(mode, "all") 兜底，写错标签会静默扫成全市场而不是报错。"""

    def test_known_pool_modes_are_accepted(self):
        for mode in POOL_MODES:
            self.assertEqual(ScanOptions(pool_mode=mode).pool_mode, mode)

    def test_unknown_pool_mode_is_rejected(self):
        for wrong in ("沪深全部A股", "全部A股", "沪深京全部A股", ""):
            with self.assertRaises(ValueError):
                ScanOptions(pool_mode=wrong)

    def test_daily_scan_profile_builds_a_scheduled_all_market_option(self):
        """定时任务的口径必须与管理员手动全市场扫描一致，且不得打开逐股回测。"""
        options = ScanOptions(scan_limit=settings.DAILY_SCAN_TOP_ROWS, trigger="scheduled", **settings.DAILY_SCAN)
        self.assertEqual(options.pool_mode, "沪深京全部 A 股")
        self.assertEqual(options.trigger, "scheduled")
        self.assertFalse(options.include_backtest)
        self.assertFalse(options.run_portfolio)

    def test_full_market_mode_is_the_scope_shared_by_page_and_schedule(self):
        """只有全市场口径会落盘给所有账号看，因此页面与定时任务必须引用同一个常量。"""
        self.assertIn(FULL_MARKET_POOL_MODE, POOL_MODES)
        self.assertEqual(settings.DAILY_SCAN["pool_mode"], FULL_MARKET_POOL_MODE)


class _ConcurrencyService(_FakeService):
    """记录采集阶段的在途并发数与历史调用参数，用来验证并发只发生在采集层。"""

    def __init__(self, delay: float = 0.0, failing: tuple[str, ...] = ()) -> None:
        super().__init__()
        self.delay = delay
        self.failing = set(failing)
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.history_calls: list[tuple[str, str | None]] = []

    def get_scan_quote(self, candidate, source_meta):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            return super().get_scan_quote(candidate, source_meta)
        finally:
            with self.lock:
                self.active -= 1

    def get_history(self, code, count=1200, adjusted=True, market_id=None, expected_latest_trading_date=None):
        self.history_calls.append((code, expected_latest_trading_date))
        if code in self.failing:
            raise RuntimeError("模拟历史源失败")
        return super().get_history(code, count=count, adjusted=adjusted, market_id=market_id)


class CollectionConcurrencyTests(unittest.TestCase):
    """采集并发必须保序、按预期交易日复用缓存、并且把进度限制在主线程少量上报。"""

    @staticmethod
    def _candidates(count: int) -> list[dict]:
        return [{"code": f"6000{index:02d}", "name": f"样本{index}"} for index in range(count)]

    def _collect(self, service, candidates, progress=None, expected="2026-10-15", workers=4, gate=4):
        with patch.object(settings, "SCAN_COLLECTION_WORKERS", workers), patch.object(
            settings, "MAX_CONCURRENT_REQUESTS", gate
        ):
            return _collect_candidates(
                service, candidates, {"status": "测试"}, expected, progress
            )

    def test_four_way_collection_keeps_candidate_order_and_runs_concurrently(self):
        service = _ConcurrencyService(delay=0.02)
        candidates = self._candidates(12)
        records = self._collect(service, candidates)
        self.assertEqual([record["code"] for record in records], [item["code"] for item in candidates])
        self.assertGreater(service.max_active, 1, "4 路配置下采集应真正并发出在途请求")
        self.assertTrue(all("history" in record for record in records))
        self.assertEqual(len(service.history_calls), len(candidates))
        self.assertEqual(
            set(service.history_calls),
            {(item["code"], "2026-10-15") for item in candidates},
            "预期交易日必须传给每一只股票，供历史缓存复用判断",
        )

    def test_single_worker_falls_back_to_serial_collection(self):
        service = _ConcurrencyService(delay=0.01)
        records = self._collect(service, self._candidates(6), workers=1)
        self.assertEqual(len(records), 6)
        self.assertEqual(service.max_active, 1, "1 线程时必须保留串行回退路径")

    def test_progress_reports_are_throttled_but_end_at_n_over_n(self):
        messages: list[tuple[str, int | None, int | None]] = []
        records = self._collect(
            _ConcurrencyService(), self._candidates(120),
            progress=lambda message, completed, total: messages.append((message, completed, total)),
        )
        self.assertEqual(len(records), 120)
        completed_reports = [item for item in messages if item[1] is not None]
        self.assertEqual(
            [item[1] for item in completed_reports], [1, 51, 101, 120],
            "进度应按 50 的整数倍上报并补齐末条，而不是逐只刷屏",
        )
        self.assertEqual(completed_reports[-1][2], 120)
        self.assertEqual(len(messages), 5, "120 只股票只应有 1 条起始消息 + 4 条完成消息")

    def test_single_stock_failure_does_not_abort_the_batch(self):
        service = _ConcurrencyService(failing=("600003",))
        records = self._collect(service, self._candidates(6))
        self.assertEqual([record["code"] for record in records], [f"6000{index:02d}" for index in range(6)])
        failed = records[3]
        self.assertIn("RuntimeError", failed["error"])
        self.assertTrue(all("history" in record for index, record in enumerate(records) if index != 3))

    def test_cancel_stops_submission_and_checkpoints_in_flight_results(self):
        class Cancelled(BaseException):
            """模拟页面重跑或用户停止时抛出的控制异常。"""

        cancelled = threading.Event()
        fetched = []
        saved = []
        callbacks = []
        main_thread = threading.current_thread()

        def collect(service, candidate, meta, reference, **kwargs):
            fetched.append(candidate["code"])
            if candidate["code"] != "600000":
                self.assertTrue(cancelled.wait(5))
            return {"code": candidate["code"]}

        def progress(message, completed=None, total=None):
            if completed:
                callbacks.append(completed)
                cancelled.set()
                raise Cancelled()

        def checkpoint(index, record):
            self.assertIs(threading.current_thread(), main_thread)
            saved.append(record["code"])

        with patch.object(settings, "SCAN_COLLECTION_WORKERS", 2), patch.object(
            settings, "MAX_CONCURRENT_REQUESTS", 2,
        ), patch("services.scan_pipeline._collect_one", side_effect=collect):
            with self.assertRaises(Cancelled):
                _collect_candidates(None, self._candidates(20), {}, "2026-10-15", progress, on_record=checkpoint)
        self.assertLessEqual(len(fetched), 2, "中断后不能继续提交剩余股票")
        self.assertEqual(set(saved), set(fetched), "已在途的股票收尾后也必须保存断点")
        self.assertEqual(callbacks, [1], "中断收尾不能再次触发页面操作")


class PipelineOrderTests(unittest.TestCase):
    """组合回测与逐股评分都依赖扫描期间收集的逐股历史，释放必须排在两者之后。

    原页面靠代码恰好写对才成立；搬进服务层后用这条用例把次序钉住。
    """

    def _run(self, pool_mode: str = "用户自定义"):
        service = _FakeService()
        calls: list[tuple] = []

        def fake_portfolio(histories, panel, **kwargs):
            service.portfolio_histories = histories
            calls.append(("portfolio", len(histories), sum(len(frame) for frame in histories.values())))
            return {"metrics": {"total_return": 0.1}}

        def fake_analyze(history, **kwargs):
            calls.append(("analyze", kwargs.get("code"), len(history)))
            return {
                "latest_features": {}, "models": {20: []}, "trade_plan": {},
                "prediction_reference_date": "2026-10-15", "market_context_id": "ctx", "factor_score": 50.0,
            }

        context = {
            "context_id": "fake-context", "reference_date": "2026-10-15", "objective_mode": "lambdarank",
            "training_window_days": 750, "neutralization": {"status": "未启用"}, "limitation": "测试用上下文",
            "models": {}, "predictions": {},
        }
        panel = pd.DataFrame({"date": ["2026-10-15"], "code": ["600000"]})
        options = ScanOptions(
            pool_mode=pool_mode, custom_codes="600000,000001" if pool_mode == "用户自定义" else "",
            enable_neutralization=False, use_extra_factors=False, run_portfolio=True,
        )
        candidates = [{"code": "600000", "name": "浦发银行"}, {"code": "000001", "name": "平安银行"}]
        with patch("services.scan_pipeline.train_market_context", return_value=context) as train_context, patch(
            "services.scan_pipeline.load_context", return_value=(context, panel)
        ), patch("services.scan_pipeline.run_portfolio_strategy", side_effect=fake_portfolio), patch(
            "services.scan_pipeline.analyze_stock", side_effect=fake_analyze
        ), patch("services.scan_pipeline.rank_candidates", return_value=[]):
            result = run_scan_pipeline(service, options, candidates, {"status": "测试"}, None)
        self.train_context_kwargs = train_context.call_args.kwargs
        return service, result, calls

    def test_portfolio_runs_before_scoring_and_history_survives_both(self):
        service, result, calls = self._run()
        self.assertEqual(calls[0][0], "portfolio", "组合回测必须早于逐股评分：它同样要用逐股历史")
        self.assertEqual(calls[0][1], 2, "组合回测应拿到全部有效股票的逐股历史")
        self.assertGreater(calls[0][2], 0, "组合回测拿到的逐股历史不能是空的")
        analyzed = [call for call in calls if call[0] == "analyze"]
        self.assertEqual(len(analyzed), 2)
        for call in analyzed:
            self.assertGreater(call[2], 0, f"{call[1]} 的逐股历史在评分前已被释放")
        self.assertFalse(result.portfolio_result is None)

    def test_scan_stats_count_scored_and_failed(self):
        _, result, _ = self._run()
        self.assertEqual(result.scan_stats, {"scanned": 2, "scored": 2, "failed": 0})
        self.assertTrue(result.finished_at)

    def test_custom_pool_does_not_persist_shared_model(self):
        self._run()
        self.assertFalse(self.train_context_kwargs["persist"])

    def test_full_market_pool_persists_shared_model(self):
        self._run(FULL_MARKET_POOL_MODE)
        self.assertTrue(self.train_context_kwargs["persist"])


if __name__ == "__main__":
    unittest.main()
