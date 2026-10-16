"""验证无界面扫描编排：不依赖 Streamlit、输入校验、以及组合回测与逐股历史的先后次序。"""

import subprocess
import sys
import unittest
from unittest.mock import patch

import pandas as pd

from config import settings
from services.scan_pipeline import POOL_MODES, ScanOptions, run_scan_pipeline


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

    def get_history(self, code, count=1200, adjusted=True, market_id=None):
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


class PipelineOrderTests(unittest.TestCase):
    """组合回测与逐股评分都依赖扫描期间收集的逐股历史，释放必须排在两者之后。

    原页面靠代码恰好写对才成立；搬进服务层后用这条用例把次序钉住。
    """

    def _run(self):
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
            pool_mode="用户自定义", custom_codes="600000,000001",
            enable_neutralization=False, use_extra_factors=False, run_portfolio=True,
        )
        candidates = [{"code": "600000", "name": "浦发银行"}, {"code": "000001", "name": "平安银行"}]
        with patch("services.scan_pipeline.train_market_context", return_value=context), patch(
            "services.scan_pipeline.load_context", return_value=(context, panel)
        ), patch("services.scan_pipeline.run_portfolio_strategy", side_effect=fake_portfolio), patch(
            "services.scan_pipeline.analyze_stock", side_effect=fake_analyze
        ), patch("services.scan_pipeline.rank_candidates", return_value=[]):
            result = run_scan_pipeline(service, options, candidates, {"status": "测试"}, None)
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


if __name__ == "__main__":
    unittest.main()
