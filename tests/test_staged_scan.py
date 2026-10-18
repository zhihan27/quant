"""先采集后训练：采集冻结快照，训练不再访问行情网络，选项与截面必须一致。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from config import settings
from services import acquisition_store
from services.market_data import MarketDataError
from services.ml_pipeline import TARGET_KIND, load_context
from services.panel_store import PanelStore
from services.scan_pipeline import (
    FULL_MARKET_POOL_MODE,
    ScanOptions,
    collect_scan_snapshot,
    train_scan_snapshot,
)


def _history(end: str = "2026-10-15", bars: int = 400) -> pd.DataFrame:
    dates = pd.bdate_range(end=end, periods=bars)
    return pd.DataFrame({
        "date": [stamp.date().isoformat() for stamp in dates],
        "open": [10.0] * bars,
        "close": [10.0 + index * 0.01 for index in range(bars)],
        "high": [10.5] * bars,
        "low": [9.5] * bars,
        "volume": [1_000_000] * bars,
        "amount": [10_000_000] * bars,
    })


class _FakeService:
    def __init__(self, panels: PanelStore, reference: str = "2026-10-15") -> None:
        self.panels = panels
        self.reference = reference
        self.frame = _history(reference)
        self.history_calls: list[str] = []
        self.benchmark_error = False

    def get_scan_quote(self, candidate, source_meta):
        return {
            "name": candidate["name"], "price": 10.0, "volume_lots": 1000,
            "change_pct": 1.0, "quote_valid": True,
            "quote_time": f"{self.reference} 15:00:00", "data_status": "实时",
        }

    def get_history(self, code, count=1200, adjusted=True, market_id=None, expected_latest_trading_date=None):
        self.history_calls.append(code)
        if code == settings.MARKET_INDEX_CODE:
            if self.benchmark_error:
                raise MarketDataError("模拟基准不可用")
            return self.frame.copy(), {"status": "实时", "source": "测试"}
        self.panels.write_history(code, self.frame, "qfq" if adjusted else "raw", {"source": "测试"})
        return self.frame.copy(), {"status": "实时", "source": "测试"}


class _Governance:
    def __init__(self) -> None:
        self.after_scan_calls = []

    def scoring_paused(self) -> bool:
        return False

    def after_scan(self, context, panel):
        self.after_scan_calls.append(context["context_id"])
        return {}

    def notes(self, summary):
        return []


class StagedScanTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.panels = PanelStore(self.root / "panel")
        self.service = _FakeService(self.panels)
        self.patches = [
            patch.object(settings, "ACQUISITION_DIR", self.root / "acquisitions"),
            patch.object(settings, "MODEL_DIR", self.root / "models"),
            patch.object(settings, "DATA_DIR", self.root / "data"),
            patch.object(settings, "ACQUISITION_SNAPSHOT_LIMIT", 3),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.addCleanup(self.temporary.cleanup)
        self.candidates = [
            {"code": "600000", "name": "浦发银行"},
            {"code": "000001", "name": "平安银行"},
        ]

    def _options(self, **overrides) -> ScanOptions:
        values = {
            "pool_mode": FULL_MARKET_POOL_MODE, "custom_codes": "",
            "enable_neutralization": False, "use_extra_factors": False,
            "run_portfolio": False, "include_backtest": False,
        }
        values.update(overrides)
        return ScanOptions(**values)

    def _collect(self, options=None):
        with patch("services.scan_pipeline.collect_candidates", return_value=(self.candidates, {"status": "测试"})):
            return collect_scan_snapshot(self.service, options or self._options())

    def _context(self, reference: str | None = None) -> dict:
        return {
            "context_id": "a" * 32,
            "model_version": settings.MODEL_VERSION,
            "target_kind": TARGET_KIND,
            "reference_date": reference or self.service.reference,
            "objective_mode": "lambdarank",
            "training_window_days": 750,
            "neutralization": {"status": "未启用"},
            "extra_factors": {"families": {}, "status": "未启用"},
            "limitation": "测试上下文",
            "models": {}, "predictions": {}, "reference_closes": {},
        }

    def _train_patches(self, context=None, extra=None):
        context = context or self._context()
        captured: dict = {}

        def fake_train(histories, benchmark, **kwargs):
            captured["histories"] = histories
            captured["benchmark"] = benchmark
            captured["kwargs"] = kwargs
            panel = pd.DataFrame({"date": [self.service.reference], "code": ["marker"]})
            panel.attrs["marker"] = "memory-panel"
            panel.attrs["context_id"] = context["context_id"]
            if kwargs.get("panel_out") is not None:
                kwargs["panel_out"]["panel"] = panel
            return context

        def fake_analyze(history, **kwargs):
            return {
                "latest_features": {}, "models": {20: []}, "trade_plan": {},
                "prediction_reference_date": self.service.reference,
                "market_context_id": context["context_id"], "factor_score": 50.0,
            }

        patches = [
            patch("services.scan_pipeline.train_market_context", side_effect=fake_train),
            patch("services.scan_pipeline.analyze_stock", side_effect=fake_analyze),
            patch("services.scan_pipeline.rank_candidates", side_effect=lambda records: records),
        ]
        patches.extend(extra or [])
        self.addCleanup(lambda: [item.stop() for item in patches])
        started = [item.start() for item in patches]
        return captured, started[0]

    def test_collect_then_train_does_not_redownload_history(self):
        snapshot = self._collect()
        self.assertEqual(snapshot.status, "ready")
        self.assertEqual(self.service.history_calls.count(settings.MARKET_INDEX_CODE), 1)
        self.assertEqual(self.service.history_calls.count(settings.MARKET_SMALL_INDEX_CODE), 1)
        self.assertEqual(len(self.service.history_calls), 4)
        self.service.history_calls.clear()

        captured, _ = self._train_patches()
        result = train_scan_snapshot(self.service, self._options(), snapshot.snapshot_id)
        self.assertEqual(self.service.history_calls, [], "训练阶段不得重新访问行情")
        self.assertEqual(set(captured["histories"]), {"600000", "000001"})
        self.assertEqual(
            pd.to_datetime(captured["benchmark"]["date"]).max().strftime("%Y-%m-%d"),
            snapshot.reference_date,
        )
        self.assertFalse(captured["kwargs"]["persist"], "快照训练必须先只在内存训练")
        active, _ = load_context()
        self.assertEqual(active["context_id"], result.context_id, "校验成功后全市场模型才发布")
        self.assertEqual(result.scan_stats, {"scanned": 2, "scored": 2, "failed": 0})

    def test_empty_pool_and_benchmark_failure_leave_failed_snapshots(self):
        with patch("services.scan_pipeline.collect_candidates", return_value=([], {"status": "测试", "error": "股票池为空"})):
            failed = collect_scan_snapshot(self.service, self._options())
        self.assertEqual(failed.status, "failed")
        self.assertIn("股票池为空", failed.meta["error"])

        self.service.benchmark_error = True
        failed_benchmark = collect_scan_snapshot(self.service, self._options())
        self.assertEqual(failed_benchmark.status, "failed")
        self.assertIn("沪深 300", failed_benchmark.meta["error"])

    def test_weekend_collection_uses_benchmark_date_without_quote_source_time(self):
        """股票池没有逐股源时间时，周六下载不能要求存在周六日线。"""
        self.service.reference = "2026-10-16"
        self.service.frame = _history(self.service.reference)
        quote = {
            "name": "普通公司", "price": 10.0, "volume_lots": 1000,
            "change_pct": 1.0, "quote_valid": True, "quote_time": None,
            "fetched_at": "2026-10-17T21:01:08+08:00",
        }
        with patch.object(self.service, "get_scan_quote", return_value=quote):
            snapshot = self._collect()
        self.assertEqual(snapshot.status, "ready")
        self.assertEqual(snapshot.meta["counts"]["success"], 2)
        loaded = acquisition_store.load_snapshot(snapshot.snapshot_id, self.panels)
        self.assertEqual(set(loaded.histories), {"600000", "000001"})

    def test_training_rejects_changed_collection_options(self):
        snapshot = self._collect(self._options(pool_mode="用户自定义", custom_codes="600000"))
        captured, _ = self._train_patches()
        with self.assertRaises(ValueError):
            train_scan_snapshot(
                self.service, self._options(pool_mode="用户自定义", custom_codes="000001"),
                snapshot.snapshot_id,
            )
        self.assertNotIn("histories", captured, "选项不一致时必须在训练前拒绝")

    def test_training_rejects_reference_date_mismatch(self):
        snapshot = self._collect()
        governance = _Governance()
        captured, _ = self._train_patches(context=self._context("2026-10-14"))
        with self.assertRaises(ValueError):
            train_scan_snapshot(self.service, self._options(), snapshot.snapshot_id, governance=governance)
        self.assertFalse(captured["kwargs"]["persist"])
        self.assertFalse((self.root / "models").exists(), "错配模型不能留下活动文件")
        self.assertFalse((self.root / "data").exists(), "错配面板不能落盘")
        self.assertEqual(governance.after_scan_calls, [])

    def test_in_memory_panel_is_used_for_portfolio_instead_of_active_panel(self):
        custom = self._options(pool_mode="用户自定义", custom_codes="600000,000001", run_portfolio=True)
        snapshot = self._collect(self._options(pool_mode="用户自定义", custom_codes="600000,000001"))
        captured: dict = {}

        def fake_portfolio(histories, panel, **kwargs):
            captured["panel"] = panel
            return {"metrics": {}}

        self._train_patches(extra=[
            patch("services.scan_pipeline.run_portfolio_strategy", side_effect=fake_portfolio),
            patch("services.scan_pipeline.load_context", side_effect=AssertionError("不应回读活动面板")),
        ])
        result = train_scan_snapshot(self.service, custom, snapshot.snapshot_id)
        # 自定义池训练也要用本次面板，而不是活动模型里的旧面板。
        self.assertEqual(captured["panel"].attrs.get("marker"), "memory-panel")
        self.assertIsNotNone(result.portfolio_result)

    def test_governance_only_runs_for_persisted_full_market_context(self):
        custom = self._options(pool_mode="用户自定义", custom_codes="600000,000001")
        custom_snapshot = self._collect(custom)
        full_market = self._options()
        full_snapshot = self._collect(full_market)
        governance = _Governance()
        captured, _ = self._train_patches(extra=[
            patch("services.scan_pipeline.load_context", return_value=(self._context(), None)),
        ])
        train_scan_snapshot(self.service, custom, custom_snapshot.snapshot_id, governance=governance)
        self.assertFalse(captured["kwargs"]["persist"])
        self.assertEqual(governance.after_scan_calls, [], "自定义池不得归档活动模型")

        train_scan_snapshot(self.service, full_market, full_snapshot.snapshot_id, governance=governance)
        self.assertFalse(captured["kwargs"]["persist"])
        self.assertTrue((self.root / "models" / f"market_context_v{settings.MODEL_VERSION}.joblib").exists())
        self.assertEqual(governance.after_scan_calls, ["a" * 32])

    def test_missing_panel_store_file_is_reported_when_training(self):
        snapshot = self._collect()
        (self.panels.history_path("600000", "qfq")).unlink()
        self._train_patches()
        result = train_scan_snapshot(self.service, self._options(), snapshot.snapshot_id)
        self.assertEqual(result.scan_stats["scored"], 1)
        self.assertEqual(result.scan_stats, {"scanned": 2, "scored": 1, "failed": 1})
        self.assertEqual(result.failed_rows[0]["代码"], "600000")
        self.assertTrue(
            any("1 只股票历史不可用" in note["text"] for note in result.notes),
            result.notes,
        )

    def test_disabled_extra_factors_are_not_passed_to_training(self):
        snapshot = self._collect()
        captured, _ = self._train_patches()
        train_scan_snapshot(self.service, self._options(), snapshot.snapshot_id)
        self.assertIsNone(captured["kwargs"]["extra_factors"])

    def test_backtest_uses_frozen_raw_history_without_network(self):
        options = self._options(include_backtest=True)
        snapshot = self._collect(options)
        self._train_patches()
        captured = []

        def backtest(raw, **kwargs):
            captured.append(raw)
            return {"status": "测试回测", "metrics": {}}

        # 即使共享不复权历史之后被覆盖，训练仍应使用快照内的原始日线。
        for candidate in self.candidates:
            self.panels.write_history(candidate["code"], _history("2026-10-16"), "raw")
        with patch.object(self.service, "get_history", side_effect=AssertionError("训练阶段禁止联网")), patch(
            "services.scan_pipeline.run_backtest", side_effect=backtest,
        ):
            result = train_scan_snapshot(self.service, options, snapshot.snapshot_id)
        self.assertEqual(len(captured), 2)
        self.assertTrue(all(pd.to_datetime(frame["date"]).max().strftime("%Y-%m-%d") == snapshot.reference_date for frame in captured))
        self.assertEqual(result.scan_stats["scored"], 2)

    def test_backtest_requires_data_prepared_during_collection(self):
        snapshot = self._collect()
        captured, _ = self._train_patches()
        with self.assertRaisesRegex(ValueError, "回测日线"):
            train_scan_snapshot(self.service, self._options(include_backtest=True), snapshot.snapshot_id)
        self.assertNotIn("histories", captured, "缺少回测数据时应在训练前拒绝")

    def test_missing_raw_file_is_reported_without_network_fallback(self):
        options = self._options(include_backtest=True)
        snapshot = self._collect(options)
        (snapshot.directory / "raw" / "code=600000.parquet").unlink()
        self._train_patches()
        with patch.object(self.service, "get_history", side_effect=AssertionError("训练阶段禁止联网")), patch(
            "services.scan_pipeline.run_backtest", return_value={"status": "测试", "metrics": {}},
        ) as backtest:
            result = train_scan_snapshot(self.service, options, snapshot.snapshot_id)
        self.assertEqual(backtest.call_count, 1)
        failed = next(row for row in result.detail_rows if row["代码"] == "600000")
        self.assertIn("回测不可用", failed["回测状态"])


if __name__ == "__main__":
    unittest.main()
