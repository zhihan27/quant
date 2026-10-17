"""采集快照：可续跑、原子落盘、按截面日期回读历史并冻结训练输入。"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from config import settings
from services import acquisition_store
from services.panel_store import PanelStore


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


def _options(**overrides):
    values = {
        "pool_mode": "用户自定义", "custom_codes": "600000,000001", "scan_limit": 10,
        "enable_neutralization": False, "use_extra_factors": False,
        "objective_mode": "lambdarank", "training_window": 750,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _record(code: str, history: pd.DataFrame | None = None, error: str | None = None) -> dict:
    if error:
        return {"code": code, "name": code, "error": error}
    return {
        "code": code, "name": f"样本{code}", "quote": {"price": 10.0},
        "history_meta": {"source": "测试"}, "entry_trade_status": "可正常评估",
        "history": history,
    }


class AcquisitionStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.panel_store = PanelStore(self.root / "panel")
        self.patches = [
            patch.object(settings, "ACQUISITION_DIR", self.root / "acquisitions"),
            patch.object(settings, "MODEL_DIR", self.root / "models"),
            patch.object(settings, "ACQUISITION_SNAPSHOT_LIMIT", 3),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.addCleanup(self.temporary.cleanup)
        self.reference = "2026-10-15"
        self.benchmark = _history(self.reference)

    def _store_history(self, code: str, frame: pd.DataFrame) -> None:
        self.panel_store.write_history(code, frame, "qfq", {"source": "测试"})

    def _new_run(self, options=None, reference=None, candidates=None):
        options = options or _options()
        reference = reference or self.reference
        candidates = candidates or [{"code": "600000", "name": "样本600000"}]
        run = acquisition_store.create_or_resume(options, {"status": "测试"}, reference, candidates)
        return run, candidates

    def test_round_trip_freezes_benchmark_and_reloads_cleaned_history(self):
        run, candidates = self._new_run()
        history = _history()
        self._store_history("600000", history)
        record = _record("600000", history)
        acquisition_store.checkpoint(run, record)
        snapshot = acquisition_store.finalize(
            run, [record], self.benchmark, {"status": "实时"}, store=self.panel_store
        )
        self.assertEqual(snapshot.status, "ready")
        self.assertEqual(acquisition_store.latest_snapshot(ready_only=True).snapshot_id, snapshot.snapshot_id)
        self.assertEqual(acquisition_store.summary(snapshot.snapshot_id)["counts"]["success"], 1)

        loaded = acquisition_store.load_snapshot(snapshot.snapshot_id, store=self.panel_store)
        self.assertEqual(loaded.reference_date, self.reference)
        self.assertEqual(list(loaded.histories), ["600000"])
        self.assertEqual(list(loaded.benchmark["date"])[-1], self.reference)
        for column in acquisition_store.STORAGE_COLUMNS:
            self.assertNotIn(column, loaded.histories["600000"].columns)
        self.assertEqual(loaded.extra_factors, {}, "关闭扩展因子时不得重新装配筹码")

    def test_extra_factors_are_restored_only_when_enabled(self):
        run, _ = self._new_run(options=_options(use_extra_factors=True))
        history = _history()
        self._store_history("600000", history)
        snapshot = acquisition_store.finalize(
            run, [_record("600000", history)], self.benchmark, {}, store=self.panel_store,
        )
        loaded = acquisition_store.load_snapshot(snapshot.snapshot_id, self.panel_store)
        self.assertIn("chip", loaded.extra_factors)
        self.assertEqual(loaded.extra_factors["chip"]["600000"].shape, loaded.histories["600000"].shape)

    def test_resumed_snapshot_excludes_removed_candidates(self):
        candidates = [{"code": "600000"}, {"code": "000001"}]
        run, _ = self._new_run(candidates=candidates)
        history = _history()
        for candidate in candidates:
            code = candidate["code"]
            self._store_history(code, history)
            acquisition_store.checkpoint(run, _record(code, history))
        resumed = acquisition_store.create_or_resume(_options(), {"status": "新名单"}, self.reference, candidates[:1])
        prior, _, _ = acquisition_store.resume_records(resumed, candidates[:1], self.reference, self.panel_store)
        snapshot = acquisition_store.finalize(resumed, prior, self.benchmark, {}, store=self.panel_store)
        loaded = acquisition_store.load_snapshot(snapshot.snapshot_id, self.panel_store)
        self.assertEqual(list(loaded.histories), ["600000"])
        self.assertEqual(snapshot.meta["counts"]["total"], 1)
        self.assertEqual(acquisition_store.summary(snapshot.snapshot_id)["recorded_success"], 1)
        self.assertEqual(snapshot.meta["source_meta"]["status"], "新名单")

    def test_legacy_weekly_file_is_rejected_in_training_and_resume(self):
        run, candidates = self._new_run()
        history = _history()
        self._store_history("600000", history)
        acquisition_store.checkpoint(run, _record("600000", history))
        weekly = history.copy()
        weekly["date"] = pd.date_range(end=self.reference, periods=400, freq="W-THU")
        # 直接写入旧版路径，模拟没有周期元数据的历史污染。
        weekly.to_parquet(self.panel_store.history_path("600000"), index=False)
        resumed = acquisition_store.create_or_resume(_options(), {}, self.reference, candidates)
        _, done, skipped = acquisition_store.resume_records(resumed, candidates, self.reference, self.panel_store)
        self.assertEqual(done, set())
        self.assertEqual(skipped, 0)
        snapshot = acquisition_store.finalize(resumed, list(resumed.prior.values()), self.benchmark, {}, store=self.panel_store)
        self.assertEqual(snapshot.status, "failed")
        self.assertIn("日线口径", acquisition_store._read_items(run.directory)["600000"]["error"])

    def test_items_jsonl_skips_corrupt_lines_and_last_record_wins(self):
        run, _ = self._new_run()
        acquisition_store.checkpoint(run, _record("600000", error="第一次失败"))
        acquisition_store.checkpoint(run, _record("600000", error="第二次失败"))
        with (run.directory / "items.jsonl").open("a", encoding="utf-8") as handle:
            handle.write('{"code": "000001", "error": "半行')
        items = acquisition_store._read_items(run.directory)
        self.assertEqual(items["600000"]["error"], "第二次失败")
        self.assertNotIn("000001", items)

    def test_history_is_truncated_to_reference_and_short_history_becomes_failure(self):
        run, candidates = self._new_run(candidates=[
            {"code": "600000", "name": "样本600000"}, {"code": "000001", "name": "样本000001"},
        ])
        long_history = _history(end="2026-10-20")
        short_history = _history(bars=100)
        self._store_history("600000", long_history)
        self._store_history("000001", short_history)
        records = [_record("600000", long_history), _record("000001", short_history)]
        snapshot = acquisition_store.finalize(
            run, records, self.benchmark, {"status": "实时"}, store=self.panel_store
        )
        self.assertEqual(snapshot.status, "ready")
        self.assertEqual(snapshot.meta["counts"]["success"], 1)
        self.assertEqual(snapshot.meta["counts"]["failed"], 1)
        loaded = acquisition_store.load_snapshot(snapshot.snapshot_id, store=self.panel_store)
        self.assertEqual(
            pd.to_datetime(loaded.histories["600000"]["date"]).max().strftime("%Y-%m-%d"),
            self.reference,
        )
        self.assertEqual(len(loaded.failures), 1)
        self.assertIn("样本不足", loaded.failures[0]["error"])

    def test_resume_skips_completed_codes_and_retries_failures(self):
        candidates = [
            {"code": "600000", "name": "样本600000"},
            {"code": "000001", "name": "样本000001"},
        ]
        run, _ = self._new_run(candidates=candidates)
        history = _history()
        self._store_history("600000", history)
        acquisition_store.checkpoint(run, _record("600000", history))
        acquisition_store.checkpoint(run, _record("000001", error="超时"))

        resumed = acquisition_store.create_or_resume(
            _options(), {"status": "测试"}, self.reference, candidates
        )
        self.assertTrue(resumed.resumed)
        self.assertEqual(resumed.snapshot_id, run.snapshot_id)
        ordered, done, skipped = acquisition_store.resume_records(
            resumed, candidates, self.reference, self.panel_store
        )
        self.assertEqual([record["code"] for record in ordered], ["600000"])
        self.assertEqual(done, {"600000"})
        self.assertEqual(skipped, 1)
        self.assertNotIn("000001", done, "失败记录必须重新采集")

    def test_reference_change_abandons_unfinished_snapshot(self):
        run, candidates = self._new_run()
        newer = "2026-10-16"
        new_run = acquisition_store.create_or_resume(
            _options(), {"status": "测试"}, newer, candidates
        )
        self.assertFalse(new_run.resumed)
        self.assertNotEqual(new_run.snapshot_id, run.snapshot_id)
        old_meta = json.loads((run.directory / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(old_meta["status"], "abandoned")

    def test_model_version_mismatch_is_rejected(self):
        run, _ = self._new_run()
        history = _history()
        self._store_history("600000", history)
        snapshot = acquisition_store.finalize(
            run, [_record("600000", history)], self.benchmark, {"status": "实时"}, store=self.panel_store
        )
        meta_path = run.directory / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["model_version"] = "0.0.1"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        with self.assertRaises(ValueError):
            acquisition_store.load_snapshot(snapshot.snapshot_id, store=self.panel_store)
        with self.assertRaises(ValueError):
            acquisition_store.load_snapshot("../outside", store=self.panel_store)

    def test_prune_keeps_latest_and_archived_references(self):
        def ready_snapshot(snapshot_id: str, created_at: str) -> Path:
            directory = self.root / "acquisitions" / snapshot_id
            directory.mkdir(parents=True)
            (directory / "meta.json").write_text(json.dumps({
                "snapshot_id": snapshot_id, "status": "ready", "created_at": created_at,
                "reference_date": self.reference, "options": {},
            }), encoding="utf-8")
            return directory

        old = ready_snapshot("snapshot0001", "2026-10-10T15:00:00+08:00")
        referenced = ready_snapshot("snapshot0002", "2026-10-11T15:00:00+08:00")
        latest = ready_snapshot("snapshot0003", "2026-10-12T15:00:00+08:00")
        failed_old = self.root / "acquisitions" / "failed0001"
        failed_old.mkdir(parents=True)
        (failed_old / "meta.json").write_text(json.dumps({
            "snapshot_id": failed_old.name, "status": "failed",
            "created_at": "2026-10-01T15:00:00+08:00", "options": {},
        }), encoding="utf-8")
        failed_new = self.root / "acquisitions" / "failed0002"
        failed_new.mkdir(parents=True)
        (failed_new / "meta.json").write_text(json.dumps({
            "snapshot_id": failed_new.name, "status": "failed",
            "created_at": "2026-10-02T15:00:00+08:00", "options": {},
        }), encoding="utf-8")
        archive = self.root / "models" / settings.MODEL_ARCHIVE_DIRNAME / "ctx"
        archive.mkdir(parents=True)
        (archive / "registry.json").write_text(json.dumps({
            "context_id": "ctx", "source_snapshot_id": "snapshot0002",
        }), encoding="utf-8")

        acquisition_store._prune_snapshots(keep=1, current=latest.name)
        self.assertFalse(old.exists(), "超过保留数量的 ready 快照应清理")
        self.assertTrue(referenced.exists(), "被归档模型引用的快照不能删")
        self.assertTrue(latest.exists())
        self.assertFalse(failed_old.exists(), "更早的失败目录应清理")


class StreamlitFreeTests(unittest.TestCase):
    def test_importing_acquisition_store_does_not_load_streamlit(self):
        import subprocess
        import sys

        code = "import sys; import services.acquisition_store; sys.exit(1 if 'streamlit' in sys.modules else 0)"
        completed = subprocess.run(
            [sys.executable, "-c", code], cwd=settings.PROJECT_ROOT,
            capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
