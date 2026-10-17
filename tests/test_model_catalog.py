"""已保存模型目录与按 id 加载：版本、文件完整性、路径穿越与旧模型回看。"""

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd

from config import settings
from services import model_catalog
from services.database import DatabaseError
from services.ml_pipeline import HORIZONS, historical_models_for_stock, load_context_by_id, models_for_stock


class ModelCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model_dir = self.root / "models"
        self.model_dir.mkdir()
        self.patch = patch.object(settings, "MODEL_DIR", self.model_dir)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.addCleanup(self.temporary.cleanup)
        self.version = settings.MODEL_VERSION

    def _archive(self, context_id: str, created_at: str = "2026-10-15T15:00:00+08:00", **overrides):
        directory = self.model_dir / settings.MODEL_ARCHIVE_DIRNAME / context_id
        directory.mkdir(parents=True)
        registry = {
            "context_id": context_id, "model_version": self.version, "created_at": created_at,
            "reference_date": "2026-10-15", "passed": True,
            "objective_mode": "lambdarank", "training_window_days": 750,
            "pool_mode": "沪深京全部 A 股", "universe_size": 100,
            "source_snapshot_id": "snapshot0001",
        }
        registry.update(overrides)
        (directory / "registry.json").write_text(json.dumps(registry, ensure_ascii=False), encoding="utf-8")
        for name in ("market_context_v{}.joblib", "market_context_v{}.json", "ml_panel_v{}.parquet"):
            (directory / name.format(self.version)).write_bytes(b"")
        return directory

    def _active(self, context_id: str, archived: bool = False):
        summary = {
            "context_id": context_id, "model_version": self.version,
            "created_at": "2026-10-16T15:00:00+08:00", "reference_date": "2026-10-16",
            "objective_mode": "lambdarank", "training_window_days": 750,
            "source": {"pool_mode": "沪深京全部 A 股", "snapshot_id": "snapshot0002"},
            "universe_codes": ["600000", "000001"],
        }
        path = self.model_dir / f"market_context_v{self.version}.json"
        path.write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
        if not archived:
            (self.model_dir / f"market_context_v{self.version}.joblib").write_bytes(b"")
        return summary

    def test_catalog_filters_versions_and_missing_files_and_marks_active(self):
        self._archive("a" * 32, "2026-10-14T15:00:00+08:00")
        self._archive("b" * 32, "2026-10-15T15:00:00+08:00")
        self._archive("c" * 32, model_version="0.0.1")
        broken = self._archive("d" * 32)
        (broken / f"ml_panel_v{self.version}.parquet").unlink()
        self._active("b" * 32, archived=True)

        models = model_catalog.saved_models(None)
        identifiers = [item["context_id"] for item in models]
        self.assertEqual(identifiers, ["b" * 32, "a" * 32], "旧版本和缺文件的归档必须过滤")
        self.assertTrue(models[0]["is_active"])
        self.assertFalse(models[1]["is_active"])
        self.assertEqual(models[0]["source"], "archive")

    def test_active_model_without_archive_is_appended_once(self):
        self._archive("a" * 32)
        self._active("e" * 32, archived=False)
        models = model_catalog.saved_models(None)
        self.assertEqual([item["context_id"] for item in models], ["e" * 32, "a" * 32])
        self.assertEqual(models[0]["source"], "active")
        self.assertTrue(models[0]["is_active"])

    def test_database_failure_falls_back_to_archive_directory(self):
        self._archive("a" * 32)
        with patch("services.model_governance.registry_versions", side_effect=DatabaseError("库不可用")):
            models = model_catalog.saved_models(repository=object())
        self.assertEqual([item["context_id"] for item in models], ["a" * 32])

    def test_database_rows_are_deduped_against_archive(self):
        self._archive("a" * 32)
        rows = [
            {"context_id": "a" * 32, "model_version": self.version},
            {"context_id": "b" * 32, "model_version": self.version},  # 归档不存在，必须丢弃
            {"context_id": "c" * 32, "model_version": "0.0.1"},
        ]
        with patch("services.model_governance.registry_versions", return_value=rows):
            models = model_catalog.saved_models(repository=object())
        self.assertEqual([item["context_id"] for item in models], ["a" * 32])


class ContextByIdTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model_dir = self.root / "models"
        self.model_dir.mkdir()
        self.patch = patch.object(settings, "MODEL_DIR", self.model_dir)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.addCleanup(self.temporary.cleanup)
        self.version = settings.MODEL_VERSION

    def _write(self, context_id: str, context_id_in_model: str | None = None, with_panel: bool = False):
        directory = self.model_dir / settings.MODEL_ARCHIVE_DIRNAME / context_id
        directory.mkdir(parents=True)
        context = {
            "context_id": context_id_in_model or context_id,
            "model_version": self.version,
            "target_kind": "excess_return_vs_csi300",
            "reference_date": "2026-10-15",
            "models": {}, "predictions": {}, "reference_closes": {},
        }
        joblib.dump(context, directory / f"market_context_v{self.version}.joblib")
        (directory / "registry.json").write_text(json.dumps({
            "context_id": context_id, "model_version": self.version,
        }), encoding="utf-8")
        if with_panel:
            panel = pd.DataFrame({"date": ["2026-10-15"], "code": ["600000"]})
            panel.attrs["context_id"] = context_id
            panel.to_parquet(directory / f"ml_panel_v{self.version}.parquet", index=False)
        return context

    def test_loads_archived_context_and_panel_by_id(self):
        context = self._write("a" * 32, with_panel=True)
        loaded, panel = load_context_by_id("a" * 32, include_panel=True)
        self.assertEqual(loaded["context_id"], context["context_id"])
        self.assertIsNotNone(panel)
        self.assertEqual(panel.attrs.get("context_id"), "a" * 32)

    def test_rejects_invalid_ids_mismatched_content_and_missing_archives(self):
        self._write("b" * 32, context_id_in_model="c" * 32)
        for bad in ("../../etc/passwd", "not-a-context-id", "f" * 32):
            self.assertEqual(load_context_by_id(bad), (None, None))
        self.assertEqual(load_context_by_id("b" * 32), (None, None), "模型内 context_id 不一致必须拒绝")


class HistoricalModelsTests(unittest.TestCase):
    def _context(self):
        predictions = {
            horizon: {"600000": {"LightGBM": {
                "predicted_excess_return": 0.02, "prediction_zscore": 1.5,
                "prediction_percentile": 0.9, "model_rank": 5, "score_kind": "rank",
            }}}
            for horizon in HORIZONS
        }
        return {
            "context_id": "a" * 32, "reference_date": "2026-10-15",
            "reference_closes": {"600000": 10.0},
            "models": {
                horizon: [{"model": "LightGBM", "validated": True, "status": "通过"}]
                for horizon in HORIZONS
            },
            "predictions": predictions,
        }

    def test_historical_rows_use_the_models_own_reference_date(self):
        rows = historical_models_for_stock(self._context(), "600000")
        self.assertEqual([row["horizon"] for row in rows], list(HORIZONS))
        self.assertTrue(all(row["predicted_excess_return"] == 0.02 for row in rows))
        self.assertTrue(all(row["reference_date"] == "2026-10-15" for row in rows))
        self.assertEqual(historical_models_for_stock(self._context(), "000001"), [])

    def test_old_model_never_scores_a_newer_date(self):
        context = self._context()
        newer = models_for_stock(context, "600000", "2026-10-16", 10.0, 20)
        self.assertTrue(all(not item["validated"] for item in newer))
        same_day = models_for_stock(context, "600000", "2026-10-15", 10.0, 20)
        self.assertTrue(any(item["validated"] for item in same_day))


if __name__ == "__main__":
    unittest.main()
