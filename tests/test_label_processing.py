"""验证截面截尾、训练与评估隔离、三十日标签和摘要归档一致性。"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from config import settings
from services.feature_engineering import add_forward_returns
from services.label_processing import winsorize_training_labels
from services.ml_pipeline import fit_panel_model, save_context_summary
from services.model_governance import holdout_report, training_baseline
from services.model_monitor import candidate_validation_rows, evaluate_pass_rate_alerts, summarize_pass_rate
from services.stock_scoring import _make_model


class LabelWinsorizationTests(unittest.TestCase):
    """验证仅截同日上尾，不混入其他日期或改写真实标签。"""

    def test_upper_99_percentile_caps_120_percent_to_35_percent(self):
        labels = pd.Series([*np.linspace(-0.2, 0.34, 99), 0.35, 1.2])
        dates = pd.Series(pd.Timestamp("2024-01-02"), index=labels.index)
        original = labels.copy()
        result, report = winsorize_training_labels(labels, dates, upper_quantile=0.99)
        self.assertAlmostEqual(result.iloc[-1], 0.35)
        self.assertEqual(report["clipped_samples"], 1)
        self.assertEqual(result.iloc[0], -0.2)
        pd.testing.assert_series_equal(labels, original)

    def test_future_cross_section_cannot_change_previous_threshold(self):
        labels = pd.Series([0.0, 0.1, 0.2, 0.3, 1.2] * 2, index=range(10, 20))
        dates = pd.Series(pd.to_datetime(["2024-01-02"] * 5 + ["2024-01-03"] * 5), index=labels.index)
        first, _ = winsorize_training_labels(labels, dates, 0.975)
        labels.iloc[5:] = 999.0
        second, _ = winsorize_training_labels(labels, dates, 0.975)
        pd.testing.assert_series_equal(first.iloc[:5], second.iloc[:5])

    def test_small_or_missing_cross_section_is_not_fabricated(self):
        labels = pd.Series([-0.5, 0.0, 0.1, 1.2, np.nan])
        dates = pd.Series(pd.Timestamp("2024-01-02"), index=labels.index)
        result, report = winsorize_training_labels(labels, dates, 0.99)
        pd.testing.assert_series_equal(result, labels)
        self.assertEqual(report["skipped_dates"], 1)
        self.assertEqual(report["clipped_samples"], 0)

    def test_disabled_and_invalid_configuration(self):
        labels = pd.Series(np.arange(10, dtype=float))
        dates = pd.Series(pd.Timestamp("2024-01-02"), index=labels.index)
        result, report = winsorize_training_labels(labels, dates, 1.0)
        pd.testing.assert_series_equal(result, labels)
        self.assertFalse(report["enabled"])
        for quantile in (0.5, 1.01, float("nan")):
            with self.subTest(quantile=quantile), self.assertRaises(ValueError):
                winsorize_training_labels(labels, dates, quantile)
        with self.assertRaises(ValueError):
            winsorize_training_labels(labels, dates.reset_index(drop=True).set_axis(range(1, 11)))

    def test_thirty_day_labels_use_benchmark_calendar(self):
        dates = pd.bdate_range("2024-01-01", periods=50)
        stock = pd.DataFrame({"date": dates, "close": 100 * 1.01 ** np.arange(50)})
        benchmark = pd.DataFrame({"date": dates, "close": 100 * 1.002 ** np.arange(50)})
        frame = add_forward_returns(stock, (30,), benchmark)
        self.assertAlmostEqual(frame.loc[0, "target_30"], 1.01 ** 30 - 1.002 ** 30)
        self.assertEqual(frame.loc[0, "label_end_30"], dates[30])
        self.assertTrue(frame["target_30"].tail(30).isna().all())

    def test_training_clips_but_oos_actual_and_baseline_remain_raw(self):
        dates = pd.bdate_range("2020-01-01", periods=420)
        labels = np.array([-0.1, -0.08, -0.04, 0.0, 0.02, 0.04, 0.08, 0.1, 0.35, 1.2])
        panel = pd.DataFrame({
            "date": np.repeat(dates, 10), "code": np.tile(np.arange(10).astype(str), len(dates)),
            "probe": np.tile(np.arange(10), len(dates)), "target_30": np.tile(labels, len(dates)),
            "forward_return_30": np.tile(labels + 0.01, len(dates)),
            "label_end_30": np.repeat(pd.Series(dates).shift(-30).to_numpy(), 10),
        })
        panel.attrs["model_feature_columns"] = ["probe"]
        original = panel["target_30"].copy()
        fits, evaluations = [], []

        class Recorder:
            """记录估计器实际收到的标签，避免只验证预处理函数。"""

            def fit(self, features, target):
                fits.append((features.index.copy(), target.copy()))
                self.feature_importances_ = np.ones(features.shape[1])
                return self

            def predict(self, features):
                return features.iloc[:, 0].to_numpy(dtype=float) * 0.01

        def evaluate(frame, *args, **kwargs):
            evaluations.append(frame.copy())
            return {"validated": True, "rank_ic": 0.1}

        cutoff = dates[390]
        with patch("services.ml_pipeline.evaluate_oos", side_effect=evaluate):
            result = fit_panel_model(panel, "LightGBM", 30, lambda name: Recorder(), cutoff=cutoff, objective="regression")
        self.assertTrue(result["validated"])
        self.assertTrue(fits)
        for indices, target in fits:
            self.assertLess(float(target.max()), 1.2)
            self.assertTrue(panel.loc[indices, "label_end_30"].lt(cutoff).all())
        self.assertAlmostEqual(evaluations[0]["actual"].max(), 1.2)
        self.assertTrue(np.allclose(evaluations[0]["baseline"], np.median(labels)))
        self.assertGreater(result["label_processing"]["final"]["clipped_samples"], 0)
        pd.testing.assert_series_equal(panel["target_30"], original)

    def test_regression_losses_map_to_both_native_estimators(self):
        expected = {"l1": ("regression_l1", "reg:absoluteerror"),
                    "l2": ("regression", "reg:squarederror"),
                    "huber": ("huber", "reg:pseudohubererror")}
        for loss, objectives in expected.items():
            with self.subTest(loss=loss), patch.object(settings, "REGRESSION_LOSS", loss):
                for name, objective in zip(("LightGBM", "XGBoost"), objectives):
                    self.assertEqual(_make_model(name).get_params()["objective"], objective)

    def test_holdout_baseline_matches_valid_features_and_training_window(self):
        dates = pd.bdate_range("2024-01-01", periods=4)
        panel = pd.DataFrame({
            "date": np.repeat(dates[:3], 2), "target_30": [1.0, 2.0, 3.0, 4.0, 100.0, 200.0],
            "label_end_30": dates[3], "factor": [1.0, 1.0, 1.0, 1.0, np.nan, 1.0],
        })
        start = dates[3] + pd.Timedelta(days=1)
        value = training_baseline(panel, start, "target_30", "label_end_30", features=["factor"], window=1)
        self.assertEqual(value, 200.0)
        value = training_baseline(panel, start, "target_30", "label_end_30", features=["factor"], window=0)
        self.assertEqual(value, 3.0)


class ModelHealthIntegrityTests(unittest.TestCase):
    """验证候选统计与入选统计分离，并防止归档摘要再次被覆盖。"""

    def test_candidates_include_rejected_objectives_and_unavailable_is_separate(self):
        attempts = [{"objective": "lambdarank", "validated": True},
                    {"objective": "regression", "validated": False}]
        rows = [{"pass": True, "model_type": "lgb", "target_type": "rank",
                 "passed_gates": {"rank_ic": True}, "details": {"objective_mode": "auto", "context_id": "new", "objective_selection": attempts}},
                {"pass": False, "target_type": "unknown", "details": {"objective_mode": "auto", "context_id": "old"}}]
        summary = summarize_pass_rate(rows)
        self.assertEqual(summary["pass_rate"], 0.5)
        self.assertEqual(summary["availability_rate"], 0.5)
        self.assertEqual(summary["evaluated_pass_rate"], 1.0)
        self.assertEqual(summary["candidate_records"], 2)
        self.assertEqual(summary["candidate_pass_rate"], 0.5)
        self.assertEqual(summary["by_candidate_target_type"]["reg"]["passed"], 0)
        self.assertEqual(summary["latest_batch"]["pass_rate"], 1.0)
        self.assertEqual(summary["batch_count"], 2)
        self.assertEqual(len(candidate_validation_rows(rows)), 2)

    def test_small_mae_sample_does_not_borrow_rank_sample_size(self):
        rows = [{"pass": True, "passed_gates": {"ndcg": True}} for index in range(20)]
        rows.extend({"pass": False, "passed_gates": {"mae": False}} for index in range(4))
        alerts = evaluate_pass_rate_alerts(rows, [])
        self.assertFalse([alert for alert in alerts if alert.get("details", {}).get("gate") == "mae"])

    def test_holdout_report_only_exposes_current_model_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "holdout.json"
            with patch("services.model_governance._report_path", return_value=path):
                for version in ("2.1.0", None):
                    path.write_text(json.dumps({"model_version": version, "passed": True}), encoding="utf-8")
                    self.assertIsNone(holdout_report())
                payload = {"model_version": settings.MODEL_VERSION, "passed": True}
                path.write_text(json.dumps(payload), encoding="utf-8")
                self.assertEqual(holdout_report(), payload)

    def test_atomic_summary_update_preserves_hardlinked_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            active, archived = root / "active.json", root / "archived.json"
            first = {"context_id": "first", "models": {}}
            save_context_summary(first, active)
            os.link(active, archived)
            save_context_summary({"context_id": "second", "models": {}}, active)
            self.assertEqual(json.loads(archived.read_text(encoding="utf-8"))["context_id"], "first")
            self.assertEqual(json.loads(active.read_text(encoding="utf-8"))["context_id"], "second")
            self.assertFalse(os.path.samefile(active, archived))
            self.assertFalse(list(root.glob("*.tmp")))

    def test_invalid_summary_cannot_damage_existing_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "summary.json"
            save_context_summary({"context_id": "first", "models": {}}, path)
            with self.assertRaises(ValueError):
                save_context_summary({"context_id": "bad", "models": {}, "value": float("nan")}, path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["context_id"], "first")


if __name__ == "__main__":
    unittest.main()
