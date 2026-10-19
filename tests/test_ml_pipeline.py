"""用确定性测试样本验证超额标签、日期截面及未来数据隔离。"""

import ctypes
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from config import settings
from services.feature_engineering import FEATURE_COLUMNS, add_forward_returns, normalize_cross_section
from services.ml_pipeline import (
    HORIZONS, MODEL_NAMES, TARGET_KIND, _fold_row_positions, _memory_metrics, _prepare_validation_data,
    _select_fold_features, _validation_cache_key,
    evaluate_oos, fit_panel_model,
    load_context, models_for_stock, normalize_predictions, predict_panel, save_context, train_market_context, unavailable_model,
)
from services.stock_scoring import analyze_stock, walk_forward_predictions


def _panel(days=420, stocks=10):
    dates = pd.bdate_range("2020-01-01", periods=days)
    rows = []
    for day, date in enumerate(dates):
        for stock in range(stocks):
            value = (stock - (stocks - 1) / 2) / 3
            row = {"date": date, "code": str(stock).zfill(6), "close": 10.0}
            row.update({"xs_" + column: value for column in FEATURE_COLUMNS})
            for horizon in settings.MODEL_HORIZONS:
                mature = day + horizon < days
                outcome = value * 0.02 + 0.01 * np.sin(day * 0.7 + stock * 1.4)
                row[f"target_{horizon}"] = outcome if mature else np.nan
                row[f"forward_return_{horizon}"] = outcome + 0.01 if mature else np.nan
                row[f"label_end_{horizon}"] = dates[day + horizon] if mature else pd.NaT
            rows.append(row)
    return pd.DataFrame(rows)


class _ExactModel:
    """可记录训练索引的确定性估计器，仅供验证日期隔离。"""

    def fit(self, x, y):
        self.fit_indices = x.index.copy()
        self.feature_importances_ = np.ones(x.shape[1])
        return self

    def predict(self, x):
        return x.iloc[:, 0].to_numpy() * 0.02


class LabelAndNormalizationTests(unittest.TestCase):
    """验证市场上涨时落后股票仍产生负超额标签。"""

    def test_relative_strength_is_not_a_duplicate_model_input(self):
        self.assertNotIn("relative_strength_20", FEATURE_COLUMNS)

    def test_empty_vetoed_panel_does_not_train_or_persist(self):
        panel = pd.DataFrame(columns=["date", "code", "close"])
        panel.attrs.update({
            "health_veto": {"status": "财务健康否决已排除全部候选"},
            "model_feature_columns": [],
        })
        panel_output = {}
        with patch("services.ml_pipeline.build_market_panel", return_value=panel), \
                patch("services.ml_pipeline.save_context") as save_context_mock:
            context = train_market_context({}, pd.DataFrame(), persist=True, panel_out=panel_output)
        self.assertIs(panel_output["panel"], panel)
        self.assertIn("未训练或发布模型", context["status"])
        self.assertEqual(set(context["models"]), set(HORIZONS))
        self.assertTrue(all(len(context["models"][horizon]) == len(MODEL_NAMES) for horizon in HORIZONS))
        save_context_mock.assert_not_called()

    def test_labels_are_benchmark_excess_and_last_rows_are_unknown(self):
        dates = pd.bdate_range("2024-01-01", periods=20)
        stock = pd.DataFrame({
            "date": dates, "open": 100 * 1.005 ** np.arange(20), "close": 100 * 1.01 ** np.arange(20),
        })
        benchmark = pd.DataFrame({
            "date": dates, "open": 100 * 1.002 ** np.arange(20), "close": 100 * 1.02 ** np.arange(20),
        })
        result = add_forward_returns(stock, (5,), benchmark)
        self.assertAlmostEqual(result.loc[0, "target_5"], 1.005**5 - 1.002**5)
        self.assertGreater(result.loc[0, "target_5"], 0)
        self.assertTrue(result["target_5"].tail(6).isna().all())
        self.assertEqual(result.loc[0, "label_end_5"], dates[6])
        self.assertAlmostEqual(result.loc[0, "research_target_5"], 1.01**5 - 1.02**5)
        self.assertEqual(result.loc[0, "research_label_end_5"], dates[5])
        self.assertTrue(add_forward_returns(stock, (5,))["target_5"].isna().all())

    def test_missing_intermediate_open_refuses_execution_label(self):
        dates = pd.bdate_range("2024-01-01", periods=15)
        benchmark = pd.DataFrame({"date": dates, "open": 100.0, "close": 100.0})
        stock = pd.DataFrame({"date": dates, "open": np.arange(15) + 100.0,
                              "close": np.arange(15) + 100.0}).drop(index=5)
        result = add_forward_returns(stock, (5,), benchmark)
        self.assertTrue(pd.isna(result.loc[0, "target_5"]))
        self.assertTrue(pd.isna(result.loc[0, "label_end_5"]))

    def test_feature_zscore_is_computed_per_date(self):
        dates = pd.bdate_range("2024-01-01", periods=2)
        panel = pd.DataFrame({"date": np.repeat(dates, 5), "code": list("abcde") * 2,
                              "factor": [1, 2, 3, 4, 5, 100, 200, 300, 400, 500]})
        result = normalize_cross_section(panel, ["factor"])
        self.assertTrue(np.allclose(result.groupby("date")["xs_factor"].mean(), 0))
        self.assertTrue(np.allclose(result.groupby("date")["xs_factor"].std(ddof=0), 1))
        panel.loc[panel["date"].eq(dates[1]), "factor"] *= -100
        altered = normalize_cross_section(panel, ["factor"])
        self.assertTrue(np.allclose(result["xs_factor"].head(5), altered["xs_factor"].head(5)))

    def test_positive_raw_forecasts_have_positive_and_negative_relative_scores(self):
        frame = pd.DataFrame({"date": ["2024-01-01"] * 5, "code": list("abcde"), "prediction": [0.1, 0.2, 0.3, 0.4, 0.5]})
        result = normalize_predictions(frame)
        self.assertAlmostEqual(result["prediction_zscore"].mean(), 0)
        self.assertAlmostEqual(result["prediction_zscore"].std(ddof=0), 1)
        self.assertLess(result["prediction_zscore"].min(), 0)
        self.assertGreater(result["prediction_zscore"].max(), 0)
        frame["prediction"] = 0.5
        self.assertTrue(normalize_predictions(frame)["prediction_zscore"].isna().all())

    def test_small_or_duplicate_cross_section_is_rejected(self):
        frame = pd.DataFrame({"date": ["2024-01-01"] * 4, "code": list("abcd"), "factor": [1, 2, 3, 4]})
        self.assertTrue(normalize_cross_section(frame, ["factor"])["xs_factor"].isna().all())
        with self.assertRaises(ValueError):
            normalize_cross_section(pd.concat([frame, frame.iloc[[0]]]), ["factor"])


class PanelValidationTests(unittest.TestCase):
    """检验Rank IC、基线、分组收益和真实模型训练。"""

    def evaluation_frame(self):
        panel = _panel(100)
        frame = panel[["date", "code"]].copy()
        frame["actual"] = panel["xs_momentum_5"] * 0.02 + 0.01 * np.sin((panel.index // 10) * 0.7 + (panel.index % 10) * 1.4)
        frame["prediction"] = panel["xs_momentum_5"] * 0.02
        frame["baseline"] = 0.0
        frame["absolute_return"] = frame["actual"] + 0.01
        return frame

    def test_later_feature_is_selected_only_after_training_coverage_is_sufficient(self):
        dates = pd.bdate_range("2024-01-01", periods=45)
        frame = pd.DataFrame({
            "date": np.repeat(dates, 10),
            "code": np.tile([str(stock).zfill(6) for stock in range(10)], len(dates)),
            "xs_base": np.tile(np.arange(10, dtype=float), len(dates)),
            "xs_late_chip": np.nan,
        })
        frame.loc[frame["date"].ge(dates[20]), "xs_late_chip"] = np.tile(
            np.arange(10, dtype=float), int(frame["date"].ge(dates[20]).sum() / 10),
        )
        feature_pool = ["xs_base", "xs_late_chip"]
        early_training = frame.loc[frame["date"].le(dates[19])]
        later_training = frame.loc[frame["date"].le(dates[44])]

        self.assertEqual(_select_fold_features(early_training, feature_pool, 8), ["xs_base"])
        self.assertEqual(
            _select_fold_features(later_training, feature_pool, 8),
            ["xs_base", "xs_late_chip"],
        )

    def test_walk_forward_folds_admit_late_feature_after_mature_history_accumulates(self):
        panel = _panel(days=420, stocks=10)
        dates = pd.DatetimeIndex(panel["date"].drop_duplicates())
        panel["xs_late_chip"] = np.where(
            panel["date"].ge(dates[180]), panel["xs_momentum_5"] * 2, np.nan,
        )
        panel.attrs["model_feature_columns"] = [
            *(f"xs_{column}" for column in FEATURE_COLUMNS), "xs_late_chip",
        ]

        result = fit_panel_model(
            panel, "LightGBM", 20, lambda _: _ExactModel(), objective="regression",
        )
        folds = result["oos_metrics"]["folds"]

        self.assertNotIn("xs_late_chip", folds[0]["feature_columns"])
        self.assertTrue(any("xs_late_chip" in fold["feature_columns"] for fold in folds[1:]))
        self.assertIn("xs_late_chip", result["features_used"])

    def test_rank_ic_and_top_bottom_accept_correct_order(self):
        result = evaluate_oos(self.evaluation_frame(), 20, 10)
        self.assertTrue(result["validated"])
        self.assertGreater(result["rank_ic"], 0.9)
        self.assertIsNotNone(result["ndcg_at_20"])
        self.assertEqual(result["ndcg_at_20_days"], result["ndcg_days"])
        self.assertGreater(result["top_excess_return"], 0)
        self.assertLess(result["bottom_excess_return"], 0)

    def test_wrong_order_or_biased_mae_is_blocked(self):
        frame = self.evaluation_frame()
        frame["prediction"] *= -1
        self.assertFalse(evaluate_oos(frame, 20, 10)["validated"])
        frame = self.evaluation_frame()
        frame["prediction"] += 0.2
        self.assertFalse(evaluate_oos(frame, 20, 10)["validated"])
        self.assertFalse(evaluate_oos(frame.head(100), 20, 10)["validated"])

    def test_degenerate_ic_does_not_fabricate_significance(self):
        frame = self.evaluation_frame()
        frame["actual"] = frame["prediction"]
        result = evaluate_oos(frame, 20, 10)
        self.assertFalse(result["validated"])
        self.assertIsNone(result["rank_ic_hac_t"])

    def test_evaluation_does_not_copy_unrelated_panel_metadata(self):
        class MetadataGuard:
            def __deepcopy__(self, memo):
                raise AssertionError("评估临时表不应深拷贝原始面板 attrs")

        frame = self.evaluation_frame()
        metadata = MetadataGuard()
        frame.attrs["unrelated_panel_metadata"] = metadata

        result = evaluate_oos(frame, 20, 10)

        self.assertIn("ndcg", result)
        self.assertIs(frame.attrs["unrelated_panel_metadata"], metadata)

    def test_overlapping_labels_are_purged_by_endpoint(self):
        panel = _panel()
        models = []
        def factory(name):
            model = _ExactModel()
            models.append(model)
            return model
        cutoff = panel["date"].drop_duplicates().iloc[380]
        result = fit_panel_model(panel, "LightGBM", 20, factory, cutoff=cutoff)
        self.assertTrue(result["validated"])
        for fold in result["oos_metrics"]["folds"]:
            self.assertLess(fold["latest_training_label_end"], fold["test_start"])
        for model in models:
            self.assertTrue(panel.loc[model.fit_indices, "label_end_20"].lt(cutoff).all())

    def test_real_lightgbm_and_xgboost_share_excess_target(self):
        panel = _panel()
        live = panel.loc[panel["date"].eq(panel["date"].max())]
        for name in ("LightGBM", "XGBoost"):
            result = fit_panel_model(panel, name, 20)
            self.assertTrue(result["validated"], result["status"])
            self.assertEqual(result["target_kind"], TARGET_KIND)
            self.assertIsNone(result["predicted_price"])
            predicted = predict_panel(result, live)
            self.assertAlmostEqual(predicted["prediction_zscore"].mean(), 0, places=7)
            self.assertAlmostEqual(predicted["prediction_zscore"].std(ddof=0), 1, places=7)

    def test_future_values_cannot_change_historical_fit(self):
        panel = _panel()
        cutoff = panel["date"].drop_duplicates().iloc[380]
        first = fit_panel_model(panel, "LightGBM", 20, lambda _: _ExactModel(), cutoff=cutoff)
        panel.loc[panel["date"].ge(cutoff), "target_20"] = 999
        second = fit_panel_model(panel, "LightGBM", 20, lambda _: _ExactModel(), cutoff=cutoff)
        self.assertEqual(first["oos_metrics"], second["oos_metrics"])

    def test_unsorted_input_keeps_identical_validation_metrics_and_reports_progress(self):
        panel = _panel(days=420, stocks=10)
        progress = []
        sorted_result = fit_panel_model(
            panel, "LightGBM", 20, lambda _: _ExactModel(), objective="regression",
        )
        shuffled = panel.sample(frac=1.0, random_state=17).reset_index(drop=True)
        shuffled.attrs.update(panel.attrs)
        shuffled_result = fit_panel_model(
            shuffled, "LightGBM", 20, lambda _: _ExactModel(), objective="regression",
            progress=progress.append,
        )

        self.assertEqual(sorted_result["oos_metrics"]["dates"], shuffled_result["oos_metrics"]["dates"])
        self.assertEqual(sorted_result["oos_metrics"]["oos_samples"], shuffled_result["oos_metrics"]["oos_samples"])
        for key in ("rank_ic", "rank_ic_hac_t", "ndcg", "ndcg_random_baseline", "mae", "baseline_mae"):
            first, second = sorted_result["oos_metrics"].get(key), shuffled_result["oos_metrics"].get(key)
            if first is not None and second is not None:
                self.assertAlmostEqual(first, second, places=12, msg=key)
        self.assertTrue(any("第1/" in message and "原生拟合" in message for message in progress))
        execution = shuffled_result["execution"]
        self.assertEqual(execution["features"], len(shuffled_result["features_used"]))
        self.assertGreater(execution["elapsed_seconds"], 0)
        self.assertIn("native_fit", execution["stage_seconds"])
        self.assertEqual(execution["threads"], settings.TRAINING_N_JOBS)
        self.assertEqual(shuffled_result["oos_metrics"]["ndcg_protocol"], settings.NDCG_PROTOCOL)

    def test_validation_slicing_supports_multiple_datetime_precisions(self):
        reference_counts = None
        nanosecond_panel = None
        for unit in ("s", "ms", "us", "ns"):
            panel = _panel(days=420, stocks=10)
            panel["date"] = panel["date"].to_numpy().astype(f"datetime64[{unit}]")
            for horizon in settings.MODEL_HORIZONS:
                column = f"label_end_{horizon}"
                panel[column] = panel[column].to_numpy().astype(f"datetime64[{unit}]")
            prepared = _prepare_validation_data(panel, 20, None, settings.TRAINING_WINDOW_DAYS)
            train_positions, test_positions = _fold_row_positions(prepared, prepared["folds"][0])
            counts = (len(train_positions), len(test_positions))
            self.assertGreater(counts[0], 0)
            self.assertGreater(counts[1], 0)
            if reference_counts is None:
                reference_counts = counts
            else:
                self.assertEqual(counts, reference_counts)
            if unit == "ns":
                nanosecond_panel = panel

        result = fit_panel_model(
            nanosecond_panel, "LightGBM", 20, lambda _: _ExactModel(), objective="regression",
        )
        self.assertTrue(result.get("oos_metrics", {}).get("folds"))

    def test_windows_peak_memory_api_declares_ctypes_signatures(self):
        kernel32 = SimpleNamespace(GetCurrentProcess=MagicMock(return_value=123))
        psapi = SimpleNamespace(GetProcessMemoryInfo=MagicMock())

        def populate_memory(handle, pointer, size):
            pointer._obj.PeakWorkingSetSize = 8 * 1024 * 1024
            return 1

        psapi.GetProcessMemoryInfo.side_effect = populate_memory
        fake_windll = SimpleNamespace(kernel32=kernel32, psapi=psapi)
        with patch("services.ml_pipeline.os.name", "nt"), patch.object(
            ctypes, "windll", fake_windll, create=True,
        ):
            result = _memory_metrics()

        self.assertEqual(result["peak_memory_mb"], 8.0)
        self.assertEqual(result["memory_status"], "操作系统进程生命周期峰值工作集")
        self.assertEqual(kernel32.GetCurrentProcess.argtypes, [])
        self.assertIsNotNone(kernel32.GetCurrentProcess.restype)
        self.assertEqual(len(psapi.GetProcessMemoryInfo.argtypes), 3)
        self.assertIsNotNone(psapi.GetProcessMemoryInfo.restype)

    def test_validation_cache_reuses_complete_folds_and_repairs_corruption(self):
        class CountingModel(_ExactModel):
            fit_count = 0

            def fit(self, x, y):
                type(self).fit_count += 1
                return super().fit(x, y)

        panel = _panel(days=420, stocks=10)
        CountingModel.fit_count = 0
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(settings, "VALIDATION_CACHE_ENABLED", True), \
                    patch.object(settings, "VALIDATION_CACHE_DIR", Path(directory)), \
                    patch("services.ml_pipeline.make_ranker", side_effect=lambda _: CountingModel()):
                cold = fit_panel_model(panel, "LightGBM", 20, objective="lambdarank")
                cold_fit_count = CountingModel.fit_count
                hot = fit_panel_model(panel, "LightGBM", 20, objective="lambdarank")

                self.assertEqual(cold["execution"]["cache"]["hits"], 0)
                self.assertGreater(cold["execution"]["cache"]["misses"], 0)
                self.assertEqual(hot["execution"]["cache"]["hits"], cold["execution"]["cache"]["misses"])
                self.assertEqual(hot["execution"]["stage_seconds"]["native_fit"], 0.0)
                self.assertEqual(cold["oos_metrics"], hot["oos_metrics"])
                self.assertEqual(cold["execution"]["validation_identity"], hot["execution"]["validation_identity"])
                self.assertFalse(cold["execution"]["validation_cache_replay"])
                self.assertTrue(hot["execution"]["validation_cache_replay"])
                self.assertTrue(all(
                    item["source"] == "hot" for item in hot["execution"]["validation_cache_sources"]
                ))
                expected_final_fit = 1 if hot["validated"] else 0
                self.assertEqual(CountingModel.fit_count - cold_fit_count, expected_final_fit)

                first_key = hot["execution"]["folds"][0]["cache_key"]
                (Path(directory) / f"{first_key}.vfc").write_bytes(b"broken cache")
                before_repair = CountingModel.fit_count
                repaired = fit_panel_model(panel, "LightGBM", 20, objective="lambdarank")
                self.assertEqual(repaired["execution"]["cache"]["hits"], hot["execution"]["cache"]["hits"] - 1)
                self.assertEqual(CountingModel.fit_count - before_repair, 1 + expected_final_fit)

    def test_cache_identity_changes_for_data_revision_date_append_and_configuration(self):
        panel = _panel(days=420, stocks=10)
        prepared = _prepare_validation_data(panel, 20, None, settings.TRAINING_WINDOW_DAYS)
        fold = prepared["folds"][0]
        common = ("LightGBM", 20, "lambdarank", settings.TRAINING_WINDOW_DAYS, 1, fold, prepared,
                  len(prepared["labelled"]), 1000, {"python": "test", "packages": {}})
        base = _validation_cache_key(*common)
        base_with_model_parameters = _validation_cache_key(*common, {"n_estimators": 120})

        revised = panel.copy()
        revised.loc[0, "target_20"] += 0.001
        revised_prepared = _prepare_validation_data(revised, 20, None, settings.TRAINING_WINDOW_DAYS)
        appended_prepared = _prepare_validation_data(_panel(days=421, stocks=10), 20, None, settings.TRAINING_WINDOW_DAYS)
        configured = panel.copy()
        configured.attrs["style_factors"] = {"configuration_id": "changed-style-config"}
        configured_prepared = _prepare_validation_data(configured, 20, None, settings.TRAINING_WINDOW_DAYS)

        self.assertNotEqual(base, _validation_cache_key(*common[:5], revised_prepared["folds"][0], revised_prepared,
                                                         len(revised_prepared["labelled"]), 1000, common[-1]))
        self.assertNotEqual(base, _validation_cache_key(*common[:5], appended_prepared["folds"][0], appended_prepared,
                                                         len(appended_prepared["labelled"]), 1000, common[-1]))
        self.assertNotEqual(base, _validation_cache_key(*common[:5], configured_prepared["folds"][0], configured_prepared,
                                                         len(configured_prepared["labelled"]), 1000, common[-1]))
        self.assertNotEqual(
            base_with_model_parameters,
            _validation_cache_key(*common, {"n_estimators": 121}),
        )

    def test_real_regressors_train_thirty_day_winsorized_labels(self):
        panel = _panel()
        original = panel["target_30"].copy()
        for name in ("LightGBM", "XGBoost"):
            result = fit_panel_model(panel, name, 30, objective="regression")
            self.assertTrue(result["validated"], result["status"])
            self.assertEqual(result["regression_loss"], "l1")
            self.assertEqual(result["validation_retrain_days"], settings.OOS_RETRAIN_DAYS)
            self.assertGreater(result["label_processing"]["final"]["clipped_samples"], 0)
            for fold in result["oos_metrics"]["folds"]:
                self.assertLess(fold["latest_training_label_end"], fold["test_start"])
                days = pd.bdate_range(fold["test_start"], fold["test_end"])
                self.assertLessEqual(len(days), settings.OOS_RETRAIN_DAYS)
        pd.testing.assert_series_equal(panel["target_30"], original)

    def test_snapshot_version_and_context_lineage(self):
        panel = _panel(10)
        panel.attrs["context_id"] = "unit-test-context"
        context = {"context_id": "unit-test-context", "model_version": settings.MODEL_VERSION,
                   "target_kind": TARGET_KIND, "models": {}, "predictions": {}, "reference_closes": {}}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(settings, "DATA_DIR", Path(directory) / "data"), patch.object(settings, "MODEL_DIR", Path(directory) / "models"):
                save_context(context, panel)
                restored, stored = load_context(include_panel=True)
                self.assertEqual(restored["context_id"], stored.attrs["context_id"])
                self.assertEqual(len(panel), len(stored))

    def test_single_stock_has_no_fabricated_forecast_without_context(self):
        dates = pd.bdate_range("2024-01-01", periods=100)
        bars = pd.DataFrame({"date": dates, "open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0, "volume": 1000.0})
        result = analyze_stock(bars, 20, code="600000", market_context={})
        self.assertEqual(result["prediction_reference_price"], 10)
        for models in result["models"].values():
            self.assertTrue(all(not item["validated"] and item["prediction_zscore"] is None for item in models))

    def test_stale_stock_or_mismatched_context_is_rejected(self):
        context = {"reference_date": "2024-01-01", "reference_closes": {"600000": 10}}
        result = models_for_stock(context, "600000", "2024-01-02", 10, 20)
        self.assertTrue(all(not item["validated"] for item in result))

    def test_walk_forward_refits_before_each_prediction_block(self):
        panel = _panel()
        features = panel.loc[panel["code"].eq("000009"), ["date", "close"]].reset_index(drop=True)
        cutoffs = []
        def fit(frame, name, horizon, cutoff=None):
            cutoffs.append(cutoff)
            return {"model": name, "validated": True, "estimator": _ExactModel(),
                    "features_used": ["xs_momentum_5"], "status": "测试通过", "oos_metrics": {}}
        with patch("services.ml_pipeline.fit_panel_model", side_effect=fit):
            predictions, log = walk_forward_predictions(features, 320, code="000009", market_panel=panel)
        self.assertEqual(len(cutoffs), 10)
        self.assertTrue(predictions.iloc[:320].isna().all())
        self.assertTrue(predictions.iloc[320:].gt(0).all())
        self.assertTrue(all(item["trained_through"] < item["prediction_start"] for item in log))


if __name__ == "__main__":
    unittest.main()
