"""用确定性测试样本验证超额标签、日期截面及未来数据隔离。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from config import settings
from services.feature_engineering import FEATURE_COLUMNS, add_forward_returns, normalize_cross_section
from services.ml_pipeline import (
    TARGET_KIND, evaluate_oos, fit_panel_model, load_context, models_for_stock,
    normalize_predictions, predict_panel, save_context, unavailable_model,
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
            for horizon in (5, 20, 60):
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

    def test_labels_are_benchmark_excess_and_last_rows_are_unknown(self):
        dates = pd.bdate_range("2024-01-01", periods=20)
        stock = pd.DataFrame({"date": dates, "close": 100 * 1.01 ** np.arange(20)})
        benchmark = pd.DataFrame({"date": dates, "close": 100 * 1.02 ** np.arange(20)})
        result = add_forward_returns(stock, (5,), benchmark)
        self.assertAlmostEqual(result.loc[0, "target_5"], 1.01**5 - 1.02**5)
        self.assertLess(result.loc[0, "target_5"], 0)
        self.assertTrue(result["target_5"].tail(5).isna().all())
        self.assertEqual(result.loc[0, "label_end_5"], dates[5])
        self.assertTrue(add_forward_returns(stock, (5,))["target_5"].isna().all())

    def test_missing_endpoint_never_shifts_stock_horizon(self):
        dates = pd.bdate_range("2024-01-01", periods=15)
        benchmark = pd.DataFrame({"date": dates, "close": 100.0})
        stock = pd.DataFrame({"date": dates, "close": np.arange(15) + 100.0}).drop(index=5)
        result = add_forward_returns(stock, (5,), benchmark)
        self.assertTrue(pd.isna(result.loc[0, "target_5"]))
        self.assertEqual(result.loc[1, "label_end_5"], dates[6])

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

    def test_rank_ic_and_top_bottom_accept_correct_order(self):
        result = evaluate_oos(self.evaluation_frame(), 20, 10)
        self.assertTrue(result["validated"])
        self.assertGreater(result["rank_ic"], 0.9)
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
