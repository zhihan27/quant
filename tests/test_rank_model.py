"""验证排序目标、NDCG 检验与滚动训练窗口。"""

import unittest

import numpy as np
import pandas as pd

from config import settings
from services.feature_engineering import FEATURE_COLUMNS
from services.ml_pipeline import evaluate_oos, fit_panel_model, predict_panel, relative_signal
from services.rank_model import (
    SCORE_KIND_RANK, SCORE_KIND_RETURN, CrossSectionRanker, ndcg_at_k, random_ndcg_baseline, relevance_grades,
)


def _panel(days=1200, stocks=20, drift=0.0):
    dates = pd.bdate_range("2018-01-01", periods=days)
    rows = []
    for day, date in enumerate(dates):
        for stock in range(stocks):
            quality = (stock - (stocks - 1) / 2) / 5
            row = {"date": date, "code": str(stock).zfill(6), "close": 10.0}
            row.update({"xs_" + column: quality for column in FEATURE_COLUMNS})
            for horizon in (5, 20, 60):
                mature = day + horizon < days
                outcome = drift + quality * 0.02 + 0.004 * np.sin(day * 0.7 + stock * 1.4)
                row[f"target_{horizon}"] = outcome if mature else np.nan
                row[f"forward_return_{horizon}"] = outcome + 0.01 if mature else np.nan
                row[f"label_end_{horizon}"] = dates[day + horizon] if mature else pd.NaT
            rows.append(row)
    return pd.DataFrame(rows)


class RelevanceGradeTests(unittest.TestCase):
    """相关度等级必须只在一个交易日内部比较。"""

    def test_grades_are_per_date_and_cover_all_levels(self):
        dates = pd.to_datetime(["2024-01-01"] * 5 + ["2024-01-02"] * 5)
        target = pd.Series([0.5, 0.4, 0.3, 0.2, 0.1] * 2)
        grades = relevance_grades(target, pd.Series(dates), grade_count=5)
        self.assertEqual(grades.tolist(), [4.0, 3.0, 2.0, 1.0, 0.0] * 2)

    def test_equal_targets_share_one_grade(self):
        grades = relevance_grades(pd.Series([0.1, 0.1, 0.1]), pd.Series(["d"] * 3), grade_count=5)
        self.assertEqual(grades.nunique(), 1)

    def test_missing_target_makes_missing_grade(self):
        grades = relevance_grades(pd.Series([0.1, np.nan]), pd.Series(["d", "d"]), grade_count=5)
        self.assertTrue(np.isnan(grades.iloc[1]))
        with self.assertRaises(ValueError):
            relevance_grades(pd.Series([0.1, 0.2]), pd.Series(["d", "d"]), grade_count=1)


class NdcgTests(unittest.TestCase):
    """NDCG 必须奖励把高等级股票排在前面，并拒绝没有排序能力的分数。"""

    def test_perfect_order_scores_one_and_reverse_scores_below(self):
        grades = np.array([4.0, 3.0, 2.0, 1.0, 0.0])
        self.assertAlmostEqual(ndcg_at_k(grades, np.array([5.0, 4.0, 3.0, 2.0, 1.0]), 2), 1.0)
        self.assertLess(ndcg_at_k(grades, np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 2), 0.5)
        self.assertTrue(np.isnan(ndcg_at_k(np.array([1.0]), np.array([1.0]), 1)))

    def test_flat_scores_are_indistinguishable_from_random(self):
        grades = np.array([4.0, 3.0, 2.0, 1.0, 0.0, 4.0, 3.0, 2.0, 1.0, 0.0])
        baseline = random_ndcg_baseline(grades, 3)
        flat = ndcg_at_k(grades, np.zeros(len(grades)), 3)
        self.assertLessEqual(abs(flat - baseline), 0.5)
        self.assertGreater(baseline, 0.0)


class RelativeSignalTests(unittest.TestCase):
    """排序模型按截面相对位置给信号，回归模型才要求预期超额为正。"""

    def test_rank_model_uses_percentile_instead_of_raw_score(self):
        bottom = {"validated": True, "score_kind": SCORE_KIND_RANK, "prediction_zscore": -1.2, "prediction_percentile": 0.1}
        top = {"validated": True, "score_kind": SCORE_KIND_RANK, "prediction_zscore": 0.4, "prediction_percentile": 0.9}
        self.assertLess(relative_signal(bottom), 0.0)
        self.assertAlmostEqual(relative_signal(top), 0.4)
        # 排序分数整体为负不代表看空，只要仍在截面均值之上就应保留正信号。
        top["predicted_excess_return"] = -0.3
        self.assertAlmostEqual(relative_signal(top), 0.4)

    def test_regression_model_keeps_absolute_excess_gate(self):
        item = {"validated": True, "score_kind": SCORE_KIND_RETURN, "prediction_zscore": 1.5, "predicted_excess_return": -0.01}
        # 预期超额为负时不给买入信号，但保留负值以便参与排序。
        self.assertLessEqual(relative_signal(item), 0.0)
        item["predicted_excess_return"] = 0.01
        self.assertAlmostEqual(relative_signal(item), 1.5)

    def test_unvalidated_or_missing_score_yields_no_signal(self):
        self.assertIsNone(relative_signal({"validated": False, "prediction_zscore": 1.0}))
        self.assertIsNone(relative_signal({"validated": True, "prediction_zscore": None}))


class ScanDisplayTests(unittest.TestCase):
    """扫描页展示不得把无量纲排序分当成预期超额收益。"""

    def test_rank_model_is_shown_as_percentile(self):
        from services.scanning import validated_return

        rank_model = {"validated": True, "score_kind": SCORE_KIND_RANK, "predicted_excess_return": 0.8, "prediction_percentile": 0.93}
        self.assertAlmostEqual(validated_return(rank_model), 0.93)
        regression = {"validated": True, "score_kind": SCORE_KIND_RETURN, "predicted_excess_return": 0.021, "prediction_percentile": 0.93}
        self.assertAlmostEqual(validated_return(regression), 0.021)

    def test_missing_percentile_and_unvalidated_models_are_explained(self):
        from services.scanning import validated_return

        self.assertIn("百分位", validated_return({"validated": True, "score_kind": SCORE_KIND_RANK}))
        self.assertEqual(validated_return({"validated": False, "status": "样本不足"}), "样本不足")


class RankEvaluationTests(unittest.TestCase):
    """排序目标的验证必须依赖 NDCG，而不是无量纲分数上的绝对误差。"""

    def evaluation_frame(self, prediction=None):
        panel = _panel(days=100, stocks=10)
        frame = panel[["date", "code"]].copy()
        frame["actual"] = panel["xs_momentum_5"] * 0.02 + 0.01 * np.sin((panel.index // 10) * 0.7 + (panel.index % 10) * 1.4)
        frame["prediction"] = frame["actual"] if prediction is None else prediction
        frame["baseline"] = 0.0
        frame["absolute_return"] = frame["actual"] + 0.01
        return frame

    def test_rank_score_ignores_mae_and_checks_ndcg(self):
        frame = self.evaluation_frame()
        # 加噪后每日Rank IC不再恒定，HAC才能给出有效方差。
        frame["prediction"] = frame["actual"] + 0.01 * np.sin(np.arange(len(frame)) * 1.7)
        result = evaluate_oos(frame, 20, 10, score_kind=SCORE_KIND_RANK)
        self.assertTrue(result["validated"], result["validation_rule"])
        self.assertEqual(result["score_kind"], SCORE_KIND_RANK)
        self.assertIsNone(result["mae"])
        self.assertGreater(result["ndcg"], result["ndcg_random_baseline"])
        self.assertGreater(result["ndcg_ratio"], 1.05)

    def test_flat_rank_score_is_rejected(self):
        frame = self.evaluation_frame()
        frame["prediction"] = 0.0
        self.assertFalse(evaluate_oos(frame, 20, 10, score_kind=SCORE_KIND_RANK)["validated"])

    def test_regression_score_still_checks_mae_baseline(self):
        frame = self.evaluation_frame()
        frame["prediction"] += 0.2
        result = evaluate_oos(frame, 20, 10)
        self.assertIsNotNone(result["mae"])
        self.assertFalse(result["validated"])


class CrossSectionRankerTests(unittest.TestCase):
    """真实排序器必须能在截面内学到优胜劣汰的顺序。"""

    def test_ranker_learns_synthetic_ordering(self):
        panel = _panel(days=200, stocks=20)
        features = [f"xs_{column}" for column in FEATURE_COLUMNS]
        sample = panel.dropna(subset=["target_20"])
        for name in ("LightGBM", "XGBoost"):
            try:
                ranker = CrossSectionRanker(name).fit_cross_section(sample[features], sample["target_20"], sample["date"])
            except ImportError as exc:
                self.skipTest(f"缺少依赖：{exc.name}")
            self.assertEqual(ranker.groups_trained, sample["date"].nunique())
            self.assertGreater(ranker.samples_trained, 0)
            self.assertEqual(ranker.score_kind, SCORE_KIND_RANK)
            predicted = ranker.predict(sample[features])
            self.assertEqual(len(predicted), len(sample))
            self.assertEqual(len(ranker.feature_importances_), len(features))

    def test_ranker_rejects_sample_without_labels(self):
        panel = _panel(days=40, stocks=10)
        features = [f"xs_{column}" for column in FEATURE_COLUMNS]
        with self.assertRaises(ValueError):
            CrossSectionRanker("LightGBM").fit_cross_section(panel[features], pd.Series(np.nan, index=panel.index), panel["date"])


class RollingTrainingWindowTests(unittest.TestCase):
    """滚动窗口必须只保留最近N个交易日，且窗口内的标签仍不跨越测试期。"""

    def test_window_truncates_training_history(self):
        panel = _panel(days=1200)
        result = fit_panel_model(panel, "LightGBM", 20, window=250, objective="lambdarank")
        self.assertTrue(result["validated"], result["status"])
        self.assertEqual(result["training_window_days"], 250)
        self.assertEqual(result["score_kind"], SCORE_KIND_RANK)
        folds = result["oos_metrics"]["folds"]
        self.assertGreaterEqual(len(folds), 2)
        for fold in folds:
            self.assertLessEqual(fold["training_dates"], 250)
            self.assertLess(fold["latest_training_label_end"], fold["test_start"])
        self.assertLess(folds[0]["trained_from"], folds[-1]["trained_from"])

    def test_unlimited_window_uses_all_history(self):
        panel = _panel(days=1200)
        result = fit_panel_model(panel, "LightGBM", 20, window=0, objective="regression")
        folds = result["oos_metrics"]["folds"]
        self.assertGreater(folds[-1]["training_dates"], 250)
        self.assertEqual(folds[-1]["trained_from"], folds[0]["trained_from"])

    def test_objective_must_be_supported(self):
        with self.assertRaises(ValueError):
            fit_panel_model(_panel(days=300), "LightGBM", 20, objective="未知目标")

    def test_auto_mode_records_both_objectives(self):
        panel = _panel(days=1200)
        result = fit_panel_model(panel, "LightGBM", 20, objective="auto", window=400)
        attempts = {item["objective"]: item for item in result["objective_selection"]}
        self.assertEqual(set(attempts), {"regression", "lambdarank"})
        self.assertEqual(attempts["lambdarank"]["score_kind"], SCORE_KIND_RANK)
        self.assertEqual(attempts["regression"]["score_kind"], SCORE_KIND_RETURN)
        self.assertIn(result["objective"], {"regression", "lambdarank"})
        self.assertTrue(result["validated"], result["status"])

    def test_rank_model_avoids_all_bullish_scores(self):
        """回归模型在全市场上涨时给出 100% 为正的分数，排序模型不应如此。"""
        panel = _panel(days=420, stocks=20, drift=0.05)
        live = panel.loc[panel["date"].eq(panel["date"].max())]
        ratios = {}
        for objective in ("regression", "lambdarank"):
            result = fit_panel_model(panel, "LightGBM", 20, objective=objective)
            self.assertTrue(result["validated"], result["status"])
            predicted = predict_panel(result, live)
            ratios[objective] = float((predicted["prediction"] > 0).mean())
        self.assertEqual(ratios["regression"], 1.0)
        self.assertLess(ratios["lambdarank"], 1.0)

    def test_default_objective_comes_from_settings(self):
        result = fit_panel_model(_panel(days=300), "LightGBM", 20)
        self.assertEqual(result["objective"], settings.DEFAULT_OBJECTIVE_MODE)


if __name__ == "__main__":
    unittest.main()
