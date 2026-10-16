"""验证 PSI / KS 的计算与三类漂移的阈值判定：全部使用构造分布，不依赖真实行情。

PSI 与 KS 都是手写实现（仓库没有现成实现，scipy 又只是 statsmodels 的传递依赖、
未在 requirements.txt 声明），因此这里的用例必须对已知答案，而不是「跑通就算过」。
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from config import settings
from services.drift_monitor import (
    LEVEL_INSUFFICIENT,
    LEVEL_OK,
    LEVEL_SEVERE,
    LEVEL_UNAVAILABLE,
    LEVEL_WATCH,
    categorical_stability_index,
    classify_psi,
    concept_drift_row,
    evaluate_drift_alerts,
    feature_drift_rows,
    kolmogorov_smirnov,
    population_stability_index,
    prediction_drift_row,
)


def _normal(size: int, mean: float = 0.0, deviation: float = 1.0, seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).normal(mean, deviation, size)


class StabilityIndexTests(unittest.TestCase):
    """PSI 的取值边界与已知答案。"""

    def test_identical_sample_is_nearly_zero(self):
        values = _normal(4000)
        psi = population_stability_index(values, values, bins=10)
        self.assertIsNotNone(psi)
        self.assertLess(psi, 0.01, "同一份样本与自己做 PSI 应当约等于 0")

    def test_same_distribution_different_draw_is_small(self):
        psi = population_stability_index(_normal(4000, seed=1), _normal(4000, seed=2), bins=10)
        self.assertLess(psi, settings.DRIFT_PSI_WATCH, "同分布的两组抽样不该被判成漂移")

    def test_mean_shift_raises_psi_above_severe(self):
        psi = population_stability_index(_normal(4000), _normal(4000, mean=1.0), bins=10)
        self.assertGreater(psi, settings.DRIFT_PSI_SEVERE)

    def test_psi_matches_hand_computed_value(self):
        """三分组构造：参考样本 0.5 分位恰好落在 0 与 1 之间，分箱边界因此确定可知。

        参考样本按边界 -inf / 0.5 / inf 分成 500:500，当前样本分成 400:600。
        """
        reference = np.concatenate([np.zeros(500), np.ones(200), np.full(300, 2.0)])
        current = np.concatenate([np.zeros(400), np.ones(200), np.full(400, 2.0)])
        expected = float((0.4 - 0.5) * np.log(0.4 / 0.5) + (0.6 - 0.5) * np.log(0.6 / 0.5))
        psi = population_stability_index(reference, current, bins=2)
        self.assertIsNotNone(psi)
        self.assertAlmostEqual(psi, expected, places=6)

    def test_constant_reference_yields_none_not_zero(self):
        """参考列近乎常数时分不出箱，此时返回 0.0 会假装「没有漂移」。"""
        self.assertIsNone(population_stability_index(np.zeros(1000), _normal(100)))

    def test_categorical_index_uses_union_of_categories(self):
        reference = ["银行", "银行", "医药"]
        current = ["银行", "医药", "医药"]
        psi = categorical_stability_index(reference, current)
        expected = float((2 / 3 - 1 / 3) * np.log((2 / 3) / (1 / 3))
                         + (1 / 3 - 2 / 3) * np.log((1 / 3) / (2 / 3)))
        self.assertAlmostEqual(psi, expected, places=6)

    def test_new_category_in_current_window_is_counted(self):
        """参考窗口没出现过的类别按参考占比 0 计入，否则「冒出新行业」反而算不出来。"""
        psi = categorical_stability_index(["银行"] * 10, ["银行"] * 9 + ["军工"])
        self.assertIsNotNone(psi)
        self.assertGreater(psi, 0)

    def test_empty_input_returns_none_not_zero(self):
        """算不出来要返回 None。假造一个 0 会被读成「完全没有漂移」，比缺失更糟。"""
        self.assertIsNone(population_stability_index(np.array([]), _normal(100)))
        self.assertIsNone(population_stability_index(_normal(100), np.array([])))
        self.assertIsNone(categorical_stability_index([], ["银行"]))

    def test_non_finite_values_are_ignored(self):
        psi = population_stability_index(
            np.concatenate([_normal(500), [np.nan, np.inf]]), _normal(500), bins=5
        )
        self.assertIsNotNone(psi)
        self.assertTrue(np.isfinite(psi))


class KolmogorovSmirnovTests(unittest.TestCase):
    """两样本 KS 统计量：同分布为 0，完全分离为 1。"""

    def test_identical_sample_is_zero(self):
        self.assertEqual(kolmogorov_smirnov(_normal(500), _normal(500)), 0.0)

    def test_fully_separated_samples_is_one(self):
        self.assertEqual(kolmogorov_smirnov(np.arange(0, 50.0), np.arange(100, 150.0)), 1.0)

    def test_known_two_point_distribution(self):
        """参照 {0}、当前 {0, 1}：经验分布最大间距出现在 0 处，为 0.5。"""
        self.assertAlmostEqual(kolmogorov_smirnov(np.zeros(10), np.array([0.0] * 5 + [1.0] * 5)), 0.5)

    def test_median_shift_is_about_half(self):
        statistic = kolmogorov_smirnov(_normal(4000), _normal(4000, mean=1.0))
        self.assertGreater(statistic, 0.3)
        self.assertLess(statistic, 0.6)

    def test_empty_input_returns_none(self):
        self.assertIsNone(kolmogorov_smirnov(np.array([]), _normal(10)))


class ClassifyTests(unittest.TestCase):
    """PSI 阈值的分级：0.1 关注、0.25 严重，样本不足优先于两者。"""

    def test_thresholds(self):
        self.assertEqual(classify_psi(0.05, 10_000), LEVEL_OK)
        self.assertEqual(classify_psi(settings.DRIFT_PSI_WATCH, 10_000), LEVEL_OK, "阈值本身不算超线")
        self.assertEqual(classify_psi(0.15, 10_000), LEVEL_WATCH)
        self.assertEqual(classify_psi(settings.DRIFT_PSI_SEVERE, 10_000), LEVEL_WATCH)
        self.assertEqual(classify_psi(0.30, 10_000), LEVEL_SEVERE)

    def test_insufficient_sample_wins_over_threshold(self):
        """样本不足时 PSI 本身不可信，即便数值很大也不能报「严重」。"""
        self.assertEqual(classify_psi(0.9, settings.DRIFT_MINIMUM_SAMPLES - 1), LEVEL_INSUFFICIENT)
        self.assertEqual(classify_psi(0.9, settings.DRIFT_MINIMUM_SAMPLES), LEVEL_SEVERE)

    def test_uncomputable_psi_is_not_labelled_insufficient(self):
        """「算不出来」与「样本不足」是两句不同的事实陈述，不能合并成一个等级。"""
        self.assertEqual(classify_psi(None, 10_000), LEVEL_UNAVAILABLE)


def _panel(days: int = 900, stocks: int = 30, industry_shift: bool = False) -> pd.DataFrame:
    """造一份结构上与本地面板一致的面板：日期 × 股票，含漂移监控用到的列。"""
    generator = np.random.default_rng(11)
    dates = pd.bdate_range(end="2026-10-16", periods=days)
    rows = []
    industries = ["银行", "医药", "电子", "食品饮料", "机械"]
    for date_index, date in enumerate(dates):
        # 最后 1/4 的交易日人为把波动率整体抬高，用来触发漂移判定。
        drift = date_index > days * 0.75
        for stock in range(stocks):
            rows.append({
                "date": date, "code": f"{600000 + stock:06d}",
                "volatility_20": float(generator.normal(0.02 + (0.06 if drift else 0.0), 0.002)),
                "turnover_20": float(abs(generator.normal(2.0, 0.3))),
                "amihud_20": float(abs(generator.normal(1e-6, 2e-7))),
                "log_market_cap": float(generator.normal(23.0, 1.0)),
                "industry": industries[0] if (industry_shift and drift) else industries[stock % len(industries)],
            })
    return pd.DataFrame(rows)


class FeatureDriftTests(unittest.TestCase):
    """每个配置维度都必须出一行：算得出来的给数，算不出来的给原因。"""

    def test_every_configured_dimension_is_reported(self):
        rows = feature_drift_rows(_panel(), current_days=20, reference_days=400)
        self.assertEqual({row["feature_name"] for row in rows}, set(settings.DRIFT_MONITORED_FEATURES))

    def test_shifted_feature_is_flagged_severe(self):
        rows = {row["feature_name"]: row for row in feature_drift_rows(_panel(), current_days=20, reference_days=400)}
        self.assertEqual(rows["波动率"]["threshold_exceeded"], LEVEL_SEVERE)
        self.assertGreater(rows["波动率"]["psi"], settings.DRIFT_PSI_SEVERE)
        self.assertEqual(rows["换手率"]["threshold_exceeded"], LEVEL_OK, "没动过的维度不该被判成漂移")

    def test_valuation_dimension_is_marked_unavailable_with_reason(self):
        """行情源不提供 PE/PB/PS，这一维只能如实标注未接入，不能静默跳过。"""
        rows = {row["feature_name"]: row for row in feature_drift_rows(_panel(), current_days=20, reference_days=400)}
        valuation = rows["估值"]
        self.assertEqual(valuation["threshold_exceeded"], LEVEL_UNAVAILABLE)
        self.assertIsNone(valuation["psi"])
        self.assertTrue(valuation["note"], "未接入必须写明原因")

    def test_categorical_dimension_uses_concentration_not_psi(self):
        rows = {row["feature_name"]: row for row in feature_drift_rows(_panel(), current_days=20, reference_days=400)}
        industry = rows["行业分布"]
        self.assertIsNotNone(industry["psi"])
        self.assertIsNone(industry["ks_stat"], "行业是类别列，不做 KS")

    def test_industry_shift_is_detected(self):
        rows = {row["feature_name"]: row
                for row in feature_drift_rows(_panel(industry_shift=True), current_days=20, reference_days=400)}
        self.assertNotEqual(rows["行业分布"]["threshold_exceeded"], LEVEL_OK)

    def test_insufficient_samples_are_labelled_not_faked(self):
        rows = {row["feature_name"]: row
                for row in feature_drift_rows(_panel(days=40, stocks=3), current_days=5, reference_days=20)}
        self.assertEqual(rows["波动率"]["threshold_exceeded"], LEVEL_INSUFFICIENT)
        self.assertTrue(rows["波动率"]["note"])

    def test_windows_are_recorded_for_audit(self):
        rows = feature_drift_rows(_panel(), current_days=20, reference_days=400)
        numeric = next(row for row in rows if row["feature_name"] == "波动率")
        self.assertTrue(numeric["window_start"] and numeric["window_end"])
        self.assertTrue(numeric["reference_start"] and numeric["reference_end"])
        self.assertGreater(numeric["sample_size"], 0)

    def test_coverage_dimension_is_unavailable_without_extra_factor_columns(self):
        """面板没带扩展因子列时这一维不可计算，必须写明原因而不是从仪表盘上静静消失。"""
        rows = {row["feature_name"]: row for row in feature_drift_rows(_panel(), current_days=20, reference_days=400)}
        coverage = rows["财务因子覆盖率"]
        self.assertEqual(coverage["threshold_exceeded"], LEVEL_UNAVAILABLE)
        self.assertIn("未启用扩展因子", coverage["note"])

    def test_empty_panel_yields_no_rows(self):
        self.assertEqual(feature_drift_rows(pd.DataFrame()), [])


class ConceptDriftTests(unittest.TestCase):
    """滚动概念漂移与连续为负的判定。"""

    def _metrics(self, ics: list[float]) -> dict:
        return {"rank_ic": None, "ndcg_ratio": 1.1,
                "daily_groups": [{"rank_ic": value, "top_excess_return": value / 2} for value in ics]}

    def test_negative_streak_counts_only_the_tail(self):
        # 中间那次为正会把连续计数清零，只有尾部连续的负数才算「连续为负」。
        row = concept_drift_row(self._metrics([0.05] * 10 + [-0.01] * 6), "2026-03-01", window=20)
        self.assertEqual(row["negative_streak"], 6)
        self.assertEqual(row["window"], 16, "可用样本外交易日只有 16 天，窗口按实际可用量收窄")

    def test_positive_ic_resets_the_streak(self):
        row = concept_drift_row(self._metrics([-0.01] * 20 + [0.05]), "2026-03-01", window=20)
        self.assertEqual(row["negative_streak"], 0)

    def test_sustained_negative_window_reports_negative_rank_ic(self):
        row = concept_drift_row(self._metrics([0.01] * 10 + [-0.02] * 20), "2026-03-01", window=30)
        self.assertEqual(row["negative_streak"], 20)
        self.assertLess(row["rank_ic"], 0)
        self.assertLess(row["hac_t"], 0)

    def test_trend_is_computed_against_the_previous_window(self):
        improving = concept_drift_row(self._metrics([-0.02] * 30 + [0.08] * 30), "2026-03-01", window=30)
        worsening = concept_drift_row(self._metrics([0.08] * 30 + [-0.02] * 30), "2026-03-01", window=30)
        self.assertEqual(improving["trend"], "好转")
        self.assertEqual(worsening["trend"], "恶化")

    def test_ndcg_note_states_it_is_whole_period(self):
        row = concept_drift_row(self._metrics([0.05] * 30), "2026-03-01", window=30)
        self.assertIn("整个样本外区间", row["note"])

    def test_metrics_without_daily_groups_yields_nothing(self):
        self.assertIsNone(concept_drift_row(None, "2026-03-01"))
        self.assertIsNone(concept_drift_row({}, "2026-03-01"))


class PredictionDriftTests(unittest.TestCase):
    """Top-N 组合的集中度与分数分布。"""

    def _context(self, scores: dict[str, float], reference_date: str = "2026-10-16") -> dict:
        # 周期键与生产一致用 int：HORIZONS = (5, 20, 60)。
        return {
            "reference_date": reference_date,
            "predictions": {20: {code: {"LightGBM": {"predicted_excess_return": value}}
                                 for code, value in scores.items()}},
        }

    def _panel(self) -> pd.DataFrame:
        generator = np.random.default_rng(3)
        return pd.DataFrame([
            {"date": "2026-10-16", "code": f"{600000 + stock:06d}",
             "industry": "银行" if stock < 5 else "医药",
             "log_market_cap": float(generator.normal(23.0, 1.0))}
            for stock in range(10)
        ])

    def test_score_distribution_uses_raw_predictions(self):
        scores = {f"{600000 + stock:06d}": float(stock) for stock in range(10)}
        row = prediction_drift_row(self._context(scores), self._panel(), top_n=5)
        self.assertEqual(row["stocks"], 5)
        # Top-5 是分数最高的 5、6、7、8、9，均值 7。
        self.assertAlmostEqual(row["score_mean"], 7.0)
        self.assertGreater(row["score_std"], 0)

    def test_industry_concentration_is_herfindahl(self):
        """Top-4 刻意选成 3 只银行 + 1 只医药，HHI 应为 (3/4)² + (1/4)² = 0.625。"""
        scores = {"600000": 0.9, "600001": 0.8, "600002": 0.7, "600005": 0.6}
        scores.update({f"{600000 + stock:06d}": 0.1 for stock in (3, 4, 6, 7, 8, 9)})
        row = prediction_drift_row(self._context(scores), self._panel(), top_n=4)
        self.assertEqual(row["stocks"], 4)
        self.assertAlmostEqual(row["industry_hhi"], 0.625)

    def test_market_cap_quantile_is_relative_to_the_whole_cross_section(self):
        scores = {f"{600000 + stock:06d}": 1.0 - stock / 100 for stock in range(10)}
        row = prediction_drift_row(self._context(scores), self._panel(), top_n=3)
        self.assertIsNotNone(row["market_cap_quantile_mean"])
        self.assertGreater(row["market_cap_quantile_mean"], 0.0)
        self.assertLessEqual(row["market_cap_quantile_mean"], 1.0)

    def test_no_validated_model_is_recorded_as_unavailable(self):
        """没有任何通过验证的模型时如实标注，不编一组 0 出来。"""
        row = prediction_drift_row({"reference_date": "2026-10-16", "predictions": {20: {}}}, self._panel())
        self.assertEqual(row["hhi_threshold_exceeded"], LEVEL_UNAVAILABLE)
        self.assertTrue(row["note"])

    def test_panel_without_the_report_date_is_noted(self):
        scores = {f"{600000 + stock:06d}": 0.1 for stock in range(10)}
        shifted = prediction_drift_row(self._context(scores, "2026-10-15"), self._panel(), top_n=5)
        self.assertIsNone(shifted["industry_hhi"])
        self.assertIsNone(shifted["market_cap_quantile_mean"])
        self.assertIn("没有", shifted["note"])

    def test_context_without_reference_date_yields_nothing(self):
        self.assertIsNone(prediction_drift_row({"predictions": {20: {}}}, self._panel()))


class DriftAlertTests(unittest.TestCase):
    """三类漂移的告警阈值。"""

    def test_severe_feature_drift_is_critical(self):
        alerts = evaluate_drift_alerts(
            [{"feature_name": "波动率", "date": "2026-10-16", "psi": 0.4,
              "threshold_exceeded": LEVEL_SEVERE, "sample_size": 1000}], [], [])
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["level"], "critical")
        self.assertEqual(alerts[0]["category"], "drift")

    def test_watch_feature_drift_is_warning(self):
        alerts = evaluate_drift_alerts(
            [{"feature_name": "波动率", "date": "2026-10-16", "psi": 0.15,
              "threshold_exceeded": LEVEL_WATCH, "sample_size": 1000}], [], [])
        self.assertEqual(alerts[0]["level"], "warning")

    def test_normal_and_unavailable_do_not_alert(self):
        self.assertEqual(evaluate_drift_alerts([
            {"feature_name": "波动率", "date": "2026-10-16", "psi": 0.01, "threshold_exceeded": LEVEL_OK},
            {"feature_name": "估值", "date": "2026-10-16", "psi": None, "threshold_exceeded": LEVEL_UNAVAILABLE},
            {"feature_name": "换手率", "date": "2026-10-16", "psi": None,
             "threshold_exceeded": LEVEL_INSUFFICIENT},
        ], [], []), [])

    def test_alert_message_carries_the_reason_not_just_the_number(self):
        alerts = evaluate_drift_alerts(
            [{"feature_name": "波动率", "date": "2026-10-16", "psi": 0.4,
              "threshold_exceeded": LEVEL_SEVERE, "sample_size": 1000, "note": "样本不足的说明"}], [], [])
        self.assertIn("波动率", alerts[0]["message"])
        self.assertIn("样本不足的说明", alerts[0]["message"])

    def test_negative_streak_threshold(self):
        base = {"date": "2026-10-16", "rank_ic": -0.02, "hac_t": -1.0, "trend": "恶化"}
        below = evaluate_drift_alerts([], [{**base, "negative_streak": settings.DRIFT_IC_NEGATIVE_DAYS - 1}], [])
        self.assertEqual(below, [])
        at = evaluate_drift_alerts([], [{**base, "negative_streak": settings.DRIFT_IC_NEGATIVE_DAYS}], [])
        self.assertEqual(len(at), 1)
        self.assertEqual(at[0]["level"], "critical")
        self.assertIn("连续", at[0]["title"])

    def test_hhi_above_history_percentile_alerts(self):
        row = {"date": "2026-10-16", "top_n": 20, "industry_hhi": 0.42, "hhi_history_p95": 0.30,
               "hhi_threshold_exceeded": LEVEL_WATCH}
        alerts = evaluate_drift_alerts([], [], [row])
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["level"], "warning")
        self.assertIn("95", alerts[0]["title"])

    def test_hhi_normal_and_insufficient_do_not_alert(self):
        self.assertEqual(evaluate_drift_alerts([], [], [
            {"date": "2026-10-16", "top_n": 20, "industry_hhi": 0.2, "hhi_threshold_exceeded": LEVEL_OK},
            {"date": "2026-10-16", "top_n": 20, "industry_hhi": 0.9,
             "hhi_threshold_exceeded": LEVEL_INSUFFICIENT},
            {"date": "2026-10-16", "top_n": 20, "industry_hhi": None,
             "hhi_threshold_exceeded": LEVEL_UNAVAILABLE},
        ]), [])


if __name__ == "__main__":
    unittest.main()
