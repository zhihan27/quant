"""验证最终留出集：区间过滤、冻结模型只用留出集之前的数据、换手率与费用口径。

全部使用构造面板，不依赖真实行情。构造的关键是一根**探针特征列**：它的值约等于对应
周期的标签，注入的估计器直接把它当预测输出。这样冻结模型能稳定通过训练期验证，
断言就可以集中在真正要验的东西上——留出集的数据到底有没有漏进训练，
评估帧到底有没有跑出留出集区间。探针只存在于本文件的构造数据里，不是产品行为。
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import pandas as pd

from config import settings
from services import model_governance
from services.backtest import cost_rates
from services.ml_pipeline import HORIZONS, MINIMUM_RANK_IC, fit_panel_model as real_fit_panel_model
from services.model_governance import (
    FinalHoldoutError,
    build_holdout_evaluation,
    evaluate_final_holdout,
    holdout_window,
    portfolio_turnover,
    training_baseline,
)

START = pd.Timestamp(settings.FINAL_HOLDOUT_START)
TRAIN_DATES = 380
HOLDOUT_DATES = 120
CODES = 20
# 换手率与费用后收益固定的主周期，与 services/model_governance.py::TURNOVER_HORIZON 一致。
TURNOVER_HORIZON = model_governance.TURNOVER_HORIZON


def _panel(train_dates: int = TRAIN_DATES, holdout_dates: int = HOLDOUT_DATES) -> pd.DataFrame:
    generator = np.random.default_rng(2026)
    dates = [*pd.bdate_range(end=START - pd.Timedelta(days=1), periods=train_dates),
             *pd.bdate_range(start=START, periods=holdout_dates)]
    rows = []
    for date in dates:
        for index in range(CODES):
            row = {"date": date, "code": f"{600000 + index:06d}"}
            for horizon in HORIZONS:
                target = float(generator.normal(0.0, 0.05))
                row[f"target_{horizon}"] = target
                row[f"label_end_{horizon}"] = date + pd.tseries.offsets.BDay(horizon)
                row[f"forward_return_{horizon}"] = target + 0.01
                row[f"probe_{horizon}"] = target + float(generator.normal(0.0, 0.002))
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.attrs["model_feature_columns"] = [f"probe_{horizon}" for horizon in HORIZONS]
    return frame


class _ProbeEstimator:
    """把探针列当预测输出的假估计器，同时记录每一次训练与预测用到的行。"""

    def __init__(self, horizon: int, record: list[tuple[str, int, list[int]]]):
        self.horizon, self.record = horizon, record

    def fit(self, features: pd.DataFrame, _target: pd.Series):
        self.record.append(("fit", self.horizon, list(features.index)))
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        self.record.append(("predict", self.horizon, list(features.index)))
        return features[f"probe_{self.horizon}"].to_numpy(dtype=float)


def _run_holdout(panel: pd.DataFrame, context: dict | None = None):
    """跑一次留出集评估，返回报告与「哪一次训练/预测用了哪些行」的记录。"""
    record: list[tuple[str, int, list[int]]] = []

    def stub(frame, name, horizon, model_factory=None, cutoff=None, window=None, objective=None):
        return real_fit_panel_model(
            frame, name, horizon, model_factory=lambda _name: _ProbeEstimator(horizon, record),
            cutoff=cutoff, window=window, objective=objective,
        )

    with mock.patch.object(model_governance, "fit_panel_model", stub):
        report = build_holdout_evaluation(
            panel, context or {"objective_mode": "lambdarank", "training_window_days": settings.TRAINING_WINDOW_DAYS}
        )
    return report, record


def _last(record, kind: str, horizon: int) -> list[int]:
    return next(indices for entry_kind, entry_horizon, indices in reversed(record)
                if entry_kind == kind and entry_horizon == horizon)


class HoldoutFilterTests(unittest.TestCase):
    """需求 F-1：冻结模型的每次训练都不能碰到留出集，评估帧不能跑出留出集区间。"""

    @classmethod
    def setUpClass(cls):
        cls.panel = _panel()
        cls.report, cls.record = _run_holdout(cls.panel)

    def test_report_marks_the_configured_window(self):
        self.assertEqual(self.report["holdout_start"], str(START.date()))
        self.assertEqual(self.report["holdout_end"], str(self.panel["date"].max().date()))
        self.assertEqual(self.report["holdout_dates"], HOLDOUT_DATES)

    def test_training_never_sees_holdout_dates(self):
        fits = [entry for entry in self.record if entry[0] == "fit"]
        self.assertTrue(fits, "至少要真的训练过")
        for _kind, horizon, indices in fits:
            dates = self.panel.loc[indices, "date"]
            self.assertLess(dates.max(), START, f"{horizon} 日模型的训练样本落进了留出集")

    def test_training_never_sees_holdout_labels(self):
        """标签跨进留出集同样算漏数据：训练期用到的每个标签结束日都必须早于留出集起点。"""
        for _kind, horizon, indices in [entry for entry in self.record if entry[0] == "fit"]:
            endpoints = self.panel.loc[indices, f"label_end_{horizon}"]
            self.assertLess(endpoints.max(), START, f"{horizon} 日模型的训练标签跨进了留出集")

    def test_evaluation_frame_covers_exactly_the_holdout_window(self):
        for horizon in HORIZONS:
            predicted = _last(self.record, "predict", horizon)
            expected = self.panel.loc[self.panel["date"] >= START]
            self.assertEqual(set(predicted), set(expected.index), f"{horizon} 日评估帧不只是留出集")
            dates = self.panel.loc[predicted, "date"]
            self.assertGreaterEqual(dates.min(), START)
            self.assertLessEqual(dates.max(), self.panel["date"].max())

    def test_frozen_model_metrics_are_reported_per_horizon(self):
        self.assertTrue(self.report["passed"])
        self.assertIsNone(self.report["failed_horizon"])
        for horizon in HORIZONS:
            line = self.report["horizons"][str(horizon)]
            self.assertTrue(line["usable"])
            self.assertGreater(line["metrics"]["rank_ic"], MINIMUM_RANK_IC)
            self.assertEqual(line["metrics"]["dates"], HOLDOUT_DATES)

    def test_report_states_the_training_window_is_short_of_configuration(self):
        """留出集之前的可训练历史不足配置窗口时必须如实写明，不能只说「已通过」。"""
        line = self.report["horizons"]["20"]
        self.assertLess(line["training"]["available_dates"], settings.TRAINING_WINDOW_DAYS)
        self.assertFalse(line["training"]["window_sufficient"])
        self.assertEqual(line["training"]["window_days"], settings.TRAINING_WINDOW_DAYS)
        self.assertTrue(line["training"]["last_fold_through"] < str(START.date()))

    def test_configured_end_trims_the_holdout(self):
        end = str((START + pd.Timedelta(days=100)).date())
        with mock.patch.object(settings, "FINAL_HOLDOUT_END", end):
            report, record = _run_holdout(self.panel)
        self.assertEqual(report["holdout_end"], end)
        # 设了终点时标签也必须落在终点之内，否则会把终点之后的行情算进「留出集」。
        expected = self.panel.loc[(self.panel["date"] >= START) & (self.panel["label_end_20"] <= pd.Timestamp(end))]
        self.assertEqual(set(_last(record, "predict", 20)), set(expected.index))
        self.assertLess(len(expected), len(self.panel.loc[self.panel["date"] >= START]))

    def test_short_horizon_is_reported_as_not_passing_not_silently_dropped(self):
        """终点设得早时 60 日的标签几乎全跨出终点，评估只能给出「没通过」，不能拿别的周期顶上。"""
        end = str((START + pd.Timedelta(days=100)).date())
        with mock.patch.object(settings, "FINAL_HOLDOUT_END", end):
            report, record = _run_holdout(self.panel)
        self.assertFalse(report["passed"])
        self.assertEqual(report["failed_horizon"], "60")
        line = report["horizons"]["60"]
        self.assertTrue(line["usable"], "还有少量样本可评估，就应当如实给出指标")
        self.assertFalse(line["metrics"]["validated"])
        expected = self.panel.loc[(self.panel["date"] >= START) & (self.panel["label_end_60"] <= pd.Timestamp(end))]
        self.assertEqual(set(_last(record, "predict", 60)), set(expected.index))

    def test_horizon_without_usable_holdout_samples_is_labelled_not_forced(self):
        end = str((START + pd.Timedelta(days=45)).date())
        with mock.patch.object(settings, "FINAL_HOLDOUT_END", end):
            report, _record = _run_holdout(self.panel)
        line = report["horizons"]["60"]
        self.assertFalse(line["usable"], "60 日的标签全部跨出终点，没有样本可评估")
        self.assertIn("样本不足", line["status"])
        self.assertIsNone(line["metrics"])

    def test_unlimited_window_is_not_silently_replaced_by_the_default(self):
        """窗口 0 的含义是「用上全部历史」，不能被当成空值换成 750。"""
        report, _record = _run_holdout(self.panel, {"objective_mode": "lambdarank", "training_window_days": 0})
        self.assertEqual(report["training_window_days"], 0)
        self.assertTrue(report["horizons"]["20"]["training"]["window_sufficient"])


class TurnoverTests(unittest.TestCase):
    """需求 A：换手率与费用后收益的口径。纯函数，构造成员变更即可对账。"""

    def _evaluation(self, periods: int = 2) -> pd.DataFrame:
        """每期 10 只股票，Top 取前 1/5 即 2 只；第二期把 B 换成 C，单边换手 0.5。

        逐日相邻地看会把同一段持仓期数成多次调仓，因此这里必须铺满日历日：
        20 个交易日一格，取样点才是 0、20、40……，中间那些日子不该参与换手率。
        """
        dates = pd.bdate_range(start="2026-01-01", periods=20 * (periods - 1) + 1)
        rows = []
        for index, date in enumerate(dates):
            period = index // 20
            sampled = index % 20 == 0
            members = ("A", "B") if period == 0 else ("A", "C")
            for code in "ABCDEFGHIJ":
                rows.append({
                    "date": date,
                    "code": code,
                    "prediction": 1.0 if (sampled and code in members) else 0.0,
                    "actual": (0.01 if period == 0 else 0.02) if (sampled and code in members) else 0.0,
                })
        return pd.DataFrame(rows)

    def test_turnover_and_cost_match_hand_computed_values(self):
        buy, sell = cost_rates()
        result = portfolio_turnover(self._evaluation(), TURNOVER_HORIZON)
        self.assertEqual(result["status"], "已计算")
        self.assertEqual(result["periods"], 2)
        self.assertAlmostEqual(result["average_turnover"], 0.5, msg="首期换手记 0 不算进平均")
        self.assertAlmostEqual(result["per_period_cost"], 0.5 * (buy + sell))
        self.assertAlmostEqual(result["gross_excess_return"], (0.01 + 0.02) / 2)
        self.assertAlmostEqual(result["after_cost_return"], (0.01 + 0.02) / 2 - 0.5 * (buy + sell))
        self.assertEqual(result["cost_rates"], {"buy": buy, "sell": sell})
        self.assertAlmostEqual(result["annualized_turnover"], 0.5 * 252 / TURNOVER_HORIZON)

    def test_unchanged_members_are_zero_turnover(self):
        frame = self._evaluation()
        latest = frame["date"].eq(frame["date"].max())
        # 第二期不换人：B 留下、C 挪出 Top。
        frame.loc[latest & frame["code"].eq("B"), ["prediction", "actual"]] = [1.0, 0.02]
        frame.loc[latest & frame["code"].eq("C"), ["prediction", "actual"]] = [0.0, 0.0]
        result = portfolio_turnover(frame, TURNOVER_HORIZON)
        self.assertAlmostEqual(result["average_turnover"], 0.0)
        self.assertAlmostEqual(result["per_period_cost"], 0.0)
        self.assertAlmostEqual(result["after_cost_return"], result["gross_excess_return"])

    def test_single_period_reports_insufficient_not_zero_turnover(self):
        result = portfolio_turnover(self._evaluation(periods=1), TURNOVER_HORIZON)
        self.assertEqual(result["periods"], 1)
        self.assertIn("不足", result["status"])
        self.assertNotIn("after_cost_return", result)

    def test_empty_evaluation_is_reported_as_such(self):
        self.assertIn("没有可用样本", portfolio_turnover(pd.DataFrame(), TURNOVER_HORIZON)["status"])


class WindowAndBaselineTests(unittest.TestCase):
    """留出集区间与训练基线的取数口径。"""

    def test_end_defaults_to_the_latest_cross_section(self):
        panel = _panel(holdout_dates=30)
        _start, end = holdout_window(panel)
        self.assertEqual(end, panel["date"].max())

    def test_configured_end_is_used(self):
        with mock.patch.object(settings, "FINAL_HOLDOUT_END", "2024-06-28"):
            _start, end = holdout_window(_panel(holdout_dates=10))
        self.assertEqual(end, pd.Timestamp("2024-06-28"))

    def test_empty_window_is_refused(self):
        with mock.patch.object(settings, "FINAL_HOLDOUT_END", "2023-12-29"):
            with self.assertRaises(FinalHoldoutError):
                holdout_window(_panel(holdout_dates=10))

    def test_baseline_uses_only_pre_holdout_labelled_samples(self):
        panel = _panel()
        # 人为把一段留出集样本的标签改成极值：基线取错窗口时会立刻偏离。
        panel.loc[panel["date"] >= START, "target_20"] = 99.0
        expected = float(panel.loc[(panel["date"] < START) & (panel["label_end_20"] < START), "target_20"].median())
        self.assertAlmostEqual(training_baseline(panel, START, "target_20", "label_end_20"), expected)
        self.assertLess(abs(expected), 1.0)

    def test_baseline_excludes_labels_crossing_into_the_holdout(self):
        panel = _panel()
        # 只留下跨进留出集的那些样本，此时没有任何可用标签。
        crossed = panel.loc[(panel["date"] < START) & (panel["label_end_20"] >= START)]
        self.assertTrue(len(crossed) > 0, "构造数据本身要含跨期样本，否则这条断言是空的")
        with self.assertRaises(FinalHoldoutError):
            training_baseline(crossed, START, "target_20", "label_end_20")

    def test_baseline_refuses_without_any_pre_holdout_sample(self):
        with self.assertRaises(FinalHoldoutError):
            training_baseline(_panel().loc[lambda frame: frame["date"] >= START], START, "target_20", "label_end_20")


class RefusalTests(unittest.TestCase):
    """留出集评估的失败姿态：宁可报错退出，也不留下一份看起来通过的空报告。"""

    def test_empty_panel_is_refused(self):
        with self.assertRaises(FinalHoldoutError):
            build_holdout_evaluation(pd.DataFrame())

    def test_panel_without_holdout_rows_is_refused(self):
        panel = _panel(holdout_dates=0)
        with self.assertRaises(FinalHoldoutError):
            build_holdout_evaluation(panel)

    def test_single_holdout_day_is_refused(self):
        with self.assertRaises(FinalHoldoutError):
            build_holdout_evaluation(_panel(holdout_dates=1))

    def test_panel_without_pre_holdout_history_is_refused(self):
        """留出集之前没有可训练历史时，冻结模型训练不出来，必须整体报错而不是出一份空报告。"""
        panel = _panel(train_dates=0)
        with self.assertRaises(FinalHoldoutError) as context:
            build_holdout_evaluation(panel)
        self.assertIn("没有任何周期的冻结模型", str(context.exception))

    def test_disabled_switch_refuses_before_touching_the_database(self):
        with mock.patch.object(settings, "ENABLE_FINAL_HOLDOUT", False):
            with self.assertRaises(FinalHoldoutError):
                evaluate_final_holdout(mock.Mock(), panel=_panel())


if __name__ == "__main__":
    unittest.main()
