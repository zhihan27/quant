"""新增因子族并入排序特征矩阵：覆盖率门控与自动装配。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from config import settings
from services.chip_factors import CHIP_FACTOR_COLUMNS
from services.extra_factors import (
    EXTRA_FACTOR_FAMILIES, HEALTH_VETO_COLUMN, accepted_extra_columns, apply_health_veto,
    attach_extra_factors, build_extra_factors, factor_correlation, summarize_extra_factors,
)
from services.financial_factors import FINANCIAL_FACTOR_COLUMNS
from services.intraday_factors import INTRADAY_FACTOR_COLUMNS, store_intraday_bars
from services.panel_store import PanelStore

from tests.test_intraday_factors import _decaying, _minutes


def _panel(rows: int = 10) -> pd.DataFrame:
    dates = pd.date_range("2026-01-05", periods=rows, freq="B")
    return pd.DataFrame({"date": dates, "code": ["600519"] * rows, "close": np.linspace(10.0, 11.0, rows)})


def _history(bars: int = 200) -> pd.DataFrame:
    """构造满足筹码分布最短历史要求（120 根）的日线。"""
    dates = pd.date_range("2025-01-02", periods=bars, freq="B")
    closes = 10.0 + np.sin(np.arange(bars) / 7.0) * 0.5
    frame = pd.DataFrame({"date": dates, "close": closes, "volume": 1000.0})
    frame["open"] = frame["close"]
    frame["high"] = frame["close"] + 0.2
    frame["low"] = frame["close"] - 0.2
    frame["amount"] = frame["close"] * frame["volume"]
    frame["turnover_rate"] = 1.0
    frame.attrs["turnover_rate_unit"] = "percent"
    return frame


class CoverageGateTests(unittest.TestCase):
    def test_family_above_threshold_is_accepted(self) -> None:
        panel = _panel()
        panel["chip_cost_range_width"] = 0.1
        panel["chip_winner_rate"] = 0.5
        accepted, decisions = accepted_extra_columns(panel, families={"chip": tuple(CHIP_FACTOR_COLUMNS)})
        self.assertEqual(accepted, list(CHIP_FACTOR_COLUMNS))
        self.assertTrue(decisions["chip"]["accepted"])

    def test_family_below_threshold_is_rejected(self) -> None:
        """覆盖率不达标时整族不参与训练，而不是把大量缺失值塞进特征矩阵。"""
        panel = _panel(rows=10)
        panel["chip_cost_range_width"] = [0.1] * 4 + [np.nan] * 6
        panel["chip_winner_rate"] = [0.5] * 4 + [np.nan] * 6
        accepted, decisions = accepted_extra_columns(panel, families={"chip": tuple(CHIP_FACTOR_COLUMNS)})
        self.assertEqual(accepted, [])
        self.assertFalse(decisions["chip"]["accepted"])
        self.assertAlmostEqual(decisions["chip"]["coverage"], 0.4, places=9)

    def test_threshold_boundary_is_inclusive(self) -> None:
        panel = _panel(rows=10)
        for column in CHIP_FACTOR_COLUMNS:
            panel[column] = [1.0] * 5 + [np.nan] * 5
        accepted, _ = accepted_extra_columns(
            panel, families={"chip": tuple(CHIP_FACTOR_COLUMNS)}, minimum_coverage=0.5
        )
        self.assertEqual(accepted, list(CHIP_FACTOR_COLUMNS))

    def test_family_needs_every_column_not_just_one(self) -> None:
        """只覆盖到半列不算覆盖，必须整族齐全。"""
        panel = _panel()
        panel["chip_cost_range_width"] = 0.1
        accepted, decisions = accepted_extra_columns(panel, families={"chip": tuple(CHIP_FACTOR_COLUMNS)})
        self.assertEqual(accepted, [])
        self.assertFalse(decisions["chip"]["accepted"])

    def test_absent_family_is_reported_without_error(self) -> None:
        accepted, decisions = accepted_extra_columns(_panel())
        self.assertEqual(accepted, [])
        for name in EXTRA_FACTOR_FAMILIES:
            self.assertFalse(decisions[name]["accepted"])

    def test_empty_panel_is_not_accepted(self) -> None:
        accepted, decisions = accepted_extra_columns(pd.DataFrame(columns=["date", "code"]))
        self.assertEqual(accepted, [])
        self.assertFalse(decisions["chip"]["accepted"])

    def test_default_threshold_comes_from_settings(self) -> None:
        panel = _panel(rows=10)
        for column in CHIP_FACTOR_COLUMNS:
            panel[column] = [1.0] * 5 + [np.nan] * 5
        _, decisions = accepted_extra_columns(panel, families={"chip": tuple(CHIP_FACTOR_COLUMNS)})
        self.assertEqual(decisions["chip"]["accepted"], settings.EXTRA_FACTOR_MINIMUM_COVERAGE <= 0.5)


class AttachTests(unittest.TestCase):
    def test_empty_panel_short_circuits(self) -> None:
        panel = pd.DataFrame(columns=["date", "code"])
        result, report = attach_extra_factors(panel, {"chip": {}})
        self.assertEqual(report["families"], {})
        self.assertTrue(result.empty)

    def test_no_families_reports_untouched(self) -> None:
        result, report = attach_extra_factors(_panel(), None)
        self.assertEqual(report["status"], "未提供新增因子")

    def test_chip_family_is_attached_and_accepted(self) -> None:
        history = _history()
        panel = pd.DataFrame(
            {"date": history["date"].tail(10).to_numpy(), "code": ["600519"] * 10, "close": 10.0}
        )
        result, report = attach_extra_factors(panel, {"chip": {"600519": history}})
        self.assertTrue(report["decisions"]["chip"]["accepted"])
        self.assertIn("chip_cost_range_width", report["accepted_columns"])
        self.assertTrue(result["chip_cost_range_width"].notna().all())

    def test_intraday_family_outside_its_dates_stays_empty(self) -> None:
        """日内因子的日期与截面不重叠时不并入，门控随后会拒绝该族。"""
        factors = pd.DataFrame(
            {
                "date": [pd.Timestamp("2025-06-02")],
                "code": ["600519"],
                **{column: [0.5] for column in INTRADAY_FACTOR_COLUMNS},
            }
        )
        result, report = attach_extra_factors(_panel(), {"intraday": factors})
        self.assertFalse(report["decisions"]["intraday"]["accepted"])
        self.assertTrue(result[INTRADAY_FACTOR_COLUMNS[0]].isna().all())


class HealthVetoTests(unittest.TestCase):
    """财务健康一票否决：只否决显式判定为困境的行，未知不等于不健康。"""

    def _panel(self, flags: list) -> pd.DataFrame:
        rows = len(flags)
        return pd.DataFrame(
            {
                "date": pd.bdate_range("2026-01-05", periods=rows),
                "code": [f"{index:06d}" for index in range(rows)],
                "close": 10.0,
                HEALTH_VETO_COLUMN: flags,
            }
        )

    def test_only_explicit_true_is_vetoed(self) -> None:
        frame, report = apply_health_veto(self._panel([True, False, None, np.nan]))
        self.assertEqual(list(frame["code"]), ["000001", "000002", "000003"])
        self.assertEqual(report["vetoed"], 1)
        self.assertTrue(report["applied"])

    def test_report_counts_codes_and_dates(self) -> None:
        frame = self._panel([True, True, False])
        frame["code"] = ["600519", "600519", "000001"]
        _, report = apply_health_veto(frame)
        self.assertEqual(report["vetoed"], 2)
        self.assertEqual(report["codes"], 1)
        self.assertEqual(report["dates"], 2)
        self.assertAlmostEqual(report["veto_rate"], 2 / 3, places=9)

    def test_panel_without_the_flag_is_untouched(self) -> None:
        frame = self._panel([True, False]).drop(columns=[HEALTH_VETO_COLUMN])
        result, report = apply_health_veto(frame)
        self.assertEqual(len(result), 2)
        self.assertEqual(report["vetoed"], 0)
        self.assertIn("没有财务健康标记", report["status"])

    def test_nothing_to_veto_reports_zero(self) -> None:
        result, report = apply_health_veto(self._panel([False, False]))
        self.assertEqual(len(result), 2)
        self.assertEqual(report["veto_rate"], 0.0)
        self.assertNotIn("applied", report)

    def test_veto_that_would_empty_the_panel_returns_no_candidates(self) -> None:
        """全部否决时必须返回空候选池，不能把高风险标的重新放行。"""
        result, report = apply_health_veto(self._panel([True, True]))
        self.assertTrue(result.empty)
        self.assertTrue(report["applied"])
        self.assertEqual(report["vetoed"], 2)
        self.assertIn("无可交易面板", report["status"])

    def test_attrs_survive_the_trim(self) -> None:
        frame = self._panel([True, False])
        frame.attrs["model_feature_columns"] = ["xs_momentum_20"]
        result, _ = apply_health_veto(frame)
        self.assertEqual(result.attrs["model_feature_columns"], ["xs_momentum_20"])


class SummaryTests(unittest.TestCase):
    def test_summary_reports_per_family_coverage(self) -> None:
        panel = _panel(rows=4)
        for column in CHIP_FACTOR_COLUMNS:
            panel[column] = [1.0, 1.0, np.nan, np.nan]
        summary = summarize_extra_factors(panel)
        self.assertEqual(summary["rows"], 4)
        self.assertAlmostEqual(summary["chip"]["coverage"], 0.5, places=9)
        self.assertEqual(summary["chip"]["rows"], 2)
        self.assertEqual(summary["intraday"]["rows"], 0)

    def test_summary_on_empty_panel(self) -> None:
        summary = summarize_extra_factors(pd.DataFrame(columns=["date", "code"]))
        self.assertEqual(summary["rows"], 0)
        self.assertEqual(summary["financial"]["coverage"], 0.0)


class CorrelationTests(unittest.TestCase):
    def test_identical_series_correlate_fully(self) -> None:
        panel = _panel(rows=30)
        panel["base"] = np.linspace(-1.0, 1.0, 30)
        panel["copy"] = panel["base"]
        panel["reversed"] = -panel["base"]
        scores = factor_correlation(panel, ["copy", "reversed"], "base")
        self.assertAlmostEqual(scores["copy"], 1.0, places=9)
        self.assertAlmostEqual(scores["reversed"], -1.0, places=9)

    def test_missing_reference_returns_empty(self) -> None:
        self.assertEqual(factor_correlation(_panel(), ["close"], "not_here"), {})


class BuildTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = PanelStore(root=Path(tempfile.mkdtemp(prefix="extra_test_")))

    def test_no_local_data_reports_honestly(self) -> None:
        collected, report = build_extra_factors(self.store, ["600519"], families={"intraday", "financial"})
        self.assertEqual(collected, {})
        self.assertIn("本地暂无分钟线", report["intraday"]["status"])
        self.assertIn("尚无 PIT 财务面板", report["financial"]["status"])

    def test_intraday_is_assembled_from_local_bars(self) -> None:
        store_intraday_bars(self.store, "600519", _minutes(_decaying(), day="2026-01-05"))
        collected, report = build_extra_factors(self.store, ["600519"], families={"intraday"})
        self.assertIn("intraday", collected)
        self.assertEqual(report["intraday"]["codes_computed"], 1)

    def test_chip_uses_histories_without_network(self) -> None:
        histories = {"600519": _history()}
        collected, report = build_extra_factors(
            self.store, ["600519"], histories=histories, families={"chip"}
        )
        self.assertIs(collected["chip"], histories)
        self.assertEqual(report["chip"]["codes"], 1)

    def test_unrequested_family_is_not_collected(self) -> None:
        collected, _ = build_extra_factors(
            self.store, ["600519"], histories={"600519": _history()}, families={"intraday"}
        )
        self.assertNotIn("chip", collected)

    def test_status_lists_assembled_families(self) -> None:
        collected, report = build_extra_factors(
            self.store, ["600519"], histories={"600519": _history()}, families={"chip"}
        )
        self.assertIn("chip", report["status"])


if __name__ == "__main__":
    unittest.main()
