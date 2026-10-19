"""PIT 财务因子的口径与无未来函数约束。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from config import settings
from services.financial_factors import (
    FINANCIAL_FACTOR_COLUMNS, altman_z, attach_financial_factors, compute_financial_factors,
    health_flag, quarter_end_grid, to_single_quarter,
)

QUARTER_ENDS = [
    "2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31",
    "2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31",
]


def _facts(code: str = "600000", ann_dates: dict[str, str] | None = None) -> pd.DataFrame:
    """构造 8 个报告期的财务长表：利润表与现金流量表按累计口径，资产负债表按时点口径。

    单季营收 100、单季扣非净利 10、单季归母净利 12、单季经营现金流 18、单季利润总额 14、
    单季财务费用 1；资产负债表各项在全部报告期保持恒定，便于手算校验。
    """
    rows = []
    for position, text in enumerate(QUARTER_ENDS):
        within_year = position % 4 + 1
        period = pd.Timestamp(text)
        ann = pd.Timestamp((ann_dates or {}).get(text, (period + pd.Timedelta(days=30)).strftime("%Y-%m-%d")))
        rows.append(
            {
                "code": code,
                "report_period": period,
                "report_kind": "annual" if within_year == 4 else ("q1" if within_year == 1 else ("semi" if within_year == 2 else "q3")),
                "ann_date": ann,
                "value_available_at": ann,
                "TOTAL_OPERATE_INCOME": 100.0 * within_year,
                "DEDUCT_PARENT_NETPROFIT": 10.0 * within_year,
                "PARENT_NETPROFIT": 12.0 * within_year,
                "NETCASH_OPERATE": 18.0 * within_year,
                "TOTAL_PROFIT": 14.0 * within_year,
                "FINANCE_EXPENSE": 1.0 * within_year,
                "TOTAL_ASSETS": 1000.0,
                "TOTAL_LIABILITIES": 500.0,
                "TOTAL_EQUITY": 500.0,
                "TOTAL_PARENT_EQUITY": 500.0,
                "TOTAL_CURRENT_ASSETS": 400.0,
                "TOTAL_CURRENT_LIAB": 200.0,
                "UNASSIGN_RPOFIT": 150.0,
            }
        )
    return pd.DataFrame(rows)


class SingleQuarterTests(unittest.TestCase):
    def test_first_quarter_keeps_cumulative_value(self) -> None:
        index = pd.to_datetime(["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"])
        single = to_single_quarter(pd.Series([10.0, 30.0, 60.0, 100.0], index=index))
        self.assertAlmostEqual(single.iloc[0], 10.0)
        self.assertAlmostEqual(single.iloc[1], 20.0)
        self.assertAlmostEqual(single.iloc[2], 30.0)
        self.assertAlmostEqual(single.iloc[3], 40.0)

    def test_missing_previous_period_does_not_impersonate_single_quarter(self) -> None:
        """缺上一报告期时留空，不用本期累计值冒充单季，否则会把半年利润当成一个季度。"""
        index = pd.to_datetime(["2024-03-31", "2024-09-30"])
        single = to_single_quarter(pd.Series([10.0, 60.0], index=index))
        self.assertAlmostEqual(single.iloc[0], 10.0)
        self.assertTrue(pd.isna(single.iloc[1]))


class QuarterGridTests(unittest.TestCase):
    def test_grid_covers_every_quarter_between_bounds(self) -> None:
        grid = quarter_end_grid(pd.Timestamp("2023-06-30"), pd.Timestamp("2024-06-30"))
        self.assertEqual(
            [stamp.strftime("%Y-%m-%d") for stamp in grid],
            ["2023-06-30", "2023-09-30", "2023-12-31", "2024-03-31", "2024-06-30"],
        )

    def test_reversed_bounds_return_empty(self) -> None:
        self.assertEqual(quarter_end_grid(pd.Timestamp("2024-06-30"), pd.Timestamp("2023-06-30")), [])


class FactorValueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.factors = compute_financial_factors(_facts())
        self.last = self.factors.iloc[-1]

    def test_missing_value_revision_time_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "修订可得时间"):
            compute_financial_factors(_facts().drop(columns=["value_available_at"]))

    def test_effective_date_uses_value_revision_time(self) -> None:
        facts = _facts()
        revised_at = pd.Timestamp("2025-02-15")
        facts.loc[facts["report_period"].eq(pd.Timestamp("2024-12-31")), "value_available_at"] = revised_at
        factors = compute_financial_factors(facts)
        final = factors.loc[factors["report_period"].eq(pd.Timestamp("2024-12-31"))].iloc[0]
        self.assertEqual(pd.Timestamp(final["available_from"]), revised_at)

    def test_ttm_roe_uses_deducted_profit_and_average_equity(self) -> None:
        # 单季扣非 10 连续四季 → TTM 40；归母净资产恒为 500 → 平均 500。
        self.assertAlmostEqual(self.last["fundamental_roe_ttm_deducted"], 40.0 / 500.0, places=6)

    def test_ocf_to_profit_ratio(self) -> None:
        # TTM 经营现金流 72 / TTM 归母净利 48 = 1.5
        self.assertAlmostEqual(self.last["fundamental_ocf_to_profit"], 1.5, places=6)

    def test_flat_series_produces_zero_growth(self) -> None:
        self.assertAlmostEqual(self.last["fundamental_revenue_yoy"], 0.0, places=9)
        self.assertAlmostEqual(self.last["fundamental_deducted_profit_yoy"], 0.0, places=9)
        self.assertAlmostEqual(self.last["fundamental_deducted_profit_yoy_accel"], 0.0, places=9)

    def test_altman_z_matches_hand_computation(self) -> None:
        # X1=(400-200)/1000=0.2  X2=150/1000=0.15  X3=(14+1)*4/1000=0.06  X4=500/500=1.0
        expected = 6.56 * 0.2 + 3.26 * 0.15 + 6.72 * 0.06 + 1.05 * 1.0
        self.assertAlmostEqual(self.last["fundamental_altman_z"], expected, places=9)

    def test_healthy_company_is_not_vetoed(self) -> None:
        self.assertEqual(self.last["fundamental_health_flag"], "safe")
        self.assertFalse(bool(self.last["fundamental_health_veto"]))

    def test_all_factor_columns_present(self) -> None:
        for column in FINANCIAL_FACTOR_COLUMNS:
            self.assertIn(column, self.factors.columns)

    def test_insufficient_history_yields_no_factors(self) -> None:
        short = _facts().head(int(settings.FUNDAMENTAL_MINIMUM_QUARTERS) - 1)
        result = compute_financial_factors(short)
        self.assertTrue(result[FINANCIAL_FACTOR_COLUMNS].isna().all().all())


class FinancialIndustryShapeTests(unittest.TestCase):
    """金融行业报表的形状：银行／保险／证券的资产负债表没有流动与非流动的划分，利润表没有财务费用。

    缺的这两个科目正是 Altman Z'' 的 X1 与 X3，所以金融股的 Z 必须留空——Z'' 本就不适用于金融企业，
    算一个不能解释的数比留空更糟。但经营现金流与权益类科目都在，盈利质量与成长因子必须照常算出来，
    不能因为 Z 算不了就让整只股票在财务因子上落空。
    """

    def _bank_facts(self, code: str = "000001") -> pd.DataFrame:
        """把通用企业报表去掉金融行业没有的三个科目，即数据源里银行模板的实际形状。"""
        return _facts(code).drop(
            columns=["TOTAL_CURRENT_ASSETS", "TOTAL_CURRENT_LIAB", "FINANCE_EXPENSE"]
        )

    def test_bank_has_no_altman_z(self) -> None:
        factors = compute_financial_factors(self._bank_facts())
        self.assertTrue(factors["fundamental_altman_z"].isna().all())

    def test_bank_still_gets_quality_and_growth_factors(self) -> None:
        factors = compute_financial_factors(self._bank_facts())
        for column in (
            "fundamental_roe_ttm_deducted", "fundamental_ocf_to_profit", "fundamental_revenue_yoy",
            "fundamental_deducted_profit_yoy", "fundamental_deducted_profit_yoy_accel",
        ):
            with self.subTest(column=column):
                self.assertTrue(factors[column].notna().any(), f"{column} 对银行不应整列为空")

    def test_missing_z_alone_never_vetoes(self) -> None:
        """算不出来的 Z 是「未知」而不是「困境」：不能因为缺科目就把金融股一票否决。"""
        factors = compute_financial_factors(self._bank_facts())
        self.assertFalse(factors["fundamental_health_veto"].fillna(False).astype(bool).any())

    def test_general_company_still_gets_a_z(self) -> None:
        """对照：通用企业模板下 Z'' 必须算得出来，否则这条测试挡不住「一律留空」的退化解。"""
        self.assertTrue(compute_financial_factors(_facts())["fundamental_altman_z"].notna().any())


class HealthFlagTests(unittest.TestCase):
    def test_low_z_is_distress(self) -> None:
        z = pd.Series([0.5, 2.0, 3.5])
        flag, veto = health_flag(z, pd.Series([np.nan, np.nan, np.nan]))
        self.assertEqual(list(flag), ["distress", "grey", "safe"])
        self.assertEqual(list(veto), [True, False, False])

    def test_negative_operating_cashflow_forces_distress(self) -> None:
        """利润为正但经营现金流为负时，即使 Z 值落在安全区也判为困境。"""
        z = pd.Series([3.5])
        flag, veto = health_flag(z, pd.Series([-0.2]))
        self.assertEqual(list(flag), ["distress"])
        self.assertTrue(bool(veto.iloc[0]))

    def test_altman_requires_all_components(self) -> None:
        frame = pd.DataFrame({"TOTAL_ASSETS": [1000.0], "TOTAL_LIABILITIES": [500.0],
                              "TOTAL_CURRENT_ASSETS": [400.0], "TOTAL_CURRENT_LIAB": [200.0],
                              "UNASSIGN_RPOFIT": [150.0], "TOTAL_PARENT_EQUITY": [500.0],
                              "_ttm_ebit": [np.nan]})
        self.assertTrue(pd.isna(altman_z(frame).iloc[0]))


class AvailableFromTests(unittest.TestCase):
    def test_effective_date_delays_when_annual_filed_after_q1(self) -> None:
        """一季报常早于上一年年报披露；此时若一季报的 TTM 引用了年报数据，生效日必须后延到年报公告日。"""
        ann = {
            "2023-12-31": "2024-04-28",
            "2024-03-31": "2024-04-20",
            "2024-06-30": "2024-08-20",
            "2024-09-30": "2024-10-25",
            "2024-12-31": "2025-04-25",
        }
        factors = compute_financial_factors(_facts(ann_dates=ann))
        row = factors.loc[factors["report_period"].eq(pd.Timestamp("2024-03-31"))].iloc[0]
        self.assertEqual(pd.Timestamp(row["available_from"]), pd.Timestamp("2024-04-28"))
        self.assertGreater(pd.Timestamp(row["available_from"]), pd.Timestamp("2024-04-20"))

    def test_effective_date_is_latest_of_lookback_window(self) -> None:
        factors = compute_financial_factors(_facts())
        last = factors.iloc[-1]
        self.assertEqual(pd.Timestamp(last["available_from"]), pd.Timestamp("2025-01-30"))

    def test_lookback_window_spans_five_periods(self) -> None:
        """回看窗口覆盖 4 个单季加 1 个上期累计，因此最早一季的生效日由窗口内最晚公告日决定。"""
        factors = compute_financial_factors(_facts())
        for _, row in factors.iterrows():
            self.assertTrue(pd.notna(row["available_from"]))
        first = factors.iloc[0]
        self.assertEqual(pd.Timestamp(first["available_from"]), pd.Timestamp("2023-04-30"))


class AttachTests(unittest.TestCase):
    def test_factor_only_visible_strictly_after_effective_date(self) -> None:
        """生效日当天不得生效：前一日与当日只能取到上一期，次一交易日起才轮到最新一期。"""
        factors = compute_financial_factors(_facts())
        effective = pd.Timestamp(factors.iloc[-1]["available_from"])
        panel = pd.DataFrame(
            {
                "date": [effective - pd.Timedelta(days=1), effective, effective + pd.Timedelta(days=1)],
                "code": ["600000"] * 3,
            }
        )
        attached, report = attach_financial_factors(panel, factors)
        periods = [pd.Timestamp(value) for value in attached["fundamental_period"]]
        self.assertEqual(periods[0], pd.Timestamp("2024-09-30"))
        self.assertEqual(periods[1], pd.Timestamp("2024-09-30"), "生效日当天不得启用最新一期")
        self.assertEqual(periods[2], pd.Timestamp("2024-12-31"))
        self.assertAlmostEqual(attached["fundamental_roe_ttm_deducted"].iloc[2], 0.08, places=6)
        self.assertGreater(report["coverage"], 0.0)

    def test_trace_columns_record_source_period(self) -> None:
        facts = _facts()
        factors = compute_financial_factors(facts)
        effective = pd.Timestamp(factors.iloc[-1]["available_from"])
        panel = pd.DataFrame({"date": [effective + pd.Timedelta(days=1)], "code": ["600000"]})
        attached, _ = attach_financial_factors(panel, factors)
        self.assertEqual(pd.Timestamp(attached["fundamental_period"].iloc[0]), pd.Timestamp("2024-12-31"))

    def test_empty_panel_is_returned_untouched(self) -> None:
        panel = pd.DataFrame(columns=["date", "code"])
        result, report = attach_financial_factors(panel, None)
        self.assertEqual(report["coverage"], 0.0)
        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
