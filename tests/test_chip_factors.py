"""筹码分布集中度因子：获利盘方向、区间宽度语义、停牌日处理与无未来函数。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from config import settings
from services.chip_factors import (
    CHIP_FACTOR_COLUMNS, attach_chip_factors, chip_distribution, turnover_fraction,
)

BARS = 200


def _bars(closes: np.ndarray, band: float, turnover: float = 0.5) -> pd.DataFrame:
    """按收盘价序列造日线：高低价在中枢上下 band 对称展开，成交额由成交量与收盘价折算。"""
    frame = pd.DataFrame({"date": pd.bdate_range("2025-01-02", periods=len(closes)), "close": closes})
    frame["high"] = frame["close"] + band
    frame["low"] = frame["close"] - band
    frame["volume"] = 1000.0
    frame["amount"] = frame["close"] * frame["volume"]
    # 行情源以百分比给出换手率，0.5 即 0.5%。
    frame["turnover_rate"] = turnover
    frame["code"] = "600519"
    return frame


def _flat(bars: int = BARS, price: float = 10.0, band: float = 0.1) -> pd.DataFrame:
    """窄幅横盘：高低价贴在价格中枢附近，筹码高度集中。"""
    return _bars(np.full(bars, price), band)


def _rising(bars: int = BARS) -> pd.DataFrame:
    """单边上行：筹码堆积在下方，现价位于成本分布顶端。"""
    return _bars(np.linspace(10.0, 20.0, bars), 0.1)


def _falling(bars: int = BARS) -> pd.DataFrame:
    """单边下行：筹码堆积在上方，现价位于成本分布底端。"""
    return _bars(np.linspace(20.0, 10.0, bars), 0.1)


def _dispersed(bars: int = BARS) -> pd.DataFrame:
    """宽幅震荡：每日振幅巨大且中枢来回摆动，筹码发散。"""
    return _bars(10.0 + np.sin(np.arange(bars) / 3.0), 2.0)


def _factor(history: pd.DataFrame, column: str, position: int = -1) -> float:
    factors, _, _ = chip_distribution(history)
    return float(factors[column].iloc[position])


class WinnerRateTests(unittest.TestCase):
    def test_rising_market_leaves_almost_every_holder_profitable(self) -> None:
        """现价站上全部历史成本时，获利盘比例应逼近 1。"""
        self.assertGreater(_factor(_rising(), "chip_winner_rate"), 0.9)

    def test_falling_market_leaves_almost_every_holder_underwater(self) -> None:
        """现价跌破全部历史成本时，获利盘比例应逼近 0。"""
        self.assertLess(_factor(_falling(), "chip_winner_rate"), 0.1)

    def test_symmetric_band_splits_holders_in_half(self) -> None:
        """价格钉在中枢上时，成本分布对称，获利盘应在一半附近。"""
        self.assertAlmostEqual(_factor(_flat(), "chip_winner_rate"), 0.5, delta=0.05)

    def test_rate_is_a_probability(self) -> None:
        for history in (_flat(), _rising(), _falling(), _dispersed()):
            factors, _, _ = chip_distribution(history)
            usable = factors["chip_winner_rate"].dropna()
            self.assertTrue(((usable >= 0.0) & (usable <= 1.0)).all())


class CostRangeWidthTests(unittest.TestCase):
    def test_narrow_band_is_tighter_than_wide_band(self) -> None:
        """横盘窄幅的 90% 成本区间必须显著窄于宽幅震荡。"""
        narrow = _factor(_flat(), "chip_cost_range_width")
        wide = _factor(_dispersed(), "chip_cost_range_width")
        self.assertLess(narrow, wide / 5.0)

    def test_width_is_relative_to_close(self) -> None:
        """区间宽度以现价为单位，价格整体平移不应改变宽度。"""
        first = _factor(_flat(band=0.1), "chip_cost_range_width")
        second = _factor(_bars(np.full(BARS, 50.0), 0.5), "chip_cost_range_width")
        self.assertAlmostEqual(first, second, places=6)

    def test_width_is_non_negative_and_finite(self) -> None:
        factors, _, _ = chip_distribution(_dispersed())
        usable = factors["chip_cost_range_width"].dropna()
        self.assertFalse(usable.empty)
        self.assertTrue((usable >= 0.0).all())


class DegenerateInputTests(unittest.TestCase):
    def test_one_word_board_does_not_crash_and_reads_as_fully_concentrated(self) -> None:
        """一字板全天无振幅：不抛异常，成本区间收敛到 0、持有人全部获利。"""
        factors, _, _ = chip_distribution(_flat(band=0.0))
        self.assertAlmostEqual(_factor(_flat(band=0.0), "chip_cost_range_width"), 0.0, places=9)
        self.assertAlmostEqual(float(factors["chip_winner_rate"].iloc[-1]), 1.0, places=9)

    def test_short_history_produces_no_factors(self) -> None:
        """历史不足最短长度时不出因子，不用短样本冒充稳态筹码。"""
        short = settings.CHIP_MINIMUM_BARS - 1
        factors, _, report = chip_distribution(_flat(bars=short))
        self.assertTrue(factors["chip_cost_range_width"].isna().all())
        self.assertEqual(report["factor_bars"], 0)
        self.assertIn(str(settings.CHIP_MINIMUM_BARS), report["status"])

    def test_minimum_bars_boundary_is_inclusive(self) -> None:
        factors, _, report = chip_distribution(_flat(bars=settings.CHIP_MINIMUM_BARS))
        self.assertEqual(report["factor_bars"], 1)
        self.assertTrue(np.isfinite(float(factors["chip_cost_range_width"].iloc[-1])))


class ValidationTests(unittest.TestCase):
    def test_missing_columns_are_named(self) -> None:
        frame = _flat().drop(columns=["low"])
        with self.assertRaises(ValueError) as caught:
            chip_distribution(frame)
        self.assertIn("low", str(caught.exception))

    def test_too_few_bins_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            chip_distribution(_flat(), bins=4)

    def test_coverage_must_be_a_fraction(self) -> None:
        for coverage in (0.0, 1.0, 1.5):
            with self.assertRaises(ValueError):
                chip_distribution(_flat(), coverage=coverage)

    def test_duplicate_dates_are_rejected(self) -> None:
        frame = _flat()
        frame.loc[1, "date"] = frame.loc[0, "date"]
        with self.assertRaises(ValueError):
            chip_distribution(frame)


class SuspendedBarTests(unittest.TestCase):
    def test_suspended_bar_neither_adds_chips_nor_decays_them(self) -> None:
        """停牌日不成交：既不新增筹码也不衰减存量，历史一侧的因子必须逐位不变。"""
        clean = _flat()
        dirty = clean.copy()
        dirty["is_usable"] = True
        # 复权异常造成的假高低价：若被当成真实成交会污染整条成本分布。
        # 该日换手率给足，确保「被当成真实成交」时污染足够显著、能被下面的对照组观察到。
        dirty.loc[150, ["is_usable", "high", "low", "close", "amount", "turnover_rate"]] = [
            False, 999.0, 1.0, 500.0, 0.0, 50.0,
        ]

        before, _, _ = chip_distribution(clean)
        after, _, report = chip_distribution(dirty)

        for column in CHIP_FACTOR_COLUMNS:
            np.testing.assert_array_equal(after[column].iloc[:150].to_numpy(), before[column].iloc[:150].to_numpy())
        self.assertTrue(np.isnan(float(after["chip_cost_range_width"].iloc[150])))
        self.assertEqual(report["accumulated_bars"], len(clean) - 1)
        # 停牌日之后只差「少做一次衰减与补入」，量级必须仍在噪声范围内，说明假价格没有渗进来。
        np.testing.assert_allclose(
            after["chip_cost_range_width"].iloc[151:].to_numpy(),
            before["chip_cost_range_width"].iloc[151:].to_numpy(),
            rtol=1e-9,
        )

    def test_unsuspended_bogus_prices_do_contaminate(self) -> None:
        """对照组：同一份假价格若被当成真实成交，成本分布会被显著改变。

        与上一条只差 `is_usable` 一个字段，用来证明「不被污染」确实是停牌标记换来的，
        而不是这条构造本身就对结果没有影响。
        """
        clean = _flat()
        polluted = clean.copy()
        polluted["is_usable"] = True
        polluted.loc[150, ["high", "low", "close", "amount", "turnover_rate"]] = [
            999.0, 1.0, 500.0, 0.0, 50.0,
        ]

        before, _, _ = chip_distribution(clean)
        after, _, _ = chip_distribution(polluted)

        clean_width = float(before["chip_cost_range_width"].iloc[-1])
        polluted_width = float(after["chip_cost_range_width"].iloc[-1])
        self.assertGreater(abs(polluted_width - clean_width) / clean_width, 0.5)

    def test_all_bars_usable_matches_row_count(self) -> None:
        _, _, report = chip_distribution(_flat())
        self.assertEqual(report["accumulated_bars"], report["bars"])


class NoLookaheadTests(unittest.TestCase):
    def test_future_bars_do_not_change_past_factors(self) -> None:
        """第 t 日的分布只由第 t 日及以前的成交决定，改写未来不得回灌历史。"""
        original = _flat()
        mutated = original.copy()
        mutated.loc[160:, ["high", "low", "close"]] = [100.0, 50.0, 80.0]

        before, _, _ = chip_distribution(original)
        after, _, _ = chip_distribution(mutated)

        for column in CHIP_FACTOR_COLUMNS:
            np.testing.assert_array_equal(after[column].iloc[:160].to_numpy(), before[column].iloc[:160].to_numpy())
        # 对照组：被改写的尾部本身必须变了，否则上面的相等断言可能恒真。
        self.assertFalse(
            np.array_equal(after["chip_cost_range_width"].iloc[160:].to_numpy(),
                           before["chip_cost_range_width"].iloc[160:].to_numpy())
        )


class TurnoverFractionTests(unittest.TestCase):
    def test_percent_scale_is_converted(self) -> None:
        converted = turnover_fraction(pd.Series([0.5, 0.8]))
        np.testing.assert_allclose(converted.to_numpy(), [0.005, 0.008])

    def test_fraction_scale_is_left_alone(self) -> None:
        converted = turnover_fraction(pd.Series([0.005, 0.008]))
        np.testing.assert_allclose(converted.to_numpy(), [0.005, 0.008])

    def test_scale_split_point_is_pinned(self) -> None:
        """0.05 是判定的分界：恰好在界上按小数处理，越界即按百分比除以 100。"""
        at_boundary = turnover_fraction(pd.Series([0.05]))
        past_boundary = turnover_fraction(pd.Series([0.06]))
        self.assertAlmostEqual(float(at_boundary.iloc[0]), 0.05, places=9)
        self.assertAlmostEqual(float(past_boundary.iloc[0]), 0.0006, places=9)

    def test_values_are_clipped_into_range(self) -> None:
        """越界的换手率截断到区间端点：超过 100% 记满仓换手，负数记当日无成交。"""
        converted = turnover_fraction(pd.Series([150.0, -1.0]))
        self.assertAlmostEqual(float(converted.iloc[0]), 1.0, places=9)
        self.assertAlmostEqual(float(converted.iloc[1]), 0.0, places=9)

    def test_missing_series_is_not_guessed(self) -> None:
        converted = turnover_fraction(pd.Series([np.nan, np.nan]))
        self.assertTrue(converted.isna().all())


class AttachTests(unittest.TestCase):
    def _panel(self, code: str = "600519", rows: int = 5) -> pd.DataFrame:
        dates = pd.bdate_range("2025-01-02", periods=BARS)[-rows:]
        return pd.DataFrame({"date": dates, "code": [code] * rows, "close": 10.0})

    def test_factors_are_joined_on_date(self) -> None:
        panel, report = attach_chip_factors(self._panel(), {"600519": _flat()})
        self.assertEqual(report["codes_computed"], 1)
        self.assertAlmostEqual(report["coverage"], 1.0, places=9)
        self.assertTrue(panel["chip_winner_rate"].notna().all())

    def test_code_absent_from_panel_is_skipped(self) -> None:
        _, report = attach_chip_factors(self._panel(code="000001"), {"600519": _flat()})
        self.assertEqual(report["codes_computed"], 0)

    def test_broken_history_does_not_block_the_others(self) -> None:
        histories = {"000001": _flat().drop(columns=["high"]), "600519": _flat()}
        _, report = attach_chip_factors(self._panel(), histories)
        self.assertEqual(report["codes_computed"], 1)
        self.assertEqual(len(report["failures"]), 1)
        self.assertIn("000001", report["failures"][0])

    def test_short_history_leaves_columns_empty(self) -> None:
        """覆盖率如实报 0，不用池均值把缺失的股票填出来。"""
        panel, report = attach_chip_factors(self._panel(), {"600519": _flat(bars=10)})
        self.assertEqual(report["coverage"], 0.0)
        self.assertTrue(panel["chip_cost_range_width"].isna().all())

    def test_empty_panel_short_circuits(self) -> None:
        _, report = attach_chip_factors(pd.DataFrame(columns=["date", "code"]), {"600519": _flat()})
        self.assertIn("面板为空", report["status"])

    def test_progress_is_throttled_and_reaches_the_last_stock(self) -> None:
        """逐只上报会在页面上写上千次元素，必须节流；但末条要报到最后一只。"""
        histories = {f"{600000 + offset:06d}": _flat(bars=30) for offset in range(120)}
        messages: list[str] = []
        attach_chip_factors(self._panel(), histories, progress=messages.append)
        self.assertLess(len(messages), len(histories))
        self.assertTrue(messages[-1].endswith(f"（{len(histories)}/{len(histories)}）"))

    def test_columns_exist_even_without_histories(self) -> None:
        panel, _ = attach_chip_factors(self._panel(), {})
        for column in CHIP_FACTOR_COLUMNS:
            self.assertIn(column, panel)


if __name__ == "__main__":
    unittest.main()
