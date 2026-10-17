"""模块四/五：日内波动率衰减与订单流不平衡代理的无未来函数约束与口径。"""

from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from config import settings
from services.intraday_factors import (
    INTRADAY_FACTOR_COLUMNS, SESSION_BUCKETS, intraday_day_factors, load_intraday_factors,
    order_flow_imbalance, store_intraday_bars,
)
from services.panel_store import PanelStore

BARS_PER_BUCKET = 30
# 事故当轮页面上的池子规模：5174 以内最后一个 50 的整倍数是 5150，
# 只按整倍数上报时字幕会停在「5151/5174」。回归用例必须用这个规模才有鉴别力。
POOL_SIZE = 5174


def _bucket_times(day: str) -> list[pd.Timestamp]:
    """按连续竞价的 8 个 30 分钟区间生成整点分钟线时间戳。"""
    stamps: list[pd.Timestamp] = []
    for start, _end in SESSION_BUCKETS:
        begin = pd.Timestamp(f"{day} {start.strftime('%H:%M')}")
        stamps.extend(pd.date_range(begin, periods=BARS_PER_BUCKET, freq="min"))
    return stamps


def _minutes(steps: list[float], day: str = "2026-01-05", base: float = 10.0) -> pd.DataFrame:
    """构造分钟的三角波收盘价：每个区间的收盘价从 base 出发再回到 base。

    区间首尾都等于 base，跨区间收益率恒为 0，日内波动率就只由区间内的振幅 ``step`` 决定，
    便于用手算的单调关系校验衰减斜率。振幅按 ``steps`` 逐区间给出。
    """
    closes: list[float] = []
    for step in steps:
        closes.extend(base + np.concatenate([np.linspace(0.0, step, BARS_PER_BUCKET // 2),
                                             np.linspace(step, 0.0, BARS_PER_BUCKET // 2)]))
    frame = pd.DataFrame(
        {
            "date": _bucket_times(day),
            "close": closes,
            "volume": 1000.0,
        }
    )
    frame["open"] = frame["close"]
    frame["high"] = frame["close"] + 0.02
    frame["low"] = frame["close"] - 0.02
    return frame


def _decaying() -> list[float]:
    """波动率逐区间衰减：开盘振幅最大，尾盘最小。"""
    return [0.01 * (8 - index) for index in range(len(SESSION_BUCKETS))]


def _rising() -> list[float]:
    return list(reversed(_decaying()))


class VolatilityDecayTests(unittest.TestCase):
    def test_decaying_volatility_gives_negative_slope(self) -> None:
        row = intraday_day_factors(_minutes(_decaying())).iloc[0]
        self.assertLess(row["intraday_vol_decay_slope"], 0.0)

    def test_rising_volatility_gives_positive_slope(self) -> None:
        row = intraday_day_factors(_minutes(_rising())).iloc[0]
        self.assertGreater(row["intraday_vol_decay_slope"], 0.0)

    def test_flat_volatility_gives_zero_slope(self) -> None:
        """振幅恒定时斜率应贴近 0；不严格等于 0 是因为当日首根 K 线没有收益率可算。"""
        flat = intraday_day_factors(_minutes([0.05] * len(SESSION_BUCKETS))).iloc[0]
        decaying = intraday_day_factors(_minutes(_decaying())).iloc[0]
        self.assertLess(abs(flat["intraday_vol_decay_slope"]), 1e-5)
        self.assertLess(abs(flat["intraday_vol_decay_slope"]), abs(decaying["intraday_vol_decay_slope"]) / 100.0)

    def test_tail_open_ratio_below_one_when_volatility_decays(self) -> None:
        row = intraday_day_factors(_minutes(_decaying())).iloc[0]
        self.assertLess(row["intraday_vol_tail_open_ratio"], 1.0)

    def test_tail_open_ratio_above_one_when_volatility_rises(self) -> None:
        row = intraday_day_factors(_minutes(_rising())).iloc[0]
        self.assertGreater(row["intraday_vol_tail_open_ratio"], 1.0)

    def test_missing_tail_buckets_yields_no_ratio(self) -> None:
        """只有上午 4 个区间时尾盘波动不存在，不能拿开盘区间硬凑出比值。"""
        frame = _minutes(_decaying())
        morning = frame.loc[frame["date"].dt.hour < 12]
        row = intraday_day_factors(morning).iloc[0]
        self.assertTrue(pd.isna(row["intraday_vol_tail_open_ratio"]))

    def test_too_few_buckets_yields_no_slope(self) -> None:
        """回归至少需要 4 个区间，否则斜率没有统计意义。"""
        frame = _minutes(_decaying())
        early = frame.loc[frame["date"] < pd.Timestamp("2026-01-05 10:31")]
        row = intraday_day_factors(early).iloc[0]
        self.assertTrue(pd.isna(row["intraday_vol_decay_slope"]))

    def test_short_day_is_skipped_entirely(self) -> None:
        frame = _minutes(_decaying()).head(int(settings.INTRADAY_MINIMUM_BARS_PER_DAY) - 1)
        self.assertTrue(intraday_day_factors(frame).empty)

    def test_missing_columns_raise(self) -> None:
        with self.assertRaises(ValueError):
            intraday_day_factors(pd.DataFrame({"date": [pd.Timestamp("2026-01-05 09:31")], "close": [10.0]}))


class OrderFlowTests(unittest.TestCase):
    def test_all_rising_closes_is_full_net_buying(self) -> None:
        frame = _monotone(np.linspace(10.0, 11.0, 60))
        row = intraday_day_factors(frame).iloc[0]
        self.assertAlmostEqual(row["intraday_ofi_proxy"], 1.0, places=9)

    def test_all_falling_closes_is_full_net_selling(self) -> None:
        frame = _monotone(np.linspace(11.0, 10.0, 60))
        row = intraday_day_factors(frame).iloc[0]
        self.assertAlmostEqual(row["intraday_ofi_proxy"], -1.0, places=9)

    def test_flat_closes_cancel_out(self) -> None:
        """价格全程持平没有方向信息，主动买卖对半分，净额为零。"""
        frame = _monotone(np.full(60, 10.0))
        row = intraday_day_factors(frame).iloc[0]
        self.assertAlmostEqual(row["intraday_ofi_proxy"], 0.0, places=9)

    def test_zero_volume_yields_no_signal(self) -> None:
        frame = _monotone(np.linspace(10.0, 11.0, 60))
        frame["volume"] = 0.0
        row = intraday_day_factors(frame).iloc[0]
        self.assertTrue(pd.isna(row["intraday_ofi_proxy"]))

    def test_standard_ofi_is_cumulative_net_order_flow(self) -> None:
        """接入 L2 后使用的标准 OFI（Cont 等 2014），返回逐事件的**累计**净委托流。

        各事件的口径：买价上行取新买量、买价持平取买量差、买价下行取负的旧买量；
        卖价下行取负的新卖量、卖价持平取负的卖量差、卖价上行取正的旧卖量
        （卖价被抬升说明卖盘被吃掉，是买方力量）。
        """
        result = order_flow_imbalance(
            bid_price=[10.0, 10.1, 10.1], bid_size=[100.0, 200.0, 150.0],
            ask_price=[10.2, 10.2, 10.3], ask_size=[100.0, 120.0, 150.0],
        )
        self.assertEqual(float(result[0]), 0.0)
        # 买价上行 +200；卖价持平 → -(120-100) = -20。
        self.assertAlmostEqual(float(result[1]), 200.0 - 20.0, places=9)
        # 买价持平 → +(150-200) = -50；卖价上行 → +120。累计 180-50+120。
        self.assertAlmostEqual(float(result[2]), 180.0 - 50.0 + 120.0, places=9)

    def test_ask_price_rise_is_positive_flow(self) -> None:
        """卖价被抬升意味着卖盘被吃掉，应记为正的委托流而不是负的。"""
        result = order_flow_imbalance(
            bid_price=[10.0, 10.0], bid_size=[100.0, 100.0],
            ask_price=[10.2, 10.3], ask_size=[100.0, 100.0],
        )
        self.assertGreater(float(result[1]), 0.0)

    def test_mismatched_lengths_raise(self) -> None:
        with self.assertRaises(ValueError):
            order_flow_imbalance([10.0], [100.0, 90.0], [10.2], [100.0])


def _monotone(closes: np.ndarray, day: str = "2026-01-05") -> pd.DataFrame:
    """单区间连续竞价序列，用于把 OFI 代理隔离出来观察。"""
    stamps = pd.date_range(pd.Timestamp(f"{day} 09:31"), periods=len(closes), freq="min")
    frame = pd.DataFrame({"date": stamps, "close": closes, "volume": 1000.0})
    frame["open"] = frame["close"]
    frame["high"] = frame["close"] + 0.05
    frame["low"] = frame["close"] - 0.05
    return frame


class VwapPositionTests(unittest.TestCase):
    def test_close_at_daily_high_is_one(self) -> None:
        frame = _minutes([0.05] * len(SESSION_BUCKETS))
        frame["high"] = frame["close"] + 0.10
        frame["low"] = frame["close"] - 0.10
        row = intraday_day_factors(frame).iloc[0]
        self.assertAlmostEqual(row["intraday_vwap_position"], 0.5, places=6)

    def test_position_is_bounded_between_zero_and_one(self) -> None:
        for step in (0.01, 0.1, 1.0):
            with self.subTest(step=step):
                row = intraday_day_factors(_minutes([step] * len(SESSION_BUCKETS))).iloc[0]
                self.assertGreaterEqual(row["intraday_vwap_position"], 0.0)
                self.assertLessEqual(row["intraday_vwap_position"], 1.0)

    def test_volume_units_do_not_shift_the_position(self) -> None:
        """成交量按「手」或按「股」给出应得到同一个位置，不能因为单位不同而漂移。"""
        frame = _minutes(_decaying())
        in_shares = intraday_day_factors(frame).iloc[0]["intraday_vwap_position"]
        frame["volume"] = frame["volume"] / 100.0
        in_lots = intraday_day_factors(frame).iloc[0]["intraday_vwap_position"]
        self.assertAlmostEqual(in_shares, in_lots, places=9)


class NoLookaheadTests(unittest.TestCase):
    def test_one_day_only_depends_on_its_own_minutes(self) -> None:
        """改写后一交易日的分钟线不得改变前一交易日的因子，否则就是未来函数。"""
        first = _minutes(_decaying(), day="2026-01-05")
        second = _minutes(_rising(), day="2026-01-06")
        baseline = intraday_day_factors(pd.concat([first, second], ignore_index=True))

        tampered_second = _minutes([0.5] * len(SESSION_BUCKETS), day="2026-01-06")
        tampered = intraday_day_factors(pd.concat([first, tampered_second], ignore_index=True))

        baseline_row = baseline.loc[baseline["date"].eq(pd.Timestamp("2026-01-05"))].iloc[0]
        tampered_row = tampered.loc[tampered["date"].eq(pd.Timestamp("2026-01-05"))].iloc[0]
        for column in INTRADAY_FACTOR_COLUMNS:
            self.assertAlmostEqual(baseline_row[column], tampered_row[column], places=12, msg=column)

    def test_second_day_row_does_change(self) -> None:
        """对照组：确实改动了的那一天本身应当变化，否则上面的断言可能是恒真。"""
        first = _minutes(_decaying(), day="2026-01-05")
        baseline = intraday_day_factors(pd.concat([first, _minutes(_rising(), day="2026-01-06")], ignore_index=True))
        tampered = intraday_day_factors(
            pd.concat([first, _minutes([0.5] * len(SESSION_BUCKETS), day="2026-01-06")], ignore_index=True)
        )
        self.assertNotAlmostEqual(
            baseline.iloc[-1]["intraday_vol_decay_slope"], tampered.iloc[-1]["intraday_vol_decay_slope"], places=9
        )


class PersistedBarsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = PanelStore(root=Path(tempfile.mkdtemp(prefix="intraday_test_")))

    def test_load_recomputes_factors_from_local_bars(self) -> None:
        bars = _minutes(_decaying(), day="2026-01-05")
        store_intraday_bars(self.store, "600519", bars)
        factors, report = load_intraday_factors(self.store, ["600519", "000001"])
        self.assertEqual(report["codes_computed"], 1)
        self.assertEqual(report["codes_without_bars"], 1)
        row = factors.iloc[0]
        self.assertEqual(row["code"], "600519")
        self.assertEqual(pd.Timestamp(row["date"]).date(), date(2026, 1, 5))
        self.assertLess(row["intraday_vol_decay_slope"], 0.0)

    def test_missing_bars_produce_empty_frame_with_columns(self) -> None:
        factors, report = load_intraday_factors(self.store, ["600519"])
        self.assertTrue(factors.empty)
        for column in ["date", "code", *INTRADAY_FACTOR_COLUMNS]:
            self.assertIn(column, factors.columns)
        self.assertIn("本地暂无分钟线", report["status"])

    def test_progress_reaches_the_last_code(self) -> None:
        """钉住「卡在 5151/5174」：节流不能把最后几只吞掉，末条必须是 N/N。

        规模必须用事故当轮的真实池子：只按整倍数上报时，5174 以内最后一个整倍数是 5150，
        末条停在 5151/5174，剩下 23 只被无声吞掉。若把池子取成 51 这种「末尾恰好是整倍数」
        的规模，旧写法会凑巧打到最后一只，这条用例就废了。
        """
        codes = [f"{600000 + offset:06d}" for offset in range(POOL_SIZE)]
        messages: list[str] = []
        load_intraday_factors(self.store, codes, progress=messages.append)
        self.assertEqual(messages[0], f"正在汇总分钟线（1/{POOL_SIZE}）")
        self.assertEqual(messages[-1], f"正在汇总分钟线（{POOL_SIZE}/{POOL_SIZE}）")
        self.assertLess(len(messages), len(codes))

    def test_reloaded_bars_keep_all_buckets(self) -> None:
        """落盘再读回后仍能算出 8 个区间，说明时间戳没有在存储环节丢失精度。"""
        store_intraday_bars(self.store, "600519", _minutes(_decaying(), day="2026-01-05"))
        factors, _ = load_intraday_factors(self.store, ["600519"])
        self.assertEqual(int(factors.iloc[0]["intraday_buckets"]), len(SESSION_BUCKETS))


if __name__ == "__main__":
    unittest.main()
