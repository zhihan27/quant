"""验证加载期的行情异常清洗：前复权防跳空、长期停牌剔除与一字板流动性锁定。"""

import unittest

import numpy as np
import pandas as pd

from services.data_cleaning import FLAG_COLUMNS, clean_daily_history, price_limit_rate, tradable_mask


def _bars(closes, start="2024-01-01"):
    dates = pd.bdate_range(start, periods=len(closes))
    close = pd.Series(closes, dtype=float)
    return pd.DataFrame({
        "date": dates, "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": 1_000_000.0,
    })


class PriceLimitRateTests(unittest.TestCase):
    """涨跌幅上限必须按板块与制度变更日期区分。"""

    def test_board_limits(self):
        self.assertEqual(price_limit_rate("600000"), 0.10)
        self.assertEqual(price_limit_rate("000001"), 0.10)
        self.assertEqual(price_limit_rate("300001"), 0.20)
        self.assertEqual(price_limit_rate("300001", "2020-08-21"), 0.10)
        self.assertEqual(price_limit_rate("300001", "2020-08-24"), 0.20)
        self.assertEqual(price_limit_rate("688001"), 0.20)
        self.assertEqual(price_limit_rate("830001"), 0.30)
        self.assertEqual(price_limit_rate("600000", is_st=True), 0.05)


class CleaningTests(unittest.TestCase):
    """清洗只标注与剔除不可信 K 线，不改写价格。"""

    def test_suspension_and_long_suspension_are_flagged(self):
        bars = _bars([10.0] * 25)
        bars.loc[5:24, ["open", "high", "low", "close"]] = np.nan
        bars.loc[5:24, "volume"] = 0.0
        cleaned, report = clean_daily_history(bars, code="600000", adjusted=True)
        self.assertTrue(cleaned.loc[10, "is_suspended"])
        self.assertTrue(cleaned.loc[20, "is_long_suspension"])
        self.assertFalse(cleaned.loc[4, "is_suspended"])
        self.assertEqual(report["suspended_bars"], 20)
        self.assertEqual(report["long_suspension_bars"], 20)
        self.assertEqual(report["usable_bars"], 5)
        self.assertIn("连续停牌", report["long_suspension_rule"])
        for column in FLAG_COLUMNS:
            self.assertIn(column, cleaned.columns)

    def test_adjusted_history_with_price_gap_is_unusable(self):
        # 前复权序列出现远超 10% 上限的跳空，说明复权口径不一致。
        bars = _bars([10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 15.0, 15.1])
        cleaned, report = clean_daily_history(bars, code="600000", adjusted=True)
        self.assertTrue(cleaned.loc[6, "is_adjustment_gap"])
        self.assertFalse(cleaned.loc[6, "is_usable"])
        self.assertEqual(report["adjustment_gap_bars"], 1)
        self.assertEqual(report["adjustment"], "前复权")

    def test_raw_history_ex_dividend_gap_is_not_removed(self):
        bars = _bars([10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 15.0, 15.1])
        cleaned, report = clean_daily_history(bars, code="600000", adjusted=False)
        self.assertFalse(cleaned.loc[6, "is_adjustment_gap"])
        self.assertTrue(cleaned.loc[6, "is_usable"])
        self.assertEqual(report["adjustment_gap_bars"], 0)
        self.assertEqual(report["adjustment"], "不复权")

    def test_locked_limit_days_lock_liquidity(self):
        closes = [10.0] * 5 + [11.0, 12.1]
        bars = _bars(closes)
        for position in (5, 6):
            bars.loc[position, ["open", "high", "low"]] = bars.loc[position, "close"]
        cleaned, report = clean_daily_history(bars, code="600000", adjusted=True)
        self.assertTrue(cleaned.loc[5, "is_locked_up"])
        self.assertTrue(cleaned.loc[6, "is_locked_up"])
        self.assertEqual(report["locked_up_bars"], 2)
        # 一字涨停只锁流动性，价格本身仍然可信。
        self.assertTrue(cleaned.loc[5, "is_usable"])

    def test_new_listing_gap_is_not_treated_as_anomaly(self):
        bars = _bars([10.0, 30.0, 33.0, 36.0, 40.0, 44.0, 48.0, 52.0])
        cleaned, report = clean_daily_history(bars, code="300001", adjusted=True)
        self.assertEqual(report["adjustment_gap_bars"], 0)
        self.assertTrue(cleaned.loc[1, "is_usable"])

    def test_invalid_input_is_rejected(self):
        with self.assertRaises(ValueError):
            clean_daily_history(pd.DataFrame({"date": [], "open": []}))
        duplicated = pd.concat([_bars([10.0, 10.1]), _bars([10.0, 10.1]).tail(1)], ignore_index=True)
        with self.assertRaises(ValueError):
            clean_daily_history(duplicated, code="600000")

    def test_tradable_mask_blocks_stale_and_locked_bars(self):
        bars = _bars([10.0, 11.0, 9.0, 9.9])
        for position, column in ((1, "is_locked_up"), (2, "is_locked_down"), (3, "is_usable")):
            bars[column] = False
        bars.loc[1, "is_locked_up"] = True
        bars.loc[2, "is_locked_down"] = True
        bars["is_usable"] = True
        bars.loc[3, "is_usable"] = False
        buyable = tradable_mask(bars, "buy")
        sellable = tradable_mask(bars, "sell")
        self.assertEqual(buyable.tolist(), [True, False, True, False])
        self.assertEqual(sellable.tolist(), [True, True, False, False])
        with self.assertRaises(ValueError):
            tradable_mask(bars, "hold")


if __name__ == "__main__":
    unittest.main()
