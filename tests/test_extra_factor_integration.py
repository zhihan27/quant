"""新增因子并入排序特征矩阵：口径一致性与覆盖率门控在训练入口上的实际表现。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from services.chip_factors import CHIP_FACTOR_COLUMNS
from services.feature_engineering import FEATURE_COLUMNS
from services.financial_factors import FINANCIAL_FACTOR_COLUMNS
from services.ml_pipeline import _select_fold_features, build_market_panel

# 截面中性化要求单日至少 20 只股票，否则每日截面会被整体跳过、退化成 Z 分数。
STOCKS = 24
DAYS = 260


def _histories() -> dict[str, pd.DataFrame]:
    """确定性日线：每只股票有各自的漂移，便于截面因子产生真实差异。"""
    dates = pd.bdate_range("2023-01-02", periods=DAYS)
    histories = {}
    for stock in range(STOCKS):
        step = np.arange(DAYS)
        drift = (stock - STOCKS / 2) / 5000.0
        close = 10.0 * (1.0 + drift) ** step * (1.0 + 0.01 * np.sin(step / 7.0))
        # 成交量必须随日波动：恒定量会让量价相关系数无定义，导致相关因子整列为空。
        volume = (1000.0 + stock) * (1.0 + 0.3 * np.sin(step / 5.0 + stock))
        frame = pd.DataFrame({"date": dates, "close": close, "volume": volume})
        frame["open"] = frame["close"]
        frame["high"] = frame["close"] * 1.01
        frame["low"] = frame["close"] * 0.99
        frame["amount"] = frame["close"] * frame["volume"]
        frame["turnover_rate"] = 1.0
        frame.attrs["turnover_rate_unit"] = "percent"
        histories[str(stock).zfill(6)] = frame
    return histories


def _exposures(codes: list[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
    """逐股逐日的点时流通市值和行业，用于验证中性化整段历史。

    行业分成 3 组、每组 8 只，满足中性化对单一行业最少 2 只股票的要求。
    """
    return pd.DataFrame([
        {
            "date": date, "code": code, "price": 10.0,
            "market_cap_float": 1e9 * (position + 1), "industry": f"行业{position % 3}",
        }
        for position, code in enumerate(codes)
        for date in dates
    ])


def _financial_factors(codes: list[str], vetoed: set[str]) -> pd.DataFrame:
    """可直接并入的 PIT 财务因子长表：生效日早于全部交易日，因此每个交易日都取得到。"""
    rows = []
    for code in codes:
        row: dict = {
            "code": code,
            "available_from": pd.Timestamp("2023-01-01"),
            "report_period": pd.Timestamp("2022-12-31"),
            "fundamental_health_flag": "distress" if code in vetoed else "safe",
            "fundamental_health_veto": code in vetoed,
        }
        for column in FINANCIAL_FACTOR_COLUMNS:
            row[column] = 0.05
        rows.append(row)
    return pd.DataFrame(rows)


class ExtraFactorIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.histories = _histories()
        self.benchmark = _histories()["000000"].copy()
        self.benchmark["close"] = 3000.0 * (1.0 + 0.0002) ** np.arange(DAYS)
        self.benchmark["high"] = self.benchmark["close"] * 1.005
        self.benchmark["low"] = self.benchmark["close"] * 0.995

    def test_absent_extra_factors_keep_existing_feature_set(self) -> None:
        panel = build_market_panel(self.histories, self.benchmark)
        self.assertEqual(panel.attrs["model_feature_columns"], [f"xs_{column}" for column in FEATURE_COLUMNS])
        self.assertEqual(panel.attrs["extra_factors"]["status"], "未提供新增因子")

    def test_build_market_panel_reports_per_stock_progress(self) -> None:
        """面板构造的两段逐股循环都要上报，否则页面会停在上一条字幕上像卡死。

        全市场扫描下这一段（构造因子 + 并入筹码）要跑好几分钟，
        不接进度回调的话，「正在汇总分钟线」之后就是一段无输出的黑洞。
        """
        messages: list[str] = []
        build_market_panel(
            self.histories, self.benchmark, extra_factors={"chip": self.histories},
            progress=messages.append,
        )
        features = [text for text in messages if text.startswith("正在构造截面因子")]
        chips = [text for text in messages if "的筹码分布" in text]
        self.assertEqual(features[-1], f"正在构造截面因子（{STOCKS}/{STOCKS}）")
        self.assertTrue(chips[-1].endswith(f"（{STOCKS}/{STOCKS}）"))

    def test_chip_factors_are_added_to_model_features(self) -> None:
        panel = build_market_panel(self.histories, self.benchmark, extra_factors={"chip": self.histories})
        features = panel.attrs["model_feature_columns"]
        for column in CHIP_FACTOR_COLUMNS:
            self.assertIn(f"xs_{column}", features)
        self.assertEqual(panel.attrs["extra_factors"]["model_features"], len(CHIP_FACTOR_COLUMNS))

    def test_extra_factors_are_neutralized_alongside_base_factors(self) -> None:
        """基础因子做了中性化时新增因子不能只做 Z 分数，否则会把市值与行业暴露重新带进模型。"""
        dates = pd.DatetimeIndex(self.histories[sorted(self.histories)[0]]["date"])
        exposures = _exposures(sorted(self.histories), dates)
        panel = build_market_panel(
            self.histories, self.benchmark, exposures=exposures, extra_factors={"chip": self.histories}
        )
        features = panel.attrs["model_feature_columns"]
        self.assertTrue(any(name.startswith("neutral_") for name in features))
        for column in CHIP_FACTOR_COLUMNS:
            self.assertIn(f"neutral_{column}", features)
            self.assertNotIn(f"xs_{column}", features)
        self.assertIn("neutralization", panel.attrs["extra_factors"])
        self.assertGreater(panel.attrs["extra_factors"]["neutralization"]["coverage"], 0.5)

    def test_late_family_coverage_is_selected_inside_each_training_fold(self) -> None:
        """日内族仅在末日有值时保留为候选，但不能进入此前的训练折。"""
        dates = pd.bdate_range("2023-01-02", periods=DAYS)
        intraday = pd.DataFrame(
            {
                "date": [dates[-1]] * STOCKS,
                "code": sorted(self.histories),
                "intraday_vol_decay_slope": np.linspace(-1e-4, -1e-5, STOCKS),
                "intraday_vol_tail_open_ratio": np.linspace(0.2, 0.9, STOCKS),
                "intraday_ofi_proxy": np.linspace(-0.5, 0.5, STOCKS),
                "intraday_vwap_position": np.linspace(0.1, 0.9, STOCKS),
            }
        )
        panel = build_market_panel(
            self.histories, self.benchmark, extra_factors={"intraday": intraday}
        )
        features = panel.attrs["model_feature_columns"]
        self.assertEqual(features[:len(FEATURE_COLUMNS)], [f"xs_{column}" for column in FEATURE_COLUMNS])
        self.assertTrue(all(f"xs_{column}" in features for column in intraday.columns[2:]))
        self.assertFalse(panel.attrs["extra_factors"]["decisions"]["intraday"]["accepted"])
        early = panel.loc[panel["date"] < dates[-1]]
        selected = _select_fold_features(early, features, 8)
        self.assertFalse(any(name.startswith("xs_intraday_") for name in selected))

    def test_missing_financial_data_never_vetoes(self) -> None:
        """没有 PIT 财务面板时一股都不否决：未知不等于不健康。"""
        panel = build_market_panel(self.histories, self.benchmark, extra_factors={"chip": self.histories})
        self.assertEqual(panel.attrs["health_veto"]["vetoed"], 0)
        self.assertEqual(len(set(panel["code"].astype(str))), STOCKS)

    def test_health_veto_removes_codes_from_the_tradable_panel(self) -> None:
        codes = sorted(self.histories)
        vetoed = set(codes[:3])
        panel = build_market_panel(
            self.histories, self.benchmark,
            extra_factors={"financial": _financial_factors(codes, vetoed)},
        )
        report = panel.attrs["health_veto"]
        self.assertTrue(report["applied"])
        self.assertEqual(report["codes"], len(vetoed))
        self.assertEqual(set(panel["code"].astype(str)), set(codes) - vetoed)

    def test_veto_takes_effect_before_the_cross_section_is_standardized(self) -> None:
        """被否决的标的不得参与自己所在截面的均值与标准差。

        标准化若先于否决执行，留存下来的行是「从含否决股票的截面里算出来的 Z 分数再被裁掉」，
        逐日均值就不再为零；反过来先否决再标准化，留存截面的逐日均值必然回到零。
        """
        codes = sorted(self.histories)
        vetoed = set(codes[:3])
        panel = build_market_panel(
            self.histories, self.benchmark,
            extra_factors={"financial": _financial_factors(codes, vetoed)},
        )
        means = panel.groupby("date")[f"xs_{FEATURE_COLUMNS[0]}"].mean().abs()
        self.assertLess(float(means.max()), 1e-9)

    def test_incomplete_family_is_not_partially_admitted(self) -> None:
        """只并入半族列时整族不参与，避免模型悄悄只用一半因子。"""
        panel = build_market_panel(
            self.histories, self.benchmark, extra_factors={"chip": self.histories}
        )
        self.assertIn("chip_cost_range_width", panel)
        self.assertIn("chip_winner_rate", panel)
        self.assertTrue(panel.attrs["extra_factors"]["decisions"]["chip"]["accepted"])


if __name__ == "__main__":
    unittest.main()
