"""覆盖短线滚动验证和撮合回放的关键边界。"""

import unittest
from unittest.mock import patch

import pandas as pd

from config.short_term import SHORT_TERM_FEATURES
from services.short_term.execution_replay import run_execution_replay
from services.short_term.validation import walk_forward_validation


class _ScoreByFirstFeature:
    """用稳定的首因子分数替代训练器，专测验证切窗。"""

    def fit_cross_section(self, features, target, dates):
        return None

    def predict(self, features):
        return features["return_1d"].to_numpy()


def _validation_panel(days=32, stocks=5):
    dates = pd.bdate_range("2024-01-02", periods=days)
    rows = []
    for day, date in enumerate(dates):
        for stock in range(stocks):
            row = {
                "code": f"{600000 + stock}",
                "date": date,
                "label_end_date": date,
                "target_net_return": (stock - 2) * 0.01 + day * 0.0001,
                "return_5d": float(stock),
            }
            row.update({column: float(stock) + day * 0.001 for column in SHORT_TERM_FEATURES})
            rows.append(row)
    return pd.DataFrame(rows), dates


class ShortTermValidationTests(unittest.TestCase):
    """确认每折训练集只保留最近配置窗口。"""

    def test_training_window_is_rolling_not_expanding(self):
        panel, dates = _validation_panel()
        with patch("services.short_term.validation.make_ranker", return_value=_ScoreByFirstFeature()):
            result = walk_forward_validation(
                panel,
                horizon=3,
                train_days=26,
                test_days=2,
                top_k=2,
            )
        self.assertEqual(result["fold_count"], 3)
        self.assertEqual([fold["train_days"] for fold in result["folds"]], [26, 26, 26])
        self.assertEqual(result["folds"][1]["train_start"], dates[2].date().isoformat())
        self.assertEqual(result["folds"][1]["train_end"], dates[27].date().isoformat())


def _execution_inputs():
    dates = pd.bdate_range("2024-01-02", periods=3)
    bars = pd.DataFrame(
        [
            {"code": "600000", "date": dates[0], "open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0, "volume": 1000},
            {"code": "600000", "date": dates[1], "open": 10.05, "high": 10.2, "low": 9.8, "close": 10.1, "volume": 300},
            {"code": "600000", "date": dates[2], "open": 10.2, "high": 10.3, "low": 10.0, "close": 10.2, "volume": 300},
        ]
    )
    universe = pd.DataFrame(
        [{"code": "600000", "date": dates[0], "available_at": "2024-01-02T15:00:00+08:00", "is_member": True}]
    )
    status_rows = []
    for date in dates[1:]:
        status_rows.append(
            {
                "code": "600000",
                "date": date,
                "available_at": f"{date.date().isoformat()}T09:00:00+08:00",
                "is_listed": True,
                "is_st": False,
                "is_suspended": False,
                "is_limit_up": False,
                "is_limit_down": False,
            }
        )
    security_status = pd.DataFrame(status_rows)
    signals = pd.DataFrame(
        [{"code": "600000", "date": dates[0], "signal_time": "2024-01-02T15:05:00+08:00", "ranking_score": 1.0, "reference_price": 10.0}]
    )
    terms = {
        "order_shares": 200,
        "lot_size": 100,
        "volume_unit": "shares",
        "max_participation_rate": 0.5,
        "entry_limit_offset_bps": 100,
        "exit_limit_offset_bps": 100,
        "entry_valid_sessions": 1,
        "exit_grace_sessions": 0,
        "commission_rate": 0.001,
        "minimum_commission": 5,
        "stamp_duty_rate": 0.001,
        "slippage_bps": 0,
        "allow_st": False,
    }
    return signals, bars, universe, security_status, terms


class ShortTermExecutionReplayTests(unittest.TestCase):
    """确认回放保留未成交、部分成交、最低佣金及点时阻断。"""

    def test_partial_fill_minimum_fee_and_t_plus_one_exit(self):
        signals, bars, universe, status, terms = _execution_inputs()
        result = run_execution_replay(signals, bars, universe, status, terms, horizon=1, top_k=1)
        trade = result["trades"][0]
        self.assertEqual(result["status"], "daily_bar_proxy_not_queue_truth")
        self.assertEqual(trade["status"], "closed_partial_entry")
        self.assertEqual(trade["filled_shares"], 100)
        self.assertEqual(trade["unfilled_shares"], 100)
        self.assertEqual(trade["entry_fill_count"], 1)
        self.assertEqual(trade["exit_fill_count"], 1)
        self.assertAlmostEqual(trade["buy_cost_including_minimum_commission"], 1005.0 + 5.0)

    def test_missing_point_in_time_universe_blocks_replay(self):
        signals, bars, _, status, terms = _execution_inputs()
        result = run_execution_replay(signals, bars, pd.DataFrame(), status, terms, horizon=1, top_k=1)
        self.assertEqual(result["status"], "blocked_missing_inputs")
        self.assertFalse(result["promotion_evidence"])
        self.assertTrue(any("历史股票池" in blocker for blocker in result["blockers"]))


if __name__ == "__main__":
    unittest.main()
