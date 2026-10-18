"""验证个股页风格因子值、点时缺失与逐模型实际入模展示。"""

import math
import unittest

import pandas as pd
from streamlit.testing.v1 import AppTest

from services.stock_style import build_stock_style_analysis


class StockStyleAnalysisTests(unittest.TestCase):
    """股票级快照与历史版本严格区分，并沿用模型保存的实际特征名单。"""

    def setUp(self):
        self.quote = {
            "code": "600000", "market_cap_total": 10_000_000_000,
            "market_cap_float": 5_000_000_000, "pe_ttm": 10,
            "pe_dynamic": 13, "pb": 2, "ps_ttm": None,
            "quote_time": "2026-10-16T15:00:00+08:00",
            "fetched_at": "2026-10-16T15:01:00+08:00",
            "source": "测试行情源", "data_status": "最近收盘行情（非实时）",
            "quote_valid": True, "valuation_unit": "ratio", "market_cap_unit": "CNY",
        }
        self.history = pd.DataFrame([{
            "code": "600000", "date": "2020-01-02", "market_cap_total": 2_000_000_000,
            "market_cap_float": 1_000_000_000, "pe_ttm": 20, "pb": 1.5, "ps_ttm": 2.5,
            "net_profit_ttm": None, "book_equity": None, "revenue_ttm": None,
            "source": "测试点时源", "collected_at": "2020-01-02T16:00:00",
            "available_at": "2020-01-02T15:30:00", "revision_id": "初版",
            "unit": "CNY/share", "quality": "字段完整", "size_pit": True, "value_pit": False,
        }])
        self.models = {
            20: [{
                "model": "LightGBM", "validated": True, "prediction_percentile": 0.9,
                "status": "同日模型判断可用",
                "features_used": ["xs_log_market_cap_total", "xs_earnings_yield"],
            }],
        }
        self.context = {
            "models": {20: [{
                "model": "LightGBM",
                "features_used": ["xs_log_market_cap_total", "xs_earnings_yield"],
            }]},
            "style_factors": {
                "requested": {"size": True, "value": True}, "status": "已通过覆盖门控",
            },
        }

    def test_current_snapshot_and_historical_point_in_time_values_are_separate(self):
        report = build_stock_style_analysis(
            "600000", "2020-01-02", self.quote, self.models,
            history=self.history, context=self.context,
        )
        factors = {item["factor"]: item for item in report["factors"]}
        self.assertAlmostEqual(factors["earnings_yield"]["current_value"], 0.1)
        self.assertAlmostEqual(
            factors["log_market_cap_total"]["historical_value"], math.log(2_000_000_000)
        )
        self.assertIsNone(factors["earnings_yield"]["historical_value"])
        self.assertIn("历史财报披露", factors["earnings_yield"]["historical_status"])
        self.assertEqual(factors["log_market_cap_total"]["effective_models"], ["LightGBM / 20日"])
        self.assertEqual(factors["earnings_yield"]["effective_models"], ["LightGBM / 20日"])
        self.assertIn("市值", report["fundamental_status"])

    def test_code_or_unit_mismatch_suppresses_snapshot_factors(self):
        quote = {**self.quote, "code": "000001"}
        report = build_stock_style_analysis("600000", "2020-01-02", quote, {})
        self.assertTrue(all(item["current_value"] is None for item in report["factors"]))
        self.assertEqual(report["factors"][0]["current_status"], "快照股票代码不匹配")

    def test_individual_style_table_renders_actual_model_features(self):
        script = (
            "from services.stock_style import build_stock_style_analysis\n"
            "from services.research_ui import show_stock_style_analysis\n"
            f"quote = {self.quote!r}\n"
            f"models = {self.models!r}\n"
            f"context = {self.context!r}\n"
            "report = build_stock_style_analysis('600000', '2020-01-02', quote, models, context=context)\n"
            "show_stock_style_analysis(report)\n"
        )
        app = AppTest.from_string(script).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 5)
        self.assertEqual(app.metric[0].value, "100.00")
        self.assertEqual(len(app.dataframe), 2)
        self.assertTrue(any("本股有效截面模型已使用" in item.value for item in app.caption))


if __name__ == "__main__":
    unittest.main()
