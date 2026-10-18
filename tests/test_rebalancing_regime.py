"""验证人工调仓分类、缺失保护、市场时点隔离和真实 PostgreSQL 存储。"""

import os
import secrets
import unittest
from datetime import date, datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from streamlit.testing.v1 import AppTest

from config import settings
from services.database import connection_settings
from services.market_regime import classify_market, latest_regime, save_regime
from services.market_data import MarketDataService
from services.rebalancing import build_checklist, ranking_snapshot
from services.timing_strategy import build_trade_plan


def _rank(code, rank=1, price=10):
    """构造已过模型与风控门槛的冻结排名行。"""
    return {"code": code, "name": "样本", "rank": rank, "score": 0.9,
        "reference_date": "2026-10-16", "model_validated": True, "price": price,
        "quote_time": "2026-10-16 15:00:00", "quote_valid": True,
        "volume_lots": 10000, "amount_yuan": 5e7, "ma20_gap": 0.05,
        "entry_allowed": True, "entry_trade_status": "可正常评估",
        "entry_generated_at": "2026-10-16T18:00:00+08:00", "entry_valid_until": "2026-10-23T18:00:00+08:00"}


def _position(code, price=10):
    """构造含费用的真实持仓字段。"""
    return {"code": code, "name": "样本", "shares": 100, "available_shares": 100,
            "buy_date": date(2026, 10, 15), "buy_price": price, "fees": 0}


class ChecklistTests(unittest.TestCase):
    """缺失排名、过期行情与未知风控条件均不能触发买入。"""

    def scan(self, rows):
        return {"model_version": settings.MODEL_VERSION, "context_id": "本轮模型",
            "pool_mode": "沪深京全部 A 股", "reference_date": "2026-10-16", "actionable_rankings": rows,
            "rows": []}

    def run_checklist(self, positions, scan, **kwargs):
        return build_checklist(positions, scan, top_n=2,
            now=datetime(2026, 10, 18, 9, tzinfo=ZoneInfo("Asia/Shanghai")), **kwargs)

    def test_sell_buy_hold_and_reason_priority(self):
        rows = [_rank("600000", 3), _rank("600001", 1), _rank("600002", 2), _rank("600003", 4, 8)]
        result = self.run_checklist([_position("600000"), _position("600001"), _position("600003")], self.scan(rows))
        self.assertEqual({row["代码"] for row in result["sell"]}, {"600000", "600003"})
        self.assertEqual(result["buy"][0]["代码"], "600002")
        self.assertEqual(result["hold"][0]["代码"], "600001")
        self.assertIn("止损", result["sell"][1]["原因"])

    def test_missing_stock_is_not_assumed_outside_top_n(self):
        result = self.run_checklist([_position("600009")], self.scan([_rank("600001")]))
        self.assertFalse(result["sell"])
        self.assertEqual(result["review"][0]["代码"], "600009")

    def test_stale_degraded_and_future_scan_do_not_generate_actions(self):
        for change in ({"reference_date": "2026-09-01"}, {"reference_date": "2026-10-20"}, {"model_version": "旧版本"}):
            result = self.run_checklist([_position("600000")], {**self.scan([_rank("600000", 3, 8)]), **change})
            self.assertFalse(result["sell"])
            self.assertFalse(result["buy"])
        result = self.run_checklist([], self.scan([_rank("600001")]), scoring_paused=True)
        self.assertFalse(result["buy"])

    def test_model_pause_does_not_hide_independent_position_stop_loss(self):
        result = self.run_checklist([_position("600000")], self.scan([_rank("600000", 1, 8)]), scoring_paused=True)
        self.assertEqual(result["sell"][0]["代码"], "600000")
        self.assertIn("止损", result["sell"][0]["原因"])
        self.assertFalse(result["buy"])

    def test_liquidity_limit_and_model_gates(self):
        for change in ({"amount_yuan": None}, {"amount_yuan": 0}, {"entry_allowed": False},
                       {"entry_trade_status": "接近涨跌停"}, {"quote_time": "2026-10-15"}, {"quote_valid": False},
                       {"reference_date": "2026-10-15"}):
            row = {**_rank("600001"), **change}
            self.assertFalse(self.run_checklist([], self.scan([row]))["buy"])

    def test_multiple_lots_today_t_plus_one_and_no_duplicate_buy(self):
        position = _position("600001")
        newer = {**position, "buy_date": date(2026, 10, 16), "buy_price": 12}
        result = build_checklist([position, newer], self.scan([_rank("600001", 1, 12)]), top_n=2,
            now=datetime(2026, 10, 16, 18, tzinfo=ZoneInfo("Asia/Shanghai")))
        self.assertEqual(result["hold"][0]["股数"], 200)
        self.assertEqual(result["hold"][0]["当前确认可卖"], 100)
        self.assertFalse(result["buy"])

    def test_after_close_universe_quote_uses_complete_bar_for_next_day_checklist(self):
        service = MarketDataService.__new__(MarketDataService)
        quote = service.quote_from_universe_row(
            {"code": "600001", "price": 99, "volume_lots": 10000, "amount_yuan": 5e7},
            {"fetched_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()},
        )
        self.assertIsNone(quote["quote_time"])
        self.assertTrue(quote["quote_valid"])
        plan = build_trade_plan(10, 0.02, 9, 8, 1, True,
            volume=10000, average_volume=9000, data_status="最近收盘全市场行情快照（非实时）",
            as_of="2026-10-16T18:00:00+08:00", evaluation_time="2026-10-16T18:00:00+08:00")
        self.assertFalse(plan["signal_active"])
        rows = ranking_snapshot([{
            "code": "600001", "quote": quote, "entry_trade_status": "可正常评估",
            "features": {"date": pd.Timestamp("2026-10-16"), "close": 10,
                         "volume": 10000, "amount": 5e7, "ma20_gap": 0.05},
            "analysis": {"models": {20: [{"validated": True, "prediction_percentile": 0.9}]}, "trade_plan": plan},
        }], "2026-10-16")
        result = self.run_checklist([], self.scan(rows))
        self.assertEqual(result["buy"][0]["参考价格"], 10)
        held = self.run_checklist([_position("600001")], self.scan(rows))
        self.assertEqual(held["hold"][0]["代码"], "600001")
        for change in ({"daily_bar_date": "2026-10-15"}, {"entry_valid_until": "2026-10-17"},
                       {"entry_generated_at": "2026-10-19"}, {"entry_valid_until": None},
                       {"quote_valid": False}, {"amount_yuan": None}):
            self.assertFalse(self.run_checklist([], self.scan([{**rows[0], **change}]))["buy"])
        before_close = build_checklist([], self.scan(rows),
            now=datetime(2026, 10, 16, 14, tzinfo=ZoneInfo("Asia/Shanghai")))
        self.assertFalse(before_close["buy"])


class MarketRegimeTests(unittest.TestCase):
    """市场统计仅使用共同交易日和当时行情。"""

    def data(self):
        dates = pd.bdate_range("2020-01-01", periods=80)
        large = pd.DataFrame({"date": dates, "close": 100 * 1.001 ** np.arange(80)})
        small = pd.DataFrame({"date": dates, "close": 100 * 1.004 ** np.arange(80)})
        histories = {f"60000{index}": pd.DataFrame({"date": dates, "close": 10 + np.arange(80) / 10,
            "amount": (np.arange(80) + 1) * 1e6, "turnover_rate": (np.arange(80) + 1) / 10}) for index in range(6)}
        exposures = pd.DataFrame({"code": list(histories), "industry": ["行业甲", "行业乙", "行业丙"] * 2,
            "industry_source": "申万一级行业（AkShare 行业成分）"})
        return histories, large, small, exposures

    def test_metrics_labels_industries_and_future_invariance(self):
        histories, large, small, exposures = self.data()
        reference = str(large.iloc[59]["date"].date())
        first = classify_market(histories, large, small, reference, exposures)
        self.assertGreater(first["relative_momentum"], 0.02)
        self.assertIn("小盘相对占优", first["label"])
        self.assertIn("放量高热", first["label"])
        self.assertEqual(len(first["top_industries"]), 3)
        # 样本口径与行业映射说明只是提示，指标已计算时不应误报降级。
        self.assertTrue(first["warnings"])
        self.assertFalse(first["degraded"])
        styled = classify_market(histories, large, small, reference, exposures, explicit_style=True)
        self.assertFalse(styled["degraded"])
        self.assertTrue(any("显式风格" in warning for warning in styled["warnings"]))
        for frame in [large, small, *histories.values()]:
            frame.loc[frame.index >= 60, "close"] = 999999
            if "amount" in frame:
                frame.loc[frame.index >= 60, "amount"] = 999999999999
        self.assertEqual(first, classify_market(histories, large, small, reference, exposures))

    def test_missing_or_misaligned_indices_explicitly_degrade(self):
        histories, large, small, exposures = self.data()
        result = classify_market(histories, large, small.iloc[:-1], str(large.iloc[-1]["date"].date()))
        self.assertIsNone(result["relative_momentum"])
        self.assertTrue(result["degraded"])
        self.assertIn("申万", "；".join(result["warnings"]))
        result = classify_market({}, None, None, "2020-01-01")
        self.assertIsNone(result["amount_yuan"])
        self.assertIsNone(result["turnover_percentile"])


class ResearchDisplayTests(unittest.TestCase):
    """真实渲染共用展示组件，缺失、部分接入与实际入模状态均可读。"""

    def test_market_and_actual_feature_status_render_without_exceptions(self):
        script = (
            "from services.research_ui import show_market_regime, show_style_status\n"
            "show_market_regime(None)\n"
            "show_style_status({'status': '覆盖已核验', 'model_feature_columns': ['xs_log_market_cap_total'], "
            "'features': {'log_market_cap_total': {'coverage': 0.9, 'accepted': True, 'quality_reasons': {'缺失': 2}}}}, "
            "['xs_log_market_cap_total'])\n"
        )
        regime = {"label": "小盘相对占优", "reference_date": "2026-10-16", "scope": "扫描样本",
            "relative_momentum": 0.03, "amount_yuan": 1e9, "amount_percentile": None,
            "turnover_percentile": 0.8, "top_amount_industries": ["银行", "电子", "机械"],
            "top_industries": [{"industry": "银行", "daily_return": 0.01, "amount_yuan": 1e8, "stocks": 10}],
            "warnings": ["部分指标未接入"]}
        with patch("services.research_ui.latest_regime", return_value=regime):
            app = AppTest.from_string(script).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 4)
        self.assertEqual(app.metric[2].value, "未接入")
        self.assertEqual(len(app.dataframe), 2)
        self.assertTrue(any("实际入模" in item.value for item in app.markdown))
        with patch("services.research_ui.latest_regime", return_value=None):
            empty = AppTest.from_string(script).run()
        self.assertFalse(empty.exception)
        self.assertIn("尚无市场状态记录", empty.info[0].value)


@unittest.skipUnless(os.environ.get("QUANT_TEST_POSTGRES") == "1", "设置QUANT_TEST_POSTGRES=1运行隔离的数据库集成测试")
class MarketRegimeDatabaseTests(unittest.TestCase):
    """随机隔离模式验证真实 SQL，测试不向业务表写入虚假市场状态。"""

    def test_roundtrip_and_old_snapshot_order(self):
        from tests.test_auth import _ScopedRepository

        schema = "quant_regime_test_" + secrets.token_hex(6)
        with psycopg.connect(**connection_settings()) as connection:
            connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
            for name in ("001_portfolio_monitor.sql", "006_market_regime.sql"):
                connection.execute((settings.PROJECT_ROOT / "sql" / name).read_text(encoding="utf-8").replace("public.", schema + "."))
        try:
            repository = _ScopedRepository(schema)
            save_regime(repository, {"reference_date": "2026-10-16", "label": "均衡", "amount_yuan": None}, "新批次", "测试")
            save_regime(repository, {"reference_date": "2026-10-15", "label": "旧快照"}, "旧批次", "测试")
            self.assertEqual(latest_regime(repository)["label"], "均衡")
        finally:
            with psycopg.connect(**connection_settings()) as connection:
                connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
