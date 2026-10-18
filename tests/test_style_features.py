"""验证历史单位、披露版本、风格覆盖门控及训练预测一致性。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from config import settings
from services.ml_pipeline import TARGET_KIND, fit_panel_model, load_context, load_context_by_id, models_for_stock, predict_panel, save_context
from services.panel_store import PanelStore
from services.style_experiments import _portfolio_metrics, _variant_configuration, comparison_fingerprint, compare_variants, experiment_panels, independent_acceptance
from services.style_factors import SIZE_COLUMNS, VALUE_COLUMNS, align_history, attach_style_factors, configuration_id_from_config, load_history, production_report_is_current, save_history
from services.valuation_data import normalize_daily_basic, reconstruct_valuation
from tests.test_ml_pipeline import _ExactModel


def _facts(days=100, stocks=6):
    """构造金额以元计、版本可追溯的历史截面。"""
    rows = []
    for date in pd.bdate_range("2020-01-01", periods=days):
        for index in range(stocks):
            cap = (index + 1) * 1e8
            rows.append({"code": f"{600000 + index:06d}", "date": date,
                "market_cap_total": cap, "market_cap_float": cap / 2,
                "pe_ttm": 10 + index, "pb": 1 + index, "ps_ttm": 2 + index,
                "net_profit_ttm": (-0.1 if index == 0 else 0.1) * cap,
                "book_equity": cap / (1 + index), "revenue_ttm": cap / (2 + index),
                "source": "测试原始披露", "available_at": date + pd.Timedelta(hours=17),
                "collected_at": pd.Timestamp("2026-10-18"), "revision_id": "原始版本",
                "unit": "CNY/share", "quality": "已核验", "size_pit": True, "value_pit": True,
                "industry": "行业甲" if index < 3 else "行业乙", "industry_pit": True})
    return pd.DataFrame(rows)


class StyleHistoryTests(unittest.TestCase):
    """历史数据只保存真实日期和版本，不填充当前值。"""

    def test_daily_basic_units_and_unverified_value(self):
        raw = pd.DataFrame({"ts_code": ["600000.SH"], "trade_date": ["20200102"],
            "total_mv": [20000], "circ_mv": [10000], "total_share": [2000],
            "float_share": [1000], "close": [10], "pe_ttm": [10], "pb": [2], "ps_ttm": [3]})
        row = normalize_daily_basic(raw).iloc[0]
        self.assertEqual(row["market_cap_total"], 2e8)
        self.assertEqual(row["total_shares"], 2e7)
        self.assertTrue(row["size_pit"])
        self.assertFalse(row["value_pit"])
        raw.loc[0, "total_mv"] *= 3
        self.assertFalse(normalize_daily_basic(raw).iloc[0]["size_pit"])

    def test_refetched_daily_basic_revision_cannot_change_past_even_after_retries(self):
        raw = pd.DataFrame({"ts_code": ["600000.SH"], "trade_date": ["20200102"],
            "total_mv": [20000], "circ_mv": [10000], "total_share": [2000],
            "float_share": [1000], "close": [10]})
        initial = normalize_daily_basic(raw)
        panel = initial[["date", "code"]]
        raw[["close", "total_mv", "circ_mv"]] *= 2
        revised = normalize_daily_basic(raw)
        with tempfile.TemporaryDirectory() as directory:
            store = PanelStore(Path(directory))
            save_history(store, initial)
            save_history(store, revised)
            save_history(store, revised)
            loaded = load_history(store, ["600000"])
            self.assertEqual(len(loaded), 2)
            self.assertEqual(align_history(panel, loaded).iloc[0]["style_market_cap_total"], 2e8)

    def test_revisions_are_preserved_but_never_backfilled(self):
        facts = _facts(1)
        revision = facts.iloc[[0]].copy()
        revision["market_cap_total"] = 999e9
        revision["available_at"] = pd.Timestamp("2020-02-01")
        revision["revision_id"] = "后续修订"
        panel = facts[["date", "code"]]
        changed = pd.concat([facts, revision], ignore_index=True)
        pd.testing.assert_frame_equal(align_history(panel, facts), align_history(panel, changed))
        with tempfile.TemporaryDirectory() as directory:
            store = PanelStore(Path(directory))
            self.assertEqual(save_history(store, facts)["new_rows"], len(facts))
            self.assertEqual(save_history(store, revision)["new_rows"], 1)
            self.assertEqual(save_history(store, revision)["new_rows"], 0, "重复版本不能再计为新增")
            loaded = load_history(store, ["600000"])
            self.assertEqual(len(loaded), 2)
            mutated = facts.iloc[[0]].copy()
            mutated["market_cap_total"] = 123
            save_history(store, mutated)
            self.assertEqual(store.read_partition("valuation", "600000").iloc[0]["market_cap_total"], 1e8)

    def test_date_alignment_has_no_forward_fill(self):
        facts = _facts(1)
        panel = pd.concat([facts[["date", "code"]], facts[["date", "code"]].assign(date=pd.Timestamp("2020-01-02"))])
        aligned = align_history(panel, facts)
        self.assertTrue(aligned.loc[aligned["date"].eq("2020-01-02"), "style_market_cap_total"].isna().all())
        with self.assertRaises(ValueError):
            save_history(PanelStore(), facts.assign(unit="万元"))

    def test_announcement_revision_ttm_and_capital_events(self):
        metadata = {"code": "600000", "source": "公告版本", "collected_at": pd.Timestamp("2026-10-18"),
                    "unit": "CNY/share", "pit_verified": True}
        shares = pd.DataFrame([
            {**metadata, "ann_date": "2019-01-01", "effective_date": "2019-01-02", "revision_id": "股本1", "total_shares": 100, "float_shares": 80},
            {**metadata, "ann_date": "2020-04-29", "effective_date": "2020-04-30", "revision_id": "股本2", "total_shares": 200, "float_shares": 160}])
        statements = pd.DataFrame([
            {**metadata, "report_period": "2019-03-31", "ann_date": "2019-04-29", "revision_id": "前年一季", "net_profit_ytd": 10, "revenue_ytd": 100, "book_equity": 200},
            {**metadata, "report_period": "2019-12-31", "ann_date": "2020-03-31", "revision_id": "年报", "net_profit_ytd": 50, "revenue_ytd": 500, "book_equity": 220},
            {**metadata, "report_period": "2020-03-31", "ann_date": "2020-04-29", "revision_id": "当期一季", "net_profit_ytd": -5, "revenue_ytd": 120, "book_equity": 225},
            {**metadata, "report_period": "2020-03-31", "ann_date": "2020-05-05", "revision_id": "一季修订", "net_profit_ytd": -10, "revenue_ytd": 110, "book_equity": 215}])
        prices = pd.DataFrame({"code": "600000", "date": pd.to_datetime(["2020-04-29", "2020-04-30", "2020-05-06"]), "raw_close": [10, 5, 5]})
        result = reconstruct_valuation(prices, shares, statements)
        self.assertEqual(result["market_cap_total"].tolist(), [1000, 1000, 1000])
        self.assertEqual(result["net_profit_ttm"].tolist(), [50, 35, 30])
        self.assertEqual(result["revenue_ttm"].tolist(), [500, 520, 510])
        # 添加未来公告不能改变过去任一日的输入。
        baseline = reconstruct_valuation(prices.iloc[:2], shares, statements.iloc[:3])
        pd.testing.assert_frame_equal(result.iloc[:2].reset_index(drop=True), baseline)


class StyleFactorTests(unittest.TestCase):
    """显式风格特征须按截面处理并通过覆盖门控。"""

    def test_negative_earnings_and_nonpositive_values(self):
        facts = _facts(2)
        facts.loc[1, "market_cap_total"] = 0
        facts.loc[2, "book_equity"] = -1
        facts.loc[3, "revenue_ttm"] = -1
        facts.loc[4, "net_profit_ttm"] = np.nan
        facts.loc[4, "pe_ttm"] = -10
        with patch.object(settings, "STYLE_WINSOR_QUANTILES", (0, 1)):
            frame, report = attach_style_factors(facts[["date", "code"]], facts, size=True, value=True, research=True)
        self.assertLess(frame.iloc[0]["earnings_yield"], 0)
        self.assertTrue(pd.isna(frame.iloc[1]["log_market_cap_total"]))
        self.assertTrue(pd.isna(frame.iloc[2]["book_to_price"]))
        self.assertTrue(pd.isna(frame.iloc[3]["sales_to_price"]))
        self.assertTrue(pd.isna(frame.iloc[4]["earnings_yield"]))
        self.assertIn("quality_reasons", report["features"]["earnings_yield"])

    def test_missing_data_and_production_approval_fail_closed(self):
        facts = _facts(100)
        _, report = attach_style_factors(facts[["date", "code"]], None, size=True, value=True, research=True)
        self.assertEqual(report["accepted_columns"], [])
        with tempfile.TemporaryDirectory() as directory, patch.object(settings, "STYLE_APPROVAL_PATH", Path(directory) / "不存在.json"):
            _, report = attach_style_factors(facts[["date", "code"]], facts, size=True)
        self.assertEqual(report["accepted_columns"], [])
        self.assertIn("独立验收", report["status"])

    def test_revoked_acceptance_also_blocks_existing_model_and_scan_caches(self):
        from services.scan_store import load_scan_result
        from services.style_factors import configuration_id, processing_config

        report = {"protocol": settings.STYLE_FEATURE_PROTOCOL, "configuration_id": configuration_id(size=True, value=False),
            "config": processing_config(size=True, value=False), "requested": {"size": True, "value": False},
            "accepted_columns": ["log_market_cap_total"]}
        approval = {"protocol": settings.STYLE_FEATURE_PROTOCOL, "model_version": settings.MODEL_VERSION,
            "configuration_id": report["configuration_id"], "variant": "size", "passed": True}
        context = {"context_id": "a" * 32, "model_version": settings.MODEL_VERSION,
            "target_kind": TARGET_KIND, "style_factors": report,
            "feature_configuration_id": report["configuration_id"],
            "reference_date": "2026-10-16", "reference_closes": {"600000": 10.0},
            "models": {20: [{"model": "LightGBM", "validated": True}]},
            "predictions": {20: {"600000": {"LightGBM": {"prediction_percentile": 0.9}}}}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "acceptance.json"
            scan_path = root / "scan.json"
            with patch.object(settings, "MODEL_DIR", root), patch.object(settings, "DATA_DIR", root), patch.object(settings, "STYLE_APPROVAL_PATH", target):
                target.write_text(json.dumps(approval), encoding="utf-8")
                save_context(context, pd.DataFrame({"code": ["600000"]}))
                archive = root / settings.MODEL_ARCHIVE_DIRNAME / context["context_id"]
                archive.mkdir(parents=True)
                (archive / f"market_context_v{settings.MODEL_VERSION}.joblib").write_bytes(
                    (root / f"market_context_v{settings.MODEL_VERSION}.joblib").read_bytes())
                scan_path.write_text(json.dumps({"model_version": settings.MODEL_VERSION, "style_factors": report}), encoding="utf-8")
                self.assertIsNotNone(load_context()[0])
                cached, _ = load_context_by_id(context["context_id"])
                self.assertIsNotNone(cached)
                self.assertTrue(models_for_stock(cached, "600000", "2026-10-16", 10.0, 20)[0]["validated"])
                self.assertIsNotNone(load_scan_result(scan_path))
                target.write_text(json.dumps({**approval, "passed": False}), encoding="utf-8")
                self.assertIsNone(load_context()[0])
                self.assertIsNone(load_context_by_id(context["context_id"])[0])
                rejected = models_for_stock(cached, "600000", "2026-10-16", 10.0, 20)
                self.assertTrue(all(not row["validated"] and row["prediction_percentile"] is None for row in rejected))
                self.assertIsNone(load_scan_result(scan_path))

    def test_acceptance_is_bound_to_actual_base_features_and_processing_rules(self):
        facts = _facts(1)
        panel = facts[["date", "code"]].copy()
        panel.attrs["model_feature_columns"] = ["xs_momentum_5"]
        _, report = attach_style_factors(panel, facts, size=True, value=False, research=True)
        with tempfile.TemporaryDirectory() as directory, patch.object(settings, "STYLE_APPROVAL_PATH", Path(directory) / "acceptance.json"):
            settings.STYLE_APPROVAL_PATH.write_text(json.dumps({
                "passed": True, "variant": "size", "protocol": settings.STYLE_FEATURE_PROTOCOL,
                "model_version": settings.MODEL_VERSION, "configuration_id": report["configuration_id"],
            }), encoding="utf-8")
            self.assertTrue(production_report_is_current(report))
            _, accepted = attach_style_factors(panel, facts, size=True, value=False)
            self.assertTrue(accepted["accepted_columns"])
            for columns, neutralized in ((["neutral_momentum_5"], True), (["xs_momentum_5", "xs_chip"], False)):
                panel.attrs["model_feature_columns"] = columns
                _, changed = attach_style_factors(panel, facts, size=True, value=False, neutralization_enabled=neutralized)
                self.assertNotEqual(changed["configuration_id"], report["configuration_id"])
                self.assertFalse(changed["accepted_columns"])
            for name in ("EXTRA_FACTOR_MINIMUM_COVERAGE", "NEUTRALIZATION_MINIMUM_STOCKS", "NEUTRALIZATION_MINIMUM_INDUSTRY_STOCKS"):
                with patch.object(settings, name, getattr(settings, name) + 1):
                    self.assertFalse(production_report_is_current(report))

    def test_future_coverage_and_extremes_cannot_change_past(self):
        facts = _facts(100)
        panel = facts[["date", "code"]]
        first, report = attach_style_factors(panel, facts, size=True, value=True, research=True)
        altered = facts.copy()
        future = altered["date"].ge(pd.Timestamp("2020-05-01"))
        altered.loc[future, "market_cap_total"] = np.nan
        changed, new_report = attach_style_factors(panel, altered, size=True, value=True, research=True)
        pd.testing.assert_frame_equal(first.loc[first["date"].lt("2020-05-01")], changed.loc[changed["date"].lt("2020-05-01")])
        self.assertEqual(report["accepted_columns"], new_report["accepted_columns"])

    def test_features_flow_into_real_training_and_prediction(self):
        from tests.test_ml_pipeline import _panel

        base = _panel(days=420, stocks=10)
        facts = _facts(420, 10)
        base["code"] = facts["code"].to_numpy()
        base.attrs["model_feature_columns"] = ["xs_momentum_5"]
        panel, report = attach_style_factors(base, facts, size=True, value=True, research=True)
        panel.attrs["model_feature_columns"] = ["xs_momentum_5", *report["model_feature_columns"]]
        model = fit_panel_model(panel, "LightGBM", 20, model_factory=lambda _: _ExactModel(), objective="regression")
        self.assertTrue(model["validated"], model["status"])
        self.assertIn("xs_log_market_cap_total", model["features_used"])
        self.assertFalse(any(column.startswith("neutral_") for column in report["model_feature_columns"]))
        predicted = predict_panel(model, panel.loc[panel["date"].eq(panel["date"].max())])
        self.assertEqual(len(predicted), 10)


class StyleExperimentTests(unittest.TestCase):
    """独立验收不会反复消费区间或把失败变成通过。"""

    def test_four_variants_use_the_same_samples_and_actual_features(self):
        from tests.test_ml_pipeline import _panel

        base = _panel(days=540, stocks=10)
        facts = _facts(540, 10)
        base["code"] = facts["code"].to_numpy()
        base.attrs["model_feature_columns"] = ["xs_momentum_5"]
        dates = sorted(base["date"].unique())
        start, end = str(pd.Timestamp(dates[350]).date()), str(pd.Timestamp(dates[500]).date())
        with patch("services.style_experiments.build_market_panel", return_value=base):
            panels = experiment_panels({}, pd.DataFrame(), facts, start)
        keys = {name: list(zip(frame["date"], frame["code"])) for name, frame in panels.items()}
        self.assertTrue(all(value == keys["base"] for value in keys.values()))
        report = compare_variants(panels, start, end, "regression", lambda _: _ExactModel())
        self.assertEqual(set(report["results"]), {"base", "size", "value", "size_value"})
        for name, result in report["results"].items():
            self.assertTrue(result["usable"], result["status"])
            self.assertTrue(result["all_requested_features_available"])
            self.assertIsNotNone(result["metrics"]["raw_label_mae"])
            self.assertGreater(result["portfolio"]["periods"], 0)
        self.assertEqual(report["results"]["base"]["features_used"], ["xs_momentum_5"])
        self.assertIn("xs_log_market_cap_float", report["results"]["size_value"]["features_used"])
        # 验收报告必须能被相同实际特征和目标的生产入口识别。
        comparison = {**report, "selected_variant": "size"}
        baseline = {"usable": True, "features_used": ["xs_momentum_5"],
            "metrics": {"rank_ic": 0.1},
            "portfolio": {"after_cost_excess_return": 0.01, "maximum_drawdown": -0.2}}
        candidate = {"usable": True, "features_used": ["xs_momentum_5", *[f"xs_{column}" for column in SIZE_COLUMNS]],
            "metrics": {"validated": True, "rank_ic": 0.2, "raw_label_mae": 0.01, "raw_label_baseline_mae": 0.02},
            "portfolio": {"after_cost_excess_return": 0.02, "maximum_drawdown": -0.1,
                          "industry_pit_coverage": 1.0, "average_log_market_cap": 20}}
        with tempfile.TemporaryDirectory() as directory, patch.object(settings, "STYLE_APPROVAL_PATH", Path(directory) / "acceptance.json"):
            with patch("services.style_experiments.evaluate_variant", side_effect=[baseline, candidate]):
                accepted = independent_acceptance(panels, comparison, "2022-01-01", "2022-03-01")
            self.assertTrue(accepted["passed"])
            _, production = attach_style_factors(base, facts, size=True, value=False, objective="regression")
            self.assertTrue(production["accepted_columns"])
            self.assertEqual(production["configuration_id"], accepted["configuration_id"])
        original = comparison_fingerprint(panels, end)
        # 只追加未来数据不改变对照证据；改动过去特征必须使旧验收失效。
        changed = {name: frame.copy() for name, frame in panels.items()}
        for frame in changed.values():
            frame.loc[frame["date"].gt(end), "xs_momentum_5"] = 999
        self.assertEqual(original, comparison_fingerprint(changed, end))
        changed["base"].loc[0, "xs_momentum_5"] = 999
        self.assertNotEqual(original, comparison_fingerprint(changed, end))

    def test_independent_interval_must_follow_validation_and_is_single_use(self):
        from services.style_factors import configuration_id

        comparison = {"selected_variant": "size", "configuration_id": configuration_id("regression"),
            "model_version": settings.MODEL_VERSION, "validation_end": "2020-06-01",
            "objective": "regression", "experiment_id": "测试实验"}
        panels = {name: pd.DataFrame({"date": pd.to_datetime([]), "code": pd.Series(dtype=str)}) for name in ("base", "size")}
        comparison["data_fingerprint"] = comparison_fingerprint(panels, comparison["validation_end"])
        comparison["variant_configuration_ids"] = {name: configuration_id_from_config(_variant_configuration(panel, name, "regression"))
                                                  for name, panel in panels.items()}
        failed = {"usable": False, "features_used": [], "metrics": None, "portfolio": None}
        with tempfile.TemporaryDirectory() as directory, patch("services.style_experiments.evaluate_variant", return_value=failed):
            target = Path(directory) / "acceptance.json"
            panels["size"].attrs["model_feature_columns"] = ["neutral_momentum_5"]
            with self.assertRaisesRegex(ValueError, "实际入模特征"):
                independent_acceptance(panels, comparison, "2020-07-01", "2020-10-01", target)
            self.assertFalse(target.exists())
            panels["size"].attrs.pop("model_feature_columns")
            with self.assertRaises(ValueError):
                independent_acceptance(panels, comparison, "2020-05-01", "2020-10-01", target)
            result = independent_acceptance(panels, comparison, "2020-07-01", "2020-10-01", target)
            self.assertFalse(result["passed"])
            self.assertEqual(result["configuration_id"], comparison["variant_configuration_ids"]["size"])
            self.assertTrue(target.exists())
            with self.assertRaises(ValueError):
                independent_acceptance(panels, comparison, "2020-07-01", "2020-10-01", target)
            with self.assertRaises(ValueError):
                independent_acceptance(panels, comparison, "2020-08-01", "2020-11-01", target)

    def test_equal_weight_rebalance_charges_drift_even_with_unchanged_members(self):
        dates = pd.bdate_range("2020-01-01", periods=21)
        panel = pd.DataFrame({"date": dates, "code": "0"})
        evaluation = pd.DataFrame([{
            "date": date, "code": str(index), "prediction": 10 - index,
            "absolute_return": 1.0 if period == 0 and index == 0 else 0.0,
            "actual": 1.0 if period == 0 and index == 0 else 0.0,
        } for period, date in enumerate((dates[0], dates[20])) for index in range(10)])
        with patch("services.style_experiments.cost_rates", return_value=(0.01, 0.01)):
            result = _portfolio_metrics(evaluation, panel, 20)
        # 首期等权持仓涨成 2/3 和 1/3，次期各买卖 1/6，末期还需平仓。
        expected = (1.5 - 0.01) * (1 - (1 / 6) * 0.02 - 0.01) - 1
        self.assertAlmostEqual(result["net_total_return"], expected)
        self.assertAlmostEqual(result["average_turnover"], (0.5 + 1 / 6) / 2)
