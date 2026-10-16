"""验证列式面板存储、DuckDB 截面计算与市值行业中性化。"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from services.exposures import build_exposure_frame, parse_sw_records
from services.feature_engineering import FEATURE_COLUMNS, normalize_cross_section
from services.ml_pipeline import build_market_panel
from services.neutralization import attach_market_cap_exposure, neutralize_cross_section
from services.panel_store import PanelStore


def _history(code="600000", days=30):
    dates = pd.bdate_range("2024-01-01", periods=days)
    return pd.DataFrame({
        "date": dates, "open": 10.0, "high": 10.2, "low": 9.8,
        "close": 10.0 + np.arange(days) * 0.1, "volume": 1000.0, "code": code,
    })


def _panel(day_count=4, stock_count=8):
    dates = pd.bdate_range("2024-01-01", periods=day_count)
    rows = []
    for day, date in enumerate(dates):
        for stock in range(stock_count):
            market_cap = 10.0 + stock
            rows.append({
                "date": date, "code": f"6000{stock:02d}", "close": 10.0 + stock,
                "market_cap_float": market_cap * 1e8, "price": 10.0 + stock, "industry": f"行业{stock % 3}",
                "momentum_5": stock * 1.0 + day * 0.1, "momentum_20": stock * 2.0, "volatility_20": stock * -0.5,
            })
    return pd.DataFrame(rows)


class PanelStoreTests(unittest.TestCase):
    """逐股 Parquet 与因子面板必须可原子读写并保持数值一致。"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = PanelStore(root=Path(self.directory.name) / "panel")

    def tearDown(self):
        self.directory.cleanup()

    def test_history_round_trip_keeps_values_and_codes(self):
        frame = _history()
        path = self.store.write_history("600000", frame, meta={"source": "单元测试"})
        self.assertTrue(path.exists())
        self.assertEqual(path.name, "code=600000.parquet")
        restored = self.store.read_history("600000")
        self.assertEqual(len(restored), len(frame))
        self.assertTrue(np.allclose(restored["close"], frame["close"]))
        self.assertEqual(restored["code"].iloc[0], "600000")
        self.assertEqual(self.store.history_codes(), ["600000"])
        self.assertIsNone(self.store.read_history("999999"))

    def test_panel_round_trip_and_summary(self):
        panel = _panel()
        self.store.save_panel(panel, version="test", context_id="ctx")
        restored = self.store.load_panel(version="test")
        self.assertEqual(len(restored), len(panel))
        self.assertEqual(restored.attrs.get("context_id"), "ctx")
        summary = self.store.summary()
        self.assertGreaterEqual(summary["history_files"], 0)
        self.assertGreaterEqual(summary["panel_files"], 1)
        self.assertGreater(summary["bytes"], 0)
        self.assertIn("engine", summary)
        self.assertTrue(summary["engine"])

    def test_cross_section_zscore_matches_pandas_implementation(self):
        panel = _panel(day_count=6, stock_count=10)
        columns = ["momentum_5", "momentum_20", "volatility_20"]
        expected = normalize_cross_section(panel, columns, 5, 0.8)
        actual = self.store.cross_section_zscore(panel, columns, minimum_stocks=5, minimum_coverage=0.8)
        for column in columns:
            self.assertTrue(
                np.allclose(actual[f"xs_{column}"], expected[f"xs_{column}"], equal_nan=True),
                f"{column} 与 pandas 结果不一致",
            )
        self.assertTrue(actual[["date", "code"]].equals(expected[["date", "code"]]))

    def test_exposures_round_trip_and_meta(self):
        frame = build_exposure_frame(pd.DataFrame({
            "code": ["600000"], "name": ["浦发银行"], "price": [10.0],
            "market_cap_total": [1e11], "market_cap_float": [8e10], "industry": ["银行"],
        }))
        self.store.save_exposures(frame)
        self.store.save_exposures_meta({"exposures": {"status": "单元测试"}})
        restored = self.store.load_exposures()
        self.assertEqual(restored["code"].iloc[0], "600000")
        self.assertEqual(self.store.exposures_meta()["exposures"]["status"], "单元测试")


class NeutralizationTests(unittest.TestCase):
    """残差必须与市值和行业暴露正交，且不引入跨日信息。"""

    def test_residual_is_orthogonal_to_exposures(self):
        # 截面股票数必须高于中性化最小样本要求，否则当日会被跳过。
        panel = _panel(day_count=3, stock_count=21)
        panel["log_market_cap"] = np.log(panel["market_cap_float"])
        result, report = neutralize_cross_section(panel, ["momentum_5", "momentum_20"])
        self.assertEqual(report["dates"], 3)
        self.assertEqual(report["skipped_dates"], 0)
        self.assertIn("neutral_momentum_5", result.columns)
        for _, group in result.groupby("date"):
            self.assertAlmostEqual(float(group["neutral_momentum_5"].mean()), 0.0, places=9)
            correlation = group["neutral_momentum_5"].corr(group["log_market_cap"])
            self.assertLess(abs(float(correlation)), 1e-6)
        self.assertGreater(report["removed_variance"]["momentum_20"], 0.0)

    def test_small_cross_section_is_skipped_not_fabricated(self):
        panel = _panel(day_count=2, stock_count=4)
        panel["log_market_cap"] = np.log(panel["market_cap_float"])
        result, report = neutralize_cross_section(panel, ["momentum_5"], minimum_stocks=20)
        self.assertEqual(report["dates"], 0)
        self.assertEqual(report["skipped_dates"], 2)
        self.assertTrue(result["neutral_momentum_5"].isna().all())
        self.assertIn("没有满足最小截面要求", report["status"])

    def test_missing_factor_is_distinguished_from_thin_cross_section(self):
        panel = _panel(day_count=2, stock_count=21)
        panel["log_market_cap"] = np.log(panel["market_cap_float"])
        panel["momentum_20"] = np.nan  # 模拟某一因子在数据源中整体缺失
        _, report = neutralize_cross_section(panel, ["momentum_5", "momentum_20"])
        self.assertEqual(report["dates"], 0)
        self.assertEqual(report["factor_missing_dates"], 2)
        self.assertIn("因子存在整体缺失", report["status"])

    def test_missing_exposure_column_returns_report_only(self):
        panel = _panel()
        result, report = neutralize_cross_section(panel, ["momentum_5"])
        self.assertEqual(report["status"], "缺少因子或暴露列，未执行中性化")
        self.assertTrue(result["neutral_momentum_5"].isna().all())

    def test_historical_market_cap_is_approximated_and_reported(self):
        panel = _panel(day_count=2, stock_count=6)
        exposures = pd.DataFrame({
            "code": [f"6000{stock:02d}" for stock in range(6)],
            "market_cap_float": [(10.0 + stock) * 1e8 for stock in range(6)],
            "price": [10.0 + stock for stock in range(6)],
            "industry": [f"行业{stock % 3}" for stock in range(6)],
        })
        result, report = attach_market_cap_exposure(panel, exposures)
        self.assertEqual(report["codes_with_exposure"], 6)
        expected = np.log(1e8 * panel["close"])
        self.assertTrue(np.allclose(result["log_market_cap"], expected))
        self.assertEqual(result["industry"].iloc[0], "行业0")
        self.assertIn("近似", report["limitation"])

    def test_market_panel_adopts_neutralized_features(self):
        panel = _panel(day_count=4, stock_count=21)
        panel["log_market_cap"] = np.log(panel["market_cap_float"])
        result, report = neutralize_cross_section(panel, ["momentum_5"])
        self.assertEqual(report["dates"], 4)
        self.assertIn("neutral_momentum_5", result.columns)

    def test_build_market_panel_without_exposure_keeps_zscore_features(self):
        dates = pd.bdate_range("2020-01-01", periods=300)
        histories = {}
        for stock in range(8):
            frame = pd.DataFrame({
                "date": dates, "open": 10.0, "high": 10.5, "low": 9.5,
                "close": 10.0 * (1 + 0.001 * (stock + 1)) ** np.arange(300),
                "volume": 1000.0 + stock, "amount": 1e6,
            })
            histories[f"6000{stock:02d}"] = frame
        benchmark = pd.DataFrame({"date": dates, "close": 3000.0 * 1.0005 ** np.arange(300)})
        panel = build_market_panel(histories, benchmark)
        self.assertEqual(len(panel.attrs["model_feature_columns"]), len(FEATURE_COLUMNS))
        self.assertIn("未启用", panel.attrs["neutralization"]["status"])
        self.assertEqual(panel.attrs["exposures"]["status"], "未接入市值与行业暴露")

    def test_build_market_panel_uses_neutral_factors_when_covered(self):
        dates = pd.bdate_range("2020-01-01", periods=300)
        histories, codes = {}, []
        for stock in range(30):
            code = f"6000{stock:02d}"
            codes.append(code)
            close = (10.0 + stock) * (1 + 0.001 * (stock + 1)) ** np.arange(300)
            # 成交量必须逐日波动，否则量价相关系数与量价背离因子恒为缺失。
            volume = (1000.0 + stock) * (1 + 0.3 * np.sin(np.arange(300) * 0.35 + stock))
            histories[code] = pd.DataFrame({
                "date": dates, "open": 10.0, "high": 10.5, "low": 9.5, "close": close,
                "volume": volume, "amount": volume * close, "turnover_rate": 1.0 + stock,
            })
        benchmark = pd.DataFrame({"date": dates, "close": 3000.0 * 1.0005 ** np.arange(300)})
        exposures = pd.DataFrame({
            "code": codes,
            "market_cap_float": [(10.0 + stock) * 1e9 for stock in range(30)],
            "price": [10.0 + stock for stock in range(30)],
            "industry": [f"行业{stock % 5}" for stock in range(30)],
        })
        panel = build_market_panel(histories, benchmark, exposures=exposures)
        self.assertEqual(panel.attrs["model_feature_columns"], [f"neutral_{column}" for column in FEATURE_COLUMNS])
        self.assertEqual(panel.attrs["neutralization"]["model_features"], len(FEATURE_COLUMNS))
        self.assertIn("已中性化", panel.attrs["neutralization"]["status"])
        self.assertIn("neutral_momentum_5", panel.columns)


class ExposureParsingTests(unittest.TestCase):
    """申万行业成分解析必须分开成功记录与逐行业失败原因。"""

    def test_parses_members_and_records_failures(self):
        lines = [
            '{"code": "000001", "industry": "银行"}',
            '{"code": "600000", "industry": "银行"}',
            '{"industry": "煤炭", "error": "ConnectionError: 超时"}',
            "",
            "不是JSON",
        ]
        members, failures = parse_sw_records(lines)
        self.assertEqual([item["code"] for item in members], ["000001", "600000"])
        self.assertEqual(members[0]["industry"], "银行")
        self.assertEqual(len(failures), 2)
        self.assertTrue(any("煤炭" in item for item in failures))

    def test_exposure_frame_falls_back_to_eastmoney_industry(self):
        universe = pd.DataFrame({
            "code": [1, "600000"], "name": ["平安银行", "浦发银行"], "price": [11.0, 10.0],
            "market_cap_total": [2e11, 1e11], "market_cap_float": [1e11, 8e10], "industry": ["银行", "银行"],
        })
        frame = build_exposure_frame(universe)
        self.assertEqual(frame["code"].tolist(), ["000001", "600000"])
        self.assertTrue(frame["industry_source"].str.contains("东方财富").all())
        self.assertEqual(frame["industry_em"].iloc[0], "银行")
        industry_map = pd.DataFrame({"code": ["600000"], "industry": ["申万银行"]})
        merged = build_exposure_frame(universe, industry_map)
        self.assertEqual(merged.loc[merged["code"].eq("600000"), "industry"].iloc[0], "申万银行")
        self.assertEqual(merged.loc[merged["code"].eq("000001"), "industry"].iloc[0], "银行")


if __name__ == "__main__":
    unittest.main()
