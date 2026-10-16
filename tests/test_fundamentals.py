"""PIT 财务面板：公告日与报表数值的合并，以及按生效日的严格 ASOF 对齐。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from services.fundamentals import (
    COLUMNS, STATEMENT_FIELDS, STATEMENT_UNION_FIELDS, VALUE_COLUMNS, _fetch_statement_templates,
    align_asof_pit, merge_fundamental_panels,
)
from services.market_data import MarketDataError
from services.panel_store import PanelStore


def _epoch_ms(day: str) -> int:
    """巨潮的发布时间是毫秒时间戳，按北京时间取自然日。"""
    return int(pd.Timestamp(day, tz="Asia/Shanghai").timestamp() * 1000)


def _values_row(code: str, report_date: str, **fields: float) -> dict:
    """东财原始形状：代码列叫 SECURITY_CODE，报告期叫 REPORT_DATE。"""
    row: dict = {
        "SECURITY_CODE": code,
        "REPORT_DATE": f"{report_date} 00:00:00",
        "report_period": pd.Timestamp(report_date),
        "report_kind": "annual",
    }
    for column in VALUE_COLUMNS:
        row[column] = fields.get(column, np.nan)
    return row


class _StubClient:
    """网络出口替身：按 reportName 返回东财形状的响应，不发起任何请求。"""

    def __init__(self, payloads: dict[str, list[dict]] | None = None, failing: set[str] | None = None) -> None:
        self.payloads = payloads or {}
        self.failing = failing or set()
        self.requested: list[str] = []

    def get_json(self, url: str, params: dict, **_kwargs) -> dict:
        name = params["reportName"]
        self.requested.append(name)
        if name in self.failing:
            raise MarketDataError(f"模板不可用 {name}")
        # 东财会为请求的每个字段都返回一列、取不到的填 null；替身照此补齐，列集合才与线上一致。
        wanted = [column for column in str(params["columns"]).split(",") if column]
        rows = [{**dict.fromkeys(wanted), **row} for row in self.payloads.get(name, [])]
        return {"success": True, "result": {"data": rows, "count": len(rows)}}


def _template_row(code: str, report_date: str, **fields: float) -> dict:
    row: dict = {"SECURITY_CODE": code, "REPORT_DATE": f"{report_date} 00:00:00"}
    row.update(fields)
    return row


def _templates(statement: str, *names: str) -> list[tuple[str, tuple[str, ...]]]:
    """按真实定义取出指定模板及其科目，避免测试自带一份会与产品漂移的科目表。"""
    declared = dict(STATEMENT_FIELDS[statement])
    return [(name, declared[name]) for name in names]


def _balance_templates(*names: str) -> list[tuple[str, tuple[str, ...]]]:
    return _templates("balance", *names)


def _cashflow_templates(*names: str) -> list[tuple[str, tuple[str, ...]]]:
    return _templates("cashflow", *names)


class StatementTemplateTests(unittest.TestCase):
    """资产负债表与现金流量表按行业分四套模板；只取通用模板会静默丢掉整个金融板块。"""

    def test_balance_and_cashflow_cover_every_industry_template(self) -> None:
        """通用模板不含银行／保险／证券，只取它等于让整个金融板块拿不到这两张表。"""
        for statement, stem in (("balance", "BALANCE"), ("cashflow", "CASHFLOW")):
            with self.subTest(statement=statement):
                names = [name for name, _fields in STATEMENT_FIELDS[statement]]
                self.assertEqual(names, [f"{prefix}{stem}" for prefix in ("G", "B", "I", "S")])

    def test_income_only_needs_the_general_template(self) -> None:
        """利润表的通用模板本身就覆盖全部行业，多取三套只会让同一只股票出现重复行。"""
        names = [name for name, _fields in STATEMENT_FIELDS["income"]]
        self.assertEqual(names, ["GINCOME"])

    def test_financial_balance_templates_ask_for_no_current_asset_split(self) -> None:
        """金融行业资产负债表没有流动／非流动划分；点名请求不存在的科目会让整次请求失败。"""
        templates = dict(STATEMENT_FIELDS["balance"])
        for name in ("BBALANCE", "IBALANCE", "SBALANCE"):
            with self.subTest(template=name):
                self.assertEqual(
                    [field for field in templates[name] if field.startswith("TOTAL_CURRENT")], []
                )
        self.assertIn("TOTAL_CURRENT_ASSETS", templates["GBALANCE"])

    def test_union_fields_keep_every_general_column(self) -> None:
        """并集列集合必须保住通用模板的科目，否则合并后这些列会整列消失。"""
        self.assertEqual(
            list(STATEMENT_UNION_FIELDS["balance"]), list(dict(STATEMENT_FIELDS["balance"])["GBALANCE"])
        )

    def _columns(self, statement: str = "balance") -> list[str]:
        """该报表的并集列集合，与 fetch_statement_values 传给抓取层的完全一致。"""
        return [*COLUMNS, *STATEMENT_UNION_FIELDS[statement]]

    def test_templates_are_concatenated_into_one_frame(self) -> None:
        columns = self._columns()
        client = _StubClient(
            {
                "RPT_F10_FINANCE_GBALANCE": [_template_row("600519", "2023-12-31", TOTAL_ASSETS=2.7e11)],
                "RPT_F10_FINANCE_BBALANCE": [_template_row("000001", "2023-12-31", TOTAL_ASSETS=5.6e12)],
            }
        )
        frame = _fetch_statement_templates(
            client, 5, _balance_templates("GBALANCE", "BBALANCE", "IBALANCE", "SBALANCE"),
            columns, "2023-12-31", 500,
        )
        self.assertEqual(sorted(frame["SECURITY_CODE"]), ["000001", "600519"])
        self.assertEqual(list(frame.columns), columns)

    def test_bank_row_survives_the_cross_statement_merge(self) -> None:
        """银行走 B 模板：资产表与现金流表并集后必须同时拿到总资产与经营现金流。

        两者缺一，财务健康与经营现金流因子对银行就永远落空；这也正是修复前 000001 的表现。
        """
        client = _StubClient(
            {
                "RPT_F10_FINANCE_BBALANCE": [
                    _template_row("000001", "2023-12-31", TOTAL_ASSETS=5.6e12)
                ],
                "RPT_F10_FINANCE_BCASHFLOW": [
                    _template_row("000001", "2023-12-31", NETCASH_OPERATE=9.25e10)
                ],
            }
        )
        # 与 fetch_statement_values 一致：两张表按 (代码, 报告期) 外连接成一行。
        balance = _fetch_statement_templates(
            client, 5, _balance_templates("GBALANCE", "BBALANCE"), self._columns(), "2023-12-31", 500
        )
        cashflow = _fetch_statement_templates(
            client, 5, _cashflow_templates("GCASHFLOW", "BCASHFLOW"),
            self._columns("cashflow"), "2023-12-31", 500,
        )
        merged = balance.merge(cashflow, on=COLUMNS, how="outer", validate="one_to_one")
        row = merged.loc[merged["SECURITY_CODE"] == "000001"].iloc[0]
        self.assertAlmostEqual(float(row["TOTAL_ASSETS"]), 5.6e12, places=0)
        self.assertAlmostEqual(float(row["NETCASH_OPERATE"]), 9.25e10, places=0)

    def test_same_code_across_templates_keeps_the_first(self) -> None:
        client = _StubClient(
            {
                "RPT_F10_FINANCE_GBALANCE": [_template_row("600519", "2023-12-31", TOTAL_ASSETS=1.0)],
                "RPT_F10_FINANCE_BBALANCE": [_template_row("600519", "2023-12-31", TOTAL_ASSETS=2.0)],
            }
        )
        frame = _fetch_statement_templates(
            client, 5, _balance_templates("GBALANCE", "BBALANCE"), self._columns(), "2023-12-31", 500
        )
        self.assertEqual(len(frame), 1)
        self.assertAlmostEqual(float(frame.iloc[0]["TOTAL_ASSETS"]), 1.0, places=9)

    def test_all_templates_empty_returns_empty_frame_with_columns(self) -> None:
        columns = self._columns()
        frame = _fetch_statement_templates(
            _StubClient(), 5, _balance_templates("GBALANCE", "BBALANCE"), columns, "2023-12-31", 500
        )
        self.assertTrue(frame.empty)
        self.assertEqual(list(frame.columns), columns)

    def test_every_template_is_requested(self) -> None:
        client = _StubClient()
        _fetch_statement_templates(
            client, 5, _balance_templates("GBALANCE", "BBALANCE"), self._columns(), "2023-12-31", 500
        )
        self.assertEqual(client.requested, ["RPT_F10_FINANCE_GBALANCE", "RPT_F10_FINANCE_BBALANCE"])

    def test_any_template_failure_propagates(self) -> None:
        """任一模板失败必须向上抛出：少取一套就是静默丢掉一整个板块，覆盖率门控看不出来。"""
        client = _StubClient(
            {"RPT_F10_FINANCE_GBALANCE": [_template_row("600519", "2023-12-31", TOTAL_ASSETS=1.0)]},
            failing={"RPT_F10_FINANCE_IBALANCE"},
        )
        with self.assertRaises(MarketDataError):
            _fetch_statement_templates(
                client, 5, _balance_templates("GBALANCE", "IBALANCE"), self._columns(), "2023-12-31", 500
            )


class MergeFundamentalPanelsTests(unittest.TestCase):
    def _calendar(self, rows: list[tuple[str, str, str]] | None = None) -> pd.DataFrame:
        rows = rows or [("600519", "2023-12-31", "2024-04-03"), ("000001", "2023-12-31", "2024-03-15")]
        return pd.DataFrame(
            [
                {
                    "code": code,
                    "report_period": pd.Timestamp(period),
                    "report_kind": "annual",
                    "ann_date": _epoch_ms(announced),
                    "title": f"{period[:4]} 年年度报告",
                }
                for code, period, announced in rows
            ]
        )

    def _values(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                _values_row("600519", "2023-12-31", **{VALUE_COLUMNS[0]: 2.0e11}),
                _values_row("000001", "2023-12-31", **{VALUE_COLUMNS[0]: 5.0e12}),
                # 300750 有报表数值但没有公告日，属于单边数据，必须被内连接丢掉。
                _values_row("300750", "2023-12-31", **{VALUE_COLUMNS[0]: 7.0e11}),
            ]
        )

    def test_values_code_column_is_renamed_from_security_code(self) -> None:
        merged = merge_fundamental_panels(self._calendar(), self._values())
        self.assertEqual(sorted(merged["code"]), ["000001", "600519"])

    def test_announcement_date_survives_the_merge(self) -> None:
        merged = merge_fundamental_panels(self._calendar(), self._values())
        annual = merged.set_index("code")["ann_date"]
        self.assertEqual(annual["600519"], pd.Timestamp("2024-04-03"))
        self.assertEqual(annual["000001"], pd.Timestamp("2024-03-15"))

    def test_value_columns_are_numeric(self) -> None:
        merged = merge_fundamental_panels(self._calendar(), self._values())
        self.assertTrue(all(str(dtype).startswith("float") for dtype in merged[VALUE_COLUMNS].dtypes))

    def test_earliest_announcement_wins(self) -> None:
        """同一报告期有多条公告时取最早发布的一条，那才是数据首次公开的时点。"""
        calendar = self._calendar(
            [("600519", "2023-12-31", "2024-04-03"), ("600519", "2023-12-31", "2024-03-20")]
        )
        merged = merge_fundamental_panels(calendar, self._values())
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged["ann_date"].iloc[0], pd.Timestamp("2024-03-20"))

    def test_code_is_zero_padded_on_both_sides(self) -> None:
        calendar = self._calendar([(1, "2023-12-31", "2024-04-03")])
        values = pd.DataFrame([_values_row(1, "2023-12-31", **{VALUE_COLUMNS[0]: 1.0})])
        merged = merge_fundamental_panels(calendar, values)
        self.assertEqual(merged["code"].iloc[0], "000001")

    def test_values_without_a_code_column_yield_nothing(self) -> None:
        """代码列缺失时返回空表，而不是带着错误的列名继续往下走。"""
        broken = pd.DataFrame({"REPORT_DATE": ["2023-12-31"], "report_period": [pd.Timestamp("2023-12-31")]})
        self.assertTrue(merge_fundamental_panels(self._calendar(), broken).empty)

    def test_either_side_empty_yields_nothing(self) -> None:
        self.assertTrue(merge_fundamental_panels(self._calendar(), pd.DataFrame()).empty)
        self.assertTrue(merge_fundamental_panels(pd.DataFrame(), self._values()).empty)

    def test_rows_without_announcement_date_are_dropped(self) -> None:
        """公告时间无法解析时不保留该行，避免用报告期冒充公告日。"""
        calendar = self._calendar()
        calendar.loc[0, "ann_date"] = np.nan
        merged = merge_fundamental_panels(calendar, self._values())
        self.assertEqual(list(merged["code"]), ["000001"])


class AlignAsofPitTests(unittest.TestCase):
    """严格生效日：公告日当天不生效，次一交易日起才计入。"""

    def setUp(self) -> None:
        self.store = PanelStore(root=Path(tempfile.mkdtemp(prefix="fundamentals_test_")))

    def _left(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-04-03", "2024-04-08"]),
                "code": ["600519", "600519"],
            }
        )

    def _right(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "code": ["600519", "600519"],
                "report_period": pd.to_datetime(["2023-09-30", "2023-12-31"]),
                "available_from": pd.to_datetime(["2023-10-25", "2024-04-03"]),
                "fundamental_roe_ttm_deducted": [0.07, 0.08],
            }
        )

    def _engines(self) -> list[tuple[str, PanelStore | None]]:
        engines: list[tuple[str, PanelStore | None]] = [("pandas", None)]
        if self.store.duckdb_available():
            engines.append(("duckdb", self.store))
        return engines

    def test_announcement_day_itself_is_not_effective(self) -> None:
        columns = ["report_period", "fundamental_roe_ttm_deducted"]
        for engine, store in self._engines():
            with self.subTest(engine=engine):
                aligned = align_asof_pit(store, self._left(), self._right(), columns)
                self.assertEqual(
                    [pd.Timestamp(value) for value in aligned["report_period"]],
                    [pd.Timestamp("2023-09-30"), pd.Timestamp("2023-12-31")],
                )
                np.testing.assert_allclose(aligned["fundamental_roe_ttm_deducted"].to_numpy(), [0.07, 0.08])

    def test_effective_date_can_be_returned_alongside_the_value(self) -> None:
        """生效日同时作为对齐键与输出列时不得选出同名列（曾经会抛重复键）。"""
        columns = ["report_period", "available_from", "fundamental_roe_ttm_deducted"]
        for engine, store in self._engines():
            with self.subTest(engine=engine):
                aligned = align_asof_pit(store, self._left(), self._right(), columns)
                self.assertEqual(
                    list(aligned["available_from"]),
                    [pd.Timestamp("2023-10-25"), pd.Timestamp("2024-04-03")],
                )

    def test_no_earlier_fact_leaves_a_gap(self) -> None:
        left = pd.DataFrame({"date": pd.to_datetime(["2023-10-01"]), "code": ["600519"]})
        for engine, store in self._engines():
            with self.subTest(engine=engine):
                aligned = align_asof_pit(store, left, self._right(), ["fundamental_roe_ttm_deducted"])
                self.assertTrue(np.isnan(aligned["fundamental_roe_ttm_deducted"].iloc[0]))

    def test_empty_facts_produce_empty_columns(self) -> None:
        aligned = align_asof_pit(None, self._left(), pd.DataFrame(), ["fundamental_roe_ttm_deducted"])
        self.assertIn("fundamental_roe_ttm_deducted", aligned)
        self.assertTrue(aligned["fundamental_roe_ttm_deducted"].isna().all())

    def test_other_codes_are_not_carried_over(self) -> None:
        left = pd.DataFrame({"date": pd.to_datetime(["2024-04-08"]), "code": ["000001"]})
        for engine, store in self._engines():
            with self.subTest(engine=engine):
                aligned = align_asof_pit(store, left, self._right(), ["fundamental_roe_ttm_deducted"])
                self.assertTrue(np.isnan(aligned["fundamental_roe_ttm_deducted"].iloc[0]))


if __name__ == "__main__":
    unittest.main()
