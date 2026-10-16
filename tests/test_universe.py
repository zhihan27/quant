"""股票池生存者偏差控制：退市、退市整理期与 ST 标的必须在扫描前被排除。"""

from __future__ import annotations

import unittest

import pandas as pd

from services.universe import (
    STATUS_DELISTED, STATUS_DELISTING, STATUS_NORMAL, STATUS_ST, STATUS_UNKNOWN,
    annotate, classify_name_status, exclude_reason, filter_universe, is_excluded_status,
    parse_reference_records, status_for,
)


def _reference() -> pd.DataFrame:
    """交易所退市名单与 ST 名单合并后的状态表。"""
    return pd.DataFrame(
        [
            {"code": "600001", "name": "退市海润", "stock_status": STATUS_DELISTED,
             "status_source": "上交所退市名单", "delist_date": "2020-05-21"},
            {"code": "000002", "name": "ST 中天", "stock_status": STATUS_ST,
             "status_source": "全市场 ST 名单", "delist_date": None},
        ]
    )


class NameStatusTests(unittest.TestCase):
    def test_plain_name_is_normal(self) -> None:
        self.assertEqual(classify_name_status("贵州茅台"), STATUS_NORMAL)

    def test_st_prefixes_are_caught(self) -> None:
        for name in ("ST 中天", "*ST 康得", "S*ST 兰光", "SST 前锋", "PT 水仙"):
            with self.subTest(name=name):
                self.assertEqual(classify_name_status(name), STATUS_ST)

    def test_delisting_period_names_are_caught(self) -> None:
        """退市整理期简称三种写法都含「退」字。"""
        for name in ("退市海润", "海润退", "国华退"):
            with self.subTest(name=name):
                self.assertEqual(classify_name_status(name), STATUS_DELISTING)

    def test_ex_rights_prefix_is_stripped_before_matching(self) -> None:
        """除权除息日的 XD/XR/DR 前缀是临时标记，不能因此判定为正常股而漏掉 ST。"""
        self.assertEqual(classify_name_status("XD*ST 康得"), STATUS_ST)
        self.assertEqual(classify_name_status("DR 贵州茅台"), STATUS_NORMAL)

    def test_blank_name_is_unknown_not_normal(self) -> None:
        """拿不到简称时必须留 UNKNOWN，不能默认当作正常股放行。"""
        for name in (None, "", "   ", float("nan")):
            with self.subTest(name=repr(name)):
                self.assertEqual(classify_name_status(name), STATUS_UNKNOWN)

    def test_name_containing_letters_is_not_misread(self) -> None:
        """简称中偶现的字母组合不能被当成 ST 前缀。"""
        self.assertEqual(classify_name_status("TCL 科技"), STATUS_NORMAL)


class ExclusionTests(unittest.TestCase):
    def test_normal_status_is_not_excluded(self) -> None:
        self.assertFalse(is_excluded_status(STATUS_NORMAL))
        self.assertFalse(is_excluded_status(STATUS_UNKNOWN))

    def test_all_three_hard_rules_are_excluded(self) -> None:
        for status in (STATUS_DELISTED, STATUS_DELISTING, STATUS_ST):
            with self.subTest(status=status):
                self.assertTrue(is_excluded_status(status))

    def test_reason_mentions_source(self) -> None:
        self.assertIn("上交所退市名单", exclude_reason(STATUS_DELISTED, "上交所退市名单"))


class StatusResolutionTests(unittest.TestCase):
    def test_reference_list_wins_over_heuristic(self) -> None:
        """名单里有记录时以名单为准，即使简称看起来正常。"""
        status, source = status_for("600001", "贵州茅台", {"600001": {"stock_status": STATUS_DELISTED,
                                                                     "status_source": "上交所退市名单"}})
        self.assertEqual(status, STATUS_DELISTED)
        self.assertEqual(source, "上交所退市名单")

    def test_heuristic_fills_gap_when_list_misses(self) -> None:
        status, source = status_for("300999", "*ST 新研", {})
        self.assertEqual(status, STATUS_ST)
        self.assertEqual(source, "证券简称启发式")

    def test_code_is_normalised_to_six_digits(self) -> None:
        status, _ = status_for(1, "退市海润", {"000001": {"stock_status": STATUS_DELISTED,
                                                        "status_source": "上交所退市名单"}})
        self.assertEqual(status, STATUS_DELISTED)


class ReferenceParsingTests(unittest.TestCase):
    def test_delisted_record_before_st_record_wins(self) -> None:
        """同一代码同时出现在两个名单时保留退市记录，因为它带退市日期、信息更完整。"""
        lines = [
            '{"kind": "delisted", "code": "600001", "name": "退市海润", "exchange": "上交所", "delist_date": "2020-05-21"}',
            '{"kind": "st", "code": "600001", "name": "退市海润"}',
        ]
        records, failures = parse_reference_records(lines)
        self.assertEqual(failures, [])
        self.assertEqual(records["600001"]["stock_status"], STATUS_DELISTED)
        self.assertEqual(records["600001"]["delist_date"], "2020-05-21")

    def test_source_error_is_surfaced_not_swallowed(self) -> None:
        """某个名单源失败必须留下失败原因，否则会静默退化成只用简称判断。"""
        records, failures = parse_reference_records(['{"kind": "error", "source": "全市场 ST 名单", "error": "timeout"}'])
        self.assertEqual(records, {})
        self.assertEqual(len(failures), 1)
        self.assertIn("全市场 ST 名单", failures[0])

    def test_malformed_code_is_reported(self) -> None:
        _, failures = parse_reference_records(['{"kind": "delisted", "code": "6000", "name": "退市海润"}'])
        self.assertEqual(len(failures), 1)


class FilterUniverseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = pd.DataFrame(
            [
                {"code": "600519", "name": "贵州茅台"},
                {"code": "600001", "name": "退市海润"},
                {"code": "000002", "name": "ST 中天"},
                {"code": "300123", "name": "海润退"},
                {"code": "002415", "name": "海康威视"},
            ]
        )

    def test_delisted_st_and_delisting_are_removed(self) -> None:
        kept, report = filter_universe(self.pool, _reference())
        self.assertEqual(sorted(kept["code"]), ["002415", "600519"])
        self.assertEqual(report["excluded"], 3)
        self.assertEqual(report["checked"], 5)

    def test_heuristic_alone_still_removes_delisting_names(self) -> None:
        """没有名单时也不能放行「退」字简称，否则退市整理期标的会进入回测。"""
        kept, report = filter_universe(self.pool, None)
        self.assertNotIn("300123", set(kept["code"]))
        self.assertNotIn("600001", set(kept["code"]))

    def test_excluded_codes_are_traceable(self) -> None:
        _, report = filter_universe(self.pool, _reference())
        self.assertIn("600001", report["excluded_codes"])
        self.assertIn(STATUS_DELISTED, report["by_status"])
        self.assertIn(STATUS_ST, report["by_status"])

    def test_limit_is_stated_honestly(self) -> None:
        """回测股票池取自本次在市标的，无法还原历史成分，这一点必须在报告里写明。"""
        _, report = filter_universe(self.pool, _reference())
        self.assertIn("无法还原历史成分股", report["limitation"])

    def test_missing_code_column_is_not_treated_as_pass(self) -> None:
        frame = pd.DataFrame([{"name": "贵州茅台"}])
        kept, report = filter_universe(frame, _reference())
        self.assertEqual(report["excluded"], 0)
        self.assertIn("缺少代码列", report["status"])

    def test_empty_pool_returns_empty_report(self) -> None:
        kept, report = filter_universe(pd.DataFrame(columns=["code", "name"]), _reference())
        self.assertTrue(kept.empty)
        self.assertEqual(report["kept"], 0)


class AnnotateTests(unittest.TestCase):
    def test_annotate_marks_reason_only_for_excluded(self) -> None:
        annotated = annotate(pd.DataFrame([{"code": "600519", "name": "贵州茅台"}]), _reference())
        self.assertFalse(bool(annotated["excluded"].iloc[0]))
        self.assertEqual(annotated["exclude_reason"].iloc[0], "")

    def test_annotate_survives_empty_frame(self) -> None:
        result = annotate(pd.DataFrame(columns=["code", "name"]), _reference())
        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
