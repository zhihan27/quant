"""验证扫描结果的落盘与读回：原子写、行数截断、明细剔除、以及坏文件不抛异常。"""

import json
import tempfile
import unittest
from pathlib import Path

from config import settings
from services.scan_pipeline import ScanOptions, ScanResult
from services.scan_store import build_payload, load_scan_result, save_scan_result


def _result(scan_limit: int = 2, trigger: str = "manual") -> ScanResult:
    options = ScanOptions(pool_mode="用户自定义", custom_codes="600000", scan_limit=scan_limit, trigger=trigger)
    return ScanResult(
        options=options,
        source_meta={"status": "测试"},
        scan_stats={"scanned": 10, "scored": 8, "failed": 2},
        rows=[{"代码": f"60000{index}", "综合评分": 90 - index} for index in range(5)],
        detail_rows=[{"代码": f"60000{index}"} for index in range(5)],
        failed_rows=[{"代码": "600009", "状态": "停牌"}],
        usable=[{"code": "600000", "name": "浦发银行"}],
        notes=[{"level": "caption", "text": "测试血缘说明"}],
        portfolio_result={
            "metrics": {"total_return": 0.12, "trades": 3},
            "trades": [{"code": "600000", "entry_date": "2026-01-05"}],
            "daily_equity": [{"date": "2026-01-05", "equity": 1.0}],
            "training_log": [{"prediction_start": "2026-01-05", "models": []}],
            "open_positions": [{"code": "600000"}],
        },
        context_id="ctx-123", reference_date="2026-10-15", finished_at="2026-10-15T15:40:00+08:00",
    )


class ScanStoreTests(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.path = Path(self._directory.name) / "last_scan.json"

    def tearDown(self):
        self._directory.cleanup()

    def test_round_trip_keeps_the_fields_the_page_renders(self):
        save_scan_result(_result(), self.path)
        payload = load_scan_result(self.path)
        self.assertEqual(payload["model_version"], settings.MODEL_VERSION)
        self.assertEqual(payload["trigger"], "manual")
        self.assertEqual(payload["reference_date"], "2026-10-15")
        self.assertEqual(payload["finished_at"], "2026-10-15T15:40:00+08:00")
        self.assertEqual(payload["scan_stats"], {"scanned": 10, "scored": 8, "failed": 2})
        self.assertEqual(payload["pool_mode"], "用户自定义")
        self.assertEqual(payload["notes"][0]["text"], "测试血缘说明")

    def test_rows_are_truncated_to_the_display_limit(self):
        """全池逐股结果序列化后是几十兆，落盘只留页面要渲染的那几行。"""
        save_scan_result(_result(scan_limit=2), self.path)
        payload = load_scan_result(self.path)
        self.assertEqual(len(payload["rows"]), 2)
        self.assertEqual(len(payload["detail_rows"]), 2)

    def test_portfolio_details_are_not_persisted(self):
        save_scan_result(_result(), self.path)
        portfolio = load_scan_result(self.path)["portfolio"]
        self.assertEqual(portfolio["metrics"], {"total_return": 0.12, "trades": 3})
        for key in ("trades", "daily_equity", "training_log", "open_positions"):
            self.assertNotIn(key, portfolio, f"{key} 随回测长度增长，不该进落盘文件")

    def test_scheduled_trigger_survives_so_the_page_can_say_who_ran_it(self):
        save_scan_result(_result(trigger="scheduled"), self.path)
        self.assertEqual(load_scan_result(self.path)["trigger"], "scheduled")

    def test_write_is_atomic_and_leaves_no_temporary_file(self):
        save_scan_result(_result(), self.path)
        self.assertFalse(self.path.with_name(self.path.name + ".tmp").exists())

    def test_preview_payload_matches_the_saved_file_field_by_field(self):
        """自定义代码试跑不落盘，但渲染口径必须与读回落盘文件完全一致。

        页面上的试跑结果直接用 build_payload 渲染；两条路径一旦分叉，同一张表就会
        因为来源不同而少几个字段，而且只在试跑时才看得出来。
        """
        result = _result()
        save_scan_result(result, self.path)
        payload = load_scan_result(self.path)
        preview = build_payload(result)
        self.assertEqual(set(preview), set(payload))
        self.assertEqual(preview["pool_mode"], payload["pool_mode"])
        self.assertEqual(preview["rows"], payload["rows"])
        self.assertEqual(preview["scan_stats"], payload["scan_stats"])

    def test_missing_file_returns_none(self):
        self.assertIsNone(load_scan_result(self.path))

    def test_corrupt_file_returns_none_instead_of_raising(self):
        self.path.write_text("{ 这不是 JSON", encoding="utf-8")
        self.assertIsNone(load_scan_result(self.path))

    def test_result_from_another_model_version_is_discarded(self):
        """模型版本升级后旧排名与当前模型对不上，宁可显示空白也不显示错的结果。"""
        payload = json.loads(json.dumps({"model_version": "0.0.1", "rows": [{"代码": "600000"}]}))
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        self.assertIsNone(load_scan_result(self.path))

    def test_non_dict_payload_is_discarded(self):
        self.path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        self.assertIsNone(load_scan_result(self.path))


if __name__ == "__main__":
    unittest.main()
