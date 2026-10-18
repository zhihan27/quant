"""验证通过率统计与告警判定：全部使用构造记录，不依赖真实行情与真实数据库。

判定逻辑（``summarize_pass_rate`` / ``evaluate_pass_rate_alerts``）是纯函数，
本文件因此完全不连库；写库与取数路径由 :mod:`tests.test_model_governance` 里的
隔离模式集成用例覆盖。
"""

from __future__ import annotations

import os
import secrets
import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from config import settings
from services.database import connection_settings
from services.ml_pipeline import evaluation_gates
from services.rank_model import SCORE_KIND_RANK, SCORE_KIND_RETURN
from services.model_monitor import (
    GATE_LABELS,
    MODEL_TYPE_CODES,
    NO_METRICS_GATE,
    evaluate_pass_rate_alerts,
    summarize_pass_rate,
    validation_record,
    validation_records,
)

SHANGHAI = ZoneInfo("Asia/Shanghai")


def _metrics(**overrides) -> dict:
    """一份能通过全部门槛的排序目标样本外指标。"""
    metrics = {
        "dates": 60, "rank_ic": 0.05, "rank_ic_hac_t": 2.5, "score_kind": SCORE_KIND_RANK,
        "ndcg_ratio": 1.2, "top_excess_return": 0.03, "universe_excess_return": 0.0,
        "top_bottom_spread": 0.05, "mae": None, "baseline_mae": None, "folds": [],
    }
    metrics.update(overrides)
    return metrics


def _row(passed: bool, scan_time: datetime, model_type: str = "lgb", target_type: str = "rank",
         failed_gate: str | None = None, gates: dict | None = None) -> dict:
    return {
        "pass": passed, "scan_time": scan_time, "model_type": model_type, "target_type": target_type,
        "failed_gate": failed_gate,
        "passed_gates": gates if gates is not None else {"dates": True, "rank_ic": passed},
    }


class GateTests(unittest.TestCase):
    """判定门槛必须在评估与监控之间只有一份实现，否则两边会各走一路。"""

    def test_all_gates_pass(self):
        gates = evaluation_gates(_metrics())
        self.assertTrue(gates["passed"])
        self.assertIsNone(gates["failed_gate"])

    def test_first_failing_gate_is_reported(self):
        # Rank IC 与 HAC t 同时不达标时，报出来的应当是先判的那一道。
        gates = evaluation_gates(_metrics(rank_ic=0.001, rank_ic_hac_t=0.5))
        self.assertFalse(gates["passed"])
        self.assertEqual(gates["failed_gate"], "rank_ic")
        self.assertFalse(gates["gates"]["rank_ic"])
        self.assertFalse(gates["gates"]["ic_t"])

    def test_ranking_and_regression_judge_different_last_gate(self):
        ranking = evaluation_gates(_metrics(ndcg_ratio=1.0))
        self.assertEqual(ranking["failed_gate"], "ndcg")
        self.assertNotIn("mae", ranking["gates"])
        regression = evaluation_gates(_metrics(
            score_kind=SCORE_KIND_RETURN, ndcg_ratio=None, mae=0.2, baseline_mae=0.1,
        ))
        self.assertEqual(regression["failed_gate"], "mae")
        self.assertNotIn("ndcg", regression["gates"])

    def test_insufficient_dates_fails_first(self):
        self.assertEqual(evaluation_gates(_metrics(dates=39))["failed_gate"], "dates")


class PassRateTests(unittest.TestCase):
    """通过率的分母与门槛失败率的分母是两件事，混用会给出错误的结论。"""

    def setUp(self):
        self.now = datetime(2026, 10, 16, 15, 30, tzinfo=SHANGHAI)

    def test_grouping_and_gate_failure_rate(self):
        rows = [
            _row(True, self.now),
            _row(False, self.now - timedelta(days=1), failed_gate="rank_ic"),
            _row(True, self.now - timedelta(days=2), model_type="xgb", target_type="reg"),
            _row(False, self.now - timedelta(days=3), model_type="xgb", target_type="rank",
                 failed_gate="rank_ic"),
        ]
        # 每条记录只判了两道门槛（dates 与 rank_ic），rank_ic 失败 2 次。
        summary = summarize_pass_rate(rows, days=30)
        self.assertEqual(summary["records"], 4)
        self.assertEqual(summary["passed"], 2)
        self.assertEqual(summary["pass_rate"], 0.5)
        self.assertEqual(summary["by_model_type"]["lgb"], {"records": 2, "passed": 1, "pass_rate": 0.5})
        self.assertEqual(summary["by_model_type"]["xgb"]["records"], 2)
        self.assertEqual(summary["by_target_type"]["rank"]["records"], 3)
        self.assertEqual(summary["by_gate"]["rank_ic"]["evaluated"], 4)
        self.assertEqual(summary["by_gate"]["rank_ic"]["failures"], 2)
        self.assertEqual(summary["by_gate"]["rank_ic"]["failure_rate"], 0.5)
        self.assertFalse(summary["by_gate"]["rank_ic"]["label"] == "rank_ic", "门槛必须给出中文说明")

    def test_gate_denominator_excludes_records_that_never_ran_it(self):
        """排序目标判 NDCG、回归目标判 MAE，拿全部记录当分母会把对方的通过算成自己的失败。"""
        rows = [
            _row(True, self.now, gates={"ndcg": True}),
            _row(False, self.now, model_type="xgb", target_type="reg", gates={"mae": False}),
        ]
        summary = summarize_pass_rate(rows, days=30)
        self.assertEqual(summary["by_gate"]["ndcg"]["evaluated"], 1)
        self.assertEqual(summary["by_gate"]["ndcg"]["failure_rate"], 0.0)
        self.assertEqual(summary["by_gate"]["mae"]["failure_rate"], 1.0)

    def test_empty_window_reports_no_rate_rather_than_zero(self):
        summary = summarize_pass_rate([], days=30)
        self.assertEqual(summary["records"], 0)
        self.assertIsNone(summary["pass_rate"], "没有数据时通过率必须是 None，0% 与「没有数据」不是一回事")
        self.assertFalse(summary["sample_sufficient"])
        self.assertTrue(summary["sample_note"])
        self.assertIsNone(summary["recent_failure"])

    def test_recent_failure_is_the_latest_one(self):
        rows = [
            _row(False, self.now, failed_gate="ic_t"),
            _row(True, self.now - timedelta(days=1)),
            _row(False, self.now - timedelta(days=2), failed_gate="ndcg"),
        ]
        failure = summarize_pass_rate(rows, days=30)["recent_failure"]
        self.assertEqual(failure["failed_gate"], "ic_t")
        self.assertEqual(failure["reason"], GATE_LABELS["ic_t"])

    def test_cached_validation_replays_are_deduplicated_in_health_statistics(self):
        replay_identity = "same-validation"
        older = _row(False, self.now - timedelta(days=2), failed_gate="rank_ic")
        older.update(
            model_version=settings.MODEL_VERSION,
            details={"horizon": 20, "validation_identity": replay_identity},
        )
        replay = _row(False, self.now - timedelta(days=1), failed_gate="rank_ic")
        replay.update(
            model_version=settings.MODEL_VERSION,
            details={"horizon": 20, "validation_identity": replay_identity},
        )

        summary = summarize_pass_rate([replay, older], days=30)

        self.assertEqual(summary["records"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["batch_count"], 1)
        self.assertEqual(summary["recent_failure"]["scan_time"], replay["scan_time"].isoformat())


class PassRateAlertTests(unittest.TestCase):
    """三条告警各自触发与不触发的边界。"""

    def setUp(self):
        self.now = datetime(2026, 10, 16, 15, 30, tzinfo=SHANGHAI)

    def _streaks(self, passes: list[bool], model_type: str = "lgb", target_type: str = "rank") -> list[dict]:
        return [
            {"model_type": model_type, "target_type": target_type, "pass": passed,
             "failed_gate": None if passed else "rank_ic", "scan_time": self.now,
             "position": index + 1}
            for index, passed in enumerate(passes)
        ]

    def _many(self, passed: int, failed: int, gates: dict | None = None) -> list[dict]:
        rows = [_row(True, self.now, gates=gates) for _ in range(passed)]
        rows += [_row(False, self.now, failed_gate="rank_ic", gates=gates) for _ in range(failed)]
        return rows

    def test_three_consecutive_failures_alert(self):
        alerts = evaluate_pass_rate_alerts(
            self._many(0, 20), self._streaks([False, False, False])
        )
        consecutive = [item for item in alerts if "连续" in item["title"]]
        self.assertEqual(len(consecutive), 1)
        self.assertEqual(consecutive[0]["level"], "critical")
        self.assertEqual(consecutive[0]["details"]["consecutive"], settings.CONSECUTIVE_FAILURE_ALERT)

    def test_two_consecutive_failures_do_not_alert(self):
        alerts = evaluate_pass_rate_alerts(self._many(0, 20), self._streaks([False, False]))
        self.assertFalse([item for item in alerts if "连续" in item["title"]])

    def test_three_horizons_in_one_scan_do_not_form_a_failure_streak(self):
        streaks = [{**row, "horizon": horizon, "position": 1}
                   for row, horizon in zip(self._streaks([False] * 3), (5, 20, 60))]
        self.assertEqual(evaluate_pass_rate_alerts(self._many(0, 6), streaks), [])

    def test_versions_do_not_share_a_failure_streak(self):
        streaks = [{**row, "model_version": str(index), "horizon": 20, "position": 1}
                   for index, row in enumerate(self._streaks([False] * 3))]
        self.assertEqual(evaluate_pass_rate_alerts(self._many(0, 6), streaks), [])

    def test_failure_alert_identifies_the_horizon(self):
        streaks = [{**row, "horizon": 20} for row in self._streaks([False] * 3)]
        alerts = evaluate_pass_rate_alerts(self._many(0, 6), streaks)
        self.assertEqual(len(alerts), 1)
        self.assertIn("20日", alerts[0]["title"])

    def test_a_pass_in_the_streak_resets_it(self):
        """连续要求的是「最近 N 次都不通过」，中间过了一次就不算连续。"""
        alerts = evaluate_pass_rate_alerts(
            self._many(0, 20), self._streaks([False, False, True, False, False])
        )
        self.assertFalse([item for item in alerts if "连续" in item["title"]])

    def test_low_monthly_pass_rate_alerts_only_with_enough_records(self):
        rows = self._many(1, 5)
        self.assertFalse([item for item in evaluate_pass_rate_alerts(rows, []) if "通过率低于" in item["title"]],
                         "样本不足时不该报通过率")
        enough = self._many(6, 20)
        self.assertLess(6 / 26, settings.PASS_RATE_MINIMUM)
        titles = [item["title"] for item in evaluate_pass_rate_alerts(enough, [])]
        self.assertTrue(any("通过率低于" in title for title in titles))

    def test_gate_failure_rate_alert(self):
        # 26 条记录里 20 条卡在 rank_ic，失败率 76.9% > 50%。
        alerts = evaluate_pass_rate_alerts(self._many(6, 20), [])
        gate_alerts = [item for item in alerts if "失败率超过" in item["title"]]
        self.assertEqual(len(gate_alerts), 1)
        self.assertEqual(gate_alerts[0]["level"], "warning")
        self.assertEqual(gate_alerts[0]["details"]["gate"], "rank_ic")

    def test_healthy_window_produces_no_alerts(self):
        self.assertEqual(evaluate_pass_rate_alerts(self._many(26, 0), self._streaks([True, True, True])), [])


class ValidationRecordTests(unittest.TestCase):
    """模型上下文 → 日志行的翻译：缺指标是常态，必须如实记为「未生成样本外指标」。"""

    def _context(self, model: dict, horizon: str = "20") -> dict:
        return {
            "context_id": "ctx-1", "model_version": settings.MODEL_VERSION,
            "reference_date": "2026-10-16", "objective_mode": "lambdarank",
            "training_window_days": 750, "models": {horizon: [model]}, "predictions": {},
        }

    def test_validated_model_records_full_metrics(self):
        model = {
            "model": "LightGBM", "horizon": 20, "score_kind": SCORE_KIND_RANK, "validated": True,
            "training_dates": 700, "status": "通过样本外基线、Rank IC与分组收益检验",
            "oos_metrics": _metrics(folds=[
                {"trained_from": "2021-11-03", "trained_through": "2023-12-29", "training_dates": 530},
                {"trained_from": "2022-06-01", "trained_through": "2024-06-28", "training_dates": 750},
            ]),
        }
        record = validation_record(self._context(model), model, 20)
        self.assertTrue(record["pass"])
        self.assertEqual(record["model_type"], "lgb")
        self.assertEqual(record["target_type"], "rank")
        # 训练区间取所有折的并集，不是最后一折——只报最后一折会低估整个训练期。
        self.assertEqual(record["window_start"], "2021-11-03")
        self.assertEqual(record["window_end"], "2024-06-28")
        self.assertEqual(record["details"]["fold_training_dates"], 750)

    def test_model_without_oos_metrics_is_recorded_as_failure_not_skipped(self):
        model = {"model": "XGBoost", "horizon": 20, "objective": "regression",
                 "validated": False, "status": "截面样本不足，未生成预测", "oos_metrics": None}
        record = validation_record(self._context(model), model, 20)
        self.assertFalse(record["pass"])
        self.assertEqual(record["failed_gate"], NO_METRICS_GATE)
        self.assertEqual(record["target_type"], "reg")
        self.assertIsNone(record["rank_ic"])

    def test_nan_metrics_become_null(self):
        model = {"model": "LightGBM", "horizon": 5, "score_kind": SCORE_KIND_RANK, "validated": False,
                 "status": "未通过", "oos_metrics": _metrics(rank_ic=float("nan"), rank_ic_hac_t=float("inf"))}
        record = validation_record(self._context(model, "5"), model, 5)
        self.assertIsNone(record["rank_ic"])
        self.assertIsNone(record["hac_t"])

    def test_validation_audit_persists_cache_identity_and_sources_for_auto_mode(self):
        execution = {
            "validation_identity": "validation-id",
            "validation_cache_sources": [{"validation_segment": 1, "source": "hot"}],
            "validation_cache_replay": True,
        }
        model = {
            "model": "LightGBM", "horizon": 20, "objective": "lambdarank",
            "validated": False, "oos_metrics": _metrics(),
            "objective_selection": [{
                "objective": "lambdarank", "selected": True, "execution": execution,
            }],
        }

        record = validation_record(self._context(model), model, 20)

        self.assertEqual(record["details"]["validation_identity"], "validation-id")
        self.assertEqual(record["details"]["validation_cache_sources"], execution["validation_cache_sources"])
        self.assertTrue(record["details"]["validation_cache_replay"])

    def test_records_cover_every_model_and_horizon(self):
        context = {
            "context_id": "ctx", "model_version": settings.MODEL_VERSION, "models": {
                "20": [{"model": name, "objective": "lambdarank", "oos_metrics": None} for name in ("LightGBM", "XGBoost")],
                "5": [], "60": [],
            },
        }
        records = validation_records(context, trigger="scheduled")
        self.assertEqual(len(records), 2)
        self.assertEqual({record["model_type"] for record in records}, set(MODEL_TYPE_CODES.values()))
        for record in records:
            self.assertEqual(record["details"]["trigger"], "scheduled")

    def test_record_is_json_serializable(self):
        """记录要经 psycopg 的 Jsonb 落库，numpy 标量会让写入直接抛错。"""
        import json

        model = {"model": "LightGBM", "horizon": 20, "score_kind": SCORE_KIND_RANK, "validated": True,
                 "training_dates": 700, "status": "通过", "oos_metrics": _metrics()}
        record = validation_record(self._context(model), model, 20)
        json.dumps({key: value for key, value in record.items() if key != "scan_time"}, allow_nan=False)


@unittest.skipUnless(os.environ.get("QUANT_TEST_POSTGRES") == "1", "设置QUANT_TEST_POSTGRES=1运行隔离的数据库集成测试")
class ValidationLogDatabaseTests(unittest.TestCase):
    """验证真实落库与回读：追加写入、序列化、以及取数过滤。"""

    @classmethod
    def setUpClass(cls):
        cls.schema = "quant_monitor_test_" + secrets.token_hex(6)
        with psycopg.connect(**connection_settings()) as connection:
            connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(cls.schema)))
            for migration in sorted((settings.PROJECT_ROOT / "sql").glob("*.sql")):
                connection.execute(migration.read_text(encoding="utf-8").replace("public.", cls.schema + "."))

        class _ScopedCursor:
            """executemany 一类批量写入走游标，同样要把 public. 换成测试模式。

            少了这一层，``log_validation`` 的批量写入会落到真实的 public 模式，
            而读取走的是测试模式——测试看起来「通过」或「查无数据」，实际污染了主库。
            """

            def __init__(self, cursor, schema):
                self.cursor, self.schema = cursor, schema

            def execute(self, query, parameters=None):
                return self.cursor.execute(query.replace("public.", self.schema + "."), parameters)

            def executemany(self, query, parameters=None):
                return self.cursor.executemany(query.replace("public.", self.schema + "."), parameters)

            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                self.cursor.close()

        class _ScopedConnection:
            def __init__(self, connection, schema):
                self.connection, self.schema = connection, schema

            def execute(self, query, parameters=None):
                return self.connection.execute(query.replace("public.", self.schema + "."), parameters)

            def cursor(self):
                return _ScopedCursor(self.connection.cursor(), self.schema)

        class _ScopedRepository:
            def __init__(self, schema):
                self.schema = schema

            def connect(self):
                from contextlib import contextmanager

                @contextmanager
                def _open():
                    with psycopg.connect(**connection_settings(), row_factory=dict_row) as connection:
                        yield _ScopedConnection(connection, self.schema)
                return _open()

        cls.repository = _ScopedRepository(cls.schema)

    @classmethod
    def tearDownClass(cls):
        with psycopg.connect(**connection_settings()) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(cls.schema)))

    def setUp(self):
        # 用例之间清空检验日志：追加写入的语义让上一轮的记录留在这里，
        # 下一条用例数出来的行数会把它们一起算进去。
        with self.repository.connect() as connection:
            connection.execute("TRUNCATE public.quant_model_validation_log")

    def test_records_round_trip_and_filter(self):
        from services.model_monitor import fetch_validation_rows, get_pass_rate, get_recent_failures, log_validation

        model = {"model": "LightGBM", "horizon": 20, "score_kind": SCORE_KIND_RANK, "validated": True,
                 "training_dates": 700, "status": "通过", "oos_metrics": _metrics()}
        passing = {"context_id": "ctx", "model_version": settings.MODEL_VERSION,
                   "models": {"20": [model]}}
        failing = {"context_id": "ctx", "model_version": settings.MODEL_VERSION, "models": {"20": [
            {"model": "XGBoost", "horizon": 20, "objective": "lambdarank", "validated": False,
             "status": "未通过样本外基线", "oos_metrics": _metrics(rank_ic=0.001)},
        ]}}
        written = log_validation(self.repository, validation_records(passing) + validation_records(failing))
        self.assertEqual(written, 2)
        rows = fetch_validation_rows(self.repository, days=1)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(fetch_validation_rows(self.repository, days=1, model_type="xgb")), 1)
        self.assertEqual(get_pass_rate(self.repository, days=1)["pass_rate"], 0.5)
        failures = get_recent_failures(self.repository)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["model_label"], "XGBoost")
        self.assertTrue(failures[0]["failed_gate_label"])

    def test_append_only_history_is_kept(self):
        """同一个模型版本反复重训，每次结果都要留下来，覆盖写会把「模型变差了」擦掉。"""
        from services.model_monitor import fetch_validation_rows, log_validation

        model = {"model": "LightGBM", "horizon": 5, "score_kind": SCORE_KIND_RANK, "validated": True,
                 "status": "通过", "oos_metrics": _metrics()}
        context = {"context_id": "ctx", "model_version": settings.MODEL_VERSION, "models": {"5": [model]}}
        log_validation(self.repository, validation_records(context))
        log_validation(self.repository, validation_records(context))
        self.assertEqual(len(fetch_validation_rows(self.repository, days=1)), 2)

    def test_streaks_separate_horizons_and_deduplicate_training_batches(self):
        from services.model_monitor import fetch_failure_streaks, log_validation

        context = {"context_id": "batch-1", "model_version": settings.MODEL_VERSION, "models": {
            horizon: [{"model": "LightGBM", "objective": "lambdarank", "status": "样本不足"}]
            for horizon in (5, 20, 60)
        }}
        log_validation(self.repository, validation_records(context))
        log_validation(self.repository, validation_records(context))
        streaks = fetch_failure_streaks(self.repository, 3)
        self.assertEqual(len(streaks), 3)
        self.assertTrue(all(row["position"] == 1 for row in streaks))
        self.assertEqual(evaluate_pass_rate_alerts([], streaks), [])
        for batch in ("batch-2", "batch-3"):
            context["context_id"] = batch
            log_validation(self.repository, validation_records(context))
        streaks = fetch_failure_streaks(self.repository, 3)
        self.assertEqual(len(streaks), 9)
        self.assertEqual({row["details"]["horizon"] for row in evaluate_pass_rate_alerts([], streaks)}, {"5", "20", "60"})

    def test_streaks_do_not_mix_model_versions(self):
        from services.model_monitor import fetch_failure_streaks, log_validation

        for version in ("old", settings.MODEL_VERSION):
            for batch in range(2):
                context = {"context_id": f"{version}-{batch}", "model_version": version, "models": {
                    20: [{"model": "LightGBM", "objective": "lambdarank", "status": "样本不足"}]
                }}
                log_validation(self.repository, validation_records(context))
        self.assertEqual(evaluate_pass_rate_alerts([], fetch_failure_streaks(self.repository, 3)), [])


if __name__ == "__main__":
    unittest.main()
