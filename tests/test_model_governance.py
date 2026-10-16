"""模型治理的数据库侧行为：只评估一次的留出集、归档与回滚安全、降级状态机、告警去重。

纯计算部分（留出集的过滤与换手率口径、PSI 与漂移阈值、通过率统计）已由
``test_final_holdout.py``、``test_drift_monitor.py``、``test_model_monitor.py`` 覆盖，
本文件只回答「真的读写库、真的读写模型文件时会发生什么」：

* 第二次 :func:`evaluate_final_holdout` 是不是真的被挡在训练之前，``--force`` 有没有留审计；
* 回滚是不是真的把活动文件换了回去、该拒绝的四种情形是不是都拒绝、回滚本身还可不可以再回滚；
* 归档裁剪有没有把活动版本一起删掉；
* 降级状态机与审计有没有留痕，严重的特征漂移会不会自动降级（且**不**自动回滚）；
* 告警在去重窗口内被抑制时，不会被算成「今天有 N 条告警」；
* 扫描产出的三张漂移表与检验日志是否真的落库，重复扫描会不会刷重复行。

连库用例照 ``tests/test_auth.py`` 的模式：``QUANT_TEST_POSTGRES=1`` 加随机模式隔离，
模型文件读写全部落在临时目录，绝不碰仓库里的 ``models/`` 与 ``data/``。

``_ScopedConnection`` 连 ``cursor()`` 一起接管：检验日志与三张漂移表的批量写入走
``executemany``，只接管 ``execute`` 会让那些写入落到真实的 public 模式，而读取走的是
测试模式——用例会「看起来通过」，实际污染主库。
"""

from __future__ import annotations

import os
import secrets
import shutil
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pandas as pd
import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from config import settings
from services import alerting, drift_monitor, model_governance, model_monitor
from services.database import DatabaseError, connection_settings
from services.feature_engineering import build_features
from services.ml_pipeline import HORIZONS, MODEL_NAMES, TARGET_KIND, load_context, save_context
from services.model_governance import (
    FinalHoldoutError,
    ScanGovernance,
    acknowledge_alert,
    archive_context,
    clear_degradation,
    current_model_status,
    degrade_model,
    evaluate_final_holdout,
    final_holdout_record,
    holdout_report,
    model_status,
    recent_alerts,
    registry_versions,
    rollback_to,
)
from services.rank_model import SCORE_KIND_RANK
from services.stock_scoring import analyze_stock

TABLE_LIST = (
    "public.quant_model_validation_log,public.quant_feature_drift,public.quant_concept_drift,"
    "public.quant_prediction_drift,public.quant_model_registry,public.quant_model_final_holdout,"
    "public.quant_model_alerts,public.quant_model_health_state,public.quant_model_governance_events"
)


class _ScopedCursor:
    """把游标上的 SQL 也映射到测试模式，``executemany`` 的批量写入同样不能漏。"""

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
        @contextmanager
        def _open():
            with psycopg.connect(**connection_settings(), row_factory=dict_row) as connection:
                yield _ScopedConnection(connection, self.schema)
        return _open()


def _create_schema(schema: str) -> None:
    """把 sql/ 下的全部迁移灌进随机模式，顺带验证迁移脚本本身能执行。"""
    with psycopg.connect(**connection_settings()) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        for migration in sorted((settings.PROJECT_ROOT / "sql").glob("*.sql")):
            connection.execute(migration.read_text(encoding="utf-8").replace("public.", schema + "."))


def _drop_schema(schema: str) -> None:
    with psycopg.connect(**connection_settings()) as connection:
        connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


# ---------------------------------------------------------------------------
# 降级闸门：不连库，只验 analyze_stock 在暂停标志下的姿态
# ---------------------------------------------------------------------------

CODE = "600000"


def _history(days: int = 90) -> pd.DataFrame:
    generator = np.random.default_rng(7)
    close = 10.0 * np.cumprod(1.0 + generator.normal(0.0005, 0.01, days))
    return pd.DataFrame({
        "date": pd.bdate_range(end="2026-10-16", periods=days),
        "open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
        "volume": generator.integers(1_000_000, 2_000_000, days).astype(float),
    })


def _scoring_context(history: pd.DataFrame) -> dict:
    """一份能通过数据血缘校验的上下文：参考日期与收盘价直接取自真实因子表的最后一根。"""
    latest = build_features(history).iloc[-1]
    prediction = {"predicted_excess_return": 0.02, "prediction_zscore": 1.1,
                  "prediction_percentile": 0.9, "model_rank": 1.0, "score_kind": SCORE_KIND_RANK}
    return {
        "context_id": "ctx-gate", "model_version": settings.MODEL_VERSION, "target_kind": TARGET_KIND,
        "reference_date": str(latest["date"])[:10], "reference_closes": {CODE: float(latest["close"])},
        "models": {horizon: [
            {"model": name, "horizon": horizon, "score_kind": SCORE_KIND_RANK, "objective": "lambdarank",
             "validated": True, "status": "通过样本外基线、Rank IC与分组收益检验"}
            for name in MODEL_NAMES
        ] for horizon in HORIZONS},
        "predictions": {horizon: {CODE: {name: dict(prediction) for name in MODEL_NAMES}} for horizon in HORIZONS},
        "limitation": "构造数据",
    }


class ScoringGateTests(unittest.TestCase):
    """需求 F-7：``scoring_paused=True`` 时不得输出任何评分。"""

    def setUp(self):
        self.history = _history()
        self.context = _scoring_context(self.history)
        self.price = float(build_features(self.history).iloc[-1]["close"])

    def _analyze(self, **overrides):
        arguments = {"history": self.history, "current_price": self.price, "code": CODE,
                     "market_context": self.context, "quote_status": "已更新"}
        arguments.update(overrides)
        return analyze_stock(**arguments)

    def test_unpaused_analysis_carries_the_model_score(self):
        result = self._analyze()
        for horizon in HORIZONS:
            for model in result["models"][horizon]:
                self.assertTrue(model["validated"], model["status"])
                self.assertEqual(model["predicted_excess_return"], 0.02)
        # 计划里的这一项是页面判断「模型支持买入」的唯一依据，必须有值。
        self.assertIs(result["trade_plan"].get("model_relative_signal_positive"), True)

    def test_paused_analysis_emits_no_score_at_all(self):
        result = self._analyze(scoring_paused=True)
        for horizon in HORIZONS:
            for model in result["models"][horizon]:
                self.assertFalse(model["validated"], "降级时不能留下任何「已验证」的模型")
                self.assertEqual(model["status"], settings.DEGRADED_SCORING_MESSAGE)
                # 分数不是「保留但不用」，而是根本不存在：对象上留着数值就可能被别处读走。
                self.assertIsNone(model["predicted_excess_return"])
                self.assertIsNone(model["prediction_zscore"])
                self.assertIsNone(model["prediction_percentile"])
        plan = result["trade_plan"]
        self.assertEqual(plan["status"], "规则化参考（模型尚未通过样本外验证）")
        self.assertIsNone(plan["model_relative_signal_positive"])
        self.assertFalse(plan["signal_active"])

    def test_scoring_paused_defaults_to_off(self):
        """不传标志时必须与既有行为完全一致，否则所有老调用方都会静默改变输出。"""
        self.assertEqual(self._analyze()["models"][20][0]["validated"],
                         self._analyze(scoring_paused=False)["models"][20][0]["validated"])

    def test_incomplete_history_is_reported_before_the_degradation(self):
        """历史不完整比模型降级更贴近这只股票的真实处境，先说更具体的那一个。"""
        result = self._analyze(history_ready=False, scoring_paused=True)
        self.assertEqual(result["models"][20][0]["status"], "历史行情不完整或过期，拒绝输出模型预测")


# ---------------------------------------------------------------------------
# 告警策略：不连库的纯函数部分
# ---------------------------------------------------------------------------

class AlertPolicyTests(unittest.TestCase):
    """降级只看严重级告警，且原因要能一眼看出是「连续失败」还是「严重漂移」。"""

    def test_only_critical_alerts_cause_degradation(self):
        warning = {"level": "warning", "title": "门槛失败率超过 50%"}
        self.assertEqual(alerting.critical_alerts([warning]), [])
        self.assertIsNone(alerting.degradation_reason([warning]))

    def test_degradation_reason_joins_every_critical_title(self):
        alerts = [{"level": "critical", "title": "特征漂移严重：波动率"},
                  {"level": "warning", "title": "预测漂移：行业集中度偏高"},
                  {"level": "critical", "title": "LightGBM 连续 3 次未通过验证"}]
        self.assertEqual(alerting.degradation_reason(alerts),
                         "特征漂移严重：波动率；LightGBM 连续 3 次未通过验证")

    def test_webhook_without_address_is_not_an_error(self):
        """未配置通道是正常状态，不该让定时任务每天记一条无意义的异常。"""
        with mock.patch.dict(os.environ, {settings.ALERT_WEBHOOK_ENV: ""}, clear=False):
            result = alerting.send_webhook([{"level": "critical", "category": "drift",
                                             "title": "x", "message": "y"}])
        self.assertFalse(result["configured"])
        self.assertFalse(result["delivered"])
        self.assertIn(settings.ALERT_WEBHOOK_ENV, result["note"])


# ---------------------------------------------------------------------------
# 连库用例
# ---------------------------------------------------------------------------

@unittest.skipUnless(os.environ.get("QUANT_TEST_POSTGRES") == "1", "设置QUANT_TEST_POSTGRES=1运行隔离的数据库集成测试")
class ModelGovernanceDatabaseTests(unittest.TestCase):
    """留出集闸门、归档回滚、降级状态机与告警去重的真实读写。"""

    @classmethod
    def setUpClass(cls):
        cls.schema = "quant_governance_test_" + secrets.token_hex(6)
        _create_schema(cls.schema)
        cls.repository = _ScopedRepository(cls.schema)

    @classmethod
    def tearDownClass(cls):
        _drop_schema(cls.schema)

    def setUp(self):
        # 每个用例独立：库表清空，模型文件落在各自的临时目录里。
        self.model_dir = Path(tempfile.mkdtemp())
        self.data_dir = Path(tempfile.mkdtemp())
        self.patches = [mock.patch.object(settings, "MODEL_DIR", self.model_dir),
                        mock.patch.object(settings, "DATA_DIR", self.data_dir)]
        for patch in self.patches:
            patch.start()
        with self.repository.connect() as connection:
            connection.execute(f"TRUNCATE {TABLE_LIST}")

    def tearDown(self):
        for patch in self.patches:
            patch.stop()
        shutil.rmtree(self.model_dir, ignore_errors=True)
        shutil.rmtree(self.data_dir, ignore_errors=True)

    # -- 归档与回滚 ---------------------------------------------------------

    def _publish(self, context_id: str, validated: bool = True) -> tuple[dict, pd.DataFrame]:
        """用真实的 save_context 落一份活动产出（模型、摘要、面板三个文件）。"""
        context = {
            "context_id": context_id, "model_version": settings.MODEL_VERSION, "target_kind": TARGET_KIND,
            "created_at": "2026-10-16T15:00:00+08:00", "reference_date": "2026-10-16",
            "objective_mode": "lambdarank", "training_window_days": 750, "limitation": "构造数据",
            "models": {"20": [{"model": "LightGBM", "horizon": 20, "validated": validated, "status": "构造"}]},
            "predictions": {"20": {}}, "reference_closes": {CODE: 10.0},
        }
        panel = pd.DataFrame({"date": pd.to_datetime(["2026-10-15", "2026-10-16"]),
                              "code": [CODE, CODE], "close": [10.0, 10.0]})
        panel.attrs["context_id"] = context_id
        save_context(context, panel)
        return context, panel

    def _archive(self, context_id: str, validated: bool = True) -> None:
        context, panel = self._publish(context_id, validated)
        archive_context(self.repository, context, panel, passed=validated,
                        base_rate=1.0 if validated else 0.0)

    def _active_context_id(self) -> str | None:
        """活动产出的 context_id；load_context 会顺带校验摘要与面板是否对得上。"""
        context, _panel = load_context(include_panel=False)
        return (context or {}).get("context_id")

    def _registry(self) -> dict[str, dict]:
        with self.repository.connect() as connection:
            rows = connection.execute("SELECT * FROM public.quant_model_registry").fetchall()
        return {row["context_id"]: dict(row) for row in rows}

    def _events(self) -> list[dict]:
        with self.repository.connect() as connection:
            return connection.execute(
                "SELECT * FROM public.quant_model_governance_events ORDER BY id"
            ).fetchall()

    def test_registry_marks_only_archived_passing_versions_as_rollbackable(self):
        self._archive("ctx-old", validated=True)
        self._archive("ctx-failed", validated=False)
        # 未通过验证的那份仍然要登记（「模型变差了」本身就是信息），但不参与回滚候选。
        registry = self._registry()
        self.assertEqual(set(registry), {"ctx-old", "ctx-failed"})
        self.assertFalse(registry["ctx-failed"]["passed"])
        self.assertTrue(registry["ctx-failed"]["is_active"], "最新产出恒为活动版本")
        versions = {row["context_id"]: row for row in registry_versions(self.repository)}
        self.assertTrue(versions["ctx-old"]["rollbackable"])
        self.assertFalse(versions["ctx-failed"]["rollbackable"])

    def test_archive_keeps_the_linked_copy_after_a_new_version_replaces_the_file(self):
        """归档用硬链接，靠「save_context 从不原地改写」保证链接指向旧内容。"""
        self._archive("ctx-old")
        archived = self.model_dir / settings.MODEL_ARCHIVE_DIRNAME / "ctx-old"
        self._archive("ctx-new")
        self.assertEqual(joblib.load(archived / f"market_context_v{settings.MODEL_VERSION}.joblib")["context_id"],
                         "ctx-old")
        self.assertEqual(self._active_context_id(), "ctx-new")

    def test_rollback_swaps_the_active_files_and_stays_reversible(self):
        self._archive("ctx-old")
        self._archive("ctx-new")
        self.assertEqual(joblib.load(self.model_dir / f"market_context_v{settings.MODEL_VERSION}.joblib")["context_id"],
                         "ctx-new")
        result = rollback_to(self.repository, "ctx-old", operator="admin")
        self.assertEqual(result["context_id"], "ctx-old")
        self.assertEqual(self._active_context_id(), "ctx-old")
        self.assertEqual(
            pd.read_parquet(self.data_dir / f"ml_panel_v{settings.MODEL_VERSION}.parquet").attrs["context_id"],
            "ctx-old",
        )
        registry = self._registry()
        self.assertTrue(registry["ctx-old"]["is_active"])
        self.assertFalse(registry["ctx-new"]["is_active"])
        event = self._events()[-1]
        self.assertEqual((event["operation"], event["operator"]), ("回滚模型", "admin"))
        self.assertEqual(event["before_value"]["context_id"], "ctx-new")
        # 回滚前先把当时的活动版本归档，所以「回滚点错了」还能再回滚回去。
        versions = {row["context_id"]: row for row in registry_versions(self.repository)}
        self.assertTrue(versions["ctx-new"]["rollbackable"], "被换下去的那一版必须还能再回滚")
        self.assertTrue((self.model_dir / settings.MODEL_ARCHIVE_DIRNAME / "ctx-new").exists())

    def test_rollback_refuses_a_version_that_never_passed(self):
        self._archive("ctx-failed", validated=False)
        self._archive("ctx-new")
        with self.assertRaises(FinalHoldoutError) as context:
            rollback_to(self.repository, "ctx-failed", operator="admin")
        self.assertIn("没有通过样本外验证", str(context.exception))
        self.assertEqual(self._active_context_id(), "ctx-new")

    def test_rollback_refuses_the_active_version(self):
        self._archive("ctx-old")
        with self.assertRaises(FinalHoldoutError) as context:
            rollback_to(self.repository, "ctx-old", operator="admin")
        self.assertIn("已经是当前活动模型", str(context.exception))

    def test_rollback_refuses_an_unknown_version(self):
        with self.assertRaises(FinalHoldoutError) as context:
            rollback_to(self.repository, "ctx-nothing", operator="admin")
        self.assertIn("没有找到", str(context.exception))

    def test_rollback_refuses_when_the_archive_directory_is_gone(self):
        self._archive("ctx-old")
        self._archive("ctx-new")
        shutil.rmtree(self.model_dir / settings.MODEL_ARCHIVE_DIRNAME / "ctx-old")
        versions = {row["context_id"]: row for row in registry_versions(self.repository)}
        self.assertFalse(versions["ctx-old"]["rollbackable"], "归档没了，页面上的回滚按钮就该是灰的")
        with self.assertRaises(FinalHoldoutError) as context:
            rollback_to(self.repository, "ctx-old", operator="admin")
        self.assertIn("归档目录已不存在", str(context.exception))
        self.assertEqual(self._active_context_id(), "ctx-new")

    def test_rollback_refuses_an_incomplete_archive(self):
        self._archive("ctx-old")
        self._archive("ctx-new")
        archived = self.model_dir / settings.MODEL_ARCHIVE_DIRNAME / "ctx-old"
        (archived / f"ml_panel_v{settings.MODEL_VERSION}.parquet").unlink()
        with self.assertRaises(FinalHoldoutError) as context:
            rollback_to(self.repository, "ctx-old", operator="admin")
        self.assertIn("归档文件不完整", str(context.exception))
        self.assertEqual(self._active_context_id(), "ctx-new")

    def test_archive_pruning_never_removes_the_active_version(self):
        for name in ("ctx-a", "ctx-b", "ctx-c"):
            self._archive(name)
        # 强制一个确定的新旧顺序，避免同一微秒内的插入顺序不确定。
        with self.repository.connect() as connection:
            connection.execute("UPDATE public.quant_model_registry SET created_at=now()-interval '3 hours' "
                               "WHERE context_id=%s", ("ctx-a",))
            connection.execute("UPDATE public.quant_model_registry SET created_at=now()-interval '2 hours' "
                               "WHERE context_id=%s", ("ctx-b",))
        removed = model_governance._prune_archive(self.repository, keep=1)
        self.assertEqual(removed, 1)
        archive_root = self.model_dir / settings.MODEL_ARCHIVE_DIRNAME
        self.assertFalse((archive_root / "ctx-a").exists())
        self.assertTrue((archive_root / "ctx-b").exists())
        self.assertTrue((archive_root / "ctx-c").exists(), "活动版本无论多旧都不删")
        registry = self._registry()
        self.assertIsNone(registry["ctx-a"]["archive_path"], "删了目录就要把登记路径清掉，否则页面还显示可回滚")
        self.assertTrue(registry["ctx-c"]["is_active"])
        self.assertTrue(registry["ctx-c"]["archive_path"])

    # -- 最终留出集 ---------------------------------------------------------

    @staticmethod
    def _canned_report(passed: bool) -> dict:
        """构造一份「冻结模型算出来的」留出集报告，避免用例真的去训模型。"""
        return {
            "model_version": settings.MODEL_VERSION, "context_id": "ctx-holdout",
            "holdout_start": "2024-01-02", "holdout_end": "2026-10-15",
            "evaluated_at": "2026-10-16T15:30:00+08:00", "holdout_dates": 690,
            "passed": passed, "failed_horizon": None if passed else "60",
            "turnover": {"status": "已计算", "average_turnover": 0.4, "after_cost_return": 0.012},
            "summary": {"rank_ic": {"20": 0.03}, "average_turnover": 0.4},
        }

    def test_final_holdout_runs_once_and_force_overwrites_with_audit(self):
        context = {"model_version": settings.MODEL_VERSION, "context_id": "ctx-holdout"}
        panel = pd.DataFrame({"date": pd.to_datetime(["2026-10-15", "2026-10-16"]), "code": [CODE, CODE]})
        trained: list[bool] = []

        def build(frame, ctx=None, progress=None):
            trained.append(True)
            return self._canned_report(passed=len(trained) == 1)

        with mock.patch.object(model_governance, "build_holdout_evaluation", side_effect=build):
            first = evaluate_final_holdout(self.repository, operator="admin", context=context, panel=panel)
            self.assertTrue(first["passed"])
            self.assertFalse(first["forced"])
            record = final_holdout_record(self.repository)
            self.assertEqual((record["operator"], record["forced"], record["passed"]), ("admin", False, True))
            self.assertEqual(record["report_path"], str(self.model_dir / settings.FINAL_HOLDOUT_REPORT_NAME))
            self.assertFalse(holdout_report()["forced"])

            with self.assertRaises(FinalHoldoutError) as refused:
                evaluate_final_holdout(self.repository, operator="admin", context=context, panel=panel)
            self.assertIn("只允许评估一次", str(refused.exception))
            self.assertEqual(len(trained), 1, "被拒绝时不该白跑一遍冻结模型训练")

            forced = evaluate_final_holdout(self.repository, operator="admin", force=True,
                                            context=context, panel=panel)
            self.assertTrue(forced["forced"])
            self.assertFalse(forced["passed"])

        record = final_holdout_record(self.repository)
        self.assertTrue(record["forced"])
        self.assertFalse(record["passed"], "强制重评要用新结论覆盖旧结论")
        self.assertFalse(holdout_report()["passed"], "落盘的报告也要跟着覆盖")
        self.assertEqual([event["operation"] for event in self._events()],
                         ["最终留出集评估", "强制重新评估最终留出集"])
        self.assertIn("force", self._events()[-1]["note"])

    def test_final_holdout_refuses_a_context_from_another_version(self):
        with self.assertRaises(FinalHoldoutError) as context:
            evaluate_final_holdout(self.repository, context={"model_version": "9.9.9", "context_id": "ctx-other"},
                                   panel=pd.DataFrame({"date": pd.to_datetime(["2026-10-15"])}))
        self.assertIn("不一致", str(context.exception))
        self.assertIsNone(final_holdout_record(self.repository))

    # -- 降级状态机与告警 ---------------------------------------------------

    def test_degradation_state_machine_is_audited(self):
        self.assertFalse(model_status(self.repository)["degraded"], "没有记录就是未降级，不是未知")
        degraded = degrade_model(self.repository, "特征漂移严重：波动率 PSI 12.400", operator="scheduled")
        self.assertTrue(degraded["degraded"])
        self.assertEqual(degraded["updated_by"], "scheduled")
        self.assertEqual(degraded["scoring_message"], settings.DEGRADED_SCORING_MESSAGE)

        clear_degradation(self.repository, operator="admin", note="确认是构造数据的误报")
        status = model_status(self.repository)
        self.assertFalse(status["degraded"])
        self.assertIsNone(status["reason"])
        self.assertIsNone(status["scoring_message"])
        events = self._events()
        self.assertEqual([event["operation"] for event in events], ["模型降级", "解除降级"])
        self.assertTrue(events[1]["before_value"]["degraded"])
        self.assertFalse(events[1]["after_value"]["degraded"])
        self.assertEqual(events[1]["note"], "确认是构造数据的误报")

    def test_status_read_failure_is_reported_instead_of_looking_healthy(self):
        broken = mock.Mock()
        broken.connect.side_effect = DatabaseError("连接失败")
        status = current_model_status(broken)
        self.assertFalse(status["degraded"])
        self.assertIn("连接失败", status["error"])
        # 扫描侧同样的取舍：读不到就照常出分，但必须把原因留下来给页面显示。
        governance = ScanGovernance(broken, trigger="manual")
        self.assertFalse(governance.scoring_paused())
        self.assertTrue(governance.errors)

    def test_scoring_paused_follows_the_health_state(self):
        self.assertFalse(ScanGovernance(self.repository).scoring_paused())
        degrade_model(self.repository, "构造的降级原因")
        self.assertTrue(ScanGovernance(self.repository).scoring_paused())

    def test_alerts_are_deduplicated_within_the_window(self):
        alert = {"level": "critical", "category": "drift", "title": "特征漂移严重：波动率", "message": "构造"}
        self.assertEqual(len(alerting.log_alerts(self.repository, [alert])), 1)
        self.assertEqual(alerting.log_alerts(self.repository, [alert]), [], "同一条消息在去重窗口内只写一次")

        result = alerting.dispatch_alerts(self.repository, [alert])
        self.assertEqual(result["stored"], 0)
        self.assertEqual(result["suppressed"], 1, "被抑制的不算「今天有告警」")

        other = {**alert, "title": "概念漂移：Rolling Rank IC 连续 5 个交易日为负"}
        self.assertEqual(len(alerting.log_alerts(self.repository, [other])), 1, "标题不同就不是同一条消息")
        other_version = {**alert, "model_version": "9.9.9"}
        self.assertEqual(len(alerting.log_alerts(self.repository, [other_version])), 1,
                         "换了模型版本要重新报，否则新版本会继承旧版本的静默")

    # -- 扫描集成 -----------------------------------------------------------

    @staticmethod
    def _panel(days: int = 100, codes: int = 30, volatility_shift: bool = False) -> pd.DataFrame:
        """构造面板：最近 20 个交易日是漂移判定的当前窗口。

        稳定维度按「每只股票一个固定水平」生成：两个窗口里每只股票出现次数不同但取值相同，
        分布因此完全一致，PSI 恒为 0——断言不受随机噪声影响。漂移维度才真的平移分布。
        """
        generator = np.random.default_rng(2026)
        dates = pd.bdate_range(end="2026-10-16", periods=days)
        rows = []
        for index, date in enumerate(dates):
            shifted = volatility_shift and index >= days - 20
            for code_index in range(codes):
                rows.append({
                    "date": date, "code": f"{600000 + code_index:06d}",
                    "volatility_20": float(generator.normal(0.08 if shifted else 0.02, 0.0008)),
                    "turnover_20": 0.01 + 0.001 * code_index,
                    "amihud_20": 1e-6 * (1 + 0.01 * code_index),
                    "log_market_cap": 22.0 + 0.1 * code_index,
                    "industry": ("银行", "医药", "电子")[code_index % 3],
                })
        return pd.DataFrame(rows)

    def _scan_context(self, panel: pd.DataFrame, as_json_keys: bool = False) -> dict:
        """一份扫描产出的上下文。

        ``as_json_keys`` 为真时周期键是 str——那是摘要 JSON 往返之后的形态，
        与训练产物里的 int 键并存，两条路径都必须能写进概念漂移与预测漂移。
        """
        report_date = str(pd.to_datetime(panel["date"]).max().date())
        codes = sorted(panel["code"].astype(str).unique())
        metrics = {
            "dates": 60, "rank_ic": 0.05, "rank_ic_hac_t": 2.4, "ndcg_ratio": 1.2,
            "score_kind": SCORE_KIND_RANK, "top_excess_return": 0.03,
            # Top 超额门槛比的是「相对股票池均值」的超额，缺了 universe_excess_return
            # 这一项会被判成没过——构造指标时少一个键，通过率就会从 50% 变成 0%。
            "universe_excess_return": 0.0, "top_bottom_spread": 0.05,
            "mae": None, "baseline_mae": None,
            "folds": [{"trained_from": "2021-11-03", "trained_through": "2023-12-29", "training_dates": 530}],
            "daily_groups": [{"date": f"2026-08-{index:02d}", "rank_ic": 0.02 + 0.001 * index}
                             for index in range(1, 31)],
        }
        key = "20" if as_json_keys else 20
        return {
            "context_id": "ctx-scan", "model_version": settings.MODEL_VERSION, "target_kind": TARGET_KIND,
            "reference_date": report_date, "objective_mode": "lambdarank", "training_window_days": 750,
            "models": {key: [
                {"model": "LightGBM", "horizon": 20, "objective": "lambdarank", "score_kind": SCORE_KIND_RANK,
                 "validated": True, "training_dates": 530,
                 "status": "通过样本外基线、Rank IC与分组收益检验", "oos_metrics": metrics},
                {"model": "XGBoost", "horizon": 20, "objective": "lambdarank",
                 "validated": False, "status": "截面样本不足，未生成预测", "oos_metrics": None},
            ]},
            "predictions": {key: {code: {"LightGBM": {"predicted_excess_return": 0.03 - 0.001 * index}}
                                  for index, code in enumerate(codes)}},
        }

    def _scan(self, panel: pd.DataFrame, context: dict | None = None, **kwargs) -> dict:
        """按真实顺序跑一轮扫描治理：先落盘活动产出，再写通过率、漂移、归档与告警。

        ``after_scan`` 归档的是 ``save_context`` 刚写下的那三个文件，所以这里必须先发布一份，
        否则归档会（正确地）报「活动文件缺失」，回滚候选也就永远是空的。
        """
        self._publish("ctx-scan")
        return ScanGovernance(self.repository, **kwargs).after_scan(context or self._scan_context(panel), panel)

    def _drift_row(self, table: str, **filters) -> dict | None:
        clause = "".join(f" AND {name}=%s" for name in filters)
        with self.repository.connect() as connection:
            return connection.execute(
                f"SELECT * FROM public.{table} WHERE 1=1{clause}", tuple(filters.values())
            ).fetchone()

    def test_scan_writes_all_three_drift_tables_and_the_validation_log(self):
        panel = self._panel()
        summary = self._scan(panel, trigger="scheduled")
        self.assertEqual(summary["errors"], [])
        self.assertEqual(summary["validation_rows"], 2, "通过的和没通过的都要写，分母不能被悄悄缩小")
        self.assertEqual(summary["feature_rows"], len(settings.DRIFT_MONITORED_FEATURES))
        self.assertEqual(summary["concept_rows"], 1, "只有通过验证的模型才有概念漂移可算")
        self.assertEqual(summary["prediction_rows"], 1)
        self.assertEqual(summary["pass_rate"]["pass_rate"], 0.5)
        self.assertEqual(summary["alerts"], 0)
        self.assertFalse(summary["degraded"])

        report_date = str(panel["date"].max().date())
        concept = self._drift_row("quant_concept_drift", date=report_date)
        self.assertEqual(concept["window_days"], 30, "可用样本外交易日只有 30 天，窗口按实际可用量收窄")
        self.assertEqual(concept["negative_streak"], 0)
        # 没有历史 HHI 就不给等级：样本不足不等于「正常」。
        prediction = self._drift_row("quant_prediction_drift", date=report_date)
        self.assertEqual(prediction["hhi_threshold_exceeded"], drift_monitor.LEVEL_INSUFFICIENT)
        self.assertIn("历史样本不足", prediction["note"])
        self.assertEqual(prediction["stocks"], settings.DRIFT_PREDICTION_TOP_N)

        # 重复扫描只更新当天的行，不追加重复行。
        self._scan(panel, trigger="scheduled")
        with self.repository.connect() as connection:
            counts = connection.execute(
                "SELECT (SELECT count(*) FROM public.quant_concept_drift) AS concept,"
                "(SELECT count(*) FROM public.quant_feature_drift) AS feature,"
                "(SELECT count(*) FROM public.quant_prediction_drift) AS prediction"
            ).fetchone()
        self.assertEqual(counts["concept"], 1)
        self.assertEqual(counts["feature"], len(settings.DRIFT_MONITORED_FEATURES))
        self.assertEqual(counts["prediction"], 1)

    def test_scan_writes_drift_for_json_shaped_horizon_keys(self):
        """摘要 JSON 往返后周期键是 str。若只认 int，预测漂移会被写成「本轮没有通过验证的模型」——
        那是把「取不到预测」说成「模型都没通过」，因此这里连同落库结果一起校验。"""
        panel = self._panel()
        summary = self._scan(panel, context=self._scan_context(panel, as_json_keys=True), trigger="manual")
        self.assertEqual(summary["concept_rows"], 1)
        self.assertEqual(summary["prediction_rows"], 1)
        report_date = str(panel["date"].max().date())
        concept = self._drift_row("quant_concept_drift", date=report_date)
        self.assertGreater(concept["rank_ic"], 0, "周期键必须取到了模型，否则这里会是一条空行")
        prediction = self._drift_row("quant_prediction_drift", date=report_date)
        self.assertEqual(prediction["hhi_threshold_exceeded"], drift_monitor.LEVEL_INSUFFICIENT)
        self.assertEqual(prediction["stocks"], settings.DRIFT_PREDICTION_TOP_N, "取到预测才凑得出 Top-N 组合")
        self.assertIsNotNone(prediction["industry_hhi"])

    def test_valuation_dimension_is_reported_as_unavailable_not_fabricated(self):
        """行情源没有 PE/PB，这一维必须是一条写明原因的「未接入」，不能跳过也不能算个假 PSI。"""
        rows = {row["feature_name"]: row for row in drift_monitor.feature_drift_rows(self._panel())}
        valuation = rows["估值"]
        self.assertEqual(valuation["threshold_exceeded"], drift_monitor.LEVEL_UNAVAILABLE)
        self.assertIsNone(valuation["psi"])
        self.assertIn("PE/PB", valuation["note"])
        self.assertEqual(drift_monitor.evaluate_drift_alerts([valuation], [], []), [])

    def test_scan_registers_a_rollback_candidate(self):
        """扫描必须真的登记版本。归档只被回滚调用过的话（老实现就是如此），
        「模型版本与回滚」页永远是空列表，出了一次坏模型就只能干看着。"""
        self._archive("ctx-good")
        summary = self._scan(self._panel())
        self.assertEqual(summary["archive"]["context_id"], "ctx-scan")
        self.assertTrue(summary["archive"]["passed"])
        registry = self._registry()
        self.assertEqual(set(registry), {"ctx-good", "ctx-scan"})
        self.assertTrue(registry["ctx-scan"]["is_active"], "本轮产物就是当前活动版本")
        self.assertFalse(registry["ctx-good"]["is_active"], "上一版退位，但仍然可回滚")
        versions = {row["context_id"]: row for row in registry_versions(self.repository)}
        self.assertTrue(versions["ctx-good"]["rollbackable"])

    def test_scan_archives_even_when_no_model_passes(self):
        """没有模型通过验证时也要登记：不通过回滚候选，但「这版变差了」本身是信息。"""
        context = self._scan_context(self._panel())
        context["models"][20] = [dict(model, validated=False, oos_metrics=None) for model in context["models"][20]]
        summary = self._scan(self._panel(), context=context)
        self.assertFalse(summary["archive"]["passed"])
        self.assertEqual(summary["pass_rate"]["pass_rate"], 0.0)
        registry = self._registry()
        self.assertFalse(registry["ctx-scan"]["passed"])
        self.assertFalse({row["context_id"]: row for row in registry_versions(self.repository)}
                         ["ctx-scan"]["rollbackable"])

    def test_severe_drift_degrades_automatically_but_never_rolls_back(self):
        self._archive("ctx-good")
        panel = self._panel(volatility_shift=True)
        summary = self._scan(panel, trigger="scheduled")
        self.assertEqual(summary["alerts"], 1, "只有波动率这一维真的漂移了")
        self.assertTrue(summary["degraded"], "严重漂移必须自动降级，否则系统照常出分")
        self.assertIn("特征漂移严重：波动率", summary["degradation_reason"])
        status = model_status(self.repository)
        self.assertEqual(status["scoring_message"], settings.DEGRADED_SCORING_MESSAGE)
        self.assertEqual(status["updated_by"], "scheduled")
        alerts = recent_alerts(self.repository, only_open=True)
        self.assertEqual([alert["level"] for alert in alerts], ["critical"])
        acknowledge_alert(self.repository, alerts[0]["id"], "admin")
        self.assertEqual(recent_alerts(self.repository, only_open=True), [])
        # 自动动作只到「标降级 + 停分 + 写告警」：生产模型一个字节都不该被换掉。
        self.assertEqual(self._active_context_id(), "ctx-scan", "降级不换模型，活动版本仍是本轮产出")
        self.assertTrue((self.model_dir / settings.MODEL_ARCHIVE_DIRNAME / "ctx-good").exists())

    def test_persistent_drift_degrades_again_after_a_manual_clear(self):
        """管理员解除降级是「我确认过了」，不是「模型没事了」：漂移还在，下一轮就再降回来。"""
        panel = self._panel(volatility_shift=True)
        self._scan(panel)
        clear_degradation(self.repository, operator="admin", note="先解一次看看到底是不是误报")
        self.assertFalse(model_status(self.repository)["degraded"])
        summary = self._scan(panel)
        self.assertTrue(summary["degraded"])
        self.assertEqual(summary["alert_result"]["suppressed"], 1, "告警被去重抑制，但降级判定照常执行")

    def test_failure_streaks_expose_the_position_the_judgement_reads(self):
        """判定函数按 ``position`` 取「最近 N 次」。取数 SQL 漏掉这一列时，
        只要有任意一条检验记录，连续失败告警就会 KeyError——
        而它被扫描管线兜住，表现为页面上一句「模型治理未完成」，通过率告警悄悄失效。"""
        panel = self._panel()
        self._scan(panel)
        streaks = model_monitor.fetch_failure_streaks(self.repository, limit=settings.CONSECUTIVE_FAILURE_ALERT)
        self.assertTrue(streaks)
        for row in streaks:
            self.assertIn("position", row)
        self.assertEqual(model_monitor.check_pass_rate_alerts(self.repository), [],
                         "每个序列只有一次记录，还够不上连续失败")

    def test_scan_without_model_context_reports_the_gap_instead_of_silence(self):
        summary = ScanGovernance(self.repository).after_scan(None, self._panel())
        self.assertTrue(any("未生成模型上下文" in message for message in summary["errors"]))
        self.assertEqual(summary["validation_rows"], 0)
        self.assertEqual(summary["feature_rows"], 0)


if __name__ == "__main__":
    unittest.main()
