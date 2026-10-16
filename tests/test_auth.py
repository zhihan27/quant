"""验证认证边界；PostgreSQL 集成测试使用独立临时模式。"""

import json
import os
import secrets
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from streamlit.testing.v1 import AppTest

from config import settings
from services.auth import (
    INITIAL_ADMIN_PASSWORD,
    AuthService,
    AuthenticationError,
    token_digest,
    validate_password,
)
from services.database import connection_settings

# 导航始终注册的业务页；账号管理与模型健康页只对管理员注册，因此不属于公开路由。
BUSINESS_PAGES = ["1_个股分析.py", "2_量化选股.py", "3_自选与持仓.py", "4_盘中监控.py"]
ACCOUNT_PAGE = "5_账号管理.py"
MODEL_HEALTH_PAGE = "6_模型健康.py"


class AuthInputTests(unittest.TestCase):
    """正式密码和未登录路由不得绕过认证。"""

    def test_weak_password_is_rejected(self):
        # 7位密码用于钉住8位下限；弱口令表本身不拦截 admin123，初始管理员的公开口令依赖页面告警。
        for password in ("admin", "123456", "Ab1!xyz", "x" * 129):
            with self.assertRaises(ValueError):
                validate_password(password)

    def test_all_routes_require_login_without_database(self):
        with patch("services.auth_ui.get_auth_service", side_effect=AssertionError("未登录首屏不应连接数据库")):
            app = AppTest.from_file(settings.PROJECT_ROOT / "app.py", default_timeout=20).run()
            for name in BUSINESS_PAGES:
                app.switch_page(str(settings.PROJECT_ROOT / "pages" / name)).run()
                self.assertFalse(app.exception)
                self.assertEqual([item.value for item in app.title], ["量化研究登录"])
                self.assertEqual([item.label for item in app.text_input], ["用户名", "密码"])

    def test_account_page_is_not_a_reachable_route_without_database(self):
        """账号管理页对未登录者不注册，未注册的脚本路径无法通过页面地址直接打开。"""
        with patch("services.auth_ui.get_auth_service", side_effect=AssertionError("未登录首屏不应连接数据库")):
            app = AppTest.from_file(settings.PROJECT_ROOT / "app.py", default_timeout=20).run()
            with self.assertRaises(ValueError):
                app.switch_page(str(settings.PROJECT_ROOT / "pages" / ACCOUNT_PAGE))


class _ScopedConnection:
    """仅将待测业务 SQL 映射到测试专用模式。"""

    def __init__(self, connection, schema):
        self.connection = connection
        self.schema = schema

    def execute(self, query, parameters=None):
        return self.connection.execute(query.replace("public.", self.schema + "."), parameters)


class _ScopedRepository:
    """共享真实数据库连接配置，所有测试数据写入随机模式。"""

    def __init__(self, schema):
        self.schema = schema

    @contextmanager
    def connect(self):
        with psycopg.connect(**connection_settings(), row_factory=dict_row) as connection:
            yield _ScopedConnection(connection, self.schema)


@unittest.skipUnless(os.environ.get("QUANT_TEST_POSTGRES") == "1", "设置QUANT_TEST_POSTGRES=1运行隔离的数据库集成测试")
class AuthDatabaseTests(unittest.TestCase):
    """验证真实事务、失败限流、令牌过期及密码变更撤销。"""

    @classmethod
    def setUpClass(cls):
        cls.schema = "quant_auth_test_" + secrets.token_hex(6)
        cls.repository = _ScopedRepository(cls.schema)
        with psycopg.connect(**connection_settings()) as connection:
            connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(cls.schema)))
            for migration in sorted((settings.PROJECT_ROOT / "sql").glob("*.sql")):
                connection.execute(migration.read_text(encoding="utf-8").replace("public.", cls.schema + "."))

    @classmethod
    def tearDownClass(cls):
        with psycopg.connect(**connection_settings()) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(cls.schema)))

    def setUp(self):
        with self.repository.connect() as connection:
            connection.execute("TRUNCATE public.quant_users,public.quant_auth_sessions,public.quant_login_limits CASCADE")
        self.service = AuthService(self.repository)
        self.password = "Quant-Test-Password-2026!"
        self.service.create_user("tester", self.password)

    def test_hash_and_server_side_logout(self):
        token = self.service.login("TESTER", self.password, "127.0.0.1")
        with self.repository.connect() as connection:
            row = connection.execute("SELECT password_hash FROM public.quant_users").fetchone()
            stored = connection.execute("SELECT token_hash FROM public.quant_auth_sessions").fetchone()
        self.assertTrue(row["password_hash"].startswith("$argon2id$"))
        self.assertNotIn(self.password, row["password_hash"])
        self.assertEqual(stored["token_hash"], token_digest(token))
        self.assertEqual(self.service.session(token)["username"], "tester")
        self.service.logout(token)
        self.assertIsNone(self.service.session(token))

    def test_five_failed_logins_persist_across_service_instances(self):
        for _ in range(5):
            with self.assertRaisesRegex(AuthenticationError, "用户名或密码错误"):
                self.service.login("tester", "wrong", "127.0.0.1")
        with self.assertRaisesRegex(AuthenticationError, "过于频繁"):
            AuthService(self.repository).login("tester", self.password, "127.0.0.1")
        with self.repository.connect() as connection:
            attempts = connection.execute("SELECT attempts FROM public.quant_login_limits WHERE bucket=%s", ("user:" + token_digest("tester"),)).fetchone()["attempts"]
        self.assertEqual(attempts, 6)

    def test_parallel_attempts_cannot_bypass_account_window(self):
        def reserve(_):
            try:
                self.service._reserve_login("tester", "127.0.0.1")
                return True
            except AuthenticationError:
                return False
        with ThreadPoolExecutor(max_workers=8) as executor:
            outcomes = list(executor.map(reserve, range(12)))
        self.assertEqual(sum(outcomes), 5)

    def test_session_expiry_and_disabled_user(self):
        token = self.service.login("tester", self.password, "127.0.0.1")
        with self.repository.connect() as connection:
            connection.execute("UPDATE public.quant_auth_sessions SET last_seen_at=now()-interval '31 minutes'")
        self.assertIsNone(self.service.session(token))
        token = self.service.login("tester", self.password, "127.0.0.1")
        self.service.disable_user("tester")
        self.assertIsNone(self.service.session(token))
        with self.assertRaisesRegex(AuthenticationError, "用户名或密码错误"):
            self.service.login("tester", self.password, "127.0.0.1")

    def test_temporary_password_gate_and_change_revokes_all_sessions(self):
        self.service.set_password("tester", "admin", temporary=True)
        first = self.service.login("tester", "admin", "127.0.0.1")
        second = self.service.login("tester", "admin", "127.0.0.1")
        self.assertTrue(self.service.session(first)["must_change_password"])
        self.service.change_password(first, "admin", self.password)
        self.assertIsNone(self.service.session(first))
        self.assertIsNone(self.service.session(second))
        new = self.service.login("tester", self.password, "127.0.0.1")
        self.assertFalse(self.service.session(new)["must_change_password"])

    def test_reset_revokes_all_sessions(self):
        token = self.service.login("tester", self.password, "127.0.0.1")
        self.service.set_password("tester", "Replacement-Password-2026!")
        self.assertIsNone(self.service.session(token))

    def test_forced_change_ui_prevents_business_access(self):
        self.service.set_password("tester", "admin", temporary=True)
        with patch("services.auth_ui.get_auth_service", return_value=self.service):
            app = AppTest.from_file(settings.PROJECT_ROOT / "app.py", default_timeout=20).run()
            app.text_input[0].set_value("tester")
            app.text_input[1].set_value("admin")
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertEqual([item.value for item in app.title], ["修改登录密码"])
            for name in BUSINESS_PAGES:
                app.switch_page(str(settings.PROJECT_ROOT / "pages" / name)).run()
                self.assertFalse(app.exception)
                self.assertEqual([item.value for item in app.title], ["修改登录密码"])

    def test_authenticated_pages_and_logout(self):
        token = self.service.login("tester", self.password, "127.0.0.1")
        with patch("services.auth_ui.get_auth_service", return_value=self.service), patch.dict(os.environ, {"QUANT_DISABLE_MONITOR": "1"}):
            app = AppTest.from_file(settings.PROJECT_ROOT / "app.py", default_timeout=20)
            app.session_state["auth_token"] = token
            app.run()
            for name in BUSINESS_PAGES:
                app.switch_page(str(settings.PROJECT_ROOT / "pages" / name)).run()
                self.assertFalse(app.exception, name)
                self.assertNotIn("量化研究登录", [item.value for item in app.title])
            # 普通账号看不到账号管理页：未注册，切换过去会直接报错而不是渲染后拒绝。
            with self.assertRaises(ValueError):
                app.switch_page(str(settings.PROJECT_ROOT / "pages" / ACCOUNT_PAGE))
            next(button for button in app.button if button.label == "退出登录").click().run()
            self.assertEqual([item.value for item in app.title], ["量化研究登录"])
            self.assertIsNone(self.service.session(token))


    def test_scan_trigger_is_available_only_to_administrators(self):
        """量化选股页对所有账号开放，但只有管理员能发起扫描。

        扫描训练出的模型上下文是所有账号共用的，谁扫谁就决定别人今天看到什么；
        断言「按钮不存在」而不是「点了会被拒绝」，才能拦住把权限只写进提示语的回归。
        """
        selection_page = str(settings.PROJECT_ROOT / "pages" / "2_量化选股.py")
        self.service.ensure_initial_admin()
        with patch("services.auth_ui.get_auth_service", return_value=self.service), patch(
            "services.runtime.get_repository", return_value=self.repository
        ), patch("services.runtime.get_market_data", return_value=object()), patch.dict(
            os.environ, {"QUANT_DISABLE_MONITOR": "1"}
        ):
            app = AppTest.from_file(settings.PROJECT_ROOT / "app.py", default_timeout=20)
            app.session_state["auth_token"] = self.service.login("tester", self.password, "127.0.0.1")
            app.run()
            app.switch_page(selection_page).run()
            self.assertFalse(app.exception)
            self.assertNotIn("开始扫描", [button.label for button in app.button])
            # 收走扫描权不等于收走结果：只读视图必须看得到最近一次排名。
            self.assertTrue(any("最近一次扫描" in item.value for item in app.info))

            app.session_state["auth_token"] = self.service.login("admin", INITIAL_ADMIN_PASSWORD, "127.0.0.1")
            app.run()
            app.switch_page(selection_page).run()
            self.assertFalse(app.exception)
            self.assertIn("开始扫描", [button.label for button in app.button])

    def test_read_only_view_renders_the_persisted_result_for_everyone(self):
        """收走扫描权不等于收走结果：只读账号必须真能把落盘的那一版排名渲染出来。"""
        payload = {
            "model_version": settings.MODEL_VERSION,
            "finished_at": "2026-10-15T15:42:00+08:00",
            "trigger": "scheduled",
            "pool_mode": "沪深京全部 A 股",
            "reference_date": "2026-10-15",
            "context_id": "ctx-test",
            "scan_stats": {"scanned": 5312, "scored": 5010, "failed": 302},
            "rows": [{
                "综合评分": 88.5, "代码": "600000", "名称": "浦发银行", "最新价": 10.0, "涨跌幅": 1.2,
                "参考买入区间": "9.80 ~ 10.20", "止损参考": 9.5, "交易状态": "可正常评估",
                "推荐理由": "测试理由", "主要风险": "测试风险",
            }],
            "detail_rows": [], "failed_rows": [],
            "usable": [{"code": "600000", "name": "浦发银行"}],
            "notes": [{"level": "caption", "text": "测试血缘说明"}],
            # 落盘时逐笔成交、每日净值、每段训练日志与期末持仓已被剔除，只留汇总口径。
            "portfolio": {
                "metrics": {"total_return": 0.18, "annualized_return": 0.07, "maximum_drawdown": -0.22,
                            "sharpe_ratio": 0.9, "win_rate": 0.54, "profit_loss_ratio": 1.4, "trades": 96},
                "benchmark_buy_hold_return": 0.05,
                "portfolio": {"top_n": 20, "per_stock_weight": 0.05, "max_weight": 0.05,
                              "average_holdings": 18.4, "invested_ratio": 0.92, "cash_ratio": 0.08,
                              "cap_binding": True},
                "universe": {"kept": 300, "limit": 300, "status": "已裁剪"},
                "score_range": {"start": "2025-01-02", "end": "2026-10-15", "dates": 430},
                "rebalance_every": 1, "unfilled_buys": 7, "unfilled_sells": 3,
                "universe_equal_weight_return": 0.03,
                "costs": {"commission_rate_each_side": 0.0003, "stamp_duty_sell_side": 0.0005,
                          "slippage_each_side": 0.001},
                "model_version": settings.MODEL_VERSION,
                "constraints": ["T+1", "涨跌停不成交"],
            },
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "last_scan.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            with patch("services.scan_store.SCAN_RESULT_PATH", path), patch(
                "services.auth_ui.get_auth_service", return_value=self.service
            ), patch("services.runtime.get_repository", return_value=self.repository), patch(
                "services.runtime.get_market_data", return_value=object()
            ), patch.dict(os.environ, {"QUANT_DISABLE_MONITOR": "1"}):
                app = AppTest.from_file(settings.PROJECT_ROOT / "app.py", default_timeout=20)
                app.session_state["auth_token"] = self.service.login("tester", self.password, "127.0.0.1")
                app.run()
                app.switch_page(str(settings.PROJECT_ROOT / "pages" / "2_量化选股.py")).run()
                self.assertFalse(app.exception)
                captions = [item.value for item in app.caption]
                self.assertTrue(any("每日定时任务" in text for text in captions), captions[:3])
                self.assertTrue(any("截面日期 2026-10-15" in text for text in captions), captions[:3])
                self.assertIn("600000", list(app.dataframe[0].value["代码"]))
                # 组合汇总也必须渲染得出来：明细被落盘剔除了，页面不能因此整块消失。
                self.assertTrue(any("每个交易日按评分买前" in text for text in captions))
                self.assertTrue(any("跑赢 13.00 个百分点" in item.value for item in app.info))
                self.assertFalse(any("KeyError" in str(item.value) for item in app.warning))

    def test_pending_account_cannot_log_in_before_activation(self):
        self.service.create_pending_user("newcomer")
        with self.assertRaisesRegex(AuthenticationError, "尚未激活"):
            self.service.login("newcomer", self.password, "127.0.0.1")

    def test_activation_sets_password_once_and_logs_in(self):
        self.service.create_pending_user("newcomer")
        chosen = "Self-Chosen-Password-2026!"
        self.service.activate_user("newcomer", chosen)
        self.assertEqual(self.service.session(self.service.login("newcomer", chosen, "127.0.0.1"))["username"], "newcomer")
        with self.assertRaisesRegex(ValueError, "已激活"):
            self.service.activate_user("newcomer", "Another-Password-2026!")

    def test_activation_rejects_weak_password_and_disabled_account(self):
        self.service.create_pending_user("newcomer")
        with self.assertRaises(ValueError):
            self.service.activate_user("newcomer", "Ab1!xyz")
        self.service.disable_user("newcomer")
        with self.assertRaisesRegex(ValueError, "已被禁用"):
            self.service.activate_user("newcomer", "Self-Chosen-Password-2026!")

    def test_reset_returns_account_to_pending_and_revokes_sessions(self):
        token = self.service.login("tester", self.password, "127.0.0.1")
        self.service.reset_password("tester")
        self.assertIsNone(self.service.session(token))
        with self.assertRaisesRegex(AuthenticationError, "尚未激活"):
            self.service.login("tester", self.password, "127.0.0.1")
        self.service.activate_user("tester", "Self-Chosen-Password-2026!")
        self.assertIsNotNone(self.service.login("tester", "Self-Chosen-Password-2026!", "127.0.0.1"))

    def test_initial_admin_is_created_once(self):
        self.assertTrue(self.service.ensure_initial_admin())
        self.assertFalse(self.service.ensure_initial_admin())
        with self.repository.connect() as connection:
            rows = connection.execute("SELECT username,is_admin,activated FROM public.quant_users WHERE is_admin").fetchall()
        self.assertEqual([row["username"] for row in rows], ["admin"])
        self.assertTrue(rows[0]["activated"])
        self.assertTrue(self.service.uses_initial_password("admin"))
        self.assertIsNotNone(self.service.login("admin", INITIAL_ADMIN_PASSWORD, "127.0.0.1"))

    def test_existing_admin_account_is_promoted_instead_of_duplicated(self):
        self.service.create_user("admin", self.password)
        token = self.service.login("admin", self.password, "127.0.0.1")
        self.assertTrue(self.service.ensure_initial_admin())
        # 提权同时下发初始口令，原会话必须失效。
        self.assertIsNone(self.service.session(token))
        with self.repository.connect() as connection:
            total = connection.execute("SELECT count(*) AS total FROM public.quant_users WHERE username=%s", ("admin",)).fetchone()["total"]
        self.assertEqual(total, 1)
        self.assertIsNotNone(self.service.login("admin", INITIAL_ADMIN_PASSWORD, "127.0.0.1"))

    def test_last_loginable_admin_cannot_be_disabled_or_reset(self):
        self.service.ensure_initial_admin()
        for action in (self.service.disable_user, self.service.reset_password):
            with self.assertRaisesRegex(ValueError, "将没有可登录的管理员"):
                action("admin")

    def test_admin_cannot_disable_the_account_in_use(self):
        self.service.ensure_initial_admin()
        with self.repository.connect() as connection:
            connection.execute("UPDATE public.quant_users SET is_admin=true WHERE username=%s", ("tester",))
        with self.assertRaisesRegex(ValueError, "不能禁用当前登录的账号"):
            self.service.disable_user("tester", operator="tester")
        self.assertIsNotNone(self.service.login("tester", self.password, "127.0.0.1"))

    def test_account_list_exposes_no_password_hash(self):
        self.service.create_pending_user("newcomer")
        rows = self.service.list_users()
        self.assertEqual([row["username"] for row in rows], ["tester", "newcomer"])
        self.assertNotIn("password_hash", rows[0])
        self.assertTrue(rows[0]["activated"])
        self.assertFalse(rows[1]["activated"])

    def test_admin_page_is_registered_only_for_administrators(self):
        """账号管理页是管理员专属路由，普通账号连页面路径都不存在。"""
        account_page = str(settings.PROJECT_ROOT / "pages" / ACCOUNT_PAGE)
        self.service.create_pending_user("newcomer")
        self.service.ensure_initial_admin()
        with patch("services.auth_ui.get_auth_service", return_value=self.service), patch(
            "services.runtime.get_repository", return_value=self.repository
        ), patch.dict(os.environ, {"QUANT_DISABLE_MONITOR": "1"}):
            app = AppTest.from_file(settings.PROJECT_ROOT / "app.py", default_timeout=20)
            app.session_state["auth_token"] = self.service.login("tester", self.password, "127.0.0.1")
            app.run()
            with self.assertRaises(ValueError):
                app.switch_page(account_page)

            app.session_state["auth_token"] = self.service.login("admin", INITIAL_ADMIN_PASSWORD, "127.0.0.1")
            app.run()
            app.switch_page(account_page).run()
            self.assertFalse(app.exception)
            self.assertTrue(any("账号管理" in item.value for item in app.markdown))
            # 初始口令未换时页面必须持续告警，且待激活账号出现在清单里。
            self.assertTrue(any("初始密码" in item.value for item in app.warning))
            self.assertIn("newcomer", app.selectbox[0].options)

    def test_model_health_page_is_registered_only_for_administrators(self):
        """模型健康页同样只对管理员注册；普通账号既看不到入口，也不能直接打开路径。

        这一页会显示降级状态与回滚按钮，越权可见等于把治理结论与生产模型的操作入口
        一起交出去，因此路由与页内校验两道防线都要钉住。
        """
        self.service.create_pending_user("newcomer")
        self.service.ensure_initial_admin()
        health_page = str(settings.PROJECT_ROOT / "pages" / MODEL_HEALTH_PAGE)
        with patch("services.auth_ui.get_auth_service", return_value=self.service), patch(
            "services.runtime.get_repository", return_value=self.repository
        ), patch.dict(os.environ, {"QUANT_DISABLE_MONITOR": "1"}):
            app = AppTest.from_file(settings.PROJECT_ROOT / "app.py", default_timeout=20)
            app.session_state["auth_token"] = self.service.login("tester", self.password, "127.0.0.1")
            app.run()
            # 导航项不在 AppTest 的可读元素里，但「未注册的路由打不开」是同一件事的
            # 硬证据：st.navigation 没登记这条路径，switch_page 就找不到它。
            with self.assertRaises(ValueError):
                app.switch_page(health_page)

            app.session_state["auth_token"] = self.service.login("admin", INITIAL_ADMIN_PASSWORD, "127.0.0.1")
            app.run()
            app.switch_page(health_page).run()
            self.assertFalse(app.exception)
            self.assertTrue(any("模型健康" in item.value for item in app.markdown))
            self.assertEqual([tab.label for tab in app.tabs],
                             ["通过率", "漂移看板", "最终留出集", "模型版本与回滚", "告警"])


if __name__ == "__main__":
    unittest.main()
