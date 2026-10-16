"""验证认证边界；PostgreSQL 集成测试使用独立临时模式。"""

import os
import secrets
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

# 导航始终注册的业务页；账号管理页只对管理员注册，因此不属于公开路由。
BUSINESS_PAGES = ["1_个股分析.py", "2_量化选股.py", "3_自选与持仓.py", "4_盘中监控.py"]
ACCOUNT_PAGE = "5_账号管理.py"


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


if __name__ == "__main__":
    unittest.main()
