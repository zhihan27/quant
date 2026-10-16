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
from services.auth import AuthService, AuthenticationError, token_digest, validate_password
from services.database import connection_settings


class AuthInputTests(unittest.TestCase):
    """正式密码和未登录路由不得绕过认证。"""

    def test_weak_password_is_rejected(self):
        for password in ("admin", "123456", "x" * 129):
            with self.assertRaises(ValueError):
                validate_password(password)

    def test_all_routes_require_login_without_database(self):
        with patch("services.auth_ui.get_auth_service", side_effect=AssertionError("未登录首屏不应连接数据库")):
            app = AppTest.from_file(settings.PROJECT_ROOT / "app.py", default_timeout=20).run()
            for page in sorted((settings.PROJECT_ROOT / "pages").glob("*.py")):
                app.switch_page(str(page)).run()
                self.assertFalse(app.exception)
                self.assertEqual([item.value for item in app.title], ["量化研究登录"])
                self.assertEqual([item.label for item in app.text_input], ["用户名", "密码"])


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
            for page in sorted((settings.PROJECT_ROOT / "pages").glob("*.py")):
                app.switch_page(str(page)).run()
                self.assertFalse(app.exception)
                self.assertEqual([item.value for item in app.title], ["修改登录密码"])

    def test_authenticated_pages_and_logout(self):
        token = self.service.login("tester", self.password, "127.0.0.1")
        with patch("services.auth_ui.get_auth_service", return_value=self.service), patch.dict(os.environ, {"QUANT_DISABLE_MONITOR": "1"}):
            app = AppTest.from_file(settings.PROJECT_ROOT / "app.py", default_timeout=20)
            app.session_state["auth_token"] = token
            app.run()
            for page in sorted((settings.PROJECT_ROOT / "pages").glob("*.py")):
                app.switch_page(str(page)).run()
                self.assertFalse(app.exception, page.name)
                self.assertNotIn("量化研究登录", [item.value for item in app.title])
            next(button for button in app.button if button.label == "退出登录").click().run()
            self.assertEqual([item.value for item in app.title], ["量化研究登录"])
            self.assertIsNone(self.service.session(token))


if __name__ == "__main__":
    unittest.main()
