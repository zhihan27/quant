"""PostgreSQL 账号、可撤销会话与跨进程登录限流。"""

from __future__ import annotations

import hashlib
import re
import secrets
import threading
from typing import Any

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

from services.database import QuantRepository

USERNAME_PATTERN = re.compile(r"[a-z0-9_.-]{3,32}")
PASSWORD_HASHER = PasswordHasher(time_cost=2, memory_cost=19456, parallelism=1)
HASH_SLOTS = threading.BoundedSemaphore(2)
DUMMY_HASH = PASSWORD_HASHER.hash(secrets.token_urlsafe(32))
LOGIN_FAILURE = "用户名或密码错误。"


class AuthenticationError(RuntimeError):
    """认证失败、登录过于频繁或会话失效。"""


def normalize_username(username: str) -> str:
    """规范账号名，拒绝超长或含非法字符的输入。"""
    value = str(username).strip().lower()
    if not USERNAME_PATTERN.fullmatch(value):
        raise ValueError("用户名需为3至32位字母、数字、点、短横线或下划线。")
    return value


def validate_password(password: str) -> None:
    """验证正式密码长度与常见弱密码，不修改用户输入。"""
    # 放宽到6位，同时拒绝常见数字 PIN，避免直接使用最容易猜中的口令。
    weak_passwords = {
        "admin", "password", "password1234", "123456", "123123", "112233", "654321",
        "000000", "111111", "222222", "333333", "444444", "555555", "666666",
        "777777", "888888", "999999", "123456789012",
    }
    if not 6 <= len(password) <= 128 or password.strip().lower() in weak_passwords:
        raise ValueError("新密码需为6至128位，并避免常见弱密码。")


def token_digest(token: str) -> str:
    """仅在数据库保存随机会话令牌的 SHA-256 摘要。"""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class AuthService:
    """管理共享量化工作台的授权账号，不提供公开注册。"""

    def __init__(self, repository: QuantRepository) -> None:
        self.repository = repository

    def create_user(self, username: str, password: str, temporary: bool = False) -> None:
        """创建账号；临时密码必须在首次登录后更换。"""
        username = normalize_username(username)
        if temporary:
            if not 1 <= len(password) <= 128:
                raise ValueError("临时密码长度无效。")
        else:
            validate_password(password)
        digest = PASSWORD_HASHER.hash(password)
        with self.repository.connect() as connection:
            connection.execute(
                "INSERT INTO public.quant_users(username,password_hash,must_change_password) VALUES (%s,%s,%s)",
                (username, digest, temporary),
            )

    def set_password(self, username: str, password: str, temporary: bool = False) -> None:
        """管理命令重置密码并撤销该账号的全部会话。"""
        username = normalize_username(username)
        if not temporary:
            validate_password(password)
        elif not 1 <= len(password) <= 128:
            raise ValueError("临时密码长度无效。")
        digest = PASSWORD_HASHER.hash(password)
        with self.repository.connect() as connection:
            row = connection.execute(
                "UPDATE public.quant_users SET password_hash=%s,must_change_password=%s WHERE username=%s RETURNING id",
                (digest, temporary, username),
            ).fetchone()
            if not row:
                raise ValueError("账号不存在。")
            connection.execute("DELETE FROM public.quant_auth_sessions WHERE user_id=%s", (row["id"],))
            connection.execute("DELETE FROM public.quant_login_limits WHERE bucket=%s", ("user:" + token_digest(username),))

    def disable_user(self, username: str) -> None:
        """禁用账号并立即撤销所有登录会话。"""
        with self.repository.connect() as connection:
            row = connection.execute(
                "UPDATE public.quant_users SET enabled=false WHERE username=%s RETURNING id",
                (normalize_username(username),),
            ).fetchone()
            if not row:
                raise ValueError("账号不存在。")
            connection.execute("DELETE FROM public.quant_auth_sessions WHERE user_id=%s", (row["id"],))

    def _reserve_login(self, username: str, address: str) -> None:
        # 先计数再计算密码摘要，数据库原子更新避免并发绕过和重启清零。
        limits = [("global", 120, 60), ("ip:" + token_digest(address), 30, 900),
                  ("user:" + token_digest(username), 5, 900)]
        blocked = False
        with self.repository.connect() as connection:
            connection.execute("DELETE FROM public.quant_login_limits WHERE window_started < now()-interval '1 day'")
            for bucket, maximum, seconds in limits:
                row = connection.execute(
                    "INSERT INTO public.quant_login_limits(bucket,attempts,window_started) VALUES (%s,1,now()) "
                    "ON CONFLICT(bucket) DO UPDATE SET "
                    "attempts=CASE WHEN quant_login_limits.window_started <= now()-(%s * interval '1 second') "
                    "THEN 1 ELSE LEAST(quant_login_limits.attempts+1,1000000) END, "
                    "window_started=CASE WHEN quant_login_limits.window_started <= now()-(%s * interval '1 second') "
                    "THEN now() ELSE quant_login_limits.window_started END RETURNING attempts",
                    (bucket, seconds, seconds),
                ).fetchone()
                if row["attempts"] > maximum:
                    blocked = True
                    break
        # 必须在事务提交后抛出，保证已预留的计数不会回滚。
        if blocked:
            raise AuthenticationError("登录尝试过于频繁，请稍后重试；账号限制窗口为15分钟。")

    def login(self, username: str, password: str, address: str) -> str:
        """限流后校验密码，返回仅保存在当前浏览器会话的随机令牌。"""
        try:
            username = normalize_username(username)
        except ValueError as exc:
            raise AuthenticationError(LOGIN_FAILURE) from exc
        self._reserve_login(username, address)
        if not 1 <= len(password) <= 128:
            raise AuthenticationError(LOGIN_FAILURE)
        if not HASH_SLOTS.acquire(blocking=False):
            raise AuthenticationError("登录服务繁忙，请稍后重试。")
        try:
            with self.repository.connect() as connection:
                row = connection.execute(
                    "SELECT id,password_hash,enabled FROM public.quant_users WHERE username=%s", (username,),
                ).fetchone()
            try:
                # 未知账号也执行相同的密码校验，减少账号枚举的时间差。
                PASSWORD_HASHER.verify(row["password_hash"] if row else DUMMY_HASH, password)
                valid = bool(row and row["enabled"])
            except (VerificationError, InvalidHashError):
                valid = False
            if not valid:
                raise AuthenticationError(LOGIN_FAILURE)
            token = secrets.token_urlsafe(32)
            with self.repository.connect() as connection:
                # 重新锁定账号，避免禁用或密码重置与登录并发时创建新会话。
                current = connection.execute(
                    "SELECT password_hash,enabled FROM public.quant_users WHERE id=%s FOR UPDATE", (row["id"],),
                ).fetchone()
                if not current or not current["enabled"] or current["password_hash"] != row["password_hash"]:
                    raise AuthenticationError(LOGIN_FAILURE)
                connection.execute("DELETE FROM public.quant_auth_sessions WHERE expires_at <= now() OR last_seen_at <= now()-interval '30 minutes'")
                # 每个账号最多保留5个会话，避免持续登录堆积令牌。
                connection.execute(
                    "DELETE FROM public.quant_auth_sessions WHERE token_hash IN (SELECT token_hash FROM public.quant_auth_sessions "
                    "WHERE user_id=%s ORDER BY created_at DESC OFFSET 4)", (row["id"],),
                )
                connection.execute(
                    "INSERT INTO public.quant_auth_sessions(token_hash,user_id,expires_at) VALUES (%s,%s,now()+interval '12 hours')",
                    (token_digest(token), row["id"]),
                )
                connection.execute("UPDATE public.quant_users SET last_login_at=now() WHERE id=%s", (row["id"],))
                connection.execute("DELETE FROM public.quant_login_limits WHERE bucket=%s", ("user:" + token_digest(username),))
            return token
        finally:
            HASH_SLOTS.release()

    def session(self, token: str) -> dict[str, Any] | None:
        """验证账号状态、12小时绝对期限和30分钟空闲期限。"""
        if not isinstance(token, str) or not 20 <= len(token) <= 100:
            return None
        with self.repository.connect() as connection:
            return connection.execute(
                "UPDATE public.quant_auth_sessions AS session SET last_seen_at=now() FROM public.quant_users AS account "
                "WHERE session.token_hash=%s AND session.user_id=account.id AND account.enabled "
                "AND session.expires_at>now() AND session.last_seen_at>now()-interval '30 minutes' "
                "RETURNING account.id,account.username,account.must_change_password", (token_digest(token),),
            ).fetchone()

    def logout(self, token: str) -> None:
        """撤销当前令牌，退出后不能继续使用原会话。"""
        with self.repository.connect() as connection:
            connection.execute("DELETE FROM public.quant_auth_sessions WHERE token_hash=%s", (token_digest(token),))

    def change_password(self, token: str, old_password: str, new_password: str) -> None:
        """用当前密码确认修改，完成后撤销该账号全部会话。"""
        validate_password(new_password)
        if new_password == old_password:
            raise ValueError("新密码不能与当前密码相同。")
        if not HASH_SLOTS.acquire(blocking=False):
            raise AuthenticationError("认证服务繁忙，请稍后重试。")
        try:
            account = self.session(token)
            if not account:
                raise AuthenticationError("登录已失效，请重新登录。")
            self._reserve_login(account["username"], "password-change")
            with self.repository.connect() as connection:
                row = connection.execute("SELECT password_hash FROM public.quant_users WHERE id=%s FOR UPDATE", (account["id"],)).fetchone()
                active = connection.execute(
                    "SELECT 1 FROM public.quant_auth_sessions WHERE token_hash=%s AND user_id=%s "
                    "AND expires_at>now() AND last_seen_at>now()-interval '30 minutes'",
                    (token_digest(token), account["id"]),
                ).fetchone()
                if not active:
                    raise AuthenticationError("登录已失效，请重新登录。")
                try:
                    PASSWORD_HASHER.verify(row["password_hash"], old_password)
                except (VerificationError, InvalidHashError) as exc:
                    raise AuthenticationError("当前密码错误。") from exc
                connection.execute(
                    "UPDATE public.quant_users SET password_hash=%s,must_change_password=false WHERE id=%s",
                    (PASSWORD_HASHER.hash(new_password), account["id"]),
                )
                connection.execute("DELETE FROM public.quant_auth_sessions WHERE user_id=%s", (account["id"],))
                connection.execute("DELETE FROM public.quant_login_limits WHERE bucket=%s", ("user:" + token_digest(account["username"]),))
        finally:
            HASH_SLOTS.release()
