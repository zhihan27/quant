"""在业务路由及自动刷新片段前统一校验登录状态。"""

from __future__ import annotations

import ipaddress
import os
from typing import Any

import streamlit as st


@st.cache_resource
def get_auth_service():
    """首次提交登录时初始化认证服务，登录首屏不连接数据库。"""
    from services.auth import AuthService
    from services.runtime import get_repository

    return AuthService(get_repository())


def client_address() -> str:
    """仅在受信任的本机反向代理模式下读取被覆盖的真实地址。"""
    if os.environ.get("QUANT_TRUST_PROXY") == "1":
        value = st.context.headers.get("X-Real-IP", "")
        try:
            return str(ipaddress.ip_address(value))
        except ValueError:
            return "proxy-missing"
    # 直连模式采用共享来源桶，不将可伪造的转发头用作安全依据。
    return "direct"


def clear_login() -> None:
    """清理全部页面数据，防止同一浏览器换账号后残留旧结果。"""
    for key in list(st.session_state):
        del st.session_state[key]


def require_session() -> dict[str, Any]:
    """每次业务执行或片段自动刷新时检查可撤销的数据库会话。"""
    from services.database import DatabaseError

    token = st.session_state.get("auth_token", "")
    try:
        account = get_auth_service().session(token) if token else None
    except DatabaseError:
        st.error("认证服务暂不可用，请稍后重试。")
        st.stop()
    if not account or account["must_change_password"]:
        clear_login()
        st.rerun(scope="app")
    return account


def require_login() -> dict[str, Any]:
    """未登录只显示登录表单，临时账号修改密码后才能访问业务。"""
    from services.auth import AuthenticationError
    from services.database import DatabaseError

    token = st.session_state.get("auth_token", "")
    try:
        account = get_auth_service().session(token) if token else None
    except DatabaseError:
        st.error("认证服务暂不可用，请检查服务器数据库配置。")
        st.stop()
    if token and not account:
        clear_login()
        st.warning("登录已过期或已被撤销，请重新登录。")
    if not account:
        st.markdown(
            """
            <style>
            .block-container { max-width: 1120px; padding-top: 6vh; padding-bottom: 5vh; }
            [data-testid="stSidebar"] { display: none; }
            main h1 { margin: 8px 0; color: #172b4d; font-size: 30px !important; }
            .quant-login-hero { min-height: 260px; box-sizing: border-box;
                display: flex; flex-direction: column; justify-content: space-between;
                gap: 24px; padding: 32px; border-radius: 16px; background: #1d2d3d; color: #ffffff; }
            .quant-login-brand { display: flex; align-items: center; gap: 12px; }
            .quant-login-mark { display: grid; place-items: center; width: 42px; height: 42px;
                border-radius: 10px; background: #087f8c; color: #ffffff; font-size: 21px; font-weight: 700; }
            .quant-login-brand strong { font-size: 17px; }
            .quant-login-brand small { display: block; margin-top: 3px; color: #b9c8d4; }
            .quant-login-kicker { margin: 16px 0 8px; color: #6ee7d7; font-size: 11px;
                font-weight: 700; letter-spacing: 1.5px; }
            .quant-login-hero h1 { margin: 0; color: #ffffff; font-size: 28px !important; line-height: 1.35; }
            .quant-login-hero p { max-width: 380px; margin: 8px 0 0; color: #c3d0da; line-height: 1.65; }
            .quant-login-heading { margin: 6px 0 4px; }
            .quant-login-heading small { color: #087f8c; font-size: 11px; font-weight: 700; letter-spacing: 1.2px; }
            @media (max-width: 768px) {
                .block-container { padding: 1rem .75rem 2rem; }
                .quant-login-hero { min-height: 220px; padding: 22px; }
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        left, right = st.columns([1.05, 1], gap="large", vertical_alignment="center")
        with left:
            st.markdown(
                '<section class="quant-login-hero">'
                '<div class="quant-login-brand"><span class="quant-login-mark">研</span>'
                '<div><strong>量化研究</strong><small>行情 · 策略 · 持仓 · 监控</small></div></div>'
                '<div><div class="quant-login-kicker">A SHARE RESEARCH WORKSPACE</div>'
                '<h1>量化研究工作台</h1>'
                '<p>行情、选股、策略与持仓，在统一工作台中完成。</p></div></section>',
                unsafe_allow_html=True,
            )
        with right:
            st.markdown(
                '<div class="quant-login-heading"><small>SECURE SIGN IN</small></div>',
                unsafe_allow_html=True,
            )
            st.title("量化研究登录")
            st.caption("使用管理员分配的账号继续。")
            if st.session_state.pop("auth_notice", None):
                st.success("密码已修改，请使用新密码登录。")
            with st.form("account_login", clear_on_submit=True):
                username = st.text_input("用户名", max_chars=32, autocomplete="username", placeholder="输入授权用户名")
                password = st.text_input(
                    "密码", type="password", max_chars=128, autocomplete="current-password",
                    placeholder="至少 6 位",
                )
                submitted = st.form_submit_button("登录", type="primary", use_container_width=True)
            st.caption("新密码支持 6 至 128 位；常见弱密码会被拒绝。")
            if submitted:
                try:
                    token = get_auth_service().login(username, password, client_address())
                    clear_login()
                    st.session_state["auth_token"] = token
                    st.rerun()
                except AuthenticationError as exc:
                    st.error(str(exc))
                except DatabaseError:
                    st.error("认证服务暂不可用，请联系管理员。")
        st.stop()
    render_account_sidebar(account)
    if account["must_change_password"] or st.session_state.get("auth_change_password"):
        st.title("修改登录密码")
        if account["must_change_password"]:
            st.warning("当前使用临时密码，请修改密码后进入工作台。")
        with st.form("change_password", clear_on_submit=True):
            old = st.text_input("当前密码", type="password", max_chars=128)
            new = st.text_input("新密码（至少6位）", type="password", max_chars=128)
            repeated = st.text_input("确认新密码", type="password", max_chars=128)
            submitted = st.form_submit_button("保存新密码", type="primary")
        if submitted:
            try:
                if new != repeated:
                    raise ValueError("两次输入的新密码不一致。")
                get_auth_service().change_password(token, old, new)
                clear_login()
                st.session_state["auth_notice"] = True
                st.rerun()
            except (AuthenticationError, ValueError) as exc:
                st.error(str(exc))
            except DatabaseError:
                st.error("密码修改失败，请稍后重试。")
        st.stop()
    return account


def render_account_sidebar(account: dict[str, Any]) -> None:
    """显示当前账号、密码修改和服务端退出操作。"""
    from services.database import DatabaseError

    st.sidebar.caption(f"当前账号：{account['username']}")
    if not account["must_change_password"] and st.sidebar.button("修改密码", use_container_width=True):
        st.session_state["auth_change_password"] = not st.session_state.get("auth_change_password", False)
        st.rerun()
    if st.sidebar.button("退出登录", use_container_width=True):
        try:
            get_auth_service().logout(st.session_state.get("auth_token", ""))
        except DatabaseError:
            st.error("退出失败，请稍后重试。")
            st.stop()
        clear_login()
        st.rerun()
