"""管理员账号管理页面：创建账号、重置密码、禁用与恢复。"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from services.auth import INITIAL_ADMIN_PASSWORD, AuthService
from services.auth_ui import require_session
from services.database import DatabaseError
from services.runtime import get_repository
from services.ui_helpers import show_page_header

SHANGHAI = ZoneInfo("Asia/Shanghai")

show_page_header(
    "账号管理",
    "创建账号、重置密码与禁用账号；密码始终由使用者本人设置，管理员不经手。",
    "研究工作台 / 账号管理",
)

account = require_session()
# 导航只对管理员注册本页，这里再校验一次，避免以后调整路由时越权。
if not account["is_admin"]:
    st.error("只有管理员可以管理账号。")
    st.stop()

service = AuthService(get_repository())


def account_status(row: dict) -> str:
    """把启用与激活两个状态收敛成一个可读结果。"""
    if not row["enabled"]:
        return "已禁用"
    return "待激活" if not row["activated"] else "正常"


try:
    if service.uses_initial_password(account["username"]):
        st.warning(
            f"当前账号仍在使用初始密码 {INITIAL_ADMIN_PASSWORD}，知道该默认口令的人都能登录并管理账号。"
            "请从侧边栏“修改密码”更换后再对外提供服务。",
            title="仍在使用初始密码",
        )
    users = service.list_users()
except DatabaseError as exc:
    st.error(str(exc))
    st.stop()

st.subheader("创建账号")
st.caption(
    "只需填写用户名。创建后账号处于待激活状态、暂时无法登录；"
    "请把用户名告知使用者本人，由其在本系统登录页选择“首次激活”自行设置密码，管理员不会知道该密码。"
)
with st.form("create_account", clear_on_submit=True):
    new_username = st.text_input("用户名", max_chars=32, placeholder="3至32位字母、数字、点、短横线或下划线")
    created = st.form_submit_button("创建账号", type="primary")
if created:
    try:
        service.create_pending_user(new_username)
        st.success(f"账号 {new_username.strip().lower()} 已创建，等待其本人设置密码。")
        st.rerun()
    except (ValueError, DatabaseError) as exc:
        st.error(str(exc))

st.divider()
st.subheader(f"账号清单（{len(users)} 个）")
st.dataframe(
    pd.DataFrame([
        {
            "用户名": row["username"],
            "状态": account_status(row),
            "角色": "管理员" if row["is_admin"] else "普通账号",
            "创建时间": row["created_at"].astimezone(SHANGHAI).strftime("%Y-%m-%d %H:%M"),
            "最近登录": row["last_login_at"].astimezone(SHANGHAI).strftime("%Y-%m-%d %H:%M")
            if row["last_login_at"] else "从未登录",
        }
        for row in users
    ]),
    hide_index=True,
    width="stretch",
)

st.divider()
st.subheader("账号操作")
st.caption(
    "重置会把账号退回待激活并立即撤销其全部登录会话，新密码由使用者本人重新设置；"
    "禁用后账号无法登录，可随时恢复。"
)
selected = st.selectbox("选择账号", [row["username"] for row in users])
current = next(row for row in users if row["username"] == selected)
action_columns = st.columns(3)
try:
    if action_columns[0].button("重置密码", width="stretch"):
        service.reset_password(selected)
        st.success(f"{selected} 已退回待激活，请通知本人重新设置密码。")
        st.rerun()
    if current["enabled"]:
        if action_columns[1].button("禁用账号", width="stretch"):
            service.disable_user(selected, operator=account["username"])
            st.success(f"{selected} 已禁用，其登录会话已撤销。")
            st.rerun()
    elif action_columns[1].button("恢复账号", width="stretch"):
        service.enable_user(selected)
        st.success(f"{selected} 已恢复。")
        st.rerun()
except (ValueError, DatabaseError) as exc:
    st.error(str(exc))
