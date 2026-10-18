"""供根目录和项目目录共同使用的 Streamlit 导航入口。"""

from pathlib import Path

import streamlit as st

from config import settings
from services.auth_ui import current_account, require_login
from services.ui_helpers import apply_page_style

st.set_page_config(page_title="A股量化研究", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
apply_page_style()
project = Path(__file__).resolve().parent
if st.session_state.get("auth_token"):
    with st.sidebar:
        st.markdown(
            '<div class="quant-brand"><span class="quant-brand-mark">研</span>'
            '<div><strong>量化研究</strong><small>行情 · 策略 · 持仓 · 监控</small></div></div>',
            unsafe_allow_html=True,
        )
        st.caption("从股票研究到持仓跟踪")
# 账号管理页只对管理员注册：未注册的页面既不在导航里，也无法通过 URL 直接打开。
account = current_account()
navigation = [
    st.Page(project / "pages" / "1_个股分析.py", title="个股分析", icon=":material/query_stats:", url_path="analysis", default=True),
    st.Page(project / "pages" / "2_量化选股.py", title="量化选股", icon=":material/filter_alt:", url_path="selection"),
    st.Page(project / "pages" / "3_自选与持仓.py", title="自选与持仓", icon=":material/account_balance_wallet:", url_path="portfolio"),
    st.Page(project / "pages" / "4_盘中监控.py", title="盘中监控", icon=":material/monitoring:", url_path="monitor"),
]
if account and account["is_admin"]:
    navigation.append(
        st.Page(project / "pages" / "5_账号管理.py", title="账号管理", icon=":material/manage_accounts:", url_path="accounts")
    )
    navigation.append(
        st.Page(project / "pages" / "6_模型健康.py", title="模型健康", icon=":material/health_and_safety:", url_path="model-health")
    )
page = st.navigation(navigation, position="sidebar" if account else "hidden")
require_login()

# 模型目标升级后清除会话中的旧结果，避免显示旧版绝对收益预测。
if st.session_state.get("result_model_version") != settings.MODEL_VERSION:
    for key in ("stock_analysis", "backtest_result", "scan_result", "scan_model_context", "portfolio_result"):
        st.session_state.pop(key, None)
    st.session_state["result_model_version"] = settings.MODEL_VERSION
page.run()
