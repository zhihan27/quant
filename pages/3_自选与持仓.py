"""持久化自选及分批持仓录入、编辑和参考盈亏页面。"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from config import settings
from services.database import DatabaseError
from services.portfolio import summarize_positions
from services.runtime import get_repository
from services.ui_helpers import format_number, loading_task, show_empty_state, show_page_header

show_page_header(
    "自选与持仓",
    "保存关注的股票，按买入批次管理价格、股数与费用。记录自动保留，重新打开即可继续编辑。",
    "账户工作台 / 自选与持仓",
)
st.caption("持仓为手工账本，请按券商实际持仓和可卖数量维护。")
try:
    with st.spinner("加载中 · 正在读取自选与持仓…", show_time=True):
        repository = get_repository()
except (DatabaseError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

notice = st.session_state.pop("portfolio_notice", None)
if notice:
    st.success(notice)
watch_tab, position_tab, audit_tab = st.tabs(["自选股票", "我的持仓", "修改记录"], on_change="rerun")


def _saved(message: str) -> None:
    """重新加载已提交数据，避免将编辑器旧版本继续用于下一次提交。"""
    st.session_state["portfolio_notice"] = message
    st.session_state["portfolio_revision"] = st.session_state.get("portfolio_revision", 0) + 1
    st.rerun()


revision = st.session_state.get("portfolio_revision", 0)
with watch_tab:
    if watch_tab.open:
        with st.form("add_watch"):
            st.subheader("添加关注股票")
            columns = st.columns(3, vertical_alignment="center")
            code = columns[0].text_input("股票代码", max_chars=6)
            name = columns[1].text_input("股票名称（可选）", max_chars=80)
            enabled = columns[2].checkbox("加入监控", value=True)
            note = st.text_input("备注", max_chars=2000)
            if st.form_submit_button("加入自选", type="primary"):
                try:
                    repository.save_watch(code, name, note, enabled)
                    _saved("自选已保存。")
                except (ValueError, DatabaseError) as exc:
                    st.error(str(exc))
        with st.spinner("加载中 · 正在读取自选列表…", show_time=True):
            watchlist = repository.watchlist()
        if watchlist:
            frame = pd.DataFrame(watchlist)[["code", "name", "monitor_enabled", "note"]]
            frame["remove"] = False
            with st.form("edit_watch"):
                edited = st.data_editor(
                    frame, hide_index=True, use_container_width=True, disabled=["code"], key=f"watch_editor_{revision}",
                    column_config={
                        "code": "代码", "name": "名称", "monitor_enabled": "开启监控", "note": "备注",
                        "remove": st.column_config.CheckboxColumn("移出自选"),
                    },
                )
                if st.form_submit_button("保存自选修改"):
                    try:
                        repository.edit_watchlist(edited.to_dict("records"))
                        _saved("自选修改已保存。")
                    except (ValueError, DatabaseError) as exc:
                        st.error(str(exc))
            selected = st.selectbox("打开个股分析", [row["code"] for row in watchlist], key="watch_analysis")
            st.page_link(str(settings.PROJECT_ROOT / "pages" / "1_个股分析.py"), label=f"分析 {selected}", query_params={"code": selected})
        else:
            show_empty_state("建立你的自选列表", "在上方输入股票代码，也可在个股分析或选股结果中一键加入。")

with position_tab:
    if position_tab.open:
        today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
        with st.expander("录入一笔持仓", expanded=True):
            with st.form("add_position"):
                first = st.columns(3)
                position_code = first[0].text_input("持仓代码", max_chars=6)
                position_name = first[1].text_input("持仓名称（可选）", max_chars=80)
                bought = first[2].date_input("买入日期", value=today, max_value=today)
                second = st.columns(4)
                price = second[0].number_input("买入价（元/股）", min_value=0.001, value=10.0, step=0.01, format="%.3f")
                shares = second[1].number_input("持仓股数", min_value=1, value=100, step=1)
                available = second[2].number_input("确认可卖股数", min_value=0, value=100, step=1)
                fees = second[3].number_input("本笔买入费用（元）", min_value=0.0, value=0.0, step=1.0)
                position_note = st.text_input("持仓备注", max_chars=2000)
                st.caption("支持同代码多批买入。成本包含本笔费用；今日买入批次即使填写可卖数量，T+1下也按零可卖处理。")
                if st.form_submit_button("保存持仓", type="primary"):
                    try:
                        repository.add_position({
                            "code": position_code, "name": position_name, "buy_date": bought,
                            "buy_price": price, "shares": shares, "available_shares": available,
                            "fees": fees, "note": position_note,
                        })
                        _saved("持仓已保存。")
                    except (ValueError, DatabaseError) as exc:
                        st.error(str(exc))
        with st.spinner("加载中 · 正在读取持仓批次…", show_time=True):
            positions = repository.positions()
        if positions:
            st.subheader("编辑持仓批次")
            frame = pd.DataFrame(positions)[["id", "version", "code", "name", "buy_date", "buy_price", "shares", "available_shares", "fees", "note"]]
            for column in ("buy_price", "fees"):
                frame[column] = frame[column].astype(float)
            frame["remove"] = False
            with st.form("edit_positions"):
                edited = st.data_editor(
                    frame, hide_index=True, use_container_width=True, disabled=["id", "version"], key=f"position_editor_{revision}",
                    column_config={
                        "id": None, "version": None, "code": "股票代码", "name": "名称",
                        "buy_date": st.column_config.DateColumn("买入日期", max_value=today),
                        "buy_price": st.column_config.NumberColumn("买入价", min_value=0.000001, format="%.6f"),
                        "shares": st.column_config.NumberColumn("持仓股数", min_value=1, step=1),
                        "available_shares": st.column_config.NumberColumn("确认可卖股数", min_value=0, step=1),
                        "fees": st.column_config.NumberColumn("买入费用", min_value=0.0, format="%.4f"),
                        "note": "备注", "remove": st.column_config.CheckboxColumn("删除此批次"),
                    },
                )
                if st.form_submit_button("保存持仓修改"):
                    try:
                        repository.edit_positions(edited.to_dict("records"))
                        _saved("持仓修改已保存，修改记录可追溯。")
                    except (ValueError, DatabaseError) as exc:
                        st.error(str(exc))
            st.subheader("持仓汇总")
            if st.button("刷新持仓报价"):
                from services.intraday_data import IntradayDataService

                data = IntradayDataService()
                try:
                    # 只刷新报价，不参与模型训练，不占全局计算锁。
                    with loading_task("刷新持仓报价") as task:
                        task.step("正在批量获取持仓股票报价…")
                        quotes, errors = data.get_quotes(list(dict.fromkeys(row["code"] for row in positions)))
                        task.step("正在保存可用报价并更新参考盈亏…")
                        repository.save_quotes(quotes)
                        if not quotes and errors:
                            task.fail("未获取到可用报价，请查看下方数据源提示。")
                    if errors:
                        st.warning("；".join(errors))
                except (ValueError, DatabaseError) as exc:
                    st.error(str(exc))
                finally:
                    data.close()
            summaries = summarize_positions(positions, repository.latest_quotes())
            cost = sum(row["cost"] for row in summaries)
            valued = [row for row in summaries if row["market_value"] is not None]
            metrics = st.columns(3)
            metrics[0].metric("持仓总成本", format_number(cost, 2, " 元"))
            metrics[1].metric("已有报价市值", format_number(sum(row["market_value"] for row in valued), 2, " 元"))
            metrics[2].metric("已有报价参考浮盈亏", format_number(sum(row["pnl"] for row in valued), 2, " 元"))
            if len(valued) < len(summaries):
                st.warning("部分持仓尚无报价，市值及浮盈亏仅汇总已有报价部分。")
            st.dataframe(pd.DataFrame(summaries).rename(columns={
                "code": "代码", "name": "名称", "shares": "股数", "available_shares": "T+1约束后可卖",
                "cost": "总成本", "average_cost": "含费均价", "price": "参考价格", "market_value": "参考市值",
                "pnl": "参考浮盈亏", "pnl_pct": "参考盈亏%", "quote_time": "行情源时间", "data_status": "行情状态", "lots": "批次",
            }), hide_index=True, use_container_width=True)
            st.caption("浮盈亏未扣未来卖出费用。停牌、冻结、借券及券商可卖规则请按实际情况调整可卖数量。")
        else:
            show_empty_state("记录第一笔持仓", "填写股票、买入价和股数。支持同一股票分批买入，并可随时修改各批次。")

with audit_tab:
    if audit_tab.open:
        rows = repository.position_audit()
        if rows:
            st.dataframe(pd.DataFrame(rows)[["id", "position_id", "operation", "changed_at"]].rename(columns={
                "id": "记录编号", "position_id": "持仓编号", "operation": "操作", "changed_at": "修改时间",
            }), hide_index=True, use_container_width=True)
            with st.expander("查看修改前后内容"):
                st.json(rows, expanded=False)
        else:
            st.info("暂无持仓修改记录。")
