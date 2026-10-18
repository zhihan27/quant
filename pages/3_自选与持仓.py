"""持久化自选及分批持仓录入、编辑和参考盈亏页面。"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from config import settings
from services.database import DatabaseError
from services.portfolio import summarize_positions
from services.rebalancing import build_checklist
from services.scan_store import load_scan_result
from services.model_governance import current_model_status
from services.runtime import get_repository
from services.ui_helpers import format_number, loading_task, show_empty_state, show_page_header

show_page_header(
    "自选与持仓",
    "关注股票，查看持仓与参考盈亏。",
    "账户工作台 / 自选与持仓",
)
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
        with st.spinner("加载中 · 正在读取自选列表…", show_time=True):
            watchlist = repository.watchlist()
        heading, action = st.columns([4, 1], vertical_alignment="center")
        with heading:
            st.subheader("我的自选")
            monitored = sum(bool(row["monitor_enabled"]) for row in watchlist)
            st.caption(f"共 {len(watchlist)} 只 · 已开启监控 {monitored} 只")
        with action.popover("添加自选", icon=":material/add:", width="stretch"):
            with st.form("add_watch", border=False):
                code = st.text_input("股票代码", max_chars=6, placeholder="例如 600519")
                enabled = st.checkbox("加入监控", value=True)
                with st.expander("名称与备注（可选）"):
                    name = st.text_input("股票名称（可选）", max_chars=80)
                    note = st.text_input("备注", max_chars=2000)
                if st.form_submit_button("加入自选", type="primary", width="stretch"):
                    try:
                        repository.save_watch(code, name, note, enabled)
                        _saved("自选已保存。")
                    except (ValueError, DatabaseError) as exc:
                        st.error(str(exc))
        if watchlist:
            frame = pd.DataFrame(watchlist)[["code", "name", "monitor_enabled", "note"]]
            # 删除列放第一列：编辑表从左往右读，勾选框留在最后容易被当成又一处备注。
            frame.insert(0, "remove", False)
            with st.form("edit_watch", border=False):
                edited = st.data_editor(
                    frame, hide_index=True, width="stretch", disabled=["code"], key=f"watch_editor_{revision}",
                    alt="自选股票及监控设置",
                    column_config={
                        "remove": st.column_config.CheckboxColumn("删除", width="small"),
                        "code": st.column_config.TextColumn("代码", width="small"),
                        "name": "名称", "monitor_enabled": "开启监控",
                        "note": st.column_config.TextColumn("备注", width="large"),
                    },
                )
                if st.form_submit_button("保存自选修改"):
                    try:
                        repository.edit_watchlist(edited.to_dict("records"))
                        _saved("自选修改已保存。")
                    except (ValueError, DatabaseError) as exc:
                        st.error(str(exc))
            selection, link = st.columns([3, 1], vertical_alignment="bottom")
            names = {row["code"]: row["name"] for row in watchlist}
            selected = selection.selectbox("选择股票", list(names), key="watch_analysis",
                format_func=lambda code: f"{code} {names[code] or ''}", label_visibility="collapsed")
            link.page_link(str(settings.PROJECT_ROOT / "pages" / "1_个股分析.py"),
                label="个股分析", icon=":material/query_stats:", query_params={"code": selected})
        else:
            show_empty_state("建立你的自选列表", "点击右上方「添加自选」，也可从个股分析或选股结果中加入。")

with position_tab:
    if position_tab.open:
        today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
        with st.spinner("加载中 · 正在读取持仓批次…", show_time=True):
            positions = repository.positions()
        heading, add_action, quote_action = st.columns([3, 1, 1], vertical_alignment="center")
        with heading:
            st.subheader("持仓总览")
            st.caption(f"{len({row['code'] for row in positions})} 只股票 · {len(positions)} 笔买入记录")
        # 录入移入弹层，让日常查看时先看到持仓和盈亏。
        with add_action.popover("新增持仓", icon=":material/add:", width="stretch"):
            with st.form("add_position", border=False):
                first = st.columns(2, vertical_alignment="bottom")
                position_code = first[0].text_input("持仓代码", max_chars=6)
                bought = first[1].date_input("买入日期", value=today, max_value=today)
                second = st.columns(2, vertical_alignment="bottom")
                price = second[0].number_input("买入价（元/股）", min_value=0.001, value=10.0, step=0.01, format="%.3f")
                shares = second[1].number_input(
                    "持仓股数", min_value=100, value=100, step=100,
                    help="按整手录入，加减以 100 股为单位。",
                )
                with st.expander("更多（可选）"):
                    position_name = st.text_input("持仓名称（可选）", max_chars=80)
                    optional = st.columns(2)
                    available = optional[0].number_input(
                        "确认可卖股数", min_value=0, value=100, step=100,
                        help="填写券商显示的实际可卖数量，加减以 100 股为单位；今日买入的批次不论填多少都按零可卖处理。",
                    )
                    fees = optional[1].number_input("本笔买入费用（元）", min_value=0.0, value=0.0, step=1.0)
                    position_note = st.text_input("持仓备注", max_chars=2000)
                st.caption("支持同代码多批买入。成本包含本笔费用；今日买入批次按 T+1 规则零可卖。")
                if st.form_submit_button("保存持仓", type="primary", width="stretch"):
                    try:
                        repository.add_position({
                            "code": position_code, "name": position_name, "buy_date": bought,
                            "buy_price": price, "shares": shares, "available_shares": available,
                            "fees": fees, "note": position_note,
                        })
                        _saved("持仓已保存。")
                    except (ValueError, DatabaseError) as exc:
                        st.error(str(exc))
        refresh_quotes = quote_action.button("刷新报价", icon=":material/refresh:",
            width="stretch", disabled=not positions)
        if positions:
            if refresh_quotes:
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
            metrics[0].metric("持仓成本", format_number(cost, 2, " 元"), border=True)
            metrics[1].metric("参考市值", format_number(sum(row["market_value"] for row in valued), 2, " 元") if valued else "暂无报价", border=True)
            metrics[2].metric("参考浮盈亏", format_number(sum(row["pnl"] for row in valued), 2, " 元") if valued else "暂无报价", border=True)
            if len(valued) < len(summaries):
                st.warning("部分持仓尚无报价，市值及浮盈亏仅汇总已有报价部分。")
            details = st.toggle("显示详细字段", key="position_details",
                help="增加可卖股数、总成本、买入批次和行情源时间。")
            overview_columns = ["code", "name", "shares", "average_cost", "price", "market_value", "pnl", "pnl_pct", "data_status"]
            if details:
                overview_columns = ["code", "name", "shares", "available_shares", "cost", "average_cost",
                    "price", "market_value", "pnl", "pnl_pct", "lots", "quote_time", "data_status"]
            st.dataframe(pd.DataFrame(summaries), column_order=overview_columns, hide_index=True, width="stretch",
                alt="按股票汇总的持仓数量、成本及参考盈亏", column_config={
                    "code": "代码", "name": "名称", "shares": st.column_config.NumberColumn("股数", format="%d"),
                    "available_shares": st.column_config.NumberColumn("可卖股数", format="%d", help="今日买入批次按零可卖处理，请同时核对券商实际可卖数量。"),
                    "cost": st.column_config.NumberColumn("总成本", format="%.2f"),
                    "average_cost": st.column_config.NumberColumn("成本均价", format="%.3f", help="包含买入费用。"),
                    "price": st.column_config.NumberColumn("参考价", format="%.3f"),
                    "market_value": st.column_config.NumberColumn("参考市值", format="%.2f"),
                    "pnl": st.column_config.NumberColumn("浮盈亏", format="%+.2f"),
                    "pnl_pct": st.column_config.NumberColumn("盈亏比例", format="%+.2f%%"),
                    "lots": "买入批次", "quote_time": "行情源时间", "data_status": "行情状态",
                })
            st.caption("手工持仓账本 · 浮盈亏未扣未来卖出费用，可卖数量请以券商为准。")
            editor = st.expander("编辑持仓批次", icon=":material/edit:", on_change="rerun")
            if editor.open:
                with editor:
                    frame = pd.DataFrame(positions)[["id", "version", "code", "name", "buy_date", "buy_price", "shares", "available_shares", "fees", "note"]]
                    for column in ("buy_price", "fees"):
                        frame[column] = frame[column].astype(float)
                    # 勾选删除列即可移除对应的买入批次。
                    frame.insert(0, "remove", False)
                    st.caption("修改买入记录后保存；勾选「删除」可移除对应批次。")
                    with st.form("edit_positions", border=False):
                        edited = st.data_editor(
                            frame, hide_index=True, width="stretch", disabled=["id", "version"], key=f"position_editor_{revision}",
                            alt="逐笔持仓买入记录编辑",
                            column_config={
                                "remove": st.column_config.CheckboxColumn("删除", width="small"),
                                "id": None, "version": None, "code": st.column_config.TextColumn("股票代码", width="small"),
                                "name": "名称",
                                "buy_date": st.column_config.DateColumn("买入日期", max_value=today),
                                "buy_price": st.column_config.NumberColumn("买入价", min_value=0.000001, format="%.6f"),
                                # min_value 保持 1：历史上因送股、配股产生的零股仍要能改能存，步长只管加减按钮。
                                "shares": st.column_config.NumberColumn("持仓股数", min_value=1, step=100),
                                "available_shares": st.column_config.NumberColumn("确认可卖股数", min_value=0, step=100),
                                "fees": st.column_config.NumberColumn("买入费用", min_value=0.0, format="%.4f"),
                                "note": "备注",
                            },
                        )
                        if st.form_submit_button("保存持仓修改"):
                            try:
                                repository.edit_positions(edited.to_dict("records"))
                                _saved("持仓修改已保存，修改记录可追溯。")
                            except (ValueError, DatabaseError) as exc:
                                st.error(str(exc))
        else:
            show_empty_state("记录第一笔持仓", "点击右上方「新增持仓」，填写股票、买入价和股数，支持分批买入。")
        checklist_panel = st.expander("次日人工调仓清单", icon=":material/checklist:", on_change="rerun")
        if checklist_panel.open:
            with checklist_panel:
                checklist_top_n = st.number_input("保留模型前 N 名", min_value=1, max_value=500,
                    value=settings.PORTFOLIO_TOP_N, step=1, key="checklist_top_n")
                health = current_model_status(repository)
                checklist = build_checklist(positions, load_scan_result(), top_n=int(checklist_top_n),
                    scoring_paused=bool(health.get("degraded") or health.get("error")))
                st.caption(f"排名截面：{checklist['reference_date'] or '未找到'}；成本止损比例 {settings.CHECKLIST_STOP_LOSS_RATIO:.0%}。")
                for warning in checklist["warnings"]:
                    st.caption(warning)
                categories = {"sell": "调出 / 止损", "buy": "买入候选", "hold": "维持持仓", "review": "待核查"}
                default_category = next((key for key in ("sell", "buy", "review", "hold") if checklist[key]), "sell")
                category = st.segmented_control("清单分类", list(categories), default=default_category, required=True,
                    format_func=lambda key: f"{categories[key]} · {len(checklist[key])}", key="checklist_category",
                    label_visibility="collapsed")
                if checklist[category]:
                    st.dataframe(pd.DataFrame(checklist[category]), hide_index=True,
                        alt=categories[category])
                else:
                    st.caption("当前没有符合条件的标的。")

with audit_tab:
    if audit_tab.open:
        rows = repository.position_audit()
        if rows:
            st.dataframe(pd.DataFrame(rows)[["id", "position_id", "operation", "changed_at"]].rename(columns={
                "id": "记录编号", "position_id": "持仓编号", "operation": "操作", "changed_at": "修改时间",
            }), hide_index=True, width="stretch", alt="持仓修改记录")
            with st.expander("查看修改前后内容"):
                st.json(rows, expanded=False)
        else:
            st.info("暂无持仓修改记录。")
