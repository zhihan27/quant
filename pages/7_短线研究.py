"""短线研究、账户任务、成交回填与批次风险监控页面。"""

from __future__ import annotations

import subprocess
import sys
import uuid
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from config import settings
from config.short_term import SHORT_TERM_MAIN_HORIZON, SHORT_TERM_STRATEGY_ID
from services.auth_ui import require_session
from services.database import DatabaseError
from services.runtime import get_monitor, get_repository
from services.short_term.risk import evaluate_position_risks
from services.short_term.store import ShortTermStore
from services.ui_helpers import show_page_header

SHANGHAI = ZoneInfo("Asia/Shanghai")

st.set_page_config(page_title="短线研究 | A 股量化研究", layout="wide")
login_account = require_session()
repository = get_repository()
short_store = ShortTermStore(repository)
show_page_header(
    "短线研究",
    "独立 H=1/3/5 日频研究、人工买卖任务、真实成交回填与批次风控。",
    "策略研究 / 短线研究",
)

try:
    get_monitor()
    monitor_config = repository.monitor_config()
    runs = short_store.recent_runs()
    accounts = short_store.accounts(int(login_account["id"]))
except DatabaseError as exc:
    st.error(f"短线服务暂不可用：{exc}")
    st.stop()

active_runs = [row for row in runs if row["status"] in ("queued", "running")]
if login_account["is_admin"]:
    run_columns = st.columns([1, 3], vertical_alignment="bottom")
    selected_horizon = run_columns[0].selectbox("研究周期 H", (3, 1, 5), index=0)
    run_disabled = bool(active_runs)
    if run_columns[1].button("启动日频研究任务", type="primary", disabled=run_disabled):
        run_id = None
        try:
            run_id = short_store.queue_run(int(login_account["id"]), int(selected_horizon))
            log_path = settings.LOG_DIR / f"short_term_{run_id}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(settings.PROJECT_ROOT / "scan_short_term.py"),
                "--run-id",
                run_id,
                "--horizon",
                str(int(selected_horizon)),
            ]
            with log_path.open("a", encoding="utf-8") as log_file:
                subprocess.Popen(
                    command,
                    cwd=str(settings.PROJECT_ROOT),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            st.success(f"研究任务已排队：{run_id}。完成状态会写入运行台账。")
            st.rerun()
        except (DatabaseError, OSError, ValueError) as exc:
            if run_id:
                short_store.fail_run(run_id, f"{type(exc).__name__}: {exc}")
            st.error(f"无法启动研究任务：{exc}")
elif active_runs:
    st.info("管理员提交的短线研究任务正在后台运行。")

try:
    monitor_message = monitor_config.get("last_error") or "无"
    last_cycle = monitor_config.get("last_cycle_at")
    last_cycle_text = last_cycle.astimezone(SHANGHAI).strftime("%Y-%m-%d %H:%M:%S") if last_cycle else "尚无心跳"
    poll_cols = st.columns(3)
    poll_cols[0].metric("持仓监控", "运行中" if monitor_config["enabled"] else "已关闭")
    poll_cols[1].metric("最近心跳", last_cycle_text)
    poll_cols[2].metric("本轮耗时", f"{float(monitor_config['cycle_seconds']):.1f} 秒" if monitor_config.get("cycle_seconds") is not None else "未记录")
    if monitor_message != "无":
        st.warning(f"监控状态：{monitor_message}")
except (KeyError, TypeError, ValueError):
    st.warning("尚未取得有效的监控心跳。")

if active_runs:
    st.info("后台研究计算正在进行；页面刷新不会重复启动该任务。")

run_ids = [str(row["run_id"]) for row in runs]
selected_run = None
selected_predictions = []
if run_ids:
    run_by_id = {str(row["run_id"]): row for row in runs}
    selected_run_id = st.selectbox(
        "查看研究运行",
        run_ids,
        format_func=lambda value: (
            f"H={run_by_id[value]['horizon']} · {run_by_id[value]['signal_date'] or '未完成'} · "
            f"{run_by_id[value]['status']} · {value[:8]}"
        ),
        key="short_term_run_id",
    )
    selected_run = run_by_id[selected_run_id]
    if selected_run["status"] == "research_only":
        selected_predictions = short_store.predictions(selected_run_id, limit=1000)
    validation = selected_run.get("validation") or {}
    result = selected_run.get("result") or {}
    probability_validation = result.get("probability_validation") or {}
    execution_replay = result.get("execution_replay") or {}
    point_in_time_snapshot = result.get("point_in_time_snapshot") or {}
    cost_stress = result.get("cost_stress") or {}
    forward_observation = result.get("forward_observation") or {}
    status_labels = {
        "queued": "排队中",
        "running": "运行中",
        "research_only": "研究态，未获生产验收",
        "insufficient_data": "数据不足",
        "failed": "运行失败",
    }
    st.subheader(status_labels.get(selected_run["status"], selected_run["status"]))
    promotion_gate = result.get("promotion_gate") or {}
    if promotion_gate:
        blockers = promotion_gate.get("blockers", [])
        st.error("生产晋级已阻断：" + ("；".join(blockers) if blockers else promotion_gate.get("status", "未评估")))
    if result.get("acceptance_policy_error"):
        st.caption(f"验收政策：{result['acceptance_policy_error']}")
    if selected_run["status"] == "failed":
        st.error(selected_run.get("error") or "后台任务失败，查看 logs/short_term_<run_id>.log 获取详情。")
    if selected_run["status"] == "insufficient_data":
        st.info(result.get("message") or "当前历史数据不足以完成滚动验证。")
    if validation:
        metric_cols = st.columns(4)
        metric_cols[0].metric("样本外折数", validation.get("fold_count", 0))
        metric_cols[1].metric("模型 Top-K 净收益代理", f"{validation['top_k_mean_net_return']:.2%}" if validation.get("top_k_mean_net_return") is not None else "未计算")
        metric_cols[2].metric("5 日动量基线", f"{validation['momentum_top_k_mean_net_return']:.2%}" if validation.get("momentum_top_k_mean_net_return") is not None else "未计算")
        metric_cols[3].metric("日均 Rank IC", f"{validation['mean_daily_rank_ic']:.3f}" if validation.get("mean_daily_rank_ic") is not None else "未计算")
        st.caption("收益使用前复权开盘价和项目比例成本代理，不是含真实撮合、最低佣金和历史股票池的可交易业绩。")
        if validation.get("folds"):
            fold_rows = []
            for fold in validation["folds"]:
                fold_rows.append(
                    {
                        "测试开始": fold["test_start"],
                        "测试结束": fold["test_end"],
                        "训练截面数": fold["train_days"],
                        "测试股票行": fold["test_rows"],
                        "模型 Top-K": fold["top_k_mean_net_return"],
                        "动量 Top-K": fold["momentum_top_k_mean_net_return"],
                        "全池等权": fold["universe_mean_net_return"],
                        "Rank IC": fold["mean_daily_rank_ic"],
                    }
                )
            st.dataframe(pd.DataFrame(fold_rows), hide_index=True, width="stretch")
    if probability_validation:
        st.subheader("时间隔离概率诊断")
        st.caption("当前依据前复权价格代理标签，不是已成交交易胜率；概率模块尚未生产验收。")
        if probability_validation.get("status") == "proxy_only_not_trade_probability":
            probability_cols = st.columns(3)
            probability_cols[0].metric("校准外层折数", probability_validation.get("fold_count", 0))
            probability_cols[1].metric("平均 Brier", f"{probability_validation['mean_brier_score']:.4f}")
            probability_cols[2].metric("平均 Log loss", f"{probability_validation['mean_log_loss']:.4f}")
            if probability_validation.get("reliability"):
                st.dataframe(pd.DataFrame(probability_validation["reliability"]), hide_index=True, width="stretch")
        else:
            st.info(probability_validation.get("reason") or probability_validation.get("status", "概率诊断尚未生成"))
    if execution_replay:
        st.subheader("历史撮合回放")
        st.caption("回放按原始日线、限价触价、成交量参与率和冻结费率估算；日线不能还原盘口队列，也未模拟账户级并发资金约束，结果不作为生产晋级证据。")
        if str(execution_replay.get("status", "")).startswith("blocked_"):
            st.warning("回放阻断：" + "；".join(execution_replay.get("blockers", [])))
        else:
            replay_cols = st.columns(6)
            replay_cols[0].metric("回放状态", execution_replay.get("status", "未评估"))
            replay_cols[1].metric("信号交易数", execution_replay.get("trade_count", 0))
            replay_cols[2].metric("入场 / 退出成交", f"{execution_replay.get('entry_fill_count', 0)} / {execution_replay.get('exit_fill_count', 0)}")
            replay_cols[3].metric("部分入场", execution_replay.get("partial_entry_trade_count", 0))
            replay_return = execution_replay.get("mean_closed_net_return")
            replay_cols[4].metric("平均已平仓净收益", f"{replay_return:.2%}" if replay_return is not None else "未计算")
            replay_cols[5].metric("已完全退出", execution_replay.get("closed_trade_count", 0))
            if execution_replay.get("coverage_blockers"):
                st.warning("点时覆盖缺口：" + "；".join(execution_replay["coverage_blockers"][:8]))
            if execution_replay.get("trade_log_path"):
                st.caption(f"逐笔回放记录：{execution_replay['trade_log_path']}；SHA-256：{execution_replay.get('trade_log_hash')}")
        portfolio_risk = execution_replay.get("portfolio_risk") or {}
        if portfolio_risk:
            st.caption("未平仓按最新原始收盘价估值；因缺少初始净值和并发资金分配规则，不计算组合收益率或最大回撤。")
            risk_cols = st.columns(3)
            risk_cols[0].metric("未平仓数量", portfolio_risk.get("open_position_count", 0))
            risk_cols[1].metric(
                "未平仓市值代理",
                f"{portfolio_risk['gross_open_market_value_before_exit_cost']:,.2f}"
                if portfolio_risk.get("gross_open_market_value_before_exit_cost") is not None else "未计算",
            )
            risk_cols[2].metric(
                "未实现损益代理",
                f"{portfolio_risk['unrealized_pnl_before_exit_cost']:,.2f}"
                if portfolio_risk.get("unrealized_pnl_before_exit_cost") is not None else "未计算",
            )
            for blocker in portfolio_risk.get("blockers", []):
                st.caption(blocker)
        corporate_actions = execution_replay.get("corporate_actions") or {}
        if corporate_actions:
            st.warning("公司行为口径未完成：" + "；".join(corporate_actions.get("blockers", [])))
    if point_in_time_snapshot:
        st.subheader("点时主数据")
        st.caption(
            f"状态：{point_in_time_snapshot.get('status', '未评估')}；"
            f"交易日：{point_in_time_snapshot.get('session_date', '未找到')}；"
            f"来源可得时间：{point_in_time_snapshot.get('available_at', '未找到')}；"
            f"股票池快照行数：{point_in_time_snapshot.get('membership_rows', 0)}。"
        )
        for blocker in point_in_time_snapshot.get("blockers", []):
            st.caption(blocker)
    if cost_stress:
        st.subheader("成本与容量压力诊断")
        st.caption("压力参数来自冻结验收政策；诊断通过仍不等于生产验收。")
        if cost_stress.get("scenarios"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "场景": row.get("name"),
                            "状态": row.get("status"),
                            "已平仓样本": (row.get("metrics") or {}).get("closed_sample_count"),
                            "平均净收益": (row.get("metrics") or {}).get("mean_closed_net_return"),
                            "未成交比例": (row.get("metrics") or {}).get("unfilled_rate"),
                            "未平仓比例": (row.get("metrics") or {}).get("unclosed_rate"),
                            "阻断原因": "；".join(row.get("blockers", [])),
                        }
                        for row in cost_stress["scenarios"]
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("；".join(cost_stress.get("blockers", [])) or cost_stress.get("status", "未评估"))
    if forward_observation:
        st.subheader("前向观察")
        st.caption(
            f"状态：{forward_observation.get('status', '未评估')}；"
            f"本轮新记录：{forward_observation.get('observed_count', 0)}；"
            f"仍待成熟：{forward_observation.get('pending_count', 0)}；"
            f"口径：{forward_observation.get('label_basis', '未找到')}。"
        )
        observed_rows = short_store.forward_observations(selected_run_id, limit=1000)
        if observed_rows:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "代码": row["code"],
                            "周期": row["horizon"],
                            "信号时间": row["signal_time"],
                            "入场日": row["entry_date"],
                            "退出日": row["label_end_date"],
                            "毛收益代理": row["gross_return"],
                            "净收益代理": row["target_net_return"],
                            "观察口径": row["label_basis"],
                            "来源哈希": row["source_hash"],
                        }
                        for row in observed_rows
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        for blocker in forward_observation.get("blockers", []):
            st.caption(blocker)
    for warning in result.get("warnings") or []:
        st.warning(warning)
    if selected_predictions:
        visible = pd.DataFrame(
            [
                {
                    "排名": row["rank"],
                    "代码": row["code"],
                    "排序分数": row["ranking_score"],
                    "截面百分位": row["percentile"],
                    "信号时间": row["signal_time"],
                    "状态": row["status"],
                }
                for row in selected_predictions
            ]
        )
        st.dataframe(visible.head(50), hide_index=True, width="stretch")
        st.caption("排序分数只表示同一截面相对顺序，不是上涨概率或目标价。此版本不会自动下单。")
else:
    st.info("暂无研究记录。管理员可在上方启动 H=3 主周期或 H=1/5 对照研究。")

if not accounts:
    st.subheader("建立短线账户")
    st.caption("账户与持仓按登录用户隔离。未确认资金、费用和仓位上限时，不生成带股数的买入任务。")
    with st.form("short_account_create", clear_on_submit=True):
        account_name = st.text_input("账户名称", max_chars=80, placeholder="例如：模拟账户 A")
        cash_known = st.checkbox("录入已核对的可用资金")
        initial_cash_value = st.number_input("初始资金（元）", min_value=0.0, value=0.0, step=1000.0)
        limits_known = st.checkbox("录入账户仓位限额")
        max_total_value = st.number_input("总仓位上限（元）", min_value=0.0, value=0.0, step=1000.0)
        max_single_value = st.number_input("单股仓位上限（元）", min_value=0.0, value=0.0, step=500.0)
        costs_known = st.checkbox("录入实际账户费率")
        commission_rate = st.number_input("佣金比例", min_value=0.0, max_value=0.05, value=0.0, step=0.0001, format="%.6f")
        minimum_commission = st.number_input("单笔最低佣金（元）", min_value=0.0, value=0.0, step=1.0)
        stamp_duty_rate = st.number_input("卖出印花税比例", min_value=0.0, max_value=0.05, value=0.0, step=0.0001, format="%.6f")
        slippage_rate = st.number_input("滑点估计比例", min_value=0.0, max_value=0.05, value=0.0, step=0.0001, format="%.6f")
        create_account = st.form_submit_button("创建账户", type="primary")
    if create_account:
        initial_cash = initial_cash_value if cash_known else None
        max_total = max_total_value if limits_known and max_total_value > 0 else None
        max_single = max_single_value if limits_known and max_single_value > 0 else None
        costs = (
            {
                "commission_rate": commission_rate,
                "minimum_commission": minimum_commission,
                "stamp_duty_rate": stamp_duty_rate,
                "slippage_rate": slippage_rate,
            }
            if costs_known
            else {}
        )
        try:
            short_store.create_account(
                int(login_account["id"]), account_name, initial_cash, max_total, max_single, costs
            )
            st.success("短线账户已创建。")
            st.rerun()
        except (DatabaseError, ValueError) as exc:
            st.error(str(exc))
    st.stop()

account_labels = {}
for row in accounts:
    cash_text = "未确认" if row["cash_balance"] is None else f"{float(row['cash_balance']):,.2f} 元"
    account_labels[int(row["id"])] = f"{row['name']} · 可用资金 {cash_text}"
selected_account_id = st.selectbox(
    "执行账户",
    list(account_labels),
    format_func=lambda value: account_labels[value],
    key="short_term_account_id",
)
selected_account = next(row for row in accounts if int(row["id"]) == int(selected_account_id))

research_tab, account_tab, execution_tab, events_tab = st.tabs(["研究与候选", "账户与持仓", "任务与成交", "事件台账"])

with research_tab:
    st.subheader("最新研究候选")
    if not selected_predictions:
        st.info("当前运行没有可用候选；数据不足或模型仍处于未验收状态时不生成交易信号。")
    else:
        st.caption("以下排名属于研究结果。创建人工任务需要另行设置触发条件、限价、有效期和账户风险限制。")
        selected_code = st.selectbox(
            "选择研究股票",
            [row["code"] for row in selected_predictions[:100]],
            format_func=lambda code: f"{code} · 排名 {next(row['rank'] for row in selected_predictions if row['code'] == code)}",
            key="short_term_candidate_code",
        )
        st.page_link(
            str(settings.PROJECT_ROOT / "pages" / "1_个股分析.py"),
            label=f"打开 {selected_code} 个股分析",
            query_params={
                "code": selected_code,
                "strategy_id": SHORT_TERM_STRATEGY_ID,
                "horizon": str(selected_run["horizon"]),
                "run_id": str(selected_run["run_id"]),
            },
        )

with account_tab:
    st.subheader("策略批次持仓")
    current_cash_text = "未确认" if selected_account["cash_balance"] is None else f"{float(selected_account['cash_balance']):,.2f} 元"
    st.caption(f"当前可用资金：{current_cash_text}")
    with st.form("short_cash_reconciliation"):
        reconciled_cash = st.number_input(
            "券商确认的可用资金（元）",
            min_value=0.0,
            value=float(selected_account["cash_balance"] or 0),
            step=1000.0,
        )
        reconcile_cash_clicked = st.form_submit_button("保存资金核对")
    if reconcile_cash_clicked:
        try:
            short_store.reconcile_cash_balance(int(login_account["id"]), int(selected_account_id), reconciled_cash)
            st.success("账户可用资金已按人工核对结果更新。")
            st.rerun()
        except (DatabaseError, ValueError) as exc:
            st.error(str(exc))
    positions = short_store.positions(int(login_account["id"]), int(selected_account_id))
    position_rows = [
        {
            "批次": row["id"],
            "策略": row["strategy_id"],
            "代码": row["code"],
            "买入日": row["buy_date"],
            "买入均价": row["average_buy_price"],
            "剩余股数": row["remaining_shares"],
            "已核对可卖": row["available_shares"],
            "计划退出日": row["planned_exit_date"],
            "止损价": row["stop_loss_price"],
            "止盈价": row["take_profit_price"],
            "状态": row["status"],
        }
        for row in positions
    ]
    if position_rows:
        st.dataframe(pd.DataFrame(position_rows), hide_index=True, width="stretch")
        quote_map = repository.latest_quotes([row["code"] for row in positions])
        risk_events = evaluate_position_risks(positions, quote_map)
        try:
            short_store.save_monitor_events(int(login_account["id"]), int(selected_account_id), risk_events)
        except DatabaseError as exc:
            st.error(f"无法保存持仓风控事件：{exc}")
        fresh_count = sum(1 for row in risk_events if row["status"] != "coverage_gap")
        st.caption(f"持仓批次 {len(positions)} 个；本次风险检查取得可用新鲜行情 {fresh_count}/{len(positions)} 个。")
        if risk_events:
            st.dataframe(pd.DataFrame(risk_events), hide_index=True, width="stretch")
        st.subheader("核对可卖数量")
        with st.form("short_reconcile_available"):
            position_ids = [int(row["id"]) for row in positions]
            chosen_position_id = st.selectbox(
                "持仓批次",
                position_ids,
                format_func=lambda value: next(f"{row['code']} · 批次 {row['id']} · 剩余 {row['remaining_shares']} 股" for row in positions if int(row["id"]) == value),
            )
            chosen_position = next(row for row in positions if int(row["id"]) == int(chosen_position_id))
            available_value = st.number_input(
                "券商确认的可卖股数",
                min_value=0,
                max_value=int(chosen_position["remaining_shares"]),
                value=min(int(chosen_position["available_shares"]), int(chosen_position["remaining_shares"])),
                step=100,
            )
            reconcile_clicked = st.form_submit_button("保存可卖数核对")
        if reconcile_clicked:
            try:
                short_store.reconcile_available_shares(
                    int(login_account["id"]), int(selected_account_id), int(chosen_position_id), int(available_value)
                )
                st.success("可卖股数已按人工核对结果更新。")
                st.rerun()
            except (DatabaseError, ValueError) as exc:
                st.error(str(exc))
    else:
        st.info("该账户暂无短线持仓批次。成交回填后才会建立持仓。")
    st.caption("买入批次初始可卖数量为 0；只有核对券商可卖数后，系统才允许记录对应卖出成交。")

with execution_tab:
    intents = short_store.intents(int(login_account["id"]), int(selected_account_id))
    positions = short_store.positions(int(login_account["id"]), int(selected_account_id))
    st.subheader("人工任务")
    st.caption("任务是提醒与人工执行记录，不会连接券商或自动下单。")
    intent_side = st.radio("任务方向", ["买入", "卖出"], horizontal=True, key="short_intent_side")
    if intent_side == "卖出" and not positions:
        st.info("当前账户没有可指定的持仓批次。")
    else:
        intent_key = st.session_state.setdefault("short_intent_idempotency", str(uuid.uuid4()))
        with st.form("short_intent_form", clear_on_submit=True):
            intent_position = None
            intent_code = None
            intent_details = {}
            if intent_side == "买入":
                candidate_codes = [row["code"] for row in selected_predictions[:100]]
                if candidate_codes:
                    intent_code = st.selectbox("股票代码", candidate_codes)
                else:
                    intent_code = st.text_input("股票代码", max_chars=6)
                intent_details["price_direction"] = "lte"
                st.caption("设置限价后，实际报价不高于限价时生成提醒。")
                m3_enabled_value = st.checkbox("启用分钟级入场确认")
                if m3_enabled_value:
                    m3_min_return_pct = st.number_input("最近5分钟收益下限（%）", min_value=-100.0, max_value=100.0, value=0.0, step=0.1)
                    m3_min_volume_ratio = st.number_input("最近5分钟成交量/同时间基线下限", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
                    m3_max_vwap_deviation_pct = st.number_input("相对当日VWAP最大偏离（%）", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
                    m3_max_realized_volatility_pct = st.number_input("20分钟实现波动率上限（%）", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
                    m3_max_spread_bps = st.number_input("最大买一卖一价差（基点）", min_value=0.0, max_value=1000.0, value=0.0, step=1.0)
                    m3_max_round_trip_cost_bps = st.number_input("最大预估往返成本（基点）", min_value=0.0, max_value=2000.0, value=0.0, step=1.0)
                    st.caption("成交量、VWAP偏离、实现波动率、价差和往返成本门槛必须为正；参数在任务创建时冻结。")
                else:
                    m3_min_return_pct = None
                    m3_min_volume_ratio = None
                    m3_max_vwap_deviation_pct = None
                    m3_max_realized_volatility_pct = None
                    m3_max_spread_bps = None
                    m3_max_round_trip_cost_bps = None
            else:
                intent_position = st.selectbox(
                    "卖出批次",
                    [int(row["id"]) for row in positions],
                    format_func=lambda value: next(f"{row['code']} · 批次 {row['id']} · 可卖 {row['available_shares']} 股" for row in positions if int(row["id"]) == value),
                )
                source_position = next(row for row in positions if int(row["id"]) == int(intent_position))
                intent_code = source_position["code"]
                intent_details["position_id"] = int(intent_position)
                sell_trigger = st.selectbox("卖出限价触发条件", ["目标价达到或高于", "止损价达到或低于"])
                intent_details["price_direction"] = "gte" if sell_trigger == "目标价达到或高于" else "lte"
            trigger_rule = st.text_input("触发条件或人工决策原因", max_chars=1000)
            reference_price_value = st.number_input("参考价格（元，0 表示不记录）", min_value=0.0, value=0.0, step=0.01)
            limit_price_value = st.number_input("限价（元，0 表示不记录）", min_value=0.0, value=0.0, step=0.01)
            intended_shares_value = st.number_input("计划股数（0 表示未确定）", min_value=0, value=0, step=100)
            if intent_side == "买入":
                stop_value = st.number_input("批次止损价（元，0 表示不设置）", min_value=0.0, value=0.0, step=0.01)
                take_value = st.number_input("批次止盈价（元，0 表示不设置）", min_value=0.0, value=0.0, step=0.01)
                trailing_stop_value = st.number_input("移动止盈回撤比例（%，0 表示不设置）", min_value=0.0, max_value=90.0, value=0.0, step=0.5)
                invalidation_value = st.number_input("策略失效价（元，0 表示不设置）", min_value=0.0, value=0.0, step=0.01)
                has_exit_date = st.checkbox("设置计划退出日")
                planned_exit_date_value = st.date_input("计划退出日", value=date.today())
                planned_exit_value = planned_exit_date_value if has_exit_date else None
            else:
                stop_value = 0.0
                take_value = 0.0
                m3_enabled_value = False
                m3_min_return_pct = None
                m3_min_volume_ratio = None
                m3_max_vwap_deviation_pct = None
                m3_max_realized_volatility_pct = None
                m3_max_spread_bps = None
                m3_max_round_trip_cost_bps = None
                trailing_stop_value = 0.0
                invalidation_value = 0.0
                planned_exit_value = None
            has_expiry = st.checkbox("设置任务有效期")
            expiry_date_value = st.date_input("有效期至", value=date.today())
            expiry_value = expiry_date_value if has_expiry else None
            create_intent = st.form_submit_button("创建人工任务", type="primary")
        if create_intent:
            if intent_side == "买入":
                intent_details["m3_enabled"] = bool(m3_enabled_value)
                if m3_enabled_value:
                    intent_details["m3_min_5m_return"] = float(m3_min_return_pct) / 100.0
                    intent_details["m3_min_volume_ratio"] = float(m3_min_volume_ratio)
                    intent_details["m3_max_abs_vwap_deviation"] = float(m3_max_vwap_deviation_pct) / 100.0
                    intent_details["m3_max_realized_volatility"] = float(m3_max_realized_volatility_pct) / 100.0
                    intent_details["m3_max_spread_bps"] = float(m3_max_spread_bps)
                    intent_details["m3_max_round_trip_cost_bps"] = float(m3_max_round_trip_cost_bps)
                if stop_value > 0:
                    intent_details["stop_loss_price"] = stop_value
                if take_value > 0:
                    intent_details["take_profit_price"] = take_value
                if trailing_stop_value > 0:
                    intent_details["trailing_stop_pct"] = trailing_stop_value
                if invalidation_value > 0:
                    intent_details["strategy_invalidation_price"] = invalidation_value
                if planned_exit_value:
                    intent_details["planned_exit_date"] = planned_exit_value.isoformat()
            if intent_position is not None:
                intent_details["position_id"] = int(intent_position)
            if selected_run and selected_run["status"] == "research_only":
                intent_details["research_status"] = "research_only"
            valid_until = datetime.combine(expiry_value, time(15, 0), SHANGHAI) if expiry_value else None
            try:
                short_store.create_manual_intent(
                    int(login_account["id"]),
                    int(selected_account_id),
                    intent_code,
                    "BUY" if intent_side == "买入" else "SELL",
                    trigger_rule,
                    intent_key,
                    strategy_id=SHORT_TERM_STRATEGY_ID if intent_side == "买入" else source_position["strategy_id"],
                    run_id=str(selected_run["run_id"]) if selected_run and selected_run["status"] == "research_only" else None,
                    reference_price=reference_price_value or None,
                    limit_price=limit_price_value or None,
                    intended_shares=intended_shares_value or None,
                    valid_until=valid_until,
                    details=intent_details,
                )
                st.session_state["short_intent_idempotency"] = str(uuid.uuid4())
                st.success("人工任务已记录，尚未成交。")
                st.rerun()
            except (DatabaseError, ValueError) as exc:
                st.error(str(exc))
    if intents:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "任务": row["id"],
                        "策略": row["strategy_id"],
                        "代码": row["code"],
                        "方向": row["side"],
                        "状态": row["status"],
                        "触发规则": row["trigger_rule"],
                        "参考价": row["reference_price"],
                        "限价": row["limit_price"],
                        "计划股数": row["intended_shares"],
                        "有效期": row["valid_until"],
                    }
                    for row in intents
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        cancellable = [row for row in intents if row["status"] in ("pending", "triggered", "partially_filled")]
        if cancellable:
            cancel_id = st.selectbox("取消待处理任务", [int(row["id"]) for row in cancellable], key="short_cancel_intent")
            if st.button("取消所选任务"):
                try:
                    short_store.cancel_intent(int(login_account["id"]), int(selected_account_id), cancel_id)
                    st.success("任务已取消，历史记录已保留。")
                    st.rerun()
                except (DatabaseError, ValueError) as exc:
                    st.error(str(exc))

    st.subheader("实际成交回填")
    active_intents = [row for row in intents if row["status"] in ("pending", "triggered", "partially_filled")]
    intent_options = [0] + [int(row["id"]) for row in active_intents]
    linked_intent_id = st.selectbox(
        "关联任务（可留空）",
        intent_options,
        format_func=lambda value: "无，直接回填实际成交" if value == 0 else next(
            f"任务 {row['id']} · {row['side']} {row['code']} · {row['status']}" for row in active_intents if int(row["id"]) == value
        ),
        key="short_fill_intent",
    )
    linked_intent = next((row for row in active_intents if int(row["id"]) == int(linked_intent_id)), None)
    if linked_intent:
        fill_side = linked_intent["side"]
        fill_code = linked_intent["code"]
        fill_position_id = int((linked_intent.get("details") or {}).get("position_id")) if (linked_intent.get("details") or {}).get("position_id") else None
        st.caption(f"已关联任务：{fill_side} {fill_code}。成交数量、价格和费用必须按真实回报填写。")
    else:
        fill_side_label = st.radio("成交方向", ["买入", "卖出"], horizontal=True, key="short_fill_side")
        fill_side = "BUY" if fill_side_label == "买入" else "SELL"
        fill_position_id = None
        fill_code = None
    if fill_side == "SELL" and not fill_position_id:
        sell_positions = [row for row in positions if int(row["available_shares"]) > 0]
        if not sell_positions:
            st.info("没有已核对可卖数量的持仓批次，不能回填卖出成交。")
        else:
            fill_position_id = st.selectbox(
                "卖出持仓批次",
                [int(row["id"]) for row in sell_positions],
                format_func=lambda value: next(f"{row['code']} · 批次 {row['id']} · 可卖 {row['available_shares']} 股" for row in sell_positions if int(row["id"]) == value),
                key="short_fill_position",
            )
            fill_code = next(row["code"] for row in sell_positions if int(row["id"]) == int(fill_position_id))
    fill_key = st.session_state.setdefault("short_fill_idempotency", str(uuid.uuid4()))
    with st.form("short_fill_form", clear_on_submit=True):
        if linked_intent:
            st.text_input("股票代码", value=fill_code, disabled=True)
            fill_name = st.text_input("股票名称（可选）", max_chars=80)
        elif fill_side == "BUY":
            fill_code = st.text_input("股票代码", max_chars=6)
            fill_name = st.text_input("股票名称（可选）", max_chars=80)
        else:
            st.text_input("股票代码", value=fill_code or "", disabled=True)
            fill_name = st.text_input("股票名称（可选）", max_chars=80)
        fill_time_input = st.datetime_input("成交时间（北京时间）", value=datetime.now(SHANGHAI).replace(tzinfo=None))
        fill_price = st.number_input("实际成交价（元）", min_value=0.0, value=0.0, step=0.01)
        fill_shares = st.number_input("实际成交股数", min_value=0, value=0, step=100)
        fill_fee = st.number_input("实际成交费用（元）", min_value=0.0, value=0.0, step=0.01)
        external_trade_id = st.text_input("券商成交编号（可选）", max_chars=160)
        save_fill = st.form_submit_button("保存真实成交回报", type="primary")
    if save_fill:
        if fill_time_input.tzinfo is None:
            fill_time_input = fill_time_input.replace(tzinfo=SHANGHAI)
        try:
            short_store.record_fill(
                int(login_account["id"]),
                int(selected_account_id),
                fill_code,
                fill_side,
                fill_time_input,
                fill_price,
                fill_shares,
                fill_fee,
                fill_key,
                position_id=fill_position_id,
                intent_id=int(linked_intent["id"]) if linked_intent else None,
                external_trade_id=external_trade_id or None,
                name=fill_name,
            )
            st.session_state["short_fill_idempotency"] = str(uuid.uuid4())
            st.success("真实成交已入账；持仓批次与已确认现金已同步更新。")
            st.rerun()
        except (DatabaseError, ValueError) as exc:
            st.error(str(exc))

with events_tab:
    st.subheader("账户风险与成交事件")
    events = short_store.events(int(login_account["id"]), int(selected_account_id))
    if events:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "时间": row["created_at"],
                        "代码": row["code"],
                        "策略": row["strategy_id"],
                        "事件": row["event_type"],
                        "状态": row["event_status"],
                        "详情": row["details"],
                    }
                    for row in events
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("当前账户暂无任务、风控或成交事件。")

st.caption("首版不自动下单；账户现金只有在用户录入且真实成交回填时才更新。历史未归属持仓不会被自动分配到短线策略。")
