"""短线日频选股研究页面，展示独立模型的候选排名与验证依据。"""

from __future__ import annotations

import subprocess
import sys

import pandas as pd
import streamlit as st

from config import settings
from config.short_term import SHORT_TERM_STRATEGY_ID
from services.auth_ui import require_session
from services.database import DatabaseError
from services.runtime import get_repository
from services.short_term.store import ShortTermStore
from services.ui_helpers import show_page_header

st.set_page_config(page_title="短线研究 | A 股量化研究", layout="wide")
login_account = require_session()
repository = get_repository()
short_store = ShortTermStore(repository)
show_page_header(
    "短线研究",
    "独立 H=1/3/5 日频选股研究，查看候选排名与样本外验证。",
    "策略研究 / 短线研究",
)
st.caption(
    "使用本地采集的完整日线，独立训练短线模型并生成排名；无需先训练量化选股模型。"
    "H 表示次日开盘至 H 个交易间隔后开盘的研究周期。"
)

try:
    runs = short_store.recent_runs()
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
    point_in_time_snapshot = result.get("point_in_time_snapshot") or {}
    forward_observation = result.get("forward_observation") or {}
    status_labels = {
        "queued": "排队中",
        "running": "运行中",
        "research_only": "选股研究结果（仅供研究）",
        "insufficient_data": "数据不足",
        "failed": "运行失败",
    }
    st.subheader(status_labels.get(selected_run["status"], selected_run["status"]))
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
        st.caption("排序分数只表示同一截面的相对顺序，不是上涨概率或目标价；排名靠前也可能全部下跌。")
else:
    st.info("暂无研究记录。管理员可在上方启动 H=3 主周期或 H=1/5 对照研究。")

# 候选入口直接依赖本轮研究结果，无需先创建短线账户。
st.subheader("最新研究候选")
if not selected_predictions:
    st.info("当前运行没有可用候选；请等待研究完成，或检查数据不足和运行失败的原因。")
else:
    st.caption("候选仅用于选股研究，按本轮冻结的短线模型排名查看。")
    st.caption("跳转后的个股分析仍使用原有中期模型，短线研究结论以本页排名为准。")
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
