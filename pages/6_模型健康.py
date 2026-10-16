"""管理员模型健康页：通过率趋势、漂移看板、最终留出集、模型版本与回滚。

本页是需求文模块 E 的落地。三条自我约束：

1. **数据来源与时间戳必须显示出来。** 每个区块都写明数据取自哪张表的哪个时间字段。
   一个没有时间戳的通过率图表，看不出它是今早的还是三个月前的。
2. **不能静默。** 样本不足、维度未接入、留出集没跑过，都显示成明确的说明文字，
   而不是画一条空折线或者干脆不画——页面上看不见的东西会被当成「没问题」。
3. **只读判断与写操作分开。** 判定（是否降级、能否回滚）一律由 services 层给出，
   页面只负责显示与转发点击；回滚等写操作全部经 services/model_governance.py 落审计。
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from config import settings
from services import drift_monitor, model_governance, model_monitor
from services.auth_ui import require_session
from services.database import DatabaseError
from services.runtime import get_repository
from services.ui_helpers import show_empty_state, show_page_header

SHANGHAI = ZoneInfo("Asia/Shanghai")
NO_DATA = "—"


def stamp(value) -> str:
    """统一的时间显示；空值显示为「—」而不是空白。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return NO_DATA
    if isinstance(value, str):
        return value
    return value.astimezone(SHANGHAI).strftime("%Y-%m-%d %H:%M") if getattr(value, "tzinfo", None) else str(value)


def number(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return NO_DATA
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


show_page_header(
    "模型健康",
    "模型通过率、特征与概念漂移、最终留出集结论与模型版本回滚；数据全部取自 PostgreSQL 治理表。",
    "研究工作台 / 模型健康",
)

account = require_session()
# 与账号管理页同一条防线：导航只对管理员注册本页，这里再校验一次，
# 避免以后调整路由或直接拼 URL 时越权。
if not account["is_admin"]:
    st.error("只有管理员可以查看模型健康。")
    st.stop()

repository = get_repository()
operator = account["username"]

try:
    current_status = model_governance.model_status(repository)
except DatabaseError as exc:
    st.error(str(exc))
    st.stop()

if current_status["degraded"]:
    st.error(
        f"**模型降级中，暂停评分。** 降级原因：{current_status['reason'] or '未记录'}；"
        f"置位时间 {stamp(current_status['since'])}，操作者 {current_status['updated_by'] or '未知'}。"
        "降级期间个股分析与量化选股不再输出任何模型评分，解除或回滚后自动恢复。",
        icon=":material/report:",
    )

tabs = st.tabs(["通过率", "漂移看板", "最终留出集", "模型版本与回滚", "告警"])

# ---------------------------------------------------------------------------
# 通过率
# ---------------------------------------------------------------------------
with tabs[0]:
    st.caption(
        f"数据来源：`public.quant_model_validation_log`，窗口近 {settings.PASS_RATE_WINDOW_DAYS} 天，"
        f"按 `scan_time` 倒序。门槛与训练时完全同源（`services/ml_pipeline.py::evaluation_gates`）。"
    )
    try:
        summary = model_monitor.get_pass_rate(repository)
        failures = model_monitor.get_recent_failures(repository, limit=20)
    except DatabaseError as exc:
        st.error(str(exc))
        summary, failures = None, []

    if summary is None:
        pass
    elif not summary["records"]:
        show_empty_state(
            "还没有检验记录",
            "通过率数据在每轮滚动重训后写入。请先在“量化选股”里完成一次全量扫描，或等待每日定时任务。",
        )
    else:
        columns = st.columns(4)
        columns[0].metric("窗口内记录", f"{summary['records']} 条")
        columns[1].metric(
            "总通过率",
            NO_DATA if summary["pass_rate"] is None else f"{summary['pass_rate']:.1%}",
            help=f"低于 {settings.PASS_RATE_MINIMUM:.0%} 触发告警（需窗口内至少 {summary['minimum_records']} 条记录）",
        )
        columns[2].metric("通过 / 未通过", f"{summary['passed']} / {summary['failed']}")
        columns[3].metric(
            "最近一次失败原因",
            summary["recent_failure"]["reason"] if summary["recent_failure"] else "窗口内无失败",
        )
        if not summary["sample_sufficient"]:
            st.warning(summary["sample_note"], icon=":material/info:")

        left, right = st.columns(2)
        with left:
            st.markdown("**分模型类型**")
            st.dataframe(
                pd.DataFrame([
                    {"模型": model_monitor.MODEL_TYPE_CODES_REVERSE.get(key, key),
                     "记录": row["records"], "通过": row["passed"],
                     "通过率": NO_DATA if row["pass_rate"] is None else f"{row['pass_rate']:.1%}"}
                    for key, row in summary["by_model_type"].items()
                ]),
                hide_index=True, width="stretch",
            )
            st.markdown("**分训练目标**")
            st.dataframe(
                pd.DataFrame([
                    {"目标": key, "记录": row["records"], "通过": row["passed"],
                     "通过率": NO_DATA if row["pass_rate"] is None else f"{row['pass_rate']:.1%}"}
                    for key, row in summary["by_target_type"].items()
                ]),
                hide_index=True, width="stretch",
            )
        with right:
            st.markdown("**各道门槛的失败分布**")
            st.caption(
                "分母是「被检验过该门槛的记录数」：排序目标判 NDCG、回归目标判 MAE，"
                "门槛集合不同，混用分母会把对方的通过算成自己的失败。"
            )
            st.dataframe(
                pd.DataFrame([
                    {"门槛": row["label"], "被检验次数": row["evaluated"], "未通过": row["failures"],
                     "失败率": NO_DATA if row["failure_rate"] is None else f"{row['failure_rate']:.1%}"}
                    for row in summary["by_gate"].values()
                ]),
                hide_index=True, width="stretch",
            )

        try:
            trend_rows = model_monitor.fetch_validation_rows(repository)
        except DatabaseError:
            trend_rows = []
        if trend_rows:
            trend = pd.DataFrame([
                {"时间": stamp(row["scan_time"]),
                 "通过率": 1.0 if row["pass"] else 0.0,
                 "模型": model_monitor.MODEL_TYPE_CODES_REVERSE.get(row["model_type"], row["model_type"])}
                for row in sorted(trend_rows, key=lambda item: item["scan_time"])
            ])
            # 单次记录只有 0/1，直接画点看不出趋势，先按扫描时间聚合成每次扫描的通过率。
            grouped = trend.groupby("时间", sort=False)["通过率"].mean().reset_index()
            st.markdown("**通过率趋势（每次扫描的平均通过率）**")
            st.line_chart(grouped, x="时间", y="通过率", height=240)

        st.markdown("**最近 20 条未通过记录**")
        if not failures:
            st.caption("窗口内暂无未通过记录。")
        else:
            st.dataframe(
                pd.DataFrame([
                    {"扫描时间": stamp(row["scan_time"]), "模型": row["model_label"],
                     "目标": row["target_type"], "卡在": row["failed_gate_label"],
                     "Rank IC": number(row["rank_ic"]), "HAC t": number(row["hac_t"], 2),
                     "NDCG比": number(row["ndcg_ratio"], 2)}
                    for row in failures
                ]),
                hide_index=True, width="stretch",
            )

# ---------------------------------------------------------------------------
# 漂移看板
# ---------------------------------------------------------------------------
with tabs[1]:
    st.caption(
        "数据来源：`public.quant_feature_drift` / `quant_concept_drift` / `quant_prediction_drift`，"
        "随每日扫描同批写入。特征漂移阈值：PSI > "
        f"{settings.DRIFT_PSI_WATCH} 关注、> {settings.DRIFT_PSI_SEVERE} 严重；"
        f"概念漂移：滚动 Rank IC 连续 {settings.DRIFT_IC_NEGATIVE_DAYS} 个交易日为负；"
        f"预测漂移：Top 组合行业 HHI 超过历史 {settings.DRIFT_HHI_PERCENTILE:.0%} 分位。"
    )
    try:
        feature_rows = drift_monitor.recent_feature_drift(repository)
        concept_rows = drift_monitor.recent_concept_drift(repository)
        prediction_rows = drift_monitor.recent_prediction_drift(repository)
    except DatabaseError as exc:
        st.error(str(exc))
        feature_rows = concept_rows = prediction_rows = []

    st.subheader("特征漂移")
    if not feature_rows:
        show_empty_state("还没有特征漂移记录", "漂移在每日扫描完成后写入；请先完成一次扫描。")
    else:
        latest_date = max(row["date"] for row in feature_rows)
        st.caption(f"最近一次计算日期：**{latest_date}**（共 {len(feature_rows)} 条历史记录）")
        current = [row for row in feature_rows if row["date"] == latest_date]
        st.dataframe(
            pd.DataFrame([
                {"维度": row["feature_name"], "判定": row["threshold_exceeded"],
                 "PSI": number(row["psi"]), "KS 统计量": number(row["ks_stat"]),
                 "缺失率": NO_DATA if row["missing_rate"] is None else f"{row['missing_rate']:.1%}",
                 "均值偏移(标准化)": number(row["mean_shift"], 2),
                 "样本数": row["sample_size"] if row["sample_size"] is not None else NO_DATA,
                 "当前窗口": f"{row['window_start']} ~ {row['window_end']}"
                 if row["window_start"] else NO_DATA,
                 "参考窗口": f"{row['reference_start']} ~ {row['reference_end']}"
                 if row["reference_start"] else NO_DATA,
                 "说明": row["note"] or ""}
                for row in sorted(current, key=lambda item: item["feature_name"])
            ]),
            hide_index=True, width="stretch",
            column_config={
                "判定": st.column_config.TextColumn(
                    help=f"{drift_monitor.LEVEL_SEVERE}（PSI > {settings.DRIFT_PSI_SEVERE}）以红色高亮"
                )
            },
        )
        exceeded = [row for row in current
                    if row["threshold_exceeded"] in (drift_monitor.LEVEL_WATCH, drift_monitor.LEVEL_SEVERE)]
        for row in exceeded:
            severe = row["threshold_exceeded"] == drift_monitor.LEVEL_SEVERE
            (st.error if severe else st.warning)(
                f"**{row['feature_name']}** {row['threshold_exceeded']}：PSI {number(row['psi'])}"
                f"（关注线 {settings.DRIFT_PSI_WATCH}，严重线 {settings.DRIFT_PSI_SEVERE}）"
                + (f"。{row['note']}" if row["note"] else ""),
                icon=":material/warning:" if not severe else ":material/error:",
            )
        unavailable = [row for row in current if row["threshold_exceeded"] == drift_monitor.LEVEL_UNAVAILABLE]
        for row in unavailable:
            st.info(f"**{row['feature_name']}** 未接入：{row['note'] or '无数据源'}", icon=":material/block:")

    st.divider()
    st.subheader("概念漂移")
    if not concept_rows:
        st.caption("还没有概念漂移记录。")
    else:
        st.dataframe(
            pd.DataFrame([
                {"日期": row["date"], "窗口(交易日)": row["window_days"],
                 "滚动 Rank IC": number(row["rank_ic"], 4), "ICIR": number(row["icir"], 3),
                 "HAC t": number(row["hac_t"], 2), "NDCG比": number(row["ndcg_ratio"], 2),
                 "Top 超额": number(row["top_excess"], 4), "趋势": row["trend"],
                 "连续为负(日)": row["negative_streak"] if row["negative_streak"] is not None else NO_DATA}
                for row in concept_rows[:30]
            ]),
            hide_index=True, width="stretch",
        )
        st.caption(concept_rows[0]["note"] or "")

    st.divider()
    st.subheader("预测漂移")
    if not prediction_rows:
        st.caption("还没有预测漂移记录。")
    else:
        st.dataframe(
            pd.DataFrame([
                {"日期": row["date"], "Top N": row["top_n"],
                 "行业 HHI": number(row["industry_hhi"]), "判定": row["hhi_threshold_exceeded"],
                 "历史95分位": number(row["hhi_history_p95"]),
                 "市值分位均值": number(row["market_cap_quantile_mean"], 3),
                 "分数均值": number(row["score_mean"], 5), "分数标准差": number(row["score_std"], 5),
                 "股票数": row["stocks"] if row["stocks"] is not None else NO_DATA}
                for row in prediction_rows[:30]
            ]),
            hide_index=True, width="stretch",
        )
        st.caption(
            "分数均值/标准差取自未归一化的原始预测分：prediction_zscore 每日都会被强制标准化成均值 0、"
            "标准差 1，逐日记录它是常数，没有信息量。"
        )

    st.divider()
    st.subheader("降级与恢复")
    st.caption(
        "系统在通过率或严重漂移告警时**自动**降级（只做标记、暂停评分、写告警三件事，全部可逆，"
        "不改变生产模型）。解除降级与回滚都必须由管理员在这里点击，并写入 `quant_model_governance_events`。"
    )
    if current_status["degraded"]:
        st.warning(
            f"当前模型处于降级状态（原因：{current_status['reason'] or '未记录'}，"
            f"自 {stamp(current_status['since'])}）。"
        )
        confirm_clear = st.checkbox("我确认已经查明原因，恢复输出评分", key="confirm_clear_degradation")
        if st.button("解除降级并恢复评分", type="primary", disabled=not confirm_clear):
            try:
                model_governance.clear_degradation(repository, operator, note="管理员在模型健康页手动解除")
                st.success("已解除降级，评分输出恢复。")
                st.rerun()
            except DatabaseError as exc:
                st.error(str(exc))
    else:
        st.success("当前模型未降级，评分正常输出。")
        if st.button("手动暂停评分输出（置为降级）"):
            try:
                model_governance.degrade_model(
                    repository, reason=f"管理员 {operator} 在模型健康页手动暂停", operator=operator
                )
                st.success("已暂停评分输出。")
                st.rerun()
            except DatabaseError as exc:
                st.error(str(exc))

# ---------------------------------------------------------------------------
# 最终留出集
# ---------------------------------------------------------------------------
with tabs[2]:
    st.caption(
        "数据来源：`public.quant_model_final_holdout` 与 "
        f"`{settings.MODEL_DIR / settings.FINAL_HOLDOUT_REPORT_NAME}`。"
        "留出集只允许评估一次，重复评估需管理员确认。"
    )
    report = model_governance.holdout_report()
    try:
        record = model_governance.final_holdout_record(repository)
    except DatabaseError as exc:
        st.error(str(exc))
        record = None
    if report is None and record is None:
        show_empty_state(
            "还没有做过最终留出集评估",
            "最终留出集要求所有模型冻结后再跑，且一个模型版本只允许评估一次。"
            "可在命令行执行 `python scan_daily.py --evaluate-holdout`，或在此页面下方触发。",
        )
    else:
        if record is not None:
            st.markdown(
                f"**模型版本** `{record['model_version']}`　**评估时间** {stamp(record['evaluated_at'])}　"
                f"**留出集区间** {record['holdout_start']} ~ {record['holdout_end']}　"
                f"**结论** {'通过' if record['passed'] else '未通过'}　"
                f"**操作者** {record['operator'] or '未记录'}"
                + ("　**（强制重评覆盖）**" if record["forced"] else "")
            )
        if report is None:
            st.warning("库里有评估记录但报告文件不存在，展示内容以数据库记录为准。")
        else:
            summary = report.get("summary") or {}
            columns = st.columns(4)
            columns[0].metric("留出集交易日", f"{summary.get('holdout_dates', NO_DATA)} 日")
            columns[1].metric("结论", "通过" if summary.get("passed") else "未通过",
                              help="所有可用周期的冻结模型都通过各道门槛才算通过")
            columns[2].metric("平均单边换手率",
                              NO_DATA if summary.get("average_turnover") is None
                              else f"{summary['average_turnover']:.1%}",
                              help="每 20 个交易日采样的非重叠持仓期，Top 组合成员变更比例")
            columns[3].metric("费用后超额收益",
                              NO_DATA if summary.get("after_cost_return") is None
                              else f"{summary['after_cost_return']:.2%}",
                              help="毛超额减去按换手率与 settings.DEFAULT_COSTS 计算的费用")
            st.markdown("**分周期结果**")
            st.dataframe(
                pd.DataFrame([
                    {"周期(日)": int(horizon),
                     "可用": "是" if item["usable"] else "否",
                     "Rank IC": number((item["metrics"] or {}).get("rank_ic"), 4),
                     "HAC t": number((item["metrics"] or {}).get("rank_ic_hac_t"), 2),
                     "NDCG比": number((item["metrics"] or {}).get("ndcg_ratio"), 2),
                     "Top 超额": number((item["metrics"] or {}).get("top_excess_return"), 4),
                     "Top-Bottom": number((item["metrics"] or {}).get("top_bottom_spread"), 4),
                     "样本外交易日": (item["metrics"] or {}).get("dates", NO_DATA),
                     "冻结模型训练日数": (item["training"] or {}).get("available_dates", NO_DATA),
                     "窗口是否够": "是" if (item["training"] or {}).get("window_sufficient") else "否",
                     "说明": item["status"] or ""}
                    for _, item in sorted(report.get("horizons", {}).items(), key=lambda pair: int(pair[0]))
                ]),
                hide_index=True, width="stretch",
            )
            turnover = report.get("turnover") or {}
            if turnover.get("status") == "已计算":
                st.markdown(
                    f"**换手率与费用**（{turnover['unit']}）：持仓期 {turnover['periods']} 个，"
                    f"单边换手 {turnover['average_turnover']:.1%}，年化约 {turnover['annualized_turnover']:.1f} 倍；"
                    f"毛超额 {turnover['gross_excess_return']:.2%}，单期费用 {turnover['per_period_cost']:.2%}，"
                    f"费用后 {turnover['after_cost_return']:.2%}。"
                    f"（买入费率 {turnover['cost_rates']['buy']:.4%}，卖出费率 {turnover['cost_rates']['sell']:.4%}）"
                )
            else:
                st.info(turnover.get("status") or "未计算换手率。", icon=":material/info:")
            for note in report.get("notes", []):
                st.caption(f"· {note}")
            st.info(report.get("limitation") or "", icon=":material/info:")

    st.divider()
    with st.expander("强制重新评估最终留出集（管理员操作，会覆盖已有结论）"):
        st.warning(
            "留出集一旦评估过就应当冻结结论。重评会覆盖已有记录并在 "
            "`quant_model_governance_events` 留下审计；这样做之后，留出集不再是"
            "「从未看过」的数据，结论的解释力会下降。"
        )
        confirm_force = st.checkbox("我确认要覆盖已有留出集评估记录", key="confirm_force_holdout")
        if st.button("执行强制重新评估", disabled=not confirm_force):
            holder = st.empty()
            try:
                with st.spinner("正在用留出集之前的数据重训冻结模型并评估，可能需要几分钟……"):
                    payload = model_governance.evaluate_final_holdout(
                        repository, operator=operator, force=True,
                        progress=lambda message: holder.info(message),
                    )
                holder.empty()
                st.success(f"评估完成，结论：{'通过' if payload['passed'] else '未通过'}。")
                st.rerun()
            except (model_governance.FinalHoldoutError, DatabaseError) as exc:
                holder.empty()
                st.error(str(exc))

# ---------------------------------------------------------------------------
# 模型版本与回滚
# ---------------------------------------------------------------------------
with tabs[3]:
    st.caption(
        "数据来源：`public.quant_model_registry`。活动文件恒等于最新一次产出，历史版本归档在 "
        f"`{settings.MODEL_DIR / settings.MODEL_ARCHIVE_DIRNAME}/`，只保留最近 "
        f"{settings.MODEL_ARCHIVE_LIMIT} 个通过验证的版本。"
        "「可回滚」是现场校验归档文件真的存在后得出的，不采信登记表里的布尔值。"
    )
    try:
        versions = model_governance.registry_versions(repository)
    except DatabaseError as exc:
        st.error(str(exc))
        versions = []
    if not versions:
        show_empty_state("还没有模型版本记录", "每轮全量扫描完成后会登记一次，包含归档路径与是否通过验证。")
    else:
        st.dataframe(
            pd.DataFrame([
                {"版本 ID": row["context_id"][:12], "模型版本": row["model_version"],
                 "生成时间": stamp(row["created_at"]), "参考日": row["reference_date"] or NO_DATA,
                 "通过": "是" if row["passed"] else "否",
                 "通过率": NO_DATA if row["base_rate"] is None else f"{row['base_rate']:.0%}",
                 "当前活动": "是" if row["is_active"] else "否",
                 "可回滚": "是" if row["rollbackable"] else "否",
                 "归档": row["archive_path"] or "已清理"}
                for row in versions
            ]),
            hide_index=True, width="stretch",
        )
        candidates = {f"{row['context_id'][:12]}　{stamp(row['created_at'])}　"
                      f"{row['reference_date'] or '无参考日'}": row["context_id"]
                      for row in versions if row["rollbackable"]}
        st.subheader("回滚到上一通过模型")
        if not candidates:
            st.caption(
                "当前没有可回滚的版本：回滚候选必须是「已通过验证、不是当前活动版本、且归档文件仍存在」的版本。"
            )
        else:
            choice = st.selectbox("选择要回滚到的版本", list(candidates))
            st.warning(
                "回滚会把活动模型与面板替换为该版本的归档副本，并先把**当前**活动版本归档一次"
                "（所以点错了还能再滚回来）。全程写入 `quant_model_governance_events`。",
                icon=":material/history:",
            )
            confirm_rollback = st.checkbox("我确认要回滚到所选版本", key="confirm_rollback")
            if st.button("执行回滚", type="primary", disabled=not confirm_rollback):
                try:
                    outcome = model_governance.rollback_to(repository, candidates[choice], operator)
                    st.success(
                        f"已回滚到 {outcome['context_id'][:12]}（模型版本 {outcome['model_version']}，"
                        f"参考日 {outcome['reference_date'] or '未记录'}）。重新扫描或刷新页面即可看到该版本的结果。"
                    )
                    st.rerun()
                except (model_governance.FinalHoldoutError, DatabaseError) as exc:
                    st.error(str(exc))

# ---------------------------------------------------------------------------
# 告警
# ---------------------------------------------------------------------------
with tabs[4]:
    st.caption(
        "数据来源：`public.quant_model_alerts`。写库是必达通道；Webhook 与邮件为可选通道，"
        f"当前 Webhook {'已启用' if settings.ALERT_WEBHOOK_ENABLED else '未启用'}、"
        f"邮件 {'已启用' if settings.ALERT_EMAIL_ENABLED else '未启用'}。"
        f"同一标题的告警在 {settings.ALERT_DEDUP_HOURS} 小时内只记一条。"
    )
    try:
        alerts = model_governance.recent_alerts(repository, limit=100)
    except DatabaseError as exc:
        st.error(str(exc))
        alerts = []
    if not alerts:
        show_empty_state("还没有告警", "通过率、漂移与模型生命周期相关的告警会显示在这里。")
    else:
        open_alerts = [row for row in alerts if not row["acknowledged"]]
        st.markdown(f"共 {len(alerts)} 条，其中未处理 {len(open_alerts)} 条。")
        st.dataframe(
            pd.DataFrame([
                {"时间": stamp(row["created_at"]), "级别": row["level"], "类别": row["category"],
                 "标题": row["title"], "内容": row["message"],
                 "已处理": "是" if row["acknowledged"] else "否",
                 "处理人": row["acknowledged_by"] or NO_DATA}
                for row in alerts
            ]),
            hide_index=True, width="stretch",
        )
        if open_alerts:
            labels = {f"{stamp(row['created_at'])}　{row['title']}": row["id"] for row in open_alerts}
            selected_alert = st.selectbox("选择要标记为已处理的告警", list(labels))
            if st.button("标记为已处理"):
                try:
                    model_governance.acknowledge_alert(repository, labels[selected_alert], operator)
                    st.success("已标记。")
                    st.rerun()
                except DatabaseError as exc:
                    st.error(str(exc))
