"""按股票池扫描真实行情并展示可解释候选结果。"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from services.database import DatabaseError
from services.runtime import get_market_data, get_repository

from config import settings
from services.auth_ui import require_session
from services.model_governance import ScanGovernance, current_model_status
from services.rank_model import objective_label
from services import acquisition_store
from services.scan_pipeline import (
    FULL_MARKET_POOL_MODE,
    ScanOptions,
    collect_scan_snapshot,
    train_scan_snapshot,
)
from services.scan_store import build_payload, load_scan_result, save_scan_result
from services.ui_helpers import (
    loading_task,
    show_conclusion,
    show_empty_state,
    show_page_header,
)

st.set_page_config(page_title="量化选股 | A 股量化研究", layout="wide")
show_page_header(
    "量化选股",
    "一次扫描沪深京全部 A 股，先看筛出几只候选和各自的评分，再看每只凭什么入选。",
    "研究工作台 / 量化选股",
)

# 一轮全市场扫描训练出的模型上下文是所有账号共用的：谁扫，谁就决定别人明天看到什么。
# 因此扫描只由管理员触发，平日由收盘后的定时任务统一刷新；其余账号在本页只读最近一次结果。
# 与 pages/5_账号管理.py 一样做二次校验，避免以后调整路由时越权。
account = require_session()
is_admin = bool(account["is_admin"])

# 降级状态下本页仍可正常浏览，但评分列会是空的——先在页顶说清楚为什么，
# 免得使用者以为是扫描坏了。状态只读一次，扫描管线内部再按同一次读取执行。
model_health = current_model_status(get_repository())
if model_health["degraded"]:
    st.error(
        f"**{settings.DEGRADED_SCORING_MESSAGE}。** 原因：{model_health['reason'] or '未记录'}。"
        "候选排名与因子观察仍然可用；管理员可在“模型健康”页查看告警详情、解除降级或回滚到上一通过模型。",
        icon=":material/report:",
    )
elif model_health["error"]:
    st.warning(f"{model_health['error']}；本次按未降级处理。", icon=":material/warning:")

NOTE_RENDERERS = {
    "caption": st.caption,
    "info": st.info,
    "warning": st.warning,
    "error": st.error,
}


if is_admin:
    service = get_market_data()
    with st.container(border=True):
        st.subheader("筛选条件")
        display_columns = st.columns([3, 1], vertical_alignment="bottom")
        display_columns[0].caption(
            f"扫描范围固定为「{FULL_MARKET_POOL_MODE}」：排名与评分是所有账号共用的，"
            "只扫某个板块会让其他人看不到其余板块的结论。"
        )
        scan_limit = display_columns[1].number_input(
            "表格最多显示几行", min_value=1, max_value=100, value=20, step=1,
            help="只影响页面显示多少行，实际扫描的股票数量不受它限制。",
        )
        include_backtest = st.checkbox(
            "同时看每只股票的历史模拟回测（全市场会很慢）", value=False,
            help="采集时同时冻结逐股回测所需的不复权日线，训练时离线回放历史买卖；会明显拉长采集和计算时间。旧快照未准备回测数据时需要重新采集。",
        )
        training_columns = st.columns([2, 1, 2], vertical_alignment="bottom")
        objective_mode = training_columns[0].selectbox(
            "训练目标",
            settings.RANK_OBJECTIVE_MODES,
            index=settings.RANK_OBJECTIVE_MODES.index(settings.DEFAULT_OBJECTIVE_MODE),
            format_func=objective_label,
            help="排序目标让模型只在同一天里比谁更强，避免全市场普涨时所有股票都被判为看多；回归目标直接预测能比大盘多赚多少；自动择优会两种都训练，再挑历史表现更好的那个。",
        )
        training_window = training_columns[1].number_input(
            "滚动训练窗口（交易日）",
            min_value=0,
            max_value=2000,
            value=int(settings.TRAINING_WINDOW_DAYS),
            step=50,
            help="每次重训只看最近 N 个交易日的历史，好让模型跟上 A 股风格切换；填 0 表示用上全部历史。",
        )
        enable_neutralization = training_columns[2].checkbox(
            "剔除市值和行业的影响（中性化）",
            value=True,
            help="把「大盘股天生占优」「某个行业整体在涨」这类影响先扣掉，只留下个股自身的强弱；需要联网取行业数据，数据覆盖不足时会自动退回普通算法。",
        )
        use_extra_factors = st.checkbox(
            "叠加筹码、日内与财报因子（PIT 时点对齐）",
            value=True,
            help="在原有量价因子之上，再叠加筹码分布集中度、日内波动率衰减与订单流代理、以及按公告日对齐的财报因子。财报因子只在公告日之后才生效，不会用到当时还没公布的数字；某一类因子覆盖不到的交易日会留空，覆盖率不足时模型自动忽略该类因子。",
        )
        with st.expander("更多（可选）：用自定义代码试跑"):
            custom_codes = st.text_input("自定义代码（逗号或空格分隔）", placeholder="例：600000,000001")
            st.caption("留空即扫描全部 A 股。填了代码只在本页试跑，结果不落盘，也不会覆盖其他人看到的全市场结论。")
        st.caption("扫描会先下载股票池的历史行情，再训练一个覆盖全池的模型。参与训练的股票太少（不足 5 只）或数据缺失太多（覆盖不到本次股票池的 80%）时不会给出评分。")

    # 试跑范围与正式范围二选一：留空就扫全市场，这也是唯一会落盘、给所有人看的口径。
    pool_mode = "用户自定义" if custom_codes.strip() else FULL_MARKET_POOL_MODE

    options = ScanOptions(
        pool_mode=pool_mode, custom_codes=custom_codes, scan_limit=int(scan_limit),
        objective_mode=objective_mode, training_window=int(training_window),
        enable_neutralization=enable_neutralization, use_extra_factors=use_extra_factors,
        include_backtest=include_backtest, trigger="manual",
    )

    # 采集与训练拆成两步：采集冻结一份截面快照（可中断续跑），训练只消费快照。
    # 同一份快照可以换训练目标和窗口反复训练；逐股回测日线须在采集时一并准备。
    ready_snapshot = acquisition_store.latest_snapshot(ready_only=True)
    latest_snapshot = acquisition_store.latest_snapshot()
    action_columns = st.columns(2)
    collect_clicked = action_columns[0].button(
        "① 采集股票池数据", type="primary",
        help="下载股票池的报价与历史行情并冻结成快照；中途关闭页面后再次点击会从断点续跑。",
    )
    if ready_snapshot is None:
        st.caption("还没有可训练的快照，请先点击「① 采集股票池数据」。")
    else:
        st.caption(
            f"可训练快照 {ready_snapshot.snapshot_id} · 截面日期 {ready_snapshot.reference_date}，"
            "训练只使用该快照，不会重新下载行情。"
        )
    if latest_snapshot is not None and latest_snapshot.status == "failed":
        st.warning(f"最近一次采集未完成：{latest_snapshot.meta.get('error') or '未知原因'}，可重试采集或换用上面的可训练快照。")

    if collect_clicked:
        with loading_task("采集股票池数据", "先取沪深 300 基准确定截面日期，再逐股下载报价与历史。中断后再次点击会跳过已完成股票。", exclusive=True) as task:
            snapshot = collect_scan_snapshot(service, options, task.step)
            if snapshot is None or snapshot.status != "ready":
                task.fail((snapshot.meta.get("error") if snapshot else None) or "采集未完成，请查看下方信息。")
        if snapshot is None:
            st.error("采集失败：未能建立快照。")
        else:
            source_meta = snapshot.meta.get("source_meta") or {}
            counts = snapshot.meta.get("counts") or {}
            if snapshot.status != "ready":
                st.error(snapshot.meta.get("error") or "采集未完成。")
            else:
                st.success(
                    f"快照 {snapshot.snapshot_id} 已就绪：截面 {snapshot.reference_date}，"
                    f"有效 {counts.get('success', 0):,} 只、失败 {counts.get('failed', 0):,} 只、"
                    f"续跑跳过 {counts.get('skipped', 0):,} 只。"
                )
            if source_meta:
                st.info(
                    f"源端总数 {source_meta.get('source_total_symbols', counts.get('total', 0)):,}；"
                    f"返回行数 {source_meta.get('source_rows', counts.get('total', 0)):,}；"
                    f"去重股票池 {source_meta.get('source_unique_symbols', counts.get('total', 0)):,}；"
                    f"有效股票 {source_meta.get('valid_symbols', counts.get('total', 0)):,}；"
                    f"排除 {source_meta.get('excluded_symbols', 0):,}；失败 {source_meta.get('failed_symbols', 0):,}；"
                    f"失败页 {source_meta.get('failed_pages', 0):,}；本次采集 {counts.get('total', 0):,}；"
                    f"来源 {source_meta.get('source', source_meta.get('status'))}；"
                    f"更新时间 {source_meta.get('fetched_at', '用户输入')}。"
                )
            if source_meta.get("warning") or source_meta.get("is_stale"):
                st.warning(source_meta.get("warning", "当前使用完整过期股票池缓存，行情非实时。"))
            for note in snapshot.meta.get("notes") or []:
                NOTE_RENDERERS.get(note.get("level", "caption"), st.caption)(note["text"])
            if snapshot.status == "ready":
                ready_snapshot = snapshot

    train_clicked = action_columns[1].button(
        "② 训练模型并生成排名", disabled=ready_snapshot is None,
        help="只使用已冻结的快照训练，不会重新下载行情；需要最新数据请先重新采集。",
    )
    if train_clicked and ready_snapshot is not None:
        result = None
        with loading_task("训练模型并生成排名", f"只使用采集快照 {ready_snapshot.snapshot_id}（截面 {ready_snapshot.reference_date}），不会重新下载行情。", exclusive=True) as task:
            try:
                result = train_scan_snapshot(
                    service, options, ready_snapshot.snapshot_id, task.step,
                    governance=ScanGovernance(get_repository(), trigger="manual", operator=account["username"]),
                )
            except (ValueError, OSError) as exc:
                task.fail(str(exc))
                st.error(f"训练未开始：{exc}")
        if result is not None:
            if options.pool_mode == "用户自定义":
                # 试跑只留在本次会话里：落盘文件是所有账号共用的那份全市场结论，一次试跑不该把它换掉。
                st.session_state["scan_result"] = build_payload(result)
            else:
                save_scan_result(result)
                # 回读一次再展示：这样管理员此刻看到的，与别人之后读到的严格是同一份内容。
                st.session_state["scan_result"] = load_scan_result()
else:
    st.info(
        "全市场扫描会训练一份所有账号共用的模型，因此只由管理员手动触发，"
        "平日由收盘后的定时任务统一执行。下面显示的是最近一次扫描的结果。"
    )

payload = st.session_state.get("scan_result") or load_scan_result()
rows = (payload or {}).get("rows") or []
scan_stats = (payload or {}).get("scan_stats") or {}

if not rows:
    if payload and scan_stats.get("scanned"):
        show_empty_state("本轮未生成候选结果", "扫描已经结束，但没有可展示的候选；下方提供有效数量、失败原因和数据源状态。")
    elif is_admin:
        show_empty_state("配置范围，开始发现候选股票", "首次可使用少量自定义代码体验。全市场扫描包含逐股行情获取与模型训练，耗时取决于股票数量和网络。")
    else:
        show_empty_state("还没有可看的扫描结果", "管理员或收盘后的定时任务跑完第一轮全市场扫描后，这里会显示候选排名。")
    if payload and scan_stats.get("scanned"):
        st.warning(
            f"最近一轮扫描已结束：处理 {scan_stats.get('scanned', 0):,} 只，"
            f"有效 {scan_stats.get('scored', 0):,} 只，失败或排除 {scan_stats.get('failed', 0):,} 只；"
            "本轮没有生成可展示的候选行。"
        )
        for note in payload.get("notes") or []:
            NOTE_RENDERERS.get(note.get("level", "caption"), st.caption)(note["text"])
        if is_admin:
            failed_rows = payload.get("failed_rows") or []
            if failed_rows:
                with st.expander(f"查看失败或排除股票（{len(failed_rows):,}）", expanded=True):
                    st.dataframe(pd.DataFrame(failed_rows), hide_index=True, width="stretch")
            st.caption("以下为全局历史行情采集记录（含此前任务）；本轮股票的失败原因请以上方清单为准。")
            history_summary = service.history_fetch_summary()
            st.caption(
                f"历史行情状态：已成功/缓存 {history_summary['success_count']:,}；"
                f"过期缓存 {history_summary['stale_count']:,}；获取失败 {history_summary['failure_count']:,}。"
            )
            if history_summary["failures"]:
                with st.expander("查看全局历史行情失败记录（含此前任务）", expanded=True):
                    st.dataframe(pd.DataFrame(history_summary["failures"]), hide_index=True, width="stretch")

if rows:
    st.caption(
        f"本表来自 {payload.get('finished_at', '未知时间')} 的"
        f"{'每日定时任务' if payload.get('trigger') == 'scheduled' else '管理员手动扫描'}，"
        f"股票池「{payload.get('pool_mode', '未知')}」，截面日期 {payload.get('reference_date') or '未产出'}。"
        "你正在看的不是自己刚跑出来的结果时，行情与评分也停留在那一刻。"
    )
    for note in payload.get("notes") or []:
        NOTE_RENDERERS.get(note.get("level", "caption"), st.caption)(note["text"])
    scored_rows = [row for row in rows if row.get("综合评分") is not None]
    best = max(scored_rows, key=lambda row: row["综合评分"]) if scored_rows else None
    show_conclusion(
        f"本次扫描 {scan_stats.get('scanned', 0):,} 只股票，筛出 {len(rows):,} 只候选"
        + (f"，其中评分最高的是 {best['名称']}（{best['代码']}）。" if best else "。")
        + " 评分是“同一天里相对更值得看”的排名，不是涨跌预测，也不保证赚钱。"
    )
    st.caption(
        f"扫描明细：行情与因子分析完成 {scan_stats.get('scored', 0):,} 只；"
        f"具有综合评分 {len(scored_rows):,} 只；失败或排除 {scan_stats.get('failed', 0):,} 只。"
    )
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            "综合评分": st.column_config.NumberColumn(
                "综合评分",
                format="%.1f",
                help="把因子分和模型分加权后的总分（0~100）。只和同一天的其他股票比，分数高表示相对更值得看。",
            ),
            "涨跌幅": st.column_config.NumberColumn(
                "涨跌幅", format="%.2f", help="最近一次行情相对上一收盘价的涨跌，单位是百分比。"
            ),
            "参考买入区间": st.column_config.TextColumn(
                "参考买入区间",
                help="按日常波动幅度算出的参考价，不是成交价。次日开盘不在这个区间内，就不假设买到了。",
            ),
            "止损参考": st.column_config.NumberColumn(
                "止损参考", format="%.2f", help="跌破这个价格就按规则卖出，用来限制单只股票最多亏多少。"
            ),
            "推荐理由": st.column_config.TextColumn("推荐理由", width="large"),
            "主要风险": st.column_config.TextColumn("主要风险", width="large"),
        },
    )
    detail_rows = payload.get("detail_rows") or []
    with st.expander(f"展开技术指标明细（{len(detail_rows):,} 行，给懂行的人看）"):
        st.caption(
            "截面Z分数＝这只股票的模型分比当天全市场平均值高出多少（0 是平均，正数更看好）；"
            "Rank IC＝样本外每天“模型排名”和“真实涨跌排名”的相关性，越高说明模型排序越靠谱；"
            "20日预测值在排序目标下只是同一天里的相对排名百分位，只有回归目标下才是预期超额收益；"
            "时序因子分只看这只股票自己的历史，和全市场比出来的综合评分不是一回事。"
        )
        st.dataframe(pd.DataFrame(detail_rows), hide_index=True, width="stretch")
    failed_rows = payload.get("failed_rows") or []
    if failed_rows:
        with st.expander(f"查看失败或排除股票（{len(failed_rows):,}）"):
            st.dataframe(pd.DataFrame(failed_rows), hide_index=True, use_container_width=True)
    if is_admin:
        # 行情接口的成功率与代理诊断是运维信息，只给管理员看，也只给管理员取。
        st.caption("以下为全局历史行情采集记录（含此前任务）；本轮股票的失败原因请以上方清单为准。")
        history_summary = service.history_fetch_summary()
        st.caption(
            f"历史行情状态：已成功/缓存 {history_summary['success_count']:,}；"
            f"过期缓存 {history_summary['stale_count']:,}；获取失败 {history_summary['failure_count']:,}。"
        )
        if history_summary["by_stock"]:
            with st.expander(f"查看逐股历史行情状态与接口成功率（{len(history_summary['by_stock']):,}）"):
                st.dataframe(pd.DataFrame(history_summary["by_stock"]), hide_index=True, use_container_width=True)
        if history_summary["failures"]:
            with st.expander("查看全局历史行情失败记录（含此前任务）"):
                st.dataframe(pd.DataFrame(history_summary["failures"]), hide_index=True, width="stretch")
        with st.expander("查看接口请求及代理诊断"):
            proxy = service.proxy_configuration()
            st.caption(f"检测到系统代理：{proxy or '未设置'}；实际连接路由：{service.network.status()}。详细请求记录写入 logs/market_data.log。")
            request_rows = service.request_diagnostics()
            if request_rows:
                st.dataframe(pd.DataFrame(request_rows), hide_index=True, use_container_width=True)
    usable = payload.get("usable") or []
    if usable:
        labels = [f"{item['code']}  {item['name']}" for item in usable]
        selected = st.selectbox("查看完整个股分析", labels)
        selected_code = selected.split()[0]
        if st.button("将所选股票加入自选", key="scan_add_watch"):
            try:
                selected_item = next(item for item in usable if item["code"] == selected_code)
                get_repository().save_watch(selected_code, selected_item["name"])
                st.success("已保存到自选。")
            except (DatabaseError, ValueError) as exc:
                st.error(str(exc))
        st.page_link(str(settings.PROJECT_ROOT / "pages" / "1_个股分析.py"), label=f"打开 {selected_code} 个股分析", query_params={"code": selected_code})
        with st.expander("综合评分是怎么算出来的"):
            st.write("综合评分 = 40% × 同日因子分 + 60% × 通过验证的模型分。")
            st.caption(
                "因子分是把它近期的涨跌、波动、量能放在一起打的 0~100 分；模型分是模型在同一天全市场里的相对位置。"
                "模型没通过历史检验时，只展示因子观察，不给预测分。"
                "选「排序目标」时，20 日预测值只是同一天里的相对排名，不是预期能赚多少；选「回归目标」才是预期超额收益。"
            )
