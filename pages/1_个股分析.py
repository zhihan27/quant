"""个股查询、因子和中低频模型分析页面。"""

from __future__ import annotations

from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from config import settings
from services.backtest import run_backtest, save_backtest_run
from services.database import DatabaseError
from services.market_data import MarketDataError, completed_daily_history, history_is_current
from services.model_governance import current_model_status
from services.rank_model import SCORE_KIND_RANK, objective_label
from services.stock_scoring import analyze_stock
from services.statistical_arbitrage import analyze_pair
from services.ui_helpers import format_number as _fmt
from services.ui_helpers import show_backtest_metrics as _show_backtest_metrics
from services.ui_helpers import show_candles as _show_candles
from services.ui_helpers import loading_task, show_conclusion, show_empty_state, show_page_header, show_plain_numbers
from services.runtime import get_market_data, get_repository

SHANGHAI = ZoneInfo("Asia/Shanghai")

st.set_page_config(page_title="个股分析 | A 股量化研究", layout="wide")
show_page_header(
    "个股分析",
    "输入股票代码或名称，先看结论和买卖参考价，再看它凭什么这么判断。",
    "研究工作台 / 个股分析",
)


def _fmt_amount(value: Any) -> str:
    """成交额按亿元显示：行情源给的是元，十位数放进并排的窄列会被省略号截断。"""
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value) / 1e8:,.2f} 亿元"
    except (TypeError, ValueError):
        return "—"


def _stock_conclusion(plan: dict) -> tuple[str, str]:
    """把模型判断和买卖参考浓缩成一句话，不懂指标也能直接读结论。"""
    zone = plan.get("buy_zone")
    if zone and plan.get("signal_active"):
        return (
            f"模型看多。参考在 {zone[0]} ~ {zone[1]} 元买入，跌破 {plan.get('initial_stop')} 元止损。"
            "这是研究结论，不是保证会涨。",
            "good",
        )
    if zone:
        return (f"模型这次不支持直接买入（{plan.get('status')}）。下方价格仅供复核，不要据此下单。", "warn")
    return ("模型这次没有给出可用判断，原因见下方「模型判断」。", "warn")


# 降级状态只读一次、整页共用：逐股分析里再查一次库既慢又没有意义。
model_health = current_model_status(get_repository())
scoring_paused = bool(model_health["degraded"])
if scoring_paused:
    st.error(
        f"**{settings.DEGRADED_SCORING_MESSAGE}。** 原因：{model_health['reason'] or '未记录'}；"
        f"自 {model_health['since'].astimezone(SHANGHAI).strftime('%Y-%m-%d %H:%M') if model_health['since'] else '未记录'}。"
        "下方的因子观察与价格参考仍然可用，但不再包含模型判断。",
        icon=":material/report:",
    )
elif model_health["error"]:
    st.warning(f"{model_health['error']}；本次按未降级处理，模型评分正常输出。", icon=":material/warning:")

service = get_market_data()
if st.query_params.get("code") and st.session_state.get("query_code") != st.query_params.get("code"):
    st.session_state["stock_query"] = st.query_params.get("code", "")
    st.session_state["query_code"] = st.query_params.get("code")
    st.session_state["selected_code"] = st.query_params.get("code")
    st.session_state["search_results"] = pd.DataFrame()
    st.session_state.pop("match_selection", None)

with st.form("stock_search_form"):
    search_columns = st.columns([5, 1, 1], vertical_alignment="bottom")
    with search_columns[0]:
        query = st.text_input("股票代码或名称", key="stock_query", placeholder="例如 600000 或 浦发银行")
    with search_columns[1]:
        submitted = st.form_submit_button("搜索股票", type="primary", use_container_width=True)
    with search_columns[2]:
        refresh_submitted = st.form_submit_button("刷新目录", use_container_width=True)
    st.caption("六位代码可直接定位；名称优先读取本地完整目录，需要最新名单时再刷新。")

if submitted or refresh_submitted:
    term = query.strip()
    # 新一轮搜索会换掉候选列表，上一轮的选中项对新列表无效，先清掉，
    # 否则下拉框会带着一个已不存在的值，Streamlit 只能回退到第一项。
    st.session_state.pop("match_selection", None)
    if term.isdigit() and len(term) == 6:
        st.session_state["selected_code"] = term
        st.session_state["search_results"] = pd.DataFrame()
    elif term:
        st.session_state.pop("selected_code", None)
        st.session_state["search_results"] = pd.DataFrame()
        try:
            title = "刷新股票目录" if refresh_submitted else "查询股票目录"
            detail = "正在重新获取并核对完整股票目录，完成后保存到本地缓存。" if refresh_submitted else "优先使用完整本地目录；首次获取或缓存过旧时才访问行情源。"
            with loading_task(title, detail) as task:
                task.step("正在获取完整股票目录并匹配名称…")
                results, search_meta = service.search_stocks(term, force_refresh=refresh_submitted)
            st.session_state["search_results"] = results
            st.session_state["search_meta"] = search_meta
            if results.empty:
                st.warning(f"没有找到股票；列表状态：{search_meta.get('status', '未知')}")
            else:
                st.info(f"股票列表：{search_meta.get('status')}，采集时间 {search_meta.get('fetched_at', '未知')}")
        except MarketDataError as exc:
            st.session_state["search_results"] = pd.DataFrame()
            st.error(str(exc))
    else:
        st.warning("请输入六位代码或股票名称。")

matches = st.session_state.get("search_results", pd.DataFrame())
if isinstance(matches, pd.DataFrame) and not matches.empty:
    labels = [f"{row.code}  {row.name}" for row in matches.itertuples()]
    selected_label = st.selectbox("匹配结果", labels, key="match_selection")
    selected_code = selected_label.split()[0]
    # 下拉框选中即生效。原先这里还要再点一次「分析所选股票」，而那个按钮既不分析也不跳转，
    # 只是把选择写进会话，与下方真正的「运行分析」重复，故合并为一个动作。
    st.session_state["selected_code"] = selected_code
elif st.session_state.get("selected_code"):
    selected_code = st.session_state["selected_code"]
else:
    selected_code = None

if selected_code:
    action_columns = st.columns([2, 2, 6], vertical_alignment="center")
    run_analysis = action_columns[0].button("运行分析", type="primary", key="run_analysis", use_container_width=True)
    add_watch = action_columns[1].button("加入自选", key="analysis_add_watch", use_container_width=True)
    action_columns[2].caption(f"当前股票 {selected_code} · 分析包含行情获取、因子计算与共享截面模型验证结果")
else:
    run_analysis = False
    add_watch = False
    show_empty_state("从一只股票开始研究", "搜索股票 → 选择匹配结果 → 运行分析。分析完成后可查看图表、交易参考与回测。")

if add_watch:
    try:
        current = st.session_state.get("stock_analysis", {})
        name = current.get("quote", {}).get("name", "") if current.get("code") == selected_code else ""
        get_repository().save_watch(selected_code, name)
        st.success("已保存到自选，重启后保留。")
    except (DatabaseError, ValueError) as exc:
        st.error(str(exc))

if run_analysis:
    for state_key in ("stock_analysis", "backtest_result", "pair_result"):
        st.session_state.pop(state_key, None)
    try:
        # 单只股票只读已落盘的模型、不写任何共享产物，因此不占全局计算锁，
        # 免得收盘后的全市场扫描把它挡在门外。
        with loading_task("个股分析", "依次获取行情、核对历史数据并读取同日共享截面模型。") as task:
            task.step("1 / 5 · 获取最新报价", 0, 5)
            quote = service.get_quote(selected_code)
            task.step("2 / 5 · 获取日线历史", 1, 5)
            daily, daily_meta = service.get_history(selected_code, count=1200, period="daily", adjusted=True)
            task.step("3 / 5 · 获取周线历史", 2, 5)
            try:
                weekly, weekly_meta = service.get_history(
                    selected_code, count=400, period="weekly", adjusted=True
                )
            except MarketDataError as exc:
                weekly = pd.DataFrame()
                weekly_meta = {"status": "周线数据不可用", "warning": str(exc)}
            benchmark = None
            benchmark_status = "沪深 300 指数不可用；相对市场因子未计算。"
            task.step("4 / 5 · 获取沪深 300 基准", 3, 5)
            try:
                benchmark, _ = service.get_history(
                    settings.MARKET_INDEX_CODE, count=1200, period="daily", adjusted=True, market_id=1
                )
                benchmark_status = "沪深 300 指数历史数据已获取。"
            except MarketDataError as exc:
                benchmark = None
                benchmark_status = f"沪深 300 指数不可用；相对市场因子未计算：{exc}"
            completed = completed_daily_history(daily)
            task.step("5 / 5 · 计算因子并核对 5、20、60 日超额收益模型上下文", 4, 5)
            analysis = analyze_stock(
                completed,
                current_price=quote.get("price"),
                benchmark=benchmark,
                horizons=(5, 20, 60),
                code=selected_code,
                as_of=quote.get("quote_time") or quote.get("fetched_at"),
                quote_status=quote.get("data_status"),
                history_ready=history_is_current(completed, quote, daily_meta),
                scoring_paused=scoring_paused,
            )
            st.session_state["stock_analysis"] = {
                "code": selected_code,
                "quote": quote,
                "daily": daily,
                "daily_meta": daily_meta,
                "weekly": weekly,
                "weekly_meta": weekly_meta,
                "benchmark": benchmark,
                "benchmark_status": benchmark_status,
                "analysis": analysis,
            }
            task.step("分析结果已生成", 5, 5)
    except (MarketDataError, ValueError) as exc:
        st.error(str(exc))

state = st.session_state.get("stock_analysis")
if state and state.get("code") == selected_code:
    quote = state["quote"]
    daily = state["daily"]
    weekly = state["weekly"]
    analysis = state["analysis"]
    st.subheader(f"{quote.get('name', selected_code)}（{selected_code}）")
    st.caption(
        f"市场：{quote.get('market', '未确认')} | 来源：{quote.get('source', '未知')} | "
        f"行情源时间：{quote.get('quote_time') or '未提供'} | 本地采集：{quote.get('fetched_at', '未知')} | "
        f"状态：{quote.get('data_status', '未知')}"
    )
    if quote.get("warning"):
        st.warning(f"数据提示：{quote['warning']}")

    # 拆成两行：七个指标挤在一行时每列不足 200px，成交量与成交额会被省略号截断。
    price_columns = st.columns(4)
    price_columns[0].metric("最新可用价", _fmt(quote.get("price"), 2))
    price_columns[1].metric("涨跌幅", _fmt(quote.get("change_pct"), 2, "%"))
    price_columns[2].metric("成交量", _fmt(quote.get("volume_lots"), 0, " 手"))
    price_columns[3].metric("成交额", _fmt_amount(quote.get("amount_yuan")))

    price_columns = st.columns(4)
    price_columns[0].metric("开盘", _fmt(quote.get("open"), 2))
    price_columns[1].metric("最高 / 最低", f"{_fmt(quote.get('high'), 2)} / {_fmt(quote.get('low'), 2)}")
    price_columns[2].metric("昨收", _fmt(quote.get("previous_close"), 2))

    plan = analysis.get("trade_plan", {})
    show_conclusion(*_stock_conclusion(plan))

    chart_tab, factor_tab, model_tab, signal_tab = st.tabs(["行情图表", "走势打分", "模型判断", "买卖参考"])
    with chart_tab:
        st.caption(f"周线状态：{state['weekly_meta'].get('status', '可用')}")
        st.caption(
            f"日线：{state['daily_meta'].get('adjustment')}，最新 K 线 {state['daily_meta'].get('latest_bar_date')}；"
            f"周线最新 K 线 {state['weekly_meta'].get('latest_bar_date')}。"
            "盘中未完成的当日 K 线只展示，不用于模型训练。"
        )
        _show_candles(daily, "日 K 线及成交量", 160)
        _show_candles(weekly, "周 K 线及成交量", 100)
    with factor_tab:
        st.caption(state.get("benchmark_status", ""))
        show_plain_numbers([
            (
                "走势打分",
                _fmt(analysis.get("factor_score"), 1, " / 100"),
                "把近期的涨跌、波动、量能等放在一起算出的 0~100 分（时序多因子评分）。"
                "这只股票现在跟它自己的历史比是什么水平，分数越高越强。",
            ),
        ], columns=1)
        factors = analysis.get("factor_details", [])
        if factors:
            factor_frame = pd.DataFrame(factors).rename(
                columns={"factor": "打分项", "value": "当前值", "z_score": "比平时高多少", "score": "这项得几分", "weight": "占多大分量"}
            )
            st.dataframe(
                factor_frame,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "比平时高多少": st.column_config.NumberColumn(
                        help="拿这只股票自己的历史当尺子：0 表示和平时一样，+2 表示明显比平时高，-2 表示明显比平时低（时序 Z 值）。"
                    ),
                    "占多大分量": st.column_config.NumberColumn(help="这一项在总分里占的比重（权重）。"),
                },
            )
        else:
            st.info("当前历史长度不足以计算稳定的打分。")
        latest = analysis.get("latest_features", {})
        st.write(
            {
                "近 5 个交易日涨跌": latest.get("momentum_5"),
                "近 20 个交易日涨跌": latest.get("momentum_20"),
                "近 60 个交易日涨跌": latest.get("momentum_60"),
                "波动大小（折算成年）": latest.get("volatility_20"),
                "日常波动幅度 ÷ 价格": latest.get("atr_pct"),
                "近 20 日比沪深 300 强多少": latest.get("relative_strength_20"),
            }
        )
        st.caption(
            "「日常波动幅度」指这只股票平均一天上下波动的幅度（ATR），除以价格好和别的股票横向比较；"
            "数值越大代表价格越不老实，买卖参考价也要留出更大空间。"
        )
        st.caption(analysis.get("fundamental_status", ""))
    with model_tab:
        model_rows = []
        importance_rows = []
        for horizon, models in analysis["models"].items():
            for item in models:
                metrics = item.get("oos_metrics") or {}
                ranking = item.get("score_kind") == SCORE_KIND_RANK
                percentile = item.get("prediction_percentile")
                model_rows.append(
                    {
                        "模型": item["model"],
                        "预测周期": f"{horizon} 个交易日",
                        "模型这次怎么说": item.get("status"),
                        "在当天全市场的位置": f"排在前 {100 - percentile * 100:.0f}%" if percentile is not None else "—",
                        "目标口径": objective_label(item.get("objective")) if item.get("objective") else None,
                        "模型预测值": item.get("predicted_excess_return"),
                        "同日截面Z分数": item.get("prediction_zscore"),
                        "同日截面百分位": item.get("prediction_percentile"),
                        "训练窗口（交易日）": item.get("training_window_days"),
                        "样本数": item.get("samples"),
                        "样本外Rank IC": metrics.get("rank_ic"),
                        "IC的HAC t值": metrics.get("rank_ic_hac_t"),
                        "Top组超额收益": metrics.get("top_excess_return"),
                        "Bottom组超额收益": metrics.get("bottom_excess_return"),
                        "Top-Bottom差": metrics.get("top_bottom_spread"),
                        # 排序目标下预测值无量纲，只能看 NDCG；回归目标才看绝对误差基线。
                        "样本外 NDCG@前20%": metrics.get("ndcg") if ranking else None,
                        "NDCG/随机基线": metrics.get("ndcg_ratio") if ranking else None,
                        "样本外 MAE": metrics.get("mae"),
                        "中位数基线 MAE": metrics.get("baseline_mae"),
                        "状态": item.get("status"),
                    }
                )
                if horizon == 20:
                    importance_rows.extend(
                        {"模型": item["model"], **row}
                        for row in item.get("feature_importance", [])[:5]
                    )
        if model_rows:
            st.dataframe(
                pd.DataFrame(model_rows)[["模型", "预测周期", "模型这次怎么说", "在当天全市场的位置"]],
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                "「在当天全市场的位置」＝这只股票当天的分数在候选股票里排第几，排在前 10% 表示比 90% 的股票更被看好。"
                "排名靠前只说明相对更强，不等于这只股票一定会涨。"
            )
        st.warning("模型没通过检验时不会给出预测分数，只显示它为什么没通过——宁可说“这次不知道”，也不给一个看着像结论的数字。")
        with st.expander("展开技术指标（Rank IC、NDCG、HAC t 值等）"):
            st.dataframe(pd.DataFrame(model_rows), hide_index=True, use_container_width=True)
            st.caption(
                "Rank IC 衡量模型把当天该涨的股票排在前面的能力，0.02 以上才算有点用；"
                "NDCG 看排在前 20% 的那批股票准不准，和随机乱排相比高 5% 以上才算过关；"
                "HAC t 值是判断 Rank IC 是不是碰巧的统计量，要大于 1.645。"
                "这三项加上分组收益检验，任意一项不通过就不输出评分。"
            )
            st.caption(
                "排序目标下模型预测值只是同日截面内的相对排序分，判断依据是 NDCG 相对随机基线的提升；"
                "只有回归目标下的预测值才是相对沪深 300 的预期超额收益。"
                "Z分数按同日股票池标准化，不换算绝对价格。请先在量化选股扫描至少5只股票建立共享截面模型。"
            )
            st.caption(analysis.get("cross_section_scope", ""))
            if importance_rows:
                st.markdown("**20 日模型特征重要性**")
                st.dataframe(pd.DataFrame(importance_rows), hide_index=True, use_container_width=True)
    with signal_tab:
        st.caption(f"当前状态：{plan.get('status', '不可用')}")
        if plan.get("buy_zone"):
            zone = plan["buy_zone"]
            show_plain_numbers([
                ("参考买入区间", f"{zone[0]} ~ {zone[1]} 元", "按最近波动算出的低吸区间，不是预测价，也不保证一定买得到。"),
                ("跌破就卖（止损）", f"{_fmt(plan.get('initial_stop'), 2)} 元", "跌到这里说明这次判断错了，认赔离场，别硬扛。"),
                ("涨到就先卖（止盈）", f"{_fmt(plan.get('take_profit'), 2)} 元", "到达这个价位先落袋，避免赚到的又还回去。"),
            ])
            st.write(
                {
                    "大概持有多久": plan.get("holding_period"),
                    "什么情况会作废": plan.get("invalidation"),
                    "参考价（不是成交价）": plan.get("reference_price"),
                }
            )
            with st.expander("展开判断依据（入场条件、退出规则、计算方式）"):
                st.write(
                    {
                        "入场条件": plan.get("entry_trigger"),
                        "移动止盈 / 趋势退出": plan.get("trailing_exit"),
                        "计算依据": plan.get("calculation"),
                        "生成时间 / 有效期": f"{plan.get('generated_at')} / {plan.get('valid_until')}",
                    }
                )
                st.caption(
                    "上面提到的「ATR」是这只股票平均一天上下波动的幅度。买入区间是参考价上下各 0.25 倍，止损是 −2 倍，止盈是 +3 倍；"
                    "倍数越大留给价格的空间越宽。参考价取最新一根完整日线的收盘价。"
                )
            st.warning(plan.get("risk_note", ""))
        else:
            st.info("价格或日常波动幅度（ATR）数据不足，未生成买卖参考价。")

    st.divider()
    st.subheader("历史表现回顾（模拟交易）")
    st.caption(
        "把本页的买卖规则放到历史上跑一遍，看它过去表现如何。用的是当时训练出来的模型，"
        "并且算上了 T+1（今天买的明天才能卖）、停牌和一字涨跌停根本买不到／卖不出的情况。"
        "过去这样不代表以后也这样。"
    )
    if st.button("开始模拟并保存结果", key="run_backtest"):
        try:
            with loading_task("滚动样本外回测", "回测会分时段训练模型，计算交易与持仓净值，耗时通常长于单次分析。", exclusive=True) as task:
                task.step("1 / 4 · 获取未复权行情", 0, 4)
                raw_history, raw_meta = service.get_history(
                    selected_code, count=1500, period="daily", adjusted=False
                )
                raw_history = completed_daily_history(raw_history)
                task.step("2 / 4 · 获取复权历史并对齐交易日", 1, 4)
                adjusted_history, _ = service.get_history(selected_code, count=1500, adjusted=True)
                adjusted_history = completed_daily_history(adjusted_history)
                task.step("3 / 4 · 运行滚动训练与逐日回测，请稍候…", 2, 4)
                backtest_result = run_backtest(
                    raw_history,
                    code=selected_code,
                    adjusted_history=adjusted_history,
                    benchmark=state.get("benchmark"),
                    is_st="ST" in str(quote.get("name", "")).upper(),
                )
                if backtest_result.get("metrics") is not None:
                    task.step("4 / 4 · 保存回测结果", 3, 4)
                    run_id = save_backtest_run(
                        {
                            "code": selected_code,
                            "holding_days": 20,
                            "strategy": "model",
                            "history_source": raw_meta.get("source"),
                            "history_fetched_at": raw_meta.get("fetched_at"),
                        },
                        backtest_result,
                    )
                    backtest_result["run_id"] = run_id
                backtest_result["code"] = selected_code
                st.session_state["backtest_result"] = backtest_result
        except (MarketDataError, ValueError, DatabaseError) as exc:
            st.error(str(exc))
    backtest_result = st.session_state.get("backtest_result")
    if backtest_result and backtest_result.get("code") == selected_code:
        st.info(backtest_result.get("status", ""))
        if backtest_result.get("metrics"):
            metrics = backtest_result["metrics"]
            _show_backtest_metrics(metrics, backtest_result.get("benchmark_buy_hold_return"))
            st.caption(
                f"模型版本 {backtest_result.get('model_version')}；结果 ID {backtest_result.get('run_id', '未保存')}；"
                f"未成交进场 {backtest_result.get('unfilled_entries')}，未成交退出 {backtest_result.get('unfilled_exits')}。"
            )
            if backtest_result.get("daily_equity"):
                equity_frame = pd.DataFrame(backtest_result["daily_equity"]).set_index("date").rename(
                    columns={"equity": "扣费后净值", "gross_equity": "不扣费净值"}
                )
                equity_frame.index.name = "日期"
                st.line_chart(equity_frame[["扣费后净值", "不扣费净值"]])
                st.caption(
                    "净值＝每 1 元本金变成了多少钱，起点都是 1。两条线的差距就是手续费、印花税和滑点吃掉的部分。"
                )
            if backtest_result.get("open_position"):
                st.write({"模拟结束时还拿着没卖的（已经算进上面的净值）": backtest_result["open_position"]})
            if backtest_result.get("trades"):
                st.dataframe(
                    pd.DataFrame(backtest_result["trades"]).tail(30).rename(
                        columns={
                            "entry_date": "买入日", "exit_date": "卖出日", "entry_price": "买入价", "exit_price": "卖出价",
                            "gross_return": "不扣费收益", "net_return": "扣费后收益",
                            "holding_days": "持有几天", "exit_reason": "卖出原因",
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
        else:
            st.warning(backtest_result.get("status"))

    with st.expander("配对观察：找两只长期同涨同跌的股票"):
        left, right = st.columns(2)
        code_a = left.text_input("股票 A 代码", max_chars=6, key="pair_a")
        code_b = right.text_input("股票 B 代码", max_chars=6, key="pair_b")
        if st.button("检验配对关系", key="pair_test"):
            try:
                with loading_task("配对统计分析", exclusive=True) as task:
                    task.step("1 / 3 · 获取股票 A 历史", 0, 3)
                    history_a, meta_a = service.get_history(code_a, count=1000, adjusted=True)
                    task.step("2 / 3 · 获取股票 B 历史", 1, 3)
                    history_b, meta_b = service.get_history(code_b, count=1000, adjusted=True)
                    if any("过期" in meta.get("status", "") for meta in (meta_a, meta_b)):
                        raise MarketDataError("配对历史已过期，不生成套利信号。")
                    task.step("3 / 3 · 检验协整关系与价差", 2, 3)
                    pair_result = analyze_pair(completed_daily_history(history_a), completed_daily_history(history_b), code_a, code_b)
                    st.session_state["pair_result"] = pair_result
            except (MarketDataError, ValueError) as exc:
                st.error(str(exc))
        pair_result = st.session_state.get("pair_result")
        if pair_result:
            st.caption(
                "思路：两只股票如果长期同涨同跌，暂时拉开明显差距时，可以押注差距收回去。"
                "这需要能融券做空其中一只，普通账户不一定做得到。"
            )
            st.write(pair_result.get("status"))
            if pair_result.get("current_z_score") is not None:
                show_plain_numbers([
                    ("差距拉大到平时几倍", _fmt(pair_result.get("current_z_score"), 2),
                     "两只股票当前的差距相当于平时的几倍：0 表示和平时一样，超过 ±2 才值得留意（Z 值）。"),
                    ("两只股票怎么配比", _fmt(pair_result.get("hedge_ratio_kalman"), 3),
                     "买 1 份 A 要同时卖空多少份 B，才能抵消掉大盘涨跌的影响（对冲比例）。"),
                    ("现在怎么做", pair_result.get("signal") or "暂无",
                     "差距偏大就反过来做，偏小就正着做；回到 0 附近就平仓观察。"),
                ])
            with st.expander("展开原始检验数据（beta、协整 p 值等，给懂行的人看）"):
                st.json(pair_result)



