"""个股查询、因子和中低频模型分析页面。"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import settings
from services.backtest import run_backtest, save_backtest_run
from services.database import DatabaseError
from services.market_data import MarketDataError, completed_daily_history, history_is_current
from services.rank_model import SCORE_KIND_RANK, objective_label
from services.stock_scoring import analyze_stock
from services.statistical_arbitrage import analyze_pair
from services.ui_helpers import format_number as _fmt
from services.ui_helpers import show_backtest_metrics as _show_backtest_metrics
from services.ui_helpers import show_candles as _show_candles
from services.ui_helpers import loading_task, show_empty_state, show_page_header
from services.runtime import get_market_data, get_repository

st.set_page_config(page_title="个股分析 | A 股量化研究", layout="wide")
show_page_header(
    "个股分析",
    "输入股票代码或名称，从行情走势、传统因子到模型结果，逐步了解一只股票。",
    "研究工作台 / 个股分析",
)


service = get_market_data()
if st.query_params.get("code") and st.session_state.get("query_code") != st.query_params.get("code"):
    st.session_state["stock_query"] = st.query_params.get("code", "")
    st.session_state["query_code"] = st.query_params.get("code")
    st.session_state["selected_code"] = st.query_params.get("code")
    st.session_state["search_results"] = pd.DataFrame()

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
    selected_label = st.selectbox("匹配结果", labels)
    selected_code = selected_label.split()[0]
    if st.button("分析所选股票", type="primary"):
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

    price_columns = st.columns(7)
    price_columns[0].metric("最新可用价", _fmt(quote.get("price"), 2))
    price_columns[1].metric("涨跌幅", _fmt(quote.get("change_pct"), 2, "%"))
    price_columns[2].metric("成交量", _fmt(quote.get("volume_lots"), 0, " 手"))
    price_columns[3].metric("成交额", _fmt(quote.get("amount_yuan"), 0, " 元"))
    price_columns[4].metric("开盘", _fmt(quote.get("open"), 2))
    price_columns[5].metric("最高 / 最低", f"{_fmt(quote.get('high'), 2)} / {_fmt(quote.get('low'), 2)}")
    price_columns[6].metric("昨收", _fmt(quote.get("previous_close"), 2))

    chart_tab, factor_tab, model_tab, signal_tab = st.tabs(["行情图表", "因子分析", "模型结果", "交易参考"])
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
        st.metric("时序多因子评分", _fmt(analysis.get("factor_score"), 1, " / 100"))
        factors = analysis.get("factor_details", [])
        if factors:
            factor_frame = pd.DataFrame(factors).rename(
                columns={"factor": "因子", "value": "当前值", "z_score": "时序 Z 值", "score": "子分", "weight": "权重"}
            )
            st.dataframe(factor_frame, hide_index=True, use_container_width=True)
        else:
            st.info("当前历史长度不足以计算稳定的因子评分。")
        latest = analysis.get("latest_features", {})
        st.write(
            {
                "5 日收益": latest.get("momentum_5"),
                "20 日收益": latest.get("momentum_20"),
                "60 日收益": latest.get("momentum_60"),
                "20 日年化波动率": latest.get("volatility_20"),
                "ATR / 价格": latest.get("atr_pct"),
                "相对沪深 300 20 日表现": latest.get("relative_strength_20"),
            }
        )
        st.caption(analysis.get("fundamental_status", ""))
    with model_tab:
        model_rows = []
        importance_rows = []
        for horizon, models in analysis["models"].items():
            for item in models:
                metrics = item.get("oos_metrics") or {}
                ranking = item.get("score_kind") == SCORE_KIND_RANK
                model_rows.append(
                    {
                        "模型": item["model"],
                        "预测周期": f"{horizon} 个交易日",
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
        st.dataframe(pd.DataFrame(model_rows), hide_index=True, use_container_width=True)
        st.caption(
            "排序目标下模型预测值只是同日截面内的相对排序分，判断依据是 NDCG 相对随机基线的提升；只有回归目标下的预测值才是相对沪深 300 的预期超额收益。"
            "Z分数按同日股票池标准化，不换算绝对价格。请先在量化选股扫描至少5只股票建立共享截面模型。"
        )
        st.caption(analysis.get("cross_section_scope", ""))
        if importance_rows:
            st.markdown("**20 日模型特征重要性**")
            st.dataframe(pd.DataFrame(importance_rows), hide_index=True, use_container_width=True)
        st.warning("基线、Rank IC或Top/Bottom分组检验不通过时不输出预测评分；分组持有期收益含重叠，不代表组合回测净收益。")
    with signal_tab:
        plan = analysis.get("trade_plan", {})
        st.write(f"**状态：** {plan.get('status', '不可用')}")
        if plan.get("buy_zone"):
            st.write(
                {
                    "参考买入区间": plan.get("buy_zone"),
                    "参考价（非成交价）": plan.get("reference_price"),
                    "入场条件": plan.get("entry_trigger"),
                    "初始止损参考": plan.get("initial_stop"),
                    "止盈参考": plan.get("take_profit"),
                    "移动止盈 / 趋势退出": plan.get("trailing_exit"),
                    "持仓周期": plan.get("holding_period"),
                    "生成时间 / 有效期": f"{plan.get('generated_at')} / {plan.get('valid_until')}",
                    "失效条件": plan.get("invalidation"),
                    "计算依据": plan.get("calculation"),
                }
            )
            st.warning(plan.get("risk_note", ""))
        else:
            st.info("价格或 ATR 数据不足，未生成交易价格。")

    st.divider()
    st.subheader("滚动样本外回测")
    st.caption("回测使用当时训练验证通过的模型和页面相同的交易规则；包含逐日持仓净值及T+1/停牌/涨跌停近似限制。")
    if st.button("运行并保存回测", key="run_backtest"):
        try:
            with loading_task("滚动样本外回测", "回测会分时段训练模型，计算交易与持仓净值，耗时通常长于单次分析。") as task:
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
                equity_frame = pd.DataFrame(backtest_result["daily_equity"]).set_index("date")
                st.line_chart(equity_frame[["equity", "gross_equity"]])
            if backtest_result.get("open_position"):
                st.write({"期末未平仓（已计入净值）": backtest_result["open_position"]})
            if backtest_result.get("trades"):
                st.dataframe(pd.DataFrame(backtest_result["trades"]).tail(30), hide_index=True, use_container_width=True)
        else:
            st.warning(backtest_result.get("status"))

    with st.expander("统计套利：检验一组配对股票"):
        left, right = st.columns(2)
        code_a = left.text_input("股票 A 代码", max_chars=6, key="pair_a")
        code_b = right.text_input("股票 B 代码", max_chars=6, key="pair_b")
        if st.button("检验配对关系", key="pair_test"):
            try:
                with loading_task("配对统计分析") as task:
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
            st.write(pair_result.get("status"))
            st.json(pair_result)



