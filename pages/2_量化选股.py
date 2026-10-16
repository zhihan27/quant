"""按股票池扫描真实行情并展示可解释候选结果。"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from services.database import DatabaseError
from services.runtime import get_market_data, get_repository

from config import settings
from services.backtest import run_backtest
from services.exposures import load_exposures
from services.extra_factors import build_extra_factors
from services.market_data import MarketDataError, completed_daily_history, history_is_current
from services.scanning import candidate_codes as _candidate_codes
from services.scanning import rank_candidates as _rank_candidates
from services.scanning import validated_return as _validated_return
from services.stock_scoring import analyze_stock
from services.ml_pipeline import load_context, train_market_context
from services.portfolio_backtest import run_portfolio_strategy
from services.rank_model import objective_label
from services.ui_helpers import (
    format_number,
    loading_task,
    show_backtest_metrics,
    show_conclusion,
    show_empty_state,
    show_page_header,
)

st.set_page_config(page_title="量化选股 | A 股量化研究", layout="wide")
show_page_header(
    "量化选股",
    "设定股票范围，一次扫描整池，先看筛出几只候选和各自的评分，再看每只凭什么入选。",
    "研究工作台 / 量化选股",
)


def _zone_text(zone: Any) -> str:
    """买入区间是元组，直接用会连括号和引号一起显示，这里转成「下沿 ~ 上沿」。"""
    if not zone:
        return "—"
    try:
        return f"{float(zone[0]):.2f} ~ {float(zone[1]):.2f}"
    except (TypeError, ValueError, IndexError):
        return "—"


def _percent_text(value: Any) -> str:
    """回测里的比例都是小数（0.12 表示 12%），统一转成百分比文字再展示。"""
    return "—" if value is None else format_number(value * 100, 2, "%")


def _cost_text(costs: Any) -> str:
    """把费率字典整理成一行白话，避免页面上直接露出英文字段名。"""
    if not costs:
        return "—"
    return (
        f"佣金每边 {_percent_text(costs.get('commission_rate_each_side'))}，"
        f"印花税只在卖出时收 {_percent_text(costs.get('stamp_duty_sell_side'))}，"
        f"滑点每边 {_percent_text(costs.get('slippage_each_side'))}"
    )


def _score_kind_label(score_kind: Any) -> str:
    """模型分数含义的标识转成一句人话，未登记的取值原样返回，不猜。"""
    if not score_kind:
        return "—"
    return {
        "rank_score": "同一天内的相对排名（百分位）",
        "excess_return": "预期比大盘多赚多少",
    }.get(str(score_kind), str(score_kind))


service = get_market_data()
with st.container(border=True):
    st.subheader("筛选条件")
    scope_columns = st.columns([3, 1])
    pool_mode = scope_columns[0].selectbox("股票范围", ["沪深京全部 A 股", "沪深主板", "创业板", "科创板", "北交所", "用户自定义"])
    scan_limit = scope_columns[1].number_input(
        "表格最多显示几行", min_value=1, max_value=100, value=20, step=1,
        help="只影响页面显示多少行，实际扫描的股票数量不受它限制。",
    )
    custom_codes = st.text_input("自定义代码（逗号或空格分隔）", placeholder="例：600000,000001", disabled=pool_mode != "用户自定义")
    include_backtest = st.checkbox(
        "同时看每只股票的历史模拟回测（全市场会很慢）", value=False,
        help="对每只候选股票单独回放一遍历史买卖，用来判断它的规则在过去是否有效，会明显拉长扫描时间。",
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
    portfolio_columns = st.columns([2, 1, 1, 1], vertical_alignment="bottom")
    run_portfolio = portfolio_columns[0].checkbox(
        "扫描后模拟一个组合（按评分自动买卖，含 T+1 与交易费用）",
        value=False,
        help="用本次扫描的结果，每天买入评分最高的前 N 名等权持有，按 A 股 T+1、涨跌停和停牌规则模拟成交，并逐笔扣掉佣金、印花税和滑点。",
    )
    portfolio_top_n = portfolio_columns[1].number_input("组合同时持有几只", min_value=1, max_value=100, value=int(settings.PORTFOLIO_TOP_N), step=1, disabled=not run_portfolio)
    max_position_weight = portfolio_columns[2].number_input(
        "单只股票最多占比", min_value=0.01, max_value=1.0, value=float(settings.MAX_POSITION_WEIGHT),
        step=0.01, format="%.2f", disabled=not run_portfolio,
        help="单只股票占组合的上限（1.00 表示 100%），用来避免押注过度集中。",
    )
    rebalance_every = portfolio_columns[3].number_input(
        "每隔几天调仓一次", min_value=1, max_value=60, value=1, step=1, disabled=not run_portfolio,
        help="调仓越勤，交易费用越多；填 1 表示每个交易日都按最新评分换股。",
    )
    st.caption("扫描会先下载股票池的历史行情，再训练一个覆盖全池的模型。参与训练的股票太少（不足 5 只）或数据缺失太多（覆盖不到本次股票池的 80%）时不会给出评分。")

if st.button("开始扫描", type="primary"):
    with loading_task("获取完整股票池", "首次获取或启用备用数据源时会逐页核对完整股票池，通常需要一至两分钟。") as task:
        task.step("正在读取缓存或逐页获取股票目录，并核对完整性…")
        candidates, source_meta = _candidate_codes(pool_mode, custom_codes, int(scan_limit), service)
        if not candidates:
            task.fail("股票池获取失败，请查看下方错误信息。")
    if not candidates:
        st.error(source_meta.get("error", "当前股票列表为空；可切换为用户自定义代码扫描。"))
        st.session_state["recommendations"] = []
    else:
        if source_meta.get("warning") or source_meta.get("is_stale"):
            st.warning(source_meta.get("warning", "当前使用完整过期股票池缓存，行情非实时。"))
        st.info(
            f"源端总数 {source_meta.get('source_total_symbols', len(candidates)):,}；"
            f"返回行数 {source_meta.get('source_rows', len(candidates)):,}；"
            f"去重股票池 {source_meta.get('source_unique_symbols', len(candidates)):,}；"
            f"有效股票 {source_meta.get('valid_symbols', len(candidates)):,}；"
            f"排除 {source_meta.get('excluded_symbols', 0):,}；失败 {source_meta.get('failed_symbols', 0):,}；"
            f"失败页 {source_meta.get('failed_pages', 0):,}；本次扫描 {len(candidates):,}；"
            f"来源 {source_meta.get('source', source_meta.get('status'))}；"
            f"更新时间 {source_meta.get('fetched_at', '用户输入')}。"
        )
        with loading_task("扫描与模型分析", f"本次共 {len(candidates):,} 只股票。进度按已处理数量计算，单只股票训练耗时可能不同。") as task:
            task.step("正在准备沪深 300 基准行情…")
            benchmark = None
            benchmark_status = "沪深 300 指数不可用；相对市场因子未计算。"
            try:
                benchmark, benchmark_meta = service.get_history(
                    settings.MARKET_INDEX_CODE, count=1200, adjusted=True, market_id=1
                )
                if "过期" in benchmark_meta.get("status", ""):
                    raise MarketDataError("沪深300历史已过期，不能训练超额收益标签。")
                benchmark = completed_daily_history(benchmark)
                benchmark_status = "沪深 300 指数历史数据已获取。"
            except MarketDataError as exc:
                benchmark = None
                benchmark_status = f"沪深 300 指数不可用；相对市场因子未计算：{exc}"
            if benchmark is None:
                st.warning(benchmark_status)
            records = []
            for index, candidate in enumerate(candidates):
                code = candidate["code"]
                task.step(f"正在分析 {code} · 已完成 {index:,} / {len(candidates):,} 只", index, len(candidates))
                try:
                    quote = service.get_scan_quote(candidate, source_meta)
                    if not quote.get("quote_valid", True):
                        records.append(
                            {"code": code, "name": candidate.get("name", code), "error": "当前仅有过期行情缓存，未生成模型预测"}
                        )
                        continue
                    name = quote.get("name", candidate.get("name", code))
                    if re.search(r"ST|退", str(name), flags=re.IGNORECASE):
                        records.append({"code": code, "name": name, "error": "ST 或退市整理股票已排除"})
                        continue
                    volume = quote.get("volume_lots")
                    if volume is None or volume <= 0:
                        records.append({"code": code, "name": name, "error": "停牌或无成交量，已排除"})
                        continue
                    history, history_meta = service.get_history(code, count=1200, adjusted=True)
                    history = completed_daily_history(history)
                    if not history_is_current(history, quote, history_meta):
                        records.append(
                            {"code": code, "name": name, "error": "历史行情未更新至报价对应的完整交易日，未生成模型预测"}
                        )
                        continue
                    if len(history) < settings.MINIMUM_SCAN_HISTORY_BARS:
                        records.append(
                            {
                                "code": code,
                                "name": name,
                                "error": f"历史样本不足 {settings.MINIMUM_SCAN_HISTORY_BARS} 根日线，已排除",
                            }
                        )
                        continue
                    limit_rate = 0.30 if code.startswith(("4", "8", "92")) else (0.20 if code.startswith(("300", "301", "688", "689")) else 0.10)
                    change_pct = quote.get("change_pct")
                    entry_trade_status = "可正常评估"
                    if change_pct is not None and abs(change_pct) >= limit_rate * 100 * 0.98:
                        entry_trade_status = "接近涨跌停，成交可行性需单独确认"
                    records.append(
                        {
                            "code": code,
                            "name": name,
                            "quote": quote,
                            "history_meta": history_meta,
                            "history": history,
                            "entry_trade_status": entry_trade_status,
                        }
                    )
                except Exception as exc:
                    records.append(
                        {
                            "code": code,
                            "name": candidate.get("name", code),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                finally:
                    task.step(f"已完成 {index + 1:,} / {len(candidates):,} 只", index + 1, len(candidates))
            ready = [item for item in records if not item.get("error")]
            exposures = None
            if enable_neutralization:
                task.step("正在获取流通市值与申万一级行业暴露…")
                try:
                    exposures, exposure_meta = load_exposures(service, scope="all", progress=task.step)
                    st.caption(f"暴露数据：{exposure_meta.get('status')}；{exposure_meta.get('limitation', '')}")
                except (OSError, ValueError, ImportError, MarketDataError) as exc:
                    exposures = None
                    st.warning(f"市值与行业暴露不可用，本次不做中性化：{type(exc).__name__}: {exc}")
            context = {}
            extra_factors: dict[str, Any] = {}
            extra_store = None
            if use_extra_factors:
                task.step("正在装配筹码、日内与 PIT 财报因子…")
                extra_store = service.panels
                extra_factors, extra_meta = build_extra_factors(
                    extra_store, [item["code"] for item in ready],
                    histories={item["code"]: item["history"] for item in ready}, progress=task.step,
                )
                st.caption(
                    f"新增因子：{extra_meta.get('status')}；"
                    f"日内 {extra_meta.get('intraday', {}).get('codes_computed', 0)} 只、"
                    f"财报 {extra_meta.get('financial', {}).get('codes', 0)} 只。"
                )
            try:
                context = train_market_context(
                    {item["code"]: item["history"] for item in ready}, benchmark,
                    metadata={**source_meta, "pool_mode": pool_mode}, progress=task.step,
                    exposures=exposures, objective=objective_mode, window=int(training_window),
                    extra_factors=extra_factors or None, store=extra_store,
                )
                st.info(f"共享模型上下文 {context['context_id'][:12]} · 截面日期 {context['reference_date']} · {len(ready)}只有效股票")
                st.caption(
                    f"训练目标 {objective_label(context['objective_mode'])}；滚动窗口 "
                    f"{context['training_window_days'] or '不限'}个交易日；中性化状态：{context['neutralization']['status']}。"
                )
                extra_report = context.get("extra_factors") or {}
                if extra_report.get("families"):
                    st.caption(f"新增因子并入结果：{extra_report.get('status')}")
                st.caption(context["limitation"])
            except (ValueError, OSError, ImportError) as exc:
                st.warning(f"截面模型不可用：{exc}；本次只展示传统因子观察。")
            st.session_state["portfolio_result"] = None
            if run_portfolio and context:
                with loading_task("Top-N 组合构建与摩擦成本回测", "滚动重训截面排序模型并对每日前 N 名调仓，撮合包含 T+1、涨跌停与全部交易费用，耗时随股票池扩大而增加。") as portfolio_task:
                    _, portfolio_panel = load_context(include_panel=True)
                    if portfolio_panel is None or portfolio_panel.empty:
                        st.warning("未能读取本次截面面板，组合回测已跳过。")
                    else:
                        try:
                            st.session_state["portfolio_result"] = run_portfolio_strategy(
                                {item["code"]: item["history"] for item in ready},
                                portfolio_panel,
                                benchmark=benchmark,
                                horizon=20,
                                top_n=int(portfolio_top_n),
                                max_weight=float(max_position_weight),
                                rebalance_every=int(rebalance_every),
                                objective=objective_mode,
                                window=int(training_window),
                                universe_limit=settings.PORTFOLIO_BACKTEST_MAX_STOCKS,
                                progress=lambda message: portfolio_task.step(message),
                            )
                        except (ValueError, OSError, ImportError) as exc:
                            st.warning(f"组合回测不可用：{type(exc).__name__}: {exc}")
            for item in ready:
                code, history, quote = item["code"], item["history"], item["quote"]
                item["analysis"] = analyze_stock(
                    history, current_price=quote.get("price"), benchmark=benchmark,
                    horizons=(5, 20, 60), code=code,
                    as_of=quote.get("quote_time") or quote.get("fetched_at"),
                    quote_status=quote.get("data_status"), market_context=context,
                )
                item["features"] = item["analysis"]["latest_features"]
                item["analysis"].pop("features", None)
                item["backtest"] = None
                if include_backtest and context:
                    try:
                        task.step(f"正在重新训练历史截面并回测 {code}，此步骤可能较久…")
                        raw_history, _ = service.get_history(code, count=1200, adjusted=False)
                        item["backtest"] = run_backtest(
                            completed_daily_history(raw_history), code=code, benchmark=benchmark, adjusted_history=history,
                        )
                    except (MarketDataError, ValueError) as exc:
                        item["backtest"] = {"status": f"回测不可用：{exc}", "metrics": None}
                # 结果中只保留展示及追溯数据，释放扫描期间收集的逐股历史。
                item.pop("history", None)
            task.step("所有股票处理结束，正在汇总评分与排序…", len(candidates), len(candidates))
            st.session_state["recommendations"] = _rank_candidates(records)
            st.session_state["scan_source"] = source_meta
            st.session_state["scan_stats"] = {
                "scanned": len(candidates),
                "scored": sum(1 for item in records if not item.get("error")),
                "failed": sum(1 for item in records if item.get("error")),
            }

recommendations = st.session_state.get("recommendations", [])
if not recommendations:
    show_empty_state("配置范围，开始发现候选股票", "首次可使用少量自定义代码体验。全市场扫描包含逐股行情获取与模型训练，耗时取决于股票数量和网络。")
if recommendations:
    rows = []
    detail_rows = []
    usable = []
    failed_rows = []
    for item in recommendations:
        if item.get("error"):
            failed_rows.append({"代码": item["code"], "名称": item["name"], "状态": item["error"]})
            continue
        usable.append(item)
        quote = item["quote"]
        models = item["analysis"]["models"].get(20, [])
        lightgbm = next((model for model in models if model["model"] == "LightGBM"), {})
        xgboost = next((model for model in models if model["model"] == "XGBoost"), {})
        plan = item["analysis"].get("trade_plan", {})
        backtest_metrics = (item.get("backtest") or {}).get("metrics") or {}
        # 默认表只留看得懂的部分；技术字段统一进明细表，避免首屏堆满行话。
        rows.append(
            {
                "综合评分": item.get("cross_section_score"),
                "代码": item["code"],
                "名称": item["name"],
                "最新价": quote.get("price"),
                "涨跌幅": quote.get("change_pct"),
                "参考买入区间": _zone_text(plan.get("buy_zone")),
                "止损参考": plan.get("initial_stop"),
                "交易状态": item.get("entry_trade_status", "未评估"),
                "推荐理由": item.get("reason"),
                "主要风险": item.get("risk"),
            }
        )
        detail_rows.append(
            {
                "代码": item["code"],
                "名称": item["name"],
                "行情时间": quote.get("quote_time") or quote.get("fetched_at"),
                "数据状态": quote.get("data_status"),
                "时序因子分": item["analysis"].get("factor_score"),
                "LightGBM 20日预测值": _validated_return(lightgbm),
                "LightGBM 截面Z分数": lightgbm.get("prediction_zscore"),
                "LightGBM Rank IC": (lightgbm.get("oos_metrics") or {}).get("rank_ic"),
                "LightGBM 排名": item.get("lightgbm_rank"),
                "XGBoost 20日预测值": _validated_return(xgboost),
                "XGBoost 截面Z分数": xgboost.get("prediction_zscore"),
                "XGBoost Rank IC": (xgboost.get("oos_metrics") or {}).get("rank_ic"),
                "XGBoost 排名": item.get("xgboost_rank"),
                "退出条件": plan.get("trailing_exit"),
                "样本外回测收益": backtest_metrics.get("total_return"),
                "回测状态": (item.get("backtest") or {}).get("status", "未计算"),
                "信号生成时间": plan.get("generated_at"),
                "截面日期": item["analysis"].get("prediction_reference_date"),
                "模型上下文": item["analysis"].get("market_context_id"),
            }
        )
    scan_stats = st.session_state.get("scan_stats", {})
    scored_rows = [row for row in rows if row.get("综合评分") is not None]
    best = max(scored_rows, key=lambda row: row["综合评分"]) if scored_rows else None
    show_conclusion(
        f"本次扫描 {scan_stats.get('scanned', 0):,} 只股票，筛出 {len(rows):,} 只候选"
        + (f"，其中评分最高的是 {best['名称']}（{best['代码']}）。" if best else "。")
        + " 评分是“同一天里相对更值得看”的排名，不是涨跌预测，也不保证赚钱。"
    )
    st.caption(
        f"扫描明细：有效评分 {scan_stats.get('scored', 0):,} 只；失败或排除 {scan_stats.get('failed', 0):,} 只。"
    )
    st.dataframe(
        pd.DataFrame(rows[: int(scan_limit)]),
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
    with st.expander(f"展开技术指标明细（{len(detail_rows):,} 行，给懂行的人看）"):
        st.caption(
            "截面Z分数＝这只股票的模型分比当天全市场平均值高出多少（0 是平均，正数更看好）；"
            "Rank IC＝样本外每天“模型排名”和“真实涨跌排名”的相关性，越高说明模型排序越靠谱；"
            "20日预测值在排序目标下只是同一天里的相对排名百分位，只有回归目标下才是预期超额收益；"
            "时序因子分只看这只股票自己的历史，和全市场比出来的综合评分不是一回事。"
        )
        st.dataframe(pd.DataFrame(detail_rows[: int(scan_limit)]), hide_index=True, width="stretch")
    if failed_rows:
        with st.expander(f"查看失败或排除股票（{len(failed_rows):,}）"):
            st.dataframe(pd.DataFrame(failed_rows), hide_index=True, use_container_width=True)
    history_summary = service.history_fetch_summary()
    st.caption(
        f"历史行情状态：已成功/缓存 {history_summary['success_count']:,}；"
        f"过期缓存 {history_summary['stale_count']:,}；获取失败 {history_summary['failure_count']:,}。"
    )
    if history_summary["by_stock"]:
        with st.expander(f"查看逐股历史行情状态与接口成功率（{len(history_summary['by_stock']):,}）"):
            st.dataframe(pd.DataFrame(history_summary["by_stock"]), hide_index=True, use_container_width=True)
    if history_summary["failures"]:
        with st.expander("查看逐股历史行情失败原因"):
            st.dataframe(pd.DataFrame(history_summary["failures"]), hide_index=True, use_container_width=True)
    with st.expander("查看接口请求及代理诊断"):
        proxy = service.proxy_configuration()
        st.caption(f"检测到系统代理：{proxy or '未设置'}；实际连接路由：{service.network.status()}。详细请求记录写入 logs/market_data.log。")
        request_rows = service.request_diagnostics()
        if request_rows:
            st.dataframe(pd.DataFrame(request_rows), hide_index=True, use_container_width=True)
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

portfolio_result = st.session_state.get("portfolio_result")
if portfolio_result:
    st.divider()
    st.subheader("模拟组合回测（按评分自动买卖）")
    portfolio_metrics = portfolio_result.get("metrics")
    if not portfolio_metrics:
        st.warning(portfolio_result.get("status", "组合回测未产生结果。"))
    else:
        total_return = portfolio_metrics.get("total_return")
        benchmark_return = portfolio_result.get("benchmark_buy_hold_return")
        gap_text = ""
        if total_return is not None and benchmark_return is not None:
            diff = (total_return - benchmark_return) * 100
            gap_text = (
                f"；同期沪深 300 买入不动是 {_percent_text(benchmark_return)}，"
                f"{'跑赢' if diff >= 0 else '跑输'} {abs(diff):.2f} 个百分点"
            )
        show_conclusion(
            f"历史上完全按这套规则自动买卖，模拟总收益 {_percent_text(total_return)}{gap_text}。"
            "这是回放过去、并且扣掉了手续费和滑点，不代表以后也会这样。"
        )
        show_backtest_metrics(portfolio_metrics, benchmark_return)
        portfolio = portfolio_result.get("portfolio", {})
        universe = portfolio_result.get("universe", {})
        score_range = portfolio_result.get("score_range", {})
        universe_note = universe.get("status") or f"股票池未超过 {universe.get('limit')} 只上限"
        st.caption(
            f"规则：每个交易日按评分买前 {portfolio.get('top_n')} 名、等权持有，"
            f"单只目标仓位 {_percent_text(portfolio.get('per_stock_weight'))}、上限 {_percent_text(portfolio.get('max_weight'))}；"
            f"每 {portfolio_result.get('rebalance_every')} 个交易日调仓一次。"
            f"打分区间 {score_range.get('start')} 至 {score_range.get('end')}，共 {score_range.get('dates')} 个交易日；"
            f"可用股票 {universe.get('kept')} 只（{universe_note}）。"
        )
        st.write(
            {
                "平均同时持几只": format_number(portfolio.get("average_holdings"), 1, " 只"),
                "资金投入比例": _percent_text(portfolio.get("invested_ratio")),
                "留作现金的比例": _percent_text(portfolio.get("cash_ratio")),
                "单股上限有没有被触发": "是" if portfolio.get("cap_binding") else "否",
                "已卖出的交易笔数": portfolio_metrics.get("trades"),
                "想买没买到的笔数（涨停或停牌）": portfolio_result.get("unfilled_buys"),
                "想卖没卖掉的笔数（跌停或停牌）": portfolio_result.get("unfilled_sells"),
                "整个股票池平均买一遍的收益": _percent_text(portfolio_result.get("universe_equal_weight_return")),
                "交易费用": _cost_text(portfolio_result.get("costs")),
                "模型版本": portfolio_result.get("model_version"),
            }
        )
        st.caption("模拟成交时遵守的规则：" + "；".join(portfolio_result.get("constraints", [])))
        open_positions = portfolio_result.get("open_positions") or []
        if open_positions:
            st.markdown("**回测结束时还拿着没卖的**")
            st.dataframe(
                pd.DataFrame(open_positions).rename(
                    columns={
                        "code": "代码",
                        "entry_date": "买入日",
                        "entry_price": "买入价",
                        "shares": "股数",
                        "market_value": "当前市值",
                        "unrealized_return": "浮动盈亏",
                    }
                ),
                hide_index=True,
                width="stretch",
                column_config={
                    "浮动盈亏": st.column_config.NumberColumn(
                        "浮动盈亏", help="比例数：0.05 表示浮盈 5%，已按卖出费用折算。"
                    ),
                },
            )
        trades = portfolio_result.get("trades") or []
        if trades:
            with st.expander(f"展开每一笔买卖明细（{len(trades):,} 笔）"):
                st.caption("收益都是比例数：0.05 表示赚 5%。")
                st.dataframe(
                    pd.DataFrame(trades).rename(
                        columns={
                            "entry_date": "买入日",
                            "exit_date": "卖出日",
                            "code": "代码",
                            "entry_price": "买入价",
                            "exit_price": "卖出价",
                            "gross_return": "不计费用收益",
                            "net_return": "扣费后收益",
                            "holding_days": "持有天数",
                            "exit_reason": "卖出原因",
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                )
        daily_equity = portfolio_result.get("daily_equity") or []
        if daily_equity:
            equity_frame = pd.DataFrame(daily_equity).set_index("date")[["equity", "gross_equity"]]
            equity_frame = equity_frame.rename(columns={"equity": "扣费后净值", "gross_equity": "不扣费净值"})
            equity_frame.index.name = "日期"
            st.line_chart(equity_frame)
            st.caption(
                "两条线都从 1.0 开始：扣费后净值算进了佣金、印花税和滑点，不扣费净值是同一条买卖路径但不计费用，"
                "两条线之间的差距就是交易成本吃掉的收益。"
            )
        training_log = portfolio_result.get("training_log") or []
        if training_log:
            with st.expander(f"展开每段重新训练的检验结果（{len(training_log):,} 段）"):
                # 摊平每段里每个模型的检验数值，直接看出是哪条门槛拦住了打分。
                log_rows = [
                    {
                        "打分起始日": block.get("prediction_start"),
                        "训练数据截止": block.get("trained_through"),
                        "模型": model.get("model"),
                        "训练目标": objective_label(model["objective"]) if model.get("objective") else "—",
                        "分数含义": _score_kind_label(model.get("score_kind")),
                        "是否通过验证": model.get("status"),
                        "训练交易日数": model.get("training_dates"),
                        "Rank IC": model.get("rank_ic"),
                        "NDCG 比值": model.get("ndcg_ratio"),
                        "MAE": model.get("mae"),
                        "验证规则": model.get("validation_rule"),
                    }
                    for block in training_log
                    for model in block.get("models", [])
                ]
                st.caption(
                    "每一段只用该起始日之前的历史重新训练，不会用到未来数据。"
                    "Rank IC 看排序是否靠谱；排序目标按「NDCG 与随机猜的比值」判定，回归目标看 MAE 是否优于「直接猜中位数」的基线。"
                    "只有通过验证的模型才参与打分。"
                )
                st.dataframe(pd.DataFrame(log_rows), hide_index=True, width="stretch")
