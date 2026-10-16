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
from services.market_data import MarketDataError, completed_daily_history, history_is_current
from services.scanning import candidate_codes as _candidate_codes
from services.scanning import rank_candidates as _rank_candidates
from services.scanning import validated_return as _validated_return
from services.stock_scoring import analyze_stock
from services.ml_pipeline import load_context, train_market_context
from services.portfolio_backtest import run_portfolio_strategy
from services.rank_model import objective_label
from services.ui_helpers import loading_task, show_backtest_metrics, show_empty_state, show_page_header

st.set_page_config(page_title="量化选股 | A 股量化研究", layout="wide")
show_page_header(
    "量化选股",
    "设定股票范围，结合传统因子与模型验证筛选候选，查看每只股票的入选依据。",
    "研究工作台 / 量化选股",
)


service = get_market_data()
with st.container(border=True):
    st.subheader("筛选条件")
    scope_columns = st.columns([3, 1])
    pool_mode = scope_columns[0].selectbox("股票范围", ["沪深京全部 A 股", "沪深主板", "创业板", "科创板", "北交所", "用户自定义"])
    scan_limit = scope_columns[1].number_input("展示前 N 个结果", min_value=1, max_value=100, value=20, step=1)
    custom_codes = st.text_input("自定义代码（逗号或空格分隔）", placeholder="例：600000,000001", disabled=pool_mode != "用户自定义")
    include_backtest = st.checkbox("同时计算每只股票的滚动样本外模型回测（全市场耗时较长）", value=False)
    training_columns = st.columns([2, 1, 2], vertical_alignment="bottom")
    objective_mode = training_columns[0].selectbox(
        "训练目标",
        settings.RANK_OBJECTIVE_MODES,
        index=settings.RANK_OBJECTIVE_MODES.index(settings.DEFAULT_OBJECTIVE_MODE),
        format_func=objective_label,
        help="排序目标只在同一交易日的截面内比胜负，避免回归目标在全市场上涨时给出全线看多的分数；自动择优会同时训练两种目标并按样本外 Rank IC 选择。",
    )
    training_window = training_columns[1].number_input(
        "滚动训练窗口（交易日）",
        min_value=0,
        max_value=2000,
        value=int(settings.TRAINING_WINDOW_DAYS),
        step=50,
        help="每次重训只使用最近 N 个交易日的样本，以适应 A 股风格切换；0 表示使用全部可用历史。",
    )
    enable_neutralization = training_columns[2].checkbox(
        "启用市值与行业中性化（申万一级）",
        value=True,
        help="对每个交易日的截面按流通市值对数与申万一级行业哑变量做回归，取残差作为纯净 Alpha 因子；需要联网获取暴露数据，覆盖率过低时自动退回截面 Z 分数因子。",
    )
    portfolio_columns = st.columns([2, 1, 1, 1], vertical_alignment="bottom")
    run_portfolio = portfolio_columns[0].checkbox(
        "扫描后运行 Top-N 组合回测（含 T+1 与交易费用）",
        value=False,
        help="对本次股票池滚动重训截面排序模型，每日取前 N 名等权持有，按 T+1、涨跌停与停牌约束撮合并逐笔计入佣金、印花税和滑点。",
    )
    portfolio_top_n = portfolio_columns[1].number_input("组合持仓数 Top N", min_value=1, max_value=100, value=int(settings.PORTFOLIO_TOP_N), step=1, disabled=not run_portfolio)
    max_position_weight = portfolio_columns[2].number_input("单股仓位上限", min_value=0.01, max_value=1.0, value=float(settings.MAX_POSITION_WEIGHT), step=0.01, format="%.2f", disabled=not run_portfolio)
    rebalance_every = portfolio_columns[3].number_input("调仓间隔（交易日）", min_value=1, max_value=60, value=1, step=1, disabled=not run_portfolio)
    st.caption("先获取完整股票池历史，再按交易日汇总截面训练共享模型；至少需要5只有效股票，同日覆盖须达到本次有效池的80%。")

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
            try:
                context = train_market_context(
                    {item["code"]: item["history"] for item in ready}, benchmark,
                    metadata={**source_meta, "pool_mode": pool_mode}, progress=task.step,
                    exposures=exposures, objective=objective_mode, window=int(training_window),
                )
                st.info(f"共享模型上下文 {context['context_id'][:12]} · 截面日期 {context['reference_date']} · {len(ready)}只有效股票")
                st.caption(
                    f"训练目标 {objective_label(context['objective_mode'])}；滚动窗口 "
                    f"{context['training_window_days'] or '不限'}个交易日；中性化状态：{context['neutralization']['status']}。"
                )
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
        rows.append(
            {
                "综合评分": item.get("cross_section_score"),
                "代码": item["code"],
                "名称": item["name"],
                "最新价": quote.get("price"),
                "涨跌幅": quote.get("change_pct"),
                "行情时间": quote.get("quote_time") or quote.get("fetched_at"),
                "数据状态": quote.get("data_status"),
                "交易状态": item.get("entry_trade_status", "未评估"),
                "时序因子分": item["analysis"].get("factor_score"),
                "LightGBM 20日预测值": _validated_return(lightgbm),
                "LightGBM 截面Z分数": lightgbm.get("prediction_zscore"),
                "LightGBM Rank IC": (lightgbm.get("oos_metrics") or {}).get("rank_ic"),
                "LightGBM 排名": item.get("lightgbm_rank"),
                "XGBoost 20日预测值": _validated_return(xgboost),
                "XGBoost 截面Z分数": xgboost.get("prediction_zscore"),
                "XGBoost Rank IC": (xgboost.get("oos_metrics") or {}).get("rank_ic"),
                "XGBoost 排名": item.get("xgboost_rank"),
                "参考买入区间": plan.get("buy_zone"),
                "止损参考": plan.get("initial_stop"),
                "退出条件": plan.get("trailing_exit"),
                "样本外回测收益": backtest_metrics.get("total_return"),
                "回测状态": (item.get("backtest") or {}).get("status", "未计算"),
                "信号生成时间": plan.get("generated_at"),
                "截面日期": item["analysis"].get("prediction_reference_date"),
                "模型上下文": item["analysis"].get("market_context_id"),
                "推荐理由": item.get("reason"),
                "主要风险": item.get("risk"),
            }
        )
    scan_stats = st.session_state.get("scan_stats", {})
    st.caption(
        f"本次扫描 {scan_stats.get('scanned', 0):,} 只；有效评分 {scan_stats.get('scored', 0):,} 只；"
        f"失败或排除 {scan_stats.get('failed', 0):,} 只。"
    )
    st.dataframe(pd.DataFrame(rows[: int(scan_limit)]), hide_index=True, use_container_width=True)
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
        st.caption("综合评分=40%同日截面因子分+60%通过验证的模型百分位；Z分数以同日有效股票池均值为0，标准差为1。模型验证失败只展示因子观察，不输出预测评分。排序目标下 20 日预测值是同日截面的相对排序百分位，只有在回归目标下才是预期超额收益。")

portfolio_result = st.session_state.get("portfolio_result")
if portfolio_result:
    st.divider()
    st.subheader("Top-N 组合回测（含 T+1 与交易摩擦）")
    portfolio_metrics = portfolio_result.get("metrics")
    if not portfolio_metrics:
        st.warning(portfolio_result.get("status", "组合回测未产生结果。"))
    else:
        show_backtest_metrics(portfolio_metrics, portfolio_result.get("benchmark_buy_hold_return"))
        portfolio = portfolio_result.get("portfolio", {})
        universe = portfolio_result.get("universe", {})
        score_range = portfolio_result.get("score_range", {})
        universe_note = universe.get("status") or f"股票池未超过 {universe.get('limit')} 只上限"
        st.caption(
            f"每日截面取前 {portfolio.get('top_n')} 名等权持有，单股目标仓位 {portfolio.get('per_stock_weight')}，"
            f"单股上限 {portfolio.get('max_weight')}；调仓间隔 {portfolio_result.get('rebalance_every')} 个交易日；"
            f"打分区间 {score_range.get('start')} 至 {score_range.get('end')}，共 {score_range.get('dates')} 个交易日；"
            f"股票池 {universe.get('kept')} 只（{universe_note}）。"
        )
        st.write(
            {
                "组合持仓数": portfolio.get("average_holdings"),
                "资金投入比例": portfolio.get("invested_ratio"),
                "现金比例": portfolio.get("cash_ratio"),
                "单股上限是否生效": portfolio.get("cap_binding"),
                "已平仓交易数": portfolio_metrics.get("trades"),
                "买入未成交笔数": portfolio_result.get("unfilled_buys"),
                "卖出未成交笔数": portfolio_result.get("unfilled_sells"),
                "股票池等权收益": portfolio_result.get("universe_equal_weight_return"),
                "费用明细": portfolio_result.get("costs"),
                "模型版本": portfolio_result.get("model_version"),
            }
        )
        st.caption("撮合约束：" + "；".join(portfolio_result.get("constraints", [])))
        open_positions = portfolio_result.get("open_positions") or []
        if open_positions:
            st.markdown("**回测结束时的未平仓持仓**")
            st.dataframe(pd.DataFrame(open_positions), hide_index=True, width="stretch")
        trades = portfolio_result.get("trades") or []
        if trades:
            with st.expander(f"查看组合成交明细（{len(trades):,} 笔）"):
                st.dataframe(pd.DataFrame(trades), hide_index=True, width="stretch")
        daily_equity = portfolio_result.get("daily_equity") or []
        if daily_equity:
            equity_frame = pd.DataFrame(daily_equity).set_index("date")[["equity", "gross_equity"]]
            st.line_chart(equity_frame)
            st.caption("equity 为计入佣金、印花税与滑点后的净值；gross_equity 为同一撮合路径下不计费用的毛净值。")
        training_log = portfolio_result.get("training_log") or []
        if training_log:
            with st.expander(f"查看滚动重训记录（{len(training_log):,} 段）"):
                # 摊平每段里每个模型的检验数值，直接看出是哪条门槛拦住了打分。
                rows = [
                    {"打分起始日": block.get("prediction_start"), "训练截至": block.get("trained_through"), **model}
                    for block in training_log for model in block.get("models", [])
                ]
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                st.caption("每段按起始日只用该日之前的历史重训；验证规则为当日生效的门槛，"
                           "排序目标看 NDCG 与随机基线之比，回归目标看 MAE 是否优于训练中位数基线。")
