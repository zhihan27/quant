"""按股票池扫描真实行情并展示可解释候选结果。"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from services.database import DatabaseError
from services.runtime import get_market_data, get_repository

from config import settings
from services.auth_ui import require_session
from services.rank_model import objective_label
from services.scan_pipeline import POOL_MODES, ScanOptions, collect_candidates, run_scan_pipeline
from services.scan_store import load_scan_result, save_scan_result
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

# 一轮全市场扫描训练出的模型上下文是所有账号共用的：谁扫，谁就决定别人明天看到什么。
# 因此扫描只由管理员触发，平日由收盘后的定时任务统一刷新；其余账号在本页只读最近一次结果。
# 与 pages/5_账号管理.py 一样做二次校验，避免以后调整路由时越权。
account = require_session()
is_admin = bool(account["is_admin"])

NOTE_RENDERERS = {
    "caption": st.caption,
    "info": st.info,
    "warning": st.warning,
    "error": st.error,
}


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


if is_admin:
    service = get_market_data()
    with st.container(border=True):
        st.subheader("筛选条件")
        scope_columns = st.columns([3, 1])
        pool_mode = scope_columns[0].selectbox("股票范围", POOL_MODES)
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
        options = ScanOptions(
            pool_mode=pool_mode, custom_codes=custom_codes, scan_limit=int(scan_limit),
            objective_mode=objective_mode, training_window=int(training_window),
            enable_neutralization=enable_neutralization, use_extra_factors=use_extra_factors,
            run_portfolio=run_portfolio, portfolio_top_n=int(portfolio_top_n),
            max_position_weight=float(max_position_weight), rebalance_every=int(rebalance_every),
            include_backtest=include_backtest, trigger="manual",
        )
        with loading_task("获取完整股票池", "首次获取或启用备用数据源时会逐页核对完整股票池，通常需要一至两分钟。") as task:
            candidates, source_meta = collect_candidates(service, options, task.step)
            if not candidates:
                task.fail("股票池获取失败，请查看下方错误信息。")
        if not candidates:
            # 股票池没取到就不动上一次的结果：一次失败的取数不该把界面上已有的排名擦掉。
            st.error(source_meta.get("error", "当前股票列表为空；可切换为用户自定义代码扫描。"))
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
            with loading_task("扫描与模型分析", f"本次共 {len(candidates):,} 只股票。进度按已处理数量计算，单只股票训练耗时可能不同。", exclusive=True) as task:
                result = run_scan_pipeline(service, options, candidates, source_meta, task.step)
            save_scan_result(result)
            # 回读一次再展示：这样管理员此刻看到的，与别人之后读到的严格是同一份内容。
            st.session_state["scan_result"] = load_scan_result()
            st.session_state["portfolio_result"] = result.portfolio_result
else:
    st.info(
        "全市场扫描会训练一份所有账号共用的模型，因此只由管理员手动触发，"
        "平日由收盘后的定时任务统一执行。下面显示的是最近一次扫描的结果。"
    )

payload = st.session_state.get("scan_result") or load_scan_result()
rows = (payload or {}).get("rows") or []
scan_stats = (payload or {}).get("scan_stats") or {}

if not rows:
    if is_admin:
        show_empty_state("配置范围，开始发现候选股票", "首次可使用少量自定义代码体验。全市场扫描包含逐股行情获取与模型训练，耗时取决于股票数量和网络。")
    else:
        show_empty_state("还没有可看的扫描结果", "管理员或收盘后的定时任务跑完第一轮全市场扫描后，这里会显示候选排名。")

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
        f"扫描明细：有效评分 {scan_stats.get('scored', 0):,} 只；失败或排除 {scan_stats.get('failed', 0):,} 只。"
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

# 刚跑完的组合回测带着逐笔成交等明细，来自会话；重新打开页面时只有落盘的汇总口径。
portfolio_result = st.session_state.get("portfolio_result") or (payload or {}).get("portfolio")
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
