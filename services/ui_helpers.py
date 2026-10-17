"""Streamlit 页面共享的展示格式与图表工具。"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Sequence

import streamlit as st

if TYPE_CHECKING:
    import pandas as pd


def apply_page_style() -> None:
    """为所有功能页应用统一样式，保留窄屏和键盘操作能力。"""
    st.markdown(
        """
        <style>
        :root { --quant-ink: #172b4d; --quant-accent: #087f8c; }
        [data-testid="stAppViewContainer"] { background: #f4f7fb; }
        [data-testid="stHeader"] { background: rgba(244,247,251,.94); }
        /* 顶部间距由 Streamlit 预留，避免页面标题被工具栏遮挡。 */
        .block-container { max-width: 1560px; padding-inline: 2rem; padding-bottom: 5rem; }
        [data-testid="stSidebar"] { background: #fff; border-right: 1px solid #e4ebf3; }
        [data-testid="stSidebarNav"] a { border-radius: 8px; margin: 3px 0; padding-left: 12px; }
        [data-testid="stSidebarNav"] a[aria-current="page"] {
            background: #e5f3f4; color: #075d66; border-left: 3px solid #087f8c; font-weight: 700; }
        .quant-brand { display: flex; align-items: center; gap: 12px; padding: 12px 0 18px; }
        .quant-brand-mark { display: grid; place-items: center; width: 42px; height: 42px;
            background: #087f8c; color: white; border-radius: 12px; font-weight: 700; font-size: 22px; }
        .quant-brand strong { color: #172b4d; font-size: 19px; }
        .quant-brand small { color: #687b92; display: block; margin-top: 3px; }
        .quant-page-header { padding: 8px 0 20px; margin-bottom: 12px; border-bottom: 1px solid #e0e8f1; }
        .quant-eyebrow { color: #087f8c; font-size: 12px; font-weight: 700; letter-spacing: 2px; margin-bottom: 10px; }
        .quant-page-header h1 { color: #172b4d; font-size: clamp(1.8rem,3vw,2.4rem);
            font-weight: 750; line-height: 1.3; padding: 0; margin: 0 0 12px; }
        .quant-page-header p { color: #61748c; line-height: 1.8; margin: 0; max-width: 980px; }
        h2, h3 { color: #172b4d; }
        [data-testid="stForm"], [data-testid="stVerticalBlockBorderWrapper"] > div { border-radius: 16px; }
        [data-testid="stForm"] { background: #fff; border: 1px solid #e0e8f1; padding: 1.3rem; }
        [data-testid="stMetric"] { background: #fff; border: 1px solid #e0e8f1; border-radius: 14px;
            padding: 16px 18px; min-height: 108px; box-shadow: 0 3px 12px rgba(23,43,77,.025); }
        [data-testid="stMetricLabel"] { color: #61748c; }
        [data-testid="stMetricValue"] { color: #172b4d; font-size: 1.75rem; }
        [data-testid="stTabs"] [role="tablist"] { gap: 6px; background: #e8eef5;
            padding: 5px; border-radius: 12px; margin-bottom: 14px; }
        [data-testid="stTabs"] [role="tab"] { border-radius: 8px; padding: 9px 18px; height: auto; }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
            background: #fff; color: #087f8c; box-shadow: 0 2px 6px rgba(23,43,77,.06); }
        [data-testid="stTabs"] [data-baseweb="tab-highlight"] { display: none; }
        [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"],
        [data-testid="stBaseButton-primaryFormSubmit"], [data-testid="stBaseButton-secondaryFormSubmit"] {
            border-radius: 10px; min-height: 42px; font-weight: 600; }
        [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {
            background: #087f8c; border-color: #087f8c; color: #fff; }
        [data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:hover {
            background: #066b76; border-color: #066b76; color: #fff; }
        [data-testid="stDataFrame"], [data-testid="stPlotlyChart"] {
            border: 1px solid #e0e8f1; border-radius: 14px; overflow: hidden; background: #fff; }
        [data-testid="stAlert"] { border-radius: 12px; }
        [data-testid="stExpander"] { background: #fff; border-radius: 14px; }
        [data-testid="stSpinner"] { background: #e8f4f5; color: #15596a; padding: 14px 18px; border-radius: 12px; }
        .quant-empty { padding: 42px 28px; text-align: center; border: 1px dashed #c8d7e7;
            background: #fff; border-radius: 18px; margin: 12px 0; }
        .quant-empty-symbol { color: #087f8c; font-size: 32px; margin-bottom: 10px; }
        .quant-empty h3 { font-size: 20px; padding: 0; margin: 0 0 10px; }
        .quant-empty p { color: #61748c; margin: 0; line-height: 1.8; }
        .quant-loading { position: fixed; bottom: 24px; right: 28px; z-index: 1000000;
            display: flex; align-items: center; gap: 16px; width: min(480px,calc(100vw - 40px));
            box-sizing: border-box; padding: 22px 24px; background: #123b54; color: #fff;
            border: 1px solid #2b7086; border-radius: 16px; box-shadow: 0 12px 36px rgba(15,41,65,.25);
            pointer-events: none; }
        .quant-loading strong { font-size: 19px; display: block; line-height: 1.5; }
        .quant-loading small { color: #cae5ee; display: block; margin-top: 5px; line-height: 1.6; }
        .quant-loading-ring { flex: 0 0 30px; width: 30px; height: 30px; border: 3px solid #587f94;
            border-top-color: #6ee7d7; border-radius: 50%; animation: quant-spin 1s linear infinite; }
        @keyframes quant-spin { to { transform: rotate(360deg); } }
        @media (prefers-reduced-motion: reduce) { .quant-loading-ring { animation: none; } }
        @media (max-width: 768px) {
            .block-container { padding-inline: 1rem; padding-bottom: 7rem; }
            .quant-loading { bottom: 16px; right: 20px; padding: 16px 18px; }
            .quant-loading strong { font-size: 16px; }
            [data-testid="stTabs"] [role="tab"] { padding: 8px 12px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # 当前框架未提供中文语言配置，统一补齐其内置工具栏和弹出菜单的操作文案。
    localization = Path(__file__).resolve().parents[1] / "assets" / "ui_zh_cn.js"
    st.html(
        f"<script>{localization.read_text(encoding='utf-8')}</script>",
        unsafe_allow_javascript=True,
    )


def show_page_header(title: str, description: str, section: str) -> None:
    """显示标题和功能说明，对传入文案转义后再输出到页面。"""
    st.markdown(
        f'<header class="quant-page-header"><div class="quant-eyebrow">{escape(section)}</div>'
        f'<h1>{escape(title)}</h1><p>{escape(description)}</p></header>',
        unsafe_allow_html=True,
    )


def show_empty_state(title: str, description: str) -> None:
    """在尚无查询结果时提供清晰的操作引导。"""
    st.markdown(
        '<section class="quant-empty"><div class="quant-empty-symbol" aria-hidden="true">↗</div>'
        f'<h3>{escape(title)}</h3><p>{escape(description)}</p></section>',
        unsafe_allow_html=True,
    )


def show_conclusion(text: str, tone: str = "info") -> None:
    """把本次结果的一句话结论放在最前面：先让人看懂结论，再往下看依据。"""
    {"info": st.info, "good": st.success, "warn": st.warning, "bad": st.error}[tone](text, title="一句话结论")


def show_plain_numbers(items: Sequence[tuple[str, str, str]], columns: int = 3) -> None:
    """成组展示关键数字，每个数字下面跟一句白话解释，避免只给数字不给含义。"""
    for start in range(0, len(items), columns):
        for slot, (label, value, note) in zip(st.columns(columns), items[start:start + columns]):
            slot.metric(label, value)
            slot.caption(note)


class LoadingTask:
    """维护一个加载任务的阶段提示和真实完成进度。"""

    def __init__(self, title: str, detail: str) -> None:
        self.title = title
        self.failed = False
        self.status = st.status(f"加载中 · {title}", expanded=True)
        with self.status:
            self.message = st.empty()
            self.progress = st.empty()
            st.caption(detail)
        self.step("正在准备，请稍候…")

    def step(self, message: str, completed: int | None = None, total: int | None = None) -> None:
        """更新当前步骤；只有已知完成数量和总数时才显示百分比。"""
        self.message.text(message)
        if completed is not None and total is not None and total > 0:
            self.progress.progress(min(1.0, max(0.0, completed / total)))

    def fail(self, message: str) -> None:
        """标记没有抛出异常的业务失败，避免显示为成功完成。"""
        self.failed = True
        self.message.text(message)


@contextmanager
def loading_task(
    title: str, detail: str = "请保持页面打开，完成后会自动展示结果。", *, exclusive: bool = False
) -> Iterator[LoadingTask]:
    """限制重计算并发，在耗时操作期间显示真实阶段与进度。

    exclusive=True 的任务会占住全局计算锁，同一数据库同一时刻只允许一个。
    这里曾经按标题字符串匹配来判断任务轻重，漏登记一个标题（例如
    「Top-N 组合构建与摩擦成本回测」）就等于该环节静默失去并发保护，
    因此改成由调用方显式声明。
    """
    from services.workload import compute_slot

    with compute_slot() if exclusive else nullcontext():
        with _loading_display(title, detail) as task:
            yield task


@contextmanager
def _loading_display(title: str, detail: str) -> Iterator[LoadingTask]:
    """任务结束或异常后清理浮层，保留实际处理结果。"""
    banner = st.empty()
    banner.markdown(
        '<div class="quant-loading" role="status" aria-live="polite">'
        '<span class="quant-loading-ring" aria-hidden="true"></span><div>'
        f'<strong>加载中 · {escape(title)}</strong><small>当前步骤与进度显示在任务卡片中，请稍候…</small></div></div>',
        unsafe_allow_html=True,
    )
    task = LoadingTask(title, detail)
    try:
        yield task
    except Exception:
        task.status.update(label=f"处理失败 · {title}", state="error", expanded=True)
        task.message.text("任务未完成，请查看下方错误信息后重试。")
        raise
    else:
        if task.failed:
            task.status.update(label=f"处理失败 · {title}", state="error", expanded=True)
        else:
            task.message.text("处理完成。")
            task.progress.empty()
            task.status.update(label=f"已完成 · {title}", state="complete", expanded=False)
    finally:
        banner.empty()


def format_number(value: Any, digits: int = 2, suffix: str = "") -> str:
    import pandas as pd

    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):,.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "—"


def show_candles(frame: pd.DataFrame, title: str, count: int) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if frame.empty:
        st.info(f"{title}：无可用数据。")
        return
    chart_data = frame.tail(count)
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.75, 0.25])
    figure.add_trace(
        go.Candlestick(
            x=chart_data["date"],
            open=chart_data["open"],
            high=chart_data["high"],
            low=chart_data["low"],
            close=chart_data["close"],
            name="开高低收",
            increasing_line_color="#e45756",
            decreasing_line_color="#14967f",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(x=chart_data["date"], y=chart_data["volume"], name="成交量", marker_color="#86b9c4"),
        row=2,
        col=1,
    )
    figure.update_layout(
        title=title,
        height=560,
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"color": "#42566e"},
        showlegend=False,
        xaxis_rangeslider_visible=False,
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
    )
    figure.update_yaxes(title_text="价格", row=1, col=1)
    figure.update_yaxes(title_text="成交量", row=2, col=1)
    st.plotly_chart(figure, width="stretch", alt=f"{title}的价格与成交量走势")


def show_backtest_metrics(metrics: dict[str, Any], benchmark_return: float | None) -> None:
    """用白话标签展示模拟交易结果，专业名称和含义放进悬停说明。"""
    columns = st.columns(6)
    columns[0].metric(
        "总收益（模拟）", format_number(_percent(metrics.get("total_return")), 2, "%"),
        help="按本页规则在历史上模拟交易的总盈亏比例（样本外总收益）。模拟不是预测，不代表未来也这样。",
    )
    columns[1].metric(
        "折算成年", format_number(_percent(metrics.get("annualized_return")), 2, "%"),
        help="把总收益折算成“平均一年赚多少”（年化收益），方便不同时间长度互相比较。",
    )
    columns[2].metric(
        "最大亏损幅度", format_number(_percent(metrics.get("maximum_drawdown")), 2, "%"),
        help="历史上从最高点跌到最低点的最大幅度（最大回撤），用来估计最难熬时会亏多少。",
    )
    columns[3].metric(
        "性价比（夏普）", format_number(metrics.get("sharpe_ratio"), 2),
        help="每承担一份波动能换来多少收益（夏普比率）：大于 1 算不错，小于 0 表示这段路在亏钱。",
    )
    columns[4].metric(
        "赚钱次数占比", format_number(_percent(metrics.get("win_rate")), 1, "%"),
        help="盈利的交易占全部交易的比例（胜率）。胜率高不代表一定赚钱，还要看每次赚亏多少。",
    )
    columns[5].metric(
        "平均赚亏比", format_number(metrics.get("profit_loss_ratio"), 2),
        help="平均每笔赚的钱 ÷ 平均每笔亏的钱（盈亏比）：大于 1 才有机会整体赚钱。",
    )
    st.write(
        {
            "总共交易几笔": metrics.get("trades"),
            "平均一年换几轮": metrics.get("annualized_turnover"),
            "不扣手续费的收益": metrics.get("gross_return"),
            "同期沪深 300 买入不动": benchmark_return,
        }
    )


def _percent(value: Any) -> float | None:
    import pandas as pd

    return float(value) * 100 if value is not None and pd.notna(value) else None

