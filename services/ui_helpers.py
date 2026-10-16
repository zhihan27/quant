"""Streamlit 页面共享的展示格式与图表工具。"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from html import escape
from typing import TYPE_CHECKING, Any, Iterator

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
        .block-container { max-width: 1560px; padding: 1.6rem 2rem 5rem; }
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
            .block-container { padding: 1.3rem 1rem 7rem; }
            .quant-loading { bottom: 16px; right: 20px; padding: 16px 18px; }
            .quant-loading strong { font-size: 16px; }
            [data-testid="stTabs"] [role="tab"] { padding: 8px 12px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
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
def loading_task(title: str, detail: str = "请保持页面打开，完成后会自动展示结果。") -> Iterator[LoadingTask]:
    """限制重计算并发，在耗时操作期间显示真实阶段与进度。"""
    from services.workload import compute_slot

    heavy = title in {"个股分析", "扫描与模型分析", "滚动样本外回测", "配对统计分析"}
    with compute_slot() if heavy else nullcontext():
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
            name="OHLC",
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
    st.plotly_chart(figure, use_container_width=True)


def show_backtest_metrics(metrics: dict[str, Any], benchmark_return: float | None) -> None:
    columns = st.columns(6)
    columns[0].metric("样本外总收益", format_number(_percent(metrics.get("total_return")), 2, "%"))
    columns[1].metric("年化收益", format_number(_percent(metrics.get("annualized_return")), 2, "%"))
    columns[2].metric("最大回撤", format_number(_percent(metrics.get("maximum_drawdown")), 2, "%"))
    columns[3].metric("夏普", format_number(metrics.get("sharpe_ratio"), 2))
    columns[4].metric("胜率", format_number(_percent(metrics.get("win_rate")), 1, "%"))
    columns[5].metric("盈亏比", format_number(metrics.get("profit_loss_ratio"), 2))
    st.write(
        {
            "交易数": metrics.get("trades"),
            "年化换手次数": metrics.get("annualized_turnover"),
            "扣费前收益": metrics.get("gross_return"),
            "沪深 300 全区间买入持有": benchmark_return,
        }
    )


def _percent(value: Any) -> float | None:
    import pandas as pd

    return float(value) * 100 if value is not None and pd.notna(value) else None

