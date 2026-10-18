"""市场状态与真实特征状态的共用展示，不在页面中推断入模成功。"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from config import settings
from services.database import DatabaseError
from services.market_regime import latest_regime
from services.stock_style import FACTOR_LABELS


def _format_style_value(value: float | None, percentage: bool) -> str:
    """比例用百分比展示，市值对数保留四位小数。"""
    if value is None:
        return "—"
    return f"{value:.2%}" if percentage else f"{value:.4f}"


def show_market_regime(repository: Any) -> None:
    """展示最新落库状态的日期、样本范围、指标与明确的缺失原因。"""
    try:
        regime = latest_regime(repository)
    except DatabaseError as exc:
        st.warning(f"市场状态读取失败：{exc}")
        return
    if not regime:
        st.info("尚无市场状态记录，完成一次全市场扫描后生成。")
        return
    st.info(f"市场状态：{regime['label']} · 截面 {regime['reference_date']} · 范围：{regime['scope']}")
    with st.expander("市场状态指标与数据限制"):
        metrics = st.columns(4)
        relative = regime.get("relative_momentum")
        amount = regime.get("amount_yuan")
        turnover = regime.get("turnover_percentile")
        amount_percentile = regime.get("amount_percentile")
        metrics[0].metric("小盘相对大盘短期动量", f"{relative:.2%}" if relative is not None else "未接入")
        metrics[1].metric("样本成交额（亿元）", f"{amount / 1e8:.2f}" if amount is not None else "未接入")
        metrics[2].metric("成交额历史分位", f"{amount_percentile:.1%}" if amount_percentile is not None else "未接入")
        metrics[3].metric("换手率历史分位", f"{turnover:.1%}" if turnover is not None else "未接入")
        if regime.get("top_industries"):
            st.dataframe(pd.DataFrame(regime["top_industries"]).rename(columns={
                "industry": "申万一级行业", "daily_return": "样本平均日收益",
                "amount_yuan": "样本成交额（元）", "stocks": "样本数量",
            }), hide_index=True, alt="当日领涨的三个申万一级行业及样本范围")
        if regime.get("top_amount_industries"):
            st.caption("申万一级成交额前三（不是净流入）：" + "、".join(regime["top_amount_industries"]))
        for warning in regime.get("warnings") or []:
            st.caption(warning)


def show_style_status(report: dict[str, Any] | None, features: list[str] | None = None) -> None:
    """分别呈现配置请求、覆盖门控与真实模型特征名单。"""
    report = report or {}
    st.caption(f"特征协议 {settings.STYLE_FEATURE_PROTOCOL}；{report.get('status', '尚无当前版本的训练摘要')}")
    with st.expander("显式市值与估值特征的生效状态"):
        if report.get("exposure_warning"):
            st.warning(report["exposure_warning"])
        decisions = report.get("features") or {}
        if decisions:
            st.dataframe(pd.DataFrame.from_dict(decisions, orient="index"), alt="风格特征覆盖率与排除原因")
        st.write("通过面板门控的候选列：", report.get("model_feature_columns") or [])
        if features is not None:
            st.write("已训练模型的实际入模列：", features)
        st.caption("面板候选列仍可能因训练样本不足被模型排除；实际入模以各模型 features_used 为准。")


def show_stock_style_analysis(report: dict[str, Any] | None) -> None:
    """展示个股市值、估值、缺失原因与实际模型列，观察值不计入量价走势分。"""
    st.subheader("市值与估值因子")
    if not report:
        st.info("请重新运行个股分析，生成市值、估值快照与实际入模状态。")
        return
    st.caption(
        "以下是风格特征观察，不另造估值综合分。对数市值越大表示规模越大，"
        "E/P、B/P、S/P 反映相应基本面相对总市值的比例，不直接代表买入信号。"
    )
    raw = report.get("raw") or {}
    caps = st.columns(2)
    for column, key, label in (
        (caps[0], "market_cap_total", "总市值（亿元）"),
        (caps[1], "market_cap_float", "流通市值（亿元）"),
    ):
        value = raw.get(key)
        column.metric(
            label,
            f"{value / 1e8:,.2f}" if value is not None and value > 0 else "未接入",
        )
    ratios = st.columns(3)
    for column, key, label in (
        (ratios[0], "pe_ttm", "市盈率 PE(TTM)"),
        (ratios[1], "pb", "市净率 PB"),
        (ratios[2], "ps_ttm", "市销率 PS(TTM)"),
    ):
        value = raw.get(key)
        column.metric(label, f"{value:,.2f}" if value is not None else "未接入")
    st.caption(
        f"快照来源：{report.get('quote_source') or '未提供'} · "
        f"行情时间：{report.get('quote_time') or '未提供逐股源时间'} · "
        f"采集时间：{report.get('collected_at') or '未提供'} · "
        f"{report.get('quote_status') or '行情状态未知'}"
    )
    if not report.get("quote_valid"):
        st.warning("快照未通过行情时效校验，仅供观察，不作为当前有效模型输入。")
    st.caption(
        f"历史点时值匹配 {report.get('reference_date') or '未知'} 的完整日线；"
        "当前快照不回填历史。表中是原始风格值，训练还需同日截尾、截面标准化及覆盖验收。"
    )
    rows = []
    for factor in report.get("factors") or []:
        percentage = factor["factor"] not in ("log_market_cap_total", "log_market_cap_float")
        rows.append({
            "因子": factor["label"],
            "当前快照值": _format_style_value(factor["current_value"], percentage),
            "快照状态": factor["current_status"],
            "同日点时值": _format_style_value(factor["historical_value"], percentage),
            "历史状态": factor["historical_status"],
            "本股有效模型": "、".join(factor["effective_models"]) or "未生效",
        })
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, width="stretch",
        alt="五项市值估值因子的快照、同日点时值、缺失原因与有效模型",
    )
    st.caption(report.get("fundamental_status") or "实际入模状态未知")
    with st.expander("风格因子血缘与实际入模详情"):
        requested = report.get("requested") or {}
        size_requested = requested.get("size")
        value_requested = requested.get("value")
        size_text = "未知" if size_requested is None else ("启用" if size_requested else "关闭")
        value_text = "未知" if value_requested is None else ("启用" if value_requested else "关闭")
        st.caption(
            f"所选模型请求：市值 {size_text}；估值 {value_text}。"
            f"{report.get('training_status') or ''}"
        )
        metadata = report.get("history_metadata") or {}
        if metadata.get("source"):
            st.write({
                "历史来源": metadata.get("source"),
                "历史版本": metadata.get("revision_id"),
                "当时可用时间": metadata.get("available_at"),
                "历史采集时间": metadata.get("collected_at"),
                "质量说明": metadata.get("quality"), "原始单位": metadata.get("unit"),
            })
        model_rows = []
        for item in report.get("models") or []:
            labels = [
                FACTOR_LABELS[name.removeprefix("xs_")]
                for name in item["style_features"]
            ]
            model_rows.append({
                "模型": item["model"],
                "周期（交易日）": item["horizon"],
                "训练包含的风格": "、".join(labels) or "无",
                "本股当前预测": "可用" if item["prediction_available"] else "不可用",
                "状态": item["status"], "实际 features_used": "、".join(item["features_used"]) or "未提供",
            })
        if model_rows:
            st.dataframe(
                pd.DataFrame(model_rows), hide_index=True, width="stretch",
                alt="逐模型逐周期训练特征名单与本股当前预测状态",
            )
        st.caption("训练包含风格列与本股当前预测可用是两件事；缺历史、缺覆盖或缺独立验收时不会强行启用。")
