"""秒级快照与分钟卡尔曼提醒的控制及展示页面。"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from services.database import DatabaseError
from services.auth_ui import require_session
from services.intraday_data import market_is_open, quote_is_fresh
from services.intraday_strategy import DEFAULT_MONITOR_CONFIG, validate_monitor_config
from services.runtime import get_monitor, get_repository
from services.ui_helpers import show_page_header

show_page_header(
    "盘中监控",
    "跟踪自选与持仓的行情快照，查看分钟级卡尔曼提醒及其有效状态。",
    "交易工作台 / 盘中监控",
)
st.caption("3至60秒批量快照，分钟级买卖提醒；行情快照并非逐笔数据，当前未连接自动下单。")
try:
    with st.spinner("加载中 · 正在读取监控设置与运行状态…", show_time=True):
        repository = get_repository()
        get_monitor()
        stored = repository.monitor_config()
except (DatabaseError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

configuration = {**DEFAULT_MONITOR_CONFIG, **stored["config"]}
with st.form("monitor_configuration"):
    st.subheader("监控设置")
    enabled = st.checkbox("开启后台监控", value=stored["enabled"])
    columns = st.columns(4)
    poll = columns[0].number_input("快照轮询（秒）", min_value=3, max_value=60, value=int(configuration["poll_seconds"]))
    minute = columns[1].number_input("分钟线同步（秒）", min_value=30, max_value=300, value=int(configuration["minute_seconds"]), step=30)
    entry = columns[2].number_input("入场偏离 Z", min_value=1.5, max_value=5.0, value=float(configuration["entry_z"]), step=0.1)
    exit_z = columns[3].number_input("回归偏离 Z", min_value=0.0, max_value=4.9, value=float(configuration["exit_z"]), step=0.1)
    st.caption("监控范围只有开启监控的自选和全部持仓，其余代码一律不监控。最多50只，实际轮询速度受网络和行情源限制。非交易时段快照每60秒采集，分钟线每5分钟同步。")
    if st.form_submit_button("保存监控设置", type="primary"):
        try:
            configuration = validate_monitor_config({
                "poll_seconds": poll, "minute_seconds": minute, "entry_z": entry, "exit_z": exit_z,
                # 配对代码是上一版留下的字段，这里显式清空，好让存量配置也停止配对监控。
                "pair_a": "", "pair_b": "",
            })
            repository.save_monitor_config(enabled, configuration)
            st.success("监控设置已保存，后台将在本轮结束后应用。")
        except (ValueError, DatabaseError) as exc:
            st.error(str(exc))


@st.fragment(run_every=5)
def show_live_status() -> None:
    """仅刷新监控结果，不重复提交设置或在页面重跑时创建采集任务。"""
    require_session()
    try:
        status = repository.monitor_config()
        now = datetime.now(ZoneInfo("Asia/Shanghai"))
        st.caption(f"页面状态每 5 秒更新 · 最近读取 {now:%H:%M:%S} · 行情源时间见下方表格")
        columns = st.columns(3)
        columns[0].metric("后台监控", "运行中" if status["enabled"] else "已暂停")
        columns[1].metric("最近采集耗时", f"{status['cycle_seconds']:.2f} 秒" if status["cycle_seconds"] is not None else "尚未采集")
        heartbeat = status["last_cycle_at"]
        columns[2].metric("最近心跳", heartbeat.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%H:%M:%S") if heartbeat else "尚未采集")
        if status["enabled"] and (heartbeat is None or (now - heartbeat).total_seconds() > 120):
            st.warning("尚无近期采集心跳，请检查任务、数据源或数据库状态。")
        if status["last_error"]:
            st.warning(status["last_error"])
        if not market_is_open(now):
            st.info("当前不在连续竞价时段，有效买卖提醒已关闭。")
        codes = repository.monitored_codes()
        quotes = repository.latest_quotes(codes)
        st.subheader("监控行情")
        if codes:
            rows = []
            for code in codes:
                quote = quotes.get(code, {})
                rows.append({
                    "代码": code, "名称": quote.get("name"), "价格": quote.get("price"), "涨跌幅%": quote.get("change_pct"),
                    "源时间": quote.get("quote_time"), "采集时间": quote.get("fetched_at"), "来源": quote.get("source"),
                    "可用于当前提醒": bool(status["enabled"]) and quote_is_fresh(quote, now),
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.info("请先加入自选或持仓。")
        st.subheader("分钟信号与失效原因")
        signals = repository.signals()
        if signals:
            rows = []
            for signal in signals:
                age = (now - signal["bar_time"]).total_seconds()
                details = signal["details"]
                current = bool(signal["active"]) and bool(status["enabled"]) and market_is_open(now) and 0 <= age <= 180
                current = current and quote_is_fresh(quotes.get(signal["code"], {}), now)
                if signal["peer_code"]:
                    current = current and quote_is_fresh(quotes.get(signal["peer_code"], {}), now)
                rows.append({
                    "分钟时间": signal["bar_time"], "代码": signal["code"], "配对代码": signal["peer_code"],
                    "策略": "卡尔曼分钟提醒" if signal["strategy"] == "kalman_minute" else "分钟协整配对",
                    "动作": signal["action"], "当前有效": current, "参考价": signal["reference_price"],
                    "Z值": details.get("z_score"), "买入观察下界": details.get("buy_watch_below"),
                    "止损参考": details.get("initial_stop"), "止盈参考": details.get("take_profit"),
                    "原因": signal["reason"] if current or not signal["active"] else "历史提醒已失效；" + signal["reason"],
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.info("尚无分钟信号记录，开启监控后自动采集。")
        with st.expander("查看已入库分钟线"):
            if codes:
                selected = st.selectbox("分钟图代码", codes)
                frame = pd.DataFrame(repository.minute_history(selected, 240))
                if not frame.empty:
                    st.line_chart(frame.set_index("date")[["close"]])
                else:
                    st.info("尚无分钟线。")
    except (DatabaseError, ValueError) as exc:
        st.error(str(exc))


show_live_status()
st.caption("网页关闭后，只要 Streamlit 服务仍运行，后台监控继续工作。也可运行 quant/run_monitor.bat；两种入口共享数据库锁，不会重复采集。分钟规则尚未完成真实执行验证。")
