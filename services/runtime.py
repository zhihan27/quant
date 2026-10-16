"""共享数据库及后台任务入口，页面重跑时复用资源。"""

import os
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from services.database import QuantRepository
    from services.intraday_monitor import IntradayMonitor
    from services.market_data import MarketDataService


@st.cache_resource
def get_repository() -> "QuantRepository":
    """创建业务仓库并幂等初始化 PostgreSQL 表。"""
    from services.database import QuantRepository

    return QuantRepository()


@st.cache_resource
def get_market_data() -> "MarketDataService":
    """在所有页面间共享行情服务，避免重复创建连接和初始化本地缓存。"""
    from services.market_data import MarketDataService

    return MarketDataService()


@st.cache_resource
def get_monitor() -> "IntradayMonitor":
    """创建进程内唯一监控对象，多进程由数据库锁进一步去重。"""
    from services.intraday_monitor import IntradayMonitor

    monitor = IntradayMonitor(get_repository())
    if os.environ.get("QUANT_DISABLE_MONITOR") != "1":
        monitor.start()
    return monitor
