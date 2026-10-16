"""为登录后的计算任务设置全局并发上限，避免重复提交耗尽CPU。"""

from contextlib import contextmanager
from threading import BoundedSemaphore
from typing import Iterator

import streamlit as st

WORK_SLOTS = BoundedSemaphore(1)


@contextmanager
def compute_slot() -> Iterator[None]:
    """同一数据库仅运行一个模型或扫描任务，繁忙时立即提示。"""
    from services.runtime import get_repository

    if not WORK_SLOTS.acquire(blocking=False):
        st.warning("已有计算任务正在运行，请完成后再提交。")
        st.stop()
    try:
        # 使用独立会话锁覆盖多进程；连接关闭或进程退出时自动释放。
        with get_repository().connect() as connection:
            acquired = connection.execute("SELECT pg_try_advisory_lock(706821026) AS acquired").fetchone()["acquired"]
            connection.commit()
            if not acquired:
                st.warning("服务器已有计算任务正在运行，请稍后重试。")
                st.stop()
            yield
    finally:
        WORK_SLOTS.release()
