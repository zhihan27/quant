"""为登录后的计算任务设置全局并发上限，避免重复提交耗尽CPU。"""

from contextlib import contextmanager
from threading import BoundedSemaphore
from typing import Iterator

from config import settings

WORK_SLOTS = BoundedSemaphore(1)


class ComputeBusy(RuntimeError):
    """计算槽位被占用，异常消息即面向用户的提示语。

    无界面进程捕获它并退出，页面捕获它并停下。
    """


@contextmanager
def compute_lock(repository=None) -> Iterator[None]:
    """不依赖 Streamlit 的跨进程计算锁，繁忙时抛出 ComputeBusy 而不排队。

    网页端与每日定时扫描进程都写同一份模型与面板文件，因此必须共用
    settings.COMPUTE_LOCK_KEY 这一个编号，否则两边会同时训练并互相覆盖。
    本函数刻意不引入 Streamlit，好让 scan_daily.py 那样的无界面进程直接复用。
    """
    from services.database import QuantRepository

    if not WORK_SLOTS.acquire(blocking=False):
        raise ComputeBusy("已有计算任务正在运行，请完成后再提交。")
    try:
        # 使用独立会话锁覆盖多进程；连接关闭或进程退出时自动释放。
        with (repository or QuantRepository()).connect() as connection:
            acquired = connection.execute(
                "SELECT pg_try_advisory_lock(%s::bigint) AS acquired", (settings.COMPUTE_LOCK_KEY,)
            ).fetchone()["acquired"]
            connection.commit()
            if not acquired:
                raise ComputeBusy("服务器已有计算任务正在运行，请稍后重试。")
            yield
    finally:
        WORK_SLOTS.release()


@contextmanager
def compute_slot() -> Iterator[None]:
    """同一数据库仅运行一个模型或扫描任务，繁忙时立即提示。"""
    import streamlit as st

    from services.runtime import get_repository

    try:
        with compute_lock(get_repository()):
            yield
    except ComputeBusy as exc:
        st.warning(str(exc))
        st.stop()
