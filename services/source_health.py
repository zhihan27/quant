"""行情数据源熔断：连续网络失败后短暂跳过该接口，避免逐只股票重复等超时。

只统计网络层失败——连接断开、超时、限流（429）、服务端错误（5xx）与被拒绝（403）。
单只股票没有数据、字段校验失败这类业务结果不算故障：一只停牌股不该把整条数据源判死。

计数口径是「一次逻辑请求」而不是「一次 attempt」：`_request_response` 内部会重试
多次，若按 attempt 计数，一只股票重试三次就能触发熔断。因此调用方必须在重试循环
结束后只汇报一次 outcome。

冷却结束进入半开：只放行一个探针请求，成功即关闭，失败即重新打开；其余线程在探测
期间快速失败，继续走各自的上层回退（东财→腾讯→新浪）。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import requests

from config import settings


def is_network_failure(error: BaseException) -> bool:
    """该异常是否属于数据源网络故障，值得计入熔断。"""
    if isinstance(error, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return True
    if isinstance(error, requests.exceptions.HTTPError):
        status = getattr(getattr(error, "response", None), "status_code", None)
        return status in (403, 429) or (status is not None and status >= 500)
    return False


@dataclass
class _SourceState:
    """单个接口的连续失败、冷却时间与半开探针归属。"""

    failures: int = 0
    opened_at: float | None = None
    probe_in_flight: bool = False
    probe_thread_id: int | None = None


class SourceHealth:
    """按接口维护熔断状态；线程安全，供采集线程池共享。"""

    def __init__(
        self,
        threshold: int | None = None,
        cooldown_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.threshold = max(1, int(threshold if threshold is not None else settings.SOURCE_FAILURE_THRESHOLD))
        self.cooldown_seconds = max(
            1.0, float(cooldown_seconds if cooldown_seconds is not None else settings.SOURCE_COOLDOWN_SECONDS)
        )
        self._clock = clock
        self._lock = threading.Lock()
        self._states: dict[str, _SourceState] = {}

    def _state(self, source: str) -> _SourceState:
        state = self._states.get(source)
        if state is None:
            state = _SourceState()
            self._states[source] = state
        return state

    def allow(self, source: str) -> bool:
        """熔断打开时返回 False；冷却结束只允许一个半开探针。"""
        now = self._clock()
        with self._lock:
            state = self._state(source)
            if state.opened_at is None:
                return True
            if now - state.opened_at < self.cooldown_seconds:
                return False
            if state.probe_in_flight:
                return False
            state.probe_in_flight = True
            state.probe_thread_id = threading.get_ident()
            return True

    def record_success(self, source: str) -> None:
        """网络请求成功时复位连续失败计数与熔断状态。"""
        with self._lock:
            state = self._state(source)
            state.failures = 0
            state.opened_at = None
            state.probe_in_flight = False
            state.probe_thread_id = None

    def release_probe(self, source: str) -> None:
        """请求退出时释放半开探针；业务失败与取消不会永久占住探针。"""
        with self._lock:
            state = self._state(source)
            if state.probe_thread_id == threading.get_ident():
                state.probe_in_flight = False
                state.probe_thread_id = None

    def record_failure(self, source: str) -> None:
        """记录一次逻辑请求失败；打开状态下只更新时间，不叠加非故障失败的计数。"""
        now = self._clock()
        with self._lock:
            state = self._state(source)
            if state.opened_at is not None:
                state.opened_at = now
                state.probe_in_flight = False
                state.probe_thread_id = None
                return
            state.failures += 1
            if state.failures >= self.threshold:
                state.opened_at = now
                state.failures = 0
                state.probe_in_flight = False
                state.probe_thread_id = None

    def snapshot(self, now: float | None = None) -> dict[str, dict[str, Any]]:
        """供日志与页面诊断：不包含任何请求参数或凭据。"""
        moment = self._clock() if now is None else now
        result: dict[str, dict[str, Any]] = {}
        with self._lock:
            for source, state in self._states.items():
                if state.opened_at is None:
                    status = "closed"
                    retry_after = 0.0
                elif moment - state.opened_at < self.cooldown_seconds:
                    status = "open"
                    retry_after = max(0.0, self.cooldown_seconds - (moment - state.opened_at))
                else:
                    status = "half_open"
                    retry_after = 0.0
                result[source] = {
                    "status": status,
                    "consecutive_failures": state.failures,
                    "retry_after_seconds": round(retry_after, 1),
                }
        return result
