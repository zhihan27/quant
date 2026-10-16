"""可独立运行的秒级监控任务，数据库咨询锁避免多进程重复采集。"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from services.database import DatabaseError, QuantRepository
from services.intraday_data import IntradayDataService, market_is_open
from services.intraday_strategy import analyze_intraday_stock, validate_monitor_config
from services.market_data import MarketDataError
from services.portfolio import summarize_positions


class IntradayMonitor:
    """后台轮询数据库自选与持仓，保存行情和去重后的分钟信号。"""

    def __init__(self, repository: QuantRepository) -> None:
        self.repository = repository
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._minute_synced: dict[str, float] = {}
        self._last_pruned = 0.0
        self.logger = logging.getLogger("quant.intraday")

    def start(self) -> None:
        """幂等启动后台线程，Streamlit 页面重跑不重复创建线程。"""
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._stop.clear()
                self._thread = threading.Thread(target=self.run, name="quant-intraday", daemon=True)
                self._thread.start()

    def stop(self) -> None:
        """通知工作线程在本轮网络请求结束后退出。"""
        self._stop.set()

    def run(self) -> None:
        """独占数据库级监控锁；其他应用进程等待主监控退出后接管。"""
        data = IntradayDataService()
        try:
            while not self._stop.is_set():
                try:
                    with self.repository.connect() as owner:
                        acquired = owner.execute("SELECT pg_try_advisory_lock(706821025) AS acquired").fetchone()["acquired"]
                        owner.commit()
                        if not acquired:
                            self._stop.wait(5)
                            continue
                        while not self._stop.is_set():
                            # 检测锁所在会话是否仍然有效，连接丢失后不继续独立采集。
                            owner.execute("SELECT 1")
                            owner.commit()
                            delay = self.run_cycle(data)
                            self._stop.wait(delay)
                except Exception:
                    self.logger.exception("监控循环失败，稍后重连数据库")
                    self._stop.wait(5)
        finally:
            data.close()

    def run_cycle(self, data: IntradayDataService) -> float:
        """完成一轮采集并返回下一轮等待秒数；失败代码不影响其余股票。"""
        stored = self.repository.monitor_config()
        if not stored["enabled"]:
            return 3
        started = time.monotonic()
        errors = []
        if started - self._last_pruned >= 86400:
            self.repository.prune_monitor_history()
            self._last_pruned = started
        configuration = validate_monitor_config(stored["config"])
        # 只监控开启监控的自选与全部持仓：配对等研究用代码不再顺带订阅，
        # 免得有限的单次请求额度被不在自选、也不在持仓里的代码占掉。
        codes = self.repository.monitored_codes()
        if len(codes) > 50:
            self.repository.monitor_heartbeat(0, "当前公开源单次最多监控50只，请缩小自选监控范围；未静默截断。")
            return 10
        if not codes:
            self.repository.monitor_heartbeat(0, "请先添加自选或持仓。")
            return 5
        quotes, failures = data.get_quotes(codes)
        errors.extend(failures)
        self.repository.save_quotes(quotes)
        quote_map = {quote["code"]: quote for quote in quotes}
        now = datetime.now(ZoneInfo("Asia/Shanghai"))
        minute_interval = configuration["minute_seconds"] if market_is_open(now) else 300
        for code in codes:
            if self._stop.is_set():
                break
            if time.monotonic() - self._minute_synced.get(code, 0) < minute_interval:
                continue
            # 失败请求也进入冷却，避免网络异常时每几秒重复拉取整段历史。
            self._minute_synced[code] = time.monotonic()
            try:
                frame, source = data.get_minutes(code)
                self.repository.save_minutes(code, frame, source)
            except (MarketDataError, ValueError, DatabaseError) as exc:
                errors.append(f"{code}：{exc}")
        summaries = summarize_positions(self.repository.positions(), quote_map, now)
        positions = {row["code"]: row for row in summaries}
        frames = {code: pd.DataFrame(self.repository.minute_history(code)) for code in codes}
        for code in codes:
            try:
                signal = analyze_intraday_stock(code, frames[code], quote_map.get(code, {}), positions.get(code), configuration, now)
                self.repository.save_signal(signal)
            except (ValueError, KeyError, DatabaseError) as exc:
                errors.append(f"{code}：分钟信号计算失败（{type(exc).__name__}）")
        elapsed = time.monotonic() - started
        self.repository.monitor_heartbeat(elapsed, "；".join(errors) if errors else None)
        requested = configuration["poll_seconds"] if market_is_open(now) else 60
        # 一轮耗时大于设定频率时不并发堆积请求，页面展示真实耗时。
        return max(0.5, requested - elapsed)
