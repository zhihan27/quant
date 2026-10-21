"""可独立运行的秒级监控任务，数据库咨询锁避免多进程重复采集。"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from datetime import datetime, time as clock_time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from config import settings
from services.database import DatabaseError, QuantRepository
from services.intraday_data import IntradayDataService, market_is_open, quote_is_fresh
from services.intraday_strategy import analyze_intraday_stock, validate_monitor_config
from services.market_data import MarketDataError
from services.portfolio import summarize_positions
from services.short_term.risk import (
    evaluate_intent_triggers,
    evaluate_minute_entry_confirmations,
    evaluate_position_risks,
)
from services.short_term.store import ShortTermStore


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
        now = datetime.now(ZoneInfo("Asia/Shanghai"))
        self._maybe_schedule_short_term(now)
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
        legacy_minute_codes = set(self.repository.monitored_codes())
        minute_codes = set(legacy_minute_codes)
        short_store = ShortTermStore(self.repository)
        short_positions = short_store.monitor_positions()
        short_intents = short_store.monitor_intents()
        minute_codes.update(
            intent["code"]
            for intent in short_intents
            if intent["side"] == "BUY" and (intent.get("details") or {}).get("m3_enabled")
        )
        minute_codes = sorted(minute_codes)
        codes = sorted(set(minute_codes).union(short_store.monitor_codes()))
        if len(codes) > 50:
            self._record_short_monitor_events(short_store, short_positions, short_intents, {}, now)
            self.repository.monitor_heartbeat(0, "当前公开源单次最多监控50只，请缩小自选监控范围；未静默截断。")
            return 10
        if not codes:
            self._record_short_monitor_events(short_store, short_positions, short_intents, {}, now)
            self.repository.monitor_heartbeat(0, "请先添加自选或持仓。")
            return 5
        quotes, failures = data.get_quotes(codes)
        errors.extend(failures)
        self.repository.save_quotes(quotes)
        quote_map = {quote["code"]: quote for quote in quotes}
        minute_interval = configuration["minute_seconds"] if market_is_open(now) else 300
        for code in minute_codes:
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
        frames = {code: pd.DataFrame(self.repository.minute_history(code)) for code in minute_codes}
        now = datetime.now(ZoneInfo("Asia/Shanghai"))
        self._record_short_monitor_events(short_store, short_positions, short_intents, quote_map, now, frames)
        summaries = summarize_positions(self.repository.positions(), quote_map, now)
        positions = {row["code"]: row for row in summaries}
        for code in legacy_minute_codes:
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

    def _maybe_schedule_short_term(self, now: datetime) -> None:
        """收盘后每日登记一次主周期研究任务；数据库键负责跨重启去重。"""
        if now.weekday() >= 5 or now.time() < clock_time(15, 10):
            return
        short_store = ShortTermStore(self.repository)
        run_id = short_store.queue_scheduled_run(now.date(), horizon=3)
        if run_id is None:
            return
        log_path = settings.LOG_DIR / f"short_term_{run_id}.log"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as log_file:
                subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(settings.PROJECT_ROOT) / "scan_short_term.py"),
                        "--run-id",
                        run_id,
                        "--horizon",
                        "3",
                    ],
                    cwd=str(settings.PROJECT_ROOT),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
        except OSError as exc:
            short_store.fail_run(run_id, f"{type(exc).__name__}: {exc}")
            short_store.release_failed_schedule_claim(now.date(), 3, run_id)
            self.logger.exception("无法启动收盘短线研究任务 run_id=%s", run_id)

    @staticmethod
    def _group_short_positions(positions: list[dict[str, Any]]) -> dict[tuple[int, int], list[dict[str, Any]]]:
        """按账号边界拆分风险检查，避免不同用户的持仓与提醒混用。"""
        grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for position in positions:
            key = (int(position["user_id"]), int(position["account_id"]))
            grouped.setdefault(key, []).append(position)
        return grouped

    @staticmethod
    def _group_short_intents(intents: list[dict[str, Any]]) -> dict[tuple[int, int], list[dict[str, Any]]]:
        """按账户拆分价格提醒，任务不能跨用户触发或写入事件。"""
        grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for intent in intents:
            key = (int(intent["user_id"]), int(intent["account_id"]))
            grouped.setdefault(key, []).append(intent)
        return grouped

    def _record_short_monitor_events(
        self,
        short_store: ShortTermStore,
        positions: list[dict[str, Any]],
        intents: list[dict[str, Any]],
        quotes: dict[str, dict[str, Any]],
        now: datetime,
        minute_bars: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        """持仓风控和限价提醒共用行情快照，但独立于分钟预热条件。"""
        for (user_id, account_id), grouped_positions in self._group_short_positions(positions).items():
            fresh_quotes = {
                position["code"]: quotes[position["code"]]
                for position in grouped_positions
                if position["code"] in quotes and quote_is_fresh(quotes[position["code"]], now)
            }
            short_store.update_high_watermarks(user_id, account_id, fresh_quotes)
            alerts = evaluate_position_risks(grouped_positions, quotes, now)
            short_store.save_monitor_events(user_id, account_id, alerts)
        for (user_id, account_id), grouped_intents in self._group_short_intents(intents).items():
            minute_alerts = evaluate_minute_entry_confirmations(grouped_intents, minute_bars or {}, quotes, now)
            short_store.save_intent_monitor_events(user_id, account_id, minute_alerts)
            confirmed_ids = {int(event["intent_id"]) for event in minute_alerts if event["status"] == "confirmed"}
            triggerable = [
                intent
                for intent in grouped_intents
                if not (intent.get("details") or {}).get("m3_enabled") or int(intent["id"]) in confirmed_ids
            ]
            alerts = evaluate_intent_triggers(triggerable, quotes, now)
            short_store.save_intent_monitor_events(user_id, account_id, alerts)
