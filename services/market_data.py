"""东财公开行情接口适配、本地缓存与行情状态管理。"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime, time as datetime_time, timedelta
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import settings
from services.data_cleaning import clean_daily_history
from services.finance_network import FinanceNetwork, brief_network_error
from services.panel_store import PanelStore
from services.source_health import SourceHealth, is_network_failure


class MarketDataError(RuntimeError):
    """行情接口无可用数据时抛出的异常。"""


def _now_text() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def completed_daily_history(history: pd.DataFrame) -> pd.DataFrame:
    """盘中剔除尚未收盘的当日 K 线，避免将部分日线用于预测或回测。"""
    if history.empty or "date" not in history:
        return history.copy()
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    history = history.loc[pd.to_datetime(history["date"]).dt.date <= now.date()].copy()
    if history.empty:
        return history
    if now.time() < datetime_time(15, 0):
        last_date = pd.Timestamp(history["date"].iloc[-1]).date()
        if last_date == now.date():
            return history.loc[pd.to_datetime(history["date"]).dt.date < now.date()].copy()
    return history.copy()


def history_is_current(
    history: pd.DataFrame, quote: dict[str, Any], meta: dict[str, Any],
    expected_latest_trading_date: str | None = None,
) -> bool:
    """核对历史源状态、基准交易日和真实报价日期，不把下载时间当作交易时间。"""
    if history.empty or "过期" in meta.get("status", "") or not quote.get("quote_valid", False):
        return False
    try:
        latest = pd.Timestamp(history["date"].iloc[-1]).date()
        if pd.isna(latest):
            return False
        expected_day = pd.Timestamp(expected_latest_trading_date).date() if expected_latest_trading_date else None
        if expected_day is not None and (pd.isna(expected_day) or latest < expected_day):
            return False
        # 全市场股票池通常没有逐股源时间，周末下载不能要求存在周末日线。
        # 只有基准确认的交易日能作为缺失源时间时的校验锚点。
        if not quote.get("quote_time"):
            return expected_day is not None
        source = datetime.fromisoformat(quote["quote_time"])
        if source.tzinfo is not None:
            source = source.astimezone(ZoneInfo("Asia/Shanghai"))
        if source.time() >= datetime_time(15, 0):
            return latest == source.date()
        return 0 <= (source.date() - latest).days <= 7
    except (TypeError, ValueError):
        return False


def _clean_code(value: str) -> str:
    code = re.sub(r"\D", "", str(value))
    if len(code) != 6:
        raise ValueError("请输入 6 位 A 股代码。")
    return code


def _market_id(code: str) -> int:
    return 0 if code.startswith(("4", "8", "92")) else (1 if code.startswith(("5", "6", "9")) else 0)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


class MarketDataService:
    """获取东财行情并标识实时、缓存和历史数据状态。"""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or settings.DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # 显式传入数据库路径时（测试或独立实例）列式面板与数据库放在同一目录，避免污染生产目录。
        self.panels = PanelStore(root=(Path(db_path).parent / "panel") if db_path else None)
        settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("quant.market_data")
        self.logger.setLevel(logging.INFO)
        if not any(getattr(handler, "_quant_market_data", False) for handler in self.logger.handlers):
            handler = RotatingFileHandler(
                settings.LOG_DIR / "market_data.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
            )
            handler._quant_market_data = True
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            self.logger.addHandler(handler)
        self.session = requests.Session()
        self.session.trust_env = True
        self.session.mount("https://", HTTPAdapter(max_retries=Retry(total=0)))
        self.session.mount("http://", HTTPAdapter(max_retries=Retry(total=0)))
        self._request_gate = threading.BoundedSemaphore(settings.MAX_CONCURRENT_REQUESTS)
        # 节流状态按线程隔离：4 路采集时每个线程各自维持最小请求间隔，互不串行。
        self._pace = threading.local()
        self._last_retries: dict[tuple[str, str], int] = {}
        self._history_tencent_until = 0.0
        self._db_write_lock = threading.Lock()
        self._health = SourceHealth()
        self._universe_memory: dict[str, tuple[pd.DataFrame, dict[str, Any], float]] = {}
        self._universe_retry_after: dict[str, float] = {}
        self._universe_locks: dict[str, threading.Lock] = {}
        self._universe_meta_lock = threading.Lock()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Referer": "https://quote.eastmoney.com/",
            }
        )
        self.network = FinanceNetwork(self.session, self.logger)
        self._init_db()
        self.logger.info("market_data service started proxy=%s", self.proxy_configuration())

    def _thread_pacer(self) -> threading.local:
        """惰性创建线程本地节流状态，兼容测试里 `__new__` 构造的实例。"""
        pacer = getattr(self, "_pace", None)
        if pacer is None:
            pacer = threading.local()
            self._pace = pacer
        return pacer

    def _write_lock(self) -> threading.Lock:
        lock = getattr(self, "_db_write_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._db_write_lock = lock
        return lock

    def _source_health(self) -> SourceHealth:
        health = getattr(self, "_health", None)
        if health is None:
            health = SourceHealth()
            self._health = health
        return health

    @staticmethod
    def _source_key(url: str) -> str:
        """熔断按接口而不是供应商聚合：日线故障不该连累报价接口。"""
        for prefix, key in (
            (settings.EASTMONEY_QUOTE_URL, "eastmoney.quote"),
            (settings.EASTMONEY_KLINE_URL, "eastmoney.kline"),
            (settings.EASTMONEY_LIST_URL, "eastmoney.list"),
            (settings.TENCENT_KLINE_URL, "tencent.kline"),
        ):
            if url.startswith(prefix):
                return key
        if url.startswith("https://hq.sinajs.cn/"):
            return "sina.quote"
        if "vip.stock.finance.sina.com.cn" in url:
            return "sina.universe"
        return MarketDataService._interface_name(url)

    def _pace_request(self) -> None:
        """同一线程两次请求起点至少间隔 REQUEST_INTERVAL_SECONDS。

        多线程采集时这是每线程的节流：聚合速率上限约为线程数 / 间隔。
        间隔每次实时读取 settings，测试把它 patch 成 0 时立即返回。
        """
        interval = float(settings.REQUEST_INTERVAL_SECONDS)
        pacer = self._thread_pacer()
        last = getattr(pacer, "last_request_at", 0.0)
        now = time.monotonic()
        wait = interval - (now - last)
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
        pacer.last_request_at = now

    def source_health_snapshot(self) -> dict[str, dict[str, Any]]:
        """各行情接口的熔断状态，供页面诊断与日志使用（不含请求参数）。"""
        return self._source_health().snapshot()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.execute("PRAGMA journal_mode=WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS cache_items ("
                "cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL, fetched_at TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS backtest_runs ("
                "run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, parameters TEXT NOT NULL,"
                "model_version TEXT NOT NULL, result TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS request_events ("
                "event_id INTEGER PRIMARY KEY AUTOINCREMENT, occurred_at TEXT NOT NULL,"
                "interface TEXT NOT NULL, stock_code TEXT NOT NULL, elapsed_ms REAL NOT NULL,"
                "exception_type TEXT, retry_count INTEGER NOT NULL, success INTEGER NOT NULL, detail TEXT)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS history_sync_status ("
                "code TEXT NOT NULL, period TEXT NOT NULL, adjustment TEXT NOT NULL,"
                "last_attempt_at TEXT NOT NULL, last_success_at TEXT, latest_bar_date TEXT,"
                "bar_count INTEGER NOT NULL DEFAULT 0, source TEXT, status TEXT NOT NULL,"
                "failure_reason TEXT, retry_count INTEGER NOT NULL DEFAULT 0,"
                "PRIMARY KEY(code, period, adjustment))"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_request_events_code_time "
                "ON request_events(stock_code, occurred_at)"
            )

    def _read_cache(self, key: str) -> tuple[Any, datetime] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload, fetched_at FROM cache_items WHERE cache_key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0]), datetime.fromisoformat(row[1])

    def _write_cache(self, key: str, payload: Any) -> str:
        fetched_at = _now_text()
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        # 并发采集会同时刷新多只股票的历史缓存；JSON 编码放在锁外，
        # 只把 SQLite 写入串行化，避免 4 个写者同时写大 payload 时撞 database is locked。
        with self._write_lock():
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO cache_items(cache_key, payload, fetched_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(cache_key) DO UPDATE SET payload=excluded.payload, "
                    "fetched_at=excluded.fetched_at",
                    (key, encoded, fetched_at),
                )
        return fetched_at

    @staticmethod
    def proxy_configuration() -> dict[str, str]:
        configuration: dict[str, str] = {}
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            value = os.environ.get(name)
            if not value:
                continue
            parsed = urlsplit(value if "://" in value else f"http://{value}")
            configuration[name] = f"{parsed.hostname or '已设置'}:{parsed.port or '默认端口'}"
        return configuration

    @staticmethod
    def _interface_name(url: str) -> str:
        parsed = urlsplit(url)
        return f"{parsed.netloc}{parsed.path}"

    @staticmethod
    def _exception_detail(exc: Exception, fallback: str) -> str:
        detail = getattr(exc, "stderr", None) or getattr(exc, "stdout", None)
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        if detail and str(detail).strip():
            return str(detail).strip()
        return str(exc).strip() or fallback

    def _record_request_event(
        self, interface: str, code: str, elapsed_ms: float, exception_type: str | None,
        retry_count: int, success: bool, detail: str | None
    ) -> None:
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO request_events(occurred_at, interface, stock_code, elapsed_ms,"
                    "exception_type, retry_count, success, detail) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (_now_text(), interface, code, elapsed_ms, exception_type, retry_count, int(success), (detail or "")[:500]),
                )
        except sqlite3.Error:
            self.logger.exception("failed to persist request event interface=%s code=%s", interface, code)

    def _record_history_status(
        self, code: str, period: str, adjustment: str, status: str, latest_bar_date: str | None,
        bar_count: int, source: str | None, failure_reason: str | None, retry_count: int
    ) -> None:
        attempted_at = _now_text()
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO history_sync_status(code, period, adjustment, last_attempt_at,"
                    "last_success_at, latest_bar_date, bar_count, source, status, failure_reason, retry_count) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(code, period, adjustment) DO UPDATE SET "
                    "last_attempt_at=excluded.last_attempt_at, "
                    "last_success_at=CASE WHEN excluded.status IN ('成功','缓存') "
                    "THEN excluded.last_attempt_at ELSE history_sync_status.last_success_at END, "
                    "latest_bar_date=COALESCE(excluded.latest_bar_date, history_sync_status.latest_bar_date), "
                    "bar_count=CASE WHEN excluded.bar_count > 0 THEN excluded.bar_count ELSE history_sync_status.bar_count END, "
                    "source=COALESCE(excluded.source, history_sync_status.source), status=excluded.status, "
                    "failure_reason=excluded.failure_reason, retry_count=excluded.retry_count",
                    (code, period, adjustment, attempted_at, attempted_at if status in ("成功", "缓存") else None,
                     latest_bar_date, bar_count, source, status, (failure_reason or "")[:500] or None, retry_count),
                )
        except sqlite3.Error:
            self.logger.exception("failed to persist history status code=%s period=%s", code, period)

    def history_fetch_summary(self, limit: int = 5000) -> dict[str, Any]:
        with self._connect() as connection:
            counts = dict(connection.execute(
                "SELECT status, COUNT(*) FROM history_sync_status GROUP BY status"
            ).fetchall())
            failures = connection.execute(
                "SELECT code, period, adjustment, status, last_attempt_at, latest_bar_date,"
                "failure_reason, retry_count FROM history_sync_status "
                "WHERE status IN ('失败','过期缓存') ORDER BY last_attempt_at DESC LIMIT ?",
                (max(1, min(int(limit), 5000)),),
            ).fetchall()
            history_statuses = connection.execute(
                "SELECT code, GROUP_CONCAT(DISTINCT period || '/' || adjustment), "
                "GROUP_CONCAT(DISTINCT status), MAX(last_attempt_at), MAX(last_success_at), "
                "MAX(latest_bar_date), GROUP_CONCAT(DISTINCT failure_reason) "
                "FROM history_sync_status GROUP BY code ORDER BY MAX(last_attempt_at) DESC LIMIT ?",
                (max(1, min(int(limit), 5000)),),
            ).fetchall()
            request_stats = connection.execute(
                "SELECT stock_code, COUNT(*), SUM(success), SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) "
                "FROM request_events WHERE interface LIKE '%kline/get%' "
                "OR interface = 'Sina.stock_zh_a_daily' OR interface LIKE '%newfqkline/get%' GROUP BY stock_code"
            ).fetchall()
        request_by_code = {
            row[0]: {"attempts": int(row[1]), "successes": int(row[2] or 0), "failures": int(row[3] or 0)}
            for row in request_stats
        }
        by_stock = []
        for row in history_statuses:
            requests_for_stock = request_by_code.get(row[0], {"attempts": 0, "successes": 0, "failures": 0})
            attempts = requests_for_stock["attempts"]
            by_stock.append({
                "code": row[0], "period_adjustment": row[1], "status": row[2],
                "last_attempt_at": row[3], "last_success_at": row[4], "latest_bar_date": row[5],
                "request_attempts": attempts, "request_successes": requests_for_stock["successes"],
                "request_failures": requests_for_stock["failures"],
                "success_rate_pct": round(requests_for_stock["successes"] * 100 / attempts, 1) if attempts else None,
                "failure_reason": row[6],
            })
        return {
            "success_count": int(counts.get("成功", 0) + counts.get("缓存", 0)),
            "stale_count": int(counts.get("过期缓存", 0)),
            "failure_count": int(counts.get("失败", 0)),
            "attempted_count": int(sum(counts.values())),
            "by_stock": by_stock,
            "failures": [
                {"code": row[0], "period": row[1], "adjustment": row[2],
                 "status": row[3], "last_attempt_at": row[4], "latest_bar_date": row[5],
                 "failure_reason": row[6], "retry_count": row[7]}
                for row in failures
            ],
        }

    def request_diagnostics(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT occurred_at, interface, stock_code, elapsed_ms, exception_type, "
                "retry_count, success, detail FROM request_events "
                "ORDER BY event_id DESC LIMIT ?",
                (max(1, min(int(limit), 1000)),),
            ).fetchall()
        return [
            {
                "occurred_at": row[0], "interface": row[1], "stock_code": row[2],
                "elapsed_ms": row[3], "exception_type": row[4], "retry_count": row[5],
                "success": bool(row[6]), "detail": row[7],
            }
            for row in rows
        ]

    def _request_response(
        self, url: str, params: dict[str, Any], code: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> requests.Response:
        interface = self._interface_name(url)
        stock_code = str(code or params.get("secid") or params.get("symbol") or "股票池")
        source_key = self._source_key(url)
        health = self._source_health()
        if not health.allow(source_key):
            raise MarketDataError(f"数据源暂时熔断（{source_key}），本轮直接改用备用源。")
        try:
            last_error: Exception | None = None
            for attempt in range(1, settings.REQUEST_RETRIES + 2):
                with self._request_gate:
                    self._pace_request()
                    started = time.perf_counter()
                    try:
                        response = self.network.get(
                            url, params=params,
                            timeout=(settings.REQUEST_CONNECT_TIMEOUT_SECONDS, settings.REQUEST_READ_TIMEOUT_SECONDS),
                            headers=headers,
                        )
                        response.raise_for_status()
                        elapsed_ms = (time.perf_counter() - started) * 1000
                        self._last_retries[(interface, stock_code)] = attempt - 1
                        health.record_success(source_key)
                        self._record_request_event(interface, stock_code, elapsed_ms, None, attempt - 1, True, None)
                        self.logger.info(
                            "request success interface=%s code=%s elapsed_ms=%.1f retries=%s",
                            interface, stock_code, elapsed_ms, attempt - 1,
                        )
                        return response
                    except requests.RequestException as exc:
                        last_error = exc
                        elapsed_ms = (time.perf_counter() - started) * 1000
                        status_code = getattr(getattr(exc, "response", None), "status_code", None)
                        retryable = status_code is None or status_code == 429 or status_code >= 500
                        # 历史源主动断开且直连也未恢复时立即换源，避免重复等待同一故障。
                        if url == settings.EASTMONEY_KLINE_URL and "RemoteDisconnected" in str(exc):
                            retryable = False
                        self._last_retries[(interface, stock_code)] = attempt - 1
                        self._record_request_event(
                            interface, stock_code, elapsed_ms, type(exc).__name__, attempt - 1, False, str(exc)
                        )
                        self.logger.warning(
                            "request failure interface=%s code=%s elapsed_ms=%.1f exception=%s retries=%s detail=%s",
                            interface, stock_code, elapsed_ms, type(exc).__name__, attempt - 1, str(exc)[:300],
                        )
                if not retryable or attempt > settings.REQUEST_RETRIES:
                    # 一次逻辑请求只记一次熔断结果：重试三次不等于三次故障。
                    if is_network_failure(last_error):
                        health.record_failure(source_key)
                    raise last_error
                time.sleep(settings.RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)))
            raise MarketDataError(f"请求失败：{last_error}") from last_error
        finally:
            health.release_probe(source_key)

    def _get_json(self, url: str, params: dict[str, Any], code: str | None = None) -> dict[str, Any]:
        response = self._request_response(url, params, code)
        payload = response.json()
        if not isinstance(payload, dict) or payload.get("rc", 0) != 0:
            raise MarketDataError("行情源未返回有效数据。")
        return payload

    def _cache_result(
        self, key: str, ttl_seconds: int, allow_stale: bool = True
    ) -> tuple[Any, datetime, bool] | None:
        item = self._read_cache(key)
        if item is None:
            return None
        payload, fetched = item
        age = (datetime.now(fetched.tzinfo) - fetched).total_seconds()
        if 0 <= age <= ttl_seconds:
            return payload, fetched, False
        if allow_stale:
            return payload, fetched, True
        return None

    def _request_with_cache(
        self,
        cache_key: str,
        ttl_seconds: int,
        loader: Any,
    ) -> tuple[Any, dict[str, Any]]:
        fresh = self._cache_result(cache_key, ttl_seconds, allow_stale=False)
        if fresh is not None:
            payload, fetched, _ = fresh
            return payload, {"fetched_at": fetched.isoformat(timespec="seconds"), "status": "本地缓存"}
        try:
            payload = loader()
            fetched_text = self._write_cache(cache_key, payload)
            return payload, {"fetched_at": fetched_text, "status": "接口获取"}
        except (requests.RequestException, ValueError, KeyError, TypeError, MarketDataError) as exc:
            stale = self._cache_result(cache_key, ttl_seconds, allow_stale=True)
            if stale is not None:
                payload, fetched, _ = stale
                return payload, {
                    "fetched_at": fetched.isoformat(timespec="seconds"),
                    "status": "行情源失败，使用过期缓存（非实时）",
                    "warning": str(exc),
                }
            raise MarketDataError(f"行情获取失败：{exc}") from exc

    def get_quote(self, code: str) -> dict[str, Any]:
        """优先读取有效快照，主源仅有过期缓存时继续尝试备用源。"""
        code = _clean_code(code)
        cache_key = f"quote:{code}"
        params = {
            "secid": f"{_market_id(code)}.{code}",
            "fields": "f57,f58,f43,f44,f45,f46,f47,f48,f59,f60,f162,f168,f86",
        }
        quote: dict[str, Any] | None = None
        try:
            payload, meta = self._request_with_cache(
                cache_key,
                settings.QUOTE_CACHE_SECONDS,
                lambda: self._get_json(settings.EASTMONEY_QUOTE_URL, params, code=code)["data"],
            )
            if not payload:
                raise MarketDataError("行情源没有返回该股票。")
            if payload.get("f57") and str(payload["f57"]) != code:
                raise MarketDataError("行情源返回的股票代码与请求代码不一致。")
            digits = int(payload.get("f59") or 2)
            scale = 10**digits
            price = _number(payload.get("f43"))
            previous_close = _number(payload.get("f60"))
            price = price / scale if price is not None and price > 0 else None
            previous_close = (
                previous_close / scale if previous_close is not None and previous_close > 0 else None
            )
            change = price - previous_close if price is not None and previous_close else None
            change_pct = change / previous_close * 100 if change is not None else None
            quote_timestamp = None
            if payload.get("f86"):
                try:
                    quote_timestamp = datetime.fromtimestamp(
                        int(payload["f86"]), ZoneInfo("Asia/Shanghai")
                    ).isoformat(timespec="seconds")
                except (TypeError, ValueError, OSError):
                    quote_timestamp = None
            now = datetime.now(ZoneInfo("Asia/Shanghai"))
            in_market_hours = now.weekday() < 5 and (
                datetime_time(9, 30) <= now.time() <= datetime_time(11, 30)
                or datetime_time(13, 0) <= now.time() <= datetime_time(15, 0)
            )
            if "过期" in meta["status"]:
                data_status = meta["status"]
            elif quote_timestamp:
                quote_age = (now - datetime.fromisoformat(quote_timestamp)).total_seconds()
                source_fresh = in_market_hours and 0 <= quote_age <= 600
                if source_fresh and meta["status"] == "接口获取":
                    data_status = f"盘中行情快照（源时间 {quote_timestamp}）"
                elif source_fresh and meta["status"] == "本地缓存":
                    data_status = f"本地缓存行情快照（源时间 {quote_timestamp}）"
                else:
                    data_status = f"最近可用行情快照（源时间 {quote_timestamp}，非实时）"
            else:
                data_status = "行情时间未知（非实时确认），仅供参考"
            quote = {
                "code": code,
                "name": payload.get("f58") or "未命名",
                "market": "北交所" if code.startswith(("4", "8", "92")) else ("沪市" if _market_id(code) == 1 else "深市"),
                "price": price,
                "change": change,
                "change_pct": change_pct,
                "volume_lots": _number(payload.get("f47")),
                "amount_yuan": _number(payload.get("f48")),
                "open": _scaled(payload.get("f46"), scale),
                "high": _scaled(payload.get("f44"), scale),
                "low": _scaled(payload.get("f45"), scale),
                "previous_close": previous_close,
                "turnover_rate": _number(payload.get("f168")),
                "pe_ttm": _number(payload.get("f162")),
                "quote_time": quote_timestamp,
                "source": "东方财富公开行情接口",
                "fetched_at": meta["fetched_at"],
                "data_status": data_status,
                "quote_valid": "过期" not in data_status and "未知" not in data_status,
                "warning": meta.get("warning"),
            }
            # 过期缓存只能兜底展示，不能阻止仍然可用的备用数据源。
            if quote["price"] is None or not quote["quote_valid"]:
                raise MarketDataError(quote.get("warning") or quote["data_status"])
            return quote
        except (MarketDataError, requests.RequestException, ValueError, KeyError, TypeError) as eastmoney_error:
            try:
                sina_quote = self._get_sina_quote(code, str(eastmoney_error))
                if sina_quote["quote_valid"]:
                    return sina_quote
                sina_warning = sina_quote["data_status"]
            except (MarketDataError, requests.RequestException, ValueError, KeyError, TypeError) as exc:
                sina_warning = brief_network_error(exc)
            warning = f"东方财富：{brief_network_error(eastmoney_error)}；新浪：{sina_warning}"
            try:
                return self._close_quote_fallback(
                    code, {**(quote or {"code": code}), "warning": warning}
                )
            except MarketDataError:
                # 全部源不可用时保留旧快照用于查看，仍明确标记不可参与模型。
                if quote and quote["price"] is not None:
                    quote["warning"] = warning
                    return quote
                raise

    def _get_sina_quote(self, code: str, primary_warning: str | None = None) -> dict[str, Any]:
        if code.startswith(("4", "8", "9")):
            raise MarketDataError("新浪备用快照暂不支持北交所代码。")
        market_prefix = "sh" if _market_id(code) == 1 else "sz"
        url = settings.SINA_QUOTE_URL.format(symbol=f"{market_prefix}{code}")
        response = self._request_response(
            url, {}, code, headers={"Referer": "https://finance.sina.com.cn/"}
        )
        text = response.content.decode("gbk", errors="replace")
        match = re.search(r'=[\"]([^\"]*)[\"]', text)
        if not match or not match.group(1):
            raise MarketDataError("新浪行情源未返回有效快照。")
        parts = match.group(1).split(",")
        if len(parts) < 32:
            raise MarketDataError("新浪行情源快照字段不完整。")
        price = _number(parts[3])
        previous_close = _number(parts[2])
        if price is None or price <= 0:
            raise MarketDataError("新浪行情源未返回有效成交价。")
        try:
            quote_time = datetime.fromisoformat(f"{parts[30]}T{parts[31]}+08:00")
        except ValueError as exc:
            raise MarketDataError("新浪行情源未提供有效行情时间。") from exc
        now = datetime.now(ZoneInfo("Asia/Shanghai"))
        age = (now - quote_time).total_seconds()
        in_market_hours = now.weekday() < 5 and (
            datetime_time(9, 30) <= now.time() <= datetime_time(11, 30)
            or datetime_time(13, 0) <= now.time() <= datetime_time(15, 0)
        )
        if in_market_hours and 0 <= age <= 600:
            data_status = f"盘中行情快照（新浪源时间 {quote_time.isoformat(timespec='seconds')}）"
        elif quote_time.date() == now.date() and now.time() >= datetime_time(15, 0):
            data_status = f"最近收盘行情（新浪源时间 {quote_time.isoformat(timespec='seconds')}，非实时）"
        else:
            data_status = f"新浪行情快照过期（源时间 {quote_time.isoformat(timespec='seconds')}，非实时）"
        volume_shares = _number(parts[8])
        amount_yuan = _number(parts[9])
        change = price - previous_close if previous_close and previous_close > 0 else None
        return {
            "code": code, "name": parts[0], "market": "沪市" if market_prefix == "sh" else "深市",
            "price": price, "change": change,
            "change_pct": change / previous_close * 100 if change is not None else None,
            "volume_lots": volume_shares / 100 if volume_shares is not None else None,
            "amount_yuan": amount_yuan, "open": _number(parts[1]),
            "high": _number(parts[4]), "low": _number(parts[5]),
            "previous_close": previous_close, "turnover_rate": None, "pe_ttm": None,
            "quote_time": quote_time.isoformat(timespec="seconds"), "source": "新浪财经行情接口",
            "fetched_at": _now_text(), "data_status": data_status,
            "quote_valid": "过期" not in data_status,
            "warning": f"东方财富接口失败，已切换新浪备用源：{primary_warning}",
        }

    def _close_quote_fallback(
        self, code: str, partial: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            history, meta = self.get_history(code, count=5, adjusted=False)
            if history.empty:
                raise MarketDataError("沒有可用日线行情。")
            bar = history.iloc[-1]
            last_date = str(bar["date"])[:10]
            return {
                "code": code,
                "name": partial.get("name", code),
                "market": "北交所" if code.startswith(("4", "8", "92")) else ("沪市" if _market_id(code) == 1 else "深市"),
                "price": _number(bar.get("close")),
                "change": _number(bar.get("change")),
                "change_pct": _number(bar.get("pct_change")),
                "volume_lots": _number(bar.get("volume")),
                "amount_yuan": _number(bar.get("amount")),
                "open": _number(bar.get("open")),
                "high": _number(bar.get("high")),
                "low": _number(bar.get("low")),
                "previous_close": None,
                "turnover_rate": _number(bar.get("turnover_rate")),
                "quote_time": last_date,
                "source": meta.get("source", "历史行情接口"),
                "fetched_at": meta["fetched_at"],
                "data_status": (
                    "最近可用收盘行情（非实时）"
                    if meta.get("status") in ("接口获取", "本地缓存")
                    else "过期缓存收盘行情（非实时）"
                ),
                "quote_valid": "过期" not in meta.get("status", ""),
                "warning": partial.get("warning"),
            }
        except (MarketDataError, requests.RequestException, ValueError, KeyError, TypeError) as exc:
            raise MarketDataError(f"快照与收盘行情均不可用：{exc}") from exc

    def quote_from_universe_row(
        self, row: dict[str, Any], meta: dict[str, Any]
    ) -> dict[str, Any]:
        fetched_at = meta.get("fetched_at")
        age_seconds = float("inf")
        if fetched_at:
            try:
                fetched = datetime.fromisoformat(fetched_at)
                age_seconds = (datetime.now(fetched.tzinfo) - fetched).total_seconds()
            except (TypeError, ValueError):
                pass
        now = datetime.now(ZoneInfo("Asia/Shanghai"))
        in_market_hours = now.weekday() < 5 and (
            datetime_time(9, 30) <= now.time() <= datetime_time(11, 30)
            or datetime_time(13, 0) <= now.time() <= datetime_time(15, 0)
        )
        is_stale = bool(meta.get("is_stale")) or age_seconds < 0 or age_seconds > settings.UNIVERSE_CACHE_SECONDS
        if is_stale:
            data_status = "股票池快照已过期（非实时）"
        elif in_market_hours:
            data_status = f"盘中全市场接口快照（采集时间 {fetched_at}；未提供逐股源时间）"
        else:
            data_status = f"最近收盘全市场行情快照（采集时间 {fetched_at}；非实时）"
        code = str(row.get("code", ""))
        market = "沪市" if code.startswith(("5", "6")) else ("北交所" if code.startswith(("4", "8", "92")) else "深市")
        return {
            "code": code, "name": str(row.get("name", "")), "market": market,
            "price": _number(row.get("price")), "change": _number(row.get("change")),
            "change_pct": _number(row.get("change_pct")), "volume_lots": _number(row.get("volume_lots")),
            "amount_yuan": _number(row.get("amount_yuan")), "open": _number(row.get("open")),
            "high": _number(row.get("high")), "low": _number(row.get("low")),
            "previous_close": _number(row.get("previous_close")),
            "turnover_rate": _number(row.get("turnover_rate")), "pe_ttm": None,
            "quote_time": None, "fetched_at": fetched_at, "source": meta.get("source"),
            "data_status": data_status, "quote_valid": not is_stale,
            "warning": meta.get("warning"),
        }

    def get_scan_quote(self, candidate: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
        """扫描超出快照有效期时重新读取股票池；新缓存可供后续股票共同复用。"""
        if candidate.get("price") is None:
            quote = self.get_quote(str(candidate["code"]))
        else:
            quote = self.quote_from_universe_row(candidate, meta)
        if quote["quote_valid"]:
            return quote
        universe, refreshed_meta = self.get_universe(scope=meta.get("scope", "all"))
        matched = universe.loc[universe["code"].astype(str) == str(candidate["code"])]
        if matched.empty:
            raise MarketDataError("股票已不在当前有效股票池中。")
        return self.quote_from_universe_row(matched.iloc[0].to_dict(), refreshed_meta)

    def get_history(
        self,
        code: str,
        count: int = 800,
        period: str = "daily",
        adjusted: bool = True,
        market_id: int | None = None,
        force_refresh: bool = False,
        expected_latest_trading_date: str | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        code = _clean_code(code)
        if period not in ("daily", "weekly"):
            raise ValueError("period 仅支持 daily 或 weekly。")
        klt = 101 if period == "daily" else 102
        adjust = 1 if adjusted else 0
        resolved_market_id = market_id if market_id in (0, 1) else _market_id(code)
        cache_key = f"history:v2:{code}:{period}:{count}:{adjust}:{resolved_market_id}"
        params = {
            "secid": f"{resolved_market_id}.{code}",
            "klt": klt,
            "fqt": adjust,
            "lmt": min(max(int(count), 2), 1500),
            "end": "20500101",
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        }
        source_used = {"name": "东方财富公开日线接口"}

        def parse_eastmoney_rows(response: dict[str, Any]) -> list[dict[str, Any]]:
            data = response.get("data") or {}
            if data.get("code") and str(data["code"]) != code:
                raise MarketDataError("行情源返回的 K 线代码与请求代码不一致。")
            rows = []
            for line in data.get("klines") or []:
                parts = line.split(",")
                if len(parts) < 11:
                    continue
                rows.append(
                    {
                        "date": parts[0],
                        "open": _number(parts[1]),
                        "close": _number(parts[2]),
                        "high": _number(parts[3]),
                        "low": _number(parts[4]),
                        "volume": _number(parts[5]),
                        "amount": _number(parts[6]),
                        "amplitude": _number(parts[7]),
                        "pct_change": _number(parts[8]),
                        "change": _number(parts[9]),
                        "turnover_rate": _number(parts[10]),
                    }
                )
            return rows

        def load_rows() -> dict[str, Any]:
            old_cache = self._cache_result(cache_key, settings.HISTORY_CACHE_SECONDS, allow_stale=True)
            old_rows: list[dict[str, Any]] = []
            old_fetched: datetime | None = None
            if old_cache is not None:
                old_payload, old_fetched, _ = old_cache
                old_rows = old_payload.get("rows", []) if isinstance(old_payload, dict) else old_payload
            last_full_text = old_payload.get("last_full_refresh_at") if old_cache is not None and isinstance(old_payload, dict) else None
            last_full = datetime.fromisoformat(last_full_text) if last_full_text else None
            # 前复权历史会整体改写，刷新时必须完整替换；不复权数据才允许增量拼接。
            full_refresh = (
                adjusted or not old_rows or last_full is None
                or (datetime.now(last_full.tzinfo) - last_full).total_seconds() >= settings.FULL_HISTORY_REFRESH_SECONDS
            )
            begin_date = "19900101"
            request_params = dict(params)
            if old_rows and not full_refresh:
                latest_date = max(str(row["date"])[:10] for row in old_rows)
                begin_date = (pd.Timestamp(latest_date) - pd.Timedelta(days=settings.HISTORY_INCREMENTAL_LOOKBACK_DAYS)).strftime("%Y%m%d")
                request_params["beg"] = begin_date
                request_params["lmt"] = max(40, settings.HISTORY_INCREMENTAL_LOOKBACK_DAYS * 4)
            def load_eastmoney_history() -> list[dict[str, Any]]:
                response = self._get_json(settings.EASTMONEY_KLINE_URL, request_params, code=code)
                return parse_eastmoney_rows(response)

            loaders = [
                ("东方财富", "东方财富公开日线接口", load_eastmoney_history),
                (
                    "腾讯", "腾讯证券历史行情接口",
                    lambda: self._load_tencent_history(
                        code, adjusted, "19900101" if full_refresh else begin_date,
                        period, int(count), resolved_market_id,
                    ),
                ),
            ]
            # 主源网络故障且备用源成功后，五分钟内优先复用备用源；失败仍尝试主源。
            if self._history_tencent_until > time.monotonic():
                loaders.reverse()
            errors = []
            eastmoney_network_failed = False
            for provider, source, loader in loaders:
                try:
                    new_rows = loader()
                    if not new_rows:
                        raise MarketDataError("行情源未返回历史数据。")
                    source_used["name"] = source
                    if provider == "腾讯" and eastmoney_network_failed:
                        self._history_tencent_until = time.monotonic() + 300
                    elif provider == "东方财富":
                        self._history_tencent_until = 0.0
                    break
                except (MarketDataError, requests.RequestException, ValueError, KeyError, TypeError) as exc:
                    errors.append(f"{provider}历史行情失败：{brief_network_error(exc)}")
                    if provider == "东方财富" and isinstance(exc, requests.RequestException):
                        eastmoney_network_failed = True
            else:
                # 新浪 A 股接口依赖流通股数据，不适用于 ETF 或显式指定沪市的指数。
                sina_supported = code.startswith(("00", "30", "60", "68")) and resolved_market_id == _market_id(code)
                if not sina_supported:
                    raise MarketDataError("；".join(errors))
                try:
                    new_rows = self._load_sina_history(
                        code, adjusted, "19900101" if full_refresh else begin_date, period
                    )
                    source_used["name"] = "新浪财经日线接口（AKShare）"
                except (MarketDataError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as sina_error:
                    errors.append(f"新浪备用日线失败：{brief_network_error(sina_error)}")
                    raise MarketDataError("；".join(errors)) from sina_error
            if not new_rows:
                raise MarketDataError("主备源均未返回新历史数据。")
            by_date = {} if full_refresh else {str(row["date"])[:10]: row for row in old_rows}
            by_date.update({str(row["date"])[:10]: row for row in new_rows})
            combined = [by_date[key] for key in sorted(by_date)][-min(max(int(count), 2), 1500):]
            return {
                "rows": combined, "source": source_used["name"],
                "last_full_refresh_at": _now_text() if full_refresh else last_full_text,
            }

        ttl = settings.HISTORY_CACHE_SECONDS
        cached = self._read_cache(cache_key)
        if cached is not None:
            _, fetched = cached
            now = datetime.now(ZoneInfo("Asia/Shanghai"))
            # 跨越收盘后重新读取，避免把盘中缓存沿用至收盘分析。
            if fetched.astimezone(ZoneInfo("Asia/Shanghai")).date() < now.date() or (
                fetched.time() < datetime_time(15, 0) <= now.time()
            ):
                ttl = -1
        if force_refresh:
            ttl = -1
        # 没有新交易日、且缓存抓取已晚于该交易日收盘时，直接复用前复权历史。
        # 一旦缓存最新 K 线早于预期交易日（出现新交易日），保持原有整段替换逻辑。
        if (
            ttl < 0 and not force_refresh and cached is not None and expected_latest_trading_date
            and settings.HISTORY_CACHE_REUSE_AFTER_CLOSE
        ):
            payload, fetched = cached
            rows_for_reuse = payload.get("rows", []) if isinstance(payload, dict) else payload
            latest_text = max(
                (str(row.get("date", ""))[:10] for row in rows_for_reuse if isinstance(row, dict)),
                default="",
            )
            try:
                expected_day = pd.Timestamp(str(expected_latest_trading_date)[:10]).date()
            except (TypeError, ValueError):
                expected_day = None
            fetched_local = fetched.astimezone(ZoneInfo("Asia/Shanghai"))
            expected_close = datetime.combine(
                expected_day, datetime_time(15, 0), tzinfo=ZoneInfo("Asia/Shanghai")
            ) if expected_day else None
            if latest_text and expected_day is not None and expected_close is not None:
                if latest_text >= expected_day.isoformat() and fetched_local >= expected_close:
                    ttl = 10 ** 9
        try:
            payload, meta = self._request_with_cache(cache_key, ttl, load_rows)
        except (MarketDataError, requests.RequestException, ValueError, KeyError, TypeError) as exc:
            self._record_history_status(
                code, period, "前复权" if adjusted else "不复权", "失败", None, 0, None,
                str(exc), self._last_retries.get((self._interface_name(settings.EASTMONEY_KLINE_URL), code), 0),
            )
            raise
        rows = payload.get("rows", []) if isinstance(payload, dict) else payload
        if not rows:
            message = f"{code} 未取得有效历史 K 线。"
            self._record_history_status(code, period, "前复权" if adjusted else "不复权", "失败", None, 0, None, message, 0)
            raise MarketDataError(message)
        frame = pd.DataFrame(rows)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        for column in frame.columns.difference(["date"]):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        if frame.empty or not frame["close"].gt(0).all():
            raise MarketDataError("历史行情没有有效正收盘价。")
        # 加载阶段即标注停牌、一字板与前复权异常跳空；下游因子与成交模拟只使用 is_usable 为真的 K 线。
        try:
            frame, cleaning = clean_daily_history(frame, code=code, adjusted=adjusted)
        except ValueError as exc:
            raise MarketDataError(f"行情清洗失败：{exc}") from exc
        source = payload.get("source") if isinstance(payload, dict) else None
        if meta.get("status") in ("本地缓存", "行情源失败，使用过期缓存（非实时）"):
            source = f"SQLite 历史行情缓存（原始来源：{source or '未知'}）"
        else:
            source = source or source_used["name"]
        retry_count = self._last_retries.get((self._interface_name(settings.EASTMONEY_KLINE_URL), code), 0)
        history_status = "过期缓存" if "过期" in meta.get("status", "") else ("缓存" if meta.get("status") == "本地缓存" else "成功")
        meta.update(
            {
                "source": source,
                "period": period,
                "adjustment": "前复权" if adjusted else "不复权",
                "bar_count": int(len(frame)),
                "latest_bar_date": frame["date"].iloc[-1].strftime("%Y-%m-%d") if not frame.empty else None,
                "retry_count": retry_count,
                "cleaning": cleaning,
            }
        )
        # 列式面板作为派生产物写穿保存；写入失败不影响已取得的行情，但必须显式记录。
        try:
            self.panels.write_history(code, frame, "qfq" if adjusted else "raw", meta, period=period)
            meta["panel_store"] = "已写入 Parquet 列式面板"
        except (OSError, ValueError, ImportError) as exc:
            meta["panel_store"] = f"Parquet 写入失败：{type(exc).__name__}"
            self.logger.warning("写入 Parquet 面板失败 code=%s period=%s detail=%s", code, period, exc)
        self._record_history_status(
            code, period, meta["adjustment"], history_status, meta["latest_bar_date"],
            len(frame), source, meta.get("warning"), retry_count,
        )
        return frame, meta

    def _load_tencent_history(
        self, code: str, adjusted: bool, start_date: str, period: str,
        count: int, market_id: int,
    ) -> list[dict[str, Any]]:
        """按日期向前分页读取腾讯行情，保留复权口径并统一成交量、金额单位。"""
        if code.startswith(("4", "8", "92")):
            raise MarketDataError("腾讯备用历史行情暂不支持北交所代码。")
        symbol = ("sh" if market_id == 1 else "sz") + code
        interval = "day" if period == "daily" else "week"
        adjustment = "qfq" if adjusted else ""
        is_index = symbol.startswith(("sh000", "sz399"))
        target = min(max(int(count), 2), 1500)
        # 多取一根用于计算首根展示 K 线的涨跌幅；腾讯单页最多返回 640 根。
        wanted = target + 1
        page_size = min(640, wanted)
        beginning = pd.Timestamp(start_date)
        ending = ""
        by_date: dict[str, dict[str, Any]] = {}
        for page in range(5):
            response = self._get_json(
                settings.TENCENT_KLINE_URL,
                {"param": f"{symbol},{interval},,{ending},{page_size},{adjustment}"},
                code=code,
            )
            instrument = (response.get("data") or {}).get(symbol) or {}
            snapshot = (instrument.get("qt") or {}).get(symbol) or []
            if snapshot and (len(snapshot) < 3 or str(snapshot[2]) != code):
                raise MarketDataError("腾讯行情返回的证券代码与请求不一致。")
            bars = instrument.get(f"{adjustment}{interval}")
            # 指数没有除权除息；普通股票和 ETF 不得把未复权行情冒充前复权行情。
            if bars is None and is_index:
                bars = instrument.get(interval)
            if bars is None:
                raise MarketDataError(f"腾讯未返回所需的{'前复权' if adjusted else '不复权'}{period}行情。")
            if not bars:
                break
            dates = []
            for bar in bars:
                if not isinstance(bar, list) or len(bar) < 6:
                    raise MarketDataError("腾讯 K 线字段不完整，未使用部分历史数据。")
                date = pd.Timestamp(bar[0])
                if pd.isna(date):
                    raise MarketDataError("腾讯 K 线日期无效。")
                if ending and date > pd.Timestamp(ending):
                    raise MarketDataError("腾讯历史分页未遵守截止日期，未使用部分结果。")
                dates.append(date)
                if date < beginning:
                    continue
                prices = [_number(value) for value in bar[1:5]]
                if any(value is None or value <= 0 for value in prices):
                    raise MarketDataError("腾讯 K 线存在无效价格，未使用部分历史数据。")
                volume = _number(bar[5])
                # 科创板和指数源成交量为股，其余沪深证券为手；本项目统一使用手。
                if volume is not None and (symbol.startswith("sh688") or is_index):
                    volume /= 100
                amount = _number(bar[8]) if len(bar) > 8 else None
                by_date[date.strftime("%Y-%m-%d")] = {
                    "date": date.strftime("%Y-%m-%d"),
                    "open": prices[0], "close": prices[1], "high": prices[2], "low": prices[3],
                    "volume": volume, "amount": amount * 10000 if amount is not None else None,
                    "turnover_rate": _number(bar[7]) if len(bar) > 7 else None,
                }
            earliest = min(dates)
            if len(by_date) >= wanted or earliest <= beginning or len(bars) < page_size:
                break
            if ending and earliest >= pd.Timestamp(ending):
                raise MarketDataError("腾讯历史分页未向前推进，未使用部分结果。")
            ending = (earliest - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            raise MarketDataError("腾讯历史分页超过预期次数，未使用不完整结果。")
        if not by_date:
            raise MarketDataError("腾讯未返回所选日期范围的历史行情。")
        frame = pd.DataFrame([by_date[date] for date in sorted(by_date)])
        previous_close = frame["close"].shift(1)
        frame["change"] = frame["close"] - previous_close
        frame["pct_change"] = frame["change"] / previous_close * 100
        frame["amplitude"] = (frame["high"] - frame["low"]) / previous_close * 100
        result = frame.tail(target).copy()
        return result.astype(object).where(pd.notna(result), None).to_dict("records")

    def _load_sina_history(
        self, code: str, adjusted: bool, start_date: str, period: str
    ) -> list[dict[str, Any]]:
        if not code.startswith(("00", "30", "60", "68")):
            raise MarketDataError("新浪 A 股备用接口不支持该证券类型；ETF 历史请使用东财或腾讯源。")
        symbol = ("sh" if _market_id(code) == 1 else "sz") + code
        adjustment = "qfq" if adjusted else ""
        script = f"""
import akshare as ak
import pandas as pd
import requests
from services.finance_network import FinanceNetwork

session = requests.Session()
session.headers.update({{"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/"}})
network = FinanceNetwork(session)

def bounded_get(url, **arguments):
    # 子进程同样使用有限请求超时与限定域名回退，避免库内部无限等待。
    arguments.setdefault("timeout", ({settings.REQUEST_CONNECT_TIMEOUT_SECONDS}, {settings.REQUEST_READ_TIMEOUT_SECONDS}))
    return network.get(url, **arguments)

requests.get = bounded_get
try:
    frame = ak.stock_zh_a_daily(symbol={symbol!r}, start_date={start_date!r}, end_date="21000101", adjust={adjustment!r})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    print(frame.to_json(orient="records", force_ascii=False))
finally:
    network.close()
    session.close()
"""
        interface = "Sina.stock_zh_a_daily"
        source_key = "sina.history"
        health = self._source_health()
        if not health.allow(source_key):
            raise MarketDataError("新浪备用日线源暂时熔断，本轮直接跳过。")
        try:
            last_error: Exception | None = None
            for attempt in range(1, settings.REQUEST_RETRIES + 2):
                with self._request_gate:
                    self._pace_request()
                    started = time.perf_counter()
                    try:
                        completed = subprocess.run(
                            [sys.executable, "-X", "utf8", "-c", script], capture_output=True, check=True,
                            encoding="utf-8", text=True, timeout=settings.SINA_HISTORY_TIMEOUT_SECONDS,
                            cwd=settings.PROJECT_ROOT,
                        )
                        output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
                        records = json.loads(output_lines[-1]) if output_lines else []
                        frame = pd.DataFrame(records)
                        if frame.empty:
                            raise MarketDataError("新浪财经未返回日线记录。")
                        frame = frame.rename(columns={"turnover": "turnover_rate"})
                        for column in ("open", "high", "low", "close", "volume", "amount", "turnover_rate"):
                            if column in frame:
                                frame[column] = pd.to_numeric(frame[column], errors="coerce")
                        if "volume" in frame:
                            frame["volume"] = frame["volume"] / 100
                        if "turnover_rate" in frame:
                            frame["turnover_rate"] = frame["turnover_rate"] * 100
                        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
                        frame = frame.dropna(subset=["date", "close"]).sort_values("date")
                        if period == "weekly":
                            frame = frame.set_index("date").resample("W-FRI").agg(
                                {"open": "first", "high": "max", "low": "min",
                                 "close": "last", "volume": "sum", "amount": "sum",
                                 "turnover_rate": "mean"}
                            ).dropna(subset=["close"]).reset_index()
                        frame["change"] = frame["close"].diff()
                        frame["pct_change"] = frame["close"].pct_change() * 100
                        frame["amplitude"] = (frame["high"] - frame["low"]) / frame["close"] * 100
                        result = frame.dropna(subset=["date", "close"]).tail(1500)
                        result["date"] = result["date"].dt.strftime("%Y-%m-%d")
                        rows = result.astype(object).where(pd.notna(result), None).to_dict("records")
                        elapsed_ms = (time.perf_counter() - started) * 1000
                        health.record_success(source_key)
                        self._record_request_event(interface, code, elapsed_ms, None, attempt - 1, True, None)
                        self.logger.info(
                            "request success interface=%s code=%s elapsed_ms=%.1f retries=%s",
                            interface, code, elapsed_ms, attempt - 1,
                        )
                        return rows
                    except (
                        subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError,
                        MarketDataError, ValueError, KeyError, TypeError,
                    ) as exc:
                        last_error = exc
                        elapsed_ms = (time.perf_counter() - started) * 1000
                        detail = self._exception_detail(exc, "新浪财经备用日线失败")
                        self._record_request_event(
                            interface, code, elapsed_ms, type(exc).__name__, attempt - 1, False, str(detail)
                        )
                        self.logger.warning(
                            "request failure interface=%s code=%s elapsed_ms=%.1f exception=%s retries=%s detail=%s",
                            interface, code, elapsed_ms, type(exc).__name__, attempt - 1, str(detail)[:300],
                        )
                if attempt <= settings.REQUEST_RETRIES:
                    time.sleep(settings.RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)))
            if isinstance(last_error, (subprocess.TimeoutExpired, subprocess.CalledProcessError)):
                health.record_failure(source_key)
            raise MarketDataError(f"{interface} 获取失败：{brief_network_error(last_error)}") from last_error
        finally:
            health.release_probe(source_key)

    def _universe_state(
        self,
    ) -> tuple[
        dict[str, tuple[pd.DataFrame, dict[str, Any], float]],
        dict[str, float],
        dict[str, threading.Lock],
        threading.Lock,
    ]:
        """惰性初始化股票池快照状态，兼容测试里 `__new__` 构造的实例。"""
        memory = getattr(self, "_universe_memory", None)
        if memory is None:
            memory = {}
            self._universe_memory = memory
        retry_after = getattr(self, "_universe_retry_after", None)
        if retry_after is None:
            retry_after = {}
            self._universe_retry_after = retry_after
        locks = getattr(self, "_universe_locks", None)
        if locks is None:
            locks = {}
            self._universe_locks = locks
        meta_lock = getattr(self, "_universe_meta_lock", None)
        if meta_lock is None:
            meta_lock = threading.Lock()
            self._universe_meta_lock = meta_lock
        return memory, retry_after, locks, meta_lock

    def _universe_snapshot(self, scope: str) -> tuple[pd.DataFrame, dict[str, Any]] | None:
        memory, _, _, _ = self._universe_state()
        entry = memory.get(scope)
        if entry is None:
            return None
        frame, meta, stored_at = entry
        if time.monotonic() - stored_at >= settings.UNIVERSE_CACHE_SECONDS:
            memory.pop(scope, None)
            return None
        return frame, meta

    def _remember_universe(self, scope: str, frame: pd.DataFrame, meta: dict[str, Any]) -> None:
        """按来源抓取时间记快照，避免内存缓存把已过半的磁盘 TTL 再续一整轮。"""
        age = 0.0
        try:
            fetched = datetime.fromisoformat(str(meta.get("fetched_at")))
            age = max(0.0, (datetime.now(fetched.tzinfo) - fetched).total_seconds())
        except (TypeError, ValueError):
            pass
        memory, _, _, _ = self._universe_state()
        memory[scope] = (frame.copy(), dict(meta), time.monotonic() - age)

    def _universe_lock_for(self, scope: str) -> threading.Lock:
        _, _, locks, meta_lock = self._universe_state()
        with meta_lock:
            lock = locks.get(scope)
            if lock is None:
                lock = threading.Lock()
                locks[scope] = lock
            return lock

    def get_universe(
        self, scope: str = "all", force_refresh: bool = False
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """股票池单飞读取：同一轮扫描的并发线程共用一份快照，且只真正刷新一次。

        返回副本：`services/scanning.py::candidate_codes` 会原地加列并过滤，共享同一个
        DataFrame 会让后一个调用方看到前一个调用方的中间状态。
        """
        if scope not in ("all", "main", "chinext", "star", "bj"):
            raise ValueError("scope 必须是 all、main、chinext、star 或 bj。")
        _, retry_after, _, _ = self._universe_state()
        if not force_refresh:
            snapshot = self._universe_snapshot(scope)
            if snapshot is not None:
                frame, meta = snapshot
                return frame.copy(), dict(meta)
            blocked_until = retry_after.get(scope, 0.0)
            if blocked_until > time.monotonic():
                wait = max(1, int(blocked_until - time.monotonic()))
                raise MarketDataError(f"股票池刷新连续失败，{wait} 秒内不重复请求；请稍后重试。")
        with self._universe_lock_for(scope):
            if not force_refresh:
                snapshot = self._universe_snapshot(scope)
                if snapshot is not None:
                    frame, meta = snapshot
                    return frame.copy(), dict(meta)
            try:
                frame, meta = self._load_universe(scope, force_refresh)
            except MarketDataError:
                retry_after[scope] = time.monotonic() + 30
                raise
            retry_after.pop(scope, None)
            self._remember_universe(scope, frame, meta)
            return frame.copy(), dict(meta)

    def _load_universe(
        self, scope: str, force_refresh: bool
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        cache_key = f"universe:{scope}"
        cached = None if force_refresh else self._cache_result(
            cache_key, settings.UNIVERSE_CACHE_SECONDS, allow_stale=False
        )
        if cached is not None and self._valid_universe_cache(cached[0], scope):
            payload, fetched, _ = cached
            rows = payload["rows"]
            return pd.DataFrame(rows), {
                **payload["meta"], "fetched_at": fetched.isoformat(timespec="seconds"),
                "status": "完整股票池本地缓存", "is_stale": False,
            }

        eastmoney_error: Exception | None = None
        source_warning = None
        try:
            rows, source_total = self._fetch_eastmoney_universe(scope)
            source = "东方财富公开行情接口（完整分页）"
        except (requests.RequestException, ValueError, KeyError, TypeError, MarketDataError) as exc:
            eastmoney_error = exc
            try:
                rows, source = self._load_akshare_spot(scope)
                source_total = None
                source_warning = f"东方财富分页接口失败，已切换备用源：{brief_network_error(exc)}"
            except Exception as fallback_error:
                stale = self._cache_result(cache_key, settings.UNIVERSE_CACHE_SECONDS, allow_stale=True)
                if stale is not None and self._valid_universe_cache(stale[0], scope):
                    payload, fetched, _ = stale
                    return pd.DataFrame(payload["rows"]), {
                        **payload["meta"], "fetched_at": fetched.isoformat(timespec="seconds"),
                        "status": "股票池刷新失败，使用完整过期缓存（非实时）",
                        "is_stale": True,
                        "warning": f"东方财富: {brief_network_error(eastmoney_error)}；备用源: {brief_network_error(fallback_error)}",
                    }
                self.logger.error("完整股票池获取失败；东方财富=%s；备用源=%s", eastmoney_error, fallback_error)
                raise MarketDataError(
                    f"完整股票池获取失败，未使用首屏或部分结果；东方财富: {brief_network_error(eastmoney_error)}；备用源: {brief_network_error(fallback_error)}"
                ) from fallback_error

        raw_frame = self._map_universe_rows(rows)
        source_unique_codes = int(raw_frame["code"].nunique()) if not raw_frame.empty else 0
        if scope != "all":
            raw_frame = raw_frame.loc[raw_frame["code"].str.startswith(settings.BOARD_PREFIXES[scope])].copy()
        unique_codes = int(raw_frame["code"].nunique()) if not raw_frame.empty else 0
        # 分页完整性针对源端全量：必须拿过滤号段之前的去重数比对。
        # 用过滤后的数量比对，会让任何一个未登记的号段被误报成“分页不完整”。
        if source_total is not None and source_unique_codes < source_total:
            raise MarketDataError(
                f"股票池分页不完整：源端总数 {source_total}，去重后仅 {source_unique_codes} 只；拒绝缓存不完整股票池。"
            )
        valid_frame, rejected = self._filter_universe_rows(raw_frame)
        if valid_frame.empty:
            raise MarketDataError("完整股票池中没有有效、非 ST 且有成交的 A 股行情。")
        minimums = {
            "all": settings.MINIMUM_EXPECTED_A_SHARE_UNIVERSE,
            "main": 2000, "chinext": 1000, "star": 400, "bj": 150,
        }
        if source_total is None and unique_codes < minimums[scope]:
            raise MarketDataError(
                f"备用股票池疑似只有首屏：唯一代码 {unique_codes} 只，{scope} 范围至少应有 {minimums[scope]} 只。"
            )
        if len(valid_frame) < minimums[scope]:
            raise MarketDataError(
                f"{scope} 股票池有效股票仅 {len(valid_frame)} 只，低于规模校验阈值 {minimums[scope]}。"
            )
        if scope == "all":
            bj_count = int(valid_frame["code"].str.startswith(("43", "83", "87", "92")).sum())
            if bj_count == 0:
                raise MarketDataError("完整沪深京 A 股池缺少有效北交所股票，拒绝标记为完整。")
        fetched_at = _now_text()
        duplicate_count = max(0, len(rows) - source_unique_codes)
        network_status = self.network.status()
        if network_status["direct_fallback_hosts"]:
            direct_warning = "行情代理连接失败后已通过公开行情接口直连恢复；系统代理设置保持不变。"
            source_warning = f"{source_warning}；{direct_warning}" if source_warning else direct_warning
        meta = {
            "source": source, "source_total_symbols": int(source_total or unique_codes),
            "source_rows": int(len(rows)), "source_unique_symbols": int(unique_codes),
            "valid_symbols": int(len(valid_frame)), "excluded_symbols": int(rejected + duplicate_count),
            "failed_symbols": 0, "failed_pages": int(source_warning is not None),
            "complete": True, "is_stale": False, "scope": scope, "warning": source_warning,
            "network": network_status,
        }
        payload = {"rows": valid_frame.to_dict("records"), "meta": meta}
        fetched_at = self._write_cache(cache_key, payload)
        return valid_frame, {**meta, "fetched_at": fetched_at, "status": "完整股票池获取成功"}

    def _fetch_eastmoney_universe(self, scope: str) -> tuple[list[dict[str, Any]], int]:
        filters = {
            "all": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048",
            "main": "m:1+t:2,m:0+t:6",
            "chinext": "m:0+t:80", "star": "m:1+t:23",
            "bj": "m:0+t:81+s:2048",
        }
        page_size = settings.LIST_PAGE_SIZE
        rows: list[dict[str, Any]] = []
        total_symbols: int | None = None
        for page in range(1, 101):
            params = {
                "pn": page, "pz": page_size, "po": 1, "np": 1,
                "fltt": 2, "invt": 2, "fid": "f12", "fs": filters[scope],
                # f20/f21 为总市值与流通市值，f100 为东财行业，用于截面中性化暴露。
                "fields": "f2,f3,f4,f5,f6,f8,f12,f14,f15,f16,f17,f18,f20,f21,f100,f60,f170",
            }
            payload = self._get_json(settings.EASTMONEY_LIST_URL, params, code=f"股票池:{scope}:page{page}")
            data = payload.get("data") or {}
            if total_symbols is None:
                total_symbols = int(data.get("total") or 0)
                if total_symbols <= 0:
                    raise MarketDataError("股票池接口未返回源端总数，无法验证完整性。")
            diff = data.get("diff") or []
            items = diff.values() if isinstance(diff, dict) else diff
            page_rows = [
                {
                    "code": item.get("f12"), "name": item.get("f14"),
                    "price": _number(item.get("f2")), "change_pct": _number(item.get("f3")),
                    "change": _number(item.get("f4")), "volume_lots": _number(item.get("f5")),
                    "amount_yuan": _number(item.get("f6")), "turnover_rate": _number(item.get("f8")),
                    "high": _number(item.get("f15")), "low": _number(item.get("f16")),
                    "open": _number(item.get("f17")), "previous_close": _number(item.get("f18")),
                    "market_cap_total": _number(item.get("f20")), "market_cap_float": _number(item.get("f21")),
                    "industry": item.get("f100") if isinstance(item.get("f100"), str) else None,
                }
                for item in items if item.get("f12") and item.get("f14")
            ]
            rows.extend(page_rows)
            self.logger.info("universe page scope=%s page=%s rows=%s total=%s", scope, page, len(page_rows), total_symbols)
            if len(rows) >= total_symbols:
                break
            if not page_rows:
                raise MarketDataError(
                    f"股票池分页中断：第 {page} 页为空，已取 {len(rows)}/{total_symbols} 只。"
                )
        else:
            raise MarketDataError(f"股票池页数超过安全上限，已取 {len(rows)} 只。")
        unique_codes = {str(row.get("code", "")).zfill(6) for row in rows}
        if len(unique_codes) < total_symbols:
            raise MarketDataError(
                f"股票池存在分页重复或缺失：源端 {total_symbols} 只，唯一代码 {len(unique_codes)} 只。"
            )
        return rows, total_symbols

    def _load_akshare_spot(self, scope: str) -> tuple[list[dict[str, Any]], str]:
        cache_key = f"universe:akshare_spot:{scope}"
        cached = self._cache_result(cache_key, settings.UNIVERSE_CACHE_SECONDS, allow_stale=False)
        if cached is not None:
            payload, _, _ = cached
            return payload["rows"], payload["source"]
        errors = []
        try:
            rows = self._fetch_sina_universe()
            source = "新浪财经公开行情接口（完整分页）"
            self._write_cache(cache_key, {"rows": rows, "source": source})
            return rows, source
        except (requests.RequestException, MarketDataError, ValueError, KeyError, TypeError) as exc:
            self.logger.warning("新浪完整分页备用源失败 detail=%s", exc)
            errors.append("新浪完整分页：" + brief_network_error(exc))
        if scope == "all":
            strategies = [("stock_zh_a_spot_em",), ("stock_zh_a_spot",)]
        elif scope == "bj":
            strategies = [("stock_bj_a_spot_em",)]
        else:
            function = {"main": "stock_zh_a_spot_em", "chinext": "stock_sz_a_spot_em", "star": "stock_sh_a_spot_em"}[scope]
            strategies = [(function,), ("stock_zh_a_spot",)]
        minimums = {"all": settings.MINIMUM_EXPECTED_A_SHARE_UNIVERSE, "main": 2000, "chinext": 1000, "star": 400, "bj": 150}
        for strategy in strategies:
            combined_rows: list[dict[str, Any]] = []
            try:
                for function_name in strategy:
                    script = (
                        "import akshare as ak; "
                        f"frame=ak.{function_name}(); "
                        "print(frame.to_json(orient='records',force_ascii=False))"
                    )
                    started = time.perf_counter()
                    try:
                        completed = subprocess.run(
                            [sys.executable, "-X", "utf8", "-c", script], capture_output=True, check=True,
                            encoding="utf-8", text=True, timeout=settings.AKSHARE_SPOT_TIMEOUT_SECONDS,
                        )
                        output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
                        source_rows = json.loads(output_lines[-1]) if output_lines else []
                        if not source_rows:
                            raise MarketDataError("AkShare 行情快照为空。")
                        for row in source_rows:
                            if function_name == "stock_zh_a_spot":
                                volume = _number(row.get("volume"))
                                if volume is not None:
                                    row["volume_lots"] = volume / 100
                                row["amount_yuan"] = _number(row.get("amount"))
                            combined_rows.append(row)
                        elapsed_ms = (time.perf_counter() - started) * 1000
                        self._record_request_event("AkShare." + function_name, "股票池", elapsed_ms, None, 0, True, None)
                    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError, MarketDataError) as exc:
                        elapsed_ms = (time.perf_counter() - started) * 1000
                        detail = self._exception_detail(exc, "AkShare 备用行情失败")
                        self.logger.warning("备用股票池失败 function=%s detail=%s", function_name, detail)
                        self._record_request_event("AkShare." + function_name, "股票池", elapsed_ms, type(exc).__name__, 0, False, str(detail))
                        raise MarketDataError(f"AkShare {function_name} 失败：{brief_network_error(detail)}") from exc
                frame = self._map_universe_rows(combined_rows)
                unique_codes = int(frame["code"].nunique()) if not frame.empty else 0
                scope_codes = unique_codes if scope == "all" else int(frame["code"].str.startswith(settings.BOARD_PREFIXES[scope]).sum())
                if scope_codes < minimums[scope]:
                    raise MarketDataError(f"备用源仅取得 {scope_codes} 只 {scope} 范围股票，低于完整池校验阈值 {minimums[scope]}。")
                if scope == "all" and not frame["code"].str.startswith(("43", "83", "87", "92")).any():
                    raise MarketDataError("备用全市场列表缺少北交所代码，不能标记为完整沪深京 A 股。")
                payload = {"rows": combined_rows, "source": "+".join(strategy)}
                self._write_cache(cache_key, payload)
                return combined_rows, "AkShare: " + " + ".join(strategy)
            except (MarketDataError, ValueError, KeyError, TypeError) as exc:
                errors.append(str(exc))
        raise MarketDataError("；".join(errors))

    def _fetch_sina_universe(self) -> list[dict[str, Any]]:
        """按源端总数完整分页读取沪深京行情，逐页超时，不使用35秒子进程总限时。"""
        cache_key = "universe:sina_full:v1"
        cached = self._cache_result(cache_key, settings.UNIVERSE_CACHE_SECONDS, allow_stale=False)
        if cached is not None:
            return cached[0]["rows"]
        prefix = "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
        headers = {"Referer": "https://finance.sina.com.cn/"}
        response = self._request_response(
            prefix + "Market_Center.getHQNodeStockCount", {"node": "hs_a"}, "新浪股票池总数", headers,
        )
        total = int(response.json())
        if not settings.MINIMUM_EXPECTED_A_SHARE_UNIVERSE <= total <= 15000:
            raise MarketDataError(f"新浪源端股票总数异常：{total}，拒绝缓存。")
        size = 80
        rows = []
        for page in range(1, math.ceil(total / size) + 1):
            response = self._request_response(
                prefix + "Market_Center.getHQNodeData",
                {"page": page, "num": size, "sort": "symbol", "asc": 1, "node": "hs_a", "symbol": "", "_s_r_a": "page"},
                f"新浪股票池:page{page}", headers,
            )
            page_rows = response.json()
            if not isinstance(page_rows, list) or not page_rows:
                raise MarketDataError(f"新浪股票池分页中断：第{page}页为空，已读取{len(rows)}/{total}只。")
            for item in page_rows:
                if not isinstance(item, dict) or not item.get("code") or not item.get("name"):
                    raise MarketDataError(f"新浪股票池第{page}页存在缺少代码或名称的记录。")
                volume = _number(item.get("volume"))
                rows.append({
                    "code": item["code"], "name": item["name"], "price": _number(item.get("trade")),
                    "change_pct": _number(item.get("changepercent")), "change": _number(item.get("pricechange")),
                    "volume_lots": volume / 100 if volume is not None else None, "amount_yuan": _number(item.get("amount")),
                    "turnover_rate": _number(item.get("turnoverratio")), "high": _number(item.get("high")),
                    "low": _number(item.get("low")), "open": _number(item.get("open")),
                    "previous_close": _number(item.get("settlement")),
                })
            self.logger.info("新浪股票池分页 page=%s rows=%s total=%s", page, len(page_rows), total)
        unique = {str(row["code"]).zfill(6) for row in rows}
        if len(unique) != total:
            raise MarketDataError(f"新浪股票池分页不完整或重复：源端{total}只，唯一代码{len(unique)}只。")
        if not any(code.startswith(("43", "83", "87", "92")) for code in unique):
            raise MarketDataError("新浪完整列表缺少北交所，拒绝当作沪深京全市场使用。")
        self._write_cache(cache_key, {"rows": rows, "source_total": total})
        return rows

    @staticmethod
    def _map_universe_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
        frame = pd.DataFrame(rows)
        mapping = {
            "代码": "code", "名称": "name", "最新价": "price",
            "涨跌幅": "change_pct", "涨跌额": "change", "成交量": "volume_lots",
            "成交额": "amount_yuan", "换手率": "turnover_rate", "最高": "high",
            "最低": "low", "今开": "open", "昨收": "previous_close",
            "总市值": "market_cap_total", "流通市值": "market_cap_float", "所属行业": "industry",
            "trade": "price", "pricechange": "change", "changepercent": "change_pct",
            "settlement": "previous_close", "turnover": "turnover_rate",
        }
        frame = frame.rename(columns=mapping)
        if "code" not in frame or "name" not in frame:
            raise MarketDataError("股票池缺少代码或名称字段。")
        codes = frame["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.replace(r"\D", "", regex=True).str.zfill(6)
        frame["code"] = codes
        for column in ("price", "change_pct", "change", "volume_lots", "amount_yuan", "turnover_rate", "high", "low", "open", "previous_close"):
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        # 市值与行业用于中性化暴露；缺列时补空，缺失值统一为 None 以便写入 JSON 缓存。
        for column in ("market_cap_total", "market_cap_float"):
            values = pd.to_numeric(frame[column], errors="coerce") if column in frame else pd.Series(index=frame.index, dtype=float)
            frame[column] = values.astype(object).where(values.notna(), None)
        industry = frame["industry"] if "industry" in frame else pd.Series(index=frame.index, dtype=object)
        frame["industry"] = industry.astype(object).where(industry.notna(), None)
        return frame.drop_duplicates("code", keep="last").reset_index(drop=True)

    @staticmethod
    def _filter_universe_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        codes = frame["code"].astype(str)
        # 沪深主板与科创板的号段同 BOARD_PREFIXES，创业板另列出该板块开板以来的全部号段
        # （300、301 与 2026 年启用的 302）。新增号段时必须同步 settings.BOARD_PREFIXES 与这里，
        # 否则整批新号段股票会被静默判为无效代码、从股票池消失且不报错；
        # tests/test_services.py 的 test_board_prefixes_are_all_valid_codes 会拦下漏改。
        valid_code = (
            codes.str.fullmatch(r"(?:000|001|002|003|300|301|302|600|601|603|605|688|689)\d{3}")
            | codes.str.fullmatch(r"(?:43|83|87|92)\d{4}")
        ) & codes.ne("000000")
        special = frame["name"].astype(str).str.contains(r"\*?ST|退市?|退", case=False, regex=True, na=False)
        active = pd.Series(True, index=frame.index)
        if "price" in frame:
            active &= frame["price"].notna() & frame["price"].gt(0)
        if "volume_lots" in frame:
            active &= frame["volume_lots"].notna() & frame["volume_lots"].gt(0)
        keep = valid_code & ~special & active
        valid = frame.loc[keep].copy()
        return valid.reset_index(drop=True), int(len(frame) - len(valid))

    @staticmethod
    def _valid_universe_cache(payload: Any, scope: str) -> bool:
        if not isinstance(payload, dict) or not payload.get("meta", {}).get("complete"):
            return False
        rows = payload.get("rows") or []
        meta = payload.get("meta") or {}
        if len(rows) != meta.get("valid_symbols") or not rows:
            return False
        minimums = {
            "all": settings.MINIMUM_EXPECTED_A_SHARE_UNIVERSE,
            "main": 2000, "chinext": 1000, "star": 400, "bj": 150,
        }
        if len(rows) < minimums[scope]:
            return False
        codes = pd.Series([str(row.get("code", "")).zfill(6) for row in rows])
        if scope == "all":
            return (
                int(meta.get("source_unique_symbols", 0)) >= minimums[scope]
                and codes.str.startswith(("43", "83", "87", "92")).any()
            )
        return bool(codes.str.startswith(settings.BOARD_PREFIXES[scope]).all())

    def search_stocks(
        self, query: str, limit: int = 30, force_refresh: bool = False
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        query = str(query).strip()
        if not query:
            return pd.DataFrame(columns=["code", "name"]), {"status": "请输入代码或名称"}
        if force_refresh:
            universe, meta = self.get_universe(force_refresh=True)
        else:
            # 名称搜索优先复用完整目录；最新价格仍由后续分析单独请求。
            cached = self._cache_result("universe:all", settings.UNIVERSE_CACHE_SECONDS, allow_stale=True)
            if cached is None or not self._valid_universe_cache(cached[0], "all"):
                universe, meta = self.get_universe()
            else:
                payload, fetched, is_stale = cached
                age_seconds = (datetime.now(fetched.tzinfo) - fetched).total_seconds()
                if is_stale and not 0 <= age_seconds <= settings.STOCK_DIRECTORY_STALE_SECONDS:
                    universe, meta = self.get_universe()
                else:
                    universe = pd.DataFrame(payload["rows"])
                    meta = {
                        **payload["meta"],
                        "fetched_at": fetched.isoformat(timespec="seconds"),
                        "status": "股票名称目录缓存（行情非实时）" if is_stale else "完整股票池本地缓存",
                        "is_stale": is_stale,
                    }
        if universe.empty:
            return universe, meta
        code = universe["code"].astype(str).str.zfill(6)
        name = universe["name"].astype(str)
        matched = universe.loc[code.str.contains(query, regex=False) | name.str.contains(query, case=False, regex=False)]
        if matched.empty and not meta.get("is_stale"):
            try:
                full_universe = self._load_akshare_code_table()
                full_code = full_universe["code"].astype(str).str.zfill(6)
                full_name = full_universe["name"].astype(str)
                matched = full_universe.loc[
                    full_code.str.contains(query, regex=False)
                    | full_name.str.contains(query, case=False, regex=False)
                ]
                meta = {
                    **meta,
                    "status": "AkShare 全量证券代码表查询",
                    "source": "AkShare 股票代码表",
                }
            except (MarketDataError, subprocess.SubprocessError, ValueError, KeyError) as exc:
                meta = {**meta, "warning": f"未在行情接口首屏匹配；备用代码表不可用：{exc}"}
        return matched.head(limit).reset_index(drop=True), meta

    def _load_akshare_code_table(self) -> pd.DataFrame:
        cache_key = "universe:akshare_code_names"
        cached = self._cache_result(cache_key, settings.UNIVERSE_CACHE_SECONDS, allow_stale=False)
        if cached is not None:
            payload, _, _ = cached
            return pd.DataFrame(payload)
        script = (
            "import akshare as ak, json; "
            "frame=ak.stock_info_a_code_name(); cols={str(c).lower():c for c in frame.columns}; "
            "code_col=next(cols[k] for k in ('code','代码') if k in cols); "
            "name_col=next(cols[k] for k in ('name','名称') if k in cols); "
            "rows=[{'code':str(r[code_col]).zfill(6),'name':str(r[name_col])} for _,r in frame.iterrows()]; "
            "print(json.dumps(rows,ensure_ascii=False))"
        )
        try:
            completed = subprocess.run(
                [sys.executable, "-X", "utf8", "-c", script],
                capture_output=True,
                check=True,
                encoding="utf-8",
                text=True,
                timeout=20,
            )
            output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
            rows = json.loads(output_lines[-1]) if output_lines else []
            if not rows:
                raise MarketDataError("AkShare 未返回证券代码。")
            self._write_cache(cache_key, rows)
            return pd.DataFrame(rows)
        except subprocess.TimeoutExpired as exc:
            raise MarketDataError("备用证券代码表请求超时（20 秒）。") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "备用接口失败").strip()[-300:]
            raise MarketDataError(f"备用证券代码表获取失败：{detail}") from exc


def _scaled(value: Any, scale: int) -> float | None:
    number = _number(value)
    return number / scale if number is not None and number > 0 else None

