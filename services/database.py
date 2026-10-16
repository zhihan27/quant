"""PostgreSQL 连接、版本化建表及自选、持仓、监控数据的事务存储。"""

from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Iterator
from zoneinfo import ZoneInfo

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from config import settings

SHANGHAI = ZoneInfo("Asia/Shanghai")


class DatabaseError(RuntimeError):
    """数据库连接、写入或并发编辑失败。"""


def connection_settings() -> dict[str, Any]:
    """读取本地私有配置，环境变量可覆盖连接信息，密码不进入日志。"""
    path = settings.PROJECT_ROOT / "config" / "database.local.json"
    local = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}
    defaults = {"host": "127.0.0.1", "port": 5432, "dbname": "quantify", "user": "postgres", "password": ""}
    configuration = {}
    for name, value in defaults.items():
        configuration[name] = os.environ.get(f"QUANT_PG_{name.upper()}", local.get(name, value))
    configuration["port"] = int(configuration["port"])
    return {**configuration, "connect_timeout": 5, "application_name": "quantify", "options": "-c timezone=Asia/Shanghai -c statement_timeout=10000 -c lock_timeout=5000"}


def stock_code(value: Any) -> str:
    """严格校验股票代码，防止错误代码进入持仓及行情订阅。"""
    value = str(value).strip()
    if not re.fullmatch(r"[0-9]{6}", value):
        raise ValueError("股票代码必须为六位数字。")
    return value


def _money(value: Any, field: str, positive: bool = False) -> Decimal:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{field}必须是有效数字。") from exc
    if not number.is_finite() or (number <= 0 if positive else number < 0) or number >= Decimal("1000000000000"):
        raise ValueError(f"{field}超出有效范围。")
    return number


def _integer(value: Any, field: str) -> int:
    number = _money(value, field)
    if number != number.to_integral_value() or number > 2_000_000_000:
        raise ValueError(f"{field}必须为有效的整数股数。")
    return int(number)


def validate_position(values: dict[str, Any]) -> dict[str, Any]:
    """校验持仓价格、整数股数、买入日和手工确认可卖数量。"""
    shares = _integer(values.get("shares"), "持仓股数")
    available = _integer(values.get("available_shares", shares), "可卖股数")
    if shares == 0 or available > shares:
        raise ValueError("持仓股数须大于零，可卖股数不能超过持仓股数。")
    try:
        bought = date.fromisoformat(str(values.get("buy_date"))[:10])
    except ValueError as exc:
        raise ValueError("买入日期无效。") from exc
    if bought > datetime.now(SHANGHAI).date():
        raise ValueError("买入日期不能晚于今天。")
    price = _money(values.get("buy_price"), "买入价", True).quantize(Decimal("0.000001"))
    if price <= 0:
        raise ValueError("买入价精度最多为六位小数，不能舍入为零。")
    return {
        "code": stock_code(values.get("code")), "name": str(values.get("name", "")).strip()[:80],
        "buy_date": bought, "buy_price": price,
        "shares": shares, "available_shares": available,
        "fees": _money(values.get("fees", 0), "买入费用").quantize(Decimal("0.0001")),
        "note": str(values.get("note", ""))[:2000],
    }


def _json(value: Any) -> Jsonb:
    # 数据库 JSON 采用严格的有限数字；日期及金额按文本保存完整修改快照。
    return Jsonb(json.loads(json.dumps(value, ensure_ascii=False, default=str, allow_nan=False)))


class QuantRepository:
    """管理 public 模式中的自选、分批持仓、行情和监控配置。"""

    def __init__(self, configuration: dict[str, Any] | None = None, initialize: bool = True) -> None:
        self.configuration = configuration or connection_settings()
        if initialize:
            self.initialize()

    @contextmanager
    def connect(self) -> Iterator[psycopg.Connection]:
        """每次业务操作独立事务，异常回滚，避免共享连接跨线程使用。"""
        try:
            with psycopg.connect(**self.configuration, row_factory=dict_row) as connection:
                yield connection
        except psycopg.Error as exc:
            reason = {
                "28P01": "数据库认证失败，请检查本地连接配置。",
                "3D000": "数据库 quantify 不存在，请先创建数据库。",
                "23514": "写入数据违反数据库约束。",
                "23505": "记录重复，请刷新后重试。",
                "42501": "数据库账号缺少 public 模式的访问或建表权限。",
            }.get(exc.sqlstate, "PostgreSQL 操作失败，请检查服务、连接配置或数据库状态。")
            raise DatabaseError(reason) from exc

    def initialize(self) -> None:
        """在事务和咨询锁保护下执行首次建表，不覆盖已有业务数据。"""
        migrations = [
            "001_portfolio_monitor.sql", "002_monitor_indexes.sql", "003_auth.sql", "004_admin_roles.sql",
            "005_model_governance.sql",
        ]
        with self.connect() as connection:
            exists = connection.execute("SELECT to_regclass('public.quant_schema_versions') AS name").fetchone()["name"]
            version = connection.execute("SELECT COALESCE(max(version),0) AS version FROM public.quant_schema_versions").fetchone()["version"] if exists else 0
            if version >= len(migrations):
                return
            connection.execute("SELECT pg_advisory_xact_lock(706821024)")
            # 获锁后重新确认，避免两个新进程同时启动时重复执行DDL并锁住正在使用的表。
            exists = connection.execute("SELECT to_regclass('public.quant_schema_versions') AS name").fetchone()["name"]
            version = connection.execute("SELECT COALESCE(max(version),0) AS version FROM public.quant_schema_versions").fetchone()["version"] if exists else 0
            for name in migrations[version:]:
                connection.execute((settings.PROJECT_ROOT / "sql" / name).read_text(encoding="utf-8"))

    def health(self) -> dict[str, Any]:
        """返回不包含凭据的数据库连接状态。"""
        with self.connect() as connection:
            return connection.execute("SELECT current_database() AS database, 'public' AS schema, version() AS version").fetchone()

    def watchlist(self) -> list[dict[str, Any]]:
        """读取持久化自选股票。"""
        with self.connect() as connection:
            return connection.execute("SELECT * FROM public.quant_watchlist ORDER BY created_at, code").fetchall()

    def save_watch(self, code: str, name: str = "", note: str = "", monitor_enabled: bool = True) -> None:
        """新增或编辑自选，同一代码重复添加时保留原备注和名称。"""
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO public.quant_watchlist(code,name,note,monitor_enabled) VALUES (%s,%s,%s,%s) "
                "ON CONFLICT(code) DO UPDATE SET name=COALESCE(NULLIF(EXCLUDED.name,''),quant_watchlist.name), "
                "note=COALESCE(NULLIF(EXCLUDED.note,''),quant_watchlist.note), "
                "monitor_enabled=EXCLUDED.monitor_enabled,updated_at=now()",
                (stock_code(code), str(name)[:80], str(note)[:2000], bool(monitor_enabled)),
            )

    def edit_watchlist(self, rows: list[dict[str, Any]]) -> None:
        """一次事务保存自选编辑，删除自选不会删除对应持仓。"""
        with self.connect() as connection:
            for row in rows:
                code = stock_code(row["code"])
                if row.get("remove"):
                    connection.execute("DELETE FROM public.quant_watchlist WHERE code=%s", (code,))
                else:
                    connection.execute(
                        "UPDATE public.quant_watchlist SET name=%s,note=%s,monitor_enabled=%s,updated_at=now() WHERE code=%s",
                        (str(row.get("name", ""))[:80], str(row.get("note", ""))[:2000], bool(row.get("monitor_enabled")), code),
                    )

    def positions(self) -> list[dict[str, Any]]:
        """返回全部持仓批次及用于并发编辑检测的版本号。"""
        with self.connect() as connection:
            return connection.execute("SELECT * FROM public.quant_positions ORDER BY code,buy_date,id").fetchall()

    def add_position(self, values: dict[str, Any]) -> int:
        """校验并保存一笔手工持仓，同时在同一事务内保存修改记录。"""
        values = validate_position(values)
        with self.connect() as connection:
            row = connection.execute(
                "INSERT INTO public.quant_positions(code,name,buy_date,buy_price,shares,available_shares,fees,note) "
                "VALUES (%(code)s,%(name)s,%(buy_date)s,%(buy_price)s,%(shares)s,%(available_shares)s,%(fees)s,%(note)s) RETURNING *",
                values,
            ).fetchone()
            connection.execute(
                "INSERT INTO public.quant_position_audit(position_id,operation,after_value) VALUES (%s,'新增',%s)",
                (row["id"], _json(row)),
            )
            return row["id"]

    def edit_positions(self, rows: list[dict[str, Any]]) -> None:
        """按版本校验批量修改或删除持仓；任一记录冲突则全部回滚。"""
        with self.connect() as connection:
            for item in rows:
                row = connection.execute("SELECT * FROM public.quant_positions WHERE id=%s FOR UPDATE", (int(item["id"]),)).fetchone()
                if row is None or row["version"] != int(item["version"]):
                    raise DatabaseError("持仓已被其他页面修改，请刷新后重新编辑。")
                if item.get("remove"):
                    connection.execute("DELETE FROM public.quant_positions WHERE id=%s", (row["id"],))
                    operation, after = "删除", None
                else:
                    values = validate_position(item)
                    unchanged = all(row[key] == value for key, value in values.items())
                    if unchanged:
                        continue
                    after = connection.execute(
                        "UPDATE public.quant_positions SET code=%(code)s,name=%(name)s,buy_date=%(buy_date)s, "
                        "buy_price=%(buy_price)s,shares=%(shares)s,available_shares=%(available_shares)s,fees=%(fees)s, "
                        "note=%(note)s,version=version+1,updated_at=now() WHERE id=%(id)s RETURNING *",
                        {**values, "id": row["id"]},
                    ).fetchone()
                    operation = "编辑"
                connection.execute(
                    "INSERT INTO public.quant_position_audit(position_id,operation,before_value,after_value) VALUES (%s,%s,%s,%s)",
                    (row["id"], operation, _json(row), _json(after) if after else None),
                )

    def position_audit(self, limit: int = 100) -> list[dict[str, Any]]:
        """读取最近的持仓修改记录。"""
        with self.connect() as connection:
            return connection.execute("SELECT * FROM public.quant_position_audit ORDER BY id DESC LIMIT %s", (min(max(int(limit), 1), 1000),)).fetchall()

    def monitor_config(self) -> dict[str, Any]:
        """读取监控开关、参数和最近心跳。"""
        with self.connect() as connection:
            return connection.execute("SELECT * FROM public.quant_monitor_config WHERE id=1").fetchone()

    def save_monitor_config(self, enabled: bool, configuration: dict[str, Any]) -> None:
        """持久化监控参数，应用重启后沿用上次选择。"""
        with self.connect() as connection:
            connection.execute(
                "UPDATE public.quant_monitor_config SET enabled=%s,config=%s,updated_at=now() WHERE id=1",
                (enabled, _json(configuration)),
            )

    def monitor_heartbeat(self, seconds: float, error: str | None = None) -> None:
        """记录实际采集耗时和失败信息，不把设定频率当作实际频率。"""
        with self.connect() as connection:
            connection.execute(
                "UPDATE public.quant_monitor_config SET last_cycle_at=now(),cycle_seconds=%s,last_error=%s WHERE id=1",
                (seconds, error[:2000] if error else None),
            )

    def monitored_codes(self) -> list[str]:
        """合并开启监控的自选和持仓代码，不重复订阅。"""
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT code FROM public.quant_watchlist WHERE monitor_enabled UNION SELECT code FROM public.quant_positions ORDER BY code"
            ).fetchall()
        return [row["code"] for row in rows]

    def save_quotes(self, quotes: list[dict[str, Any]]) -> None:
        """保存原始行情源时间和采集时间；同一采集快照重复提交不重复写入。"""
        with self.connect() as connection:
            for quote in quotes:
                if quote.get("price") is None:
                    continue
                source_time = quote.get("quote_time")
                source_time = datetime.fromisoformat(source_time) if source_time and len(source_time) > 10 else None
                connection.execute(
                    "INSERT INTO public.quant_quote_snapshots(code,observed_at,source_time,price,volume_lots,amount_yuan,source,data_status,payload) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                    (stock_code(quote["code"]), quote["fetched_at"], source_time, quote["price"], quote.get("volume_lots"),
                     quote.get("amount_yuan"), quote.get("source", "未知"), quote.get("data_status", "未知"), _json(quote)),
                )

    def latest_quotes(self, codes: list[str] | None = None) -> dict[str, dict[str, Any]]:
        """返回每只股票最后采集快照，保留其来源及过期状态供页面判断。"""
        if codes is None:
            codes = self.monitored_codes()
        if not codes:
            return {}
        with self.connect() as connection:
            # 每个代码通过主键索引只读最后一行，避免高频快照积累后全表扫描。
            rows = connection.execute(
                "SELECT latest.* FROM unnest(%s::text[]) AS requested(code) CROSS JOIN LATERAL "
                "(SELECT code,payload FROM public.quant_quote_snapshots WHERE code=requested.code "
                "ORDER BY observed_at DESC LIMIT 1) latest",
                (list(dict.fromkeys(codes)),),
            ).fetchall()
        return {row["code"]: row["payload"] for row in rows}

    def save_minutes(self, code: str, frame: Any, source: str) -> None:
        """幂等保存已完成的不复权分钟线。"""
        import pandas as pd

        rows = [
            (stock_code(code), row.date.to_pydatetime(), float(row.open), float(row.high), float(row.low), float(row.close),
             float(row.volume), float(row.amount) if pd.notna(row.amount) else None, source)
            for row in frame.itertuples()
        ]
        if not rows:
            return
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.executemany(
                    "INSERT INTO public.quant_minute_bars(code,bar_time,open,high,low,close,volume_lots,amount_yuan,source) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(code,bar_time) DO UPDATE SET "
                    "open=EXCLUDED.open,high=EXCLUDED.high,low=EXCLUDED.low,close=EXCLUDED.close,volume_lots=EXCLUDED.volume_lots, "
                    "amount_yuan=EXCLUDED.amount_yuan,source=EXCLUDED.source,updated_at=now()",
                    rows,
                )

    def minute_history(self, code: str, limit: int = 1800) -> list[dict[str, Any]]:
        """读取最近分钟线，返回时间升序数据。"""
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM (SELECT bar_time AS date,open,high,low,close,volume_lots AS volume,amount_yuan AS amount "
                "FROM public.quant_minute_bars WHERE code=%s ORDER BY bar_time DESC LIMIT %s) bars ORDER BY date",
                (stock_code(code), min(max(int(limit), 1), 10000)),
            ).fetchall()

    def save_signal(self, signal: dict[str, Any]) -> None:
        """每只股票每分钟每策略保留一条记录，重复轮询不会刷出重复提醒。"""
        if not signal.get("bar_time"):
            return
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO public.quant_intraday_signals(code,peer_code,strategy,bar_time,action,active,reason,reference_price,details) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(code,peer_code,strategy,bar_time) DO UPDATE SET "
                "action=EXCLUDED.action,active=EXCLUDED.active,reason=EXCLUDED.reason,reference_price=EXCLUDED.reference_price, "
                "details=EXCLUDED.details,updated_at=now()",
                (signal["code"], signal.get("peer_code", ""), signal["strategy"], signal["bar_time"], signal["action"],
                 signal["active"], signal["reason"], signal.get("reference_price"), _json(signal)),
            )

    def signals(self, limit: int = 100) -> list[dict[str, Any]]:
        """读取最新信号及失效原因。"""
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM public.quant_intraday_signals ORDER BY bar_time DESC,id DESC LIMIT %s",
                (min(max(int(limit), 1), 1000),),
            ).fetchall()

    def save_backtest(self, run_id: str, parameters: dict[str, Any], result: dict[str, Any]) -> None:
        """将回测结果保存到 PostgreSQL，便于统一追溯。"""
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO public.quant_saved_backtests(run_id,parameters,model_version,result) VALUES (%s,%s,%s,%s)",
                (run_id, _json(parameters), settings.MODEL_VERSION, _json(result)),
            )

    def prune_monitor_history(self) -> None:
        """定期清理快照30天、分钟线90天及信号180天以前的数据，持仓和审计永久保留。"""
        with self.connect() as connection:
            connection.execute("DELETE FROM public.quant_quote_snapshots WHERE observed_at < now() - interval '30 days'")
            connection.execute("DELETE FROM public.quant_minute_bars WHERE bar_time < now() - interval '90 days'")
            connection.execute("DELETE FROM public.quant_intraday_signals WHERE bar_time < now() - interval '180 days'")
