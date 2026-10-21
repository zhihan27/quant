"""采集与校验可追溯的点时股票池快照。"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from services.market_data import MarketDataService
from services.panel_store import PanelStore

LOGGER = logging.getLogger("quant.short_term.point_in_time")
SHANGHAI = ZoneInfo("Asia/Shanghai")


def collect_point_in_time_snapshot(
    panel_store: PanelStore | None = None,
    market_data: MarketDataService | None = None,
    session_date: Any | None = None,
) -> dict[str, Any]:
    """仅保存完整、非过期且含真实获取时间的当期策略股票池。"""
    store = panel_store or PanelStore()
    source = market_data or MarketDataService()
    try:
        universe, metadata = source.get_universe(scope="all", force_refresh=True)
    except Exception as exc:
        LOGGER.warning("点时股票池刷新失败 detail=%s", exc)
        return {
            "status": "unavailable",
            "membership_saved": False,
            "security_status_saved": False,
            "blockers": [f"股票池来源刷新失败：{type(exc).__name__}"],
        }

    try:
        fetched_at = pd.Timestamp(metadata.get("fetched_at"))
    except (TypeError, ValueError):
        fetched_at = pd.NaT
    if fetched_at.tzinfo is None:
        return {
            "status": "blocked_invalid_source_time",
            "membership_saved": False,
            "security_status_saved": False,
            "blockers": ["股票池来源未提供带时区的实际 fetched_at。"],
        }
    if not metadata.get("complete") or metadata.get("is_stale"):
        return {
            "status": "blocked_incomplete_source",
            "membership_saved": False,
            "security_status_saved": False,
            "source": metadata.get("source"),
            "blockers": ["股票池来源不完整或已过期，未写入点时快照。"],
        }
    if universe.empty or "code" not in universe.columns:
        return {
            "status": "blocked_empty_source",
            "membership_saved": False,
            "security_status_saved": False,
            "source": metadata.get("source"),
            "blockers": ["完整股票池没有可用证券代码。"],
        }

    snapshot_time = fetched_at.tz_convert(SHANGHAI)
    try:
        snapshot_date = pd.Timestamp(session_date) if session_date is not None else snapshot_time
        if snapshot_date.tzinfo is not None:
            snapshot_date = snapshot_date.tz_localize(None)
        snapshot_date = snapshot_date.normalize()
    except (TypeError, ValueError):
        return {
            "status": "blocked_invalid_session_date",
            "membership_saved": False,
            "security_status_saved": False,
            "blockers": ["股票池快照关联的交易日无效。"],
        }

    membership = pd.DataFrame(
        {
            "code": universe["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6),
            "date": snapshot_date,
            "available_at": snapshot_time,
            "is_member": True,
            "source": str(metadata.get("source") or "未提供"),
        }
    ).drop_duplicates(["code", "date", "available_at"], keep="last")
    if not membership["code"].str.fullmatch(r"\d{6}").all():
        return {
            "status": "blocked_invalid_source_codes",
            "membership_saved": False,
            "security_status_saved": False,
            "source": metadata.get("source"),
            "blockers": ["股票池来源包含无效证券代码，未写入点时快照。"],
        }

    path = store.save_point_in_time_snapshot("membership", membership)
    return {
        "status": "membership_snapshot_saved",
        "membership_saved": True,
        "security_status_saved": False,
        "membership_rows": int(len(membership)),
        "available_at": snapshot_time.isoformat(),
        "session_date": snapshot_date.date().isoformat(),
        "source": metadata.get("source"),
        "path": str(path),
        "blockers": [
            "当前行情接口只返回符合其活跃股票池筛选的证券，未覆盖所有非成员证券。",
            "当前接口未提供可按执行时点验证的上市、ST、停牌及涨跌停历史状态。",
        ],
    }


def filter_by_point_in_time_membership(
    panel: pd.DataFrame,
    membership_history: pd.DataFrame | None,
    signal_time_column: str = "signal_time",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """按每条信号当时已可获得的股票池记录筛选面板，缺失记录不推断成员身份。"""
    required = {"code", "date", "available_at", "is_member"}
    if panel.empty:
        return panel.copy(), {"status": "empty_panel", "rows": 0, "covered_rows": 0, "coverage_ratio": 0.0}
    if not isinstance(membership_history, pd.DataFrame) or membership_history.empty:
        return panel.iloc[0:0].copy(), {
            "status": "missing_membership_history",
            "rows": int(len(panel)),
            "covered_rows": 0,
            "member_rows": 0,
            "missing_rows": int(len(panel)),
            "coverage_ratio": 0.0,
        }
    if not required.issubset(membership_history.columns):
        missing = sorted(required.difference(membership_history.columns))
        return panel.iloc[0:0].copy(), {
            "status": "invalid_membership_history",
            "rows": int(len(panel)),
            "covered_rows": 0,
            "coverage_ratio": 0.0,
            "blockers": [f"点时股票池缺字段：{'、'.join(missing)}"],
        }

    frame = panel.copy()
    history = membership_history.copy()
    frame["code"] = frame["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    history["code"] = history["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    history["date"] = pd.to_datetime(history["date"], errors="coerce").dt.normalize()
    if signal_time_column in frame:
        signal_times = pd.to_datetime(frame[signal_time_column], errors="coerce")
    else:
        signal_times = frame["date"] + pd.Timedelta(hours=15, minutes=5)
    if getattr(signal_times.dt, "tz", None) is None:
        signal_times = signal_times.dt.tz_localize(SHANGHAI, ambiguous="NaT", nonexistent="NaT")
    else:
        signal_times = signal_times.dt.tz_convert(SHANGHAI)
    available = pd.to_datetime(history["available_at"], errors="coerce", utc=True).dt.tz_convert(SHANGHAI)
    history["available_at"] = available
    try:
        history["is_member"] = history["is_member"].map(_parse_boolean)
    except ValueError as exc:
        return panel.iloc[0:0].copy(), {
            "status": "invalid_membership_values",
            "rows": int(len(panel)),
            "covered_rows": 0,
            "coverage_ratio": 0.0,
            "blockers": [str(exc)],
        }
    if frame["date"].isna().any() or signal_times.isna().any() or available.isna().any():
        return panel.iloc[0:0].copy(), {
            "status": "invalid_pit_timestamps",
            "rows": int(len(panel)),
            "covered_rows": 0,
            "coverage_ratio": 0.0,
            "blockers": ["面板日期、信号时间或 available_at 包含无效值。"],
        }

    index: dict[tuple[str, pd.Timestamp], tuple[list[pd.Timestamp], list[bool]]] = {}
    for (code, session), group in history.groupby(["code", "date"], sort=False):
        ordered = group.sort_values("available_at")
        index[(str(code), pd.Timestamp(session))] = (
            ordered["available_at"].tolist(),
            ordered["is_member"].tolist(),
        )

    membership_values: list[bool | None] = []
    for row, cutoff in zip(frame[["code", "date"]].itertuples(index=False), signal_times):
        records = index.get((row.code, pd.Timestamp(row.date)))
        if records is None:
            membership_values.append(None)
            continue
        position = pd.Index(records[0]).searchsorted(cutoff, side="right") - 1
        membership_values.append(records[1][position] if position >= 0 else None)
    frame["_pit_is_member"] = membership_values
    covered = frame["_pit_is_member"].notna()
    selected = frame.loc[frame["_pit_is_member"].eq(True)].drop(columns="_pit_is_member").copy()
    covered_rows = int(covered.sum())
    total_rows = int(len(frame))
    return selected, {
        "status": "filtered" if covered_rows == total_rows else "partial_coverage",
        "rows": total_rows,
        "covered_rows": covered_rows,
        "member_rows": int(selected.shape[0]),
        "nonmember_rows": int((frame["_pit_is_member"] == False).sum()),
        "missing_rows": int((~covered).sum()),
        "coverage_ratio": float(covered_rows / total_rows) if total_rows else 0.0,
    }


def _parse_boolean(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    raise ValueError("点时股票池包含非布尔 is_member 值。")
