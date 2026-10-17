"""分阶段扫描的采集快照：清单可续跑，基准/暴露/新增因子冻结在快照里。

逐股历史正文不复制：采集时 ``get_history`` 已把清洗后的前复权日线写入 PanelStore，
快照只记录代码、报价与元数据。训练时按代码回读历史、重新过滤未完成日线并截断到
快照的截面日期，因此同一份快照可以反复训练，模型口径始终指向同一个交易日。

本模块不依赖 Streamlit，也不连接数据库；页面与定时任务都可以直接使用。
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from config import settings
from services.market_data import completed_daily_history
from services.panel_store import PanelStore

LOGGER = logging.getLogger("quant.acquisition")
SHANGHAI = ZoneInfo("Asia/Shanghai")
SNAPSHOT_ID_PATTERN = re.compile(r"^[0-9A-Za-z_-]{8,64}$")
COMPLETED_STATUSES = ("collecting", "failed")
STORAGE_COLUMNS = ("stored_at", "history_source", "code")
# 采集选项里决定同一份截面数据能否复用的字段；训练目标与窗口不参与指纹，
# 因为同一份快照允许用不同目标/窗口重复训练。
FINGERPRINT_FIELDS = (
    "pool_mode", "custom_codes", "scan_limit", "enable_neutralization", "use_extra_factors",
    "include_backtest",
)


def _now_text() -> str:
    return datetime.now(SHANGHAI).isoformat(timespec="seconds")


def _json_safe(value: Any) -> Any:
    """把 numpy 标量、NaN/inf 与嵌套容器转成可写进 JSON 的普通类型。"""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if value is pd.NaT or value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression=settings.PARQUET_COMPRESSION)
    os.replace(temporary, path)


def _snapshot_root() -> Path:
    # 必须在调用时读取，测试才能通过 patch settings.ACQUISITION_DIR 重定向。
    return Path(settings.ACQUISITION_DIR)


def _snapshot_path(snapshot_id: str) -> Path:
    if not SNAPSHOT_ID_PATTERN.match(str(snapshot_id or "")):
        raise ValueError("采集快照编号不合法。")
    return _snapshot_root() / snapshot_id


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        LOGGER.warning("读取快照元数据失败 path=%s", path)
        return None
    return payload if isinstance(payload, dict) else None


def _read_items(directory: Path) -> dict[str, dict[str, Any]]:
    """读取追加式清单；跳过损坏/半行，同一代码后写覆盖先写。"""
    path = directory / "items.jsonl"
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        LOGGER.warning("读取快照清单失败 path=%s detail=%s", path, exc)
        return {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict) or not record.get("code"):
            continue
        records[str(record["code"])] = record
    return records


def _append_item(directory: Path, record: dict[str, Any]) -> None:
    path = directory / "items.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in record.items() if key not in ("history", "raw_history")}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(payload), ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fingerprint(options: Any) -> dict[str, Any]:
    return {name: getattr(options, name, None) for name in FINGERPRINT_FIELDS}


def _same_snapshot(meta: dict[str, Any], options: Any, reference_date: str) -> bool:
    stored = meta.get("options") or {}
    return (
        meta.get("model_version") == settings.MODEL_VERSION
        and meta.get("reference_date") == reference_date
        and all(stored.get(name) == getattr(options, name, None) for name in FINGERPRINT_FIELDS)
    )


def _stored_history_status(
    store: PanelStore, code: str, reference_date: str
) -> tuple[str | None, int]:
    """轻量检查已落盘历史：返回 (最新日期, 行数)，不读取完整 frame。"""
    path = store.history_path(code, "qfq")
    if not path.exists():
        return None, 0
    try:
        frame = pd.read_parquet(path, columns=["date"])
    except (OSError, ValueError):
        return None, 0
    if frame.empty:
        return None, 0
    if _daily_history_error(frame):
        return None, 0
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
    if dates.empty:
        return None, 0
    latest = dates.max().strftime("%Y-%m-%d")
    return latest, int((dates.dt.strftime("%Y-%m-%d") <= reference_date).sum())


def _completed_history(
    frame: pd.DataFrame, reference_date: str
) -> pd.DataFrame:
    """训练口径的历史：重新剔除未完成日线、截断到快照截面日、去掉存储附加列。"""
    cleaned = completed_daily_history(frame)
    if cleaned.empty or "date" not in cleaned:
        return cleaned
    dates = pd.to_datetime(cleaned["date"], errors="coerce")
    cleaned = cleaned.loc[dates.notna() & (dates.dt.strftime("%Y-%m-%d") <= reference_date)].copy()
    cleaned = cleaned.drop_duplicates(subset=["date"], keep="last")
    for column in STORAGE_COLUMNS:
        if column in cleaned:
            cleaned = cleaned.drop(columns=[column])
    return cleaned.sort_values("date").reset_index(drop=True)


def _daily_history_error(frame: pd.DataFrame) -> str | None:
    """拒绝周线及旧版本曾被周线覆盖的文件，避免把周频标签当成日频。"""
    if frame.attrs.get("history_period", "daily") != "daily":
        return "历史文件不是日线，请重新采集该股"
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna().drop_duplicates().sort_values()
    if len(dates) > 1 and dates.diff().dt.days.median() >= 6:
        return "历史日期间隔不符合日线口径，可能已被旧版周线覆盖，请重新采集该股"
    return None


def _validate_history(
    record: dict[str, Any], reference_date: str, store: PanelStore
) -> tuple[pd.DataFrame | None, str | None]:
    """校验一只股票的训练用历史；PanelStore 写入是 best-effort，这里补齐或判失败。"""
    if record.get("error"):
        return None, str(record["error"])
    code = str(record["code"])
    frame = store.read_history(code, "qfq")
    if (frame is None or frame.empty) and isinstance(record.get("history"), pd.DataFrame):
        # 采集时 Parquet 写入失败但内存里仍有清洗后的日线：补写一次再校验。
        try:
            store.write_history(code, record["history"], "qfq", record.get("history_meta") or {})
        except (OSError, ValueError, ImportError) as exc:
            return None, f"Parquet 历史写入失败：{type(exc).__name__}"
        frame = store.read_history(code, "qfq")
    if frame is None or frame.empty:
        return None, "Parquet 历史文件缺失，请重新采集该股"
    error = _daily_history_error(frame)
    if error:
        return None, error
    cleaned = _completed_history(frame, reference_date)
    if cleaned.empty:
        return None, "历史行情在快照截面前为空，请重新采集该股"
    latest = pd.to_datetime(cleaned["date"]).max().strftime("%Y-%m-%d")
    if latest < reference_date:
        return None, f"历史行情仅到 {latest}，未覆盖快照截面日 {reference_date}"
    if len(cleaned) < settings.MINIMUM_SCAN_HISTORY_BARS:
        return None, f"截断后历史样本不足 {settings.MINIMUM_SCAN_HISTORY_BARS} 根日线"
    return cleaned, None


@dataclass
class AcquisitionRun:
    snapshot_id: str
    directory: Path
    meta: dict[str, Any]
    resumed: bool = False
    prior: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class AcquisitionSnapshot:
    snapshot_id: str
    directory: Path
    meta: dict[str, Any]

    @property
    def status(self) -> str:
        return str(self.meta.get("status") or "unknown")

    @property
    def reference_date(self) -> str:
        return str(self.meta.get("reference_date") or "")


@dataclass
class LoadedSnapshot:
    snapshot_id: str
    meta: dict[str, Any]
    records: list[dict[str, Any]]
    histories: dict[str, pd.DataFrame]
    benchmark: pd.DataFrame
    exposures: pd.DataFrame | None
    extra_factors: dict[str, Any]
    failures: list[dict[str, str]]

    @property
    def reference_date(self) -> str:
        return str(self.meta.get("reference_date") or "")


def _iter_snapshots() -> list[tuple[Path, dict[str, Any]]]:
    root = _snapshot_root()
    if not root.exists():
        return []
    result = []
    for directory in root.iterdir():
        if not directory.is_dir() or not SNAPSHOT_ID_PATTERN.match(directory.name):
            continue
        meta = _read_json(directory / "meta.json")
        if meta:
            result.append((directory, meta))
    return result


def _referenced_snapshot_ids() -> set[str]:
    """被归档模型引用的快照不裁剪，否则模型目录会指向已删数据。"""
    referenced: set[str] = set()
    archive_root = Path(settings.MODEL_DIR) / settings.MODEL_ARCHIVE_DIRNAME
    if not archive_root.exists():
        return referenced
    for path in archive_root.glob("*/registry.json"):
        payload = _read_json(path) or {}
        snapshot_id = payload.get("source_snapshot_id")
        if snapshot_id:
            referenced.add(str(snapshot_id))
    return referenced


def create_or_resume(
    options: Any,
    source_meta: dict[str, Any],
    reference_date: str,
    candidates: list[dict[str, Any]],
    notes: list[dict[str, str]] | None = None,
) -> AcquisitionRun:
    """查找同日同口径的未完成快照继续采集；没有则新建并把旧半成品标记为废弃。"""
    root = _snapshot_root()
    root.mkdir(parents=True, exist_ok=True)
    existing = sorted(_iter_snapshots(), key=lambda item: str(item[1].get("created_at") or ""), reverse=True)
    for directory, meta in existing:
        if meta.get("status") in COMPLETED_STATUSES and _same_snapshot(meta, options, reference_date):
            meta = {
                **meta,
                "status": "collecting",
                "error": None,
                "resumed_at": _now_text(),
                "candidate_codes": [str(item["code"]) for item in candidates],
                "source_meta": source_meta or {},
                "counts": {**(meta.get("counts") or {}), "total": len(candidates)},
                "notes": [*(meta.get("notes") or []), *(notes or [])],
            }
            _atomic_json(directory / "meta.json", meta)
            return AcquisitionRun(
                snapshot_id=directory.name, directory=directory, meta=meta, resumed=True,
                prior=_read_items(directory),
            )

    # 同一股票池的旧半成品无法与新参考日/选项拼接，直接标废弃，避免每天残留一个目录。
    for directory, meta in existing:
        stored = meta.get("options") or {}
        if meta.get("status") != "collecting":
            continue
        if (stored.get("pool_mode"), stored.get("custom_codes")) != (
            getattr(options, "pool_mode", None), getattr(options, "custom_codes", None)
        ):
            continue
        if not _same_snapshot(meta, options, reference_date):
            _atomic_json(directory / "meta.json", {**meta, "status": "abandoned", "finished_at": _now_text()})

    snapshot_id = datetime.now(SHANGHAI).strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    directory = root / snapshot_id
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "items.jsonl").touch()
    meta = {
        "snapshot_id": snapshot_id,
        "model_version": settings.MODEL_VERSION,
        "status": "collecting",
        "error": None,
        "created_at": _now_text(),
        "finished_at": None,
        "reference_date": str(reference_date),
        "benchmark_meta": None,
        "source_meta": source_meta or {},
        "options": _fingerprint(options),
        "candidate_codes": [str(item["code"]) for item in candidates],
        "counts": {"total": len(candidates), "success": 0, "failed": 0, "skipped": 0},
        "notes": notes or [],
        "exposure_meta": {},
        "extra_meta": {},
    }
    _atomic_json(directory / "meta.json", meta)
    return AcquisitionRun(snapshot_id=snapshot_id, directory=directory, meta=meta)


def checkpoint(run: AcquisitionRun, record: dict[str, Any]) -> None:
    """保存逐股断点；回测原始日线独立冻结，行情正文不写入清单。"""
    raw = record.get("raw_history")
    if isinstance(raw, pd.DataFrame) and not raw.empty:
        try:
            _atomic_frame(raw, run.directory / "raw" / f"code={record['code']}.parquet")
        except (OSError, ValueError, ImportError) as exc:
            record["backtest_error"] = f"冻结回测日线失败：{type(exc).__name__}"
            LOGGER.exception("冻结回测日线失败 code=%s", record["code"])
    _append_item(run.directory, record)


def snapshot_history(
    record: dict[str, Any], reference_date: str, store: PanelStore | None = None
) -> tuple[pd.DataFrame | None, str | None]:
    """给采集阶段的扩展因子准备历史：优先用本次内存里的 frame，续跑记录回读面板。"""
    if record.get("error"):
        return None, str(record["error"])
    frame = record.get("history")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return _validate_history(record, reference_date, store or PanelStore())
    cleaned = _completed_history(frame, reference_date)
    error = _daily_history_error(cleaned)
    if error:
        return None, error
    if cleaned.empty:
        return None, "历史行情在快照截面前为空"
    latest = pd.to_datetime(cleaned["date"]).max().strftime("%Y-%m-%d")
    if latest < reference_date:
        return None, f"历史行情仅到 {latest}，未覆盖快照截面日 {reference_date}"
    if len(cleaned) < settings.MINIMUM_SCAN_HISTORY_BARS:
        return None, f"截断后历史样本不足 {settings.MINIMUM_SCAN_HISTORY_BARS} 根日线"
    return cleaned, None


def resume_records(
    run: AcquisitionRun, candidates: list[dict[str, Any]], reference_date: str, store: PanelStore
) -> tuple[list[dict[str, Any]], set[str], int]:
    """恢复已完成的记录；仅当文件仍覆盖快照截面日时才跳过该股。"""
    ordered: list[dict[str, Any]] = []
    done: set[str] = set()
    skipped = 0
    for candidate in candidates:
        code = str(candidate["code"])
        record = run.prior.get(code)
        if not record:
            continue
        if record.get("error"):
            continue
        latest, bars = _stored_history_status(store, code, reference_date)
        if latest is None or latest < reference_date or bars < settings.MINIMUM_SCAN_HISTORY_BARS:
            continue
        if (run.meta.get("options") or {}).get("include_backtest"):
            if record.get("backtest_error") or not (run.directory / "raw" / f"code={code}.parquet").is_file():
                continue
        ordered.append(record)
        done.add(code)
        skipped += 1
    return ordered, done, skipped


def finalize(
    run: AcquisitionRun,
    records: list[dict[str, Any]],
    benchmark: pd.DataFrame | None,
    benchmark_meta: dict[str, Any] | None,
    exposures: pd.DataFrame | None = None,
    exposure_meta: dict[str, Any] | None = None,
    extra_factors: dict[str, Any] | None = None,
    extra_meta: dict[str, Any] | None = None,
    store: PanelStore | None = None,
) -> AcquisitionSnapshot:
    """逐条校验历史、冻结基准/暴露/因子，把快照置为 ready（或 failed）。"""
    store = store or PanelStore()
    reference_date = str(run.meta.get("reference_date") or "")
    notes = list(run.meta.get("notes") or [])
    validated: list[dict[str, Any]] = []
    success = failed = 0
    for record in records:
        # 续跑恢复的记录本来就在清单里；成功时无需重复追加，失败时才写一条更正。
        from_prior = "history" not in record
        cleaned, error = _validate_history(record, reference_date, store)
        if error:
            item = {
                "code": str(record.get("code")),
                "name": record.get("name") or record.get("code"),
                "error": error,
            }
            validated.append(item)
            checkpoint(run, item)
            failed += 1
            continue
        item = {key: value for key, value in record.items() if key not in ("history", "raw_history")}
        item["latest_bar_date"] = str(pd.to_datetime(cleaned["date"]).max().date())
        item["history_bars"] = int(len(cleaned))
        validated.append(item)
        if not from_prior:
            checkpoint(run, item)
        success += 1

    total = int((run.meta.get("counts") or {}).get("total") or len(records))
    skipped = int((run.meta.get("counts") or {}).get("skipped") or 0)
    status = "ready"
    error = None
    if benchmark is None or benchmark.empty:
        status = "failed"
        error = "沪深 300 基准不可用，无法生成训练快照。"
        notes.append({"level": "warning", "text": error})
    elif success == 0:
        status = "failed"
        error = "没有可用的逐股历史，快照不能用于训练。"
        notes.append({"level": "warning", "text": error})
    try:
        if benchmark is not None and not benchmark.empty:
            _atomic_frame(benchmark, run.directory / "benchmark.parquet")
        if exposures is not None and not exposures.empty:
            _atomic_frame(exposures, run.directory / "exposures.parquet")
        extra_factors = extra_factors or {}
        for name in ("intraday", "financial"):
            frame = extra_factors.get(name)
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                _atomic_frame(frame, run.directory / f"{name}.parquet")
    except (OSError, ValueError, ImportError) as exc:
        status = "failed"
        error = f"冻结快照数据失败：{type(exc).__name__}: {exc}"
        LOGGER.exception("冻结采集快照失败 snapshot=%s", run.snapshot_id)

    meta = {
        **run.meta,
        "status": status,
        "error": error,
        "finished_at": _now_text(),
        "benchmark_meta": benchmark_meta or {},
        "exposure_meta": exposure_meta or {},
        "extra_meta": extra_meta or {},
        "counts": {"total": total, "success": success, "failed": failed, "skipped": skipped},
        "notes": notes,
    }
    _atomic_json(run.directory / "meta.json", meta)
    run.meta = meta
    if status == "ready":
        _prune_snapshots(keep=settings.ACQUISITION_SNAPSHOT_LIMIT, current=run.snapshot_id)
    return AcquisitionSnapshot(snapshot_id=run.snapshot_id, directory=run.directory, meta=meta)


def mark_failed(
    run: AcquisitionRun | None,
    error: str,
    *,
    notes: list[dict[str, str]] | None = None,
) -> AcquisitionSnapshot | None:
    """采集在基准阶段就失败时留下一条可展示记录；没有 run 时返回 None。"""
    if run is None:
        return None
    meta = {**run.meta, "status": "failed", "error": error, "finished_at": _now_text()}
    meta["notes"] = [*(meta.get("notes") or []), *(notes or []), {"level": "warning", "text": error}]
    _atomic_json(run.directory / "meta.json", meta)
    run.meta = meta
    return AcquisitionSnapshot(snapshot_id=run.snapshot_id, directory=run.directory, meta=meta)


def _prune_snapshots(keep: int, current: str | None = None) -> None:
    """保留最近 keep 个 ready 快照与所有被归档引用的快照；清理废弃目录。"""
    snapshots = _iter_snapshots()
    referenced = _referenced_snapshot_ids()
    ready = sorted(
        ((directory, meta) for directory, meta in snapshots if meta.get("status") == "ready"),
        key=lambda item: str(item[1].get("created_at") or ""), reverse=True,
    )
    failed = sorted(
        ((directory, meta) for directory, meta in snapshots if meta.get("status") == "failed"),
        key=lambda item: str(item[1].get("created_at") or ""), reverse=True,
    )
    protected = {directory.name for directory, _ in ready[: max(1, int(keep))]}
    # 最近一次失败留作排障线索，更早的失败目录没有长期价值。
    protected |= {directory.name for directory, _ in failed[:1]}
    protected |= referenced
    if current:
        protected.add(current)
    for directory, meta in snapshots:
        status = meta.get("status")
        if status == "collecting" or directory.name in protected:
            continue
        if status in ("ready", "abandoned", "failed"):
            shutil.rmtree(directory, ignore_errors=True)


def latest_snapshot(ready_only: bool = False) -> AcquisitionSnapshot | None:
    snapshots = sorted(_iter_snapshots(), key=lambda item: str(item[1].get("created_at") or ""), reverse=True)
    for directory, meta in snapshots:
        if ready_only and meta.get("status") != "ready":
            continue
        if meta.get("status") == "abandoned":
            continue
        return AcquisitionSnapshot(snapshot_id=directory.name, directory=directory, meta=meta)
    return None


def summary(snapshot_id: str | None = None) -> dict[str, Any] | None:
    """页面展示用的快照摘要；未完成的快照用 jsonl 现算计数。"""
    if snapshot_id:
        directory = _snapshot_path(snapshot_id)
        meta = _read_json(directory / "meta.json")
        if not meta:
            return None
    else:
        snapshot = latest_snapshot()
        if snapshot is None:
            return None
        directory, meta = snapshot.directory, snapshot.meta
    candidates = set(str(code) for code in meta.get("candidate_codes") or [])
    items = {code: item for code, item in _read_items(directory).items() if code in candidates}
    successes = sum(1 for item in items.values() if not item.get("error"))
    failures = sum(1 for item in items.values() if item.get("error"))
    return {
        "snapshot_id": directory.name,
        "status": meta.get("status"),
        "error": meta.get("error"),
        "created_at": meta.get("created_at"),
        "finished_at": meta.get("finished_at"),
        "resumed_at": meta.get("resumed_at"),
        "reference_date": meta.get("reference_date"),
        "pool_mode": (meta.get("options") or {}).get("pool_mode"),
        "custom_codes": (meta.get("options") or {}).get("custom_codes"),
        "counts": meta.get("counts") or {},
        "recorded_success": successes,
        "recorded_failed": failures,
        "source_meta": meta.get("source_meta") or {},
        "notes": meta.get("notes") or [],
        "extra_meta": meta.get("extra_meta") or {},
        "path": str(directory),
    }


def load_snapshot(
    snapshot_id: str | None = None, store: PanelStore | None = None
) -> LoadedSnapshot:
    """加载 ready 快照：回读历史并截断到快照截面日；任何不一致都明确报错。"""
    if snapshot_id:
        directory = _snapshot_path(snapshot_id)
        meta = _read_json(directory / "meta.json")
    else:
        snapshot = latest_snapshot(ready_only=True)
        if snapshot is None:
            raise ValueError("没有可训练的采集快照，请先采集股票池数据。")
        directory, meta = snapshot.directory, snapshot.meta
    if not meta:
        raise ValueError("采集快照不存在或元数据已损坏。")
    if meta.get("status") != "ready":
        raise ValueError(f"采集快照尚未就绪（状态：{meta.get('status')}），请先完成采集。")
    if meta.get("model_version") != settings.MODEL_VERSION:
        raise ValueError("采集快照的模型版本与当前版本不一致，请重新采集。")
    reference_date = str(meta.get("reference_date") or "")
    if not reference_date:
        raise ValueError("采集快照缺少截面日期，不能训练。")
    benchmark_path = directory / "benchmark.parquet"
    if not benchmark_path.exists():
        raise ValueError("采集快照缺少沪深 300 基准，请重新采集。")
    benchmark = pd.read_parquet(benchmark_path)
    if benchmark.empty or "date" not in benchmark:
        raise ValueError("采集快照的沪深 300 基准为空，请重新采集。")
    benchmark_dates = pd.to_datetime(benchmark["date"], errors="coerce")
    benchmark = benchmark.loc[benchmark_dates.notna() & (benchmark_dates.dt.strftime("%Y-%m-%d") <= reference_date)]
    if benchmark.empty:
        raise ValueError("采集快照的沪深 300 基准未覆盖截面日期，请重新采集。")
    benchmark_latest = pd.to_datetime(benchmark["date"]).max().strftime("%Y-%m-%d")
    if benchmark_latest != reference_date:
        raise ValueError(
            f"采集快照基准最新日为 {benchmark_latest}，与截面日期 {reference_date} 不一致，请重新采集。"
        )
    store = store or PanelStore()
    items = _read_items(directory)
    order = [str(code) for code in meta.get("candidate_codes") or []]
    records: list[dict[str, Any]] = []
    histories: dict[str, pd.DataFrame] = {}
    failures: list[dict[str, str]] = []
    for code in order:
        record = items.get(code)
        if not record:
            failures.append({"code": code, "name": code, "error": "快照清单中没有该股的采集记录"})
            continue
        cleaned, error = _validate_history(record, reference_date, store)
        if error:
            failures.append({"code": code, "name": record.get("name") or code, "error": error})
            continue
        histories[code] = cleaned
        item = {**record, "history": cleaned}
        if (meta.get("options") or {}).get("include_backtest"):
            path = directory / "raw" / f"code={code}.parquet"
            try:
                raw = _completed_history(pd.read_parquet(path), reference_date)
                if raw.empty or pd.to_datetime(raw["date"]).max().strftime("%Y-%m-%d") != reference_date:
                    raise ValueError("回测日线未覆盖快照截面日")
                error = _daily_history_error(raw)
                if error:
                    raise ValueError(error)
                item["raw_history"] = raw
            except (OSError, ValueError, ImportError) as exc:
                item["backtest_error"] = record.get("backtest_error") or f"快照回测日线不可用：{exc}"
        records.append(item)
    if not histories:
        raise ValueError("采集快照里没有可用的逐股历史，请重新采集。")
    exposures = None
    exposures_path = directory / "exposures.parquet"
    if exposures_path.exists():
        exposures = pd.read_parquet(exposures_path)
    extra_factors: dict[str, Any] = {}
    if (meta.get("options") or {}).get("use_extra_factors"):
        extra_factors["chip"] = histories
        for name in ("intraday", "financial"):
            path = directory / f"{name}.parquet"
            if path.exists():
                extra_factors[name] = pd.read_parquet(path)
    return LoadedSnapshot(
        snapshot_id=directory.name, meta=meta, records=records, histories=histories,
        benchmark=benchmark, exposures=exposures, extra_factors=extra_factors, failures=failures,
    )
