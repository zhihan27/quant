"""已保存模型的轻量目录：只读摘要，不反序列化 joblib。

个股分析页要列出「可以选哪个模型」，但完整上下文有几十兆。这里只读
``models/archive/<context_id>/registry.json`` 与活动模型摘要 JSON，
再校验三个归档文件确实存在；数据库不可用时自动回落到本地归档目录。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from config import settings

LOGGER = logging.getLogger("quant.model_catalog")

_ARCHIVE_FILES = (
    "market_context_v{version}.joblib",
    "market_context_v{version}.json",
    "ml_panel_v{version}.parquet",
)


def _archive_root() -> Path:
    return settings.MODEL_DIR / settings.MODEL_ARCHIVE_DIRNAME


def _active_summary() -> dict[str, Any] | None:
    path = settings.MODEL_DIR / f"market_context_v{settings.MODEL_VERSION}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        LOGGER.exception("读取活动模型摘要失败")
        return None
    if not isinstance(payload, dict) or payload.get("model_version") != settings.MODEL_VERSION:
        return None
    return payload


def _details(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _archive_entry(context_id: str) -> dict[str, Any] | None:
    """读一个归档目录的 registry.json，并确认三份文件都在。"""
    version = settings.MODEL_VERSION
    directory = _archive_root() / context_id
    registry_path = directory / "registry.json"
    if not registry_path.is_file():
        return None
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        LOGGER.warning("模型归档登记损坏，已跳过：%s", context_id)
        return None
    if not isinstance(registry, dict):
        return None
    if registry.get("context_id") != context_id or registry.get("model_version") != version:
        return None
    if not all((directory / name.format(version=version)).exists() for name in _ARCHIVE_FILES):
        return None
    return {
        "context_id": context_id,
        "model_version": version,
        "created_at": registry.get("created_at"),
        "reference_date": registry.get("reference_date"),
        "passed": bool(registry.get("passed")),
        "objective_mode": registry.get("objective_mode"),
        "training_window_days": registry.get("training_window_days"),
        "pool_mode": registry.get("pool_mode"),
        "universe_size": registry.get("universe_size"),
        "source_snapshot_id": registry.get("source_snapshot_id"),
        "archive_path": str(directory),
        "source": "archive",
    }


def _from_repository(repository: Any) -> list[str]:
    """数据库登记只是线索；id 仍要回到归档目录逐条验证。"""
    from services.database import DatabaseError
    from services.model_governance import registry_versions

    try:
        rows = registry_versions(repository, limit=settings.MODEL_CATALOG_LIMIT * 3)
    except DatabaseError:
        LOGGER.warning("模型登记库不可用，回落到本地归档目录列表", exc_info=True)
        return []
    return [
        str(row.get("context_id"))
        for row in rows
        if row.get("context_id") and row.get("model_version") == settings.MODEL_VERSION
    ]


def _from_archive() -> list[str]:
    root = _archive_root()
    if not root.is_dir():
        return []
    ids = [path.name for path in root.iterdir() if path.is_dir() and (path / "registry.json").is_file()]
    # 新目录通常更值得展示；created_at 不完整时仍保持稳定的倒序回退。
    return sorted(ids, key=lambda name: ((root / name).stat().st_mtime, name), reverse=True)


def saved_models(repository: Any | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    """已保存模型列表，新到旧；``repository`` 为 None 或数据库失败时读本地归档。"""
    maximum = settings.MODEL_CATALOG_LIMIT if limit is None else max(int(limit), 1)
    active = _active_summary()
    active_id = str(active.get("context_id") or "") if active else ""

    ordered: list[str] = []
    seen: set[str] = set()
    candidates = _from_repository(repository) if repository is not None else []
    candidates.extend(_from_archive())
    for context_id in candidates:
        if context_id in seen:
            continue
        seen.add(context_id)
        ordered.append(context_id)

    models: list[dict[str, Any]] = []
    for context_id in ordered:
        entry = _archive_entry(context_id)
        if entry is None:
            continue
        entry["is_active"] = bool(active_id and context_id == active_id)
        models.append(entry)

    # 活动模型可能还没有归档（例如刚开始使用的新模型），仍要允许用户选择当前结果。
    if active and active_id and active_id not in {item["context_id"] for item in models}:
        version = settings.MODEL_VERSION
        if (settings.MODEL_DIR / f"market_context_v{version}.joblib").exists():
            models.append({
                "context_id": active_id,
                "model_version": version,
                "created_at": active.get("created_at"),
                "reference_date": active.get("reference_date"),
                "passed": None,
                "objective_mode": active.get("objective_mode"),
                "training_window_days": active.get("training_window_days"),
                "pool_mode": (active.get("source") or {}).get("pool_mode"),
                "universe_size": len(active.get("universe_codes") or []),
                "source_snapshot_id": (active.get("source") or {}).get("snapshot_id"),
                "archive_path": None,
                "source": "active",
                "is_active": True,
            })

    models.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return models[:maximum]
