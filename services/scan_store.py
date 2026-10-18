"""最近一次扫描结果的落盘与读回。

全市场扫描结果保存在共享 JSON；自定义试跑结果按账号保存在独立 JSON，避免登录状态变化时丢失。

只保存完整的轻量结果表，不存全池逐股分析对象，供页面先按板块筛选、再限制显示行数。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from config import settings
from services.scan_pipeline import ScanResult
from services.style_factors import production_report_is_current

LOGGER = logging.getLogger("quant.scan_store")

SCAN_RESULT_PATH = settings.DATA_DIR / "last_scan.json"
USER_SCAN_RESULT_DIR = settings.DATA_DIR / "user_scans"

# 逐笔成交、每日净值、每段训练日志与期末持仓都随回测长度增长，落盘会让文件大到没必要。
# 只读视图只展示汇总口径；这些明细留在跑出它的那个会话里，页面上对应的展开区本就有守卫。
PORTFOLIO_DETAIL_KEYS = ("trades", "daily_equity", "training_log", "open_positions")


def _compact_portfolio(portfolio_result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not portfolio_result:
        return None
    return {key: value for key, value in portfolio_result.items() if key not in PORTFOLIO_DETAIL_KEYS}


def build_payload(result: ScanResult) -> dict[str, Any]:
    """把扫描产物整理成页面直接渲染的字典。

    全市场与个人试跑共用这一份口径，确保页面展示与读回文件保持一致。
    """
    limit = int(result.options.scan_limit)
    return {
        "model_version": settings.MODEL_VERSION,
        "finished_at": result.finished_at,
        "trigger": result.options.trigger,
        "pool_mode": result.options.pool_mode,
        "reference_date": result.reference_date,
        "context_id": result.context_id,
        "scan_stats": result.scan_stats,
        "display_limit": limit,
        "rows": result.rows,
        "detail_rows": result.detail_rows,
        "failed_rows": result.failed_rows,
        "usable": result.usable,
        "notes": result.notes,
        "market_regime": result.market_regime,
        "actionable_rankings": result.actionable_rankings,
        "style_factors": result.style_factors,
        "portfolio": _compact_portfolio(result.portfolio_result),
    }


def _save_payload(payload: dict[str, Any], target: Path) -> Path:
    """原子保存扫描结果，避免并行读取到写了一半的文件。"""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(temporary, target)
    return target


def save_scan_result(result: ScanResult, path: Path | None = None) -> Path:
    """原子写入共享的最近一次全市场扫描结果。"""
    target = Path(path or SCAN_RESULT_PATH)
    _save_payload(build_payload(result), target)
    LOGGER.info("扫描结果已落盘 path=%s kind=%s", target, result.options.trigger)
    return target


def _read_payload(target: Path) -> dict[str, Any] | None:
    """读取 JSON 结果；文件缺失或损坏时返回 None。"""
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        if target.exists():
            LOGGER.warning("扫描结果不可读 path=%s detail=%s", target, exc)
        return None
    return payload if isinstance(payload, dict) else None


def load_scan_result(
    path: Path | None = None, *, require_current_model: bool = True,
) -> dict[str, Any] | None:
    """读回最近一次全市场结果；显示历史结果时可保留旧版本快照。

    模型版本或特征验收不匹配时，调仓等当前模型业务仍拒绝使用；选股页面可以选择
    显示旧扫描快照，并明确标注它对应的历史模型版本。
    """
    target = Path(path or SCAN_RESULT_PATH)
    payload = _read_payload(target)
    if payload is None:
        return None
    if require_current_model:
        if payload.get("model_version") != settings.MODEL_VERSION:
            return None
        if not production_report_is_current(payload.get("style_factors")):
            LOGGER.warning("扫描结果的风格规则或独立验收已失效，需重新训练")
            return None
    return payload


def _user_scan_path(username: str) -> Path:
    """将账号名映射为不可逆文件名，个人扫描结果不会与其他账号混用。"""
    normalized = str(username).strip().casefold()
    if not normalized:
        raise ValueError("保存个人扫描结果需要有效的用户名。")
    identifier = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return USER_SCAN_RESULT_DIR / f"{identifier}.json"


def save_user_scan_result(result: ScanResult, username: str) -> Path:
    """原子保存当前账号的自定义股票池试跑结果。"""
    target = _user_scan_path(username)
    _save_payload(build_payload(result), target)
    LOGGER.info("个人扫描结果已保存 username_hash=%s", target.stem[:12])
    return target


def load_user_scan_result(username: str) -> dict[str, Any] | None:
    """读取当前账号最近一次自定义股票池试跑结果。"""
    try:
        return _read_payload(_user_scan_path(username))
    except ValueError:
        return None
