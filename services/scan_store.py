"""最近一次扫描结果的落盘与读回。

扫描排名原先只写在 Streamlit 会话里，刷新即丢，因此除了点按钮的那个人，谁也不知道
今天扫描选出了什么。定时任务跑完之后更需要在界面上留下痕迹，故把它落成一份 JSON。

只保存完整的轻量结果表，不存全池逐股分析对象，供页面先按板块筛选、再限制显示行数。
"""

from __future__ import annotations

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

# 逐笔成交、每日净值、每段训练日志与期末持仓都随回测长度增长，落盘会让文件大到没必要。
# 只读视图只展示汇总口径；这些明细留在跑出它的那个会话里，页面上对应的展开区本就有守卫。
PORTFOLIO_DETAIL_KEYS = ("trades", "daily_equity", "training_log", "open_positions")


def _compact_portfolio(portfolio_result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not portfolio_result:
        return None
    return {key: value for key, value in portfolio_result.items() if key not in PORTFOLIO_DETAIL_KEYS}


def build_payload(result: ScanResult) -> dict[str, Any]:
    """把扫描产物整理成页面直接渲染的字典。

    落盘与不落盘两条展示路径共用这一份口径：页面上的试跑结果不写文件，但必须与
    读回的文件长得一模一样，否则同一张表会因为来源不同而少几个字段。
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


def save_scan_result(result: ScanResult, path: Path | None = None) -> Path:
    """原子写入最近一次扫描结果，避免读到写了一半的文件。"""
    target = Path(path or SCAN_RESULT_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(build_payload(result), ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(temporary, target)
    LOGGER.info("扫描结果已落盘 path=%s kind=%s", target, result.options.trigger)
    return target


def load_scan_result(path: Path | None = None) -> dict[str, Any] | None:
    """读回最近一次扫描结果；缺失、损坏或模型版本不符都返回 None，不抛异常。

    与 services/ml_pipeline.py 的 load_context 同口径：模型版本升级后旧结果自动作废，
    免得页面上显示一份与当前模型对不上的排名。
    """
    target = Path(path or SCAN_RESULT_PATH)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        if target.exists():
            LOGGER.warning("扫描结果不可读 path=%s detail=%s", target, exc)
        return None
    if not isinstance(payload, dict) or payload.get("model_version") != settings.MODEL_VERSION:
        return None
    if not production_report_is_current(payload.get("style_factors")):
        LOGGER.warning("扫描结果的风格规则或独立验收已失效，需重新训练")
        return None
    return payload
