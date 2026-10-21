"""将冻结预测与后续复权日线结果配对，保留价格代理口径。"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from config.short_term import SHORT_TERM_RESEARCH_COSTS
from services.market_data import completed_daily_history
from services.panel_store import PanelStore
from services.short_term.store import ShortTermStore

LOGGER = logging.getLogger("quant.short_term.forward_observation")
SHANGHAI = ZoneInfo("Asia/Shanghai")
LABEL_BASIS = "qfq_open_price_proxy_no_execution_state"


def observe_pending_predictions(
    store: ShortTermStore,
    panel_store: PanelStore | None = None,
    limit: int = 10000,
) -> dict[str, Any]:
    """持久化成熟预测的次日开盘入场及 H 间隔后开盘退出代理结果。"""
    panels = panel_store or PanelStore()
    predictions = store.pending_forward_observations(limit=limit)
    if not predictions:
        return {
            "status": "no_pending_predictions",
            "observed_count": 0,
            "pending_count": 0,
            "promotion_evidence": False,
            "label_basis": LABEL_BASIS,
            "blockers": ["目前没有待观察的冻结预测。"],
        }

    histories: dict[str, pd.DataFrame] = {}
    for code in sorted({str(row["code"]).zfill(6) for row in predictions}):
        history = panels.read_history(code, adjustment="qfq", period="daily")
        if history is None or history.empty or not {"date", "open"}.issubset(history.columns):
            continue
        try:
            history = completed_daily_history(history).copy()
            history["date"] = pd.to_datetime(history["date"], errors="coerce").dt.normalize()
            history["open"] = pd.to_numeric(history["open"], errors="coerce")
            history = history.dropna(subset=["date", "open"]).sort_values("date", kind="stable")
            history = history.drop_duplicates("date", keep="last").reset_index(drop=True)
            histories[code] = history
        except (KeyError, TypeError, ValueError) as exc:
            LOGGER.warning("前向观察日线无效 code=%s detail=%s", code, exc)

    observations = []
    for prediction in predictions:
        code = str(prediction["code"]).zfill(6)
        history = histories.get(code)
        if history is None:
            continue
        signal_time = pd.Timestamp(prediction["signal_time"])
        if signal_time.tzinfo is None:
            signal_time = signal_time.tz_localize(SHANGHAI)
        else:
            signal_time = signal_time.tz_convert(SHANGHAI)
        signal_date = signal_time.tz_localize(None).normalize()
        signal_positions = np.flatnonzero(history["date"].eq(signal_date).to_numpy())
        if len(signal_positions) != 1:
            continue
        horizon = int(prediction["horizon"])
        entry_index = int(signal_positions[0]) + 1
        exit_index = int(signal_positions[0]) + horizon + 1
        if exit_index >= len(history):
            continue
        entry = history.iloc[entry_index]
        exit_row = history.iloc[exit_index]
        entry_open = float(entry["open"])
        exit_open = float(exit_row["open"])
        if not np.isfinite(entry_open) or not np.isfinite(exit_open) or entry_open <= 0 or exit_open <= 0:
            continue

        buy_cost = float(SHORT_TERM_RESEARCH_COSTS["commission_rate"]) + float(SHORT_TERM_RESEARCH_COSTS["slippage_rate"])
        sell_cost = buy_cost + float(SHORT_TERM_RESEARCH_COSTS["stamp_duty_rate"])
        gross_return = exit_open / entry_open - 1.0
        target_net_return = exit_open * (1.0 - sell_cost) / (entry_open * (1.0 + buy_cost)) - 1.0
        source_basis = {
            "code": code,
            "signal_time": signal_time.isoformat(),
            "entry_date": pd.Timestamp(entry["date"]).date().isoformat(),
            "entry_open": entry_open,
            "label_end_date": pd.Timestamp(exit_row["date"]).date().isoformat(),
            "exit_open": exit_open,
            "history_source": str(history["history_source"].iloc[-1]) if "history_source" in history.columns else None,
            "costs": SHORT_TERM_RESEARCH_COSTS,
        }
        source_hash = hashlib.sha256(
            json.dumps(source_basis, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        observations.append(
            {
                "run_id": str(prediction["run_id"]),
                "code": code,
                "horizon": horizon,
                "signal_time": signal_time.isoformat(),
                "entry_date": pd.Timestamp(entry["date"]).date(),
                "entry_open": entry_open,
                "label_end_date": pd.Timestamp(exit_row["date"]).date(),
                "exit_open": exit_open,
                "gross_return": gross_return,
                "target_net_return": target_net_return,
                "label_basis": LABEL_BASIS,
                "source_hash": source_hash,
                "details": {
                    "source": "local_qfq_daily_history",
                    "costs": dict(SHORT_TERM_RESEARCH_COSTS),
                    "actual_fills_verified": False,
                },
            }
        )

    saved_count = store.save_forward_observations(observations)
    pending_count = max(len(predictions) - len(observations), 0)
    returns = [row["target_net_return"] for row in observations]
    return {
        "status": "price_proxy_observed" if saved_count else "awaiting_maturity_or_history",
        "observed_count": saved_count,
        "pending_count": pending_count,
        "sampled_pending_count": len(predictions),
        "mean_net_return": float(np.mean(returns)) if returns else None,
        "positive_return_ratio": float(np.mean(np.asarray(returns) > 0)) if returns else None,
        "label_basis": LABEL_BASIS,
        "promotion_evidence": False,
        "blockers": [
            "观察来自本地前复权日线和比例成本代理，不含真实委托、部分成交、最低佣金或公司行为税务核验。",
            "该观察用于前向研究诊断，不能作为实盘执行口径晋级证据。",
        ],
    }
