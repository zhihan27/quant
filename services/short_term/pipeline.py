"""从本地日线快照运行短线滚动研究和最新截面排序。"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from config.short_term import (
    SHORT_TERM_FEATURES,
    SHORT_TERM_MAIN_HORIZON,
    SHORT_TERM_MIN_HISTORY_BARS,
    SHORT_TERM_RESEARCH_COSTS,
    SHORT_TERM_SNAPSHOT_DIR,
    SHORT_TERM_ACCEPTANCE_POLICY_PATH,
    SHORT_TERM_STRATEGY_ID,
    SHORT_TERM_STRATEGY_VERSION,
)
from services.market_data import completed_daily_history
from services.panel_store import PanelStore
from services.short_term.features import build_short_features, build_short_labels
from services.short_term.execution_replay import run_execution_replay
from services.short_term.cost_stress import run_cost_stress_tests
from services.short_term.governance import evaluate_promotion_gate, load_acceptance_policy
from services.short_term.models import fit_short_ranker, save_short_ranker
from services.short_term.point_in_time import collect_point_in_time_snapshot
from services.short_term.probability import walk_forward_probability_validation
from services.short_term.validation import walk_forward_validation

LOGGER = logging.getLogger("quant.short_term")
SHANGHAI = ZoneInfo("Asia/Shanghai")


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "未安装"


def _save_immutable_snapshot(frame: pd.DataFrame, run_id: str) -> tuple[Path, str]:
    """原子保存每轮独立快照并返回文件哈希。"""
    directory = SHORT_TERM_SNAPSHOT_DIR / run_id
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "features.parquet"
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest


def _load_local_features(panel_store: PanelStore | None = None) -> tuple[pd.DataFrame, dict[str, int]]:
    """逐文件读取本地前复权日线，不在页面任务中重新抓取全市场行情。"""
    store = panel_store or PanelStore()
    codes = store.history_codes(adjustment="qfq", period="daily")
    frames = []
    stats = {"cached_codes": len(codes), "usable_codes": 0, "short_history_codes": 0, "invalid_history_codes": 0}
    for code in codes:
        history = store.read_history(code, adjustment="qfq", period="daily")
        if history is None or history.empty:
            stats["invalid_history_codes"] += 1
            continue
        try:
            history = completed_daily_history(history).tail(800)
            if len(history) < int(SHORT_TERM_MIN_HISTORY_BARS):
                stats["short_history_codes"] += 1
                continue
            features = build_short_features({code: history}, minimum_history=20)
            features = features.dropna(subset=list(SHORT_TERM_FEATURES))
            if features.empty:
                stats["invalid_history_codes"] += 1
                continue
            frames.append(features)
            stats["usable_codes"] += 1
        except (KeyError, TypeError, ValueError) as exc:
            stats["invalid_history_codes"] += 1
            LOGGER.warning("短线日线被排除 code=%s reason=%s", code, exc)
    if not frames:
        return pd.DataFrame(), stats
    frame = pd.concat(frames, ignore_index=True)
    frame["code"] = frame["code"].astype("category")
    return frame.sort_values(["date", "code"], kind="stable").reset_index(drop=True), stats


def _load_replay_market_inputs(
    panel_store: PanelStore, codes: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """读取原始日线及显式点时主数据文件；不以当前状态冒充历史状态。"""
    raw_codes = set(panel_store.history_codes(adjustment="raw", period="daily"))
    bars = []
    for code in sorted(codes.intersection(raw_codes)):
        history = panel_store.read_history(code, adjustment="raw", period="daily")
        if history is None or history.empty or not {"date", "open", "high", "low", "close", "volume"}.issubset(history.columns):
            continue
        history = completed_daily_history(history)
        selected = history[["date", "open", "high", "low", "close", "volume"]].copy()
        selected["code"] = str(code).zfill(6)
        bars.append(selected)
    daily_bars = pd.concat(bars, ignore_index=True) if bars else pd.DataFrame()

    def read_history_file(path: Path) -> pd.DataFrame:
        if not path.is_file():
            return pd.DataFrame()
        try:
            return pd.read_parquet(path)
        except (OSError, ValueError) as exc:
            LOGGER.warning("短线回放主数据读取失败 path=%s reason=%s", path.name, exc)
            return pd.DataFrame()

    point_in_time_universe = panel_store.load_point_in_time_history("membership")
    legacy_universe = read_history_file(panel_store.universe_dir / "membership_history.parquet")
    universe_frames = [frame for frame in (point_in_time_universe, legacy_universe) if frame is not None and not frame.empty]
    universe = pd.concat(universe_frames, ignore_index=True).drop_duplicates(
        ["code", "date", "available_at"], keep="last"
    ) if universe_frames else pd.DataFrame()
    point_in_time_status = panel_store.load_point_in_time_history("security_status")
    legacy_status = read_history_file(panel_store.universe_dir / "security_status_history.parquet")
    status_frames = [frame for frame in (point_in_time_status, legacy_status) if frame is not None and not frame.empty]
    security_status = pd.concat(status_frames, ignore_index=True).drop_duplicates(
        ["code", "date", "available_at"], keep="last"
    ) if status_frames else pd.DataFrame()
    return daily_bars, universe, security_status


def run_short_term_research(
    run_id: str, horizon: int = SHORT_TERM_MAIN_HORIZON, panel_store: PanelStore | None = None
) -> dict[str, Any]:
    """运行独立日频研究；缺少点时股票池与真实撮合字段时固定保持研究态。"""
    store = panel_store or PanelStore()
    features, data_stats = _load_local_features(store)
    if features.empty:
        return {"status": "insufficient_data", "message": "没有满足最短历史要求的本地日线。", "data": data_stats}
    latest_session = pd.Timestamp(features["date"].max()).normalize()
    point_in_time_snapshot = collect_point_in_time_snapshot(store, session_date=latest_session)
    available_at = point_in_time_snapshot.get("available_at")
    if point_in_time_snapshot.get("membership_saved") and available_at:
        latest_rows = pd.to_datetime(features["date"]).dt.normalize().eq(latest_session)
        features.loc[latest_rows, "signal_time"] = pd.Timestamp(available_at).tz_convert(SHANGHAI).tz_localize(None)
    labels = build_short_labels(features, horizon=horizon, costs=SHORT_TERM_RESEARCH_COSTS)
    label_columns = labels[["code", "date", "entry_date", "entry_open", "label_end_date", "exit_open", "target_net_return", "label_status", "label_basis"]]
    panel = features.merge(label_columns, on=["code", "date"], how="left", validate="one_to_one")
    snapshot_path, snapshot_hash = _save_immutable_snapshot(panel, run_id)
    membership_history = store.load_point_in_time_history("membership")
    legacy_membership_path = store.universe_dir / "membership_history.parquet"
    if legacy_membership_path.is_file():
        try:
            legacy_membership = pd.read_parquet(legacy_membership_path)
            membership_history = pd.concat(
                [frame for frame in (membership_history, legacy_membership) if frame is not None and not frame.empty],
                ignore_index=True,
            )
        except (OSError, ValueError) as exc:
            LOGGER.warning("读取旧版点时股票池失败 detail=%s", exc)
    validation = walk_forward_validation(
        panel,
        horizon=horizon,
        include_oos_signals=True,
        membership_history=membership_history,
    )
    oos_signals = validation.pop("_out_of_sample_signals", [])
    probability_validation = walk_forward_probability_validation(panel, horizon=horizon)
    acceptance_policy, acceptance_policy_hash, acceptance_policy_error = load_acceptance_policy(SHORT_TERM_ACCEPTANCE_POLICY_PATH)
    signal_frame = pd.DataFrame(oos_signals)
    replay_bars, universe_history, security_status_history = _load_replay_market_inputs(
        store, set(signal_frame["code"].astype(str)) if not signal_frame.empty else set()
    )
    if not signal_frame.empty and not replay_bars.empty:
        reference_prices = replay_bars[["code", "date", "close"]].rename(columns={"close": "reference_price"})
        signal_frame["code"] = signal_frame["code"].astype(str).str.zfill(6)
        signal_frame["date"] = pd.to_datetime(signal_frame["date"], errors="coerce").dt.normalize()
        reference_prices["code"] = reference_prices["code"].astype(str).str.zfill(6)
        reference_prices["date"] = pd.to_datetime(reference_prices["date"], errors="coerce").dt.normalize()
        signal_frame = signal_frame.merge(reference_prices, on=["code", "date"], how="left", validate="many_to_one")
        signal_frame["signal_time"] = pd.to_datetime(signal_frame["date"]) + pd.Timedelta(hours=15, minutes=5)
    execution_replay = run_execution_replay(
        signal_frame,
        replay_bars,
        universe_history,
        security_status_history,
        acceptance_policy.get("execution_terms") if isinstance(acceptance_policy, dict) else None,
        horizon=int(horizon),
        top_k=int(validation.get("top_k", 20)),
    )
    replay_trades = execution_replay.pop("trades", [])
    replay_trade_log_path = None
    replay_trade_log_hash = None
    if replay_trades:
        trade_log = snapshot_path.parent / "execution_replay.json"
        trade_log.write_text(json.dumps(replay_trades, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        replay_trade_log_path = str(trade_log)
        replay_trade_log_hash = hashlib.sha256(trade_log.read_bytes()).hexdigest()
    execution_replay["trade_log_path"] = replay_trade_log_path
    execution_replay["trade_log_hash"] = replay_trade_log_hash
    cost_stress = run_cost_stress_tests(
        signal_frame,
        replay_bars,
        universe_history,
        security_status_history,
        acceptance_policy,
        horizon=int(horizon),
        top_k=int(validation.get("top_k", 20)),
    )
    replay_has_coverage = execution_replay.get("status") in ("partial_coverage", "daily_bar_proxy_not_queue_truth")
    membership_coverage = validation.get("point_in_time_filter", {})
    pit_universe_verified = bool(
        oos_signals
        and membership_coverage.get("coverage_ratio") == 1.0
        and replay_has_coverage
        and not execution_replay.get("coverage_blockers")
    )
    pit_security_verified = bool(
        isinstance(security_status_history, pd.DataFrame)
        and not security_status_history.empty
        and oos_signals
        and replay_has_coverage
        and not execution_replay.get("coverage_blockers")
    )
    replay_evidence = {
        "point_in_time_universe": pit_universe_verified,
        "point_in_time_security_status": pit_security_verified,
        "execution_replay": bool(execution_replay.get("promotion_evidence")),
        "cost_stress": bool(cost_stress.get("promotion_evidence")),
        "forward_observation": False,
    }
    warnings = [
        "标签仍是前复权开盘价比例成本代理，未模拟排队、实际未成交、部分成交、最低佣金及真实批次可卖状态。",
        "回放使用原始日线触价近似，仍无法还原集合竞价、盘口队列和真实撮合；该回放不作为生产晋级证据。",
        "模型输出只表示同一截面内的相对排序，不是概率或目标价，策略尚未通过生产晋级。",
    ]
    if not pit_universe_verified or not pit_security_verified:
        warnings.insert(0, "当前本地数据没有完整的点时历史股票池或证券状态，相关结果不得视为无幸存者偏差回测。")
    promotion_gate = evaluate_promotion_gate(
        validation,
        replay_evidence,
        acceptance_policy,
    )
    result: dict[str, Any] = {
        "status": "research_only",
        "strategy_id": SHORT_TERM_STRATEGY_ID,
        "strategy_version": SHORT_TERM_STRATEGY_VERSION,
        "horizon": int(horizon),
        "signal_date": pd.Timestamp(features["date"].max()).date().isoformat(),
        "snapshot_path": str(snapshot_path),
        "snapshot_hash": snapshot_hash,
        "data": data_stats,
        "validation": validation,
        "probability_validation": probability_validation,
        "execution_replay": execution_replay,
        "cost_stress": cost_stress,
        "promotion_evidence": replay_evidence,
        "promotion_gate": promotion_gate,
        "acceptance_policy_hash": acceptance_policy_hash,
        "acceptance_policy_error": acceptance_policy_error,
        "point_in_time_snapshot": point_in_time_snapshot,
        "warnings": warnings,
    }
    try:
        ranker, candidates, training_metadata = fit_short_ranker(panel)
    except ValueError as exc:
        result.update({"status": "insufficient_data", "message": str(exc)})
        return result
    model_path = save_short_ranker(ranker, run_id)
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    manifest = {
        **training_metadata,
        "strategy_id": SHORT_TERM_STRATEGY_ID,
        "horizon": int(horizon),
        "snapshot_hash": snapshot_hash,
        "model_hash": model_hash,
        "feature_version": "daily-price-v1",
        "label_basis": "qfq_open_price_proxy_no_execution_state",
        "costs": dict(SHORT_TERM_RESEARCH_COSTS),
        "environment": {"python": __import__("platform").python_version(), "pandas": _package_version("pandas"), "lightgbm": _package_version("lightgbm")},
        "promotion_status": "research_only",
        "promotion_gate": promotion_gate,
        "acceptance_policy_hash": acceptance_policy_hash,
        "probability_validation": probability_validation,
        "execution_replay": execution_replay,
        "cost_stress": cost_stress,
        "promotion_evidence": replay_evidence,
        "acceptance_policy_error": acceptance_policy_error,
        "historical_universe_point_in_time": pit_universe_verified,
    }
    manifest_path = model_path.with_name("manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    predictions = []
    prediction_created_at = datetime.now(SHANGHAI)
    for row in candidates.itertuples(index=False):
        candidate_time = pd.Timestamp(row.signal_time)
        if candidate_time.tzinfo is None:
            candidate_time = candidate_time.tz_localize(SHANGHAI)
        if candidate_time.date() == prediction_created_at.date():
            candidate_time = prediction_created_at
        predictions.append(
            {
                "code": str(row.code),
                "signal_time": candidate_time.isoformat(),
                "ranking_score": float(row.ranking_score),
                "rank": int(row.rank),
                "percentile": float(row.percentile),
                "features": {column: float(getattr(row, column)) for column in SHORT_TERM_FEATURES},
                "status": "research_only",
            }
        )
    result.update(
        {
            "model_path": str(model_path),
            "model_hash": model_hash,
            "manifest_path": str(manifest_path),
            "training": training_metadata,
            "prediction_count": len(predictions),
            "predictions": predictions,
        }
    )
    return result
