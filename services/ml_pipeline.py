"""按交易日训练截面排序模型，预测相对沪深300的优胜劣汰并严格验证。"""

from __future__ import annotations

import json
import hashlib
import logging
import math
import os
import pickle
import platform
import re
import sys
from time import perf_counter
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from config import settings
from services.dataframe_utils import column_values_without_attrs
from services.extra_factors import EXTRA_FACTOR_FAMILIES, apply_health_veto, attach_extra_factors
from services.feature_engineering import FEATURE_COLUMNS, add_forward_returns, build_features, normalize_cross_section
from services.neutralization import (
    NEUTRAL_PREFIX, attach_market_cap_exposure, neutralize_cross_section, neutralized_feature_columns,
)
from services.panel_store import PanelStore
from services.style_factors import attach_style_factors, load_history, configuration_id, production_report_is_current
from services.progress import interval_throttled, throttled
from services.label_processing import winsorize_training_labels
from services.strategy_scoring import fuse_prediction_percentiles
from services.rank_model import (
    SCORE_KIND_RANK, SCORE_KIND_RETURN, evaluate_ndcg, make_ranker, summarize_importances,
)
from services.validation_cache import ValidationCache, build_cache_key

TARGET_KIND = "excess_return_vs_csi300"
MINIMUM_STOCKS = 5
MINIMUM_COVERAGE = 0.8
MINIMUM_TRAIN_DATES = 80
MINIMUM_FEATURE_COVERAGE_DATES = 20
MINIMUM_OOS_DATES = 40
MINIMUM_RANK_IC = 0.02
MINIMUM_IC_T = 1.645
# 排序目标的 NDCG 必须显著高于随机排序基线，才认为学到的是截面区分能力。
MINIMUM_NDCG_RATIO = 1.05
# 中性化因子覆盖率低于该比例时退回截面Z分数因子，避免样本被残差缺失过度削减。
MODEL_NAMES = ("LightGBM", "XGBoost")
HORIZONS = settings.MODEL_HORIZONS
LOGGER = logging.getLogger("quant.ml")


def unavailable_model(name: str, horizon: int, status: str, objective: str | None = None, window: int | None = None) -> dict[str, Any]:
    """无截面或验证失败时只返回状态，禁止保留未经验证的预测值。"""
    return {
        "model": name, "horizon": horizon, "target_kind": TARGET_KIND,
        "model_version": settings.MODEL_VERSION, "validated": False, "status": status,
        "ndcg_protocol": settings.NDCG_PROTOCOL,
        "objective": objective, "score_kind": None, "training_window_days": window,
        "predicted_excess_return": None, "prediction_zscore": None, "prediction_percentile": None,
        "predicted_return": None, "predicted_price": None, "oos_metrics": None,
        "feature_importance": [], "samples": 0,
    }


def relative_signal(item: dict[str, Any]) -> float | None:
    """把模型输出折算为相对信号：排序模型只看截面相对位置，回归模型额外要求预期超额为正。"""
    zscore = item.get("prediction_zscore")
    if not item.get("validated") or zscore is None:
        return None
    if item.get("score_kind") == SCORE_KIND_RANK:
        percentile = item.get("prediction_percentile")
        favourable = percentile is not None and float(percentile) >= 0.5
    else:
        value = item.get("predicted_excess_return")
        favourable = value is not None and float(value) > 0
    return float(zscore) if favourable else min(0.0, float(zscore))


def build_market_panel(
    histories: dict[str, pd.DataFrame], benchmark: pd.DataFrame, exposures: pd.DataFrame | None = None,
    extra_factors: dict[str, Any] | None = None, store: PanelStore | None = None,
    progress: Callable[[str], None] | None = None,
    style_history: pd.DataFrame | None = None, style_options: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """构造可追溯的股票日截面，可选对市值与行业暴露中性化，保留标签结束日以便按日期清除重叠样本。

    传入 ``extra_factors`` 时，筹码分布、日内高频与 PIT 财务因子按覆盖率门控自动并入排序特征矩阵；
    不传时行为与既有版本完全一致。

    ``progress`` 只用于上报本函数的逐股阶段：全市场扫描下这一段（构造因子 + 并入筹码）要跑好几分钟，
    不接进度回调的话，页面会停在上一阶段的字幕上不动，看起来像卡死。
    """
    if benchmark is None or benchmark.empty:
        raise ValueError("沪深300历史不可用，不能训练超额收益模型。")
    if len(histories) < MINIMUM_STOCKS:
        raise ValueError(f"截面模型至少需要{MINIMUM_STOCKS}只股票的历史；当前{len(histories)}只。")
    frames = []
    should_report = throttled(len(histories))
    for index, (code, history) in enumerate(histories.items()):
        if progress and should_report(index):
            progress(f"正在构造截面因子（{index + 1}/{len(histories)}）")
        features = build_features(history, benchmark)
        labelled = add_forward_returns(features, HORIZONS, benchmark)
        labelled["code"] = str(code)
        columns = ["date", "code", "close", *FEATURE_COLUMNS]
        columns += [f"{prefix}_{horizon}" for horizon in HORIZONS for prefix in ("target", "forward_return", "label_end")]
        columns += [f"{prefix}_{horizon}" for horizon in HORIZONS for prefix in (
            "research_target", "research_forward_return", "research_label_end",
        )]
        frames.append(labelled[columns])
    panel = pd.concat(frames, ignore_index=True).sort_values(["date", "code"]).reset_index(drop=True)
    # 新增因子先装配、财务健康否决先生效，之后才做截面标准化：被否决的标的不能参与自己所在截面的
    # 均值与标准差，否则「打分校准的股票池」与「实际可交易的股票池」会不一致。
    extra_report: dict[str, Any] = {"status": "未提供新增因子", "families": {}}
    if extra_factors:
        panel, extra_report = attach_extra_factors(panel, extra_factors, store=store, progress=progress)
    panel, veto_report = apply_health_veto(panel)
    panel = normalize_cross_section(panel, FEATURE_COLUMNS, MINIMUM_STOCKS, MINIMUM_COVERAGE)
    panel.attrs["extra_factor_families"] = sorted((extra_report.get("families") or {}).keys())
    model_features = [f"xs_{column}" for column in FEATURE_COLUMNS]
    neutralization: dict[str, Any] = {"status": "未启用市值与行业中性化，模型使用截面Z分数因子"}
    exposure_report: dict[str, Any] = {"status": "未接入市值与行业暴露"}
    neutralization_blocked = False
    if exposures is not None and not exposures.empty:
        panel, exposure_report = attach_market_cap_exposure(panel, exposures)
        if (
            exposure_report.get("point_in_time")
            and exposure_report.get("industry_available")
            and exposure_report.get("market_cap_coverage", 0.0) >= MINIMUM_COVERAGE
            and exposure_report.get("industry_coverage", 0.0) >= MINIMUM_COVERAGE
        ):
            panel, neutralization = neutralize_cross_section(panel, FEATURE_COLUMNS)
            model_features = neutralized_feature_columns(panel, FEATURE_COLUMNS)
            coverage = float(panel[model_features].notna().all(axis=1).mean()) if model_features else 0.0
            neutralization = {
                **neutralization, "model_features": len(model_features), "coverage": coverage,
                "selection_policy": "训练折内按历史覆盖率筛选，不按全样本覆盖率切换",
            }
        else:
            neutralization_blocked = True
            model_features = []
            neutralization = {
                "status": (
                    "点时市值或历史行业暴露覆盖不足，已阻止该中性化配置训练；"
                    f"市值覆盖 {exposure_report.get('market_cap_coverage', 0.0):.1%}，"
                    f"行业覆盖 {exposure_report.get('industry_coverage', 0.0):.1%}"
                ),
                "exposure": exposure_report,
            }
    family_reports = extra_report.get("families") or {}
    candidate_extra_columns = [
        column
        for family, columns in EXTRA_FACTOR_FAMILIES.items()
        if family in family_reports and all(column in panel for column in columns)
        for column in columns
    ]
    extra_report = {**extra_report, "model_candidate_columns": candidate_extra_columns}
    if candidate_extra_columns:
        features: list[str] = []
        coverage = 0.0
        neutralized = bool(model_features) and model_features[0].startswith(NEUTRAL_PREFIX)
        # 基础因子做过中性化时新增因子也必须一起做，否则模型会绕过中性化，
        # 顺着新增因子把市值与行业暴露重新学回来。
        if neutralized and not neutralization_blocked:
            panel, extra_neutralization = neutralize_cross_section(panel, candidate_extra_columns)
            candidates = neutralized_feature_columns(panel, candidate_extra_columns)
            coverage = float(panel[candidates].notna().all(axis=1).mean()) if candidates else 0.0
            if candidates:
                features = candidates
            extra_report = {**extra_report, "neutralization": {**extra_neutralization, "coverage": coverage}}
        if not features and not neutralization_blocked:
            # 树模型对特征的单调变换不敏感，这里统一做截面Z分数只为与基础因子保持同一口径；
            # 缺少可靠点时暴露时才使用截面Z分数，原因写进报告便于核对。
            panel = normalize_cross_section(panel, candidate_extra_columns, MINIMUM_STOCKS, MINIMUM_COVERAGE)
            features = [f"xs_{column}" for column in candidate_extra_columns if f"xs_{column}" in panel]
            if neutralized:
                extra_report["neutralization"] = {
                    **extra_report["neutralization"],
                    "status": f"新增因子中性化整体覆盖率 {coverage:.0%}；保留候选列并在训练折内筛选",
                }
        model_features = [*model_features, *features]
        extra_report = {**extra_report, "model_features": len(features)}
    if not settings.ENABLE_BASE_FEATURES:
        model_features = []
    panel.attrs["model_feature_columns"] = model_features
    if style_history is None and extra_factors is not None:
        style_history = extra_factors.get("valuation")
    if style_history is None and store is not None and (settings.ENABLE_SIZE_FEATURES or settings.ENABLE_VALUE_FEATURES):
        style_history = load_history(store, list(histories))
    panel, style_report = attach_style_factors(panel, style_history, **{
        "neutralization_enabled": exposures is not None and not exposures.empty, **(style_options or {}),
    })
    model_features = [*model_features, *style_report["model_feature_columns"]]
    if neutralization_blocked:
        model_features = []
    panel.attrs["style_factors"] = style_report
    # 截面覆盖只针对本次成功取得历史的股票池，不能冒充历史全市场成分。
    panel.attrs["universe_codes"] = sorted(histories)
    panel.attrs["exposures"] = exposure_report
    panel.attrs["neutralization"] = neutralization
    panel.attrs["extra_factors"] = extra_report
    panel.attrs["health_veto"] = veto_report
    panel.attrs["model_feature_columns"] = model_features
    return panel


def normalize_predictions(frame: pd.DataFrame, prediction_column: str = "prediction") -> pd.DataFrame:
    """每个日期独立去均值和标准化；相同预测或不足截面拒绝生成排序。"""
    result = frame.copy()
    values = pd.to_numeric(result[prediction_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    groups = values.groupby(result["date"])
    count = groups.transform("count")
    deviation = groups.transform("std", ddof=0)
    valid = count.ge(MINIMUM_STOCKS) & deviation.gt(1e-12)
    result["prediction_zscore"] = ((values - groups.transform("mean")) / deviation).where(valid)
    result["prediction_percentile"] = ((groups.rank(method="average") - 1) / (count - 1)).where(valid)
    result["model_rank"] = groups.rank(method="average", ascending=False).where(valid)
    return result


def fuse_normalized_predictions(
    prediction_frames: dict[str, pd.DataFrame], model_names: Sequence[str] = MODEL_NAMES,
) -> pd.DataFrame:
    """按生产选股口径外连接各模型百分位，并对可用模型取均值。"""
    return fuse_prediction_percentiles(prediction_frames, model_names).dropna(
        subset=["score"]
    ).sort_values(["date", "code"]).reset_index(drop=True)


def evaluation_gates(metrics: dict[str, Any]) -> dict[str, Any]:
    """逐道门槛的通过情况与第一道未通过的门槛，供评估与监控共用一份判定。

    只读取 ``evaluate_oos`` 已经算好的标量，不重算任何指标，因此抽出来不会改变任何数值。
    之所以要共用：页面上显示的「这轮为什么没通过」必须与当时真正拦下模型的理由逐字一致，
    两处各写一份判定迟早会各改一半，用户看到的失败原因就成了假的。

    判定顺序与原始实现逐字保持一致：样本天数 → Rank IC → HAC t → Top 超额 → Top-Bottom
    → 排序目标看 NDCG、回归目标看 MAE。``failed_gate`` 取其中第一道未通过的。
    """
    ranking = metrics.get("score_kind") == SCORE_KIND_RANK
    ic_mean, ic_t = metrics.get("rank_ic"), metrics.get("rank_ic_hac_t")
    top, universe = metrics.get("top_excess_return"), metrics.get("universe_excess_return")
    spread = metrics.get("top_bottom_spread")
    ndcg_ratio = metrics.get("ndcg_ratio")
    ndcg_at_20_ratio = metrics.get("ndcg_at_20_ratio")
    top_20_net_excess = metrics.get("top_20_excess_net_return")
    universe_net_excess = metrics.get("universe_net_excess_return")
    mae, baseline_mae = metrics.get("mae"), metrics.get("baseline_mae")
    # 每一项都显式判 None：监控读回的是落库/落盘后再取出的记录，
    # 未生成样本外指标时这些字段确实是空值，不能让 None 参与比较。
    gates = {
        "dates": int(metrics.get("dates") or 0) >= MINIMUM_OOS_DATES,
        "rank_ic": ic_mean is not None and float(ic_mean) > MINIMUM_RANK_IC,
        "ic_t": ic_t is not None and float(ic_t) > MINIMUM_IC_T,
        "top_excess": top is not None and universe is not None and float(top) > max(0.0, float(universe)),
        "top_bottom": spread is not None and float(spread) > 0,
    }
    if ranking:
        gates["ndcg_at_20"] = (
            ndcg_at_20_ratio is not None
            and float(ndcg_at_20_ratio) > MINIMUM_NDCG_RATIO
            and int(metrics.get("ndcg_at_20_days") or 0) >= MINIMUM_OOS_DATES
        )
        gates["top_20_net_excess"] = (
            top_20_net_excess is not None
            and universe_net_excess is not None
            and float(top_20_net_excess) > max(0.0, float(universe_net_excess))
            and int(metrics.get("top_20_dates") or 0) >= MINIMUM_OOS_DATES
        )
    else:
        gates["mae"] = mae is not None and baseline_mae is not None and float(mae) < float(baseline_mae)
    failed = next((name for name, ok in gates.items() if not ok), None)
    return {"passed": failed is None, "gates": gates, "failed_gate": failed}


def evaluate_oos(
    frame: pd.DataFrame, horizon: int, total_stocks: int | None = None, score_kind: str = SCORE_KIND_RETURN
) -> dict[str, Any]:
    """计算每日Rank IC、HAC显著性、Top/Bottom收益与排序质量基线。"""
    from statsmodels.api import OLS
    from services.backtest import cost_rates

    evaluation_columns = ["date", "code", "prediction", "actual", "baseline", "absolute_return"]
    clean = pd.DataFrame(
        {column: column_values_without_attrs(frame, column) for column in evaluation_columns},
        index=frame.index,
        copy=False,
    ).replace([np.inf, -np.inf], np.nan).dropna(
        subset=["prediction", "actual", "baseline", "absolute_return"]
    )
    required = max(MINIMUM_STOCKS, math.ceil((total_stocks or MINIMUM_STOCKS) * MINIMUM_COVERAGE))
    buy_cost, sell_cost = cost_rates()
    rows = []
    eligible_indices = []
    top_20_members: list[set[str]] = []
    for date, group in clean.groupby("date", sort=True):
        if group["code"].nunique() < required or group["prediction"].nunique() < 2 or group["actual"].nunique() < 2:
            continue
        ordered = group.sort_values("prediction", kind="stable")
        size = max(1, len(group) // 5)
        bottom, top = ordered.head(size), ordered.tail(size)
        top_20 = ordered.tail(20) if len(ordered) >= 20 else ordered.iloc[0:0]
        benchmark_return = group["absolute_return"] - group["actual"]
        net_return = (1 + group["absolute_return"]) * (1 - sell_cost) / (1 + buy_cost) - 1
        if len(top_20) == 20:
            top_20_net_return = (
                (1 + top_20["absolute_return"]) * (1 - sell_cost) / (1 + buy_cost) - 1
            )
            top_20_benchmark_return = (
                top_20["absolute_return"] - top_20["actual"]
            )
            top_20_members.append(set(top_20["code"].astype(str)))
            top_20_net_excess = float(top_20_net_return.mean() - top_20_benchmark_return.mean())
            universe_net_excess = float(net_return.mean() - benchmark_return.mean())
            top_20_net_mean = float(top_20_net_return.mean())
        else:
            top_20_net_excess = None
            universe_net_excess = None
            top_20_net_mean = None
        ic = float(group["prediction"].rank().corr(group["actual"].rank()))
        if not np.isfinite(ic):
            continue
        rows.append({
            "date": str(pd.Timestamp(date).date()), "rank_ic": ic,
            "top_excess_return": float(top["actual"].mean()), "bottom_excess_return": float(bottom["actual"].mean()),
            "top_return": float(top["absolute_return"].mean()), "bottom_return": float(bottom["absolute_return"].mean()),
            "universe_excess_return": float(group["actual"].mean()), "stocks": len(group),
            "top_20_net_return": top_20_net_mean,
            "top_20_excess_net_return": top_20_net_excess,
            "universe_net_excess_return": universe_net_excess,
        })
        eligible_indices.extend(group.index.tolist())
    evaluated = clean.loc[eligible_indices]
    ranking = score_kind == SCORE_KIND_RANK
    metrics = {"dates": len(rows), "oos_samples": len(evaluated), "daily_groups": rows, "target_kind": TARGET_KIND,
               "score_kind": score_kind, "ndcg": None, "ndcg_random_baseline": None, "ndcg_ratio": None,
               "ndcg_at_20": None, "ndcg_at_20_random_baseline": None, "ndcg_at_20_ratio": None,
               "ndcg_at_20_days": 0, "top_20_net_return": None,
               "top_20_excess_net_return": None, "universe_net_excess_return": None,
               "top_20_dates": 0, "top_20_selection_turnover": None,
               "ndcg_protocol": settings.NDCG_PROTOCOL,
               "rank_ic": None, "rank_ic_hac_t": None, "top_excess_return": None, "bottom_excess_return": None,
               "top_return": None, "bottom_return": None, "top_bottom_spread": None, "mae": None, "baseline_mae": None,
               "validated": False, "validation_rule": (
                   "排序目标：每日Rank IC>0.02且HAC t>1.645；NDCG@20高于随机排序基线5%以上；"
                   "Top 20 扣费超额为正并优于池均值；Top-Bottom为正；至少40个样本外交易日"
                   if ranking else
                   "回归目标：MAE优于训练中位数基线；每日Rank IC>0.02且HAC t>1.645；Top超额为正并优于池均值；"
                   "Top-Bottom为正；至少40个样本外交易日")}
    if not rows:
        return metrics
    daily = pd.DataFrame(rows)
    ic_mean = float(daily["rank_ic"].mean())
    estimate = OLS(daily["rank_ic"].to_numpy(), np.ones((len(daily), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": min(horizon, len(daily) - 1)})
    deviation = float(estimate.bse[0])
    ic_t = ic_mean / deviation if np.isfinite(deviation) and deviation > 1e-12 else None
    # 排序分数是无量纲的截面名次分，绝对误差没有意义，只保留相对排序质量的 NDCG 检验。
    mae = None if ranking else float((evaluated["prediction"] - evaluated["actual"]).abs().mean())
    baseline_mae = None if ranking else float((evaluated["baseline"] - evaluated["actual"]).abs().mean())
    quality = evaluate_ndcg(evaluated)
    quality_at_20 = evaluate_ndcg(evaluated, k=20)
    ndcg, ndcg_baseline = quality["ndcg"], quality["ndcg_random_baseline"]
    ndcg_ratio = ndcg / ndcg_baseline if ndcg is not None and ndcg_baseline else None
    ndcg_at_20_ratio = (
        quality_at_20["ndcg"] / quality_at_20["ndcg_random_baseline"]
        if quality_at_20["ndcg"] is not None and quality_at_20["ndcg_random_baseline"]
        else None
    )
    top = float(daily["top_excess_return"].mean())
    bottom = float(daily["bottom_excess_return"].mean())
    spread = top - bottom
    universe = float(daily["universe_excess_return"].mean())
    top_20_daily = daily.dropna(subset=["top_20_net_return", "top_20_excess_net_return", "universe_net_excess_return"])
    selection_turnover = [
        1 - len(current & previous) / 20
        for previous, current in zip(top_20_members, top_20_members[1:])
    ]
    metrics.update(rank_ic=ic_mean, rank_ic_hac_t=float(ic_t) if ic_t is not None else None,
                   ic_test_status="有效" if ic_t is not None else "IC方差退化，拒绝推断显著性", mae=mae, baseline_mae=baseline_mae,
                   ndcg=ndcg, ndcg_random_baseline=ndcg_baseline, ndcg_days=quality["ndcg_days"],
                   ndcg_ratio=ndcg_ratio,
                   ndcg_at_20=quality_at_20["ndcg"],
                   ndcg_at_20_random_baseline=quality_at_20["ndcg_random_baseline"],
                   ndcg_at_20_days=quality_at_20["ndcg_days"],
                   ndcg_at_20_ratio=ndcg_at_20_ratio,
                   top_20_net_return=(float(top_20_daily["top_20_net_return"].mean()) if not top_20_daily.empty else None),
                   top_20_excess_net_return=(
                       float(top_20_daily["top_20_excess_net_return"].mean()) if not top_20_daily.empty else None
                   ),
                   universe_net_excess_return=(
                       float(top_20_daily["universe_net_excess_return"].mean()) if not top_20_daily.empty else None
                   ),
                   top_20_dates=int(len(top_20_daily)),
                   top_20_selection_turnover=(float(np.mean(selection_turnover)) if selection_turnover else None),
                   top_excess_return=top, bottom_excess_return=bottom, top_bottom_spread=spread,
                   # 池均值超额原本只用在内联判定里、用完即弃；判定抽成 evaluation_gates 后
                   # 它必须随指标一起返回，落库后页面才能复核「Top 超额是否真的优于池均值」。
                   universe_excess_return=universe,
                   top_return=float(daily["top_return"].mean()), bottom_return=float(daily["bottom_return"].mean()))
    gates = evaluation_gates(metrics)
    metrics.update(validated=bool(gates["passed"]), gate_detail=gates["gates"], failed_gate=gates["failed_gate"])
    return metrics


def _selection_key(result: dict[str, Any]) -> tuple[int, float]:
    """目标函数对比时的择优口径：先看是否通过验证，再看样本外Rank IC。"""
    metrics = result.get("oos_metrics") or {}
    rank_ic = metrics.get("rank_ic")
    return (1 if result.get("validated") else 0, float(rank_ic) if rank_ic is not None else -1.0)


def _select_fold_features(
    training: pd.DataFrame, feature_pool: Sequence[str], required_stocks: int,
) -> list[str]:
    """只根据当前训练窗口的截面覆盖选择特征。"""
    available_columns = [column for column in feature_pool if column in training.columns]
    if training.empty or not available_columns:
        return []
    coverage = training.groupby("date", sort=True)[available_columns].count()
    qualified_dates = coverage.ge(required_stocks).sum(axis=0)
    return [
        column for column in available_columns
        if int(qualified_dates[column]) >= MINIMUM_FEATURE_COVERAGE_DATES
    ]


def _prepare_validation_data(
    panel: pd.DataFrame, horizon: int, cutoff: pd.Timestamp | None, window: int,
) -> dict[str, Any]:
    """为单个预测周期准备窄列评估输入与可复用的日期边界。"""
    started = perf_counter()
    target, endpoint = f"target_{horizon}", f"label_end_{horizon}"
    forward_return = f"forward_return_{horizon}"
    feature_pool = list(panel.attrs.get("model_feature_columns", [f"xs_{column}" for column in FEATURE_COLUMNS]))
    feature_pool = list(dict.fromkeys(column for column in feature_pool if column in panel.columns))
    required_columns = ["date", "code", *feature_pool, target, endpoint, forward_return]
    # 通过列数组构造临时表，避免复制原始面板上与当前周期无关的 attrs。
    frame = pd.DataFrame(
        {column: column_values_without_attrs(panel, column) for column in required_columns},
        index=panel.index,
        copy=False,
    )
    if cutoff is not None:
        frame = frame.loc[frame["date"].lt(cutoff)]
    frame_preparation_seconds = perf_counter() - started

    required = max(MINIMUM_STOCKS, math.ceil(frame["code"].nunique() * MINIMUM_COVERAGE))
    # 保留完整候选池，具体入模列在每个训练折内单独确定。
    features = feature_pool
    feature_calibration_seconds = 0.0
    fingerprint_started = perf_counter()
    fingerprint_columns = ["date", "code", *features, target, endpoint, forward_return]
    row_hashes = pd.util.hash_pandas_object(
        frame.loc[:, fingerprint_columns], index=True, categorize=True,
    ).to_numpy(dtype=np.uint64, copy=False)
    input_fingerprint = hashlib.sha256(row_hashes.tobytes()).hexdigest()
    fingerprint_seconds = perf_counter() - fingerprint_started
    sample_filter_started = perf_counter()
    filtered = frame.dropna(subset=[target, endpoint])
    if cutoff is not None:
        filtered = filtered.loc[filtered[endpoint].lt(cutoff)]
    # 后续拟合和预测只需实际入模列及本周期标签，不保留其他周期或诊断列。
    labelled = filtered.loc[:, ["date", "code", *features, target, endpoint, forward_return]]
    labelled_dates = pd.to_datetime(labelled["date"]).to_numpy(dtype="datetime64[ns]")
    dates = np.unique(labelled_dates)
    sample_filter_seconds = perf_counter() - sample_filter_started

    started = perf_counter()
    folds: list[dict[str, int]] = []
    if len(dates) >= 240:
        from sklearn.model_selection import TimeSeriesSplit

        first_test = int(next(TimeSeriesSplit(n_splits=3).split(dates))[1][0])
        for block_start in range(first_test, len(dates), int(settings.OOS_RETRAIN_DAYS)):
            folds.append({
                "train_start_index": max(0, block_start - window) if window > 0 else 0,
                "train_stop_index": block_start,
                "test_start_index": block_start,
                "test_stop_index": min(block_start + int(settings.OOS_RETRAIN_DAYS), len(dates)),
            })
    date_positions = np.searchsorted(dates, labelled_dates) if len(dates) else np.zeros(0, dtype=int)
    sorted_by_date = bool(labelled["date"].is_monotonic_increasing)
    fold_preparation_seconds = perf_counter() - started
    return {
        "labelled": labelled,
        "features": features,
        "dates": dates,
        "folds": folds,
        "date_positions": date_positions,
        "sorted_by_date": sorted_by_date,
        "target": target,
        "endpoint": endpoint,
        "forward_return": forward_return,
        "required_stocks": required,
        "total_stocks": int(frame["code"].nunique()),
        "input_fingerprint": input_fingerprint,
        "feature_configuration_id": (panel.attrs.get("style_factors") or {}).get("configuration_id"),
        "cutoff": str(pd.Timestamp(cutoff).isoformat()) if cutoff is not None else None,
        "preparation_seconds": {
            "frame_preparation": frame_preparation_seconds,
            "feature_calibration": feature_calibration_seconds,
            "input_fingerprint": fingerprint_seconds,
            "sample_filter": sample_filter_seconds,
            "fold_preparation": fold_preparation_seconds,
        },
    }


def _fold_row_positions(prepared: dict[str, Any], fold: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """按原始行序生成一折训练与测试位置，兼容未排序的外部面板。"""
    dates = prepared["dates"]
    labelled = prepared["labelled"]
    train_start = fold["train_start_index"]
    train_stop = fold["train_stop_index"]
    test_start_index = fold["test_start_index"]
    test_stop_index = fold["test_stop_index"]
    test_start = pd.Timestamp(dates[test_start_index])
    date_positions = prepared["date_positions"]
    if prepared["sorted_by_date"]:
        labelled_dates = pd.to_datetime(labelled["date"]).to_numpy(dtype="datetime64[ns]")
        train_start_date = dates[train_start] if train_start < train_stop else dates[train_stop]
        train_start_row = int(np.searchsorted(labelled_dates, train_start_date, side="left"))
        train_stop_row = int(np.searchsorted(labelled_dates, dates[test_start_index], side="left"))
        candidates = np.arange(train_start_row, train_stop_row, dtype=np.int64)
        mature = labelled[prepared["endpoint"]].iloc[candidates].lt(test_start).to_numpy()
        train_positions = candidates[mature]
        test_start_row = int(np.searchsorted(labelled_dates, dates[test_start_index], side="left"))
        test_end_row = int(np.searchsorted(labelled_dates, dates[test_stop_index - 1], side="right"))
        test_positions = np.arange(test_start_row, test_end_row, dtype=np.int64)
        return train_positions, test_positions

    train_mask = (date_positions >= train_start) & (date_positions < train_stop)
    train_mask &= labelled[prepared["endpoint"]].lt(test_start).to_numpy()
    test_mask = (date_positions >= test_start_index) & (date_positions < test_stop_index)
    return np.flatnonzero(train_mask), np.flatnonzero(test_mask)


def _runtime_environment() -> dict[str, Any]:
    """记录可复现实测所需的运行环境版本。"""
    from importlib.metadata import PackageNotFoundError, version

    packages = {}
    for package in ("lightgbm", "xgboost", "scikit-learn", "statsmodels", "numpy", "pandas"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": packages}


def _memory_metrics() -> dict[str, Any]:
    """读取操作系统记录的进程峰值常驻内存；不可用时明确返回缺失状态。"""
    try:
        if os.name == "nt":
            import ctypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            from ctypes import wintypes

            kernel32 = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi
            kernel32.GetCurrentProcess.argtypes = []
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            process = kernel32.GetCurrentProcess()
            ok = psapi.GetProcessMemoryInfo(
                process, ctypes.byref(counters), counters.cb,
            )
            if ok:
                return {
                    "peak_memory_mb": counters.PeakWorkingSetSize / (1024 ** 2),
                    "memory_status": "操作系统进程生命周期峰值工作集",
                }
        else:
            import resource

            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            divisor = 1024 ** 2 if sys.platform == "darwin" else 1024
            return {
                "peak_memory_mb": float(peak) / divisor,
                "memory_status": "操作系统进程生命周期峰值 RSS",
            }
    except (AttributeError, ImportError, OSError, TypeError, ValueError):
        pass
    return {"peak_memory_mb": None, "memory_status": "当前环境不支持进程峰值内存读取"}


def _execution_summary(
    name: str, horizon: int, objective: str, prepared: dict[str, Any], started: float,
    stage_seconds: dict[str, float], fold_details: list[dict[str, Any]], cache: dict[str, Any],
    model_parameters: Any = None,
) -> dict[str, Any]:
    """汇总当前模型、目标函数和验证段的工作量、耗时与运行环境。"""
    dates = prepared["dates"]
    environment = _runtime_environment()
    validation_identity = build_cache_key({
        "model_version": settings.MODEL_VERSION,
        "validation_protocol": settings.VALIDATION_PROTOCOL_VERSION,
        "model": name,
        "horizon": int(horizon),
        "objective": objective,
        "input_fingerprint": prepared["input_fingerprint"],
        "feature_configuration_id": prepared["feature_configuration_id"],
        "features": prepared["features"],
        "model_parameters": model_parameters,
        "cutoff": prepared["cutoff"],
        "folds": [
            {
                "cache_key": detail.get("cache_key"),
                "validation_segment": detail.get("validation_segment"),
                "test_start": detail.get("test_start"),
                "test_end": detail.get("test_end"),
            }
            for detail in fold_details if "validation_segment" in detail
        ],
        "environment": environment,
    })
    cache_sources = [
        {"validation_segment": detail.get("validation_segment"), "source": detail.get("cache_source")}
        for detail in fold_details if detail.get("cache_source")
    ]
    executed_sources = [item["source"] for item in cache_sources if item["source"] != "not_run"]
    return {
        "model": name,
        "horizon": horizon,
        "objective": objective,
        "target_function": objective,
        "stocks": prepared["total_stocks"],
        "features": len(prepared["features"]),
        "effective_samples": len(prepared["labelled"]),
        "training_dates": len(dates),
        "training_date_start": str(pd.Timestamp(dates[0]).date()) if len(dates) else None,
        "training_date_end": str(pd.Timestamp(dates[-1]).date()) if len(dates) else None,
        "validation_date_start": fold_details[0].get("test_start") if fold_details else None,
        "validation_date_end": fold_details[-1].get("test_end") if fold_details else None,
        "threads": int(settings.TRAINING_N_JOBS),
        "preparation_seconds": prepared["preparation_seconds"],
        "stage_seconds": stage_seconds,
        "elapsed_seconds": perf_counter() - started,
        **_memory_metrics(),
        "environment": environment,
        "validation_identity": validation_identity if any(
            source != "not_run" for source in executed_sources
        ) else None,
        "validation_cache_sources": cache_sources,
        "validation_cache_replay": bool(executed_sources) and all(source == "hot" for source in executed_sources),
        "cache": cache,
        "folds": fold_details,
    }


def _validation_cache_key(
    name: str, horizon: int, objective: str, window: int, fold_number: int,
    fold: dict[str, int], prepared: dict[str, Any], training_samples: int,
    validation_samples: int, environment: dict[str, Any], model_parameters: Any = None,
) -> str:
    """将缓存绑定到完整数据、模型处理协议、依赖版本和验证日期。"""
    dates = prepared["dates"]
    train_start = fold["train_start_index"]
    train_stop = fold["train_stop_index"]
    test_start = fold["test_start_index"]
    test_stop = fold["test_stop_index"]
    identity = {
        "validation_protocol": settings.VALIDATION_PROTOCOL_VERSION,
        "model_version": settings.MODEL_VERSION,
        "model": name,
        "objective": objective,
        "horizon": int(horizon),
        "target": prepared["target"],
        "ndcg_protocol": settings.NDCG_PROTOCOL,
        "feature_protocol": settings.STYLE_FEATURE_PROTOCOL,
        "feature_configuration_id": prepared["feature_configuration_id"],
        "features": prepared["features"],
        "input_fingerprint": prepared["input_fingerprint"],
        "model_parameters": model_parameters,
        "cutoff": prepared["cutoff"],
        "parameters": {
            "training_window_days": int(window),
            "retrain_interval_days": int(settings.OOS_RETRAIN_DAYS),
            "threads": int(settings.TRAINING_N_JOBS),
            "seed": 42,
            "rank_grade_count": int(settings.RANK_GRADE_COUNT),
            "ndcg_top_fraction": float(settings.NDCG_TOP_FRACTION),
            "regression_loss": settings.REGRESSION_LOSS,
            "label_winsor_upper_quantile": float(settings.LABEL_WINSOR_UPPER_QUANTILE),
        },
        "fold": {
            "number": int(fold_number),
            "train_start": str(pd.Timestamp(dates[train_start]).date()),
            "train_end": str(pd.Timestamp(dates[train_stop - 1]).date()),
            "test_start": str(pd.Timestamp(dates[test_start]).date()),
            "test_end": str(pd.Timestamp(dates[test_stop - 1]).date()),
            "training_samples": int(training_samples),
            "validation_samples": int(validation_samples),
        },
        "environment": environment,
    }
    return build_cache_key(identity)


def fit_panel_model(
    panel: pd.DataFrame, name: str, horizon: int, model_factory: Callable | None = None,
    cutoff: pd.Timestamp | None = None, window: int | None = None, objective: str | None = None,
    progress: Callable[[str], None] | None = None, _prepared: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """按完整交易日划分滚动样本外折，训练窗口只保留最近N个交易日，严格清除标签跨入测试期的样本。"""
    requested = objective or settings.DEFAULT_OBJECTIVE_MODE
    if requested not in settings.RANK_OBJECTIVE_MODES:
        raise ValueError(f"目标函数仅支持 {'、'.join(settings.RANK_OBJECTIVE_MODES)}；当前为 {requested}。")
    window = int(settings.TRAINING_WINDOW_DAYS if window is None else window)
    validation_block = int(settings.OOS_RETRAIN_DAYS)
    if validation_block < 1:
        raise ValueError("样本外重训间隔必须为正整数。")
    started = perf_counter()
    if progress is not None:
        original_progress = progress
        should_report_progress = interval_throttled()

        def throttled_progress(message: str) -> None:
            if should_report_progress(force=" / " in message):
                original_progress(message)

        progress = throttled_progress
    prepared = _prepared or _prepare_validation_data(panel, horizon, cutoff, window)
    labelled = prepared["labelled"]
    feature_pool = prepared["features"]
    dates = prepared["dates"]
    target, endpoint = prepared["target"], prepared["endpoint"]
    forward_return = prepared["forward_return"]
    if progress:
        progress(f"按交易日验证 {name} · {horizon}日 · 数据准备完成 · 已用 {perf_counter() - started:.1f} 秒")
    if not feature_pool:
        result = unavailable_model(name, horizon, "截面特征样本不足，未生成预测", objective=requested, window=window)
        result["execution"] = _execution_summary(
            name, horizon, requested, prepared, started, {}, [],
            {"enabled": False, "hits": 0, "misses": 0, "reason": "没有候选特征"},
        )
        return result
    cache_enabled = bool(settings.VALIDATION_CACHE_ENABLED and model_factory is None)
    validation_cache = ValidationCache(
        settings.VALIDATION_CACHE_DIR,
        settings.VALIDATION_CACHE_MAX_BYTES,
        settings.VALIDATION_CACHE_TTL_DAYS,
    ) if cache_enabled else None

    def _estimator(candidate: str) -> Any:
        # 显式注入的工厂优先，保持测试与自定义估计器的行为完全可控。
        if model_factory is not None:
            return model_factory(name)
        if candidate == "regression":
            from services.stock_scoring import _make_model

            return _make_model(name)
        return make_ranker(name)

    def _fit(
        estimator: Any, sample: pd.DataFrame, candidate: str, selected_features: Sequence[str],
    ) -> tuple[Any, dict[str, Any]]:
        labels = sample[target]
        processing = {"enabled": False, "reason": "排序目标保留原始标签，避免截尾产生并列名次"}
        if candidate == "regression":
            labels, processing = winsorize_training_labels(labels, sample["date"], minimum_stocks=MINIMUM_STOCKS)
        if hasattr(estimator, "fit_cross_section"):
            estimator.fit_cross_section(sample[list(selected_features)], labels, sample["date"])
        else:
            estimator.fit(sample[list(selected_features)], labels)
        return estimator, processing

    def _estimator_parameter_identity(candidate: str) -> dict[str, Any]:
        estimator = _estimator(candidate)
        identity: dict[str, Any] = {
            "class": f"{type(estimator).__module__}.{type(estimator).__qualname__}",
        }
        if callable(getattr(estimator, "_build", None)):
            built = estimator._build()
            getter = getattr(built, "get_params", None)
            identity["wrapper"] = {
                "name": getattr(estimator, "name", None),
                "grade_count": getattr(estimator, "grade_count", None),
                "seed": getattr(estimator, "seed", None),
            }
            identity["parameters"] = getter(deep=True) if callable(getter) else vars(built)
            return identity
        getter = getattr(estimator, "get_params", None)
        identity["parameters"] = getter(deep=True) if callable(getter) else vars(estimator)
        return identity

    def _run(candidate: str) -> dict[str, Any]:
        candidate_started = perf_counter()
        result = unavailable_model(name, horizon, "截面样本不足，未生成预测", objective=candidate, window=window)
        result["samples"] = len(labelled)
        result["training_dates"] = len(dates)
        result["score_kind"] = SCORE_KIND_RETURN if candidate == "regression" else SCORE_KIND_RANK
        stage_seconds = {key: 0.0 for key in (
            "fold_preparation", "native_fit", "prediction", "metrics", "final_fit", "cache_lookup", "cache_write",
        )}
        fold_details = []
        oos: list[pd.DataFrame] = []
        cache_summary = {
            "enabled": cache_enabled,
            "hits": 0,
            "misses": 0,
            "input_fingerprint": prepared["input_fingerprint"],
            "fingerprint_seconds": prepared["preparation_seconds"]["input_fingerprint"],
        }
        if len(dates) < 240 or labelled["code"].nunique() < MINIMUM_STOCKS:
            result["execution"] = _execution_summary(
                name, horizon, candidate, prepared, candidate_started, stage_seconds, fold_details, cache_summary,
            )
            return result
        model_parameters: Any = None
        try:
            if model_factory is None:
                model_parameters = _estimator_parameter_identity(candidate)
            else:
                model_parameters = {
                    "factory": f"{getattr(model_factory, '__module__', type(model_factory).__module__)}."
                               f"{getattr(model_factory, '__qualname__', type(model_factory).__qualname__)}",
                }
            for fold_number, fold_spec in enumerate(prepared["folds"], start=1):
                fold_started = perf_counter()
                train_positions, test_positions = _fold_row_positions(prepared, fold_spec)
                fold_prep_seconds = perf_counter() - fold_started
                stage_seconds["fold_preparation"] += fold_prep_seconds
                train_window = labelled.iloc[train_positions]
                training_date_count = int(train_window["date"].nunique())
                if training_date_count < MINIMUM_TRAIN_DATES:
                    fold_details.append({
                        "validation_segment": fold_number,
                        "test_start": str(pd.Timestamp(dates[fold_spec["test_start_index"]]).date()),
                        "test_end": str(pd.Timestamp(dates[fold_spec["test_stop_index"] - 1]).date()),
                        "training_samples": len(train_positions), "validation_samples": len(test_positions),
                        "training_dates": training_date_count, "threads": int(settings.TRAINING_N_JOBS),
                        "status": "训练日期不足，跳过该段", "preparation_seconds": fold_prep_seconds,
                        "fit_seconds": 0.0, "prediction_seconds": 0.0, "cache_source": "not_run",
                    })
                    continue
                fold_features = _select_fold_features(
                    train_window, feature_pool, prepared["required_stocks"],
                )
                if not fold_features:
                    fold_details.append({
                        "validation_segment": fold_number,
                        "test_start": str(pd.Timestamp(dates[fold_spec["test_start_index"]]).date()),
                        "test_end": str(pd.Timestamp(dates[fold_spec["test_stop_index"] - 1]).date()),
                        "training_samples": len(train_window), "validation_samples": len(test_positions),
                        "training_dates": training_date_count, "features": 0,
                        "feature_columns": [], "threads": int(settings.TRAINING_N_JOBS),
                        "status": "训练窗口内无达到覆盖门槛的特征，跳过该段",
                        "preparation_seconds": fold_prep_seconds, "fit_seconds": 0.0,
                        "prediction_seconds": 0.0, "cache_source": "not_run",
                    })
                    continue
                train = train_window.loc[train_window[fold_features].notna().any(axis=1)]
                training_date_count = int(train["date"].nunique())
                if training_date_count < MINIMUM_TRAIN_DATES:
                    fold_details.append({
                        "validation_segment": fold_number,
                        "test_start": str(pd.Timestamp(dates[fold_spec["test_start_index"]]).date()),
                        "test_end": str(pd.Timestamp(dates[fold_spec["test_stop_index"] - 1]).date()),
                        "training_samples": len(train), "validation_samples": len(test_positions),
                        "training_dates": training_date_count, "features": len(fold_features),
                        "feature_columns": fold_features, "threads": int(settings.TRAINING_N_JOBS),
                        "status": "特征覆盖筛选后训练日期不足，跳过该段",
                        "preparation_seconds": fold_prep_seconds, "fit_seconds": 0.0,
                        "prediction_seconds": 0.0, "cache_source": "not_run",
                    })
                    continue
                test_window = labelled.iloc[test_positions]
                scorable_test = test_window[fold_features].notna().any(axis=1).to_numpy()
                fold_test_positions = test_positions[scorable_test]
                if not len(fold_test_positions):
                    fold_details.append({
                        "validation_segment": fold_number,
                        "test_start": str(pd.Timestamp(dates[fold_spec["test_start_index"]]).date()),
                        "test_end": str(pd.Timestamp(dates[fold_spec["test_stop_index"] - 1]).date()),
                        "training_samples": len(train), "validation_samples": 0,
                        "training_dates": training_date_count, "features": len(fold_features),
                        "feature_columns": fold_features, "threads": int(settings.TRAINING_N_JOBS),
                        "status": "验证窗口内没有可用特征值，跳过该段",
                        "preparation_seconds": fold_prep_seconds, "fit_seconds": 0.0,
                        "prediction_seconds": 0.0, "cache_source": "not_run",
                    })
                    continue
                test_dates = pd.to_datetime(dates[fold_spec["test_start_index"]:fold_spec["test_stop_index"]])
                test_start = pd.Timestamp(test_dates[0])
                test_end = pd.Timestamp(test_dates[-1])
                training_start = pd.Timestamp(train["date"].min())
                training_end = pd.Timestamp(train["date"].max())
                latest_training_label_end = pd.Timestamp(train[endpoint].max())
                environment = _runtime_environment()
                fold_prepared = {**prepared, "features": fold_features}
                cache_key = _validation_cache_key(
                    name, horizon, candidate, window, fold_number, fold_spec, fold_prepared,
                    len(train), len(fold_test_positions), environment, model_parameters,
                ) if validation_cache is not None else None
                cached = None
                lookup_seconds = 0.0
                if validation_cache is not None and cache_key is not None:
                    lookup_started = perf_counter()
                    cached = validation_cache.load(cache_key, len(test_positions))
                    lookup_seconds = perf_counter() - lookup_started
                    stage_seconds["cache_lookup"] += lookup_seconds
                    if cached is not None:
                        metadata = cached["metadata"]
                        try:
                            cached_baseline = float(metadata["baseline"])
                            cached_processing = metadata["label_processing"]
                            if not math.isfinite(cached_baseline) or not isinstance(cached_processing, dict):
                                raise ValueError("缓存折元数据无效")
                            baseline, processing = cached_baseline, cached_processing
                        except (KeyError, TypeError, ValueError):
                            validation_cache.delete(cache_key)
                            cached = None
                    if cached is not None:
                        cache_summary["hits"] += 1
                    else:
                        cache_summary["misses"] += 1
                cache_source = "hot" if cached is not None else "cold" if cache_enabled else "disabled"
                if progress:
                    stage = "复用缓存验证段" if cached is not None else "原生拟合"
                    progress(
                        f"按交易日验证 {name} · {horizon}日 · {candidate} · 第{fold_number}/{len(prepared['folds'])}段 · "
                        f"{stage} · 训练 {len(train_positions):,} 行 / {training_date_count} 日 · "
                        f"已用 {perf_counter() - started:.1f} 秒"
                    )
                fit_seconds = 0.0
                prediction_seconds = 0.0
                write_seconds = 0.0
                if cached is not None:
                    predictions = cached["predictions"]
                else:
                    fit_started = perf_counter()
                    model, processing = _fit(_estimator(candidate), train, candidate, fold_features)
                    fit_seconds = perf_counter() - fit_started
                    stage_seconds["native_fit"] += fit_seconds
                    if progress:
                        progress(
                            f"按交易日验证 {name} · {horizon}日 · {candidate} · 第{fold_number}/{len(prepared['folds'])}段 · "
                            f"预测 · 已用 {perf_counter() - started:.1f} 秒"
                        )
                    prediction_started = perf_counter()
                    test_features = labelled.loc[:, fold_features].iloc[fold_test_positions]
                    predictions = np.asarray(model.predict(test_features), dtype=float)
                    prediction_seconds = perf_counter() - prediction_started
                    stage_seconds["prediction"] += prediction_seconds
                    baseline = float(train[target].median())
                    if validation_cache is not None and cache_key is not None:
                        write_started = perf_counter()
                        validation_cache.save(cache_key, predictions, {
                            "baseline": baseline, "label_processing": processing,
                        })
                        write_seconds = perf_counter() - write_started
                        stage_seconds["cache_write"] += write_seconds
                fold = pd.DataFrame({
                    "date": labelled["date"].to_numpy()[fold_test_positions],
                    "code": labelled["code"].to_numpy()[fold_test_positions],
                })
                fold["prediction"] = predictions
                fold["actual"] = labelled[target].to_numpy()[fold_test_positions]
                fold["absolute_return"] = labelled[forward_return].to_numpy()[fold_test_positions]
                fold["baseline"] = baseline
                oos.append(fold)
                fold_details.append({"validation_segment": fold_number,
                                     "trained_through": str(training_end.date()),
                                     "trained_from": str(training_start.date()),
                                     "training_dates": training_date_count,
                                     "latest_training_label_end": str(latest_training_label_end.date()),
                                     "test_start": str(test_start.date()), "test_end": str(test_end.date()),
                                     "training_samples": len(train), "validation_samples": len(fold_test_positions),
                                     "features": len(fold_features), "feature_columns": fold_features,
                                     "threads": int(settings.TRAINING_N_JOBS),
                                     "fit_seconds": fit_seconds, "prediction_seconds": prediction_seconds,
                                     "preparation_seconds": fold_prep_seconds, "cache_lookup_seconds": lookup_seconds,
                                     "cache_write_seconds": write_seconds, "cache_source": cache_source,
                                     "cache_key": cache_key,
                                     "label_processing": processing})
            if len(oos) < 2:
                result["status"] = "有效日期验证折不足，未生成预测"
                result["execution"] = _execution_summary(
                    name, horizon, candidate, prepared, candidate_started, stage_seconds, fold_details, cache_summary,
                    model_parameters,
                )
                return result
            evaluation = pd.concat(oos, ignore_index=True)
            if progress:
                progress(f"按交易日验证 {name} · {horizon}日 · {candidate} · 指标计算 · 已用 {perf_counter() - started:.1f} 秒")
            metrics_started = perf_counter()
            metrics = evaluate_oos(evaluation, horizon, panel["code"].nunique(), score_kind=result["score_kind"])
            stage_seconds["metrics"] = perf_counter() - metrics_started
            metrics["folds"] = [
                {key: detail[key] for key in (
                    "trained_through", "trained_from", "training_dates", "latest_training_label_end",
                    "test_start", "test_end", "feature_columns", "label_processing",
                )}
                for detail in fold_details if "trained_through" in detail
            ]
            result.update(oos_metrics=metrics, features_used=sorted({
                              column for detail in fold_details
                              for column in detail.get("feature_columns", [])
                          }),
                          regression_loss=settings.REGRESSION_LOSS if candidate == "regression" else None,
                          validation_retrain_days=validation_block,
                          label_processing={"folds": [
                              fold["label_processing"] for fold in fold_details if "label_processing" in fold
                          ]})
            if not metrics["validated"]:
                result["status"] = "未通过样本外基线、Rank IC或分组收益检验"
                result["execution"] = _execution_summary(
                    name, horizon, candidate, prepared, candidate_started, stage_seconds, fold_details, cache_summary,
                    model_parameters,
                )
                return result
            final_start = max(0, len(dates) - window) if window > 0 else 0
            final_mask = prepared["date_positions"] >= final_start
            final_window = labelled.iloc[np.flatnonzero(final_mask)]
            final_features = _select_fold_features(
                final_window, feature_pool, prepared["required_stocks"],
            )
            if not final_features:
                raise ValueError("最终训练窗口内没有达到覆盖门槛的特征")
            final = final_window.loc[final_window[final_features].notna().any(axis=1)]
            if final["date"].nunique() < MINIMUM_TRAIN_DATES:
                raise ValueError("最终训练窗口特征覆盖后的有效日期不足")
            prepared["features"] = final_features
            if progress:
                progress(f"按交易日验证 {name} · {horizon}日 · {candidate} · 最终拟合 · 已用 {perf_counter() - started:.1f} 秒")
            final_started = perf_counter()
            model, final_processing = _fit(_estimator(candidate), final, candidate, final_features)
            stage_seconds["final_fit"] = perf_counter() - final_started
            result["label_processing"]["final"] = final_processing
            importance = np.asarray(
                getattr(model, "feature_importances_", np.zeros(len(final_features))), dtype=float,
            )
            result.update(validated=True, status="通过样本外基线、Rank IC与分组收益检验", estimator=model,
                          features_used=final_features,
                          feature_importance=summarize_importances(final_features, importance))
        except ImportError as exc:
            result["status"] = f"模型依赖缺失：{exc.name}"
        except (ValueError, RuntimeError, TypeError) as exc:
            LOGGER.exception("截面模型训练失败：%s %s日", name, horizon)
            result["status"] = f"模型训练失败：{type(exc).__name__}"
        result["execution"] = _execution_summary(
            name, horizon, candidate, prepared, candidate_started, stage_seconds, fold_details, cache_summary,
            model_parameters,
        )
        LOGGER.info(
            "按交易日验证完成 model=%s horizon=%d objective=%s validated=%s stocks=%d features=%d samples=%d "
            "threads=%d elapsed=%.2fs cache_hits=%d cache_misses=%d stages=%s",
            name, horizon, candidate, result.get("validated"), prepared["total_stocks"],
            len(prepared["features"]), len(labelled),
            settings.TRAINING_N_JOBS, result["execution"]["elapsed_seconds"],
            cache_summary["hits"], cache_summary["misses"], stage_seconds,
        )
        return result

    if requested != "auto":
        return _run(requested)
    # auto：同时训练回归与排序目标，用样本外表现决定采用哪一个。
    attempts = [_run("regression"), _run("lambdarank")]
    best = max(attempts, key=_selection_key)
    best["objective_selection"] = [
        {"objective": item["objective"], "score_kind": item["score_kind"], "validated": item["validated"],
         "rank_ic": (item["oos_metrics"] or {}).get("rank_ic"), "ndcg": (item["oos_metrics"] or {}).get("ndcg"),
          "status": item["status"], "selected": item is best,
          "training_dates": item.get("training_dates"), "samples": item.get("samples"),
          "regression_loss": item.get("regression_loss"), "label_processing": item.get("label_processing"),
          "execution": item.get("execution"),
          "oos_metrics": {key: value for key, value in (item.get("oos_metrics") or {}).items() if key != "daily_groups"}}
        for item in attempts
    ]
    return best


def predict_panel(model: dict[str, Any], panel: pd.DataFrame) -> pd.DataFrame:
    """只允许通过验证的模型对完整日期截面预测，随后逐日标准化。"""
    columns = ["date", "code", "prediction", "prediction_zscore", "prediction_percentile", "model_rank"]
    if not model.get("validated") or "estimator" not in model:
        return pd.DataFrame(columns=columns)
    features = list(model.get("features_used") or [])
    if not features or panel is None or panel.empty:
        return pd.DataFrame(columns=columns)
    values = panel.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)
    scorable = values.notna().any(axis=1)
    valid = panel.loc[scorable].copy()
    values = values.loc[scorable]
    required = max(MINIMUM_STOCKS, math.ceil(panel["code"].nunique() * MINIMUM_COVERAGE))
    valid = valid.loc[valid.groupby("date")["code"].transform("nunique").ge(required)]
    if valid.empty:
        return pd.DataFrame(columns=columns)
    values = values.loc[valid.index]
    predictions = valid[["date", "code"]].copy()
    predictions["prediction"] = np.asarray(model["estimator"].predict(values), dtype=float)
    return normalize_predictions(predictions).dropna(subset=["prediction_zscore"])


def final_holdout_context_fields() -> dict[str, Any]:
    """模型上下文里的四个最终留出集字段。

    留出集只评估一次、报告写盘后长期有效，所以这里直接读报告文件而不是查库：
    ``train_market_context`` 会跑在没有数据库的进程里（单测也会在不连库的情况下直接调它），
    任何查库的写法都会把训练入口绑死在 PostgreSQL 上。

    未评估时如实写 False / None，绝不写「已评估」——报告不存在就是没评估过。
    """
    enabled = bool(settings.ENABLE_FINAL_HOLDOUT)
    report = None
    if enabled:
        path = settings.MODEL_DIR / settings.FINAL_HOLDOUT_REPORT_NAME
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                report = loaded if isinstance(loaded, dict) else None
            except (OSError, ValueError):
                LOGGER.exception("读取最终留出集报告失败，按未评估处理")
    if not report:
        return {
            "final_holdout_evaluated": False,
            "final_holdout_result": None,
            "final_holdout_start": settings.FINAL_HOLDOUT_START if enabled else None,
            "final_holdout_end": settings.FINAL_HOLDOUT_END if enabled else None,
            "final_holdout_protocol_consumed": False,
        }
    if report.get("model_version") != settings.MODEL_VERSION:
        return {
            "final_holdout_evaluated": False,
            "final_holdout_result": None,
            "final_holdout_start": report.get("holdout_start"),
            "final_holdout_end": report.get("holdout_end"),
            "final_holdout_protocol_consumed": True,
            "final_holdout_protocol_model_version": report.get("model_version"),
        }
    summary = report.get("summary") or {}
    return {
        "final_holdout_evaluated": True,
        "final_holdout_result": {
            "model_version": report.get("model_version"),
            "context_id": report.get("context_id"),
            "evaluated_at": report.get("evaluated_at"),
            "passed": report.get("passed"),
            **summary,
        },
        "final_holdout_start": report.get("holdout_start"),
        "final_holdout_end": report.get("holdout_end"),
    }


def train_market_context(
    histories: dict[str, pd.DataFrame], benchmark: pd.DataFrame, metadata: dict[str, Any] | None = None,
    progress: Callable[[str], None] | None = None, persist: bool = True,
    exposures: pd.DataFrame | None = None, objective: str | None = None, window: int | None = None,
    extra_factors: dict[str, Any] | None = None, store: PanelStore | None = None,
    panel_out: dict[str, Any] | None = None,
    style_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """一次扫描汇总历史后按滚动窗口训练共享模型，供个股页和扫描页读取相同结果。

    ``panel_out`` 是可选的可变容器：训练用的截面面板放回 ``panel_out["panel"]``，
    供组合回测与治理直接复用。这样自定义池（persist=False）也不会误读活动模型的旧面板，
    同时省掉一次几十兆 Parquet 的重读。
    """
    panel = build_market_panel(
        histories, benchmark, exposures=exposures, extra_factors=extra_factors, store=store, progress=progress,
        style_options={**(style_options or {}), "objective": objective, "training_window": window},
    )
    if panel_out is not None:
        panel_out["panel"] = panel
    if panel.empty:
        configured_window = settings.TRAINING_WINDOW_DAYS if window is None else window
        style_report = panel.attrs.get("style_factors") or {}
        context = {
            "context_id": uuid.uuid4().hex, "model_version": settings.MODEL_VERSION,
            "target_kind": TARGET_KIND, "created_at": datetime.now().astimezone().isoformat(),
            "reference_date": None, "universe_codes": sorted(histories), "source": metadata or {},
            "models": {}, "predictions": {}, "reference_closes": {},
            "objective_mode": objective or settings.DEFAULT_OBJECTIVE_MODE,
            "training_window_days": configured_window,
            "exposures": panel.attrs.get("exposures"),
            "neutralization": panel.attrs.get("neutralization"),
            "extra_factors": panel.attrs.get("extra_factors"),
            "health_veto": panel.attrs.get("health_veto"),
            "style_factors": style_report,
            "model_feature_columns": panel.attrs.get("model_feature_columns", []),
            "feature_configuration_id": style_report.get("configuration_id"),
            "status": "风险否决后候选池为空，未训练或发布模型",
            "limitation": "财务健康否决后无可交易截面；未产生模型预测。",
            **final_holdout_context_fields(),
        }
        for horizon in HORIZONS:
            context["models"][horizon] = [
                unavailable_model(
                    name, horizon, "风险否决后无可交易样本，未训练模型",
                    objective=objective, window=configured_window,
                )
                for name in MODEL_NAMES
            ]
            context["predictions"][horizon] = {}
        return context
    latest_date = panel["date"].max()
    live = panel.loc[panel["date"].eq(latest_date)]
    window = int(settings.TRAINING_WINDOW_DAYS if window is None else window)
    context = {"context_id": uuid.uuid4().hex, "model_version": settings.MODEL_VERSION, "target_kind": TARGET_KIND,
               "created_at": datetime.now().astimezone().isoformat(), "reference_date": str(latest_date.date()),
               "universe_codes": sorted(histories), "source": metadata or {}, "models": {}, "predictions": {},
               "reference_closes": dict(zip(live["code"], live["close"])),
               "objective_mode": objective or settings.DEFAULT_OBJECTIVE_MODE, "training_window_days": window,
                "label_winsor_upper_quantile": settings.LABEL_WINSOR_UPPER_QUANTILE,
                "regression_loss": settings.REGRESSION_LOSS,
                "validation_retrain_days": settings.OOS_RETRAIN_DAYS,
               "ndcg_protocol": settings.NDCG_PROTOCOL,
               "exposures": panel.attrs.get("exposures"), "neutralization": panel.attrs.get("neutralization"),
               "extra_factors": panel.attrs.get("extra_factors"),
               "style_factors": panel.attrs.get("style_factors"),
               "model_feature_columns": panel.attrs.get("model_feature_columns"),
               "feature_configuration_id": panel.attrs["style_factors"]["configuration_id"],
               "market_regime": (metadata or {}).get("market_regime"),
               # 留出集状态随每次扫描重新注入：context 每轮都是新建的字典，
               # 不从报告文件读回来的话这四个字段永远显示「未评估」。
               **final_holdout_context_fields(),
               "limitation": "截面仅含本次成功采集历史的股票池；当前股票池回溯存在幸存者偏差，分组收益为持有期重叠的诊断指标，非可执行组合净值。"}
    for horizon in HORIZONS:
        context["models"][horizon] = []
        context["predictions"][horizon] = {}
        if progress:
            progress(f"按交易日验证 {horizon}日 · 准备共享评估数据")
        prepared = _prepare_validation_data(panel, horizon, None, window)
        for name in MODEL_NAMES:
            if progress:
                progress(f"按交易日验证 {name} · {horizon}日截面排序模型")
            model = fit_panel_model(
                panel, name, horizon, objective=objective, window=window,
                progress=progress, _prepared=prepared,
            )
            model["feature_protocol"] = settings.STYLE_FEATURE_PROTOCOL
            predicted = predict_panel(model, live)
            if model["validated"] and predicted.empty:
                model.update(validated=False, status="最新截面不足或模型预测无差异，未输出排序")
            context["models"][horizon].append(model)
            for row in predicted.to_dict("records"):
                context["predictions"][horizon].setdefault(row["code"], {})[name] = {
                    "predicted_excess_return": float(row["prediction"]), "prediction_zscore": float(row["prediction_zscore"]),
                    "prediction_percentile": float(row["prediction_percentile"]), "model_rank": float(row["model_rank"]),
                    "score_kind": model["score_kind"],
                }
    panel.attrs["context_id"] = context["context_id"]
    if persist:
        save_context(context, panel)
    return context


def save_context_summary(context: dict[str, Any], path: Path) -> None:
    """原子保存模型摘要，避免原地改写污染指向同一文件的历史归档。

    参数为模型上下文与摘要目标路径；不修改模型、面板或数据库。
    """
    summary = {key: value for key, value in context.items() if key not in ("models", "predictions", "reference_closes")}
    summary["validation"] = {str(horizon): [{key: value for key, value in model.items() if key != "estimator"} for model in models] for horizon, models in context["models"].items()}
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_context(context: dict[str, Any], panel: pd.DataFrame) -> None:
    """版本化原子保存模型、截面与指标；旧版绝对收益模型不复用。"""
    import joblib

    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    identifier = context["context_id"]
    panel_path = settings.DATA_DIR / f"ml_panel_v{settings.MODEL_VERSION}.parquet"
    model_path = settings.MODEL_DIR / f"market_context_v{settings.MODEL_VERSION}.joblib"
    panel_temporary = panel_path.with_suffix("." + identifier + ".tmp")
    model_temporary = model_path.with_suffix("." + identifier + ".tmp")
    panel.to_parquet(panel_temporary, index=False)
    joblib.dump(context, model_temporary)
    os.replace(panel_temporary, panel_path)
    os.replace(model_temporary, model_path)
    save_context_summary(context, settings.MODEL_DIR / f"market_context_v{settings.MODEL_VERSION}.json")


def _context_configuration_is_current(context: dict[str, Any]) -> bool:
    """加载和每次输出预测共用验收闸门，页面缓存不能绕过验收撤销。"""
    report = context.get("style_factors")
    if not production_report_is_current(report):
        return False
    expected = report["configuration_id"] if report else configuration_id(
        context.get("objective_mode"), context.get("training_window_days"))
    return not context.get("feature_configuration_id") or context["feature_configuration_id"] == expected


def load_context(include_panel: bool = False) -> tuple[dict[str, Any] | None, pd.DataFrame | None]:
    """读取本程序生成的受信任本地模型；版本或上下文不一致时拒绝使用。"""
    import joblib

    path = settings.MODEL_DIR / f"market_context_v{settings.MODEL_VERSION}.joblib"
    if not path.exists():
        return None, None
    try:
        context = joblib.load(path)
        if context.get("model_version") != settings.MODEL_VERSION or context.get("target_kind") != TARGET_KIND:
            return None, None
        if not _context_configuration_is_current(context):
            LOGGER.warning("风格规则或独立验收已失效，拒绝继续读取旧模型权重")
            return None, None
        panel = pd.read_parquet(settings.DATA_DIR / f"ml_panel_v{settings.MODEL_VERSION}.parquet") if include_panel else None
        if panel is not None and panel.attrs.get("context_id") != context["context_id"]:
            return None, None
        return context, panel
    except (OSError, ValueError, KeyError, EOFError, ImportError, pickle.UnpicklingError):
        LOGGER.exception("读取截面模型上下文失败")
        return None, None


CONTEXT_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def _archive_directory(context_id: Any) -> Path | None:
    """把 context_id 映射到归档目录；只接受 32 位十六进制并强制留在 archive 根内。"""
    if not isinstance(context_id, str) or not CONTEXT_ID_PATTERN.fullmatch(context_id):
        return None
    root = (settings.MODEL_DIR / settings.MODEL_ARCHIVE_DIRNAME).resolve()
    directory = (root / context_id).resolve()
    try:
        directory.relative_to(root)
    except ValueError:  # 理论上不可达，防的是未来有人放宽 id 校验
        return None
    return directory


def load_context_by_id(
    context_id: str, include_panel: bool = False
) -> tuple[dict[str, Any] | None, pd.DataFrame | None]:
    """按归档 id 读取已保存模型；任何不一致都返回 ``(None, None)``，不让页面读到错配数据。"""
    import joblib

    directory = _archive_directory(context_id)
    if directory is None:
        LOGGER.warning("拒绝加载非法模型 id：%r", context_id)
        return None, None
    version = settings.MODEL_VERSION
    model_path = directory / f"market_context_v{version}.joblib"
    if not directory.is_dir() or not model_path.exists():
        LOGGER.info("模型归档不存在或已被清理：%s", context_id)
        return None, None
    try:
        registry_path = directory / "registry.json"
        if registry_path.exists():
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            if registry.get("context_id") != context_id or registry.get("model_version") != version:
                LOGGER.warning("模型归档登记与目标不一致：%s", context_id)
                return None, None
        context = joblib.load(model_path)
        if (
            context.get("context_id") != context_id
            or context.get("model_version") != version
            or context.get("target_kind") != TARGET_KIND
        ):
            LOGGER.warning("模型归档内容与 id/版本不一致：%s", context_id)
            return None, None
        panel = None
        if not _context_configuration_is_current(context):
            LOGGER.warning("归档模型特征规则或独立验收已失效：%s", context_id)
            return None, None
        if include_panel:
            panel = pd.read_parquet(directory / f"ml_panel_v{version}.parquet")
            if panel.attrs.get("context_id") != context_id:
                LOGGER.warning("模型归档面板与上下文不一致：%s", context_id)
                return None, None
        return context, panel
    except (OSError, ValueError, KeyError, EOFError, ImportError, pickle.UnpicklingError):
        LOGGER.exception("读取归档模型失败：%s", context_id)
        return None, None


def historical_models_for_stock(
    context: dict[str, Any] | None, code: str | None
) -> list[dict[str, Any]]:
    """某份模型在它自己的截面日期上对该股的判断，用于旧模型回看。

    刻意复用 ``models_for_stock``：日期与收盘价必须与该模型的
    ``reference_date`` / ``reference_closes`` 完全一致。旧模型不能给新数据打分，
    这里也不伪造「当前结论」。
    """
    if not context or not code:
        return []
    date = context.get("reference_date")
    close = (context.get("reference_closes") or {}).get(str(code))
    if not date or close is None:
        return []
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        for item in models_for_stock(context, code, date, close, horizon):
            rows.append({
                "model": item.get("model"),
                "horizon": horizon,
                "predicted_excess_return": item.get("predicted_excess_return"),
                "prediction_percentile": item.get("prediction_percentile"),
                "prediction_zscore": item.get("prediction_zscore"),
                "validated": item.get("validated"),
                "status": item.get("status"),
                "score_kind": item.get("score_kind"),
                "reference_date": date,
                "context_id": context.get("context_id"),
            })
    return rows


def models_for_stock(context: dict[str, Any] | None, code: str | None, date: str, close: float | None, horizon: int) -> list[dict[str, Any]]:
    """验证同日同价格的数据血缘后，向两个页面提供一致的模型结果。"""
    status = "请先在量化选股中扫描至少5只股票以建立同日截面模型"
    if context:
        status = "当前股票不在已训练截面内，或日线已更新，请重新扫描股票池"
        if not _context_configuration_is_current(context):
            return [unavailable_model(name, horizon, "模型特征规则或独立验收已失效，请重新验收并训练")
                    for name in MODEL_NAMES]
    valid = context and context.get("reference_date") == date and str(code) in context.get("reference_closes", {})
    if valid:
        expected = context["reference_closes"][str(code)]
        valid = close is not None and np.isfinite(close) and np.isclose(close, expected, rtol=1e-8)
    if not valid:
        return [unavailable_model(name, horizon, status) for name in MODEL_NAMES]
    result = []
    for model in context["models"].get(horizon, []):
        item = {key: value for key, value in model.items() if key != "estimator"}
        prediction = context["predictions"].get(horizon, {}).get(str(code), {}).get(model["model"])
        if item["validated"] and prediction:
            item.update(prediction)
        elif item["validated"]:
            item.update(validated=False, status="该股票最新截面特征不完整，未生成预测")
        item["context_id"] = context["context_id"]
        item["reference_date"] = date
        result.append(item)
    return result
