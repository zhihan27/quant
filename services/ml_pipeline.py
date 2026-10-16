"""按交易日训练截面排序模型，预测相对沪深300的优胜劣汰并严格验证。"""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
import uuid
from datetime import datetime
from typing import Any, Callable

import numpy as np
import pandas as pd

from config import settings
from services.extra_factors import apply_health_veto, attach_extra_factors
from services.feature_engineering import FEATURE_COLUMNS, add_forward_returns, build_features, normalize_cross_section
from services.neutralization import (
    NEUTRAL_PREFIX, attach_market_cap_exposure, neutralize_cross_section, neutralized_feature_columns,
)
from services.panel_store import PanelStore
from services.rank_model import (
    SCORE_KIND_RANK, SCORE_KIND_RETURN, evaluate_ndcg, make_ranker, summarize_importances,
)

TARGET_KIND = "excess_return_vs_csi300"
MINIMUM_STOCKS = 5
MINIMUM_COVERAGE = 0.8
MINIMUM_TRAIN_DATES = 80
MINIMUM_OOS_DATES = 40
MINIMUM_RANK_IC = 0.02
MINIMUM_IC_T = 1.645
# 排序目标的 NDCG 必须显著高于随机排序基线，才认为学到的是截面区分能力。
MINIMUM_NDCG_RATIO = 1.05
# 中性化因子覆盖率低于该比例时退回截面Z分数因子，避免样本被残差缺失过度削减。
MINIMUM_NEUTRAL_COVERAGE = 0.5
MODEL_NAMES = ("LightGBM", "XGBoost")
HORIZONS = (5, 20, 60)
LOGGER = logging.getLogger("quant.ml")


def unavailable_model(name: str, horizon: int, status: str, objective: str | None = None, window: int | None = None) -> dict[str, Any]:
    """无截面或验证失败时只返回状态，禁止保留未经验证的预测值。"""
    return {
        "model": name, "horizon": horizon, "target_kind": TARGET_KIND,
        "model_version": settings.MODEL_VERSION, "validated": False, "status": status,
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
) -> pd.DataFrame:
    """构造可追溯的股票日截面，可选对市值与行业暴露中性化，保留标签结束日以便按日期清除重叠样本。

    传入 ``extra_factors`` 时，筹码分布、日内高频与 PIT 财务因子按覆盖率门控自动并入排序特征矩阵；
    不传时行为与既有版本完全一致。
    """
    if benchmark is None or benchmark.empty:
        raise ValueError("沪深300历史不可用，不能训练超额收益模型。")
    if len(histories) < MINIMUM_STOCKS:
        raise ValueError(f"截面模型至少需要{MINIMUM_STOCKS}只股票的历史；当前{len(histories)}只。")
    frames = []
    for code, history in histories.items():
        features = build_features(history, benchmark)
        labelled = add_forward_returns(features, HORIZONS, benchmark)
        labelled["code"] = str(code)
        columns = ["date", "code", "close", *FEATURE_COLUMNS]
        columns += [f"{prefix}_{horizon}" for horizon in HORIZONS for prefix in ("target", "forward_return", "label_end")]
        frames.append(labelled[columns])
    panel = pd.concat(frames, ignore_index=True).sort_values(["date", "code"]).reset_index(drop=True)
    # 新增因子先装配、财务健康否决先生效，之后才做截面标准化：被否决的标的不能参与自己所在截面的
    # 均值与标准差，否则「打分校准的股票池」与「实际可交易的股票池」会不一致。
    extra_report: dict[str, Any] = {"status": "未提供新增因子", "families": {}}
    if extra_factors:
        panel, extra_report = attach_extra_factors(panel, extra_factors, store=store)
    panel, veto_report = apply_health_veto(panel)
    panel = normalize_cross_section(panel, FEATURE_COLUMNS, MINIMUM_STOCKS, MINIMUM_COVERAGE)
    model_features = [f"xs_{column}" for column in FEATURE_COLUMNS]
    neutralization: dict[str, Any] = {"status": "未启用市值与行业中性化，模型使用截面Z分数因子"}
    exposure_report: dict[str, Any] = {"status": "未接入市值与行业暴露"}
    if exposures is not None and not exposures.empty:
        panel, exposure_report = attach_market_cap_exposure(panel, exposures)
        panel, neutralization = neutralize_cross_section(panel, FEATURE_COLUMNS)
        candidates = neutralized_feature_columns(panel, FEATURE_COLUMNS)
        coverage = float(panel[candidates].notna().all(axis=1).mean()) if candidates else 0.0
        if candidates and coverage >= MINIMUM_NEUTRAL_COVERAGE:
            model_features = candidates
            neutralization = {**neutralization, "model_features": len(candidates), "coverage": coverage}
        else:
            neutralization = {
                **neutralization, "coverage": coverage,
                "status": f"{neutralization['status']}；中性化因子覆盖率 {coverage:.0%} 偏低，模型改用截面Z分数因子",
            }
    accepted = list(extra_report.get("accepted_columns") or [])
    if accepted:
        features: list[str] = []
        coverage = 0.0
        neutralized = bool(model_features) and model_features[0].startswith(NEUTRAL_PREFIX)
        # 基础因子做过中性化时新增因子也必须一起做，否则模型会绕过中性化，
        # 顺着新增因子把市值与行业暴露重新学回来。
        if neutralized:
            panel, extra_neutralization = neutralize_cross_section(panel, accepted)
            candidates = neutralized_feature_columns(panel, accepted)
            coverage = float(panel[candidates].notna().all(axis=1).mean()) if candidates else 0.0
            if candidates and coverage >= MINIMUM_NEUTRAL_COVERAGE:
                features = candidates
            extra_report = {**extra_report, "neutralization": {**extra_neutralization, "coverage": coverage}}
        if not features:
            # 树模型对特征的单调变换不敏感，这里统一做截面Z分数只为与基础因子保持同一口径；
            # 中性化覆盖率不足时也走这条路，原因写进报告便于核对。
            panel = normalize_cross_section(panel, accepted, MINIMUM_STOCKS, MINIMUM_COVERAGE)
            features = [f"xs_{column}" for column in accepted if f"xs_{column}" in panel]
            if neutralized:
                extra_report["neutralization"] = {
                    **extra_report["neutralization"],
                    "status": f"新增因子中性化覆盖率 {coverage:.0%} 偏低，改用截面Z分数",
                }
        model_features = [*model_features, *features]
        extra_report = {**extra_report, "model_features": len(features)}
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


def evaluate_oos(
    frame: pd.DataFrame, horizon: int, total_stocks: int | None = None, score_kind: str = SCORE_KIND_RETURN
) -> dict[str, Any]:
    """计算每日Rank IC、HAC显著性、Top/Bottom收益与排序质量基线。"""
    from statsmodels.api import OLS

    clean = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["prediction", "actual", "baseline", "absolute_return"])
    required = max(MINIMUM_STOCKS, math.ceil((total_stocks or MINIMUM_STOCKS) * MINIMUM_COVERAGE))
    rows = []
    eligible_indices = []
    for date, group in clean.groupby("date", sort=True):
        if group["code"].nunique() < required or group["prediction"].nunique() < 2 or group["actual"].nunique() < 2:
            continue
        ordered = group.sort_values("prediction", kind="stable")
        size = max(1, len(group) // 5)
        bottom, top = ordered.head(size), ordered.tail(size)
        ic = float(group["prediction"].rank().corr(group["actual"].rank()))
        if not np.isfinite(ic):
            continue
        rows.append({
            "date": str(pd.Timestamp(date).date()), "rank_ic": ic,
            "top_excess_return": float(top["actual"].mean()), "bottom_excess_return": float(bottom["actual"].mean()),
            "top_return": float(top["absolute_return"].mean()), "bottom_return": float(bottom["absolute_return"].mean()),
            "universe_excess_return": float(group["actual"].mean()), "stocks": len(group),
        })
        eligible_indices.extend(group.index.tolist())
    evaluated = clean.loc[eligible_indices]
    ranking = score_kind == SCORE_KIND_RANK
    metrics = {"dates": len(rows), "oos_samples": len(evaluated), "daily_groups": rows, "target_kind": TARGET_KIND,
               "score_kind": score_kind, "ndcg": None, "ndcg_random_baseline": None, "ndcg_ratio": None,
               "rank_ic": None, "rank_ic_hac_t": None, "top_excess_return": None, "bottom_excess_return": None,
               "top_return": None, "bottom_return": None, "top_bottom_spread": None, "mae": None, "baseline_mae": None,
               "validated": False, "validation_rule": (
                   "排序目标：每日Rank IC>0.02且HAC t>1.645；NDCG@截面前20%高于随机排序基线5%以上；"
                   "Top超额为正并优于池均值；Top-Bottom为正；至少40个样本外交易日"
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
    ndcg, ndcg_baseline = quality["ndcg"], quality["ndcg_random_baseline"]
    ndcg_ratio = ndcg / ndcg_baseline if ndcg is not None and ndcg_baseline else None
    top = float(daily["top_excess_return"].mean())
    bottom = float(daily["bottom_excess_return"].mean())
    spread = top - bottom
    passed = len(rows) >= MINIMUM_OOS_DATES and ic_mean > MINIMUM_RANK_IC and ic_t is not None and ic_t > MINIMUM_IC_T
    passed = passed and top > max(0.0, float(daily["universe_excess_return"].mean())) and spread > 0
    if ranking:
        passed = passed and ndcg_ratio is not None and ndcg_ratio > MINIMUM_NDCG_RATIO
    else:
        passed = passed and mae < baseline_mae
    metrics.update(rank_ic=ic_mean, rank_ic_hac_t=float(ic_t) if ic_t is not None else None,
                   ic_test_status="有效" if ic_t is not None else "IC方差退化，拒绝推断显著性", mae=mae, baseline_mae=baseline_mae,
                   ndcg=ndcg, ndcg_random_baseline=ndcg_baseline, ndcg_days=quality["ndcg_days"],
                   ndcg_ratio=ndcg_ratio,
                   top_excess_return=top, bottom_excess_return=bottom, top_bottom_spread=spread,
                   top_return=float(daily["top_return"].mean()), bottom_return=float(daily["bottom_return"].mean()), validated=bool(passed))
    return metrics


def _selection_key(result: dict[str, Any]) -> tuple[int, float]:
    """目标函数对比时的择优口径：先看是否通过验证，再看样本外Rank IC。"""
    metrics = result.get("oos_metrics") or {}
    return (1 if result.get("validated") else 0, float(metrics.get("rank_ic") or -1.0))


def fit_panel_model(
    panel: pd.DataFrame, name: str, horizon: int, model_factory: Callable | None = None,
    cutoff: pd.Timestamp | None = None, window: int | None = None, objective: str | None = None,
) -> dict[str, Any]:
    """按完整交易日划分滚动样本外折，训练窗口只保留最近N个交易日，严格清除标签跨入测试期的样本。"""
    from sklearn.model_selection import TimeSeriesSplit

    requested = objective or settings.DEFAULT_OBJECTIVE_MODE
    if requested not in settings.RANK_OBJECTIVE_MODES:
        raise ValueError(f"目标函数仅支持 {'、'.join(settings.RANK_OBJECTIVE_MODES)}；当前为 {requested}。")
    window = int(settings.TRAINING_WINDOW_DAYS if window is None else window)
    frame = panel.loc[panel["date"] < cutoff].copy() if cutoff is not None else panel
    target, endpoint = f"target_{horizon}", f"label_end_{horizon}"
    all_dates = frame["date"].drop_duplicates().sort_values()
    feature_pool = list(panel.attrs.get("model_feature_columns") or [f"xs_{column}" for column in FEATURE_COLUMNS])
    calibration = frame.loc[frame["date"].isin(all_dates.iloc[:80])]
    features = [
        column for column in feature_pool
        if column in calibration and calibration[column].notna().sum() >= MINIMUM_STOCKS * 20
    ]
    if not features:
        return unavailable_model(name, horizon, "截面特征样本不足，未生成预测", objective=requested, window=window)
    labelled = frame.dropna(subset=features + [target, endpoint]).copy()
    if cutoff is not None:
        labelled = labelled.loc[labelled[endpoint] < cutoff]
    dates = np.sort(labelled["date"].unique())

    def _estimator(candidate: str) -> Any:
        # 显式注入的工厂优先，保持测试与自定义估计器的行为完全可控。
        if model_factory is not None:
            return model_factory(name)
        if candidate == "regression":
            from services.stock_scoring import _make_model

            return _make_model(name)
        return make_ranker(name)

    def _fit(estimator: Any, sample: pd.DataFrame) -> Any:
        if hasattr(estimator, "fit_cross_section"):
            estimator.fit_cross_section(sample[features], sample[target], sample["date"])
        else:
            estimator.fit(sample[features], sample[target])
        return estimator

    def _run(candidate: str) -> dict[str, Any]:
        result = unavailable_model(name, horizon, "截面样本不足，未生成预测", objective=candidate, window=window)
        result["samples"] = len(labelled)
        result["training_dates"] = len(dates)
        result["score_kind"] = SCORE_KIND_RETURN if candidate == "regression" else SCORE_KIND_RANK
        if len(dates) < 240 or labelled["code"].nunique() < MINIMUM_STOCKS:
            return result
        oos: list[pd.DataFrame] = []
        fold_details = []
        try:
            for train_dates, test_dates in TimeSeriesSplit(n_splits=3).split(dates):
                test_start = pd.Timestamp(dates[test_dates[0]])
                train = labelled.loc[labelled["date"].isin(dates[train_dates]) & (labelled[endpoint] < test_start)]
                if window > 0:
                    # 滚动窗口：只用最近 window 个交易日重训，自适应市场风格切换。
                    train = train.loc[train["date"].isin(dates[train_dates][-window:])]
                test = labelled.loc[labelled["date"].isin(dates[test_dates])]
                if train["date"].nunique() < MINIMUM_TRAIN_DATES:
                    continue
                model = _fit(_estimator(candidate), train)
                predictions = np.asarray(model.predict(test[features]), dtype=float)
                fold = test[["date", "code"]].copy()
                fold["prediction"] = predictions
                fold["actual"] = test[target].to_numpy()
                fold["absolute_return"] = test[f"forward_return_{horizon}"].to_numpy()
                fold["baseline"] = float(train[target].median())
                oos.append(fold)
                fold_details.append({"trained_through": str(train["date"].max().date()),
                                     "trained_from": str(train["date"].min().date()),
                                     "training_dates": int(train["date"].nunique()),
                                     "latest_training_label_end": str(train[endpoint].max().date()),
                                     "test_start": str(test_start.date()), "test_end": str(pd.Timestamp(dates[test_dates[-1]]).date())})
            if len(oos) < 2:
                result["status"] = "有效日期验证折不足，未生成预测"
                return result
            evaluation = pd.concat(oos, ignore_index=True)
            metrics = evaluate_oos(evaluation, horizon, panel["code"].nunique(), score_kind=result["score_kind"])
            metrics["folds"] = fold_details
            result.update(oos_metrics=metrics, features_used=features)
            if not metrics["validated"]:
                result["status"] = "未通过样本外基线、Rank IC或分组收益检验"
                return result
            final = labelled.loc[labelled["date"].isin(dates[-window:] if window > 0 else dates)]
            model = _fit(_estimator(candidate), final)
            importance = np.asarray(getattr(model, "feature_importances_", np.zeros(len(features))), dtype=float)
            result.update(validated=True, status="通过样本外基线、Rank IC与分组收益检验", estimator=model,
                          feature_importance=summarize_importances(features, importance))
        except ImportError as exc:
            result["status"] = f"模型依赖缺失：{exc.name}"
        except (ValueError, RuntimeError, TypeError) as exc:
            LOGGER.exception("截面模型训练失败：%s %s日", name, horizon)
            result["status"] = f"模型训练失败：{type(exc).__name__}"
        return result

    if requested != "auto":
        return _run(requested)
    # auto：同时训练回归与排序目标，用样本外表现决定采用哪一个。
    attempts = [_run("regression"), _run("lambdarank")]
    best = max(attempts, key=_selection_key)
    best["objective_selection"] = [
        {"objective": item["objective"], "score_kind": item["score_kind"], "validated": item["validated"],
         "rank_ic": (item["oos_metrics"] or {}).get("rank_ic"), "ndcg": (item["oos_metrics"] or {}).get("ndcg"),
         "status": item["status"]}
        for item in attempts
    ]
    return best


def predict_panel(model: dict[str, Any], panel: pd.DataFrame) -> pd.DataFrame:
    """只允许通过验证的模型对完整日期截面预测，随后逐日标准化。"""
    columns = ["date", "code", "prediction", "prediction_zscore", "prediction_percentile", "model_rank"]
    if not model.get("validated") or "estimator" not in model:
        return pd.DataFrame(columns=columns)
    features = model["features_used"]
    valid = panel.dropna(subset=features)
    required = max(MINIMUM_STOCKS, math.ceil(panel["code"].nunique() * MINIMUM_COVERAGE))
    valid = valid.loc[valid.groupby("date")["code"].transform("nunique").ge(required)]
    if valid.empty:
        return pd.DataFrame(columns=columns)
    predictions = valid[["date", "code"]].copy()
    predictions["prediction"] = np.asarray(model["estimator"].predict(valid[features]), dtype=float)
    return normalize_predictions(predictions).dropna(subset=["prediction_zscore"])


def train_market_context(
    histories: dict[str, pd.DataFrame], benchmark: pd.DataFrame, metadata: dict[str, Any] | None = None,
    progress: Callable[[str], None] | None = None, persist: bool = True,
    exposures: pd.DataFrame | None = None, objective: str | None = None, window: int | None = None,
    extra_factors: dict[str, Any] | None = None, store: PanelStore | None = None,
) -> dict[str, Any]:
    """一次扫描汇总历史后按滚动窗口训练共享模型，供个股页和扫描页读取相同结果。"""
    panel = build_market_panel(histories, benchmark, exposures=exposures, extra_factors=extra_factors, store=store)
    latest_date = panel["date"].max()
    live = panel.loc[panel["date"].eq(latest_date)]
    window = int(settings.TRAINING_WINDOW_DAYS if window is None else window)
    context = {"context_id": uuid.uuid4().hex, "model_version": settings.MODEL_VERSION, "target_kind": TARGET_KIND,
               "created_at": datetime.now().astimezone().isoformat(), "reference_date": str(latest_date.date()),
               "universe_codes": sorted(histories), "source": metadata or {}, "models": {}, "predictions": {},
               "reference_closes": dict(zip(live["code"], live["close"])),
               "objective_mode": objective or settings.DEFAULT_OBJECTIVE_MODE, "training_window_days": window,
               "exposures": panel.attrs.get("exposures"), "neutralization": panel.attrs.get("neutralization"),
               "extra_factors": panel.attrs.get("extra_factors"),
               "limitation": "截面仅含本次成功采集历史的股票池；当前股票池回溯存在幸存者偏差，分组收益为持有期重叠的诊断指标，非可执行组合净值。"}
    for horizon in HORIZONS:
        context["models"][horizon] = []
        context["predictions"][horizon] = {}
        for name in MODEL_NAMES:
            if progress:
                progress(f"按交易日验证 {name} · {horizon}日截面排序模型")
            model = fit_panel_model(panel, name, horizon, objective=objective, window=window)
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
    summary = {key: value for key, value in context.items() if key not in ("models", "predictions", "reference_closes")}
    summary["validation"] = {str(horizon): [{key: value for key, value in model.items() if key != "estimator"} for model in models] for horizon, models in context["models"].items()}
    (settings.MODEL_DIR / f"market_context_v{settings.MODEL_VERSION}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


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
        panel = pd.read_parquet(settings.DATA_DIR / f"ml_panel_v{settings.MODEL_VERSION}.parquet") if include_panel else None
        if panel is not None and panel.attrs.get("context_id") != context["context_id"]:
            return None, None
        return context, panel
    except (OSError, ValueError, KeyError, EOFError, ImportError, pickle.UnpicklingError):
        LOGGER.exception("读取截面模型上下文失败")
        return None, None


def models_for_stock(context: dict[str, Any] | None, code: str | None, date: str, close: float | None, horizon: int) -> list[dict[str, Any]]:
    """验证同日同价格的数据血缘后，向两个页面提供一致的模型结果。"""
    status = "请先在量化选股中扫描至少5只股票以建立同日截面模型"
    if context:
        status = "当前股票不在已训练截面内，或日线已更新，请重新扫描股票池"
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
