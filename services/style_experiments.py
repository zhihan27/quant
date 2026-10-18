"""固定样本的四组风格对照与一次性独立验收，不发布生产模型。"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from config import settings
from services.acquisition_store import _atomic_json
from services.backtest import cost_rates
from services.ml_pipeline import build_market_panel, evaluate_oos, fit_panel_model, predict_panel
from services.style_factors import SIZE_COLUMNS, VALUE_COLUMNS, attach_style_factors, configuration_id

LOGGER = logging.getLogger("quant.style_experiments")
VARIANTS = {"base": (False, False), "size": (True, False), "value": (False, True), "size_value": (True, True)}


def comparison_fingerprint(panels: dict[str, pd.DataFrame], end: str) -> str:
    """冻结选择期及以前的股票、特征和成熟标签，允许只追加未见日期。"""
    digest = hashlib.sha256()
    last = pd.Timestamp(end)
    for variant in sorted(panels):
        panel = panels[variant]
        features = panel.attrs.get("model_feature_columns", [])
        columns = sorted(set(["date", "code", *features,
            *[column for column in panel if column.startswith("style_")]]))
        frozen = panel.loc[panel["date"].le(last), columns].sort_values(["date", "code"]).reset_index(drop=True)
        endpoint = "label_end_20"
        if endpoint in panel:
            mature = panel.loc[panel["date"].le(last), ["date", "code", endpoint, "target_20", "forward_return_20"]].copy()
            mature.loc[~mature[endpoint].le(last), [endpoint, "target_20", "forward_return_20"]] = None
            frozen = frozen.merge(mature, on=["date", "code"], validate="one_to_one")
        digest.update(json.dumps({"variant": variant, "columns": list(frozen.columns)}, sort_keys=True).encode())
        digest.update(pd.util.hash_pandas_object(frozen, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def experiment_panels(
    histories: dict[str, pd.DataFrame], benchmark: pd.DataFrame, valuation: pd.DataFrame,
    calibration_end: str, exposures: pd.DataFrame | None = None, extra_factors: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """四组使用相同基础面板与覆盖交集，防止以换股票或换区间制造改善。"""
    base = build_market_panel(histories, benchmark, exposures=exposures, extra_factors=extra_factors,
                              style_options={"size": False, "value": False})
    panels = {}
    required = []
    for variant, (size, value) in VARIANTS.items():
        frame, report = attach_style_factors(base, valuation, size=size, value=value,
                                             research=True, calibration_end=calibration_end)
        frame.attrs = {**base.attrs, "style_factors": report,
                       "model_feature_columns": [*base.attrs["model_feature_columns"], *report["model_feature_columns"]]}
        panels[variant] = frame
        required.extend(report["model_feature_columns"])
    comparison = panels["size_value"]
    # 基础组也保留相同的点时暴露用于诊断，但不把这些字段放入其模型特征。
    metadata = [column for column in comparison if column.startswith("style_")]
    if metadata:
        attrs = panels["base"].attrs.copy()
        panels["base"] = panels["base"].merge(
            comparison[["date", "code", *metadata]], on=["date", "code"], validate="one_to_one")
        panels["base"].attrs = attrs
    required = sorted(set([*base.attrs["model_feature_columns"], *required]))
    valid = comparison[required].notna().all(axis=1)
    keys = set(zip(comparison.loc[valid, "date"], comparison.loc[valid, "code"]))
    for variant, frame in panels.items():
        panels[variant] = frame.loc[[key in keys for key in zip(frame["date"], frame["code"])]].copy()
    return panels


def _portfolio_metrics(evaluation: pd.DataFrame, panel: pd.DataFrame, horizon: int) -> dict[str, Any]:
    """用非重叠持仓期、实际标签和统一费用诊断组合及风格暴露。"""
    buy, sell = cost_rates()
    previous: set[str] = set()
    returns, excesses, turnovers, caps = [], [], [], []
    industries: dict[str, list[float]] = {}
    industry_ready = {"style_industry", "style_industry_pit"}.issubset(panel)
    industry_coverage = []
    calendar = sorted(panel["date"].unique())
    sampled = calendar[::horizon]
    for date in sampled:
        group = evaluation.loc[evaluation["date"].eq(date)].sort_values(["prediction", "code"], ascending=[False, True])
        selected = group.head(max(1, len(group) // 5))
        if selected.empty:
            continue
        members = set(selected["code"])
        if previous:
            buy_turnover = len(members - previous) / len(members)
            sell_turnover = len(previous - members) / len(previous)
            fee = buy_turnover * buy + sell_turnover * sell
        else:
            buy_turnover, sell_turnover = 1.0, 0.0
            fee = buy
        turnover = (buy_turnover + sell_turnover) / 2
        returns.append(float(selected["absolute_return"].mean()) - fee)
        excesses.append(float(selected["actual"].mean()) - fee)
        turnovers.append(turnover)
        exposure = panel.loc[panel["date"].eq(date) & panel["code"].isin(members)]
        if "style_market_cap_total" in exposure:
            cap = pd.to_numeric(exposure["style_market_cap_total"], errors="coerce")
            if cap.gt(0).all() and cap.notna().all() and len(cap):
                caps.append(float(np.log(cap).mean()))
        if industry_ready:
            known = exposure["style_industry_pit"].eq(True) & exposure["style_industry"].notna()
            industry_coverage.append(float(known.mean()) if len(known) else 0.0)
            weights = exposure.loc[known, "style_industry"].value_counts(normalize=True)
            for industry, weight in weights.items():
                industries.setdefault(str(industry), []).append(float(weight))
        previous = members
    if returns:
        returns[-1] -= sell
        excesses[-1] -= sell
    wealth = np.cumprod([1.0, *[1 + value for value in returns]])
    drawdown = wealth / np.maximum.accumulate(wealth) - 1
    return {"periods": len(returns), "after_cost_excess_return": float(np.mean(excesses)) if excesses else None,
            "net_total_return": float(wealth[-1] - 1), "maximum_drawdown": float(drawdown.min()),
            "average_turnover": float(np.mean(turnovers)) if turnovers else None,
            "average_log_market_cap": float(np.mean(caps)) if caps else None,
            "industry_weights": {name: float(sum(weights) / len(returns)) for name, weights in industries.items()},
            "industry_pit_coverage": min(industry_coverage) if industry_coverage else 0.0,
            "cost_rates": {"buy": buy, "sell": sell},
            "unit": f"每 {horizon} 交易日非重叠期；首期含费用；诊断未模拟涨跌停成交限制"}


def evaluate_variant(panel: pd.DataFrame, start: str, end: str, objective: str, horizon: int = 20,
                     model_name: str = "LightGBM", model_factory: Any = None,
                     training_cutoff: str | None = None) -> dict[str, Any]:
    """冻结起点之前模型，标签结束日不得越过区间终点，保留原始标签评估。"""
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    if first >= last:
        raise ValueError("验证起点必须早于终点。")
    cutoff = pd.Timestamp(training_cutoff) if training_cutoff else first
    if cutoff > first:
        raise ValueError("冻结训练截点不能晚于评估起点。")
    model = fit_panel_model(panel, model_name, horizon, model_factory=model_factory, cutoff=cutoff,
                            objective=objective, window=settings.TRAINING_WINDOW_DAYS)
    report = {"status": model["status"], "features_used": model.get("features_used", []),
              "usable": False, "metrics": None, "portfolio": None,
              "style_factors": panel.attrs.get("style_factors"), "objective": objective}
    if not model.get("validated"):
        return report
    target, endpoint = f"target_{horizon}", f"label_end_{horizon}"
    subset = panel.loc[panel["date"].between(first, last) & panel[endpoint].le(last)]
    predictions = predict_panel(model, subset)
    evaluation = predictions[["date", "code", "prediction"]].merge(
        subset[["date", "code", target, f"forward_return_{horizon}"]], on=["date", "code"], validate="one_to_one")
    train = panel.loc[panel["date"].lt(cutoff) & panel[endpoint].lt(cutoff)].dropna(subset=[*model["features_used"], target])
    dates = sorted(train["date"].unique())
    if settings.TRAINING_WINDOW_DAYS > 0:
        train = train.loc[train["date"].isin(dates[-settings.TRAINING_WINDOW_DAYS:])]
    if evaluation.empty or train.empty:
        report["status"] = "区间内没有完整标签或截面不足"
        return report
    evaluation = evaluation.rename(columns={target: "actual", f"forward_return_{horizon}": "absolute_return"})
    evaluation["baseline"] = float(train[target].median())
    metrics = evaluate_oos(evaluation, horizon, total_stocks=panel["code"].nunique(), score_kind=model["score_kind"])
    # 排序分不能直接当收益算 MAE；仅用训练期拟合分数到收益映射，评估原始留出标签。
    scores = np.asarray(model["estimator"].predict(train[model["features_used"]]), dtype=float)
    if model.get("objective") == "regression":
        metrics["raw_label_mae"] = float((evaluation["prediction"] - evaluation["actual"]).abs().mean())
    elif np.std(scores) > 1e-12:
        slope, intercept = np.polyfit(scores, train[target].to_numpy(), 1)
        calibrated = evaluation["prediction"] * slope + intercept
        metrics["raw_label_mae"] = float((calibrated - evaluation["actual"]).abs().mean())
    else:
        metrics["raw_label_mae"] = None
    metrics["raw_label_baseline_mae"] = float((evaluation["baseline"] - evaluation["actual"]).abs().mean())
    metrics.pop("daily_groups", None)
    report.update(usable=True, metrics=metrics, portfolio=_portfolio_metrics(evaluation, subset, horizon))
    return report


def compare_variants(panels: dict[str, pd.DataFrame], start: str, end: str, objective: str | None = None,
                     model_factory: Any = None) -> dict[str, Any]:
    """固定规则评估四组，配置选择只使用验证期，返回可保存的对照报告。"""
    objective = objective or settings.DEFAULT_OBJECTIVE_MODE
    if set(panels) != set(VARIANTS):
        raise ValueError("对照必须同时包含基础、加市值、加估值和同时加入四组。")
    results = {name: evaluate_variant(panel, start, end, objective, model_factory=model_factory)
               for name, panel in panels.items()}
    candidates = []
    for name, report in results.items():
        required = (list(SIZE_COLUMNS) if VARIANTS[name][0] else []) + (list(VALUE_COLUMNS) if VARIANTS[name][1] else [])
        accepted = (report.get("style_factors") or {}).get("accepted_columns") or []
        report["all_requested_features_available"] = all(column in accepted for column in required)
        if name != "base" and report["usable"] and report["all_requested_features_available"] and report["metrics"]["validated"]:
            candidates.append(name)
    selected = max(candidates, key=lambda name: results[name]["metrics"]["rank_ic"]) if candidates else None
    report = {"protocol": settings.STYLE_FEATURE_PROTOCOL, "model_version": settings.MODEL_VERSION,
        "configuration_id": configuration_id(objective), "validation_start": start, "validation_end": end,
        "data_fingerprint": comparison_fingerprint(panels, end),
        "objective": objective, "selected_variant": selected, "results": results,
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "production_enabled": False}
    report["experiment_id"] = hashlib.sha256(json.dumps(report, sort_keys=True, default=str).encode()).hexdigest()
    return report


def independent_acceptance(
    panels: dict[str, pd.DataFrame], comparison: dict[str, Any], start: str, end: str,
    path: Path | None = None, model_factory: Any = None,
) -> dict[str, Any]:
    """仅评估验证期选定组合与基础组；同一最终区间只允许消费一次，失败保留结果。"""
    variant = comparison.get("selected_variant")
    if variant not in VARIANTS or variant == "base":
        raise ValueError("验证期未选出可用新增特征组合，不能进行独立验收。")
    if comparison.get("configuration_id") != configuration_id(comparison.get("objective")) or comparison.get("model_version") != settings.MODEL_VERSION:
        raise ValueError("对照配置或模型版本已变化，旧对照不能用于独立验收。")
    if pd.Timestamp(start) <= pd.Timestamp(comparison["validation_end"]) or pd.Timestamp(start) >= pd.Timestamp(end):
        raise ValueError("独立验收必须严格位于验证区间之后，且起点早于终点。")
    if comparison.get("data_fingerprint") != comparison_fingerprint(panels, comparison["validation_end"]):
        raise ValueError("对照期的股票池、特征或成熟标签已变化，不能沿用旧对照进行独立验收。")
    target = Path(path or settings.STYLE_APPROVAL_PATH)
    # 先排他预留区间标记；中断也视为已消费，避免反复检验同一区间。
    marker = target.parent / "style_holdouts" / f"{pd.Timestamp(start):%Y%m%d}_{pd.Timestamp(end):%Y%m%d}.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    for used in marker.parent.glob("*.json"):
        try:
            first, last = used.stem.split("_", 1)
            if pd.Timestamp(first) <= pd.Timestamp(end) and pd.Timestamp(last) >= pd.Timestamp(start):
                raise ValueError("独立区间与已消费区间重叠，不允许通过移动端点重复检验。")
        except (OSError, TypeError):
            LOGGER.exception("无法校验已消费独立区间 path=%s", used)
            raise
    try:
        with marker.open("x", encoding="utf-8") as handle:
            json.dump({"experiment_id": comparison["experiment_id"], "status": "评估中"}, handle, ensure_ascii=False)
    except FileExistsError as exc:
        raise ValueError("该独立区间已经被消费，不允许重复验收或调参。") from exc
    # 即使评估抛错或进程中断，也不让上次通过的文件继续充当本次验收结论。
    _atomic_json(target, {"passed": False, "variant": variant, "status": "独立验收评估中或中断",
                          "experiment_id": comparison["experiment_id"]})
    # 独立阶段的冻结训练始终止于原验证起点，不能把配置选择期重新作为训练样本。
    results = {name: evaluate_variant(panels[name], start, end, comparison["objective"], model_factory=model_factory,
                                      training_cutoff=comparison.get("validation_start", start))
               for name in ("base", variant)}
    baseline, candidate = results["base"], results[variant]
    gates = {"both_usable": bool(baseline["usable"] and candidate["usable"])}
    if gates["both_usable"]:
        base, metrics, portfolio = baseline["metrics"], candidate["metrics"], candidate["portfolio"]
        gates.update(
            statistical_validation=bool(metrics["validated"]),
            raw_label_mae=metrics["raw_label_mae"] is not None and metrics["raw_label_mae"] < metrics["raw_label_baseline_mae"],
            rank_ic=metrics["rank_ic"] is not None and base["rank_ic"] is not None and metrics["rank_ic"] >= base["rank_ic"],
            net_excess=portfolio["after_cost_excess_return"] is not None and baseline["portfolio"]["after_cost_excess_return"] is not None
                and portfolio["after_cost_excess_return"] > max(0, baseline["portfolio"]["after_cost_excess_return"]),
            drawdown=portfolio["maximum_drawdown"] >= baseline["portfolio"]["maximum_drawdown"],
            industry_exposure=portfolio["industry_pit_coverage"] >= settings.STYLE_MINIMUM_COVERAGE,
            size_exposure=portfolio["average_log_market_cap"] is not None,
        )
    required = [f"xs_{column}" for column in (*SIZE_COLUMNS, *VALUE_COLUMNS)
                if (column in SIZE_COLUMNS and VARIANTS[variant][0]) or (column in VALUE_COLUMNS and VARIANTS[variant][1])]
    gates["actual_model_features"] = all(column in candidate["features_used"] for column in required)
    report = {"passed": all(gates.values()), "gates": gates, "results": results,
        "variant": variant, "protocol": settings.STYLE_FEATURE_PROTOCOL, "model_version": settings.MODEL_VERSION,
            "configuration_id": configuration_id(comparison.get("objective")), "experiment_id": comparison["experiment_id"],
        "holdout_start": start, "holdout_end": end, "validation_end": comparison["validation_end"],
        "evaluated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}
    _atomic_json(marker, report)
    _atomic_json(target, report)
    LOGGER.info("显式风格独立验收完成 passed=%s variant=%s", report["passed"], variant)
    return report
