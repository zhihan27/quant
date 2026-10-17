"""多因子观察、共享截面模型读取与滚动历史预测。"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from config import settings
from services.feature_engineering import build_features
from services.timing_strategy import plan_from_features

FACTOR_WEIGHTS = {
    "momentum_5": 0.10,
    "momentum_20": 0.18,
    "momentum_60": 0.14,
    "trend_strength": 0.18,
    "relative_strength_20": 0.15,
    "volatility_20": 0.10,
    "atr_pct": 0.08,
    "volume_change_20": 0.04,
    "price_volume_corr_20": 0.03,
}
INVERTED_FACTORS = {"volatility_20", "atr_pct"}


def _factor_score(features: pd.DataFrame) -> tuple[float | None, list[dict[str, Any]]]:
    if features.empty:
        return None, []
    latest = features.iloc[-1]
    weighted_score = 0.0
    total_weight = 0.0
    details = []
    for name, weight in FACTOR_WEIGHTS.items():
        values = pd.to_numeric(features[name], errors="coerce").dropna().tail(252)
        raw = latest.get(name)
        if len(values) < 30 or pd.isna(raw):
            continue
        standard_deviation = float(values.std(ddof=0))
        z_score = (float(raw) - float(values.mean())) / standard_deviation if standard_deviation > 1e-12 else 0.0
        if name in INVERTED_FACTORS:
            z_score *= -1
        subscore = 50 + 50 * math.tanh(z_score / 2)
        weighted_score += subscore * weight
        total_weight += weight
        details.append(
            {
                "factor": name,
                "value": float(raw),
                "z_score": z_score,
                "score": subscore,
                "weight": weight,
            }
        )
    if total_weight == 0:
        return None, details
    return weighted_score / total_weight, details


def _make_model(name: str) -> Any:
    if name == "LightGBM":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=120,
            learning_rate=0.04,
            max_depth=4,
            num_leaves=15,
            min_child_samples=20,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=settings.TRAINING_N_JOBS,
            verbosity=-1,
        )
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=settings.TRAINING_N_JOBS,
        verbosity=0,
    )


def walk_forward_predictions(
    features: pd.DataFrame,
    start: int,
    horizon: int = 20,
    retrain_every: int = 20,
    code: str | None = None,
    market_panel: pd.DataFrame | None = None,
) -> tuple[pd.Series, list[dict[str, Any]]]:
    """在每个历史区间重新训练截面模型，只使用当时已成熟的超额标签。"""
    from services.ml_pipeline import fit_panel_model, load_context, predict_panel, relative_signal

    if retrain_every < 1 or horizon not in (5, 20, 60):
        raise ValueError("重新训练间隔或预测周期无效。")
    predictions = pd.Series(np.nan, index=features.index, dtype=float)
    if market_panel is None:
        _, market_panel = load_context(include_panel=True)
    if market_panel is None or code is None or str(code) not in set(market_panel["code"]):
        return predictions, [{"status": "缺少该股票历史截面上下文，模型回测未产生交易信号"}]
    training_log = []
    for block_start in range(start, len(features), retrain_every):
        cutoff = pd.Timestamp(features.iloc[block_start]["date"])
        block_end = min(block_start + retrain_every, len(features))
        end_date = pd.Timestamp(features.iloc[block_end - 1]["date"])
        # 每个区间的训练与验证都限制在历史截止点，禁止复用最终全样本模型。
        fitted = [fit_panel_model(market_panel, name, horizon, cutoff=cutoff) for name in ("LightGBM", "XGBoost")]
        live = market_panel.loc[market_panel["date"].between(cutoff, end_date)]
        values = []
        for model in fitted:
            predicted = predict_panel(model, live)
            selected = predicted.loc[predicted["code"].eq(str(code))]
            for row in selected.to_dict("records"):
                # 排序目标只看当日截面相对位置，回归目标额外要求预期超额为正。
                signal = relative_signal({**row, "validated": model.get("validated"),
                                          "score_kind": model.get("score_kind"),
                                          "predicted_excess_return": row["prediction"]})
                if signal is not None:
                    values.append({"date": row["date"], "signal": signal})
        if values:
            daily = pd.DataFrame(values).groupby("date")["signal"].mean()
            for index in range(block_start, block_end):
                predictions.loc[index] = daily.get(pd.Timestamp(features.loc[index, "date"]), np.nan)
        training_log.append({
            "prediction_start": str(cutoff.date()),
            "trained_through": str(market_panel.loc[market_panel["date"] < cutoff, "date"].max().date()),
            "target_kind": "excess_return_vs_csi300",
            "models": [{"model": item["model"], "objective": item.get("objective"), "score_kind": item.get("score_kind"),
                        "status": item["status"], "oos_metrics": item["oos_metrics"]} for item in fitted],
        })
    return predictions, training_log


def analyze_stock(
    history: pd.DataFrame,
    current_price: float | None,
    benchmark: pd.DataFrame | None = None,
    horizons: tuple[int, ...] = (5, 20, 60),
    code: str | None = None,
    as_of: str | None = None,
    quote_status: str | None = None,
    history_ready: bool = True,
    market_context: dict[str, Any] | None = None,
    scoring_paused: bool = False,
) -> dict[str, Any]:
    """供个股页和扫描页共同使用的分析入口。

    ``scoring_paused`` 由调用方从模型治理状态读出后传入，默认 False 保持既有行为不变。
    它是一个显式的入参而不是在这里查库：本函数在逐股循环里被调用成千上万次，
    每次都查一次降级状态没有意义；况且上一轮扫描之后才被降级这种情形，
    本来就该由调用方在扫描开始前统一读一次。
    """
    if any(horizon not in (5, 20, 60) for horizon in horizons):
        raise ValueError("预测周期仅支持5、20、60个交易日。")
    features = build_features(history, benchmark)
    factor_score, factor_details = _factor_score(features)
    latest = features.iloc[-1] if not features.empty else pd.Series(dtype=float)
    reference_price = float(latest["close"]) if pd.notna(latest.get("close")) else None
    from services.ml_pipeline import load_context, models_for_stock, relative_signal

    if market_context is None:
        market_context, _ = load_context()
    reference_date = str(latest.get("date", ""))[:10]
    model_results = {
        horizon: models_for_stock(market_context, code, reference_date, reference_price, horizon)
        for horizon in horizons
    }
    if not history_ready:
        from services.ml_pipeline import MODEL_NAMES, unavailable_model

        model_results = {horizon: [unavailable_model(name, horizon, "历史行情不完整或过期，拒绝输出模型预测") for name in MODEL_NAMES] for horizon in horizons}
    elif scoring_paused:
        # 降级闸门放在数据血缘校验之后：历史本来就不完整时，
        # 「历史不完整」比「模型降级」更贴近这只股票的真实处境，先说更具体的那一个。
        from services.ml_pipeline import MODEL_NAMES, unavailable_model

        model_results = {
            horizon: [unavailable_model(name, horizon, settings.DEGRADED_SCORING_MESSAGE) for name in MODEL_NAMES]
            for horizon in horizons
        }
    twenty_day = model_results.get(20, [])
    signals = [signal for signal in (relative_signal(item) for item in twenty_day) if signal is not None]
    prediction = float(np.mean(signals)) if signals else None
    plan = plan_from_features(
        row=latest,
        reference_price=reference_price,
        relative_signal=prediction,
        model_ready=prediction is not None,
        as_of=as_of,
        data_status=quote_status,
        history_ready=history_ready,
    )
    return {
        "features": features,
        "latest_features": latest.to_dict(),
        "factor_score": factor_score,
        "factor_details": factor_details,
        "models": model_results,
        "trade_plan": plan,
        "prediction_horizon": 20,
        "target_kind": "excess_return_vs_csi300",
        "market_context_id": market_context.get("context_id") if market_context else None,
        "cross_section_scope": market_context.get("limitation") if market_context else "尚未建立历史截面",
        "prediction_reference_price": reference_price,
        "prediction_reference_date": str(latest.get("date", ""))[:10],
        "fundamental_status": "未接入可验证的历史披露数据，基本面因子未参与评分",
    }

