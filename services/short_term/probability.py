"""短线概率模块的时间隔离校准诊断。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from config.short_term import SHORT_TERM_FEATURES, SHORT_TERM_TEST_WINDOW_DAYS, SHORT_TERM_TRAIN_WINDOW_DAYS


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def walk_forward_probability_validation(
    panel: pd.DataFrame,
    horizon: int,
    return_buffer: float = 0.0,
    train_days: int = SHORT_TERM_TRAIN_WINDOW_DAYS,
    test_days: int = SHORT_TERM_TEST_WINDOW_DAYS,
    calibration_fraction: float = 0.2,
) -> dict[str, Any]:
    """按训练→sigmoid校准→外层测试顺序评估价格代理事件概率。"""
    required = {"date", "label_end_date", "target_net_return", *SHORT_TERM_FEATURES}
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"概率验证缺少字段：{', '.join(missing)}。")
    if not 0.1 <= float(calibration_fraction) <= 0.4:
        raise ValueError("校准段比例必须在 10% 至 40% 之间。")
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        return {"status": "unavailable", "reason": "LightGBM 分类器未安装", "horizon": int(horizon)}

    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["label_end_date"] = pd.to_datetime(frame["label_end_date"], errors="coerce").dt.normalize()
    frame["target_net_return"] = pd.to_numeric(frame["target_net_return"], errors="coerce")
    frame = frame.dropna(subset=["date", "label_end_date", "target_net_return", *SHORT_TERM_FEATURES])
    dates = sorted(frame["date"].drop_duplicates())
    fold_metrics = []
    reliability = []
    for fold_number, start in enumerate(range(int(train_days), len(dates) - int(test_days) + 1, int(test_days)), start=1):
        test_dates = dates[start : start + int(test_days)]
        test_start = test_dates[0]
        train_dates = dates[max(0, start - int(train_days)) : start]
        calibration_count = max(1, int(len(train_dates) * float(calibration_fraction)))
        calibration_dates = train_dates[-calibration_count:]
        calibration_start = calibration_dates[0]
        train = frame.loc[
            frame["date"].isin(train_dates)
            & (frame["date"] < calibration_start)
            & (frame["label_end_date"] < calibration_start)
        ].copy()
        calibration = frame.loc[
            frame["date"].isin(calibration_dates) & (frame["label_end_date"] < test_start)
        ].copy()
        test = frame.loc[frame["date"].isin(test_dates)].copy()
        if len(train_dates) < 2 or train["date"].nunique() < 2 or calibration.empty or test.empty:
            continue
        train_y = (train["target_net_return"] > float(return_buffer)).astype(int)
        calibration_y = (calibration["target_net_return"] > float(return_buffer)).astype(int)
        test_y = (test["target_net_return"] > float(return_buffer)).astype(int)
        if train_y.nunique() < 2 or calibration_y.nunique() < 2 or test_y.nunique() < 2:
            continue
        classifier = LGBMClassifier(
            objective="binary",
            n_estimators=120,
            learning_rate=0.04,
            max_depth=4,
            num_leaves=15,
            min_child_samples=20,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=1,
            verbosity=-1,
        )
        classifier.fit(train[list(SHORT_TERM_FEATURES)], train_y)
        calibration_raw = classifier.predict_proba(calibration[list(SHORT_TERM_FEATURES)])[:, 1]
        calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
        calibrator.fit(_logit(calibration_raw), calibration_y)
        test_raw = classifier.predict_proba(test[list(SHORT_TERM_FEATURES)])[:, 1]
        test_probability = calibrator.predict_proba(_logit(test_raw))[:, 1]
        fold_metrics.append(
            {
                "fold": fold_number,
                "train_start": pd.Timestamp(train["date"].min()).date().isoformat(),
                "train_end": pd.Timestamp(train["date"].max()).date().isoformat(),
                "calibration_start": pd.Timestamp(calibration["date"].min()).date().isoformat(),
                "calibration_end": pd.Timestamp(calibration["date"].max()).date().isoformat(),
                "test_start": pd.Timestamp(test["date"].min()).date().isoformat(),
                "test_end": pd.Timestamp(test["date"].max()).date().isoformat(),
                "calibration_rows": int(len(calibration)),
                "test_rows": int(len(test)),
                "brier_score": float(brier_score_loss(test_y, test_probability)),
                "log_loss": float(log_loss(test_y, np.clip(test_probability, 1e-6, 1 - 1e-6), labels=[0, 1])),
            }
        )
        bins = pd.DataFrame({"probability": test_probability, "actual": test_y.to_numpy()})
        bins["bin"] = pd.cut(bins["probability"], bins=np.linspace(0, 1, 11), include_lowest=True, duplicates="drop")
        for bin_key, group in bins.groupby("bin", observed=True):
            reliability.append(
                {
                    "fold": fold_number,
                    "probability_bin": str(bin_key),
                    "mean_probability": float(group["probability"].mean()),
                    "observed_positive_rate": float(group["actual"].mean()),
                    "samples": int(len(group)),
                }
            )
    if not fold_metrics:
        return {
            "status": "insufficient_data",
            "horizon": int(horizon),
            "label_basis": "qfq_open_price_proxy_no_execution_state",
            "reason": "时间隔离后没有同时包含正负样本的训练、校准和外层测试折",
            "folds": [],
            "reliability": [],
        }
    return {
        "status": "proxy_only_not_trade_probability",
        "horizon": int(horizon),
        "return_buffer": float(return_buffer),
        "train_window_days": int(train_days),
        "test_window_days": int(test_days),
        "calibration_fraction": float(calibration_fraction),
        "label_basis": "qfq_open_price_proxy_no_execution_state",
        "fold_count": len(fold_metrics),
        "mean_brier_score": float(np.mean([row["brier_score"] for row in fold_metrics])),
        "mean_log_loss": float(np.mean([row["log_loss"] for row in fold_metrics])),
        "folds": fold_metrics,
        "reliability": reliability,
    }
