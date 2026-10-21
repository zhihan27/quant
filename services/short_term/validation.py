"""按信号日期滚动验证短线排序器，并与同口径基线比较。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config.short_term import SHORT_TERM_FEATURES, SHORT_TERM_TEST_WINDOW_DAYS, SHORT_TERM_TOP_K, SHORT_TERM_TRAIN_WINDOW_DAYS
from services.rank_model import make_ranker
from services.short_term.point_in_time import filter_by_point_in_time_membership


def _mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def walk_forward_validation(
    panel: pd.DataFrame,
    horizon: int,
    train_days: int = SHORT_TERM_TRAIN_WINDOW_DAYS,
    test_days: int = SHORT_TERM_TEST_WINDOW_DAYS,
    top_k: int = SHORT_TERM_TOP_K,
    include_oos_signals: bool = False,
    membership_history: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """按完整日期切分，并仅用测试开始前已成熟的标签训练。"""
    required = {"date", "label_end_date", "target_net_return", "return_5d", *SHORT_TERM_FEATURES}
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"滚动验证缺少字段：{'、'.join(missing)}。")
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["label_end_date"] = pd.to_datetime(frame["label_end_date"], errors="coerce").dt.normalize()
    frame["target_net_return"] = pd.to_numeric(frame["target_net_return"], errors="coerce")
    frame["return_5d"] = pd.to_numeric(frame["return_5d"], errors="coerce")
    frame = frame.dropna(subset=["date", "label_end_date", "target_net_return"])
    pit_filter = {
        "status": "not_applied",
        "rows": int(len(frame)),
        "covered_rows": 0,
        "member_rows": 0,
        "missing_rows": int(len(frame)),
        "coverage_ratio": 0.0,
    }
    if membership_history is not None:
        frame, pit_filter = filter_by_point_in_time_membership(frame, membership_history)
    if int(train_days) < 1 or int(test_days) < 1 or int(top_k) < 1:
        raise ValueError("滚动窗口和 Top K 必须为正整数。")
    dates = sorted(frame["date"].drop_duplicates())
    fold_results = []
    prediction_frames = []
    for fold_number, start in enumerate(range(int(train_days), len(dates) - int(test_days) + 1, int(test_days)), start=1):
        test_dates = dates[start : start + int(test_days)]
        test_start = test_dates[0]
        train_dates = dates[max(0, start - int(train_days)) : start]
        train = frame.loc[
            frame["date"].isin(train_dates)
            & (frame["date"] < test_start)
            & (frame["label_end_date"] < test_start)
        ].copy()
        test = frame.loc[frame["date"].isin(test_dates)].copy()
        train = train.dropna(subset=list(SHORT_TERM_FEATURES))
        test = test.dropna(subset=["return_5d", *SHORT_TERM_FEATURES])
        train_day_count = int(train["date"].nunique())
        if train_day_count < max(20, int(train_days) * 0.8) or train["code"].nunique() < 5 or test.empty:
            continue
        ranker = make_ranker("LightGBM")
        ranker.fit_cross_section(train[list(SHORT_TERM_FEATURES)], train["target_net_return"], train["date"])
        test["ranking_score"] = ranker.predict(test[list(SHORT_TERM_FEATURES)])
        test["fold"] = fold_number
        prediction_frames.append(test[["code", "date", "label_end_date", "target_net_return", "ranking_score", "return_5d", "fold"]])
        model_returns = []
        momentum_returns = []
        universe_returns = []
        rank_ics = []
        scored_days = 0
        for _, group in test.groupby("date", sort=True):
            group = group.replace([np.inf, -np.inf], np.nan).dropna(subset=["ranking_score", "target_net_return", "return_5d"])
            if len(group) < 5:
                continue
            count = min(int(top_k), len(group))
            model_returns.append(float(group.nlargest(count, "ranking_score")["target_net_return"].mean()))
            momentum_returns.append(float(group.nlargest(count, "return_5d")["target_net_return"].mean()))
            universe_returns.append(float(group["target_net_return"].mean()))
            rank_ic = group["ranking_score"].rank(method="average").corr(
                group["target_net_return"].rank(method="average"), method="pearson"
            )
            if pd.notna(rank_ic):
                rank_ics.append(float(rank_ic))
            scored_days += 1
        equity = (1.0 + pd.Series(model_returns, dtype=float)).cumprod()
        fold_drawdown = float((equity / equity.cummax() - 1.0).min()) if not equity.empty else None
        fold_results.append(
            {
                "fold": fold_number,
                "test_start": pd.Timestamp(test_dates[0]).date().isoformat(),
                "test_end": pd.Timestamp(test_dates[-1]).date().isoformat(),
                "train_start": pd.Timestamp(train["date"].min()).date().isoformat(),
                "train_end": pd.Timestamp(train["date"].max()).date().isoformat(),
                "train_rows": int(len(train)),
                "train_days": train_day_count,
                "test_rows": int(len(test)),
                "scored_days": scored_days,
                "top_k_mean_net_return": _mean(model_returns),
                "momentum_top_k_mean_net_return": _mean(momentum_returns),
                "universe_mean_net_return": _mean(universe_returns),
                "mean_daily_rank_ic": _mean(rank_ics),
                "max_drawdown": fold_drawdown,
            }
        )
    if not fold_results:
        return {
            "status": "insufficient_data",
            "horizon": int(horizon),
            "fold_count": 0,
            "message": "历史不足以形成符合标签成熟隔离条件的滚动测试窗口。",
            "folds": [],
            "point_in_time_filter": pit_filter,
        }
    oos = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    result = {
        "status": "measured_research_only",
        "horizon": int(horizon),
        "train_window_days": int(train_days),
        "test_window_days": int(test_days),
        "top_k": int(top_k),
        "fold_count": len(fold_results),
        "oos_rows": int(len(oos)),
        "top_k_mean_net_return": _mean([row["top_k_mean_net_return"] for row in fold_results if row["top_k_mean_net_return"] is not None]),
        "momentum_top_k_mean_net_return": _mean([row["momentum_top_k_mean_net_return"] for row in fold_results if row["momentum_top_k_mean_net_return"] is not None]),
        "universe_mean_net_return": _mean([row["universe_mean_net_return"] for row in fold_results if row["universe_mean_net_return"] is not None]),
        "mean_daily_rank_ic": _mean([row["mean_daily_rank_ic"] for row in fold_results if row["mean_daily_rank_ic"] is not None]),
        "max_drawdown": min((row["max_drawdown"] for row in fold_results if row["max_drawdown"] is not None), default=None),
        "positive_fold_ratio": float(sum((row["top_k_mean_net_return"] or 0.0) > 0 for row in fold_results) / len(fold_results)),
        "scored_days": int(sum(row["scored_days"] for row in fold_results)),
        "folds": fold_results,
        "point_in_time_filter": pit_filter,
    }
    if include_oos_signals:
        result["_out_of_sample_signals"] = [
            {
                "code": str(row.code),
                "date": pd.Timestamp(row.date).date().isoformat(),
                "ranking_score": float(row.ranking_score),
                "fold": int(row.fold),
            }
            for row in oos[["code", "date", "ranking_score", "fold"]].itertuples(index=False)
        ]
    return result
