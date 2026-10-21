"""短线排序模型拟合与独立归档。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from config.short_term import SHORT_TERM_FEATURES, SHORT_TERM_MODEL_DIR, SHORT_TERM_STRATEGY_VERSION, SHORT_TERM_TRAIN_WINDOW_DAYS
from services.rank_model import make_ranker


def fit_short_ranker(panel: pd.DataFrame) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    """用已成熟标签的最近训练窗口拟合短线排序器。"""
    required = {"date", "label_end_date", "target_net_return", *SHORT_TERM_FEATURES}
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"短线模型训练缺少字段：{'、'.join(missing)}。")
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["label_end_date"] = pd.to_datetime(frame["label_end_date"], errors="coerce").dt.normalize()
    latest_signal_date = frame["date"].max()
    matured = frame.loc[(frame["label_end_date"] <= latest_signal_date) & frame["target_net_return"].notna()].copy()
    matured = matured.dropna(subset=list(SHORT_TERM_FEATURES) + ["target_net_return"])
    available_dates = sorted(matured["date"].dropna().unique())
    if len(available_dates) < int(SHORT_TERM_TRAIN_WINDOW_DAYS):
        raise ValueError(f"成熟训练截面仅有 {len(available_dates)} 天，少于配置的 {SHORT_TERM_TRAIN_WINDOW_DAYS} 天。")
    selected_dates = set(available_dates[-int(SHORT_TERM_TRAIN_WINDOW_DAYS) :])
    train = matured.loc[matured["date"].isin(selected_dates)].copy()
    day_sizes = train.groupby("date")["code"].nunique()
    usable_dates = day_sizes[day_sizes >= 5].index
    train = train.loc[train["date"].isin(usable_dates)]
    if train["date"].nunique() < int(SHORT_TERM_TRAIN_WINDOW_DAYS) * 0.8:
        raise ValueError("成熟训练截面覆盖不足 80%，不能拟合短线模型。")
    ranker = make_ranker("LightGBM")
    ranker.fit_cross_section(train[list(SHORT_TERM_FEATURES)], train["target_net_return"], train["date"])
    current_date = pd.Timestamp(latest_signal_date)
    candidates = frame.loc[frame["date"].eq(current_date)].dropna(subset=list(SHORT_TERM_FEATURES)).copy()
    if candidates["code"].nunique() < 5:
        raise ValueError("最新有效截面不足 5 只股票，不能生成候选排名。")
    candidates["ranking_score"] = ranker.predict(candidates[list(SHORT_TERM_FEATURES)])
    candidates["rank"] = candidates["ranking_score"].rank(method="first", ascending=False).astype(int)
    candidates["percentile"] = candidates["ranking_score"].rank(pct=True, method="average")
    output = candidates.sort_values(["rank", "code"], kind="stable").reset_index(drop=True)
    metadata = {
        "strategy_version": SHORT_TERM_STRATEGY_VERSION,
        "feature_columns": list(SHORT_TERM_FEATURES),
        "training_rows": int(len(train)),
        "training_days": int(train["date"].nunique()),
        "training_codes": int(train["code"].nunique()),
        "signal_date": current_date.date().isoformat(),
        "score_semantics": "同一截面排序分数，不是上涨概率或目标价。",
    }
    return ranker, output, metadata


def save_short_ranker(ranker: Any, run_id: str) -> Path:
    """将研究模型写入独立版本目录，不覆盖活动或旧模型。"""
    path = SHORT_TERM_MODEL_DIR / SHORT_TERM_STRATEGY_VERSION / run_id / "model.joblib"
    path.parent.mkdir(parents=True, exist_ok=False)
    joblib.dump(ranker, path)
    return path
