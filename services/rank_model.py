"""Learning to Rank：把选股定义为每日截面内的优胜劣汰排序任务。

回归损失会把“当日全市场都涨”学成看多信号，排序目标只在同一交易日的股票之间比较，
配合 NDCG 与 Rank IC 检验，避免树模型产生正向偏置。
"""

from __future__ import annotations

import logging
import math
from typing import Any, Sequence

import numpy as np
import pandas as pd

from config import settings

LOGGER = logging.getLogger("quant.rank_model")
SCORE_KIND_RANK = "rank_score"
SCORE_KIND_RETURN = "excess_return"


def relevance_grades(target: pd.Series, dates: pd.Series, grade_count: int | None = None) -> pd.Series:
    """把每日截面内的超额收益转换为整数相关度等级，收益越高等级越高。"""
    grade_count = int(grade_count or settings.RANK_GRADE_COUNT)
    if grade_count < 2:
        raise ValueError("相关度等级至少为 2 级。")
    values = pd.to_numeric(target, errors="coerce")
    percentile = values.groupby(dates).rank(pct=True, method="average")
    grades = np.ceil(percentile * grade_count) - 1
    return grades.clip(lower=0, upper=grade_count - 1).astype("float64")


def ndcg_at_k(grades: np.ndarray, scores: np.ndarray, k: int) -> float:
    """按指数增益计算 NDCG@k；理想排序由同一截面的等级分布决定。"""
    grades = np.asarray(grades, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if len(grades) < 2 or k < 1:
        return float("nan")
    gains = np.power(2.0, grades) - 1.0
    order = np.argsort(-scores, kind="stable")[:k]
    discounts = 1.0 / np.log2(np.arange(2, len(order) + 2))
    ideal_order = np.argsort(-gains, kind="stable")[:k]
    ideal_discounts = 1.0 / np.log2(np.arange(2, len(ideal_order) + 2))
    ideal = float(np.sum(gains[ideal_order] * ideal_discounts))
    if ideal <= 1e-12:
        return float("nan")
    return float(np.sum(gains[order] * discounts) / ideal)


def random_ndcg_baseline(grades: np.ndarray, k: int, repeats: int = 20, seed: int = 42) -> float:
    """固定随机种子下的随机排序 NDCG 均值，作为“没有排序能力”的参照。"""
    generator = np.random.default_rng(seed)
    scores = np.zeros(len(grades), dtype=float)
    values = []
    for _ in range(max(1, repeats)):
        generator.shuffle(scores)
        value = ndcg_at_k(grades, scores, k)
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


class CrossSectionRanker:
    """把 LightGBM/XGBoost 排序器适配为与回归器一致的 fit/predict 接口。"""

    score_kind = SCORE_KIND_RANK

    def __init__(self, name: str, grade_count: int | None = None, seed: int = 42) -> None:
        self.name = name
        self.grade_count = int(grade_count or settings.RANK_GRADE_COUNT)
        self.seed = seed
        self.estimator: Any = None
        self.groups_trained = 0
        self.samples_trained = 0

    def _build(self) -> Any:
        if self.name == "LightGBM":
            from lightgbm import LGBMRanker

            return LGBMRanker(
                objective="lambdarank",
                n_estimators=120,
                learning_rate=0.04,
                max_depth=4,
                num_leaves=15,
                min_child_samples=20,
                reg_lambda=1.0,
                random_state=self.seed,
                n_jobs=1,
                verbosity=-1,
                label_gain=[float(2**index - 1) for index in range(self.grade_count)],
            )
        from xgboost import XGBRanker

        return XGBRanker(
            objective="rank:ndcg",
            n_estimators=120,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=self.seed,
            n_jobs=1,
            verbosity=0,
        )

    def fit_cross_section(self, features: pd.DataFrame, target: pd.Series, dates: pd.Series) -> "CrossSectionRanker":
        """按交易日排序后以连续分组训练排序模型，缺失标签的样本不参与。"""
        frame = pd.DataFrame(
            {"target": pd.to_numeric(target, errors="coerce").to_numpy(), "date": pd.to_datetime(dates).to_numpy()},
            index=features.index,
        )
        frame["grade"] = relevance_grades(frame["target"], frame["date"], self.grade_count)
        frame = frame.dropna(subset=["grade"])
        if frame.empty:
            raise ValueError("排序训练没有有效的相关度标签。")
        order = frame.sort_values(["date"], kind="stable").index
        ordered = frame.loc[order]
        matrix = features.loc[order]
        group_sizes = ordered.groupby("date", sort=True).size().to_numpy()
        self.groups_trained = int(len(group_sizes))
        self.samples_trained = int(len(ordered))
        self.estimator = self._build()
        labels = ordered["grade"].to_numpy(dtype=float)
        if self.name == "LightGBM":
            self.estimator.fit(matrix, labels, group=group_sizes)
        else:
            qid = np.repeat(np.arange(len(group_sizes)), group_sizes)
            self.estimator.fit(matrix, labels, qid=qid)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.estimator is None:
            raise ValueError("排序模型尚未训练。")
        return np.asarray(self.estimator.predict(features), dtype=float)

    @property
    def feature_importances_(self) -> np.ndarray:
        importance = getattr(self.estimator, "feature_importances_", None)
        if importance is None:
            return np.zeros(0, dtype=float)
        return np.asarray(importance, dtype=float)


def make_ranker(name: str, grade_count: int | None = None, seed: int = 42) -> CrossSectionRanker:
    return CrossSectionRanker(name, grade_count=grade_count, seed=seed)


def evaluate_ndcg(frame: pd.DataFrame, k: int | None = None, score_column: str = "prediction", actual_column: str = "actual") -> dict[str, float | None]:
    """按日计算预测排序的 NDCG@k 与随机基线，用于排序模型的偏置检验。"""
    clean = frame.dropna(subset=[score_column, actual_column])
    if clean.empty:
        return {"ndcg": None, "ndcg_random_baseline": None, "ndcg_days": 0}
    values = []
    baselines = []
    for _, group in clean.groupby("date", sort=True):
        if len(group) < 2:
            continue
        top = int(k or max(1, math.ceil(len(group) * settings.NDCG_TOP_FRACTION)))
        grades = relevance_grades(group[actual_column], pd.Series(["x"] * len(group), index=group.index)).to_numpy()
        scores = group[score_column].to_numpy(dtype=float)
        value = ndcg_at_k(grades, scores, top)
        if np.isfinite(value):
            values.append(value)
            baselines.append(random_ndcg_baseline(grades, top))
    if not values:
        return {"ndcg": None, "ndcg_random_baseline": None, "ndcg_days": 0}
    return {
        "ndcg": float(np.mean(values)),
        "ndcg_random_baseline": float(np.nanmean(baselines)),
        "ndcg_days": len(values),
    }


def objective_label(objective: str) -> str:
    """把目标函数标识转换为页面可读名称。"""
    return {
        SCORE_KIND_RANK: "排序目标（LambdaMART / NDCG）",
        SCORE_KIND_RETURN: "回归目标（MSE / MAE）",
        "lambdarank": "排序目标（LambdaMART / NDCG）",
        "regression": "回归目标（MSE / MAE）",
        "auto": "自动择优（同时训练排序与回归目标）",
    }.get(str(objective), str(objective))


def summarize_importances(features: Sequence[str], importance: np.ndarray) -> list[dict[str, Any]]:
    """把特征重要性归一化后倒序输出，供页面展示。"""
    values = np.asarray(importance, dtype=float)
    total = float(values.sum()) or 1.0
    return sorted(
        [{"feature": feature, "importance": float(value / total)} for feature, value in zip(features, values)],
        key=lambda item: item["importance"],
        reverse=True,
    )
