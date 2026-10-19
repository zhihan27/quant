"""生产扫描、调仓清单与组合回测共用的模型排序口径。"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import pandas as pd


TARGET_KIND = "excess_return_vs_csi300"


def mean_percentile(values: Iterable[Any]) -> float | None:
    """返回有限且位于 0 到 1 之间的百分位均值。"""
    scores = []
    for value in values:
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(score) and 0 <= score <= 1:
            scores.append(score)
    return sum(scores) / len(scores) if scores else None


def validated_model_percentile(models: Sequence[dict[str, Any]]) -> float | None:
    """只融合通过验证且目标口径一致的模型截面百分位。"""
    values = [
        model.get("prediction_percentile")
        for model in models
        if model.get("validated") and model.get("target_kind") == TARGET_KIND
    ]
    return mean_percentile(values)


def fuse_prediction_percentiles(
    prediction_frames: dict[str, pd.DataFrame], model_names: Sequence[str],
) -> pd.DataFrame:
    """按交易日与代码外连接模型百分位，并对可用模型取均值。"""
    merged: pd.DataFrame | None = None
    available: list[str] = []
    for name in model_names:
        frame = prediction_frames.get(name)
        if frame is None or frame.empty or "prediction_percentile" not in frame:
            continue
        values = frame[["date", "code", "prediction_percentile"]].rename(
            columns={"prediction_percentile": name}
        )
        merged = values if merged is None else merged.merge(
            values, on=["date", "code"], how="outer", validate="one_to_one"
        )
        available.append(name)
    if merged is None:
        return pd.DataFrame(columns=["date", "code", "score", *model_names])
    for name in available:
        merged[name] = pd.to_numeric(merged[name], errors="coerce")
        merged.loc[~merged[name].between(0, 1), name] = float("nan")
    merged["score"] = merged[available].mean(axis=1) if available else float("nan")
    return merged.reindex(columns=["date", "code", "score", *model_names])
