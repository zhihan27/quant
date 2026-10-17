"""漂移监控：特征分布、概念（预测能力）与预测结构的随时间变化。

要解决的问题：市场风格切换导致模型失效时，系统原先照常出分。没有任何机制发现
「训练分布与当前截面已经对不上」，也没有降级与回滚手段。

三类漂移分别回答不同的问题：

* **特征漂移**（``feature_drift``）——喂给模型的输入变了没有。用 PSI 与两样本 KS 比较
  参考窗口与当前窗口的分布。
* **概念漂移**（``concept_drift``）——模型和真实涨跌的关系变了没有。滚动统计 Rank IC、
  ICIR、HAC t 与 Top 超额。
* **预测漂移**（``prediction_drift``）——模型吐出来的东西变了没有。每日 Top-N 组合的
  行业集中度、市值分位与分数分布。

不伪造结果是本模块的硬约束，有三处必须写明而不是给数：

1. 行情源未提供 PE/PB，**估值维度无数据源**，如实标注「未接入」而不是算一个假 PSI；
2. 本地股票池只有几十只，单日截面在 10 个分箱下绝大多数箱计数不足 5，PSI 会剧烈抖动、
   天天误报严重。因此当前分布取最近 N 个交易日**池化**，且有效样本数低于下限时
   明确判「样本不足」，不给等级；
3. 面板里没有财报因子列时（本次扫描未启用扩展因子），**财务因子覆盖率同样不可计算**，
   同样标注而不是跳过——跳过等于让这一维从仪表盘上静静消失。

纯计算函数（``population_stability_index``、``kolmogorov_smirnov``、``classify_psi`` 等）
与取数写库函数分开，前者不碰数据库，需求文 §二-F 的 PSI 与阈值判定因此可用构造数据直接验证。

本模块刻意不引入 Streamlit：services/scan_pipeline.py 会调用它。
"""

from __future__ import annotations

import logging
import math
from typing import Any, Sequence

import numpy as np
import pandas as pd

from config import settings
from services.database import QuantRepository
from services.extra_factors import EXTRA_FACTOR_FAMILIES, summarize_extra_factors
from services.model_monitor import finite_or_none

LOGGER = logging.getLogger("quant.drift_monitor")

# 无法计算时的等级值。与「正常」严格区分：仪表盘上「未接入」是一句事实陈述，
# 「正常」是一个结论，两者看起来一样才是真正危险的。
LEVEL_OK = "正常"
LEVEL_WATCH = "关注"
LEVEL_SEVERE = "严重"
LEVEL_INSUFFICIENT = "样本不足"
LEVEL_UNAVAILABLE = "未接入"

# 加在比例上的极小值。空箱必须给一个极小数而不是跳过：跳过等于把该区间
# 从 PSI 求和里抹掉，分布变化越大反而算出来越正常。
_EPSILON = 1e-6

CONCEPT_HORIZON = 20

# 因子族的中文名，只用于写给人看的 note。
EXTRA_FACTOR_LABELS = {"chip": "筹码", "intraday": "日内", "financial": "财报"}


def population_stability_index(
    reference: Any, current: Any, bins: int = settings.DRIFT_PSI_BINS
) -> float | None:
    """PSI：参考分布的分箱上，两组比例的加权对数差之和。

    分箱边界取参考分布的分位数，因此箱内样本在参考分布下是均匀的。
    参考分布近乎常数（唯一分位点不足 3 个）时返回 None——此时分箱没有意义，
    返回 0.0 会假装「没有漂移」。
    """
    base = np.asarray(reference, dtype=float)
    base = base[np.isfinite(base)]
    now = np.asarray(current, dtype=float)
    now = now[np.isfinite(now)]
    if base.size == 0 or now.size == 0:
        return None
    edges = np.unique(np.quantile(base, np.linspace(0.0, 1.0, int(bins) + 1)))
    if edges.size < 3:
        return None
    edges[0], edges[-1] = -np.inf, np.inf
    expected = np.histogram(base, bins=edges)[0].astype(float) / base.size
    actual = np.histogram(now, bins=edges)[0].astype(float) / now.size
    return _psi_from_proportions(np.clip(expected, _EPSILON, None), np.clip(actual, _EPSILON, None))


def categorical_stability_index(reference: Any, current: Any) -> float | None:
    """类别列（行业）的 PSI：各类别占比的加权对数差之和。

    两个窗口的类别集合取并集：只出现在当前窗口的新类别按参考占比 0 计入，
    否则「冒出好几个新行业」这种最需要被发现的变化反而算不出来。
    """
    base = pd.Series(reference).dropna().astype(str)
    now = pd.Series(current).dropna().astype(str)
    if base.empty or now.empty:
        return None
    categories = sorted(set(base.unique()) | set(now.unique()))
    expected = base.value_counts(normalize=True).reindex(categories).fillna(0.0).to_numpy(dtype=float)
    actual = now.value_counts(normalize=True).reindex(categories).fillna(0.0).to_numpy(dtype=float)
    return _psi_from_proportions(np.clip(expected, _EPSILON, None), np.clip(actual, _EPSILON, None))


def _psi_from_proportions(expected: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def kolmogorov_smirnov(reference: Any, current: Any) -> float | None:
    """两样本 KS 统计量：两条经验分布函数的最大垂直距离，取值 0~1。

    用 numpy 手写而不是调 scipy：scipy 只是 statsmodels 的传递依赖，
    requirements.txt 里并未声明它，直接 import 会让精简环境里的整条扫描链断掉。
    """
    base = np.sort(np.asarray(reference, dtype=float))
    base = base[np.isfinite(base)]
    now = np.sort(np.asarray(current, dtype=float))
    now = now[np.isfinite(now)]
    if base.size == 0 or now.size == 0:
        return None
    combined = np.concatenate([base, now])
    # 在两组样本的每一个取值上比较累计比例。用 side="right" 让相等值同进同出，
    # 否则同分布的两组样本也会算出一个非零距离。
    cdf_base = np.searchsorted(base, combined, side="right") / base.size
    cdf_current = np.searchsorted(now, combined, side="right") / now.size
    return float(np.max(np.abs(cdf_base - cdf_current)))


def classify_psi(psi: float | None, sample_size: int, minimum_samples: int = settings.DRIFT_MINIMUM_SAMPLES) -> str:
    """PSI 的等级判定。样本不足时明确返回「样本不足」，不给等级。"""
    if psi is None:
        return LEVEL_UNAVAILABLE
    if sample_size < int(minimum_samples):
        return LEVEL_INSUFFICIENT
    if psi > settings.DRIFT_PSI_SEVERE:
        return LEVEL_SEVERE
    if psi > settings.DRIFT_PSI_WATCH:
        return LEVEL_WATCH
    return LEVEL_OK


def _window_dates(panel: pd.DataFrame, current_days: int, reference_days: int) -> tuple[list, list]:
    """切出参考窗口与当前窗口的交易日。

    当前窗口 = 最近 current_days 个交易日；参考窗口 = 它之前的最近 reference_days 个交易日
    （reference_days 为 0 或不足时用上全部更早的历史）。
    """
    all_dates = sorted(pd.to_datetime(panel["date"]).drop_duplicates())
    if not all_dates:
        return [], []
    current = all_dates[-current_days:] if len(all_dates) > current_days else all_dates
    earlier = all_dates[: len(all_dates) - len(current)]
    if not earlier:
        return [], current
    reference = earlier[-reference_days:] if reference_days and len(earlier) > reference_days else earlier
    return reference, current


def _mean_shift(reference: np.ndarray, current: np.ndarray) -> float | None:
    """均值偏移，按参考窗口的标准差标准化，便于不同量纲的因子横向比较。"""
    if reference.size == 0 or current.size == 0:
        return None
    deviation = float(np.std(reference, ddof=0))
    if not math.isfinite(deviation) or deviation <= 1e-12:
        return None
    return float((float(np.mean(current)) - float(np.mean(reference))) / deviation)


def _numeric_row(
    name: str, column: str, panel: pd.DataFrame, reference_dates: list, current_dates: list,
    report_date: str, note: str | None = None,
) -> dict[str, Any]:
    """构造一个数值因子的漂移行。列缺失或全空时如实标注，不给数。"""
    reference_start, reference_end = _span(reference_dates)
    current_start, current_end = _span(current_dates)
    base = {"date": report_date, "feature_name": name, "feature_name_column": column,
            "reference_start": reference_start, "reference_end": reference_end,
            "window_start": current_start, "window_end": current_end}
    if column not in panel.columns:
        return {**base, "psi": None, "ks_stat": None, "missing_rate": None, "mean_shift": None,
                "threshold_exceeded": LEVEL_UNAVAILABLE, "sample_size": 0,
                "note": note or f"本地面板中没有 {column} 列，该维无法计算"}
    reference_values = pd.to_numeric(panel.loc[panel["date"].isin(reference_dates), column], errors="coerce")
    current_values = pd.to_numeric(panel.loc[panel["date"].isin(current_dates), column], errors="coerce")
    current_valid = current_values.dropna().to_numpy(dtype=float)
    # 缺失率只在当前窗口上有意义：它回答的是「现在还能不能算得出这个因子」。
    missing_rate = float(current_values.isna().mean()) if len(current_values) else None
    psi = population_stability_index(reference_values, current_values)
    ks = kolmogorov_smirnov(reference_values, current_values)
    sample_size = int(current_valid.size)
    row = {
        **base,
        "psi": psi, "ks_stat": ks, "missing_rate": missing_rate,
        "mean_shift": _mean_shift(reference_values.dropna().to_numpy(dtype=float), current_valid),
        "threshold_exceeded": classify_psi(psi, sample_size),
        "sample_size": sample_size,
        "note": None,
    }
    if row["threshold_exceeded"] == LEVEL_INSUFFICIENT:
        row["note"] = f"当前窗口有效样本仅 {sample_size} 个，低于 {settings.DRIFT_MINIMUM_SAMPLES} 的下限，PSI 不作等级判定"
    elif row["threshold_exceeded"] == LEVEL_UNAVAILABLE:
        row["note"] = "参考窗口数据不足或近乎常数，PSI 无法计算"
    return row


def _categorical_row(
    name: str, column: str, panel: pd.DataFrame, reference_dates: list, current_dates: list, report_date: str
) -> dict[str, Any]:
    """构造类别列的漂移行。行业这类列用类别占比算 PSI，不做 KS 与均值偏移。"""
    reference_start, reference_end = _span(reference_dates)
    current_start, current_end = _span(current_dates)
    base = {"date": report_date, "feature_name": name, "feature_name_column": column,
            "reference_start": reference_start, "reference_end": reference_end,
            "window_start": current_start, "window_end": current_end,
            "ks_stat": None, "mean_shift": None}
    if column not in panel.columns:
        return {**base, "psi": None, "missing_rate": None, "threshold_exceeded": LEVEL_UNAVAILABLE,
                "sample_size": 0, "note": f"本地面板中没有 {column} 列，该维无法计算"}
    reference_values = panel.loc[panel["date"].isin(reference_dates), column]
    current_values = panel.loc[panel["date"].isin(current_dates), column]
    sample_size = int(current_values.notna().sum())
    psi = categorical_stability_index(reference_values, current_values)
    level = classify_psi(psi, sample_size)
    if level == LEVEL_UNAVAILABLE:
        note = "参考窗口没有可用类别或分位点不足，PSI 无法计算"
    else:
        note = f"类别列：按 {len(set(current_values.dropna().astype(str)))} 个行业的占比分布计算 PSI，不做 KS 与均值偏移"
        if level == LEVEL_INSUFFICIENT:
            note = f"{note}；当前窗口有效样本仅 {sample_size} 个，低于 {settings.DRIFT_MINIMUM_SAMPLES} 的下限，不作等级判定"
    return {
        **base, "psi": psi,
        "missing_rate": float(current_values.isna().mean()) if len(current_values) else None,
        "threshold_exceeded": level,
        "sample_size": sample_size,
        "note": note,
    }


def _coverage_row(
    panel: pd.DataFrame, reference_dates: list, current_dates: list, report_date: str
) -> dict[str, Any]:
    """财务因子覆盖率维度。

    这一维不是一个可做 PSI 的分布，而是一个覆盖率标量，漂移的含义是「还算不算得出来」。
    因此用当前覆盖率与参考覆盖率的差作为 mean_shift，并直接引用现有模型的覆盖率门控
    ``EXTRA_FACTOR_MINIMUM_COVERAGE`` 作为严重线——低于它就是模型真的会丢掉这一类因子。
    """
    reference_start, reference_end = _span(reference_dates)
    current_start, current_end = _span(current_dates)
    base = {"date": report_date, "feature_name": "财务因子覆盖率", "feature_name_column": None,
            "reference_start": reference_start, "reference_end": reference_end,
            "window_start": current_start, "window_end": current_end,
            "psi": None, "ks_stat": None}
    if panel.empty:
        return {**base, "missing_rate": None, "mean_shift": None, "threshold_exceeded": LEVEL_UNAVAILABLE,
                "sample_size": 0, "note": "本地面板为空，覆盖率无法计算"}
    # 必须先判断面板里到底有没有这一类列：summarize_extra_factors 对每个因子族都会返回一个键，
    # 列不存在时给出 coverage 0.0。直接拿它判定覆盖率，会把「本次扫描根本没启用扩展因子」
    # 报成覆盖率 0% 的严重漂移，进而触发告警和自动降级。
    if not any(column in panel.columns for columns in EXTRA_FACTOR_FAMILIES.values() for column in columns):
        return {**base, "missing_rate": None, "mean_shift": None, "threshold_exceeded": LEVEL_UNAVAILABLE,
                "sample_size": 0, "note": "本地面板中没有筹码/日内/财报因子列（本次扫描未启用扩展因子），该维不可计算"}
    # 缺席的族同样不能计入：面板里只有筹码因子时分母写成三个族，覆盖率会被算成 33% 判严重，
    # 而实际原因是另外两族没接入本地数据。只判定「面板里真有列」的族。
    present = [
        name for name, columns in EXTRA_FACTOR_FAMILIES.items()
        if any(column in panel.columns for column in columns)
    ]
    absent = [name for name in EXTRA_FACTOR_FAMILIES if name not in present]
    current_report = summarize_extra_factors(panel.loc[panel["date"].isin(current_dates)])
    reference_report = summarize_extra_factors(panel.loc[panel["date"].isin(reference_dates)])
    current_coverages = {name: float(current_report[name]["coverage"]) for name in present}
    reference_coverages = (
        {name: float(reference_report[name]["coverage"]) for name in present}
        if reference_report else dict(current_coverages)
    )
    current_coverage = float(np.mean(list(current_coverages.values())))
    reference_coverage = float(np.mean(list(reference_coverages.values())))
    sample_size = int(current_report["rows"])
    gate = settings.EXTRA_FACTOR_MINIMUM_COVERAGE
    shift = current_coverage - reference_coverage
    # 模型是按族整体门控的（accepted_extra_columns），因此逐族对照门控线，而不是拿各族均值糊过去。
    # 再看参考窗口：本来达标、当前掉下去的才是漂移；参考窗口就不达标的属于数据深度不足
    # （例如本地分钟线刚累积几天），判关注而不是严重——自动降级不该被这种常态触发。
    regressed = [name for name, value in current_coverages.items()
                 if value < gate and reference_coverages[name] >= gate]
    shallow = [name for name, value in current_coverages.items()
               if value < gate and name not in regressed]
    if sample_size < settings.DRIFT_MINIMUM_SAMPLES:
        level = LEVEL_INSUFFICIENT
    elif regressed:
        level = LEVEL_SEVERE
    elif shallow or shift < -settings.DRIFT_PSI_WATCH:
        level = LEVEL_WATCH
    else:
        level = LEVEL_OK
    detail = "、".join(f"{EXTRA_FACTOR_LABELS[name]} {value:.0%}" for name, value in current_coverages.items())
    note = (
        f"覆盖率 {current_coverage:.1%}（参考窗口 {reference_coverage:.1%}）；逐族 {detail}；"
        f"低于 {gate:.0%} 的族会被模型整体丢弃"
    )
    if absent:
        note = f"{note}；面板无列、不参与判定：{'、'.join(EXTRA_FACTOR_LABELS[name] for name in absent)}"
    if regressed:
        note = f"{note}；掉出门控线：{'、'.join(EXTRA_FACTOR_LABELS[name] for name in regressed)}"
    elif shallow:
        note = f"{note}；参考窗口同样未达标（数据深度不足，非漂移）：{'、'.join(EXTRA_FACTOR_LABELS[name] for name in shallow)}"
    if level == LEVEL_INSUFFICIENT:
        note = f"{note}；当前窗口仅 {sample_size} 行，低于 {settings.DRIFT_MINIMUM_SAMPLES} 的下限，不作等级判定"
    return {**base, "missing_rate": 1.0 - current_coverage, "mean_shift": shift,
            "threshold_exceeded": level, "sample_size": sample_size, "note": note}


def _valuation_row(panel: pd.DataFrame, reference_dates: list, current_dates: list, report_date: str) -> dict[str, Any]:
    """估值维度。

    本系统的行情源（东方财富股票池接口）只返回代码、名称、价格、涨跌、成交量成交额、
    市值与行业，**没有任何 PE/PB/PS 字段**；``services/exposures.py::build_exposure_frame``
    也只透传市值与行业。因此这里如实写「未接入」，而不是拿一个不相干的列冒充估值。
    要接入需要先补一个稳定的估值数据源，这属于新增数据采集，不在本次改动范围内。
    """
    reference_start, reference_end = _span(reference_dates)
    current_start, current_end = _span(current_dates)
    return {
        "date": report_date, "feature_name": "估值", "feature_name_column": None,
        "reference_start": reference_start, "reference_end": reference_end,
        "window_start": current_start, "window_end": current_end,
        "psi": None, "ks_stat": None, "missing_rate": None, "mean_shift": None,
        "threshold_exceeded": LEVEL_UNAVAILABLE, "sample_size": 0,
        "note": "未接入：现有行情源不提供 PE/PB/PS，股票池接口与暴露数据都没有估值字段；需先补一个稳定的估值数据源才能监控该维",
    }


def _span(dates: Sequence[Any]) -> tuple[str | None, str | None]:
    if not len(dates):
        return None, None
    return str(pd.Timestamp(min(dates)).date()), str(pd.Timestamp(max(dates)).date())


def feature_drift_rows(
    panel: pd.DataFrame,
    current_days: int = settings.DRIFT_CURRENT_WINDOW_DAYS,
    reference_days: int = settings.TRAINING_WINDOW_DAYS,
) -> list[dict[str, Any]]:
    """按配置的维度逐个计算特征漂移，返回可直接落库的行。

    配置里列出的每一维都会出现在结果里：算得出来的给数，算不出来的给原因。
    """
    if panel is None or panel.empty:
        return []
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    reference_dates, current_dates = _window_dates(frame, int(current_days), int(reference_days))
    if not current_dates:
        return []
    report_date = str(pd.Timestamp(current_dates[-1]).date())
    rows: list[dict[str, Any]] = []
    for name, spec in settings.DRIFT_MONITORED_FEATURES.items():
        if name == "估值":
            rows.append(_valuation_row(frame, reference_dates, current_dates, report_date))
        elif spec == settings.DRIFT_EXTRA_COVERAGE_SENTINEL:
            rows.append(_coverage_row(frame, reference_dates, current_dates, report_date))
        elif spec in settings.DRIFT_CATEGORICAL_FEATURES:
            rows.append(_categorical_row(name, spec, frame, reference_dates, current_dates, report_date))
        else:
            rows.append(_numeric_row(name, spec, frame, reference_dates, current_dates, report_date))
    return rows


def concept_drift_row(
    metrics: dict[str, Any] | None, report_date: str, window: int = settings.DRIFT_CONCEPT_WINDOW_DAYS
) -> dict[str, Any] | None:
    """从某个周期的样本外指标算出滚动概念漂移。

    ``daily_groups`` 只包含「当日股票数达标且预测有差异」的可用交易日，
    因此这里的「连续」是按可用行计，不是按自然日计——中间缺席的日子本来就没有 IC 可算，
    把它们算成中断会让连续为负的日子永远凑不满。

    注意 ndcg_ratio 是**整个样本外区间的值**，不是这一滚动窗口的值：
    daily_groups 里没有逐日 NDCG，为它单独重算需要保留逐日成员明细，那会让
    模型上下文 JSON 每次扫描都变大。这一点写在 note 里，不假装它是窗口指标。
    """
    groups = (metrics or {}).get("daily_groups") or []
    if not groups:
        return None
    frame = pd.DataFrame(groups)
    if frame.empty or "rank_ic" not in frame:
        return None
    series = pd.to_numeric(frame["rank_ic"], errors="coerce").dropna()
    if series.empty:
        return None
    current = series.tail(int(window))
    previous = series.iloc[max(0, len(series) - 2 * int(window)): len(series) - int(window)]
    ic_mean = float(current.mean())
    deviation = float(current.std(ddof=0))
    icir = ic_mean / deviation if deviation > 1e-12 else None
    negative_streak = 0
    for value in reversed(series.tolist()):
        if value < 0:
            negative_streak += 1
        else:
            break
    trend = "未知"
    if len(previous) >= 5:
        delta = ic_mean - float(previous.mean())
        threshold = 0.5 * float(previous.std(ddof=0)) if float(previous.std(ddof=0)) > 1e-12 else 0.0
        trend = "好转" if delta > threshold else ("恶化" if delta < -threshold else "持平")
    hac_t = _hac_t(current.to_numpy(dtype=float))
    tail = frame.tail(int(window))
    return {
        "date": report_date,
        "window": int(len(current)),
        "rank_ic": ic_mean,
        "icir": icir,
        "hac_t": hac_t,
        "ndcg_ratio": finite_or_none((metrics or {}).get("ndcg_ratio")),
        "top_excess": finite_or_none(pd.to_numeric(tail.get("top_excess_return"), errors="coerce").mean())
        if "top_excess_return" in tail else None,
        "trend": trend,
        "negative_streak": negative_streak,
        "note": (
            f"滚动窗口 {len(current)} 个可用样本外交易日（非自然日）；"
            f"ndcg_ratio 为整个样本外区间的值，不是本窗口的值"
        ),
    }


def _hac_t(values: np.ndarray) -> float | None:
    """对滚动 IC 序列做 Newey-West 校正的均值显著性检验。"""
    if values.size < 3 or not np.isfinite(values).all():
        return None
    from statsmodels.api import OLS

    try:
        estimate = OLS(values, np.ones((values.size, 1))).fit(
            cov_type="HAC", cov_kwds={"maxlags": min(CONCEPT_HORIZON, values.size - 1)}
        )
    except (ValueError, np.linalg.LinAlgError):
        return None
    deviation = float(estimate.bse[0])
    if not math.isfinite(deviation) or deviation <= 1e-12:
        return None
    return float(values.mean() / deviation)


def prediction_drift_row(
    context: dict[str, Any], panel: pd.DataFrame | None, top_n: int = settings.DRIFT_PREDICTION_TOP_N,
    horizon: int = CONCEPT_HORIZON,
) -> dict[str, Any] | None:
    """每日 Top-N 组合的行业集中度、市值分位与分数分布。

    分数用未归一化的 ``predicted_excess_return``。**不能用 prediction_zscore**：
    它每天都被强制标准化成均值 0、标准差 1，逐日记录恒为常数，看起来有数其实没有信息量。

    面板里不含预测列，预测只能来自模型上下文；没有任何一个模型通过验证时，
    这一行如实写「未通过验证，无可评估的预测」，而不是编一组 0。
    """
    report_date = context.get("reference_date")
    # 周期键在训练产物里是 int（HORIZONS 就是 int），而经过摘要 JSON 往返之后会变成 str。
    # 只认 int 的话，读到 JSON 版本上下文时会得出一句「本轮没有通过验证的模型」——
    # 这是把「取不到预测」说成了「模型都没通过」，比静默更糟。
    predictions = (context.get("predictions") or {}).get(horizon)
    if predictions is None:
        predictions = (context.get("predictions") or {}).get(str(horizon)) or {}
    base = {"date": report_date, "top_n": int(top_n), "industry_hhi": None,
            "market_cap_quantile_mean": None, "score_mean": None, "score_std": None,
            "stocks": 0, "note": None}
    if not report_date:
        return None
    scores = {
        code: float(np.mean([item["predicted_excess_return"] for item in per_model.values()]))
        for code, per_model in predictions.items()
        if per_model
    }
    if not scores:
        return {**base, "hhi_threshold_exceeded": LEVEL_UNAVAILABLE,
                "hhi_history_p95": None,
                "note": "本轮没有通过验证的模型，没有任何预测可供评估"}
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[: int(top_n)]
    codes = [code for code, _ in ordered]
    values = np.array([score for _, score in ordered], dtype=float)
    row = {
        **base,
        "stocks": len(codes),
        "score_mean": float(values.mean()),
        "score_std": float(values.std(ddof=0)),
    }
    if panel is None or panel.empty:
        row["note"] = "本地面板不可用，行业集中度与市值分位未计算"
        return row
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    sample = frame.loc[frame["date"].eq(str(report_date))]
    if sample.empty:
        row["note"] = f"面板中没有 {report_date} 的截面，行业集中度与市值分位未计算"
        return row
    selected = sample.loc[sample["code"].astype(str).isin(codes)]
    industry = selected["industry"].dropna().astype(str) if "industry" in selected.columns else pd.Series(dtype=str)
    if len(industry):
        shares = industry.value_counts(normalize=True).to_numpy(dtype=float)
        row["industry_hhi"] = float(np.sum(shares ** 2))
    else:
        row["note"] = "该截面没有行业字段，行业集中度未计算"
    if "log_market_cap" in sample.columns:
        pool = pd.to_numeric(sample["log_market_cap"], errors="coerce")
        ranks = pool.rank(pct=True)
        chosen = ranks.loc[selected.index].dropna()
        if len(chosen):
            row["market_cap_quantile_mean"] = float(chosen.mean())
    return row


def evaluate_drift_alerts(
    feature_rows: Sequence[dict[str, Any]],
    concept_rows: Sequence[dict[str, Any]],
    prediction_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """把三类漂移的判定结果翻译成告警。纯函数，便于用构造数据验证阈值。"""
    alerts: list[dict[str, Any]] = []
    for row in feature_rows:
        level = row.get("threshold_exceeded")
        if level not in (LEVEL_WATCH, LEVEL_SEVERE):
            continue
        psi = row.get("psi")
        alerts.append({
            "level": "critical" if level == LEVEL_SEVERE else "warning",
            "category": "drift",
            "title": f"特征漂移{level}：{row['feature_name']}",
            "message": (
                f"{row['date']} 的「{row['feature_name']}」PSI 为 "
                f"{'未计算' if psi is None else format(float(psi), '.3f')}，判定为{level}"
                f"（关注线 {settings.DRIFT_PSI_WATCH}，严重线 {settings.DRIFT_PSI_SEVERE}）。"
                + (f" {row['note']}" if row.get("note") else "")
            ),
            "details": finite_or_none({
                "feature_name": row["feature_name"], "psi": psi, "ks_stat": row.get("ks_stat"),
                "sample_size": row.get("sample_size"), "level": level, "date": row.get("date"),
            }),
        })
    for row in concept_rows:
        streak = int(row.get("negative_streak") or 0)
        if streak < settings.DRIFT_IC_NEGATIVE_DAYS:
            continue
        alerts.append({
            "level": "critical",
            "category": "drift",
            "title": f"概念漂移：Rolling Rank IC 连续 {streak} 个交易日为负",
            "message": (
                f"截至 {row['date']}，模型排序与真实涨跌的相关性已连续 {streak} 个可用样本外交易日为负"
                f"（阈值 {settings.DRIFT_IC_NEGATIVE_DAYS} 日），说明模型可能已经与当前市场风格脱节。"
            ),
            "details": finite_or_none({
                "date": row.get("date"), "negative_streak": streak,
                "rank_ic": row.get("rank_ic"), "hac_t": row.get("hac_t"), "trend": row.get("trend"),
            }),
        })
    for row in prediction_rows:
        if row.get("hhi_threshold_exceeded") != LEVEL_WATCH:
            continue
        alerts.append({
            "level": "warning",
            "category": "drift",
            "title": "预测漂移：Top 组合行业集中度超过历史 95 分位",
            "message": (
                f"{row['date']} 的 Top-{row['top_n']} 组合行业 HHI 为 {float(row['industry_hhi']):.3f}，"
                f"超过历史 95 分位 {float(row['hhi_history_p95']):.3f}，持仓过度集中在少数行业。"
            ),
            "details": finite_or_none({
                "date": row.get("date"), "top_n": row.get("top_n"),
                "industry_hhi": row.get("industry_hhi"), "history_p95": row.get("hhi_history_p95"),
            }),
        })
    return alerts


def _jsonb(value: Any) -> Any:
    from psycopg.types.json import Jsonb

    return Jsonb(finite_or_none(value) or {})


def log_feature_drift(repository: QuantRepository, rows: Sequence[dict[str, Any]]) -> int:
    """按 (date, feature_name) UPSERT 特征漂移；重复扫描不刷重复行。"""
    payload = [
        (
            row["date"], row["feature_name"], row.get("psi"), row.get("ks_stat"), row.get("missing_rate"),
            row.get("mean_shift"), row.get("threshold_exceeded") or LEVEL_OK, row.get("window_start"),
            row.get("window_end"), row.get("reference_start"), row.get("reference_end"),
            row.get("sample_size"), row.get("note"),
        )
        for row in rows
    ]
    if not payload:
        return 0
    with repository.connect() as connection:
        with connection.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO public.quant_feature_drift(date,feature_name,psi,ks_stat,missing_rate,mean_shift,"
                "threshold_exceeded,window_start,window_end,reference_start,reference_end,sample_size,note) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT(date,feature_name) DO UPDATE SET psi=EXCLUDED.psi,ks_stat=EXCLUDED.ks_stat,"
                "missing_rate=EXCLUDED.missing_rate,mean_shift=EXCLUDED.mean_shift,"
                "threshold_exceeded=EXCLUDED.threshold_exceeded,window_start=EXCLUDED.window_start,"
                "window_end=EXCLUDED.window_end,reference_start=EXCLUDED.reference_start,"
                "reference_end=EXCLUDED.reference_end,sample_size=EXCLUDED.sample_size,note=EXCLUDED.note,"
                "updated_at=now()",
                payload,
            )
    return len(payload)


def log_concept_drift(repository: QuantRepository, rows: Sequence[dict[str, Any]]) -> int:
    """按 (date, window_days) UPSERT 概念漂移。

    落库列名是 ``window_days``：``window`` 是 PostgreSQL 保留字，建表就会语法错误。
    行字典里仍沿用 ``window`` 这个键（需求文与 :func:`concept_drift_row` 的字段名）。
    """
    payload = [
        (row["date"], int(row["window"]), row.get("rank_ic"), row.get("icir"), row.get("hac_t"),
         row.get("ndcg_ratio"), row.get("top_excess"), row.get("trend") or "未知",
         row.get("negative_streak"), row.get("note"))
        for row in rows
    ]
    if not payload:
        return 0
    with repository.connect() as connection:
        with connection.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO public.quant_concept_drift(date,window_days,rank_ic,icir,hac_t,ndcg_ratio,top_excess,"
                "trend,negative_streak,note) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT(date,window_days) DO UPDATE SET rank_ic=EXCLUDED.rank_ic,icir=EXCLUDED.icir,"
                "hac_t=EXCLUDED.hac_t,ndcg_ratio=EXCLUDED.ndcg_ratio,top_excess=EXCLUDED.top_excess,"
                "trend=EXCLUDED.trend,negative_streak=EXCLUDED.negative_streak,note=EXCLUDED.note,"
                "updated_at=now()",
                payload,
            )
    return len(payload)


def history_hhi_percentile(repository: QuantRepository, report_date: str, percentile: float = settings.DRIFT_HHI_PERCENTILE) -> float | None:
    """历史 HHI 分位基准，排除当日自身。样本不足时返回 None，由调用方标注。"""
    with repository.connect() as connection:
        row = connection.execute(
            "SELECT percentile_cont(%s) WITHIN GROUP (ORDER BY industry_hhi) AS threshold, count(*) AS samples "
            "FROM public.quant_prediction_drift WHERE industry_hhi IS NOT NULL AND date < %s",
            (float(percentile), report_date),
        ).fetchone()
    # 样本数必须真的用上：只有两三天历史时 percentile_cont 给的就是历史最大值附近，
    # 于是每次集中度创出新高都判「关注」，而那不是漂移是没基准。低于下限时如实返回 None。
    if row is None or int(row["samples"] or 0) < settings.DRIFT_HHI_MINIMUM_SAMPLES:
        return None
    return finite_or_none(row["threshold"])


def log_prediction_drift(repository: QuantRepository, rows: Sequence[dict[str, Any]]) -> int:
    """按 (date, top_n) UPSERT 预测漂移，并在写入前用历史分位判定行业集中度。"""
    payload = []
    for row in rows:
        hhi = row.get("industry_hhi")
        level = LEVEL_UNAVAILABLE if hhi is None else LEVEL_OK
        threshold = None
        if hhi is not None:
            threshold = finite_or_none(history_hhi_percentile(repository, row["date"]))
            if threshold is None:
                level = LEVEL_INSUFFICIENT
                row = {**row, "note": ((row.get("note") or "") + "；历史样本不足，行业集中度不作等级判定").strip("；")}
            elif float(hhi) > float(threshold):
                level = LEVEL_WATCH
        payload.append(
            (row["date"], int(row["top_n"]), hhi, row.get("market_cap_quantile_mean"),
             row.get("score_mean"), row.get("score_std"), level, threshold, row.get("stocks"), row.get("note"))
        )
        row["hhi_threshold_exceeded"] = level
        row["hhi_history_p95"] = threshold
    if not payload:
        return 0
    with repository.connect() as connection:
        with connection.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO public.quant_prediction_drift(date,top_n,industry_hhi,market_cap_quantile_mean,"
                "score_mean,score_std,hhi_threshold_exceeded,hhi_history_p95,stocks,note) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT(date,top_n) DO UPDATE SET industry_hhi=EXCLUDED.industry_hhi,"
                "market_cap_quantile_mean=EXCLUDED.market_cap_quantile_mean,score_mean=EXCLUDED.score_mean,"
                "score_std=EXCLUDED.score_std,hhi_threshold_exceeded=EXCLUDED.hhi_threshold_exceeded,"
                "hhi_history_p95=EXCLUDED.hhi_history_p95,stocks=EXCLUDED.stocks,note=EXCLUDED.note,"
                "updated_at=now()",
                payload,
            )
    return len(payload)


def recent_feature_drift(repository: QuantRepository, limit: int = 60) -> list[dict[str, Any]]:
    """最近的各维漂移，供管理员仪表盘。"""
    with repository.connect() as connection:
        return connection.execute(
            "SELECT * FROM public.quant_feature_drift ORDER BY date DESC, feature_name LIMIT %s",
            (min(max(int(limit), 1), 1000),),
        ).fetchall()


def recent_concept_drift(repository: QuantRepository, limit: int = 120) -> list[dict[str, Any]]:
    with repository.connect() as connection:
        return connection.execute(
            "SELECT * FROM public.quant_concept_drift ORDER BY date DESC, window_days DESC LIMIT %s",
            (min(max(int(limit), 1), 1000),),
        ).fetchall()


def recent_prediction_drift(repository: QuantRepository, limit: int = 120) -> list[dict[str, Any]]:
    with repository.connect() as connection:
        return connection.execute(
            "SELECT * FROM public.quant_prediction_drift ORDER BY date DESC, top_n LIMIT %s",
            (min(max(int(limit), 1), 1000),),
        ).fetchall()
