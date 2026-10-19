"""最终留出集评估、模型归档、回滚与降级。

## 关于留出集口径的一处有意偏离

需求文 §二-A-2 要求「任何训练、验证、超参搜索、自动择优、因子选择都不能使用留出集
日期范围内的数据，滚动窗口的结束日严格早于 FINAL_HOLDOUT_START」。

**本系统没有那样做，这是有意的。** 本系统每天都要用最近 750 个交易日滚动重训生产模型，
才能给 2026 年的截面打分；把滚动窗口的结束日永久冻结在 2024-01-01 之前，等于让生产模型
再也看不到近两年的市场，日常评分会直接失去意义。两者不可兼得。

采用的折中是 **评估专用模式**：

* 留出集只在 :func:`evaluate_final_holdout` 里生效，且只约束那一次评估——
  评估用的**冻结模型**严格只用 ``date < FINAL_HOLDOUT_START`` 且标签结束日也早于它的
  样本训练，留出集对它而言确实是没见过的数据；
* 日常生产训练（``train_market_context``）保持原样，按滚动窗口使用全量历史；
* 因此「最终留出集」给出的是**这套建模流程在未参与调参的数据上表现如何**的结论，
  而不是「今天线上那个模型在留出集上表现如何」的结论。这个区别写在报告里。

## 归档与回滚

``models/archive/<context_id>/`` 保存历次产出的模型文件与面板。归档统一用硬链接：
``save_context`` 从不原地改写文件，而是写临时文件再 ``os.replace``，
因此归档里的硬链接会一直指向被替换掉的那份旧内容，**零字节拷贝**。
回滚时反向复制回活动路径，不复用硬链接（否则回滚会掏空归档目录，就再也回不去了）。

## 自动与人工的边界

通过率告警或严重漂移触发时，系统**自动**执行的只有三件事：把模型标成 degraded、
在页面上暂停输出评分、写一条告警。这三件事完全可逆、不改变生产模型。
**回滚必须由管理员在页面上点击**并留下审计——自动换掉生产模型是不可逆的静默动作，
与「所有数据可追溯、可回滚」的基调相悖。重训不单独触发，每日定时扫描本身就是重训。

本模块刻意不引入 Streamlit。
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from config import settings
from services import alerting, drift_monitor, model_monitor
from services.backtest import cost_rates
from services.database import DatabaseError, QuantRepository
from services.ml_pipeline import (
    HORIZONS, MINIMUM_COVERAGE, MINIMUM_STOCKS, MODEL_NAMES, SCORE_KIND_RANK, evaluate_oos, fit_panel_model,
    fuse_normalized_predictions, load_context, normalize_predictions,
)
from services.model_monitor import finite_or_none

LOGGER = logging.getLogger("quant.model_governance")
SHANGHAI = ZoneInfo("Asia/Shanghai")

# 留出集换手率与生产组合共用固定 Top N 口径。
# 换手率与费用后收益只在主周期上出：它是「非重叠持仓期」的指标，
# 5 日与 60 日各出一份只是噪音，而且 60 日在留出集里凑不出几个完整持仓期。
TURNOVER_HORIZON = 20

DEGRADED_SCORING_MESSAGE = settings.DEGRADED_SCORING_MESSAGE


class FinalHoldoutError(RuntimeError):
    """留出集评估的前置条件不满足，或该版本已评估过。"""


def _report_path() -> Path:
    return settings.MODEL_DIR / settings.FINAL_HOLDOUT_REPORT_NAME


# ---------------------------------------------------------------------------
# 留出集：纯计算部分
# ---------------------------------------------------------------------------

def holdout_window(panel: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """留出集的起止交易日；终点未配置时取面板最新截面。"""
    start = pd.Timestamp(settings.FINAL_HOLDOUT_START)
    if settings.FINAL_HOLDOUT_END:
        end = pd.Timestamp(settings.FINAL_HOLDOUT_END)
    else:
        end = pd.Timestamp(panel["date"].max())
    if end <= start:
        raise FinalHoldoutError(
            f"留出集区间为空：起点 {start.date()} 不早于终点 {end.date()}，请检查 "
            "FINAL_HOLDOUT_START / FINAL_HOLDOUT_END 配置。"
        )
    return start, end


def training_baseline(
    panel: pd.DataFrame, start: pd.Timestamp, target: str, endpoint: str,
    features: list[str] | None = None, window: int | None = None,
) -> float:
    """冻结模型的训练样本中位数基线。

    取的是**冻结模型真正训练过的那段样本**的中位数，不是留出集自己的中位数：
    用留出集的统计量当基线，既是拿留出集的信息反过来构造检验，也会让
    「MAE 优于基线」退化成自己跟自己比。
    """
    train = panel.loc[panel["date"] < start]
    labelled = train.dropna(subset=[target, endpoint, *(features or [])])
    labelled = labelled.loc[labelled[endpoint] < start]
    window = int(settings.TRAINING_WINDOW_DAYS if window is None else window)
    if window > 0:
        dates = np.sort(labelled["date"].unique())
        labelled = labelled.loc[labelled["date"].isin(dates[-window:])]
    if labelled.empty:
        raise FinalHoldoutError("留出集之前没有任何完整标签样本，无法计算冻结模型的基线。")
    baseline = float(labelled[target].median())
    if not np.isfinite(baseline):
        raise FinalHoldoutError("冻结模型的训练样本中位数不是有限值，无法计算基线。")
    return baseline


def _top_members(group: pd.DataFrame, top_n: int) -> set[str]:
    ordered = group.sort_values("prediction", kind="stable")
    size = min(int(top_n), len(ordered))
    if size <= 0:
        return set()
    return set(ordered.tail(size)["code"].astype(str))


def portfolio_turnover(
    evaluation: pd.DataFrame, horizon: int = TURNOVER_HORIZON, top_n: int | None = None,
) -> dict[str, Any]:
    """留出集上 Top 组合的换手率与费用后收益。纯函数。

    口径说明（不写清楚这几个数字会被误读）：

    * 只在**每 horizon 个交易日采样一次**上算。``forward_return_h`` 是 h 日前向重叠收益，
      逐日相邻地比较 Top 成员，会把同一段持仓期数成多次调仓，换手率与费用都虚高。
    * 单边换手 ``t = |本期 Top - 上期 Top| / |本期 Top|``，首期记 0（没有上一期可比）。
    * 单期费用 ``= t × (买入费率 + 卖出费率)``：换出去的那部分要卖一次、换进来的要买一次。
    * 收益为**每期的超额收益**，单位是「一个持仓期（horizon 个交易日）」，不是年化。
      费用后收益 = 毛超额 − 单期费用。
    * 费率取自 ``services/backtest.py::cost_rates``，与回测、组合回测同一套，
      这里不另写一遍费用公式。
    """
    if evaluation is None or evaluation.empty:
        return {"status": "留出集没有可用样本，未计算换手率与费用后收益", "periods": 0}
    top_n = int(settings.PORTFOLIO_TOP_N if top_n is None else top_n)
    if top_n < 1:
        raise ValueError("留出集 Top N 必须为正整数。")
    buy_cost, sell_cost = cost_rates()
    frame = evaluation.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    dates = sorted(frame["date"].unique())
    # 按持仓期采样，并丢掉最后一个不完整持仓期：它的前向收益本来就不该被当成已实现。
    sampled = [dates[index] for index in range(0, max(0, len(dates) - int(horizon)), int(horizon))]
    per_period: list[dict[str, Any]] = []
    previous: set[str] | None = None
    for date in sampled:
        group = frame.loc[frame["date"].eq(date)]
        if group.empty:
            continue
        members = _top_members(group, top_n)
        if not members:
            continue
        turnover = 0.0 if previous is None else len(members - previous) / len(members)
        per_period.append({
            "date": str(pd.Timestamp(date).date()),
            "members": len(members),
            "turnover": turnover,
            "gross_excess": float(group.loc[group["code"].astype(str).isin(members), "actual"].mean()),
        })
        previous = members
    if len(per_period) < 2:
        return {
            "status": f"留出集内完整的 {horizon} 日持仓期不足 2 个，无法计算换手率与费用后收益",
            "periods": len(per_period),
        }
    frame_periods = pd.DataFrame(per_period)
    # 首期换手记 0 只是「没有上一期可比」，不能算进平均，否则会系统性低估真实换手。
    measured = frame_periods["turnover"].iloc[1:]
    average_turnover = float(measured.mean())
    per_period_cost = average_turnover * (buy_cost + sell_cost)
    gross = float(frame_periods["gross_excess"].mean())
    return {
        "status": "已计算",
        "horizon": int(horizon),
        "periods": len(per_period),
        "top_n": top_n,
        "average_turnover": average_turnover,
        "annualized_turnover": average_turnover * 252.0 / float(horizon),
        "gross_excess_return": gross,
        "per_period_cost": per_period_cost,
        "after_cost_return": gross - per_period_cost,
        "cost_rates": {"buy": buy_cost, "sell": sell_cost},
        "unit": f"每期 {horizon} 个交易日的超额收益，非年化",
    }


def build_holdout_evaluation(
    panel: pd.DataFrame, context: dict[str, Any] | None = None, progress: Any = None,
) -> dict[str, Any]:
    """在留出集上评估冻结模型，返回报告字典。不写库、不落盘，便于用构造数据验证。

    每个周期单独训练一个冻结模型（只用留出集之前的数据）。冻结模型若在训练期就没能
    通过样本外检验，就如实记「无法评估」并跳过——**绝不退回使用生产模型**，
    生产模型见过留出集，用它评估等于自己给自己判卷。
    """
    if panel is None or panel.empty:
        raise FinalHoldoutError("本地面板为空，无法进行最终留出集评估。")
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    start, end = holdout_window(frame)
    holdout = frame.loc[frame["date"].between(start, end)]
    if holdout.empty:
        raise FinalHoldoutError(f"留出集区间 {start.date()} ~ {end.date()} 内没有任何截面数据。")
    holdout_dates = int(holdout["date"].nunique())
    if holdout_dates < 2:
        raise FinalHoldoutError(f"留出集内只有 {holdout_dates} 个交易日，样本不足以评估。")

    objective = (context or {}).get("objective_mode") or settings.DEFAULT_OBJECTIVE_MODE
    # 用 is None 判空而不是 or：窗口 0 在扫描页上的含义是「用上全部历史」，
    # 写成 or 会把它悄悄换成 750，冻结模型于是被套上一个用户明确关掉的窗口约束。
    configured_window = (context or {}).get("training_window_days")
    window = int(settings.TRAINING_WINDOW_DAYS if configured_window is None else configured_window)
    results: dict[str, Any] = {}
    # 逐日明细只在本函数内流转：它要进报告 JSON 会被序列化失败，
    # 要随 oos_metrics 落盘又会让每次扫描产出的 json 白白变大。
    evaluations: dict[str, pd.DataFrame] = {}
    for horizon in HORIZONS:
        target, endpoint = f"target_{horizon}", f"label_end_{horizon}"
        line = {"horizon": horizon, "usable": False, "status": "", "metrics": None,
                "training": None, "train_baseline": None, "model_results": {}}
        if target not in frame.columns:
            reason = f"面板中没有 {target} 列，该周期无法评估"
            line["status"] = reason
            for model_name in MODEL_NAMES:
                line["model_results"][model_name] = {
                    "model": model_name, "usable": False, "status": reason, "metrics": None,
                    "training": None, "train_baseline": None,
                }
            line["ensemble"] = {
                "usable": False, "status": reason, "metrics": None,
            }
            results[str(horizon)] = line
            continue
        evaluation_by_model: dict[str, pd.DataFrame] = {}
        normalized_by_model: dict[str, pd.DataFrame] = {}
        for model_name in MODEL_NAMES:
            if progress:
                progress(f"正在训练 {horizon} 日 {model_name} 冻结模型（只用留出集之前的数据）")
            frozen = fit_panel_model(frame, model_name, horizon, objective=objective, window=window, cutoff=start)
            model_line = {"model": model_name, "usable": False, "status": "", "metrics": None,
                          "training": None, "train_baseline": None}
            if not frozen.get("validated") or "estimator" not in frozen:
                model_line["status"] = f"冻结模型未通过训练期验证，留出集评估未进行：{frozen.get('status')}"
                line["model_results"][model_name] = model_line
                continue
            features = list(frozen["features_used"])
            baseline = training_baseline(frame, start, target, endpoint, features=features, window=window)
            subset = holdout.dropna(subset=[*features, target])
            if settings.FINAL_HOLDOUT_END:
                # 设了终点时，标签也必须落在终点之内，否则「留出集」会把终点之后的行情算进来。
                subset = subset.loc[subset[endpoint] <= end]
            required = max(MINIMUM_STOCKS, int(np.ceil(holdout["code"].nunique() * MINIMUM_COVERAGE)))
            enough_coverage = subset.groupby("date")["code"].transform("nunique").ge(required)
            subset = subset.loc[enough_coverage]
            if subset.empty or subset["date"].nunique() < 2:
                model_line["status"] = "留出集内可用样本不足，无法评估"
                line["model_results"][model_name] = model_line
                continue
            evaluation = subset[["date", "code"]].copy()
            evaluation["prediction"] = np.asarray(frozen["estimator"].predict(subset[features]), dtype=float)
            evaluation["actual"] = subset[target].to_numpy()
            evaluation["absolute_return"] = subset[f"forward_return_{horizon}"].to_numpy()
            evaluation["baseline"] = baseline
            evaluation_by_model[model_name] = evaluation
            normalized_by_model[model_name] = normalize_predictions(
                evaluation[["date", "code", "prediction"]]
            ).dropna(subset=["prediction_zscore"])
            metrics = evaluate_oos(
                evaluation, horizon, total_stocks=int(subset["code"].nunique()), score_kind=frozen["score_kind"],
            )
            folds = (frozen.get("oos_metrics") or {}).get("folds") or []
            available_dates = int(frozen.get("training_dates") or 0)
            model_line.update(
                usable=True,
                status=frozen.get("status"),
                # 逐日明细必须留在这里：报告使用 allow_nan=False 落盘，因此只保留汇总指标。
                metrics={key: value for key, value in metrics.items() if key != "daily_groups"},
                train_baseline=baseline,
                training={
                    "available_dates": available_dates,
                    "window_days": window,
                    "window_sufficient": bool(window <= 0 or available_dates >= window),
                    "first_fold_from": folds[0].get("trained_from") if folds else None,
                    "last_fold_through": folds[-1].get("trained_through") if folds else None,
                },
            )
            line["model_results"][model_name] = model_line
        ensemble = {"usable": False, "status": "并非所有配置模型均有可用留出预测", "metrics": None}
        fused = fuse_normalized_predictions(normalized_by_model, MODEL_NAMES)
        if not fused.empty:
            truth = pd.concat(
                [
                    evaluation[["date", "code", "actual", "absolute_return", "baseline"]]
                    for evaluation in evaluation_by_model.values()
                ],
                ignore_index=True,
            ).drop_duplicates(["date", "code"], keep="first")
            ensemble_evaluation = truth.merge(fused[["date", "code", "score"]],
                                              on=["date", "code"], how="inner", validate="one_to_one")
            ensemble_evaluation["prediction"] = ensemble_evaluation["score"]
            score_models = [name for name in MODEL_NAMES if name in normalized_by_model]
            ensemble_metrics = evaluate_oos(
                ensemble_evaluation, horizon, total_stocks=int(ensemble_evaluation["code"].nunique()),
                score_kind=SCORE_KIND_RANK,
            )
            ensemble.update(
                usable=True,
                status="按生产选股口径外连接可用模型的逐日百分位并取均值",
                model_names=score_models,
                metrics={key: value for key, value in ensemble_metrics.items() if key != "daily_groups"},
            )
            evaluations[str(horizon)] = ensemble_evaluation
        line["ensemble"] = ensemble
        summary_model = next(
            (name for name in MODEL_NAMES if line["model_results"].get(name, {}).get("usable")), None
        )
        if summary_model is not None:
            summary = line["model_results"][summary_model]
            line.update(
                usable=True, status=summary["status"], metrics=summary["metrics"],
                training=summary["training"], train_baseline=summary["train_baseline"],
                summary_model=summary_model,
            )
            if not ensemble["usable"]:
                evaluations[str(horizon)] = evaluation_by_model[summary_model]
        else:
            reasons = [
                f"{name}：{result['status']}"
                for name, result in line["model_results"].items()
            ]
            line["status"] = "；".join(reasons) or "所有冻结模型均未生成可用留出评估"
        results[str(horizon)] = line

    usable = {key: value for key, value in results.items() if value["usable"]}
    if not usable:
        reasons = "；".join(f"{key} 日：{value['status']}" for key, value in results.items())
        raise FinalHoldoutError(f"没有任何周期的冻结模型可用于留出集评估。{reasons}")

    if str(TURNOVER_HORIZON) in evaluations:
        turnover = portfolio_turnover(evaluations[str(TURNOVER_HORIZON)], TURNOVER_HORIZON)
    else:
        turnover = {"status": f"留出集内 {TURNOVER_HORIZON} 日周期不可用，未计算换手率与费用后收益"}
    model_checks = [
        (key, name, model_result)
        for key, value in sorted(results.items(), key=lambda item: int(item[0]))
        for name, model_result in value["model_results"].items()
    ]
    ensemble_checks = [
        (key, "ensemble", value["ensemble"])
        for key, value in sorted(results.items(), key=lambda item: int(item[0]))
        if "ensemble" in value
    ]
    all_checks = sorted(model_checks + ensemble_checks, key=lambda item: (int(item[0]), item[1]))
    passed = bool(all_checks) and all(
        bool(model_result["usable"] and model_result["metrics"].get("validated"))
        for _, _, model_result in all_checks
    )
    failed = next((item for item in all_checks
                   if not item[2]["usable"] or not item[2]["metrics"].get("validated")), None)
    worst = failed[0] if failed else None
    failed_model = failed[1] if failed else None
    return {
        "model_version": (context or {}).get("model_version") or settings.MODEL_VERSION,
        "context_id": (context or {}).get("context_id"),
        "holdout_start": str(start.date()),
        "holdout_end": str(end.date()),
        "evaluated_at": datetime.now(SHANGHAI).isoformat(timespec="seconds"),
        "holdout_dates": holdout_dates,
        "objective_mode": objective,
        "training_window_days": window,
        "passed": bool(passed),
        "failed_horizon": worst,
        "failed_model": failed_model,
        "models": list(MODEL_NAMES),
        "horizons": results,
        "turnover": turnover,
        "summary": finite_or_none({
            "holdout_dates": holdout_dates,
            "passed": bool(passed),
            "failed_horizon": worst,
            "failed_model": failed_model,
            "rank_ic": {key: value["metrics"].get("rank_ic") for key, value in usable.items()},
            "model_rank_ic": {
                name: {
                    key: model_result["metrics"].get("rank_ic") if model_result["usable"] else None
                    for key, value in results.items()
                    for model_result in [value["model_results"].get(name, {})]
                    if model_result
                }
                for name in MODEL_NAMES
            },
            "ensemble_rank_ic": {
                key: value["ensemble"]["metrics"].get("rank_ic") if value["ensemble"]["usable"] else None
                for key, value in results.items()
                if "ensemble" in value
            },
            "ndcg_ratio": {key: value["metrics"].get("ndcg_ratio") for key, value in usable.items()},
            "top_excess_return": {key: value["metrics"].get("top_excess_return") for key, value in usable.items()},
            "after_cost_return": turnover.get("after_cost_return"),
            "average_turnover": turnover.get("average_turnover"),
        }),
        "limitation": (context or {}).get("limitation") or (
            "截面仅含本次成功采集历史的股票池；当前股票池回溯存在幸存者偏差，"
            "分组收益为持有期重叠的诊断指标，非可执行组合净值。"
        ),
        "notes": [
            "留出集为「评估专用」：冻结模型只用留出集之前的数据训练，日常生产模型按滚动窗口使用全量历史，"
            "因此本报告衡量的是建模流程，不是今天线上那个模型。",
            "逐一报告配置内各模型；留出集报告本身不能证明研发人员此前未查看该区间或未据此选择方案。",
            "换手率与费用后收益按每 20 个交易日采样的非重叠持仓期计算，单边换手口径为成员变更比例；"
            "费率与回测、组合回测同源（services/backtest.py::cost_rates）。",
            "冻结模型训练不使用留出区间；本报告无法证明函数外的研发活动未查看或使用该区间。",
        ],
    }


# ---------------------------------------------------------------------------
# 数据库：留出集记录
# ---------------------------------------------------------------------------

def final_holdout_record(repository: QuantRepository, model_version: str | None = None) -> dict[str, Any] | None:
    """读取某模型版本已有的留出集评估记录；没有则返回 None。"""
    version = model_version or settings.MODEL_VERSION
    with repository.connect() as connection:
        return connection.execute(
            "SELECT * FROM public.quant_model_final_holdout WHERE model_version=%s", (version,)
        ).fetchone()


def log_governance_event(
    repository: QuantRepository, operation: str, operator: str = "", model_version: str | None = None,
    context_id: str | None = None, before: Any = None, after: Any = None, note: str | None = None,
) -> None:
    """记录一次治理操作，谁在什么时候把哪个模型换成了什么。"""
    from psycopg.types.json import Jsonb

    with repository.connect() as connection:
        connection.execute(
            "INSERT INTO public.quant_model_governance_events"
            "(operation,operator,model_version,context_id,before_value,after_value,note) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (operation, operator or "", model_version, context_id,
             Jsonb(finite_or_none(before)) if before is not None else None,
             Jsonb(finite_or_none(after)) if after is not None else None, note),
        )


def evaluate_final_holdout(
    repository: QuantRepository, model_version: str | None = None, operator: str = "",
    force: bool = False, context: dict[str, Any] | None = None, panel: pd.DataFrame | None = None,
    progress: Any = None,
) -> dict[str, Any]:
    """在所有模型冻结后运行一次的最终留出集评估。

    固定留出期按起点加数据库事务级 advisory lock 实现一次性评估，跨模型版本也不允许重跑。
    force 参数为兼容旧入口保留，但不能覆盖已经查看过的最终留出集。
    """
    if not settings.ENABLE_FINAL_HOLDOUT:
        raise FinalHoldoutError(
            "最终留出集未启用（settings.ENABLE_FINAL_HOLDOUT 为 False），拒绝评估。"
        )
    version = model_version or settings.MODEL_VERSION
    if context is None or panel is None:
        loaded_context, loaded_panel = load_context(include_panel=True)
        if not loaded_context or loaded_panel is None or loaded_panel.empty:
            raise FinalHoldoutError("尚未生成可用的模型上下文与截面面板，请先完成一次全量扫描。")
        context = context or loaded_context
        panel = loaded_panel if panel is None else panel
    if context.get("model_version") != version:
        raise FinalHoldoutError(
            f"当前模型上下文的版本是 {context.get('model_version')}，与请求评估的 {version} 不一致。"
        )

    from psycopg.types.json import Jsonb

    with repository.connect() as connection:
        connection.execute("SELECT pg_advisory_xact_lock(76625018433586)")
        start = pd.Timestamp(settings.FINAL_HOLDOUT_START).date()
        report_path = _report_path()
        if report_path.exists():
            try:
                prior_report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise FinalHoldoutError(
                    "最终留出集报告文件存在但无法验证；为避免重复查看留出集，拒绝重评。"
                ) from exc
            if not isinstance(prior_report, dict):
                raise FinalHoldoutError("最终留出集报告格式异常，拒绝重评。")
            if prior_report.get("holdout_start") == str(start):
                raise FinalHoldoutError(
                    f"最终留出集起点 {start} 已有报告文件记录；即使数据库记录缺失也不能重评。"
                )
        existing = connection.execute(
            "SELECT * FROM public.quant_model_final_holdout WHERE holdout_start=%s "
            "ORDER BY evaluated_at LIMIT 1",
            (start,),
        ).fetchone()
        if existing:
            raise FinalHoldoutError(
                f"最终留出集起点 {start} 已由模型版本 {existing['model_version']} 评估。"
                "为避免反复查看同一留出集，--force 也不能重评或覆盖。"
            )
        report = build_holdout_evaluation(panel, context, progress=progress)
        report["operator"] = operator or ""
        report["forced"] = False
        payload = finite_or_none(report)
        row = connection.execute(
            "INSERT INTO public.quant_model_final_holdout"
            "(model_version,context_id,holdout_start,holdout_end,operator,forced,passed,metrics,report_path) "
            "VALUES (%s,%s,%s,%s,%s,FALSE,%s,%s,%s) ON CONFLICT (model_version) DO NOTHING RETURNING id",
            (version, report["context_id"], report["holdout_start"], report["holdout_end"],
             operator or "", bool(report["passed"]), Jsonb(payload), str(_report_path())),
        ).fetchone()
        if row is None:
            raise FinalHoldoutError(f"模型版本 {version} 已存在留出集记录，本次结果未保存。")
    _write_report(payload)
    log_governance_event(
        repository, "最终留出集评估",
        operator=operator, model_version=version, context_id=report["context_id"],
        after={"passed": report["passed"], "holdout_start": report["holdout_start"],
               "holdout_end": report["holdout_end"]},
    )
    return payload


def _write_report(payload: dict[str, Any]) -> Path:
    """原子写入留出集报告。写临时文件再 os.replace，避免页面读到写了一半的 JSON。"""
    path = _report_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)
    return path


def holdout_report() -> dict[str, Any] | None:
    """读取当前模型版本的留出集报告，避免跨版本展示验收结论。"""
    path = _report_path()
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        LOGGER.exception("读取最终留出集报告失败")
        return None
    if not isinstance(loaded, dict) or loaded.get("model_version") != settings.MODEL_VERSION:
        return None
    return loaded


# ---------------------------------------------------------------------------
# 归档与回滚
# ---------------------------------------------------------------------------

def _link_or_copy(source: Path, target: Path) -> None:
    """优先硬链接，跨卷或文件系统不支持时退回复制。

    硬链接在这里是安全的：``save_context`` 从不原地改写文件，而是写临时文件再
    ``os.replace``，所以归档里的链接会一直指向被替换掉的那份旧内容。
    """
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _archive_summary_matches(directory: Path, version: str, context_id: str) -> bool:
    """校验归档摘要所属批次，拒绝使用被后续训练覆盖的历史元数据。"""
    path = directory / f"market_context_v{version}.json"
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
        return isinstance(summary, dict) and summary.get("context_id") == context_id and summary.get("model_version") == version
    except (OSError, ValueError):
        LOGGER.warning("模型摘要缺失或损坏：%s", path)
        return False


def archive_context(
    repository: QuantRepository, context: dict[str, Any], panel: pd.DataFrame | None,
    passed: bool, base_rate: float | None = None,
) -> dict[str, Any]:
    """把当前活动模型归档并登记。

    归档让「回滚到上一通过模型」有东西可回：活动文件恒等于最新一次产出，
    历史版本只存在于归档目录。**只有至少一个模型通过验证的产出会被登记为可回滚候选**，
    与 scan_daily.py「不拿空结果覆盖好结果」是同一个立场。
    """
    context_id = str(context.get("context_id") or "")
    if not context_id:
        raise FinalHoldoutError("模型上下文缺少 context_id，无法归档。")
    directory = settings.MODEL_DIR / settings.MODEL_ARCHIVE_DIRNAME / context_id
    directory.mkdir(parents=True, exist_ok=True)
    version = context.get("model_version") or settings.MODEL_VERSION
    sources = {
        "model": settings.MODEL_DIR / f"market_context_v{version}.joblib",
        "summary": settings.MODEL_DIR / f"market_context_v{version}.json",
        "panel": settings.DATA_DIR / f"ml_panel_v{version}.parquet",
    }
    missing = [name for name, path in sources.items() if not path.exists()]
    if missing:
        raise FinalHoldoutError(f"归档失败：活动文件缺失 {', '.join(missing)}。")
    if not _archive_summary_matches(settings.MODEL_DIR, version, context_id):
        raise FinalHoldoutError("归档失败：活动摘要与待归档的模型批次或版本不一致。")
    targets = {
        "model": directory / f"market_context_v{version}.joblib",
        "summary": directory / f"market_context_v{version}.json",
        "panel": directory / f"ml_panel_v{version}.parquet",
    }
    for name, source in sources.items():
        _link_or_copy(source, targets[name])
    # 模型目录与个股分析页靠这些字段展示「这份模型是什么口径训练出来的」，
    # 不读 joblib：反序列化几十兆的上下文只为拿几个标量没有意义。
    source_meta = context.get("source") or {}
    details = {
        "objective_mode": context.get("objective_mode"),
        "training_window_days": context.get("training_window_days"),
        "regression_loss": context.get("regression_loss"),
        "label_winsor_upper_quantile": context.get("label_winsor_upper_quantile"),
        "pool_mode": source_meta.get("pool_mode"),
        "universe_size": len(context.get("universe_codes") or []),
        "neutralization_status": (context.get("neutralization") or {}).get("status"),
        "extra_factors_status": (context.get("extra_factors") or {}).get("status"),
        "source_snapshot_id": source_meta.get("snapshot_id"),
        "limitation": context.get("limitation"),
    }
    (directory / "registry.json").write_text(
        json.dumps(finite_or_none({
            "context_id": context_id, "model_version": version,
            "created_at": context.get("created_at"), "reference_date": context.get("reference_date"),
            "passed": bool(passed), "base_rate": base_rate, **details,
        }), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    with repository.connect() as connection:
        connection.execute(
            "INSERT INTO public.quant_model_registry"
            "(context_id,model_version,reference_date,passed,base_rate,archive_path,panel_path,is_active,details) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,TRUE,%s) "
            "ON CONFLICT (context_id) DO UPDATE SET passed=EXCLUDED.passed, base_rate=EXCLUDED.base_rate, "
            "archive_path=EXCLUDED.archive_path, panel_path=EXCLUDED.panel_path, details=EXCLUDED.details, "
            "is_active=TRUE",
            (context_id, version, context.get("reference_date"), bool(passed), base_rate,
             str(directory), str(sources["panel"]), _jsonb(details)),
        )
        connection.execute(
            "UPDATE public.quant_model_registry SET is_active=FALSE, retired_at=now() "
            "WHERE is_active AND context_id<>%s",
            (context_id,),
        )
    _prune_archive(repository)
    return {"context_id": context_id, "archive_path": str(directory), "passed": bool(passed)}


def _jsonb(value: Any) -> Any:
    from psycopg.types.json import Jsonb

    return Jsonb(finite_or_none(value) or {})


def _prune_archive(repository: QuantRepository, keep: int = settings.MODEL_ARCHIVE_LIMIT) -> int:
    """裁剪归档目录，只保留最近 keep 个可回滚版本。活动版本无论多旧都不删。"""
    with repository.connect() as connection:
        rows = connection.execute(
            "SELECT context_id, archive_path FROM public.quant_model_registry "
            "WHERE passed AND NOT is_active ORDER BY created_at DESC OFFSET %s",
            (int(keep),),
        ).fetchall()
    removed = 0
    for row in rows:
        path = Path(row["archive_path"]) if row["archive_path"] else None
        if path and path.exists() and path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        removed += 1
        with repository.connect() as connection:
            connection.execute(
                "UPDATE public.quant_model_registry SET archive_path=NULL, retired_at=COALESCE(retired_at, now()) "
                "WHERE context_id=%s",
                (row["context_id"],),
            )
    return removed


def registry_versions(repository: QuantRepository, limit: int = 50) -> list[dict[str, Any]]:
    """模型版本列表，新到旧。``rollbackable`` 由归档文件是否真的存在决定，不靠登记的布尔值。"""
    with repository.connect() as connection:
        rows = connection.execute(
            "SELECT * FROM public.quant_model_registry ORDER BY created_at DESC LIMIT %s",
            (min(max(int(limit), 1), 500),),
        ).fetchall()
    result = []
    for row in rows:
        path = Path(row["archive_path"]) if row["archive_path"] else None
        # 登记的路径可能已经不存在（手工清理、磁盘迁移）。回滚按钮必须据此禁用，
        # 而不是等点了以后再报错。校验的三个文件与 rollback_to 一致，否则会出现
        # 「按钮点得动、点了报错」。
        available = bool(path and row["model_version"] == settings.MODEL_VERSION and all(
            (path / name).exists()
            for name in (f"market_context_v{row['model_version']}.joblib",
                         f"market_context_v{row['model_version']}.json",
                         f"ml_panel_v{row['model_version']}.parquet")
        ) and _archive_summary_matches(path, row["model_version"], row["context_id"]))
        result.append({**row, "rollbackable": bool(row["passed"] and not row["is_active"] and available)})
    return result


def rollback_to(repository: QuantRepository, context_id: str, operator: str) -> dict[str, Any]:
    """把活动模型换回某个历史版本。管理员操作，全程留审计。

    换之前先把**当前**活动版本归档一次：不然回滚本身就变成了不可逆动作，
    点错了就再也回不来——而「可回滚」正是本次改造要给出的承诺。
    """
    with repository.connect() as connection:
        target = connection.execute(
            "SELECT * FROM public.quant_model_registry WHERE context_id=%s", (context_id,)
        ).fetchone()
    if target is None:
        raise FinalHoldoutError(f"没有找到模型版本 {context_id} 的登记记录。")
    if not target["passed"]:
        raise FinalHoldoutError("该版本没有通过样本外验证，不能回滚到它。")
    if target["is_active"]:
        raise FinalHoldoutError("该版本已经是当前活动模型，无需回滚。")
    version = target["model_version"]
    if version != settings.MODEL_VERSION:
        raise FinalHoldoutError("该归档的训练版本与当前程序不兼容，不能作为当前活动模型回滚。")
    directory = Path(target["archive_path"]) if target["archive_path"] else None
    if directory is None or not directory.exists():
        raise FinalHoldoutError("该版本的归档目录已不存在，无法回滚。")
    archived_model = directory / f"market_context_v{version}.joblib"
    archived_panel = directory / f"ml_panel_v{version}.parquet"
    archived_summary = directory / f"market_context_v{version}.json"
    if not archived_model.exists() or not archived_panel.exists() or not archived_summary.exists():
        raise FinalHoldoutError("该版本的归档文件不完整（模型、面板或摘要缺失），无法回滚。")
    if not _archive_summary_matches(directory, version, context_id):
        raise FinalHoldoutError("该归档摘要与模型批次或版本不一致，请先修复摘要。")

    context, panel = load_context(include_panel=True)
    if context and panel is not None:
        try:
            archive_context(repository, context, panel, passed=bool(
                any(model.get("validated") for models in (context.get("models") or {}).values() for model in models)
            ))
        except (FinalHoldoutError, OSError):
            LOGGER.exception("回滚前归档当前活动模型失败，继续回滚但当前版本可能无法再回来")

    active_model = settings.MODEL_DIR / f"market_context_v{version}.joblib"
    active_panel = settings.DATA_DIR / f"ml_panel_v{version}.parquet"
    # 摘要 JSON 也要一起换回来：它是归档的组成部分，只换模型和面板会留下一份描述着
    # 「被换掉的那个版本」的摘要，下一次归档再把它当成本版本的产物存进去。
    active_summary = settings.MODEL_DIR / f"market_context_v{version}.json"
    # 复制到临时文件再 os.replace：页面随时可能在读，不能让它读到半个文件。
    for source, destination in (
        (archived_model, active_model), (archived_panel, active_panel), (archived_summary, active_summary)
    ):
        temporary = destination.with_suffix(".rollback.tmp")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    with repository.connect() as connection:
        connection.execute(
            "UPDATE public.quant_model_registry SET is_active=FALSE, retired_at=now() WHERE is_active"
        )
        connection.execute(
            "UPDATE public.quant_model_registry SET is_active=TRUE, retired_at=NULL WHERE context_id=%s",
            (context_id,),
        )
    log_governance_event(
        repository, "回滚模型", operator=operator, model_version=version, context_id=context_id,
        before={"context_id": (context or {}).get("context_id")},
        after={"context_id": context_id, "archive_path": str(directory)},
        note="管理员手动回滚到上一通过版本",
    )
    return {"context_id": context_id, "model_version": version, "reference_date": target["reference_date"]}


def delete_archived_model(repository: QuantRepository, context_id: str, operator: str) -> dict[str, Any]:
    """删除非活动模型归档，保留模型登记和治理审计。"""
    if not isinstance(context_id, str) or not context_id.strip():
        raise FinalHoldoutError("模型版本 ID 无效，无法删除。")
    context_id = context_id.strip()
    with repository.connect() as connection:
        # 在同一事务内锁定登记、删除文件并清空路径，避免回滚操作与删除并发交错。
        target = connection.execute(
            "SELECT * FROM public.quant_model_registry WHERE context_id=%s FOR UPDATE",
            (context_id,),
        ).fetchone()
        if target is None:
            raise FinalHoldoutError(f"没有找到模型版本 {context_id} 的登记记录。")
        if target["is_active"]:
            raise FinalHoldoutError("当前活动模型不能删除，请先生成或回滚到其他模型。")

        archive_root = (settings.MODEL_DIR / settings.MODEL_ARCHIVE_DIRNAME).resolve()
        archive_path = Path(target["archive_path"]).resolve() if target["archive_path"] else None
        if archive_path is not None:
            try:
                archive_path.relative_to(archive_root)
            except ValueError as exc:
                raise FinalHoldoutError("模型归档路径不在受控目录内，拒绝删除。") from exc
            if archive_path.name != context_id:
                raise FinalHoldoutError("模型归档路径与模型 ID 不一致，拒绝删除。")
            if archive_path.exists():
                if not archive_path.is_dir():
                    raise FinalHoldoutError("模型归档路径不是目录，拒绝删除。")
                shutil.rmtree(archive_path)

        connection.execute(
            "UPDATE public.quant_model_registry SET archive_path=NULL,panel_path=NULL,"
            "retired_at=COALESCE(retired_at,now()) WHERE context_id=%s AND is_active=FALSE",
            (context_id,),
        )
    log_governance_event(
        repository,
        "删除模型归档",
        operator=operator,
        model_version=target["model_version"],
        context_id=context_id,
        before={"archive_path": target["archive_path"], "passed": target["passed"]},
        after={"archive_path": None},
        note="管理员手动删除非活动模型归档，保留登记记录以便审计。",
    )
    return {"context_id": context_id, "model_version": target["model_version"]}


# ---------------------------------------------------------------------------
# 降级状态
# ---------------------------------------------------------------------------

def model_status(repository: QuantRepository, model_version: str | None = None) -> dict[str, Any]:
    """当前模型的降级状态。没有记录就是未降级——不是「未知」。"""
    version = model_version or settings.MODEL_VERSION
    with repository.connect() as connection:
        row = connection.execute(
            "SELECT * FROM public.quant_model_health_state WHERE model_version=%s", (version,)
        ).fetchone()
    if row is None:
        return {"model_version": version, "degraded": False, "reason": None,
                "since": None, "updated_by": ""}
    row = dict(row)
    row["scoring_message"] = DEGRADED_SCORING_MESSAGE if row["degraded"] else None
    return row


def current_model_status(repository: QuantRepository) -> dict[str, Any]:
    """页面用的降级状态；读库失败时返回「未降级 + 错误说明」。

    与 :meth:`ScanGovernance.scoring_paused` 同样的取舍：数据库读不到时页面照常可用，
    但 ``error`` 字段非空，页面必须把它显示出来——否则一次数据库故障会表现为
    「模型一切正常」，而实际上这一次根本没有治理数据。
    """
    try:
        status = model_status(repository)
    except DatabaseError as exc:
        return {"degraded": False, "reason": None, "since": None, "updated_by": "",
                "scoring_message": None, "error": f"读取模型降级状态失败：{exc}"}
    return {**status, "error": None}


def degrade_model(
    repository: QuantRepository, reason: str, operator: str = "scheduled",
    model_version: str | None = None,
) -> dict[str, Any]:
    """把模型标成 degraded。自动动作只到这里为止，回滚仍然由管理员决定。"""
    version = model_version or settings.MODEL_VERSION
    with repository.connect() as connection:
        connection.execute(
            "INSERT INTO public.quant_model_health_state(model_version,degraded,reason,since,updated_by) "
            "VALUES (%s,TRUE,%s,now(),%s) ON CONFLICT (model_version) DO UPDATE SET "
            "degraded=TRUE, reason=EXCLUDED.reason, since=now(), updated_by=EXCLUDED.updated_by, updated_at=now()",
            (version, reason[:2000], operator),
        )
    log_governance_event(repository, "模型降级", operator=operator, model_version=version,
                         after={"degraded": True, "reason": reason})
    LOGGER.warning("模型 %s 已降级：%s", version, reason)
    return model_status(repository, version)


def clear_degradation(
    repository: QuantRepository, operator: str, note: str = "", model_version: str | None = None
) -> dict[str, Any]:
    """解除降级。只能由管理员执行，系统不会自动解除。

    不自动解除是有意的：如果一次误报把模型降级后下一轮扫描又自动恢复，降级就等于没有；
    而如果被一次误报永久卡死，那是管理员看得到、点得动的事，比悄悄恢复要安全。
    """
    version = model_version or settings.MODEL_VERSION
    before = model_status(repository, version)
    with repository.connect() as connection:
        connection.execute(
            "INSERT INTO public.quant_model_health_state(model_version,degraded,reason,since,updated_by) "
            "VALUES (%s,FALSE,NULL,NULL,%s) ON CONFLICT (model_version) DO UPDATE SET "
            "degraded=FALSE, reason=NULL, since=NULL, updated_by=EXCLUDED.updated_by, updated_at=now()",
            (version, operator),
        )
    log_governance_event(repository, "解除降级", operator=operator, model_version=version,
                         before={"degraded": before["degraded"], "reason": before["reason"]},
                         after={"degraded": False}, note=note or None)
    return model_status(repository, version)


def recent_alerts(repository: QuantRepository, limit: int = 50, only_open: bool = False) -> list[dict[str, Any]]:
    """最近的告警，供管理员页展示。"""
    clause = "WHERE NOT acknowledged" if only_open else ""
    with repository.connect() as connection:
        return connection.execute(
            f"SELECT * FROM public.quant_model_alerts {clause} ORDER BY created_at DESC, id DESC LIMIT %s",
            (min(max(int(limit), 1), 500),),
        ).fetchall()


def acknowledge_alert(repository: QuantRepository, alert_id: int, operator: str) -> None:
    """标记告警已阅。"""
    with repository.connect() as connection:
        connection.execute(
            "UPDATE public.quant_model_alerts SET acknowledged=TRUE, acknowledged_at=now(), acknowledged_by=%s "
            "WHERE id=%s AND NOT acknowledged",
            (operator, int(alert_id)),
        )


# ---------------------------------------------------------------------------
# 扫描集成
# ---------------------------------------------------------------------------

class ScanGovernance:
    """扫描期间挂在 ``run_scan_pipeline`` 上的治理钩子。

    设计要点是**绝不把扫描搞挂**：治理是扫描的附加动作，不是前置条件。
    数据库连不上、某个漂移维度算不出来、告警通道挂了，都不该让当天的评分没有产出。
    但同样不能**静默**——每一次降级失败都记进 :attr:`errors`，由扫描管线写成一条
    「血缘说明」显示在页面上，让人看到「这次治理数据没写进去」而不是以为写了。

    构造时不连库；只有 :meth:`scoring_paused` 与 :meth:`after_scan` 会真正访问数据库。
    ``services/scan_pipeline.py`` 只按这两个方法名调用它（鸭子类型），
    定时任务与网页端传同一份实现，因此两边的治理行为不会各走一路。
    """

    def __init__(self, repository: QuantRepository, trigger: str = "manual", operator: str = ""):
        self.repository = repository
        self.trigger = trigger
        self.operator = operator
        self.errors: list[str] = []
        self.status: dict[str, Any] = {"degraded": False, "reason": None}

    def scoring_paused(self) -> bool:
        """扫描开始前读一次降级状态。读不到时按「未降级」继续，但记下原因。

        这里选择「读不到就照常出分」而不是「读不到就不出分」：数据库短暂不可用时
        整站都不该跟着停摆，而这一情形会被记进 errors 并列在页面的血缘说明里。
        """
        try:
            self.status = model_status(self.repository)
        except DatabaseError as exc:
            self.status = {"degraded": False, "reason": None}
            self.errors.append(f"读取模型降级状态失败，本次按未降级处理：{exc}")
            return False
        return bool(self.status.get("degraded"))

    def after_scan(self, context: dict[str, Any] | None, panel: pd.DataFrame | None) -> dict[str, Any]:
        """扫描结束后写通过率与漂移、判定告警、必要时自动降级。"""
        summary: dict[str, Any] = {
            "trigger": self.trigger, "errors": self.errors,
            "validation_rows": 0, "feature_rows": 0, "concept_rows": 0, "prediction_rows": 0,
            "pass_rate": None, "alerts": 0, "alert_result": None, "degraded": False, "archive": None,
        }
        if not context:
            self.errors.append("本次扫描未生成模型上下文，通过率与漂移均未写入")
            summary["errors"] = self.errors
            return summary

        # 通过率与漂移各自独立 try：一边失败不该让另一边的数据也丢掉。
        records: list[dict[str, Any]] = []
        try:
            records = model_monitor.validation_records(context, trigger=self.trigger)
            summary["validation_rows"] = model_monitor.log_validation(self.repository, records)
            summary["pass_rate"] = model_monitor.get_pass_rate(self.repository, model_version=context.get("model_version"))
        except DatabaseError as exc:
            self.errors.append(f"通过率记录写入失败：{exc}")

        # 归档登记：模型健康页的「回滚候选」只可能来自这里，不登记就永远是空列表。
        # 放在告警之前：下面告警写入失败会提前 return，回滚候选不该被告警通道连坐——
        # 而且归档说的是「活动文件现在是哪一份」，与告警判定本来就无关。
        # 归档文件缺一不可（archive_context 会校验 joblib/json/parquet 三个都存在），
        # 所以它只会登记本轮真正落盘的产物。
        try:
            base_rate = (sum(1 for record in records if record["pass"]) / len(records)) if records else None
            summary["archive"] = archive_context(
                self.repository, context, panel,
                passed=any(record["pass"] for record in records), base_rate=base_rate,
            )
        except (FinalHoldoutError, OSError, DatabaseError) as exc:
            self.errors.append(f"模型归档失败，本版本不会出现在回滚候选里：{type(exc).__name__}: {exc}")

        feature_rows: list[dict[str, Any]] = []
        concept_rows: list[dict[str, Any]] = []
        prediction_rows: list[dict[str, Any]] = []
        try:
            if panel is not None and not panel.empty:
                feature_rows = drift_monitor.feature_drift_rows(panel)
                summary["feature_rows"] = drift_monitor.log_feature_drift(self.repository, feature_rows)
            report_date = str(context.get("reference_date") or "")
            # 上下文的周期键在训练产物里是 int（HORIZONS 就是 int），而在摘要 JSON 里是 str。
            # 统一按 str 取，否则这里每次都取到空列表，概念漂移会静默地一条都不写。
            models_by_horizon = {str(key): value for key, value in (context.get("models") or {}).items()}
            for horizon in HORIZONS:
                metrics = next(
                    (model.get("oos_metrics") for model in models_by_horizon.get(str(horizon), [])
                     if model.get("validated") and model.get("oos_metrics")),
                    None,
                )
                row = drift_monitor.concept_drift_row(metrics, report_date)
                if row:
                    concept_rows.append(row)
            if concept_rows:
                summary["concept_rows"] = drift_monitor.log_concept_drift(self.repository, concept_rows)
            prediction_row = drift_monitor.prediction_drift_row(context, panel)
            if prediction_row:
                prediction_rows.append(prediction_row)
                summary["prediction_rows"] = drift_monitor.log_prediction_drift(self.repository, prediction_rows)
        except (DatabaseError, ValueError, KeyError) as exc:
            self.errors.append(f"漂移数据写入失败：{type(exc).__name__}: {exc}")

        alerts: list[dict[str, Any]] = []
        try:
            alerts.extend(model_monitor.check_pass_rate_alerts(self.repository))
        except DatabaseError as exc:
            self.errors.append(f"通过率告警判定失败：{exc}")
        if feature_rows or concept_rows or prediction_rows:
            alerts.extend(drift_monitor.evaluate_drift_alerts(feature_rows, concept_rows, prediction_rows))
        summary["alerts"] = len(alerts)
        if alerts:
            try:
                for alert in alerts:
                    alert.setdefault("model_version", settings.MODEL_VERSION)
                summary["alert_result"] = alerting.dispatch_alerts(self.repository, alerts)
            except DatabaseError as exc:
                self.errors.append(f"告警写入失败：{exc}")
                return summary

        # 自动动作到此为止：只标降级、只写告警。回滚永远要管理员点击。
        reason = alerting.degradation_reason(alerts)
        if reason and not self.status.get("degraded"):
            try:
                degrade_model(self.repository, reason, operator="scheduled")
                summary["degraded"] = True
                summary["degradation_reason"] = reason
            except DatabaseError as exc:
                self.errors.append(f"自动降级失败，模型仍按原状态输出评分：{exc}")
        summary["errors"] = self.errors
        return summary

    def notes(self, summary: dict[str, Any]) -> list[tuple[str, str]]:
        """把治理结论翻译成页面上的一行行说明。返回 (级别, 文本) 列表。"""
        lines: list[tuple[str, str]] = []
        if summary.get("validation_rows"):
            rate = summary.get("pass_rate") or {}
            text = f"模型治理：本轮写入 {summary['validation_rows']} 条检验记录"
            if rate.get("pass_rate") is not None:
                text += f"；近 {rate['days']} 天通过率 {rate['pass_rate']:.1%}（{rate['passed']}/{rate['records']}）"
            if rate.get("recent_failure"):
                text += f"；最近一次失败卡在「{rate['recent_failure']['reason']}」"
            lines.append(("caption", text))
        if summary.get("archive"):
            archive = summary["archive"]
            lines.append(("caption", (
                f"模型归档：context {str(archive.get('context_id'))[:12]}"
                + ("，已登记为回滚候选" if archive.get("passed") else "，本轮没有通过验证的模型，不进入回滚候选")
            )))
        if summary.get("feature_rows"):
            lines.append(("caption", (
                f"漂移监控：特征 {summary['feature_rows']} 项、"
                f"概念 {summary['concept_rows']} 项、预测 {summary['prediction_rows']} 项已写入"
            )))
        if summary.get("alerts"):
            lines.append(("warning", f"模型治理告警 {summary['alerts']} 条，详见“模型健康”页。"))
        if summary.get("degraded"):
            lines.append(("warning", f"模型已自动降级，暂停输出评分：{summary.get('degradation_reason') or ''}"))
        elif self.status.get("degraded"):
            lines.append(("warning", (
                f"模型当前处于降级状态，本次未输出模型评分：{self.status.get('reason') or '未记录原因'}"
            )))
        for message in summary.get("errors") or []:
            # 治理没能写入时必须显式说出来，不能让页面看起来一切正常。
            lines.append(("warning", f"模型治理未完成：{message}"))
        return lines
