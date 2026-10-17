"""模型通过率监控：把每轮训练的样本外检验结果落库，并从中读出趋势与告警。

要解决的问题：模型是变好了还是变坏了，原先没人知道。每轮训练的实际检验数值只随模型
上下文落盘，下一次扫描整个覆盖，看不出通过率趋势，也定位不到反复失败的是哪一道门槛。
本模块把每次检验追加写入 PostgreSQL，再由管理员页把它画成趋势。

分层：从数据库取数的函数（``get_pass_rate`` 等）与纯计算函数（``summarize_pass_rate`` 等）
分开。纯计算部分不碰数据库，需求文 §二-F 要求的「通过率统计是否正确」因此可以用构造数据
直接验证，不必依赖真实行情或真实数据库。

本模块刻意不引入 Streamlit：services/scan_pipeline.py 会调用它，而定时任务要在没有网页的
进程里跑同一套编排（tests/test_scan_pipeline.py 有一条子进程断言把这件事钉住）。
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any, Iterable, Sequence
from zoneinfo import ZoneInfo

import numpy as np

from config import settings
from services.database import QuantRepository
from services.ml_pipeline import evaluation_gates
from services.rank_model import SCORE_KIND_RANK

LOGGER = logging.getLogger("quant.model_monitor")
SHANGHAI = ZoneInfo("Asia/Shanghai")

# 模型名 → 需求文里的 model_type 代号。页面表格用的也是这一份映射。
MODEL_TYPE_CODES = {"LightGBM": "lgb", "XGBoost": "xgb"}
MODEL_TYPE_CODES_REVERSE = {value: key for key, value in MODEL_TYPE_CODES.items()}

# 门槛代号 → 给人看的一句话。failed_gate 落库时存代号，页面与告警用它翻译。
GATE_LABELS = {
    "dates": "样本外交易日数不足",
    "rank_ic": "Rank IC 未超过 0.02",
    "ic_t": "HAC t 未超过 1.645",
    "top_excess": "Top 组合超额未优于股票池均值",
    "top_bottom": "Top-Bottom 分组收益差未为正",
    "ndcg": "NDCG 未高于随机排序基线 5%",
    "mae": "MAE 未优于训练中位数基线",
}

# 模型未生成样本外指标时用的失败原因。它不是某一道门槛没过，而是压根没有可判定的指标，
# 与「门槛没过」必须分开显示，否则页面上会把「数据不足」说成「模型不行」。
NO_METRICS_GATE = "未生成样本外指标"


def finite_or_none(value: Any) -> Any:
    """把 NaN / Inf / numpy 标量转成可安全落库与落盘的原生值。

    ``json.dumps(allow_nan=False)`` 与 psycopg 的 Jsonb 都会因为 NaN 直接抛错，
    而模型指标在退化时经常是 ``np.float64('nan')`` 而不是 None——不是「没有值」，
    而是「算出来是 NaN」，两者都不能当成 0 写进库里。

    放在本模块是因为这里是「训练指标 → 可落库记录」的翻译层；
    drift_monitor 与 model_governance 共用这一份，避免各写一份 NaN 处理。
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {str(key): finite_or_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_or_none(item) for item in value]
    return value


def _finite(value: Any) -> float | None:
    """取一个可为空的有限浮点数，供指标字段使用。"""
    converted = finite_or_none(value)
    return float(converted) if isinstance(converted, (int, float)) and not isinstance(converted, bool) else None


def _training_span(folds: Sequence[dict[str, Any]]) -> tuple[str | None, str | None, int | None]:
    """从各折的明细里取出本次训练实际覆盖的起止交易日与折内最大训练天数。

    取的是所有折的并集而不是最后一折：模型上下文里每一折都用不同窗口重训过，
    只报最后一折会让人以为整个训练期只有那么长。
    """
    starts = [str(fold["trained_from"]) for fold in folds if fold.get("trained_from")]
    ends = [str(fold["trained_through"]) for fold in folds if fold.get("trained_through")]
    counts = [int(fold["training_dates"]) for fold in folds if fold.get("training_dates")]
    return (min(starts) if starts else None, max(ends) if ends else None, max(counts) if counts else None)


def validation_record(context: dict[str, Any], model: dict[str, Any], horizon: int, scan_time: datetime | None = None) -> dict[str, Any]:
    """把模型上下文里的一个模型翻译成一行 model_validation_log。

    ``oos_metrics`` 为空是常态而非异常：模型在样本不足或依赖缺失时会提前返回，
    此时只设了 ``status``。这类记录必须写进去并标成「未生成样本外指标」，
    直接跳过会让通过率分母悄悄变小，看起来比实际更健康。
    """
    metrics = model.get("oos_metrics") or {}
    gates = evaluation_gates(metrics) if metrics else {"passed": False, "gates": {}, "failed_gate": NO_METRICS_GATE}
    folds = metrics.get("folds") or []
    window_start, window_end, fold_days = _training_span(folds)
    # score_kind 在「训练还没跑到评估就退出」时是空的，退回看训练目标：
    # 自动择优模式下目标本身就是 rank 或 reg，比记一个 unknown 有用。
    score_kind = model.get("score_kind")
    if score_kind:
        target_type = "rank" if score_kind == SCORE_KIND_RANK else "reg"
    elif model.get("objective") in ("lambdarank", "regression"):
        target_type = "rank" if model.get("objective") == "lambdarank" else "reg"
    else:
        target_type = "unknown"
    ready = bool(model.get("validated", gates["passed"]))
    mae, baseline_mae = _finite(metrics.get("mae")), _finite(metrics.get("baseline_mae"))
    return {
        "model_version": str(context.get("model_version") or settings.MODEL_VERSION),
        "model_type": MODEL_TYPE_CODES.get(model.get("model"), "unknown"),
        "target_type": target_type,
        "scan_time": scan_time or datetime.now(SHANGHAI),
        "window_start": window_start,
        "window_end": window_end,
        "rank_ic": _finite(metrics.get("rank_ic")),
        "hac_t": _finite(metrics.get("rank_ic_hac_t")),
        "ndcg_ratio": _finite(metrics.get("ndcg_ratio")),
        "top_excess": _finite(metrics.get("top_excess_return")),
        "top_bottom": _finite(metrics.get("top_bottom_spread")),
        "mae": _finite(metrics.get("mae")),
        "training_dates": finite_or_none(model.get("training_dates")),
        "passed_gates": finite_or_none(gates["gates"]),
        "failed_gate": gates["failed_gate"] or (None if ready else "模型训练或最新截面预测不可用"),
        "pass": bool(gates["passed"] and ready),
        "details": finite_or_none({
            "context_id": context.get("context_id"),
            "horizon": horizon,
            "model": model.get("model"),
            "objective": model.get("objective"),
            "score_kind": model.get("score_kind"),
            "status": model.get("status"),
            "samples": model.get("samples"),
            "dates": metrics.get("dates"),
            "samples_evaluated": metrics.get("oos_samples"),
            "universe_excess_return": metrics.get("universe_excess_return"),
            "baseline_mae": metrics.get("baseline_mae"),
            "ic_test_status": metrics.get("ic_test_status"),
            "validation_rule": metrics.get("validation_rule"),
            "fold_training_dates": fold_days,
            "reference_date": context.get("reference_date"),
            "training_window_days": context.get("training_window_days"),
            "objective_mode": context.get("objective_mode"),
            "objective_selection": model.get("objective_selection"),
            "regression_loss": model.get("regression_loss"),
            "label_processing": model.get("label_processing"),
            "mae_relative_change": mae / baseline_mae - 1 if mae is not None and baseline_mae is not None and baseline_mae > 0 else None,
        }),
    }


def validation_records(
    context: dict[str, Any], scan_time: datetime | None = None, trigger: str = "manual"
) -> list[dict[str, Any]]:
    """把一个模型上下文展开成本轮要落库的全部检验记录。

    一次全量扫描产生每个模型、每个周期各一行；候选明细随 details 保存。
    """
    stamp = scan_time or datetime.now(SHANGHAI)
    records = []
    for horizon, models in sorted((context.get("models") or {}).items(), key=lambda item: int(item[0])):
        for model in models:
            record = validation_record(context, model, int(horizon), stamp)
            record["details"]["trigger"] = trigger
            records.append(record)
    return records


def log_validation(repository: QuantRepository, records: Iterable[dict[str, Any]]) -> int:
    """追加写入检验记录，返回写入行数。

    刻意不做 UPSERT：同一个模型版本会被反复重训，每次结果都要留下来才能看趋势，
    覆盖写等于把「模型变差了」这件事本身擦掉。
    """
    rows = list(records)
    if not rows:
        return 0
    from psycopg.types.json import Jsonb

    payload = [
        (
            row["model_version"], row["model_type"], row["target_type"], row["scan_time"],
            row.get("window_start"), row.get("window_end"), row.get("rank_ic"), row.get("hac_t"),
            row.get("ndcg_ratio"), row.get("top_excess"), row.get("top_bottom"), row.get("mae"),
            row.get("training_dates"), Jsonb(row.get("passed_gates") or {}), row.get("failed_gate"),
            bool(row["pass"]), Jsonb(row.get("details") or {}),
        )
        for row in rows
    ]
    with repository.connect() as connection:
        with connection.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO public.quant_model_validation_log("
                "model_version,model_type,target_type,scan_time,window_start,window_end,rank_ic,hac_t,"
                "ndcg_ratio,top_excess,top_bottom,mae,training_dates,passed_gates,failed_gate,pass,details) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                payload,
            )
    return len(payload)


def _rate(passed: int, total: int) -> float | None:
    """通过率；分母为零时返回 None 而不是 0，0% 与「没有数据」是两件事。"""
    return float(passed) / float(total) if total else None


def _group_summary(rows: Sequence[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        bucket = groups.setdefault(str(row.get(key)), {"records": 0, "passed": 0})
        bucket["records"] += 1
        bucket["passed"] += 1 if row.get("pass") else 0
    for bucket in groups.values():
        bucket["pass_rate"] = _rate(bucket["passed"], bucket["records"])
    return dict(sorted(groups.items()))


def candidate_validation_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """从入选日志展开全部目标候选，不把择优后的分组当成完整实验结果。

    输入为数据库检验记录；旧记录缺少候选明细时不猜测未保存的实验结果。
    """
    candidates = []
    for row in rows:
        details = row.get("details") or {}
        attempts = details.get("objective_selection")
        if not attempts:
            if details.get("objective_mode") == "auto":
                continue
            if row.get("target_type") in ("rank", "reg"):
                candidates.append(dict(row))
            continue
        for attempt in attempts:
            objective = attempt.get("objective")
            if objective not in ("regression", "lambdarank"):
                continue
            metrics = attempt.get("oos_metrics") or {}
            gates = evaluation_gates(metrics) if metrics else None
            candidates.append({
                **row, "target_type": "rank" if objective == "lambdarank" else "reg",
                "pass": bool(attempt.get("validated")),
                "passed_gates": gates["gates"] if gates else {},
                "failed_gate": gates["failed_gate"] if gates else None,
                "details": {**details, "objective": objective, "candidate": attempt},
            })
    return candidates


def summarize_pass_rate(
    rows: Sequence[dict[str, Any]], days: int = settings.PASS_RATE_WINDOW_DAYS
) -> dict[str, Any]:
    """从检验记录算出总通过率、分模型、分目标与各门槛失败率。纯函数，不碰数据库。

    门槛失败率的分母是「评估过该门槛的记录数」而不是全部记录：排序目标判 NDCG、
    回归目标判 MAE，两者门槛集合不同，拿全部记录当分母会把对方的通过算成自己的失败。
    """
    total = len(rows)
    passed = sum(1 for row in rows if row.get("pass"))
    evaluated = [row for row in rows if row.get("passed_gates")]
    candidates = candidate_validation_rows(rows)
    candidate_passed = sum(1 for row in candidates if row.get("pass"))
    batch_keys = [
        (str(row.get("model_version")), str((row.get("details") or {}).get("context_id") or row.get("scan_time")))
        for row in rows
    ]
    latest_rows = [row for row, key in zip(rows, batch_keys) if key == batch_keys[0]] if rows else []
    latest_passed = sum(1 for row in latest_rows if row.get("pass"))
    gate_total: dict[str, int] = {}
    gate_failures: dict[str, int] = {}
    for row in rows:
        for gate, ok in (row.get("passed_gates") or {}).items():
            gate_total[gate] = gate_total.get(gate, 0) + 1
            if not ok:
                gate_failures[gate] = gate_failures.get(gate, 0) + 1
    by_gate = {
        gate: {
            "label": GATE_LABELS.get(gate, gate),
            "evaluated": gate_total[gate],
            "failures": gate_failures.get(gate, 0),
            "failure_rate": _rate(gate_failures.get(gate, 0), gate_total[gate]),
        }
        for gate in sorted(gate_total)
    }
    # 记录按时间倒序取回，第一条失败就是最近一次失败的原因。
    recent_failure = next((row for row in rows if not row.get("pass")), None)
    sufficient = total >= int(settings.PASS_RATE_MINIMUM_RECORDS)
    return {
        "days": int(days),
        "records": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": _rate(passed, total),
        "evaluated_records": len(evaluated),
        "unavailable_records": total - len(evaluated),
        "availability_rate": _rate(len(evaluated), total),
        "evaluated_pass_rate": _rate(sum(1 for row in evaluated if row.get("pass")), len(evaluated)),
        "batch_count": len(set(batch_keys)),
        "latest_batch": {"records": len(latest_rows), "passed": latest_passed,
                         "pass_rate": _rate(latest_passed, len(latest_rows))},
        "candidate_records": len(candidates),
        "candidate_passed": candidate_passed,
        "candidate_pass_rate": _rate(candidate_passed, len(candidates)),
        "candidate_metrics_records": sum(1 for row in candidates if row.get("passed_gates")),
        "by_candidate_target_type": _group_summary(candidates, "target_type"),
        "candidate_detail_records": sum(1 for row in rows if (row.get("details") or {}).get("objective_selection")),
        "minimum_records": int(settings.PASS_RATE_MINIMUM_RECORDS),
        "sample_sufficient": sufficient,
        "sample_note": (
            "" if sufficient else
            f"窗口内仅 {total} 条记录（不足 {settings.PASS_RATE_MINIMUM_RECORDS} 条），通过率与门槛失败率样本不足，仅供参考"
        ),
        "by_model_type": _group_summary(rows, "model_type"),
        "by_target_type": _group_summary(rows, "target_type"),
        "by_gate": by_gate,
        "recent_failure": {
            "scan_time": recent_failure["scan_time"].isoformat() if recent_failure and recent_failure.get("scan_time") else None,
            "model_type": recent_failure.get("model_type"),
            "target_type": recent_failure.get("target_type"),
            "failed_gate": recent_failure.get("failed_gate"),
            "reason": GATE_LABELS.get(recent_failure.get("failed_gate"), recent_failure.get("failed_gate")),
        } if recent_failure else None,
    }


def fetch_validation_rows(
    repository: QuantRepository, days: int = settings.PASS_RATE_WINDOW_DAYS,
    model_type: str | None = None, target_type: str | None = None,
    model_version: str | None = None,
) -> list[dict[str, Any]]:
    """读取窗口内的检验记录，按时间倒序。"""
    with repository.connect() as connection:
        return connection.execute(
            "SELECT scan_time, model_version, model_type, target_type, window_start, window_end, "
            "rank_ic, hac_t, ndcg_ratio, top_excess, top_bottom, mae, training_dates, "
            "passed_gates, failed_gate, pass, details "
            "FROM public.quant_model_validation_log "
            "WHERE scan_time >= now() - make_interval(days => %s) "
            "AND (%s::text IS NULL OR model_type = %s::text) "
            "AND (%s::text IS NULL OR target_type = %s::text) "
            "AND (%s::text IS NULL OR model_version = %s::text) "
            "ORDER BY scan_time DESC, id DESC",
            (int(days), model_type, model_type, target_type, target_type, model_version, model_version),
        ).fetchall()


def get_pass_rate(
    repository: QuantRepository, days: int = settings.PASS_RATE_WINDOW_DAYS,
    model_type: str | None = None, target_type: str | None = None,
    model_version: str | None = None,
) -> dict[str, Any]:
    """通过率报表：总通过率、分模型、分目标、分门槛失败率。"""
    return summarize_pass_rate(fetch_validation_rows(repository, days, model_type, target_type, model_version), days=days)


def get_recent_failures(repository: QuantRepository, limit: int = 20, model_version: str | None = None) -> list[dict[str, Any]]:
    """最近的失败记录及原因，供管理员页与告警引用。"""
    with repository.connect() as connection:
        rows = connection.execute(
            "SELECT scan_time, model_version, model_type, target_type, rank_ic, hac_t, ndcg_ratio, "
            "top_excess, top_bottom, mae, failed_gate, passed_gates, details "
            "FROM public.quant_model_validation_log WHERE NOT pass "
            "AND (%s::text IS NULL OR model_version = %s::text) "
            "ORDER BY scan_time DESC, id DESC LIMIT %s",
            (model_version, model_version, min(max(int(limit), 1), 500)),
        ).fetchall()
    return [
        {
            **row,
            "failed_gate_label": GATE_LABELS.get(row.get("failed_gate"), row.get("failed_gate")),
            "model_label": MODEL_TYPE_CODES_REVERSE.get(row.get("model_type"), row.get("model_type")),
        }
        for row in rows
    ]


def fetch_failure_streaks(repository: QuantRepository, limit: int) -> list[dict[str, Any]]:
    """按版本、模型、目标和预测周期读取不同训练批次的最近结果。"""
    with repository.connect() as connection:
        return connection.execute(
            # 同一批次的重复写入只取最后一条，三个预测周期分别统计，不能合成三次失败。
            "WITH batches AS ("
            "  SELECT *, details->>'horizon' AS horizon,"
            "         row_number() OVER ("
            "           PARTITION BY model_version, model_type, target_type, details->>'horizon',"
            "                        COALESCE(NULLIF(details->>'context_id', ''), scan_time::text)"
            "           ORDER BY scan_time DESC, id DESC) AS batch_position"
            "  FROM public.quant_model_validation_log WHERE model_version = %s"
            "), ranked AS ("
            "  SELECT model_version, model_type, target_type, horizon, pass, failed_gate, scan_time,"
            "         row_number() OVER (PARTITION BY model_version, model_type, target_type, horizon "
            "                            ORDER BY scan_time DESC, id DESC) AS position"
            "  FROM batches WHERE batch_position = 1"
            ") SELECT * FROM ranked WHERE position <= %s "
            "ORDER BY model_version, model_type, target_type, horizon, position",
            (settings.MODEL_VERSION, int(limit)),
        ).fetchall()


def evaluate_pass_rate_alerts(rows: Sequence[dict[str, Any]], streaks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """三条通过率告警的判定，纯函数。

    1. 同一版本、模型、目标和预测周期连续 N 个训练批次不通过；
    2. 窗口内总通过率低于下限；
    3. 某道门槛的失败率高于上限。

    第 2、3 条都要求窗口内记录数达标才判：一次全量扫描只产生 6 行记录，
    不加最小样本量的话上线第一天就会因为「月度通过率 33%」刷屏。
    第 1 条自带次数要求，不受该下限约束。
    """
    alerts: list[dict[str, Any]] = []
    required = int(settings.CONSECUTIVE_FAILURE_ALERT)
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in streaks:
        key = (str(row.get("model_version") or settings.MODEL_VERSION), str(row["model_type"]),
               str(row["target_type"]), str(row.get("horizon") or ""))
        grouped.setdefault(key, []).append(row)
    for (version, model_type, target_type, horizon), series in sorted(grouped.items()):
        ordered = sorted(series, key=lambda item: item["position"])
        if len(ordered) < required or any(row.get("pass") for row in ordered[:required]):
            continue
        latest = ordered[0]
        gate = latest.get("failed_gate")
        period = f" · {horizon}日" if horizon else ""
        alerts.append({
            "level": "critical",
            "category": "pass_rate",
            "model_version": version,
            "title": f"{MODEL_TYPE_CODES_REVERSE.get(model_type, model_type)}{period} 连续 {required} 次未通过验证",
            "message": (
                f"模型 {MODEL_TYPE_CODES_REVERSE.get(model_type, model_type)}（目标 {target_type}{period}）"
                f"已连续 {required} 个训练批次未通过验证，最近一次卡在「{GATE_LABELS.get(gate, gate)}」。"
            ),
            "details": {
                "model_type": model_type, "target_type": target_type, "horizon": horizon,
                "consecutive": required,
                "failed_gate": gate, "latest_scan_time": latest["scan_time"].isoformat(),
            },
        })
    summary = summarize_pass_rate(rows)
    if not summary["sample_sufficient"]:
        return alerts
    if summary["pass_rate"] is not None and summary["pass_rate"] < settings.PASS_RATE_MINIMUM:
        alerts.append({
            "level": "critical",
            "category": "pass_rate",
            "title": f"近 {summary['days']} 天通过率低于 {settings.PASS_RATE_MINIMUM:.0%}",
            "message": (
                f"近 {summary['days']} 天共 {summary['records']} 条检验记录，通过 {summary['passed']} 条，"
                f"通过率 {summary['pass_rate']:.1%}，低于设定的 {settings.PASS_RATE_MINIMUM:.0%}。"
            ),
            "details": {"pass_rate": summary["pass_rate"], "records": summary["records"], "days": summary["days"]},
        })
    for gate, item in summary["by_gate"].items():
        if item["evaluated"] >= settings.PASS_RATE_MINIMUM_RECORDS and item["failure_rate"] is not None and item["failure_rate"] > settings.GATE_FAILURE_RATE_ALERT:
            alerts.append({
                "level": "warning",
                "category": "pass_rate",
                "title": f"门槛「{item['label']}」失败率超过 {settings.GATE_FAILURE_RATE_ALERT:.0%}",
                "message": (
                    f"近 {summary['days']} 天里「{item['label']}」被检验 {item['evaluated']} 次，"
                    f"未通过 {item['failures']} 次，失败率 {item['failure_rate']:.1%}，"
                    "该门槛在窗口内频繁未通过，请结合误差幅度与独立训练批次数复核。"
                ),
                "details": {"gate": gate, "failure_rate": item["failure_rate"], "evaluated": item["evaluated"]},
            })
    return alerts


def check_pass_rate_alerts(repository: QuantRepository, days: int = settings.PASS_RATE_WINDOW_DAYS) -> list[dict[str, Any]]:
    """读取窗口数据并返回通过率相关的全部告警。"""
    rows = fetch_validation_rows(repository, days=days, model_version=settings.MODEL_VERSION)
    streaks = fetch_failure_streaks(repository, limit=int(settings.CONSECUTIVE_FAILURE_ALERT))
    streaks = [row for row in streaks if row.get("model_version") == settings.MODEL_VERSION]
    return evaluate_pass_rate_alerts(rows, streaks)
