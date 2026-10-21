"""短线模型晋级门禁；缺少冻结政策或证据时默认阻断。"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from services.short_term.execution_replay import TERM_FIELDS


REQUIRED_EVIDENCE = (
    "point_in_time_universe",
    "point_in_time_security_status",
    "execution_replay",
    "cost_stress",
    "forward_observation",
)
REQUIRED_POLICY_FIELDS = (
    "version",
    "frozen_at",
    "approved_by",
    "min_folds",
    "min_scored_days",
    "min_mean_net_return",
    "min_excess_vs_momentum",
    "min_positive_fold_ratio",
    "max_drawdown",
    "execution_terms",
)


def load_acceptance_policy(path: Path) -> tuple[dict[str, Any] | None, str | None, str | None]:
    """读取冻结政策、哈希和可审计的缺失或解析原因。"""
    if not path.is_file():
        return None, None, f"验收政策文件不存在：{path}"
    try:
        raw = path.read_bytes()
        policy = json.loads(raw.decode("utf-8"))
    except OSError as exc:
        return None, None, f"验收政策文件读取失败：{type(exc).__name__}"
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, None, f"验收政策不是有效 UTF-8 JSON：{type(exc).__name__}"
    if not isinstance(policy, dict):
        return None, None, "验收政策顶层必须是 JSON 对象"
    missing = [field for field in REQUIRED_POLICY_FIELDS if policy.get(field) is None]
    if missing:
        return None, None, f"验收政策缺少字段：{', '.join(missing)}"
    if not str(policy.get("approved_by", "")).strip() or not str(policy.get("version", "")).strip():
        return None, None, "验收政策的 version 与 approved_by 不能为空"
    if not isinstance(policy.get("execution_terms"), dict):
        return None, None, "验收政策 execution_terms 必须是 JSON 对象"
    missing_terms = [field for field in TERM_FIELDS if policy["execution_terms"].get(field) is None]
    if missing_terms:
        return None, None, f"验收政策 execution_terms 缺少字段：{', '.join(missing_terms)}"
    try:
        frozen_at = datetime.fromisoformat(str(policy["frozen_at"]).replace("Z", "+00:00"))
    except ValueError:
        return None, None, "验收政策 frozen_at 必须是带时区的 ISO-8601 时间"
    if frozen_at.tzinfo is None:
        return None, None, "验收政策 frozen_at 必须包含时区"
    numeric_fields = (
        "min_folds", "min_scored_days", "min_mean_net_return", "min_excess_vs_momentum",
        "min_positive_fold_ratio", "max_drawdown",
    )
    try:
        numeric = {field: float(policy[field]) for field in numeric_fields}
    except (TypeError, ValueError, OverflowError):
        return None, None, "验收政策的数值门槛格式无效"
    if any(not math.isfinite(value) for value in numeric.values()):
        return None, None, "验收政策的数值门槛必须为有限值"
    if not numeric["min_folds"].is_integer() or not numeric["min_scored_days"].is_integer():
        return None, None, "验收政策的 min_folds 与 min_scored_days 必须为整数"
    if numeric["min_folds"] < 1 or numeric["min_scored_days"] < 1 or not 0 <= numeric["min_positive_fold_ratio"] <= 1 or numeric["max_drawdown"] < 0:
        return None, None, "验收政策的样本量、正收益比例或回撤门槛超出有效范围"
    return policy, hashlib.sha256(raw).hexdigest(), None


def evaluate_promotion_gate(
    validation: dict[str, Any], evidence: dict[str, bool], policy: dict[str, Any] | None
) -> dict[str, Any]:
    """按预先冻结的政策检查晋级条件，不自动改变模型状态。"""
    blockers: list[str] = []
    if not isinstance(policy, dict):
        blockers.append("尚未登记有效且冻结的验收政策")
    else:
        missing_policy = [field for field in REQUIRED_POLICY_FIELDS if policy.get(field) is None]
        if missing_policy:
            blockers.append(f"验收政策缺少字段：{', '.join(missing_policy)}")
        else:
            if int(validation.get("fold_count", 0)) < int(policy["min_folds"]):
                blockers.append("样本外时间折数量未达政策要求")
            if int(validation.get("scored_days", 0)) < int(policy["min_scored_days"]):
                blockers.append("样本外有效交易日数量未达政策要求")
            mean_return = validation.get("top_k_mean_net_return")
            if mean_return is None or float(mean_return) < float(policy["min_mean_net_return"]):
                blockers.append("Top-K 扣费收益未达政策要求")
            model_return = validation.get("top_k_mean_net_return")
            baseline_return = validation.get("momentum_top_k_mean_net_return")
            if model_return is None or baseline_return is None or float(model_return) - float(baseline_return) < float(policy["min_excess_vs_momentum"]):
                blockers.append("相对动量基线的增量未达政策要求")
            positive_ratio = validation.get("positive_fold_ratio")
            if positive_ratio is None or float(positive_ratio) < float(policy["min_positive_fold_ratio"]):
                blockers.append("正收益时间折比例未达政策要求")
            drawdown = validation.get("max_drawdown")
            if drawdown is None or abs(float(drawdown)) > float(policy["max_drawdown"]):
                blockers.append("最大回撤未达政策要求")
    for evidence_name in REQUIRED_EVIDENCE:
        if not bool(evidence.get(evidence_name, False)):
            blockers.append(f"缺少证据：{evidence_name}")
    return {"status": "blocked" if blockers else "eligible_for_review", "blockers": blockers}
