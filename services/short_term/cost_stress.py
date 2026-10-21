"""按冻结政策运行交易成本与容量压力场景。"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from services.short_term.execution_replay import TERM_FIELDS, run_execution_replay


def run_cost_stress_tests(
    signals: pd.DataFrame,
    daily_bars: pd.DataFrame,
    universe_history: pd.DataFrame,
    security_status_history: pd.DataFrame,
    policy: dict[str, Any] | None,
    horizon: int,
    top_k: int,
) -> dict[str, Any]:
    """执行政策显式列出的费率与容量情景，不擅自设置压力倍数。"""
    stress_policy = policy.get("cost_stress") if isinstance(policy, dict) else None
    scenarios = stress_policy.get("scenarios") if isinstance(stress_policy, dict) else None
    if not isinstance(scenarios, list) or not scenarios:
        return {
            "status": "blocked_missing_scenarios",
            "promotion_evidence": False,
            "blockers": ["冻结验收政策未定义 cost_stress.scenarios。"],
            "scenario_count": 0,
            "scenarios": [],
        }

    results = []
    names: set[str] = set()
    blockers = []
    for index, scenario in enumerate(scenarios, start=1):
        if not isinstance(scenario, dict):
            blockers.append(f"第 {index} 个成本压力场景不是 JSON 对象。")
            continue
        name = str(scenario.get("name") or "").strip()
        overrides = scenario.get("overrides")
        if not name or name in names:
            blockers.append(f"第 {index} 个场景名称缺失或重复。")
            continue
        names.add(name)
        if not isinstance(overrides, dict) or not overrides:
            blockers.append(f"场景 {name} 缺少显式 execution_terms 覆盖项。")
            continue
        unknown = sorted(set(overrides).difference(TERM_FIELDS))
        base_terms = policy.get("execution_terms") if isinstance(policy, dict) else None
        if unknown or not isinstance(base_terms, dict):
            blockers.append(f"场景 {name} 含未知费率字段或缺少基准 execution_terms。")
            continue
        terms = {**base_terms, **overrides}
        replay = run_execution_replay(
            signals,
            daily_bars,
            universe_history,
            security_status_history,
            terms,
            horizon=horizon,
            top_k=top_k,
        )
        trades = replay.pop("trades", [])
        closed_returns = [
            float(trade["net_return"])
            for trade in trades
            if trade.get("net_return") is not None and np.isfinite(float(trade["net_return"]))
        ]
        closed_count = len(closed_returns)
        trade_count = int(replay.get("trade_count", 0))
        metrics = {
            "closed_sample_count": closed_count,
            "mean_closed_net_return": float(np.mean(closed_returns)) if closed_returns else None,
            "worst_closed_net_return": float(min(closed_returns)) if closed_returns else None,
            "unfilled_rate": float(replay.get("unfilled_trade_count", 0) / trade_count) if trade_count else None,
            "unclosed_rate": float(replay.get("unclosed_trade_count", 0) / trade_count) if trade_count else None,
        }
        criteria = scenario.get("criteria")
        case_blockers = list(replay.get("blockers", [])) + list(replay.get("coverage_blockers", []))
        if replay.get("status") != "daily_bar_proxy_not_queue_truth":
            case_blockers.append("该场景的原始回放数据覆盖不完整。")
        if not isinstance(criteria, dict):
            case_blockers.append("场景未定义验收 criteria。")
        else:
            required_criteria = {"min_closed_sample_count", "min_mean_closed_net_return", "max_unfilled_rate", "max_unclosed_rate"}
            missing = sorted(required_criteria.difference(criteria))
            if missing:
                case_blockers.append(f"场景验收条件缺少字段：{'、'.join(missing)}。")
            else:
                try:
                    parsed_criteria = {
                        "min_closed_sample_count": float(criteria["min_closed_sample_count"]),
                        "min_mean_closed_net_return": float(criteria["min_mean_closed_net_return"]),
                        "max_unfilled_rate": float(criteria["max_unfilled_rate"]),
                        "max_unclosed_rate": float(criteria["max_unclosed_rate"]),
                    }
                    valid_criteria = all(math.isfinite(value) for value in parsed_criteria.values())
                    valid_criteria = valid_criteria and parsed_criteria["min_closed_sample_count"] >= 1
                    valid_criteria = valid_criteria and parsed_criteria["min_closed_sample_count"].is_integer()
                    valid_criteria = valid_criteria and 0 <= parsed_criteria["max_unfilled_rate"] <= 1
                    valid_criteria = valid_criteria and 0 <= parsed_criteria["max_unclosed_rate"] <= 1
                except (TypeError, ValueError, OverflowError):
                    valid_criteria = False
                    parsed_criteria = {}
                if not valid_criteria:
                    case_blockers.append("场景验收阈值必须是有限值，样本数为正整数，比例在 0 至 1 之间。")
                else:
                    if closed_count < int(parsed_criteria["min_closed_sample_count"]):
                        case_blockers.append("已平仓样本数未达场景要求。")
                    if (
                        metrics["mean_closed_net_return"] is None
                        or metrics["mean_closed_net_return"] < parsed_criteria["min_mean_closed_net_return"]
                    ):
                        case_blockers.append("平均已平仓净收益未达场景要求。")
                    if metrics["unfilled_rate"] is None or metrics["unfilled_rate"] > parsed_criteria["max_unfilled_rate"]:
                        case_blockers.append("未成交比例超过场景要求。")
                    if metrics["unclosed_rate"] is None or metrics["unclosed_rate"] > parsed_criteria["max_unclosed_rate"]:
                        case_blockers.append("未平仓比例超过场景要求。")
        results.append(
            {
                "name": name,
                "status": "passed_diagnostic" if not case_blockers else "blocked_or_failed",
                "metrics": metrics,
                "blockers": list(dict.fromkeys(case_blockers)),
                "execution_basis": replay.get("execution_basis"),
                "promotion_evidence": bool(replay.get("promotion_evidence")),
            }
        )

    all_passed = bool(results) and len(results) == len(scenarios) and all(
        item["status"] == "passed_diagnostic" for item in results
    ) and not blockers
    if not all_passed:
        blockers.append("至少一个压力场景缺少完整数据、未达冻结标准或无法计算。")
    return {
        "status": "passed_diagnostic" if all_passed else "blocked_or_failed",
        "promotion_evidence": all_passed and all(item["promotion_evidence"] for item in results),
        "scenario_count": len(results),
        "scenarios": results,
        "blockers": list(dict.fromkeys(blockers)),
        "basis": "policy_defined_execution_terms_scenarios",
    }
