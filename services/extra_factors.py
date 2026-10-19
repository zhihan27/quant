"""把新增因子族并入既有排序特征矩阵的统一下沉层。

筹码分布、日内高频与 PIT 财务三类因子的覆盖率差异很大：筹码是日频、基本全覆盖；财务是季度
低频但按公告日对齐后每个交易日都有值；日内因子受免费分钟线深度限制，目前只能覆盖最近若干个
交易日。因此并入模型时按**因子族**做覆盖率门控——整族因子齐全的行占比达到
``settings.EXTRA_FACTOR_MINIMUM_COVERAGE`` 才纳入训练，未达标的族整体不参与，而不是把大量缺失
值塞进特征矩阵让模型去猜。

门控口径与既有市值行业中性化一致（``panel[columns].notna().all(axis=1).mean()``），纳入后的因子
同样走截面 Z 分数标准化，与基础因子处于同一量纲，并沿用原有 Rank IC、HAC t 检验等样本外验收关卡。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from config import settings
from services.chip_factors import CHIP_FACTOR_COLUMNS, attach_chip_factors
from services.financial_factors import FINANCIAL_FACTOR_COLUMNS, attach_financial_factors, compute_financial_factors
from services.fundamentals import load_fundamental_facts
from services.intraday_factors import INTRADAY_FACTOR_COLUMNS, attach_intraday_factors, load_intraday_factors
from services.panel_store import PanelStore

LOGGER = logging.getLogger("quant.extra_factors")

# 财务健康一票否决所依据的列，由 PIT 财务因子按公告日对齐后挂到面板上。
HEALTH_VETO_COLUMN = "fundamental_health_veto"

# 因子族 → 该族用于门控的完整列集合；整族齐全才算覆盖。
EXTRA_FACTOR_FAMILIES = {
    "chip": tuple(CHIP_FACTOR_COLUMNS),
    "intraday": tuple(INTRADAY_FACTOR_COLUMNS),
    "financial": tuple(FINANCIAL_FACTOR_COLUMNS),
}


def build_extra_factors(
    store: PanelStore,
    codes: list[str],
    histories: dict[str, Any] | None = None,
    families: Any = None,
    progress: Any = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """按股票池装配三类新增因子，返回可直接交给训练入口的字典与采集报告。

    全部取自本地已落盘的面板（筹码用本次已抓到的日线，日内与财务读本地分区），打分链路不额外联网。
    未建库的族直接缺席，交由 ``attach_extra_factors`` 的覆盖率门控决定是否参与模型。
    """
    wanted = set(families or EXTRA_FACTOR_FAMILIES)
    collected: dict[str, Any] = {}
    report: dict[str, Any] = {}
    if histories and "chip" in wanted:
        collected["chip"] = histories
        report["chip"] = {"codes": len(histories), "status": "筹码分布由本次日线滚动模拟"}
    if "intraday" in wanted:
        factors, intraday_report = load_intraday_factors(store, codes, progress=progress)
        report["intraday"] = intraday_report
        if not factors.empty:
            collected["intraday"] = factors
    if "financial" in wanted:
        facts = load_fundamental_facts(store, codes)
        if facts is None or facts.empty:
            report["financial"] = {"status": "本地尚无 PIT 财务面板，也没有可验证的点时财务面板", "rows": 0}
        elif (
            "value_available_at" not in facts
            or pd.to_datetime(facts["value_available_at"], errors="coerce").isna().any()
        ):
            report["financial"] = {
                "status": "报表数值缺少修订可得时间，已禁用财务因子与健康否决",
                "rows": int(len(facts)), "point_in_time_verified": False,
            }
        else:
            factors = compute_financial_factors(facts)
            report["financial"] = {
                "rows": int(len(factors)),
                "codes": int(factors["code"].nunique()) if not factors.empty else 0,
                "status": "带数值修订可得时间的点时财务因子已计算",
                "point_in_time_verified": True,
            }
            if not factors.empty:
                collected["financial"] = factors
    report["status"] = (
        "已装配因子族：" + "、".join(sorted(collected)) if collected else "本地无可用新增因子数据"
    )
    return collected, report


def attach_extra_factors(
    panel: pd.DataFrame,
    extra_factors: dict[str, Any] | None = None,
    store: PanelStore | None = None,
    progress: Any = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """把三类新增因子并入日截面面板，返回并入后的面板与逐族报告。

    ``extra_factors`` 的取值沿用各模块既有入口：``chip`` 传逐股日线字典（筹码需滚动模拟），
    ``intraday`` 与 ``financial`` 传各自算好的因子长表。
    """
    if panel is None or panel.empty:
        return panel, {"status": "面板为空，未并入新增因子", "families": {}}
    extra_factors = extra_factors or {}
    result = panel
    families: dict[str, Any] = {}
    if extra_factors.get("chip") is not None:
        result, families["chip"] = attach_chip_factors(result, extra_factors["chip"], progress=progress)
    if extra_factors.get("intraday") is not None:
        result, families["intraday"] = attach_intraday_factors(result, extra_factors["intraday"], progress=progress)
    if extra_factors.get("financial") is not None:
        result, families["financial"] = attach_financial_factors(
            result, extra_factors["financial"], store=store, progress=progress
        )
    if not families:
        return result, {"status": "未提供新增因子", "families": {}}
    accepted, decisions = accepted_extra_columns(result)
    report = {
        "families": families,
        "decisions": decisions,
        "accepted_columns": accepted,
        "minimum_coverage": float(settings.EXTRA_FACTOR_MINIMUM_COVERAGE),
        "status": (
            f"新增因子并入 {len(accepted)} 列：" + "、".join(sorted({key for key, item in decisions.items() if item['accepted']}))
            if accepted
            else "新增因子覆盖率均未达标，模型仅使用基础因子"
        ),
    }
    return result, report


def apply_health_veto(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """把财务健康标记为真的行排除出可交易截面，返回裁剪后的面板与否决报告。

    只否决字段**显式为真**的行：拿不到 PIT 财务数据的行是「未知」而不是「健康」，不能因为取不到
    数据就被误杀。调用方必须在任何截面标准化之前执行，被否决的标的不得参与自己所在截面的均值与
    标准差——否则「打分校准的股票池」与「实际可交易的股票池」就会不一致。这一条与股票池的
    退市／ST 过滤同属硬性风险规则。
    """
    total = int(len(panel)) if panel is not None else 0
    if panel is None or panel.empty or HEALTH_VETO_COLUMN not in panel:
        return panel, {
            "status": "面板上没有财务健康标记，未执行否决", "rows": total, "vetoed": 0, "veto_rate": 0.0,
        }
    vetoed = panel[HEALTH_VETO_COLUMN].fillna(False).astype(bool)
    count = int(vetoed.sum())
    if not count:
        return panel, {
            "status": "本期没有触发财务健康否决的标的", "rows": total, "vetoed": 0, "veto_rate": 0.0,
        }
    kept = panel.loc[~vetoed].reset_index(drop=True)
    if kept.empty:
        kept.attrs = dict(panel.attrs)
        codes = int(panel.loc[vetoed, "code"].astype(str).nunique()) if "code" in panel else 0
        dates = int(panel.loc[vetoed, "date"].nunique()) if "date" in panel else 0
        return kept, {
            "status": f"财务健康否决已排除全部 {count} 行候选（{codes} 只股票、{dates} 个交易日），无可交易面板",
            "rows": total, "vetoed": count, "veto_rate": 1.0,
            "codes": codes, "dates": dates, "applied": True,
        }
    kept.attrs = dict(panel.attrs)
    codes = int(panel.loc[vetoed, "code"].astype(str).nunique()) if "code" in panel else 0
    dates = int(panel.loc[vetoed, "date"].nunique()) if "date" in panel else 0
    return kept, {
        "rows": total,
        "vetoed": count,
        "veto_rate": count / total,
        "codes": codes,
        "dates": dates,
        "applied": True,
        "status": f"财务健康否决已排除 {count} 行（{codes} 只股票、{dates} 个交易日）",
    }


def accepted_extra_columns(
    panel: pd.DataFrame,
    families: dict[str, tuple[str, ...]] | None = None,
    minimum_coverage: float | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """按覆盖率门控挑出可纳入模型的特征列，并给出逐族判定依据。"""
    minimum = float(settings.EXTRA_FACTOR_MINIMUM_COVERAGE if minimum_coverage is None else minimum_coverage)
    accepted: list[str] = []
    decisions: dict[str, Any] = {}
    total = len(panel) if panel is not None else 0
    for name, columns in (families or EXTRA_FACTOR_FAMILIES).items():
        present = [column for column in columns if column in panel]
        if not present or total == 0:
            decisions[name] = {"coverage": 0.0, "accepted": False, "columns": []}
            continue
        coverage = float(panel[present].notna().all(axis=1).mean())
        # 只并入了一部分列说明该族的装配不完整，宁可整族不用，也不让模型悄悄只用半族因子。
        complete_family = len(present) == len(columns)
        passed = complete_family and coverage >= minimum
        decisions[name] = {"coverage": coverage, "accepted": passed, "columns": present if passed else []}
        if passed:
            accepted.extend(present)
    return accepted, decisions


def summarize_extra_factors(panel: pd.DataFrame) -> dict[str, Any]:
    """当前面板上各因子族的实际覆盖率，用于测试记录与页面展示。"""
    total = len(panel) if panel is not None else 0
    summary: dict[str, Any] = {"rows": int(total)}
    for name, columns in EXTRA_FACTOR_FAMILIES.items():
        present = [column for column in columns if column in panel]
        if not present or total == 0:
            summary[name] = {"coverage": 0.0, "rows": 0}
            continue
        complete = panel[present].notna().all(axis=1)
        summary[name] = {"coverage": float(complete.mean()), "rows": int(complete.sum())}
    return summary


def factor_correlation(
    panel: pd.DataFrame, columns: list[str], reference: str
) -> dict[str, float]:
    """新增因子与参考因子的截面秩相关，用于判断是否存在重复计价。"""
    if panel is None or panel.empty or reference not in panel:
        return {}
    scores: dict[str, float] = {}
    for column in columns:
        if column not in panel:
            continue
        values = pd.to_numeric(panel[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        base = pd.to_numeric(panel[reference], errors="coerce").replace([np.inf, -np.inf], np.nan)
        paired = pd.concat([values, base], axis=1).dropna()
        if len(paired) < 3:
            continue
        correlation = paired.iloc[:, 0].corr(paired.iloc[:, 1], method="spearman")
        if pd.notna(correlation):
            scores[column] = float(correlation)
    return scores
