"""PIT 点时财报多因子：盈利质量、成长动能与财务健康。

所有因子只在**已正式公告**的数据上计算，且把「生效日」定义为计算中实际用到的那几个报告期里
最晚的公告日。这样处理的原因是一个容易被忽略的时序陷阱：上一年年报与本年一季报的公告日均落在
四月，一季报常常**早于**年报披露。若一季报的 TTM 指标直接引用年报数据，就等于用到了当时尚未
公开的数字。以生效日为锚做 ASOF JOIN，可以让这类指标自动延后到真正可得的时点才生效。

因子口径：

- ``fundamental_roe_ttm_deducted``：TTM 扣非归母净利润 / 平均归母净资产，衡量主营真实的赚钱能力。
- ``fundamental_ocf_to_profit``：TTM 经营现金流净额 / TTM 归母净利润。大于 1 说明利润有现金支撑；
  亏损时该比值失去解释力，直接留空而不强行给数。
- ``fundamental_revenue_yoy``：单季营业收入同比。
- ``fundamental_deducted_profit_yoy``：单季扣非归母净利润同比。
- ``fundamental_deducted_profit_yoy_accel``：同比增速的环比变化，即「加速」。
  单季净利润本身季节性强，直接做环比会把季节性当成加速，因此改用同比增速的变化来度量。
- ``fundamental_altman_z``：Altman Z''（1993）财务困境预警。

Altman 口径说明：原始 Z（1968）的 X4 需要**权益市值**，而本面板刻意只依赖财报数值、不掺入价格，
因此采用面向非制造业与新兴市场的 Z'' 变体——它本身就以**账面权益**计量 X4，系数与口径自洽，
不像「拿原始 Z 的系数配账面权益」那样张冠李戴。A 股股票池同时包含制造与非制造企业，Z'' 的适用
范围也更贴合。判定阈值见 ``settings.ALTMAN_Z_DISTRESS_THRESHOLD`` 与 ``ALTMAN_Z_SAFE_THRESHOLD``。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from config import settings
from services.fundamentals import align_asof_pit
from services.panel_store import PanelStore

LOGGER = logging.getLogger("quant.financial_factors")

FINANCIAL_FACTOR_COLUMNS = [
    "fundamental_roe_ttm_deducted",
    "fundamental_ocf_to_profit",
    "fundamental_revenue_yoy",
    "fundamental_deducted_profit_yoy",
    "fundamental_deducted_profit_yoy_accel",
    "fundamental_altman_z",
]
FLAG_COLUMNS = ["fundamental_health_flag", "fundamental_health_veto"]
ATTACH_COLUMNS = [*FINANCIAL_FACTOR_COLUMNS, *FLAG_COLUMNS, "fundamental_period", "fundamental_ann_date"]

# 利润表与现金流量表按年初至今累计口径披露，资产负债表科目是时点数，两者还原方式不同。
CUMULATIVE_COLUMNS = (
    "TOTAL_OPERATE_INCOME",
    "OPERATE_INCOME",
    "TOTAL_PROFIT",
    "OPERATE_PROFIT",
    "FINANCE_EXPENSE",
    "PARENT_NETPROFIT",
    "DEDUCT_PARENT_NETPROFIT",
    "NETCASH_OPERATE",
)
_TTM_QUARTERS = 4
# 一次 TTM 计算最多回看 5 个报告期（4 个单季 + 1 个上期累计），生效日按这一窗口取最晚公告日。
_LOOKBACK_PERIODS = _TTM_QUARTERS + 1
_QUARTER_ENDS = ((3, 31), (6, 30), (9, 30), (12, 31))
# Altman Z''（1993）：X1 营运资金、X2 留存收益、X3 息税前利润、X4 账面权益，均除以总资产或总负债。
ALTMAN_COEFFICIENTS = {"x1": 6.56, "x2": 3.26, "x3": 6.72, "x4": 1.05}
_HEALTH_LABELS = {"safe": "健康", "grey": "灰色", "distress": "困境"}


def quarter_end_grid(first: pd.Timestamp, last: pd.Timestamp) -> list[pd.Timestamp]:
    """生成覆盖 ``[first, last]`` 的完整季度末网格，用于对齐缺失报告期。"""
    if pd.isna(first) or pd.isna(last) or last < first:
        return []
    grid: list[pd.Timestamp] = []
    for year in range(first.year, last.year + 1):
        for month, day in _QUARTER_ENDS:
            stamp = pd.Timestamp(year=year, month=month, day=day)
            if first <= stamp <= last:
                grid.append(stamp)
    return grid


def previous_quarter_end(stamp: pd.Timestamp) -> pd.Timestamp:
    """返回某个季度末的上一个季度末。"""
    if stamp.month == 3:
        return pd.Timestamp(year=stamp.year - 1, month=12, day=31)
    if stamp.month == 6:
        return pd.Timestamp(year=stamp.year, month=3, day=31)
    if stamp.month == 9:
        return pd.Timestamp(year=stamp.year, month=6, day=30)
    return pd.Timestamp(year=stamp.year, month=9, day=30)


def to_single_quarter(series: pd.Series) -> pd.Series:
    """把年初至今的累计值还原为单季值：一季度即累计值，其余为上期累计值之差。

    只有当前一行的报告期**恰好是上一个季度末**时才做差。否则（中间缺报告期）直接留空——用隔了
    两个季度的累计值相减会把两个季度并成一个「单季」，是无声的口径错误，宁可缺数不可错数。
    """
    values = pd.to_numeric(series, errors="coerce").astype(float)
    index = pd.DatetimeIndex(values.index)
    result = values.copy()
    for position, stamp in enumerate(index):
        if stamp.month == 3:
            continue
        adjacent = position > 0 and index[position - 1] == previous_quarter_end(stamp)
        result.iloc[position] = values.iloc[position] - values.iloc[position - 1] if adjacent else np.nan
    return result


def _growth(series: pd.Series, lag: int) -> pd.Series:
    """同比增速；基期取绝对值做分母，使基期为负时增速的方向仍然正确。"""
    base = series.shift(lag)
    usable = base.abs() > 1e-6
    return ((series - base) / base.abs()).where(usable & series.notna() & base.notna())


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """仅在分母为正时计算比值，其余留空，避免负分母制造出方向相反的高分。"""
    return (numerator / denominator.where(denominator > 1e-6)).replace([np.inf, -np.inf], np.nan)


def _column_or_na(frame: pd.DataFrame, column: str) -> pd.Series:
    """取列；整列不存在时按空值处理。

    金融行业的资产负债表没有流动与非流动的划分，`TOTAL_CURRENT_ASSETS` / `TOTAL_CURRENT_LIAB`
    在银行／保险／证券的报表里根本不存在，合并后既可能是空列也可能整列缺位。缺科目是正常形状，
    对应分量留空即可，不该让整只股票在财务因子上落空。
    """
    if column in frame:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index)


def altman_z_components(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """计算 Z'' 的四个分量；分母非正时对应分量为空。"""
    assets = _column_or_na(frame, "TOTAL_ASSETS")
    liabilities = _column_or_na(frame, "TOTAL_LIABILITIES")
    working_capital = _column_or_na(frame, "TOTAL_CURRENT_ASSETS") - _column_or_na(frame, "TOTAL_CURRENT_LIAB")
    return {
        "x1": _safe_ratio(working_capital, assets),
        "x2": _safe_ratio(_column_or_na(frame, "UNASSIGN_RPOFIT"), assets),
        "x3": _safe_ratio(frame["_ttm_ebit"], assets),
        "x4": _safe_ratio(_column_or_na(frame, "TOTAL_PARENT_EQUITY"), liabilities),
    }


def altman_z(frame: pd.DataFrame) -> pd.Series:
    """Altman Z''：按系数加权四个分量；任一分量缺失时整体留空。"""
    components = altman_z_components(frame)
    total = sum(ALTMAN_COEFFICIENTS[key] * value for key, value in components.items())
    complete = pd.concat(components.values(), axis=1).notna().all(axis=1)
    return total.where(complete)


def health_flag(z: pd.Series, ocf_to_profit: pd.Series) -> tuple[pd.Series, pd.Series]:
    """按 Altman Z'' 分档，并把经营现金流为负的公司直接判为困境。

    利润为正但经营现金流为负，说明利润没有现金支撑，这类公司在 Altman 分档之上再加一道否决。
    """
    distress = float(settings.ALTMAN_Z_DISTRESS_THRESHOLD)
    safe = float(settings.ALTMAN_Z_SAFE_THRESHOLD)
    flag = pd.Series(pd.NA, index=z.index, dtype="object")
    flag = flag.mask(z.notna() & z.lt(distress), "distress")
    flag = flag.mask(z.notna() & z.ge(distress) & z.lt(safe), "grey")
    flag = flag.mask(z.notna() & z.ge(safe), "safe")
    cashflow_flag = ocf_to_profit.notna() & ocf_to_profit.lt(float(settings.MINIMUM_CASHFLOW_TO_PROFIT))
    flag = flag.mask(cashflow_flag & flag.ne("distress"), "distress")
    veto = flag.eq("distress")
    return flag, veto


def compute_financial_factors(facts: pd.DataFrame) -> pd.DataFrame:
    """逐股计算 PIT 财务因子，输出带生效日 ``available_from`` 的长表。

    生效日取该行计算中实际回看窗口内最晚的公告日，确保 ASOF 对齐后每个交易日只用已公开数据。
    """
    required = {"code", "report_period", "ann_date"}
    missing = required - set(facts.columns)
    if missing:
        raise ValueError(f"财务因子缺少字段：{', '.join(sorted(missing))}")
    if facts.empty:
        return pd.DataFrame(columns=["code", "report_period", "available_from", *FINANCIAL_FACTOR_COLUMNS, *FLAG_COLUMNS])
    frame = facts.copy()
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["report_period"] = pd.to_datetime(frame["report_period"])
    frame["ann_date"] = pd.to_datetime(frame["ann_date"])
    frame = frame.sort_values(["code", "report_period"])
    frames = [_factors_for_code(code, group) for code, group in frame.groupby("code", sort=True)]
    combined = pd.concat(frames, ignore_index=True)
    return combined.loc[:, ["code", "report_period", "available_from", *FINANCIAL_FACTOR_COLUMNS, *FLAG_COLUMNS]]


def empty_factor_frame() -> pd.DataFrame:
    """历史不足时的空结果；列结构必须与正常输出一致，避免下游按列名取值时报缺失。"""
    frame = pd.DataFrame(index=pd.DatetimeIndex([], name="report_period"))
    for column in FINANCIAL_FACTOR_COLUMNS:
        frame[column] = pd.Series(dtype="float64")
    frame["fundamental_health_flag"] = pd.Series(dtype="object")
    frame["fundamental_health_veto"] = pd.Series(dtype="bool")
    frame["available_from"] = pd.Series(dtype="datetime64[ns]")
    return frame


def _factors_for_code(code: str, group: pd.DataFrame) -> pd.DataFrame:
    """单只股票在其完整季度网格上计算因子；缺失报告期保持为空。"""
    group = group.drop_duplicates("report_period", keep="last").sort_values("report_period")
    grid = quarter_end_grid(group["report_period"].min(), group["report_period"].max())
    if len(grid) < int(settings.FUNDAMENTAL_MINIMUM_QUARTERS):
        return _assemble(code, empty_factor_frame())
    indexed = group.set_index("report_period").reindex(grid)
    indexed.index.name = "report_period"

    single = {column: to_single_quarter(indexed[column]) for column in CUMULATIVE_COLUMNS if column in indexed}
    ttm = {column: series.rolling(_TTM_QUARTERS, min_periods=_TTM_QUARTERS).sum() for column, series in single.items()}
    # EBIT：利润总额加回财务费用，取 TTM 口径与资产表的时点数配平。
    indexed["_ttm_ebit"] = ttm.get("TOTAL_PROFIT", pd.Series(np.nan, index=indexed.index)) + ttm.get(
        "FINANCE_EXPENSE", pd.Series(np.nan, index=indexed.index)
    )

    equity = _column_or_na(indexed, "TOTAL_PARENT_EQUITY")
    average_equity = (equity + equity.shift(_TTM_QUARTERS)) / 2.0
    roe = _safe_ratio(ttm.get("DEDUCT_PARENT_NETPROFIT", pd.Series(np.nan, index=indexed.index)), average_equity)
    ocf_to_profit = _safe_ratio(
        ttm.get("NETCASH_OPERATE", pd.Series(np.nan, index=indexed.index)),
        ttm.get("PARENT_NETPROFIT", pd.Series(np.nan, index=indexed.index)),
    )
    revenue_yoy = _growth(single.get("TOTAL_OPERATE_INCOME", pd.Series(np.nan, index=indexed.index)), _TTM_QUARTERS)
    deducted_yoy = _growth(single.get("DEDUCT_PARENT_NETPROFIT", pd.Series(np.nan, index=indexed.index)), _TTM_QUARTERS)
    acceleration = deducted_yoy - deducted_yoy.shift(1)
    z = altman_z(indexed)
    flag, veto = health_flag(z, ocf_to_profit)

    result = pd.DataFrame(
        {
            "fundamental_roe_ttm_deducted": roe,
            "fundamental_ocf_to_profit": ocf_to_profit,
            "fundamental_revenue_yoy": revenue_yoy,
            "fundamental_deducted_profit_yoy": deducted_yoy,
            "fundamental_deducted_profit_yoy_accel": acceleration,
            "fundamental_altman_z": z,
            "fundamental_health_flag": flag,
            "fundamental_health_veto": veto,
            "available_from": _available_from(indexed["ann_date"]),
        },
        index=indexed.index,
    )
    return _assemble(code, result)


def _available_from(ann_dates: pd.Series) -> pd.Series:
    """回看窗口内最晚的公告日；窗口内有报告期缺公告日时该行留空，不放行未公开数据。"""
    values = pd.to_datetime(ann_dates).to_numpy()
    result: list[Any] = []
    for position in range(len(values)):
        window = values[max(0, position - _LOOKBACK_PERIODS + 1) : position + 1]
        window = window[~pd.isna(window)]
        result.append(window.max() if window.size else pd.NaT)
    return pd.Series(result, index=ann_dates.index, dtype="datetime64[ns]")


def _assemble(code: str, result: pd.DataFrame) -> pd.DataFrame:
    """补上代码列并把报告期还原为普通列。"""
    frame = result.reset_index()
    if "report_period" not in frame:
        frame["report_period"] = pd.NaT
    frame.insert(0, "code", code)
    return frame


def attach_financial_factors(
    panel: pd.DataFrame,
    factors: pd.DataFrame | None,
    store: PanelStore | None = None,
    progress: Any = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """把 PIT 财务因子按生效日严格对齐到日截面，缺失留空不用池均值填充。"""
    if panel is None or panel.empty or "code" not in panel:
        return panel, {"status": "面板为空，未并入财务因子", "coverage": 0.0, "columns": FINANCIAL_FACTOR_COLUMNS}
    result = panel.copy()
    for column in ATTACH_COLUMNS:
        if column not in result:
            result[column] = pd.NA
    if factors is None or factors.empty:
        return result, {"status": "没有可并入的财务因子", "coverage": 0.0, "columns": FINANCIAL_FACTOR_COLUMNS}
    usable = factors.dropna(subset=["available_from"]).copy()
    usable = usable.loc[usable[FINANCIAL_FACTOR_COLUMNS].notna().any(axis=1)]
    if usable.empty:
        return result, {"status": "财务因子全部缺少生效日，未并入", "coverage": 0.0, "columns": FINANCIAL_FACTOR_COLUMNS}
    if progress:
        progress(f"正在按公告日对齐 {len(usable)} 条财务因子记录")
    aligned = align_asof_pit(store, result, usable, [*FINANCIAL_FACTOR_COLUMNS, *FLAG_COLUMNS])
    for column in [*FINANCIAL_FACTOR_COLUMNS, *FLAG_COLUMNS]:
        result[column] = aligned[column].to_numpy()
    # 生效期单独对齐一份，便于追溯每个交易日到底用了哪一期财报、何时起可用。
    trace = align_asof_pit(store, result, usable, ["report_period", "available_from"])
    result["fundamental_period"] = trace["report_period"].to_numpy()
    result["fundamental_ann_date"] = trace["available_from"].to_numpy()
    coverage = float(result[FINANCIAL_FACTOR_COLUMNS].notna().any(axis=1).mean()) if len(result) else 0.0
    veto_rate = float(result["fundamental_health_veto"].fillna(False).astype(bool).mean()) if len(result) else 0.0
    return result, {
        "rows_with_factors": int(result[FINANCIAL_FACTOR_COLUMNS].notna().any(axis=1).sum()),
        "coverage": coverage,
        "veto_rate": veto_rate,
        "columns": FINANCIAL_FACTOR_COLUMNS,
        "status": f"财务因子面板覆盖率 {coverage:.0%}，其中困境否决占比 {veto_rate:.0%}",
        "rule": "生效日取计算回看窗口内最晚公告日，交易日严格晚于生效日才可使用。",
    }


def financial_factor_summary(factors: pd.DataFrame | None) -> dict[str, Any]:
    """财务因子的分布概览，用于测试记录与页面展示。"""
    if factors is None or factors.empty:
        return {"rows": 0, "status": "未计算财务因子"}
    summary: dict[str, Any] = {"rows": int(len(factors)), "codes": int(factors["code"].nunique())}
    for column in FINANCIAL_FACTOR_COLUMNS:
        values = pd.to_numeric(factors[column], errors="coerce").dropna()
        if values.empty:
            continue
        summary[column] = {
            "median": float(values.median()),
            "q05": float(values.quantile(0.05)),
            "q95": float(values.quantile(0.95)),
        }
    flags = factors["fundamental_health_flag"].value_counts(dropna=True).to_dict()
    summary["health_flags"] = {_HEALTH_LABELS.get(key, key): int(value) for key, value in flags.items()}
    return summary
