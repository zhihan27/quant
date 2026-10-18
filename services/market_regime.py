"""根据同一交易日及以前的真实行情识别市场状态，缺项显式降级。"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from config import settings
from services.database import DatabaseError, _json

LOGGER = logging.getLogger("quant.market_regime")


def _history(frame: pd.DataFrame | None, date: pd.Timestamp) -> pd.DataFrame:
    """过滤未来日期与重复交易日，不填充缺失行情。"""
    if frame is None or frame.empty or "date" not in frame:
        return pd.DataFrame()
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    return result.loc[result["date"].le(date)].sort_values("date").drop_duplicates("date", keep="last")


def classify_market(
    histories: dict[str, pd.DataFrame], benchmark: pd.DataFrame | None, small_index: pd.DataFrame | None,
    reference_date: str, exposures: pd.DataFrame | None = None, full_market: bool = False,
    explicit_style: bool = False,
) -> dict[str, Any]:
    """返回大小盘相对动量、成交额、换手分位与申万行业前三；仅使用截至指定日的数据。"""
    date = pd.Timestamp(reference_date).normalize()
    warnings: list[str] = []
    result: dict[str, Any] = {"reference_date": str(date.date()), "label": "数据不足，无法识别市场状态",
        "relative_momentum": None, "index_ratio": None, "amount_yuan": None,
        "turnover_percentile": None, "amount_percentile": None, "top_industries": [],
        "warnings": warnings, "scope": "扫描股票池", "stocks": len(histories),
        "sources": {"indices": "中证1000 / 沪深300 日线", "liquidity": "扫描日线成交额与换手率"}}
    large, small = _history(benchmark, date), _history(small_index, date)
    if not large.empty and not small.empty and "close" in large and "close" in small:
        aligned = large[["date", "close"]].merge(small[["date", "close"]], on="date", suffixes=("_large", "_small"))
        aligned = aligned.set_index("date").apply(pd.to_numeric, errors="coerce")
        aligned = aligned.where(aligned > 0)
        window = int(settings.MARKET_REGIME_MOMENTUM_DAYS)
        # 两个指数必须共享当前端点和同一窗口起点，缺失交易日不能各自往前凑。
        calendar = large["date"].drop_duplicates().sort_values()
        if len(calendar) > window and calendar.iloc[-1] == date:
            start = calendar.iloc[-window - 1]
            if date in aligned.index and start in aligned.index and aligned.loc[[start, date]].notna().all().all():
                momentum = aligned.loc[date] / aligned.loc[start] - 1
                result["relative_momentum"] = float(momentum["close_small"] - momentum["close_large"])
                result["index_ratio"] = float(aligned.loc[date, "close_small"] / aligned.loc[date, "close_large"])
    if result["relative_momentum"] is None:
        warnings.append("大小盘指数缺失、交易日未对齐或动量窗口不足")
    chunks = []
    for code, history in histories.items():
        frame = _history(history, date)
        if frame.empty:
            continue
        chunk = frame[["date"]].copy()
        chunk["code"] = code
        for column in ("amount", "turnover_rate", "close"):
            chunk[column] = pd.to_numeric(frame[column], errors="coerce") if column in frame else np.nan
        chunk["daily_return"] = chunk["close"].pct_change(fill_method=None)
        chunk.loc[chunk["amount"].le(0), "amount"] = np.nan
        chunks.append(chunk)
    if chunks:
        panel = pd.concat(chunks, ignore_index=True)
        current = panel.loc[panel["date"].eq(date)].copy()
        required = max(5, int(np.ceil(len(histories) * 0.8)))
        amount = panel.groupby("date")["amount"].agg(["sum", "count"])
        totals = amount["sum"].where(amount["count"].ge(required))
        turnover = panel.groupby("date")["turnover_rate"].agg(["mean", "count"])
        turnover = turnover["mean"].where(turnover["count"].ge(required))
        for series, name in ((totals, "amount"), (turnover, "turnover")):
            trailing = series.tail(int(settings.MARKET_REGIME_PERCENTILE_DAYS)).dropna()
            if date in series.index and pd.notna(series.loc[date]):
                if name == "amount":
                    result["amount_yuan"] = float(series.loc[date])
                if len(trailing) >= settings.MARKET_REGIME_MINIMUM_DAYS:
                    result[f"{name}_percentile"] = float(trailing.le(series.loc[date]).mean())
            if result[f"{name}_percentile"] is None:
                warnings.append(f"{name} 历史覆盖不足，未计算分位")
        if full_market and len(current) >= settings.MINIMUM_EXPECTED_A_SHARE_UNIVERSE and current["amount"].notna().mean() >= 0.8:
            result["scope"] = "本轮全市场有效股票（不含停牌及采集失败）"
        else:
            warnings.append("流动性仅代表扫描样本，不能视为全市场总量")
        if exposures is not None and not exposures.empty and {"code", "industry", "industry_source"}.issubset(exposures):
            industry_map = exposures.loc[exposures["industry_source"].astype(str).str.contains("申万一级", regex=False)]
            industry_map = industry_map.drop_duplicates("code", keep="last").set_index("code")["industry"]
            current["industry"] = current["code"].map(industry_map)
            result["industry_coverage"] = float(current["industry"].notna().mean()) if len(current) else 0.0
            sectors = current.dropna(subset=["industry"]).groupby("industry").agg(
                daily_return=("daily_return", "mean"), amount_yuan=("amount", lambda values: values.sum(min_count=1)),
                stocks=("code", "nunique"))
            sectors = sectors.dropna(subset=["daily_return"])
            leaders = sectors.sort_values("daily_return", ascending=False).head(3)
            liquid = sectors.dropna(subset=["amount_yuan"]).sort_values("amount_yuan", ascending=False).head(3)
            result["top_industries"] = [{"industry": str(name), "daily_return": float(row["daily_return"]),
                "amount_yuan": float(row["amount_yuan"]) if pd.notna(row["amount_yuan"]) else None,
                "stocks": int(row["stocks"])} for name, row in leaders.iterrows()]
            result["top_amount_industries"] = list(liquid.index)
            warnings.append("行业为当前申万分类映射，历史成分未经点时核验；成交额不等同资金净流入")
        if not result["top_industries"]:
            warnings.append("申万一级行业缺失，行业轮动未接入")
    else:
        warnings.append("没有可用股票日线，流动性与行业轮动未计算")
    gap = result["relative_momentum"]
    if gap is not None:
        style = "小盘相对占优" if gap >= settings.MARKET_REGIME_STYLE_GAP else (
            "大盘相对占优" if gap <= -settings.MARKET_REGIME_STYLE_GAP else "大小盘风格均衡")
        heat = result["turnover_percentile"]
        amount_heat = result["amount_percentile"]
        state = ""
        if heat is not None and amount_heat is not None:
            state = "；放量高热" if min(heat, amount_heat) >= 0.8 else (
                "；缩量偏冷" if max(heat, amount_heat) <= 0.2 else "；流动性平稳")
        result["label"] = style + state
        if explicit_style and abs(gap) >= settings.MARKET_REGIME_STYLE_GAP:
            warnings.append("市场风格分化明显，显式风格模型需复核市值暴露；无法仅凭状态断言模型失效")
    result["degraded"] = bool(warnings)
    for warning in warnings:
        LOGGER.warning("市场状态降级 date=%s detail=%s", reference_date, warning)
    return result


def save_regime(repository: Any, regime: dict[str, Any], context_id: str | None, trigger: str) -> None:
    """将市场状态随扫描批次追加落库；SQL 使用参数绑定，失败向调用方反馈。"""
    try:
        with repository.connect() as connection:
            connection.execute(
                "INSERT INTO public.quant_market_regime(reference_date,context_id,model_version,trigger,label,details) "
                "VALUES (%s,%s,%s,%s,%s,%s)",
                (regime["reference_date"], context_id, settings.MODEL_VERSION, trigger, regime["label"], _json(regime)),
            )
    except DatabaseError:
        LOGGER.exception("市场状态落库失败 date=%s", regime.get("reference_date"))
        raise


def latest_regime(repository: Any) -> dict[str, Any] | None:
    """读取最新交易日的扫描状态，旧快照重训不覆盖更新的交易日。"""
    with repository.connect() as connection:
        row = connection.execute(
            "SELECT details,created_at FROM public.quant_market_regime ORDER BY reference_date DESC,created_at DESC,id DESC LIMIT 1"
        ).fetchone()
    return {**row["details"], "recorded_at": str(row["created_at"])} if row else None
