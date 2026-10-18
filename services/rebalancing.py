"""使用持仓与冻结完整排名生成次日人工调仓清单，不发送交易指令。"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from config import settings
from services.portfolio import summarize_positions


def _number(value: Any) -> float | None:
    """有限数值才可参与止损和风控判断。"""
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _quote_is_current(row: dict[str, Any], date: pd.Timestamp, now: datetime) -> bool:
    """排名行和行情必须来自同一截面，且行情时间不能晚于复核时点。"""
    if pd.isna(date) or row.get("reference_date") != str(date.date()):
        return False
    if row.get("daily_bar_date"):
        # 日线来源日期独立记录，不把快照采集时间冒充逐股行情时间。
        bar_date = pd.to_datetime(row["daily_bar_date"], errors="coerce")
        if pd.isna(bar_date) or bar_date.date() != date.date():
            return False
        close_time = pd.Timestamp(date.date()).tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)
        return bool(close_time <= pd.Timestamp(now))
    stamp = pd.to_datetime(row.get("quote_time"), errors="coerce")
    if pd.isna(stamp):
        return False
    stamp = stamp.tz_localize("Asia/Shanghai") if stamp.tzinfo is None else stamp.tz_convert("Asia/Shanghai")
    return bool(stamp.date() == date.date() and stamp <= pd.Timestamp(now))


def _signal_is_current(row: dict[str, Any], now: datetime) -> bool:
    """信号生成时间与有效期都必须明确，缺失或过期时不生成调入任务。"""
    stamps = [pd.to_datetime(row.get(key), errors="coerce")
              for key in ("entry_generated_at", "entry_valid_until")]
    if any(pd.isna(stamp) for stamp in stamps):
        return False
    generated, expires = [stamp.tz_localize("Asia/Shanghai") if stamp.tzinfo is None
                          else stamp.tz_convert("Asia/Shanghai") for stamp in stamps]
    return bool(generated <= pd.Timestamp(now) <= expires)


def ranking_snapshot(records: list[dict[str, Any]], reference_date: str | None) -> list[dict[str, Any]]:
    """保留所有股票的模型排名和交易门控，展示行数限制不参与任务单判断。"""
    ranked = []
    for item in records:
        models = (item.get("analysis") or {}).get("models") or {}
        validated = [model for model in models.get(20, []) if model.get("validated")
                     and _number(model.get("prediction_percentile")) is not None]
        score = sum(float(model["prediction_percentile"]) for model in validated) / len(validated) if validated else None
        features = item.get("features") or {}
        quote = item.get("quote") or {}
        plan = (item.get("analysis") or {}).get("trade_plan") or {}
        bar_date = pd.to_datetime(features.get("date"), errors="coerce")
        daily_bar_date = str(bar_date.date()) if pd.notna(bar_date) else None
        has_daily_bar = daily_bar_date is not None
        ranked.append({"code": item["code"], "name": item.get("name", item["code"]),
            "score": score, "rank": None, "reference_date": reference_date,
            "model_validated": bool(validated),
            "price": _number(features.get("close") if has_daily_bar else quote.get("price")),
            "daily_bar_date": daily_bar_date,
            "quote_time": quote.get("quote_time"), "quote_valid": bool(quote.get("quote_valid")),
            "amount_yuan": _number(features.get("amount") if has_daily_bar else quote.get("amount_yuan", quote.get("amount"))),
            "volume_lots": _number(features.get("volume") if has_daily_bar else quote.get("volume_lots")),
            "ma20_gap": _number(features.get("ma20_gap")),
            "entry_allowed": bool(plan.get("close_signal_active")),
            "entry_generated_at": plan.get("generated_at"), "entry_valid_until": plan.get("valid_until"),
            "entry_trade_status": item.get("entry_trade_status"), "error": item.get("error")})
    eligible = sorted([row for row in ranked if row["score"] is not None], key=lambda row: (-row["score"], row["code"]))
    for rank, row in enumerate(eligible, 1):
        row["rank"] = rank
    return ranked


def build_checklist(
    positions: list[dict[str, Any]], scan: dict[str, Any] | None, top_n: int = settings.PORTFOLIO_TOP_N,
    now: datetime | None = None, scoring_paused: bool = False,
) -> dict[str, Any]:
    """返回调出、调入、维持和待核查清单；缺数或过期排名不推断为跌出 Top N。"""
    if top_n < 1:
        raise ValueError("Top N 必须为正整数。")
    now = now or datetime.now(ZoneInfo("Asia/Shanghai"))
    if now.tzinfo is None:
        now = now.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    result = {"sell": [], "buy": [], "hold": [], "review": [], "warnings": [],
              "reference_date": (scan or {}).get("reference_date"), "context_id": (scan or {}).get("context_id")}
    date = pd.to_datetime(result["reference_date"], errors="coerce")
    usable = bool(scan and scan.get("model_version") == settings.MODEL_VERSION and scan.get("context_id")
                  and scan.get("pool_mode") == "沪深京全部 A 股" and pd.notna(date)
                  and 0 <= (now.date() - date.date()).days <= settings.CHECKLIST_MAXIMUM_SCAN_AGE_DAYS)
    rows = (scan or {}).get("actionable_rankings") or []
    ranking_ready = usable and not scoring_paused and any(row.get("rank") is not None and row.get("model_validated") for row in rows)
    if not ranking_ready:
        result["warnings"].append("缺少有效全市场模型排名、扫描已过期或模型降级；不生成基于排名的买卖任务")
    lookup = {str(row["code"]).zfill(6): row for row in rows}
    quotes = {code: {"price": row.get("price"), "quote_time": row.get("quote_time")} for code, row in lookup.items()}
    summaries = summarize_positions(positions, quotes, now=now)
    held = {row["code"] for row in summaries}
    for position in summaries:
        row = lookup.get(position["code"], {})
        price = _number(row.get("price"))
        price_ready = bool(usable and price is not None and price > 0 and row.get("quote_valid")
                           and _quote_is_current(row, date, now))
        item = {"代码": position["code"], "名称": position["name"], "股数": position["shares"],
                "当前确认可卖": position["available_shares"], "模型排名": row.get("rank"),
                "模型评分": row.get("score"), "参考价格": price if price_ready else None,
                "参考浮盈亏": position["pnl"] if price_ready else None,
                "参考盈亏%": position["pnl_pct"] if price_ready else None}
        bought_after_scan = any(str(batch["code"]) == position["code"] and pd.Timestamp(batch["buy_date"]).date() > date.date()
                                for batch in positions) if pd.notna(date) else True
        if not price_ready or row.get("error") or bought_after_scan:
            item["原因"] = "缺少同日有效行情或持仓晚于排名截面，需人工核查"
            result["review"].append(item)
            continue
        reasons = []
        stop = position["average_cost"] * (1 - settings.CHECKLIST_STOP_LOSS_RATIO)
        item["固定成本止损线"] = round(stop, 4)
        if price <= stop:
            reasons.append("触发固定成本初始止损")
        gap = _number(row.get("ma20_gap"))
        if gap is not None and gap < 0:
            reasons.append("收盘跌破20日均线，触发趋势退出")
        if ranking_ready and row.get("model_validated") and row.get("rank") is not None and row["rank"] > top_n:
            reasons.append(f"模型排名已跌出 Top {top_n}")
        if reasons:
            item["原因"] = "；".join(reasons)
            result["sell"].append(item)
        elif not ranking_ready or not row.get("model_validated") or row.get("rank") is None or gap is None:
            item["原因"] = "排名或趋势数据不完整，未假定应继续持有或调出"
            result["review"].append(item)
        else:
            item["原因"] = f"仍在 Top {top_n} 且未触发止损或趋势退出"
            result["hold"].append(item)
    if ranking_ready:
        for row in sorted(rows, key=lambda item: item.get("rank") or math.inf):
            if row["code"] in held or row.get("rank") is None or row["rank"] > top_n:
                continue
            amount, volume = _number(row.get("amount_yuan")), _number(row.get("volume_lots"))
            allowed = bool(row.get("model_validated") and row.get("quote_valid") and row.get("entry_allowed")
                and _signal_is_current(row, now)
                and row.get("entry_trade_status") == "可正常评估" and not row.get("error")
                and amount is not None and amount >= settings.CHECKLIST_MINIMUM_AMOUNT
                and volume is not None and volume > 0 and _quote_is_current(row, date, now)
                and _number(row.get("price")) is not None and row["price"] > 0)
            item = {"代码": row["code"], "名称": row["name"], "模型排名": row["rank"],
                    "模型评分": row["score"], "参考价格": row.get("price"), "成交额（元）": amount}
            item["原因"] = "进入全市场前列，流动性、趋势与风控门槛通过" if allowed else "进入前列但流动性、行情或风控门槛未通过，禁止自动补零放行"
            result["buy" if allowed else "review"].append(item)
    result["warnings"].append("任务单供下一实际交易日盘前复核；开盘价、停牌、涨跌停与券商可卖数须重新确认")
    return result
