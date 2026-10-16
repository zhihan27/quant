"""按持仓批次汇总成本、可卖数量和带时间标记的参考浮盈亏。"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

from services.intraday_data import quote_is_fresh


def summarize_positions(
    positions: list[dict[str, Any]], quotes: dict[str, dict[str, Any]], now: datetime | None = None
) -> list[dict[str, Any]]:
    """同代码多批持仓加权汇总，买入当日可卖数强制为零，缺失报价不以零估值。"""
    now = now or datetime.now(ZoneInfo("Asia/Shanghai"))
    summary = {}
    for position in positions:
        code = position["code"]
        row = summary.setdefault(code, {
            "code": code, "name": position["name"], "shares": 0, "available_shares": 0, "cost": Decimal(0), "lots": 0,
        })
        row["shares"] += int(position["shares"])
        row["available_shares"] += int(position["available_shares"]) if position["buy_date"] < now.date() else 0
        row["cost"] += Decimal(str(position["buy_price"])) * int(position["shares"]) + Decimal(str(position["fees"]))
        row["lots"] += 1
    for code, row in summary.items():
        quote = quotes.get(code, {})
        price = quote.get("price")
        row["average_cost"] = float(row["cost"] / row["shares"])
        row["cost"] = float(row["cost"])
        row["price"] = float(price) if price is not None else None
        row["market_value"] = row["price"] * row["shares"] if price is not None else None
        row["pnl"] = row["market_value"] - row["cost"] if price is not None else None
        row["pnl_pct"] = row["pnl"] / row["cost"] * 100 if price is not None else None
        row["quote_time"] = quote.get("quote_time")
        row["data_status"] = "盘中快照" if quote_is_fresh(quote, now) else (quote.get("data_status", "尚无报价") + "；参考估值")
    return list(summary.values())
