"""历史估值来源适配与披露版本驱动的点时重建，禁止使用当前股本回填。"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from config import settings
from services.style_factors import normalize_history

LOGGER = logging.getLogger("quant.valuation_data")


def fetch_daily_basic(code: str, start: str, end: str, session: Any = None) -> pd.DataFrame:
    """按股票分年采集历史日指标，令单次记录数远低于接口上限；令牌只读环境变量。"""
    token = os.environ.get("TUSHARE_TOKEN", "")
    if not token:
        raise ValueError("未配置 TUSHARE_TOKEN；历史估值保持未接入，不使用当前快照替代。")
    if len(code) != 6 or not code.isdigit():
        raise ValueError("股票代码必须为六位数字。")
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    if first > last:
        raise ValueError("开始日期不能晚于结束日期。")
    suffix = "SH" if code.startswith("6") else ("BJ" if code.startswith(("4", "8", "92")) else "SZ")
    frames = []
    client = session or requests
    for year in range(first.year, last.year + 1):
        left = max(first, pd.Timestamp(year, 1, 1)).strftime("%Y%m%d")
        right = min(last, pd.Timestamp(year, 12, 31)).strftime("%Y%m%d")
        try:
            response = client.post("https://api.tushare.pro", json={
                "api_name": "daily_basic", "token": token,
                "params": {"ts_code": f"{code}.{suffix}", "start_date": left, "end_date": right},
                "fields": "ts_code,trade_date,close,total_share,float_share,total_mv,circ_mv,pe_ttm,pb,ps_ttm",
            }, timeout=(settings.REQUEST_CONNECT_TIMEOUT_SECONDS, settings.REQUEST_READ_TIMEOUT_SECONDS))
            response.raise_for_status()
            payload = response.json()
            if payload.get("code") != 0:
                # 不原样记录服务端响应，避免上游错误文本包含令牌。
                raise ValueError("历史指标接口拒绝请求，请核对账号权限、积分与请求配额。")
            data = payload.get("data") or {}
            chunk = pd.DataFrame(data.get("items") or [], columns=data.get("fields") or [])
            if len(chunk) >= 6000:
                raise ValueError("历史指标返回达到上限，可能被截断，未保存该批数据。")
            if not chunk.empty:
                frames.append(chunk)
        except (requests.RequestException, ValueError, KeyError):
            LOGGER.warning("历史日指标采集失败 code=%s year=%s", code, year)
            raise
    if not frames:
        return pd.DataFrame()
    return normalize_daily_basic(pd.concat(frames, ignore_index=True))


def normalize_daily_basic(frame: pd.DataFrame) -> pd.DataFrame:
    """将万元/万股换算为元/股；历史市值可用，未经版本核验的估值仅供追溯。"""
    required = {"ts_code", "trade_date", "total_mv", "circ_mv", "total_share", "float_share", "close"}
    if not required.issubset(frame.columns):
        raise ValueError("daily_basic 缺少历史市值、股本或未复权收盘价字段。")
    result = frame.copy()
    result["code"] = result["ts_code"].astype(str).str.split(".").str[0]
    result["date"] = pd.to_datetime(result["trade_date"].astype(str), format="%Y%m%d", errors="raise")
    now = datetime.now(ZoneInfo("Asia/Shanghai")).replace(tzinfo=None)
    for target, source in (("market_cap_total", "total_mv"), ("market_cap_float", "circ_mv"),
                           ("total_shares", "total_share"), ("float_shares", "float_share")):
        result[target] = pd.to_numeric(result[source], errors="coerce") * 10000
    result["raw_close"] = pd.to_numeric(result["close"], errors="coerce")
    valid = pd.Series(True, index=result.index)
    for cap, shares in (("market_cap_total", "total_shares"), ("market_cap_float", "float_shares")):
        reconstructed = result["raw_close"] * result[shares]
        valid &= reconstructed.gt(0) & np.isclose(result[cap], reconstructed, rtol=0.005, atol=10000)
    valid &= result["float_shares"].le(result["total_shares"])
    result["available_at"] = result["date"] + pd.Timedelta(hours=17)
    result["collected_at"] = now
    result["source"] = "Tushare daily_basic"
    result["unit"] = "CNY/share"
    result["size_pit"] = valid
    result["value_pit"] = False
    result["quality"] = np.where(valid, "历史日市值；估值修订版本未经核验", "价格股本与市值校验失败")
    result["revision_id"] = [hashlib.sha256(json.dumps(row, sort_keys=True, default=str).encode()).hexdigest()[:24]
                             for row in frame.to_dict("records")]
    return normalize_history(result)


def reconstruct_valuation(prices: pd.DataFrame, shares: pd.DataFrame, statements: pd.DataFrame) -> pd.DataFrame:
    """用同日未复权价格、历史股本事件及公告后版本重建估值。

    股本须提供 effective_date、ann_date、source、revision_id、collected_at、unit；
    财报须提供 report_period、ann_date、利润/收入累计值与净资产的原始披露版本。
    不接受将当前报表最早公告日伪装成原始版本的输入。
    """
    price_fields = {"code", "date", "raw_close"}
    meta = {"code", "ann_date", "source", "revision_id", "collected_at", "unit", "pit_verified"}
    if not price_fields.issubset(prices) or not (meta | {"effective_date", "total_shares", "float_shares"}).issubset(shares):
        raise ValueError("点时重建缺少未复权价格、历史股本事件或版本血缘。")
    if not statements.empty and not (meta | {"report_period", "net_profit_ytd", "revenue_ytd", "book_equity"}).issubset(statements):
        raise ValueError("财报必须包含原始披露版本及累计利润、收入、净资产。")
    if prices.duplicated(["code", "date"]).any():
        raise ValueError("未复权价格存在同股同日重复记录。")
    rows = []
    share_data, financial = shares.copy(), statements.copy()
    for frame in (share_data, financial):
        if frame.empty:
            continue
        if not frame["unit"].eq("CNY/share").all():
            raise ValueError("点时重建输入金额须为元、股本须为股。")
        frame["ann_date"] = pd.to_datetime(frame["ann_date"], errors="raise").dt.normalize()
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        frame.drop(frame.index[~frame["pit_verified"].eq(True)], inplace=True)
        for column in ("source", "revision_id", "collected_at"):
            if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
                raise ValueError(f"点时重建输入 {column} 不能为空。")
        collected = pd.to_datetime(frame["collected_at"], errors="raise")
        if isinstance(collected.dtype, pd.DatetimeTZDtype):
            collected = collected.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
        frame["collected_at"] = collected
    for column in ("total_shares", "float_shares"):
        share_data[column] = pd.to_numeric(share_data[column], errors="coerce")
    share_data["effective_date"] = pd.to_datetime(share_data["effective_date"], errors="raise")
    if not financial.empty:
        financial["report_period"] = pd.to_datetime(financial["report_period"], errors="raise")
        for column in ("net_profit_ytd", "revenue_ytd", "book_equity"):
            financial[column] = pd.to_numeric(financial[column], errors="coerce")
    for row in prices.sort_values(["date", "code"]).to_dict("records"):
        code, date = str(row["code"]).zfill(6), pd.Timestamp(row["date"]).normalize()
        known = share_data.loc[share_data["code"].eq(code) & share_data["ann_date"].lt(date) & share_data["effective_date"].le(date)]
        if known.empty:
            continue
        capital = known.sort_values(["effective_date", "ann_date", "revision_id"]).iloc[-1]
        price = pd.to_numeric(pd.Series([row["raw_close"]]), errors="coerce").iloc[0]
        record = {"code": code, "date": date, "market_cap_total": price * float(capital["total_shares"]),
                  "market_cap_float": price * float(capital["float_shares"]), "raw_close": price,
                  "available_at": date + pd.Timedelta(hours=17), "collected_at": capital["collected_at"],
                  "source": str(capital["source"]), "revision_id": str(capital["revision_id"]),
                  "unit": "CNY/share", "quality": "未复权价乘当时股本；仅使用前一日及以前公告版本",
                  "size_pit": bool(np.isfinite(price) and price > 0
                      and np.isfinite(capital["total_shares"]) and np.isfinite(capital["float_shares"])
                      and 0 < capital["float_shares"] <= capital["total_shares"]), "value_pit": False}
        if not financial.empty:
            known_financial = financial.loc[financial["code"].eq(code) & financial["ann_date"].lt(date) & financial["report_period"].lt(date)]
            facts = known_financial.sort_values(["ann_date", "revision_id"]).drop_duplicates("report_period", keep="last").set_index("report_period")
            if not facts.empty:
                period = facts.index.max()
                latest = facts.loc[period]
                record["book_equity"] = latest["book_equity"]
                used = facts.loc[[period]]
                for source, target in (("net_profit_ytd", "net_profit_ttm"), ("revenue_ytd", "revenue_ttm")):
                    if period.month == 12:
                        record[target] = latest[source]
                    else:
                        annual, prior = pd.Timestamp(period.year - 1, 12, 31), period - pd.DateOffset(years=1)
                        if annual in facts.index and prior in facts.index:
                            record[target] = latest[source] + facts.loc[annual, source] - facts.loc[prior, source]
                            used = facts.loc[[period, annual, prior]]
                record["value_pit"] = True
                record["source"] += "；" + "；".join(sorted(used["source"].astype(str).unique()))
                record["revision_id"] += ";" + ";".join(used["revision_id"].astype(str))
                record["financial_ann_date"] = used["ann_date"].max()
                record["report_period"] = period
                record["collected_at"] = max(capital["collected_at"], used["collected_at"].max())
                # 保留原始倍数供追溯；亏损、负净资产或异常收入不生成无定义倍数。
                for multiple, numerator in (("pe_ttm", "net_profit_ttm"), ("pb", "book_equity"), ("ps_ttm", "revenue_ttm")):
                    value = record.get(numerator, np.nan)
                    record[multiple] = record["market_cap_total"] / value if record["size_pit"] and value > 0 else np.nan
        rows.append(record)
    return normalize_history(pd.DataFrame(rows)) if rows else pd.DataFrame()
