"""限时批量快照和已完成分钟线适配，保留源时间并隔离过期行情。"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, time as clock_time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from services.database import stock_code
from services.finance_network import FinanceNetwork
from services.market_data import MarketDataError, _market_id, _number

SHANGHAI = ZoneInfo("Asia/Shanghai")


def market_is_open(now: datetime | None = None) -> bool:
    """只在工作日连续竞价时段启用信号，节假日还须通过行情源日期校验。"""
    now = now.astimezone(SHANGHAI) if now else datetime.now(SHANGHAI)
    return now.weekday() < 5 and (
        clock_time(9, 30) <= now.time() < clock_time(11, 30)
        or clock_time(13, 0) <= now.time() < clock_time(14, 57)
    )


def quote_is_fresh(quote: dict[str, Any], now: datetime | None = None, max_age: int = 20) -> bool:
    """秒级监控须有独立源时间，采集时间不替代实际行情时间。"""
    now = now or datetime.now(SHANGHAI)
    try:
        timestamp = datetime.fromisoformat(quote["quote_time"])
        price = float(quote["price"])
        age = (now - timestamp).total_seconds()
        return bool(quote.get("quote_valid")) and market_is_open(now) and np.isfinite(price) and price > 0 and 0 <= age <= max_age
    except (KeyError, TypeError, ValueError):
        return False


def normalize_minutes(frame: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """统一沪深京一分钟线；保守排除当前分钟、集合竞价及无成交记录。"""
    columns = ["date", "open", "high", "low", "close", "volume", "amount"]
    if frame.empty:
        raise MarketDataError("行情源未返回分钟线。")
    frame = frame.copy()
    timestamps = pd.to_datetime(frame["date"], errors="raise")
    frame["date"] = timestamps.dt.tz_localize(SHANGHAI) if timestamps.dt.tz is None else timestamps.dt.tz_convert(SHANGHAI)
    for column in columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce") if column in frame else np.nan
    current = pd.Timestamp(now or datetime.now(SHANGHAI)).floor("min")
    times = frame["date"].dt.time
    sessions = times.between(clock_time(9, 31), clock_time(11, 30)) | times.between(clock_time(13, 1), clock_time(15, 0))
    frame = frame.loc[sessions & (frame["date"] < current) & (frame["volume"] > 0)].copy()
    prices = frame[["open", "high", "low", "close"]]
    if frame.empty:
        raise MarketDataError("没有已经完成且有成交的分钟线。")
    if not np.isfinite(prices.to_numpy()).all() or not np.isfinite(frame["volume"]).all() or (prices <= 0).any(axis=None):
        raise MarketDataError("分钟线包含无效价格或成交量。")
    if (frame["high"] < prices[["open", "close"]].max(axis=1)).any() or (frame["low"] > prices[["open", "close"]].min(axis=1)).any():
        raise MarketDataError("分钟线开收盘价超出最高最低价。")
    frame["amount"] = frame["amount"].where(np.isfinite(frame["amount"]) & (frame["amount"] >= 0), None)
    return frame[columns].sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


class IntradayDataService:
    """批量轮询公开快照与一分钟线，采用短超时并明确返回逐股失败。"""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.trust_env = True
        self.session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/"})
        self.network = FinanceNetwork(self.session)
        self._last_request = 0.0

    def close(self) -> None:
        """关闭线程私有 HTTP 会话。"""
        self.session.close()
        self.network.close()

    def _get(self, url: str, params: dict[str, Any] | None = None) -> requests.Response:
        wait = 0.25 - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()
        response = self.network.get(url, params=params, timeout=(3, 5))
        response.raise_for_status()
        return response

    @staticmethod
    def _sina_quote(code: str, text: str) -> dict[str, Any]:
        parts = text.split(",")
        if len(parts) < 32:
            raise MarketDataError("新浪快照字段不完整。")
        price = _number(parts[3])
        if price is None or price <= 0:
            raise MarketDataError("新浪快照没有有效价格。")
        timestamp = datetime.fromisoformat(f"{parts[30]}T{parts[31]}+08:00")
        now = datetime.now(SHANGHAI)
        age = (now - timestamp).total_seconds()
        previous = _number(parts[2])
        return {
            "code": code, "name": parts[0], "price": price, "previous_close": previous,
            "change_pct": (price / previous - 1) * 100 if previous and previous > 0 else None,
            "open": _number(parts[1]), "high": _number(parts[4]), "low": _number(parts[5]),
            "volume_lots": (_number(parts[8]) or 0) / 100, "amount_yuan": _number(parts[9]),
            "source": "新浪财经行情快照", "quote_time": timestamp.isoformat(), "fetched_at": now.isoformat(),
            "quote_valid": 0 <= age <= 20,
            "data_status": "盘中快照" if market_is_open(now) and 0 <= age <= 20 else "最近可用快照（非实时）",
        }

    def _eastmoney_quote(self, code: str) -> dict[str, Any]:
        payload = self._get("https://push2.eastmoney.com/api/qt/stock/get", {
            "secid": f"{_market_id(code)}.{code}", "fields": "f57,f58,f43,f44,f45,f46,f47,f48,f59,f60,f86",
        }).json()
        data = payload.get("data") or {}
        if str(data.get("f57")) != code:
            raise MarketDataError("东方财富未返回匹配的股票快照。")
        scale = 10 ** int(data.get("f59") or 2)
        price = _number(data.get("f43"))
        if price is None or price <= 0:
            raise MarketDataError("东方财富未返回有效价格。")
        timestamp = datetime.fromtimestamp(int(data["f86"]), SHANGHAI) if data.get("f86") else None
        now = datetime.now(SHANGHAI)
        age = (now - timestamp).total_seconds() if timestamp else float("inf")
        previous = _number(data.get("f60"))
        previous = previous / scale if previous and previous > 0 else None
        return {
            "code": code, "name": data.get("f58", code), "price": price / scale, "previous_close": previous,
            "change_pct": ((price / scale) / previous - 1) * 100 if previous else None,
            "volume_lots": _number(data.get("f47")), "amount_yuan": _number(data.get("f48")),
            "source": "东方财富行情快照", "quote_time": timestamp.isoformat() if timestamp else None,
            "fetched_at": now.isoformat(), "quote_valid": 0 <= age <= 20,
            "data_status": "盘中快照" if market_is_open(now) and 0 <= age <= 20 else "最近可用快照（非实时）",
        }

    def get_quotes(self, codes: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
        """沪深股票批量拉取，北交所和失败代码逐股备用；不复用30秒旧缓存。"""
        codes = list(dict.fromkeys(stock_code(code) for code in codes))
        quotes = {}
        errors = []
        supported = [code for code in codes if not code.startswith(("4", "8", "92"))]
        if supported:
            symbols = [("sh" if _market_id(code) == 1 else "sz") + code for code in supported]
            try:
                response = self._get("https://hq.sinajs.cn/list=" + ",".join(symbols))
                text = response.content.decode("gbk", errors="replace")
                for match in re.finditer(r'hq_str_(?:sh|sz)([0-9]{6})="([^"]*)"', text):
                    if match.group(1) not in supported:
                        continue
                    try:
                        quote = self._sina_quote(match.group(1), match.group(2))
                        quotes[quote["code"]] = quote
                    except (ValueError, MarketDataError):
                        continue
            except requests.RequestException:
                pass
        for code in codes:
            if code not in quotes:
                try:
                    quotes[code] = self._eastmoney_quote(code)
                except (requests.RequestException, ValueError, TypeError, KeyError, MarketDataError) as exc:
                    errors.append(f"{code}：快照获取失败（{type(exc).__name__}）")
        return list(quotes.values()), errors

    def get_minutes(self, code: str) -> tuple[pd.DataFrame, str]:
        """优先读取东财近五日一分钟线，沪深失败时回退新浪不复权分钟线。"""
        code = stock_code(code)
        try:
            payload = self._get("https://push2his.eastmoney.com/api/qt/stock/trends2/get", {
                "secid": f"{_market_id(code)}.{code}", "ndays": 5, "iscr": 0,
                "fields1": "f1,f2,f3,f4,f5,f6,f7,f8", "fields2": "f51,f52,f53,f54,f55,f56,f57,f58",
            }).json()
            data = payload.get("data") or {}
            if data.get("code") and str(data["code"]) != code:
                raise MarketDataError("分钟线股票代码不匹配。")
            rows = [line.split(",")[:8] for line in data.get("trends", [])]
            frame = pd.DataFrame(rows, columns=["date", "open", "close", "high", "low", "volume", "amount", "average"])
            return normalize_minutes(frame), "东方财富一分钟线（不复权）"
        except (requests.RequestException, ValueError, TypeError, KeyError, MarketDataError) as primary:
            if code.startswith(("4", "8", "92")):
                raise MarketDataError("北交所分钟线暂不可用，未生成分钟信号。") from primary
            symbol = ("sh" if _market_id(code) == 1 else "sz") + code
            try:
                response = self._get("https://quotes.sina.cn/cn/api/jsonp_v2.php/=/CN_MarketDataService.getKLineData", {
                    "symbol": symbol, "scale": 1, "ma": "no", "datalen": 1200,
                })
                text = response.text
                frame = pd.DataFrame(json.loads(text[text.index("["):text.rindex("]") + 1]))
                frame = frame.rename(columns={"day": "date"})
                frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce") / 100
                return normalize_minutes(frame), "新浪一分钟线（不复权）"
            except (requests.RequestException, ValueError, TypeError, KeyError, MarketDataError) as fallback:
                raise MarketDataError(f"分钟线主备源均不可用（{type(fallback).__name__}）。") from primary
