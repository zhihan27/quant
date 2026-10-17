"""日内高频因子：波动率衰减与订单流不平衡代理。

模块四（波动率衰减）：把连续竞价时段切成 8 个等长 30 分钟区间，逐区间计算分钟收益率标准差，
再对区间序号回归得到衰减斜率，并给出「尾盘 30 分钟 / 开盘 30 分钟」波动比，用来刻画盘中多头
能量衰竭或抛压释放的过程。

模块五（OFI 代理）：本地暂无 L2 逐档快照时，用分钟线做两个代理——按价格变动方向拆分的主动
买卖量净额（``intraday_ofi_proxy``），以及成交均价在当日高低区间中的位置
（``intraday_vwap_position``）。若后续接入 L2 盘口，可直接改用标准 ``order_flow_imbalance``，
该函数已按最优买卖价与挂单量变动的口径实现。

无未来函数约束：某交易日的日内因子只由该交易日自身的分钟线计算，不跨日、不插补。
"""

from __future__ import annotations

import logging
from datetime import time as clock_time
from typing import Any

import numpy as np
import pandas as pd

from config import settings
from services.panel_store import PanelStore
from services.progress import throttled

LOGGER = logging.getLogger("quant.intraday_factors")

INTRADAY_FACTOR_COLUMNS = [
    "intraday_vol_decay_slope",
    "intraday_vol_tail_open_ratio",
    "intraday_ofi_proxy",
    "intraday_vwap_position",
]
# 连续竞价 240 分钟等分为 8 个 30 分钟区间；上午 4 段、下午 4 段。
SESSION_BUCKETS = (
    (clock_time(9, 31), clock_time(10, 0)),
    (clock_time(10, 1), clock_time(10, 30)),
    (clock_time(10, 31), clock_time(11, 0)),
    (clock_time(11, 1), clock_time(11, 30)),
    (clock_time(13, 1), clock_time(13, 30)),
    (clock_time(13, 31), clock_time(14, 0)),
    (clock_time(14, 1), clock_time(14, 30)),
    (clock_time(14, 31), clock_time(15, 0)),
)
OPEN_BUCKETS = 1
TAIL_BUCKETS = 2
MINIMUM_BUCKETS = 4


def order_flow_imbalance(
    bid_price: Any, bid_size: Any, ask_price: Any, ask_size: Any
) -> np.ndarray:
    """标准 OFI（Cont 等 2014）：最优买卖价与挂单量变动带来的净委托流。

    仅供未来接入 L2 逐档快照时使用；本地无 L2 数据时以 ``ofi_proxy`` 替代。
    """
    bid_price = np.asarray(bid_price, dtype=float)
    bid_size = np.asarray(bid_size, dtype=float)
    ask_price = np.asarray(ask_price, dtype=float)
    ask_size = np.asarray(ask_size, dtype=float)
    if not (len(bid_price) == len(bid_size) == len(ask_price) == len(ask_size)):
        raise ValueError("OFI 输入长度必须一致。")
    flow = np.zeros(len(bid_price), dtype=float)
    for position in range(1, len(bid_price)):
        previous_bid, current_bid = bid_price[position - 1], bid_price[position]
        if current_bid > previous_bid:
            flow[position] = flow[position - 1] + bid_size[position]
        elif current_bid == previous_bid:
            flow[position] = flow[position - 1] + (bid_size[position] - bid_size[position - 1])
        else:
            flow[position] = flow[position - 1] - bid_size[position - 1]
        previous_ask, current_ask = ask_price[position - 1], ask_price[position]
        if current_ask < previous_ask:
            flow[position] -= ask_size[position]
        elif current_ask == previous_ask:
            flow[position] -= ask_size[position] - ask_size[position - 1]
        else:
            flow[position] += ask_size[position - 1]
    return flow


def _bucket_index(times: pd.Series) -> pd.Series:
    """把分钟线时间映射到 8 个区间序号；不属于任何区间的记录标为 -1。"""
    index = pd.Series(-1, index=times.index, dtype=int)
    for position, (start, end) in enumerate(SESSION_BUCKETS):
        index = index.mask(times.between(start, end), position)
    return index


def _mean_of_finite(values: np.ndarray) -> float:
    """只在有限值上求均值；全缺失时返回缺失，避免对空切片取均值产生噪声警告。"""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else float("nan")


def _decay_slope(values: np.ndarray) -> float:
    """区间波动对区间序号的最小二乘斜率；区间过少或方差退化时返回缺失。"""
    mask = np.isfinite(values)
    if int(mask.sum()) < MINIMUM_BUCKETS:
        return float("nan")
    x = np.arange(len(values), dtype=float)[mask]
    y = values[mask]
    x_centred = x - x.mean()
    denominator = float(np.sum(x_centred**2))
    if denominator <= 1e-12:
        return float("nan")
    return float(np.sum(x_centred * (y - y.mean())) / denominator)


def intraday_day_factors(minutes: pd.DataFrame, minimum_bars: int | None = None) -> pd.DataFrame:
    """按交易日计算日内因子，每个交易日只使用自身的分钟线。"""
    required = {"date", "high", "low", "close", "volume"}
    missing = required - set(minutes.columns)
    if missing:
        raise ValueError(f"日内因子缺少字段：{', '.join(sorted(missing))}")
    minimum_bars = int(settings.INTRADAY_MINIMUM_BARS_PER_DAY if minimum_bars is None else minimum_bars)
    frame = minutes.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce") if column in frame else np.nan
    if "amount" in frame:
        frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce")
    frame = frame.sort_values("date").reset_index(drop=True)
    frame["trading_day"] = frame["date"].dt.normalize()
    frame["bucket"] = _bucket_index(frame["date"].dt.time)

    rows: list[dict[str, Any]] = []
    for day, group in frame.groupby("trading_day", sort=True):
        group = group.reset_index(drop=True)
        if len(group) < minimum_bars:
            continue
        returns = group["close"].pct_change(fill_method=None)
        bucket_volatility = np.full(len(SESSION_BUCKETS), np.nan, dtype=float)
        for position in range(len(SESSION_BUCKETS)):
            sample = returns.loc[group["bucket"].eq(position)]
            if sample.notna().sum() >= 2:
                bucket_volatility[position] = float(sample.std(ddof=1))
        present = np.isfinite(bucket_volatility).sum()
        decay_slope = _decay_slope(bucket_volatility)
        # 尾盘区间缺失（如盘中未收盘）时比值不成立，不用部分数据硬凑。
        head = _mean_of_finite(bucket_volatility[:OPEN_BUCKETS])
        tail = _mean_of_finite(bucket_volatility[-TAIL_BUCKETS:])
        ratio = float(tail / head) if np.isfinite(head) and head > 1e-12 and np.isfinite(tail) else float("nan")
        rows.append(
            {
                "date": day,
                "intraday_vol_decay_slope": decay_slope,
                "intraday_vol_tail_open_ratio": ratio,
                "intraday_ofi_proxy": _ofi_proxy(group),
                "intraday_vwap_position": _vwap_position(group),
                "intraday_buckets": int(present),
                "intraday_bars": int(len(group)),
            }
        )
    columns = ["date", *INTRADAY_FACTOR_COLUMNS, "intraday_buckets", "intraday_bars"]
    return pd.DataFrame(rows, columns=columns)


def _ofi_proxy(group: pd.DataFrame) -> float:
    """主动买卖量代理：按分钟收盘价涨跌方向归集成交量，返回净买入占比。

    分钟线不含逐笔方向，这里用「价格上涨记主动买、下跌记主动卖、持平对半分」的 tick rule 近似，
    属于代理指标而非真实委托流。
    """
    volume = pd.to_numeric(group["volume"], errors="coerce").fillna(0.0)
    change = pd.to_numeric(group["close"], errors="coerce").diff()
    buy = volume.where(change > 0, 0.0) + volume.where(change == 0, 0.0) / 2.0
    sell = volume.where(change < 0, 0.0) + volume.where(change == 0, 0.0) / 2.0
    total = float(buy.sum() + sell.sum())
    if total <= 0:
        return float("nan")
    return float((buy.sum() - sell.sum()) / total)


def _vwap_position(group: pd.DataFrame) -> float:
    """成交均价在当日最高最低区间中的位置：0 贴近最低价，1 贴近最高价。

    高价源不保证 ``amount`` 与 ``volume`` 的单位一致（成交量常以「手」计），直接相除会得到
    偏离量级的价格，因此这里改用「分钟典型价按成交量加权」，与成交量单位无关。
    """
    volume = pd.to_numeric(group["volume"], errors="coerce")
    typical = pd.concat(
        [
            pd.to_numeric(group["high"], errors="coerce"),
            pd.to_numeric(group["low"], errors="coerce"),
            pd.to_numeric(group["close"], errors="coerce"),
        ],
        axis=1,
    ).mean(axis=1)
    weighted = volume.where(volume > 0)
    total_volume = float(weighted.sum())
    if total_volume > 0 and weighted.notna().any():
        vwap = float((typical * weighted).sum(skipna=True) / weighted.sum(skipna=True))
    else:
        vwap = float(typical.mean())
    high = float(pd.to_numeric(group["high"], errors="coerce").max())
    low = float(pd.to_numeric(group["low"], errors="coerce").min())
    if not np.isfinite(vwap) or not np.isfinite(high) or not np.isfinite(low) or high - low <= 1e-12:
        return float("nan")
    return float(np.clip((vwap - low) / (high - low), 0.0, 1.0))


# ------------------------------------------------------------------ 分钟线落盘累积
def merge_intraday_bars(existing: pd.DataFrame | None, incoming: pd.DataFrame, max_days: int | None = None) -> pd.DataFrame:
    """合并新旧分钟线：按时间戳去重保留最新，并只保留最近 max_days 个交易日。"""
    frames = [frame for frame in (existing, incoming) if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "amount"])
    merged = pd.concat(frames, ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged = merged.dropna(subset=["date"]).sort_values("date")
    merged = merged.drop_duplicates("date", keep="last").reset_index(drop=True)
    max_days = int(settings.INTRADAY_PANEL_MAX_DAYS if max_days is None else max_days)
    if max_days > 0:
        days = np.sort(merged["date"].dt.normalize().unique())
        if len(days) > max_days:
            kept = set(days[-max_days:])
            merged = merged.loc[merged["date"].dt.normalize().isin(kept)].reset_index(drop=True)
    return merged


def store_intraday_bars(store: PanelStore, code: str, bars: pd.DataFrame, max_days: int | None = None) -> dict[str, Any]:
    """把新抓到的分钟线并入本地面板，返回累积后的覆盖报告。"""
    existing = store.read_partition("intraday_bars", code)
    merged = merge_intraday_bars(existing, bars, max_days)
    store.write_partition("intraday_bars", code, merged)
    days = pd.to_datetime(merged["date"]).dt.normalize()
    return {
        "code": str(code).zfill(6),
        "bars": int(len(merged)),
        "trading_days": int(days.nunique()),
        "first": str(days.min().date()) if len(days) else None,
        "last": str(days.max().date()) if len(days) else None,
    }


def load_intraday_factors(
    store: PanelStore, codes: list[str], progress: Any = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """从本地累积的分钟线分区重算日内因子，打分链路无需再次联网抓取。

    仅使用已落盘的历史分钟线，因此覆盖范围取决于本地累积深度，不做任何插补。
    """
    frames: list[pd.DataFrame] = []
    empty: list[str] = []
    should_report = throttled(len(codes))
    for index, code in enumerate(codes):
        code = str(code).zfill(6)
        if progress and should_report(index):
            progress(f"正在汇总分钟线（{index + 1}/{len(codes)}）")
        bars = store.read_partition("intraday_bars", code)
        if bars is None or bars.empty:
            empty.append(code)
            continue
        try:
            factors = intraday_day_factors(bars)
        except (ValueError, KeyError, TypeError) as exc:
            LOGGER.warning("本地分钟线无法计算日内因子 code=%s detail=%s", code, exc)
            empty.append(code)
            continue
        if factors.empty:
            empty.append(code)
            continue
        factors["code"] = code
        frames.append(factors)
    columns = ["date", "code", *INTRADAY_FACTOR_COLUMNS, "intraday_buckets", "intraday_bars"]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    return combined, {
        "codes_requested": len(codes),
        "codes_computed": len(frames),
        "rows": int(len(combined)),
        "codes_without_bars": len(empty),
        "status": (
            f"从本地分钟线汇总 {len(frames)} 只股票的日内因子" if frames else "本地暂无分钟线，未产生日内因子"
        ),
        "limitation": "日内因子深度取决于本地累积的分钟线交易日数，未覆盖到的交易日留空不插补。",
    }


def collect_intraday_factors(
    store: PanelStore,
    codes: list[str],
    service: Any = None,
    progress: Any = None,
    persist_bars: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """逐股抓取分钟线、累积落盘并计算日内因子，返回长表与覆盖报告。"""
    from services.intraday_data import IntradayDataService, MarketDataError, normalize_minutes

    service = service or IntradayDataService()
    frames: list[pd.DataFrame] = []
    failures: list[str] = []
    for index, code in enumerate(codes):
        code = str(code).zfill(6)
        if progress:
            progress(f"正在获取 {code} 的分钟线（{index + 1}/{len(codes)}）")
        try:
            minutes, source = service.get_minutes(code)
        except (MarketDataError, ValueError, OSError) as exc:
            failures.append(f"{code}：{type(exc).__name__}")
            continue
        except Exception as exc:  # 单个标的失败不应中断整池采集
            failures.append(f"{code}：{type(exc).__name__}")
            LOGGER.warning("分钟线获取失败 code=%s detail=%s", code, exc)
            continue
        try:
            if persist_bars:
                store_intraday_bars(store, code, normalize_minutes(minutes))
            factors = intraday_day_factors(minutes)
        except (ValueError, KeyError, TypeError) as exc:
            failures.append(f"{code}：{type(exc).__name__}")
            continue
        if factors.empty:
            failures.append(f"{code}：当日分钟线不足，未产生日内因子")
            continue
        factors["code"] = code
        factors["intraday_source"] = source
        frames.append(factors)
    columns = ["date", "code", *INTRADAY_FACTOR_COLUMNS, "intraday_buckets", "intraday_bars", "intraday_source"]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    report = {
        "codes_requested": len(codes),
        "codes_computed": len(frames),
        "rows": int(len(combined)),
        "failures": failures[:20],
        "failed_codes": len(failures),
        "status": (
            f"日内因子已计算 {len(frames)} 只股票的分钟线特征" if frames else "未取得可用分钟线，未产生日内因子"
        ),
        "limitation": (
            "免费分钟线接口单次仅返回最近约 5 个交易日（1 分钟）或约 30 个交易日（5 分钟），"
            "本地按日累积落盘以逐步加长历史深度；代理指标不等同于 L2 逐笔委托流。"
        ),
    }
    return combined, report


def attach_intraday_factors(
    panel: pd.DataFrame, factors: pd.DataFrame | None, progress: Any = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """把日内因子按 (交易日, 代码) 并入日截面面板，缺失留空不用池均值填充。"""
    if panel is None or panel.empty or "code" not in panel:
        return panel, {"status": "面板为空，未并入日内因子"}
    result = panel.copy()
    for column in INTRADAY_FACTOR_COLUMNS:
        if column not in result:
            result[column] = np.nan
    if factors is None or factors.empty:
        return result, {"status": "没有可并入的日内因子", "coverage": 0.0, "columns": INTRADAY_FACTOR_COLUMNS}
    reference = factors.copy()
    reference["code"] = reference["code"].astype(str).str.zfill(6)
    reference["date"] = pd.to_datetime(reference["date"])
    reference = reference.drop_duplicates(["date", "code"], keep="last").set_index(["date", "code"])
    keys = pd.MultiIndex.from_arrays(
        [pd.to_datetime(result["date"]), result["code"].astype(str).str.zfill(6)]
    )
    aligned = reference.reindex(keys)
    for column in INTRADAY_FACTOR_COLUMNS:
        result[column] = aligned[column].to_numpy()
    coverage = float(result[INTRADAY_FACTOR_COLUMNS].notna().all(axis=1).mean()) if len(result) else 0.0
    return result, {
        "rows_with_factors": int(result[INTRADAY_FACTOR_COLUMNS].notna().all(axis=1).sum()),
        "coverage": coverage,
        "columns": INTRADAY_FACTOR_COLUMNS,
        "status": f"日内因子面板覆盖率 {coverage:.0%}",
    }
