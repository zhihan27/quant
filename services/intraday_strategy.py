"""基于已完成分钟线的卡尔曼和协整研究提醒，独立于日频选股模型。"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from services.intraday_data import quote_is_fresh

SHANGHAI = ZoneInfo("Asia/Shanghai")
DEFAULT_MONITOR_CONFIG = {
    "poll_seconds": 5, "minute_seconds": 60, "entry_z": 2.5, "exit_z": 0.5,
    "pair_a": "", "pair_b": "", "borrow_confirmed": False,
}


def validate_monitor_config(values: dict[str, Any]) -> dict[str, Any]:
    """校验持久化监控参数，避免零间隔死循环和无效配对。"""
    from services.database import stock_code

    result = {**DEFAULT_MONITOR_CONFIG, **values}
    result["poll_seconds"] = int(result["poll_seconds"])
    result["minute_seconds"] = int(result["minute_seconds"])
    result["entry_z"] = float(result["entry_z"])
    result["exit_z"] = float(result["exit_z"])
    if not 3 <= result["poll_seconds"] <= 60 or not 30 <= result["minute_seconds"] <= 300:
        raise ValueError("快照轮询须为3至60秒，分钟线同步须为30至300秒。")
    if not np.isfinite([result["entry_z"], result["exit_z"]]).all() or not 1.5 <= result["entry_z"] <= 5 or not 0 <= result["exit_z"] < result["entry_z"]:
        raise ValueError("入场阈值须为1.5至5，回归阈值须小于入场阈值。")
    result["pair_a"], result["pair_b"] = str(result["pair_a"]).strip(), str(result["pair_b"]).strip()
    if result["pair_a"] or result["pair_b"]:
        result["pair_a"], result["pair_b"] = stock_code(result["pair_a"]), stock_code(result["pair_b"])
        if result["pair_a"] == result["pair_b"]:
            raise ValueError("配对股票不能为同一只。")
    result["borrow_confirmed"] = bool(result["borrow_confirmed"])
    return result


def _signal(code: str, frame: pd.DataFrame, strategy: str, reason: str) -> dict[str, Any]:
    return {
        "code": code, "strategy": strategy, "bar_time": pd.Timestamp(frame.iloc[-1]["date"]).isoformat() if not frame.empty else None,
        "active": False, "action": "等待", "reason": reason, "reference_price": None,
    }


def _recent(frame: pd.DataFrame, now: datetime) -> bool:
    if frame.empty:
        return False
    last = pd.Timestamp(frame.iloc[-1]["date"])
    return last.date() == now.date() and 0 <= (pd.Timestamp(now) - last).total_seconds() <= 180


def _known_minutes(frame: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """策略入口再次清除当前及未来分钟，数据库回放也遵守相同时间边界。"""
    if frame.empty:
        return frame
    dates = pd.to_datetime(frame["date"])
    return frame.loc[dates < pd.Timestamp(now).floor("min")].sort_values("date").drop_duplicates("date", keep="last").copy()


def analyze_intraday_stock(
    code: str, frame: pd.DataFrame, quote: dict[str, Any], position: dict[str, Any] | None = None,
    configuration: dict[str, Any] | None = None, now: datetime | None = None,
) -> dict[str, Any]:
    """利用一分钟创新残差及持仓成本生成观察或退出提醒，不假设即时成交。"""
    now = now or datetime.now(SHANGHAI)
    configuration = validate_monitor_config(configuration or {})
    frame = _known_minutes(frame, now)
    result = _signal(code, frame, "kalman_minute", "需要当日至少35根已完成分钟线。")
    if frame.empty:
        return result
    today = frame.loc[pd.to_datetime(frame["date"]).dt.date == now.date()].copy()
    if len(today) < 35:
        return result
    closes = today["close"].to_numpy(dtype=float)
    differences = np.diff(closes[:30])
    noise = max(float(np.var(differences, ddof=1)), float(closes[29] * 0.0002) ** 2)
    # 噪声参数仅取当日最初30根，之后递推时先预测再更新，不用当前价回填先验。
    transition = np.array([[1.0, 1.0], [0.0, 1.0]])
    state = np.array([closes[29], float(np.mean(differences[-5:]))])
    covariance = np.eye(2) * noise
    process = np.diag([noise * 0.05, noise * 0.005])
    innovation = prior_price = variance = 0.0
    for price in closes[30:]:
        prior = transition @ state
        prior_covariance = transition @ covariance @ transition.T + process
        prior_price = float(prior[0])
        innovation = float(price - prior[0])
        variance = float(prior_covariance[0, 0] + noise)
        gain = prior_covariance[:, 0] / variance
        state = prior + gain * innovation
        covariance = (np.eye(2) - np.outer(gain, [1.0, 0.0])) @ prior_covariance
    z_score = innovation / np.sqrt(variance)
    previous_close = today["close"].shift(1)
    true_range = pd.concat([
        today["high"] - today["low"], (today["high"] - previous_close).abs(), (today["low"] - previous_close).abs(),
    ], axis=1).max(axis=1)
    atr = float(true_range.tail(14).mean())
    price = float(quote["price"]) if quote.get("price") is not None else float(closes[-1])
    holding = int((position or {}).get("shares", 0))
    sellable = int((position or {}).get("available_shares", 0))
    cost = float((position or {}).get("average_cost", price))
    result.update({
        "reference_price": price, "z_score": float(z_score), "kalman_fair": prior_price, "atr": atr,
        "vwap": float(np.average(closes, weights=today["volume"].to_numpy(dtype=float))),
        "buy_watch_below": round(prior_price - configuration["entry_z"] * np.sqrt(variance), 3),
        "initial_stop": round(max(0.001, (cost if holding else price) - 2 * atr), 3),
        "take_profit": round((cost if holding else price) + 3 * atr, 3),
        "available_shares": sellable, "quote_time": quote.get("quote_time"),
        "reason": "未达到卡尔曼残差或持仓退出条件。",
    })
    if not quote_is_fresh(quote, now) or not _recent(today, now):
        result["reason"] = "非连续竞价时段、行情源时间过期或分钟线未更新；提醒已失效。"
        return result
    gaps = pd.to_datetime(today["date"]).diff().dt.total_seconds().tail(20).dropna()
    # 午休的91分钟间隔合法，其余最近缺失分钟会停止发出新的有效提醒。
    if ((gaps > 90) & (gaps != 5460)).any():
        result["reason"] = "最近分钟线不连续，暂停有效提醒。"
        return result
    if holding and (price <= result["initial_stop"] or price >= result["take_profit"] or z_score >= configuration["entry_z"]):
        result["action"] = "卖出提醒" if sellable else "卖出条件触发但不可卖"
        result["active"] = sellable > 0
        result["reason"] = "持仓触及ATR退出参考或残差过热。" if sellable else "触及退出参考，但T+1或手工可卖数量限制阻止卖出。"
    elif not holding and z_score <= -configuration["entry_z"]:
        volume_ready = float(today.iloc[-1]["volume"]) >= 0.8 * float(today.iloc[-21:-1]["volume"].mean())
        recovery = closes[-1] > closes[-2]
        cost_ready = (prior_price - price) / price > 0.002
        if volume_ready and recovery and cost_ready:
            result.update(active=True, action="买入观察", reason="分钟残差超跌、收盘回升且量能达标；预期回归幅度超过0.2%的研究成本门槛。")
        else:
            result["reason"] = "残差超跌，等待价格回升、量能及成本门槛同时满足。"
    return result


def analyze_intraday_pair(
    code_a: str, code_b: str, first: pd.DataFrame, second: pd.DataFrame,
    quotes: dict[str, dict[str, Any]], configuration: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """只用今日之前的分钟线检验协整，今日逐根预测对冲比例并计算先验残差。"""
    now = now or datetime.now(SHANGHAI)
    configuration = validate_monitor_config(configuration or {})
    first, second = _known_minutes(first, now), _known_minutes(second, now)
    result = _signal(code_a, first, "cointegration_minute", "配对历史不足。")
    result["peer_code"] = code_b
    if first.empty or second.empty:
        return result
    merged = first[["date", "close"]].merge(second[["date", "close"]], on="date", suffixes=("_a", "_b")).sort_values("date")
    if merged.empty:
        return result
    result["bar_time"] = pd.Timestamp(merged.iloc[-1]["date"]).isoformat()
    dates = pd.to_datetime(merged["date"]).dt.date
    training = merged.loc[dates < now.date()].tail(960)
    today = merged.loc[dates == now.date()]
    if len(training) < 240 or len(today) < 35:
        result["reason"] = "需要至少240根历史训练分钟线和35根今日已完成分钟线。"
        return result
    a = training["close_a"].to_numpy(dtype=float)
    b = training["close_b"].to_numpy(dtype=float)
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        result["reason"] = "训练价格缺乏变化，无法检验协整。"
        return result
    design = np.column_stack([b, np.ones(len(b))])
    beta = np.linalg.lstsq(design, a, rcond=None)[0]
    middle = len(a) // 2
    try:
        pvalues = [float(coint(a, b)[1]), float(coint(a[:middle], b[:middle])[1]), float(coint(a[middle:], b[middle:])[1])]
        sub_betas = [float(np.polyfit(b[:middle], a[:middle], 1)[0]), float(np.polyfit(b[middle:], a[middle:], 1)[0])]
    except (ValueError, np.linalg.LinAlgError):
        result["reason"] = "协整检验失败，未生成有效配对提醒。"
        return result
    stable = np.isfinite(pvalues + sub_betas).all() and max(pvalues) < 0.05 and 0.05 < beta[0] < 20 and abs(sub_betas[0] - sub_betas[1]) / abs(beta[0]) < 0.3
    result.update(training_pvalues=pvalues, training_beta=float(beta[0]), training_bars=len(training))
    if not stable:
        result["reason"] = "此前分钟线未通过协整与对冲比例稳定性检验。"
        return result
    residuals = list(a - design @ beta)
    noise = max(float(np.var(residuals, ddof=1)), (float(a[-1]) * 0.0002) ** 2)
    covariance = np.linalg.pinv(design.T @ design) * noise
    process = np.diag([1e-8, noise * 1e-4])
    z_score = innovation = 0.0
    prior_beta = beta.copy()
    for row in today.itertuples():
        observation = np.array([row.close_b, 1.0])
        prior_beta = beta.copy()
        predicted_covariance = covariance + process
        innovation = float(row.close_a - observation @ prior_beta)
        center = float(np.mean(residuals[-60:]))
        scale = max(float(np.std(residuals[-60:], ddof=1)), np.sqrt(noise) * 0.2)
        z_score = (innovation - center) / scale
        gain = predicted_covariance @ observation / float(observation @ predicted_covariance @ observation + noise)
        beta = prior_beta + gain * innovation
        covariance = (np.eye(2) - np.outer(gain, observation)) @ predicted_covariance
        residuals.append(innovation)
    last = today.iloc[-1]
    result.update(z_score=float(z_score), hedge_ratio=float(prior_beta[0]), reference_price=float(last["close_a"]))
    if not all(quote_is_fresh(quotes.get(code, {}), now) for code in (code_a, code_b)) or not _recent(today, now):
        result["reason"] = "配对行情过期或不同步，提醒已失效。"
        return result
    gaps = pd.to_datetime(today["date"]).diff().dt.total_seconds().tail(20).dropna()
    if ((gaps > 90) & (gaps != 5460)).any():
        result["reason"] = "最近配对分钟线不连续，暂停有效提醒。"
        return result
    # 原始分钟线发生大幅隔夜跳变时无法区分公司行为，先停止提醒。
    if max(abs(float(today.iloc[0][f"close_{leg}"]) / float(training.iloc[-1][f"close_{leg}"]) - 1) for leg in ("a", "b")) > 0.08:
        result["reason"] = "配对存在大幅隔夜跳变，需排查除权及结构变化。"
        return result
    if abs(z_score) >= configuration["entry_z"]:
        result["action"] = "多A空B观察" if z_score < 0 else "空A多B观察"
        result["reason"] = "训练期协整稳定，分钟先验残差偏离。"
        gross = float(last["close_a"]) + abs(float(prior_beta[0])) * float(last["close_b"])
        if abs(innovation) / gross < 0.002:
            result["reason"] = "回归幅度不足以覆盖0.2%的研究成本门槛。"
        elif not configuration["borrow_confirmed"]:
            result["reason"] = "价差触发研究条件，但未确认借券可得性，仅供观察。"
        else:
            result["active"] = True
            result["reason"] += "已手工确认借券条件；仍需按实际可卖持仓和券商规则复核。"
    elif abs(z_score) <= configuration["exit_z"]:
        result.update(action="价差回归提醒", active=False, reason="价差进入回归区间，请核对实际配对头寸。")
    else:
        result["reason"] = "配对通过训练检查，当前偏离未达到入场阈值。"
    return result
