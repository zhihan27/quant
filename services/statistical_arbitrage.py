"""配对筛选、协整稳定性、卡尔曼对冲系数与价差实验。"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _kalman_hedge_ratio(log_a: np.ndarray, log_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """以二维线性状态空间递推估计 beta 和截距。"""
    state = np.zeros(2, dtype=float)
    covariance = np.eye(2, dtype=float)
    transition_noise = np.eye(2, dtype=float) * 1e-5
    observation_noise = 1e-3
    betas = np.zeros(len(log_a), dtype=float)
    intercepts = np.zeros(len(log_a), dtype=float)
    for index, (dependent, independent) in enumerate(zip(log_a, log_b)):
        design = np.asarray([independent, 1.0])
        covariance = covariance + transition_noise
        error = dependent - float(design @ state)
        variance = float(design @ covariance @ design + observation_noise)
        gain = covariance @ design / variance
        state = state + gain * error
        covariance = covariance - np.outer(gain, design) @ covariance
        betas[index] = state[0]
        intercepts[index] = state[1]
    return betas, intercepts


def _simple_metrics(returns: pd.Series, trades: int) -> dict[str, Any]:
    values = returns.dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return {"observations": 0, "total_return": None, "sharpe_ratio": None, "maximum_drawdown": None, "trades": trades}
    equity = np.cumprod(1 + values)
    peak = np.maximum.accumulate(np.concatenate(([1.0], equity)))[1:]
    volatility = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return {
        "observations": int(len(values)),
        "total_return": float(equity[-1] - 1),
        "sharpe_ratio": float(np.mean(values) / volatility * math.sqrt(252)) if volatility > 1e-12 else None,
        "maximum_drawdown": float(np.min(equity / peak - 1)),
        "trades": int(trades),
    }


def _pair_weights(position: float, beta: float) -> tuple[float, float]:
    """按双腿总名义敞口为1分配权重，正仓位表示做多A、做空B。"""
    if not np.isfinite(beta) or beta <= 0:
        return 0.0, 0.0
    return position / (1 + beta), -position * beta / (1 + beta)


def analyze_pair(
    history_a: pd.DataFrame,
    history_b: pd.DataFrame,
    code_a: str,
    code_b: str,
    commission_rate: float = 0.0003,
    slippage_rate: float = 0.001,
    stamp_duty_rate: float = 0.0005,
    borrow_rate_annual: float = 0.0,
) -> dict[str, Any]:
    """使用训练期筛选配对，按延后执行的对冲仓位计算测试期表现并返回最新研究信号。"""
    try:
        from statsmodels.tsa.stattools import coint
    except ImportError:
        return {"status": "缺少statsmodels，不生成信号"}
    costs = [commission_rate, slippage_rate, stamp_duty_rate, borrow_rate_annual]
    if not all(np.isfinite(value) and 0 <= value < 1 for value in costs):
        raise ValueError("费用必须为有效的非负比例且小于1。")
    if code_a == code_b:
        return {"status": "配对股票不能相同；不生成信号"}
    frames = []
    for history, name in ((history_a, "a"), (history_b, "b")):
        frame = history[["date", "close"]].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        if frame["date"].duplicated().any():
            raise ValueError("配对历史包含重复日期。")
        frames.append(frame.rename(columns={"close": f"close_{name}"}))
    pair = frames[0].merge(frames[1], on="date", how="inner").dropna().sort_values("date").reset_index(drop=True)
    pair = pair.loc[
        np.isfinite(pair["close_a"]) & np.isfinite(pair["close_b"])
        & pair["close_a"].gt(0) & pair["close_b"].gt(0)
    ].reset_index(drop=True)
    if len(pair) < 180:
        return {"status": f"样本不足：需要至少180个有效共同交易日，当前{len(pair)}；不生成信号"}
    log_a = np.log(pair["close_a"].to_numpy(dtype=float))
    log_b = np.log(pair["close_b"].to_numpy(dtype=float))
    returns = pair[["close_a", "close_b"]].pct_change(fill_method=None)
    split_index = int(len(pair) * 0.7)
    middle = split_index // 2
    training_correlation = float(returns.iloc[1:split_index]["close_a"].corr(returns.iloc[1:split_index]["close_b"]))
    try:
        p_train = float(coint(log_a[:split_index], log_b[:split_index])[1])
        p_first = float(coint(log_a[:middle], log_b[:middle])[1])
        p_second = float(coint(log_a[middle:split_index], log_b[middle:split_index])[1])
    except (ValueError, np.linalg.LinAlgError):
        return {"status": "训练期协整检验失败；不生成信号"}
    betas, intercepts = _kalman_hedge_ratio(log_a, log_b)
    # 用更新前状态的预测误差计算价差，标准化基准也只读取此前观测。
    prior_beta = np.r_[0.0, betas[:-1]]
    prior_intercept = np.r_[0.0, intercepts[:-1]]
    spread = pd.Series(log_a - prior_beta * log_b - prior_intercept)
    history_spread = spread.shift(1)
    mean = history_spread.rolling(60, min_periods=40).mean()
    deviation = history_spread.rolling(60, min_periods=40).std(ddof=0)
    z_score = (spread - mean) / deviation.replace(0, np.nan)
    training_z = z_score.iloc[60:split_index].dropna()
    if len(training_z) < 40:
        return {"status": "训练价差样本不足；不生成信号"}
    entry = float(training_z.abs().quantile(0.95))
    beta_history = betas[60:split_index]
    beta_mean = float(beta_history.mean())
    beta_cv = float(beta_history.std() / max(abs(beta_mean), 1e-9))
    stable = bool(
        training_correlation >= 0.70 and p_train < 0.05
        and p_first < 0.05 and p_second < 0.05
        and beta_mean > 0 and beta_cv <= 0.50 and 1.5 <= entry <= 3.5
    )
    position = 0.0
    holding = 0
    previous_weights = np.zeros(2)
    strategy_returns = []
    path = []
    trade_count = 0
    stop_threshold = max(4.0, entry + 1)
    for index in range(split_index, len(pair)):
        # 收盘生成信号，下一日收盘执行，再下一日才计持仓收益，避免成交在已知收盘价上。
        decision = index - 2
        prior_position = position
        z = z_score.iloc[decision]
        if not stable or pd.isna(z) or not np.isfinite(betas[decision]) or betas[decision] <= 0:
            position = 0.0
        elif position and (abs(z) <= 0.5 or abs(z) >= stop_threshold or holding >= 20):
            position = 0.0
        elif position == 0 and entry <= abs(z) < stop_threshold:
            position = -1.0 if z > 0 else 1.0
        holding = holding + 1 if position and position == prior_position else int(position != 0)
        weights = np.asarray(_pair_weights(position, float(betas[decision])))
        changes = weights - previous_weights
        turnover = float(np.abs(changes).sum())
        sell_notional = float((-changes).clip(min=0).sum())
        fee = turnover * (commission_rate + slippage_rate) + sell_notional * stamp_duty_rate
        borrow = float((-weights).clip(min=0).sum()) * borrow_rate_annual / 252
        leg_returns = returns.iloc[index].to_numpy(dtype=float)
        gross = float(weights @ leg_returns)
        net = (1 - fee - borrow) * (1 + gross) - 1
        strategy_returns.append(net)
        if position != prior_position:
            trade_count += 1
        path.append({
            "date": pair.iloc[index]["date"].strftime("%Y-%m-%d"),
            "signal_date": pair.iloc[decision]["date"].strftime("%Y-%m-%d"),
            "position": position, "weight_a": float(weights[0]), "weight_b": float(weights[1]),
            "gross_return": gross, "net_return": net, "cost": fee + borrow,
        })
        # 持仓权重随价格漂移，下一次再平衡费用按实际权重变化计入。
        previous_weights = weights * (1 + leg_returns) / max(1 + gross, 1e-9)
    validation = _simple_metrics(pd.Series(strategy_returns, dtype=float), trade_count)
    current_p = None
    try:
        current_p = float(coint(log_a[-180:], log_b[-180:])[1])
    except (ValueError, np.linalg.LinAlgError):
        pass
    # 最新稳定性只控制当前信号，绝不改变已经生成的历史回测路径。
    current_stable = stable and current_p is not None and current_p < 0.05 and betas[-1] > 0
    latest_z = float(z_score.iloc[-1]) if pd.notna(z_score.iloc[-1]) else None
    signal = "无套利信号"
    if current_stable and latest_z is not None:
        if entry <= latest_z < stop_threshold:
            signal = f"价差偏高：做空{code_a} / 做多{code_b}"
        elif -stop_threshold < latest_z <= -entry:
            signal = f"价差偏低：做多{code_a} / 做空{code_b}"
        elif abs(latest_z) <= 0.5:
            signal = "价差回归区间：平仓观察"
    return {
        "status": "配对通过当前及训练期筛选；实验性研究信号" if current_stable else "当前配对未通过稳定性筛选，不生成套利信号",
        "codes": [code_a, code_b], "common_observations": len(pair),
        "training_stable": stable, "current_stable": bool(current_stable),
        "training_end": pair.iloc[split_index - 1]["date"].strftime("%Y-%m-%d"),
        "return_correlation": training_correlation,
        "cointegration_p_value": p_train, "first_period_p_value": p_first,
        "second_period_p_value": p_second, "current_period_p_value": current_p,
        "hedge_ratio_kalman": float(betas[-1]), "hedge_ratio_cv": beta_cv,
        "entry_threshold": entry if stable else None, "current_z_score": latest_z,
        "signal": signal, "exit_condition": f"|Z|≤0.5、|Z|≥{stop_threshold:.2f}或持有20日",
        "validation": validation, "backtest_path": path,
        "costs": {"commission_each_side": commission_rate, "slippage_each_side": slippage_rate,
                  "stamp_duty_sell_side": stamp_duty_rate, "borrow_rate_annual": borrow_rate_annual},
        "warning": "日频配对研究；回测双腿总名义敞口为1。融券可得性、真实借券费、分红及盘口执行未完整模拟，不能据此直接实盘。",
    }
