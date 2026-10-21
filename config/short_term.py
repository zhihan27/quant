"""短线研究策略的独立配置，不修改旧模型周期。"""

from __future__ import annotations

from config import settings

SHORT_TERM_STRATEGY_ID = "cn_a_short_daily_v1"
SHORT_TERM_STRATEGY_VERSION = "1.0.0"
SHORT_TERM_MODEL_DIR = settings.MODEL_DIR / "short_term"
SHORT_TERM_SNAPSHOT_DIR = settings.DATA_DIR / "short_term" / "snapshots"
SHORT_TERM_ACCEPTANCE_POLICY_PATH = settings.DATA_DIR / "short_term" / "acceptance_policy.json"
SHORT_TERM_HORIZONS = (1, 3, 5)
SHORT_TERM_MAIN_HORIZON = 3
SHORT_TERM_MIN_HISTORY_BARS = 320
SHORT_TERM_TRAIN_WINDOW_DAYS = 504
SHORT_TERM_TEST_WINDOW_DAYS = 63
SHORT_TERM_TOP_K = 20
SHORT_TERM_FEATURE_VERSION = "daily-price-v1"

# 这些特征只依赖 signal_time 当时及此前的已完成日线。
SHORT_TERM_FEATURES = (
    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    "overnight_return",
    "intraday_return",
    "close_sma5_gap",
    "close_sma10_gap",
    "close_sma20_gap",
    "volatility_5d",
    "volatility_20d",
    "downside_volatility_20d",
    "atr14_pct",
    "range_location_20d",
    "volume_ratio_5_20",
    "amount_ratio_5_20",
    "amihud_20d",
)

# 沿用项目现有比例成本作为研究代理；实际账户费率和最低佣金仍需另行校准。
SHORT_TERM_RESEARCH_COSTS = dict(settings.DEFAULT_COSTS)
