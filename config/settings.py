from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"
DB_PATH = DATA_DIR / "quant.db"
# 列式面板存储根目录：逐股日线、因子面板与暴露数据均以 Parquet 保存。
PANEL_DIR = DATA_DIR / "panel"
PARQUET_COMPRESSION = "zstd"

EASTMONEY_QUOTE_URL = "https://push2.eastmoney.com/api/qt/stock/get"
EASTMONEY_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
EASTMONEY_LIST_URL = "https://82.push2.eastmoney.com/api/qt/clist/get"
SINA_QUOTE_URL = "https://hq.sinajs.cn/list={symbol}"
TENCENT_KLINE_URL = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
REQUEST_TIMEOUT_SECONDS = 10
REQUEST_CONNECT_TIMEOUT_SECONDS = 5
REQUEST_READ_TIMEOUT_SECONDS = 10
REQUEST_RETRIES = 2
RETRY_BACKOFF_SECONDS = 0.5
MAX_CONCURRENT_REQUESTS = 1
LIST_PAGE_SIZE = 100
AKSHARE_SPOT_TIMEOUT_SECONDS = 35
SINA_HISTORY_TIMEOUT_SECONDS = 20
HISTORY_INCREMENTAL_LOOKBACK_DAYS = 7
FULL_HISTORY_REFRESH_SECONDS = 604800
QUOTE_CACHE_SECONDS = 30
UNIVERSE_CACHE_SECONDS = 900
# 仅用于代码和名称检索，过期目录不会被当作实时行情使用。
STOCK_DIRECTORY_STALE_SECONDS = 7 * 24 * 60 * 60
HISTORY_CACHE_SECONDS = 21600
REQUEST_INTERVAL_SECONDS = 0.25
MINIMUM_SCAN_HISTORY_BARS = 320
MINIMUM_EXPECTED_A_SHARE_UNIVERSE = 4000
# 交易所号段。2026 年创业板启用 302 号段，新股发行还会继续扩充号段，
# 因此股票池过滤、缓存完整性校验与涨跌停制度判定统一引用这一份定义。
# 只改其中一处会让新号段的股票从股票池里凭空消失，或在涨跌停判定里按错误比例计算。
BOARD_PREFIXES = {
    "main": ("00", "60"),
    "chinext": ("300", "301", "302"),
    "star": ("688", "689"),
    "bj": ("43", "83", "87", "92"),
}
# 涨跌停制度只看交易所大类：北交所涵盖 43/83/87/92 起首的 4x/8x 全部号段，
# 比股票池过滤用的具体号段更宽，故不并入 BOARD_PREFIXES。
BSE_LIMIT_PREFIXES = ("4", "8", "92")
MARKET_INDEX_CODE = "000300"
MODEL_VERSION = "2.1.0"

DEFAULT_COSTS = {
    "commission_rate": 0.0003,
    "stamp_duty_rate": 0.0005,
    "slippage_rate": 0.001,
}

# 训练范式：排序目标（LambdaMART/NDCG）替代回归目标，消除“全线看多”的正向偏置。
RANK_OBJECTIVE_MODES = ("lambdarank", "regression", "auto")
DEFAULT_OBJECTIVE_MODE = "lambdarank"
RANK_GRADE_COUNT = 5
# 滚动窗口：每次重训只用最近 N 个交易日，自适应风格切换；0 表示不限制。
TRAINING_WINDOW_DAYS = 750
# 前向滚动验证的最短训练期，低于该值不产生该折模型。
WALK_FORWARD_RETRAIN_DAYS = 20
# NDCG 诊断的截断比例（前 20% 截面）。
NDCG_TOP_FRACTION = 0.2

# 中性化：每日截面对市值与行业哑变量做多元线性回归，取残差作为纯净 Alpha。
NEUTRALIZATION_MINIMUM_STOCKS = 20
NEUTRALIZATION_MINIMUM_INDUSTRY_STOCKS = 2
INDUSTRY_CACHE_SECONDS = 604800
EXPOSURE_CACHE_SECONDS = 900
# 申万一级行业接口按行业逐个读取，子进程整体限时；超时后使用已落盘的部分结果。
SW_INDUSTRY_TOTAL_TIMEOUT_SECONDS = 240
SW_INDUSTRY_MINIMUM_INDUSTRIES = 25

# 行情清洗：长期停牌剔除与一字板流动性锁定。
LONG_SUSPENSION_DAYS = 20
# 前复权序列出现超出涨跌幅上限的价格跳空时按异常处理，容忍度用于覆盖不同板块制度。
ADJUSTMENT_GAP_TOLERANCE = 1.2

# 组合构建：截面前 N 名进入组合，单股仓位上限限制资金集中度。
PORTFOLIO_TOP_N = 20
MAX_POSITION_WEIGHT = 0.05
PORTFOLIO_BACKTEST_MAX_STOCKS = 300

# 股票池状态：退市与 ST 名单来自交易所公开接口，缓存一天，避免每次扫描重复拉取。
UNIVERSE_REFERENCE_TIMEOUT_SECONDS = 180
UNIVERSE_STATUS_CACHE_SECONDS = 86400

# 筹码分布：按价格分箱堆积，换手衰减系数控制历史筹码的遗忘速度。
CHIP_PRICE_BINS = 120
CHIP_DECAY_FACTOR = 1.0
# 成本区间宽度取 90% 筹码覆盖范围；获利盘比例按当日收盘价在成本分布中的位置计算。
CHIP_COVERAGE_RATIO = 0.90
# 筹码分布至少需要的历史交易日；不足时不出因子，不用短样本冒充稳态筹码。
CHIP_MINIMUM_BARS = 120

# 日内高频因子落盘：分钟线按 code=*.parquet 累积，随时间增长历史深度。
INTRADAY_PANEL_DIRNAME = "intraday"
# 东财分钟接口的单次可得窗口有限，累积超过该交易日数后只保留最近部分，控制磁盘占用。
INTRADAY_PANEL_MAX_DAYS = 250
INTRADAY_MINIMUM_BARS_PER_DAY = 30

# PIT 财务因子：财报归属期与公告日的对齐容忍度。
FUNDAMENTAL_REFERENCE_TIMEOUT_SECONDS = 240
FUNDAMENTAL_CACHE_SECONDS = 604800
# 单季同比至少需要的历史报告期数；TTM 需要连续四个季度。
FUNDAMENTAL_MINIMUM_QUARTERS = 5
# Altman Z-Score 财务困境预警：低于该阈值的标的按需求降权/否决。
# 采用 Z''（1993，非制造业与新兴市场）变体的判定阈值；该变体的 X4 以账面权益计量，
# 与 PIT 财务面板不掺入价格的口径自洽，因此不能沿用原始 Z（1968）的 1.81 / 2.99。
ALTMAN_Z_DISTRESS_THRESHOLD = 1.10
ALTMAN_Z_SAFE_THRESHOLD = 2.60
# 现金流质量：经营性现金流净额占净利润比例低于该值视为现金流恶化。
MINIMUM_CASHFLOW_TO_PROFIT = 0.0

# 新增因子并入排序特征矩阵的最低覆盖率；沿用中性化因子的同类门控口径。
EXTRA_FACTOR_MINIMUM_COVERAGE = 0.5

