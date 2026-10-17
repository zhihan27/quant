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

# 跨进程计算互斥：同一数据库同时只允许一个重计算任务（扫描、组合回测、滚动重训）。
# 网页端与每日定时扫描进程必须共用这一个编号——两者写的是同一份
# models/market_context_v{MODEL_VERSION}.joblib 与 data/ml_panel_v{MODEL_VERSION}.parquet，
# 各自另起一个编号等于放弃互斥，两边会同时训练并互相覆盖，
# 而 load_context(include_panel=True) 靠 context_id 比对，读到的模型与面板会来自不同批次。
# 建表锁 706821024 与盘中监控锁 706821025 分别见 services/database.py、services/intraday_monitor.py。
COMPUTE_LOCK_KEY = 706821026

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

# 每日收盘后的定时全市场扫描。触发时刻由部署层决定（deploy/quant-scan.timer 或
# Windows 计划任务），这里只管扫描参数，口径与管理员手动全市场扫描一致。
# 组合回测不影响模型且是最慢的一段，逐股回测在全市场下不可接受，故定时任务都不开。
DAILY_SCAN = {
    "pool_mode": "沪深京全部 A 股",
    "objective_mode": DEFAULT_OBJECTIVE_MODE,
    "training_window": TRAINING_WINDOW_DAYS,
    "enable_neutralization": True,
    "use_extra_factors": True,
    "run_portfolio": False,
    "include_backtest": False,
}
# 落盘保留的展示行数：只存页面要渲染的表格，不存全池逐股分析结果。
DAILY_SCAN_TOP_ROWS = 100
DAILY_SCAN_LOG_MAX_BYTES = 5_000_000

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

# ---------------------------------------------------------------------------
# 模型治理：最终留出集、通过率监控、漂移监控、告警与降级
# ---------------------------------------------------------------------------

# 最终留出集：一个从未参与任何调参、且只允许评估一次的数据段。
#
# 注意这个名字与需求文一致，但它**只控制 evaluate_final_holdout 那一次一次性评估**，
# 不改变日常生产训练。需求文 §二-A-2 要求「任何训练都不能使用留出集日期范围内的数据，
# 滚动窗口结束日严格早于 FINAL_HOLDOUT_START」；本系统有意不那样做：
# 生产模型必须用全量历史滚动重训才能给 2026 年的截面打分，把训练窗口永久冻结在
# 2024-01-01 之前，日常评分就失去了意义。因此留出集的口径是「评估专用」——
# 冻结模型只用留出集之前的数据训练，日常生产模型不引用留出集。这是对需求文的有意偏离，
# 详见 services/model_governance.py 顶部说明与 test_records 记录。
ENABLE_FINAL_HOLDOUT: bool = True
FINAL_HOLDOUT_START: str = "2024-01-01"
FINAL_HOLDOUT_END: str | None = None  # None 表示到最新可用截面
# 留出集评估报告文件名（写在 settings.MODEL_DIR 下）。
FINAL_HOLDOUT_REPORT_NAME = "final_holdout_report.json"

# 通过率监控。每轮训练每个模型每个周期各写一行，一次全量扫描约产生 6 行。
PASS_RATE_WINDOW_DAYS = 30
PASS_RATE_MINIMUM = 0.60
# 连续多少次不通过触发告警。按「同一次扫描里的同一个模型/周期」计连续，不跨周期混算。
CONSECUTIVE_FAILURE_ALERT = 3
GATE_FAILURE_RATE_ALERT = 0.50
# 比例类告警的最小样本量：月内不足这么多条记录时不判通过率与门槛失败率，
# 否则每次扫描只有 6 行，上线第一天就会刷屏。连续失败告警不看这个下限（它自带 3 次的要求）。
PASS_RATE_MINIMUM_RECORDS = 20

# 漂移监控阈值。
DRIFT_PSI_WATCH = 0.10
DRIFT_PSI_SEVERE = 0.25
DRIFT_IC_NEGATIVE_DAYS = 5
DRIFT_HHI_PERCENTILE = 0.95
# 概念漂移的滚动窗口长度（可用样本外交易日数）。
DRIFT_CONCEPT_WINDOW_DAYS = 60
# 特征漂移的当前分布取最近 N 个交易日池化，而不是最新一个截面日：
# 本地股票池只有几十只，单日截面在 10 个分箱下绝大多数箱计数不足 5，
# PSI 会剧烈抖动、天天误报严重。
DRIFT_CURRENT_WINDOW_DAYS = 20
DRIFT_PSI_BINS = 10
# 参与漂移计算的最小有效样本数，低于该值判定为「样本不足」并如实标注，不给数。
DRIFT_MINIMUM_SAMPLES = 200
# 预测漂移里历史 HHI 95 分位基准所需的最少历史交易日（约一个月）。低于它时 percentile_cont
# 退化成「历史最大值附近」，每天创出新高都会判关注；此时该维判「样本不足」而不是照常给阈值。
DRIFT_HHI_MINIMUM_SAMPLES = 20

# 漂移监控维度 → 面板列名。
# 值为 None 表示该维度当前无法计算，drift_monitor 必须写一条带原因的「未接入」记录，
# 而不是跳过它——遵守「缺失数据要明确标注、不能静默」。
# 值为哨兵 __EXTRA_FACTOR_COVERAGE__ 表示该维不是单列分布，而是筹码/日内/财报三族因子的
# 覆盖率，由 services/extra_factors.py::summarize_extra_factors 计算。
DRIFT_EXTRA_COVERAGE_SENTINEL = "__EXTRA_FACTOR_COVERAGE__"
DRIFT_MONITORED_FEATURES = {
    "估值": None,  # 行情源未提供 PE/PB/PS，见 drift_monitor 的未接入说明
    "财务因子覆盖率": DRIFT_EXTRA_COVERAGE_SENTINEL,
    "波动率": "volatility_20",
    "换手率": "turnover_20",
    "Amihud": "amihud_20",
    "市值分布": "log_market_cap",
    "行业分布": "industry",
}
# 行业不是数值列：特征漂移维度按类别占比算 PSI，不做 KS 与均值偏移（见 drift_monitor._categorical_row）；
# Top-N 组合的行业集中度（HHI）是另一回事，在预测漂移维度单独计算。
DRIFT_CATEGORICAL_FEATURES = ("industry",)
# 预测漂移里的 Top-N 组合规模。与 PORTFOLIO_TOP_N 同口径。
DRIFT_PREDICTION_TOP_N = PORTFOLIO_TOP_N

# 模型归档与降级。归档保留最近 N 个「至少一个模型通过验证」的版本，
# 回滚只在这批里选，缺归档文件的版本一律拒绝回滚。
MODEL_ARCHIVE_LIMIT = 5
MODEL_ARCHIVE_DIRNAME = "archive"
# 降级时替代模型评分的那句话。放在这里而不是 model_governance.py，
# 是因为 services/stock_scoring.py 也要用它，而它被 backtest 反向依赖，
# 从 model_governance 取会形成循环导入。
DEGRADED_SCORING_MESSAGE = "模型降级中，暂停评分"

# 告警通道。写库与写日志是必达通道，Webhook 与邮件默认关闭，打开后地址与凭据
# 只从环境变量读，不进本文件（沿用 QUANT_PG_* 的先例，敏感信息不硬编码）。
ALERT_ENABLED = True
ALERT_WEBHOOK_ENABLED = False
ALERT_EMAIL_ENABLED = False
ALERT_WEBHOOK_TIMEOUT_SECONDS = 5
ALERT_SMTP_TIMEOUT_SECONDS = 10
ALERT_EMAIL_RECIPIENTS: tuple[str, ...] = ()
# 同一标题的告警在该小时数内只写一条：每日扫描都会重新判定一次，
# 不做去重的话「连续 3 次未通过」会天天各写一条，告警列表很快就被同一条消息淹掉。
ALERT_DEDUP_HOURS = 20

# 告警通道的环境变量名。写在这里而不是散落在 alerting.py 的字符串里，
# 便于部署时一眼看到需要配什么；对应的值一律不进仓库。
ALERT_WEBHOOK_ENV = "QUANT_ALERT_WEBHOOK_URL"
ALERT_SMTP_HOST_ENV = "QUANT_ALERT_SMTP_HOST"
ALERT_SMTP_PORT_ENV = "QUANT_ALERT_SMTP_PORT"
ALERT_SMTP_USER_ENV = "QUANT_ALERT_SMTP_USER"
ALERT_SMTP_PASSWORD_ENV = "QUANT_ALERT_SMTP_PASSWORD"
ALERT_SMTP_SENDER_ENV = "QUANT_ALERT_SMTP_SENDER"

