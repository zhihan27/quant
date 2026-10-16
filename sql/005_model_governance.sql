-- 模型治理：最终留出集评估、通过率监控、特征/概念/预测漂移、模型归档与告警。
--
-- 表名沿用仓库既有的 quant_ 前缀。需求文写作 model_validation_log 等，
-- 这里一一对应为 quant_model_validation_log、quant_feature_drift、
-- quant_concept_drift、quant_prediction_drift、quant_model_final_holdout，
-- 只为与 quant_watchlist / quant_positions 等同族命名保持一致，字段名与需求文一致。
--
-- 注意：迁移执行方式是 tests/test_auth.py 里朴素的字符串替换（把 public. 换成随机模式名），
-- 因此本文件里不得出现字符串字面量形式的 public.

-- 通过率监控：每次滚动重训 / 每日扫描 / 管理员手动扫描后的实际检验数值。
-- 与 quant_saved_backtests 一样按追加写入，不覆盖历史，才能看出趋势。
CREATE TABLE IF NOT EXISTS public.quant_model_validation_log (
    id             BIGSERIAL PRIMARY KEY,
    model_version  TEXT        NOT NULL,
    model_type     TEXT        NOT NULL,
    target_type    TEXT        NOT NULL,
    scan_time      TIMESTAMPTZ NOT NULL DEFAULT now(),
    window_start   DATE,
    window_end     DATE,
    rank_ic        DOUBLE PRECISION,
    hac_t          DOUBLE PRECISION,
    ndcg_ratio     DOUBLE PRECISION,
    top_excess     DOUBLE PRECISION,
    top_bottom     DOUBLE PRECISION,
    mae            DOUBLE PRECISION,
    training_dates INTEGER,
    passed_gates   JSONB       NOT NULL DEFAULT '{}'::jsonb,
    failed_gate    TEXT,
    pass           BOOLEAN     NOT NULL,
    details        JSONB       NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS quant_model_validation_log_time_idx
    ON public.quant_model_validation_log(scan_time DESC, id DESC);
CREATE INDEX IF NOT EXISTS quant_model_validation_log_model_idx
    ON public.quant_model_validation_log(model_version, model_type, target_type);
CREATE INDEX IF NOT EXISTS quant_model_validation_log_pass_idx
    ON public.quant_model_validation_log(pass, scan_time DESC);
-- 连续失败告警按「同一个模型类型 + 同一个目标」分组取最近若干次，这个索引直接服务该查询。
CREATE INDEX IF NOT EXISTS quant_model_validation_log_series_idx
    ON public.quant_model_validation_log(model_type, target_type, scan_time DESC, id DESC);

COMMENT ON TABLE  public.quant_model_validation_log IS '每轮训练的样本外检验明细；通过率与门槛失败率的唯一数据来源';
COMMENT ON COLUMN public.quant_model_validation_log.model_type IS 'lgb 或 xgb';
COMMENT ON COLUMN public.quant_model_validation_log.target_type IS 'rank（排序目标）或 reg（回归目标）';
COMMENT ON COLUMN public.quant_model_validation_log.window_start IS '该次训练实际用到的样本起始交易日，空表示未生成样本外指标';
COMMENT ON COLUMN public.quant_model_validation_log.window_end IS '该次训练实际用到的样本结束交易日';
COMMENT ON COLUMN public.quant_model_validation_log.passed_gates IS '逐道门槛的通过情况，形如 {"dates": true, "rank_ic": false}';
COMMENT ON COLUMN public.quant_model_validation_log.failed_gate IS '第一道未通过的门槛；全部通过时为 NULL';
COMMENT ON COLUMN public.quant_model_validation_log.details IS '完整指标快照与数据血缘，含训练目标、窗口与状态说明';

-- 最终留出集：一个模型版本只允许一次评估。UNIQUE(model_version) 是「只跑一次」的真正闸门，
-- 两个管理员同时点也只能成功一个，不靠应用层判断。
CREATE TABLE IF NOT EXISTS public.quant_model_final_holdout (
    id            BIGSERIAL PRIMARY KEY,
    model_version TEXT        NOT NULL UNIQUE,
    context_id    TEXT,
    holdout_start DATE        NOT NULL,
    holdout_end   DATE        NOT NULL,
    evaluated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    operator      TEXT        NOT NULL DEFAULT '',
    forced        BOOLEAN     NOT NULL DEFAULT FALSE,
    passed        BOOLEAN     NOT NULL,
    metrics       JSONB       NOT NULL DEFAULT '{}'::jsonb,
    report_path   TEXT
);
CREATE INDEX IF NOT EXISTS quant_model_final_holdout_time_idx
    ON public.quant_model_final_holdout(evaluated_at DESC);

COMMENT ON TABLE  public.quant_model_final_holdout IS '一次性最终留出集评估结果；同一 model_version 只能有一行';
COMMENT ON COLUMN public.quant_model_final_holdout.context_id IS '被评估的那一份模型上下文的 ID；MODEL_VERSION 是常量，同版本下每次扫描都会换 context_id';
COMMENT ON COLUMN public.quant_model_final_holdout.forced IS 'TRUE 表示是管理员用 --force 覆盖已有评估记录后写入的';
COMMENT ON COLUMN public.quant_model_final_holdout.metrics IS '留出集的完整指标，含 Rank IC、HAC t、NDCG、Top 超额、Top-Bottom、换手率与费用后收益';

-- 特征漂移。date + feature_name 唯一，重复扫描 UPSERT 不刷重复行。
CREATE TABLE IF NOT EXISTS public.quant_feature_drift (
    id                 BIGSERIAL PRIMARY KEY,
    date               DATE   NOT NULL,
    feature_name       TEXT   NOT NULL,
    psi                DOUBLE PRECISION,
    ks_stat            DOUBLE PRECISION,
    missing_rate       DOUBLE PRECISION,
    mean_shift         DOUBLE PRECISION,
    threshold_exceeded TEXT   NOT NULL DEFAULT '正常',
    window_start       DATE,
    window_end         DATE,
    reference_start    DATE,
    reference_end      DATE,
    sample_size        INTEGER,
    note               TEXT,
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (date, feature_name)
);
CREATE INDEX IF NOT EXISTS quant_feature_drift_date_idx ON public.quant_feature_drift(date DESC);
CREATE INDEX IF NOT EXISTS quant_feature_drift_level_idx ON public.quant_feature_drift(threshold_exceeded, date DESC);

COMMENT ON TABLE  public.quant_feature_drift IS '特征分布漂移；参考分布取训练窗口，当前分布取最近滚动窗口';
COMMENT ON COLUMN public.quant_feature_drift.threshold_exceeded IS '正常 / 关注（PSI>0.1）/ 严重（PSI>0.25）/ 样本不足';
COMMENT ON COLUMN public.quant_feature_drift.mean_shift IS '当前窗口均值减参考窗口均值，按参考窗口标准差标准化';
COMMENT ON COLUMN public.quant_feature_drift.note IS '该维度不可计算时写明原因（如无数据源、该维无列），不静默留空';
COMMENT ON COLUMN public.quant_feature_drift.sample_size IS '参与本次计算的有效样本数；过小则判定为样本不足而非照常给数';

-- 概念漂移：滚动 Rank IC / ICIR / HAC t / NDCG / Top 超额。
-- 列名用 window_days 而不是需求文里的 window：window 是 PostgreSQL 的保留字，
-- 建表语句会直接语法错误（LINE ... window INTEGER NOT NULL）。
-- 需求文里该字段的含义是「滚动窗口长度」，语义不变。
CREATE TABLE IF NOT EXISTS public.quant_concept_drift (
    id          BIGSERIAL PRIMARY KEY,
    date        DATE NOT NULL,
    window_days INTEGER NOT NULL,
    rank_ic     DOUBLE PRECISION,
    icir        DOUBLE PRECISION,
    hac_t       DOUBLE PRECISION,
    ndcg_ratio  DOUBLE PRECISION,
    top_excess  DOUBLE PRECISION,
    trend       TEXT NOT NULL DEFAULT '未知',
    negative_streak INTEGER,
    note        TEXT,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (date, window_days)
);
CREATE INDEX IF NOT EXISTS quant_concept_drift_date_idx ON public.quant_concept_drift(date DESC, window_days);

COMMENT ON TABLE  public.quant_concept_drift IS '概念漂移：模型预测能力随时间的变化';
COMMENT ON COLUMN public.quant_concept_drift.window_days IS '滚动窗口长度（可用样本外交易日数）；需求文里叫 window，因保留字改名';
COMMENT ON COLUMN public.quant_concept_drift.icir IS 'Rank IC 均值除以标准差，衡量预测能力的稳定性';
COMMENT ON COLUMN public.quant_concept_drift.trend IS '好转 / 持平 / 恶化 / 未知，由本行与上一行对比得出';
COMMENT ON COLUMN public.quant_concept_drift.negative_streak IS '截至本日、连续 Rank IC 为负的样本外交易日数（按可用行计，非自然日）';
COMMENT ON COLUMN public.quant_concept_drift.note IS '口径说明：滚动窗口按可用样本外行计；ndcg_ratio 是整个样本外区间的值，不是本窗口的值';

-- 预测漂移：Top-N 组合的行业集中度与市值分位、分数分布。
-- 分数用未归一化的原始预测分；prediction_zscore 每日被强制标准化，逐日记录恒为常数，没有信息量。
CREATE TABLE IF NOT EXISTS public.quant_prediction_drift (
    id                       BIGSERIAL PRIMARY KEY,
    date                     DATE NOT NULL,
    top_n                    INTEGER NOT NULL,
    industry_hhi             DOUBLE PRECISION,
    market_cap_quantile_mean DOUBLE PRECISION,
    score_mean               DOUBLE PRECISION,
    score_std                DOUBLE PRECISION,
    hhi_threshold_exceeded   TEXT NOT NULL DEFAULT '正常',
    hhi_history_p95          DOUBLE PRECISION,
    stocks                   INTEGER,
    note                     TEXT,
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (date, top_n)
);
CREATE INDEX IF NOT EXISTS quant_prediction_drift_date_idx ON public.quant_prediction_drift(date DESC);

COMMENT ON TABLE  public.quant_prediction_drift IS '预测漂移：每日 Top-N 组合的行业集中度、市值分位与分数分布';
COMMENT ON COLUMN public.quant_prediction_drift.industry_hhi IS 'Top-N 组合的申万一级行业赫芬达尔指数，越接近 1 越集中';
COMMENT ON COLUMN public.quant_prediction_drift.hhi_threshold_exceeded IS '正常 / 关注，超过历史 95 分位为关注';
COMMENT ON COLUMN public.quant_prediction_drift.hhi_history_p95 IS '判定所用的历史 95 分位基准，一并落库便于复核';

-- 模型归档注册表：回滚的依据。活动文件恒等于最新一次产出，历史版本进 models/archive/。
CREATE TABLE IF NOT EXISTS public.quant_model_registry (
    context_id     TEXT PRIMARY KEY,
    model_version  TEXT        NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    reference_date DATE,
    passed         BOOLEAN     NOT NULL DEFAULT FALSE,
    base_rate      DOUBLE PRECISION,
    archive_path   TEXT,
    panel_path     TEXT,
    is_active      BOOLEAN     NOT NULL DEFAULT FALSE,
    retired_at     TIMESTAMPTZ,
    details        JSONB       NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS quant_model_registry_active_idx ON public.quant_model_registry(is_active) WHERE is_active;
CREATE INDEX IF NOT EXISTS quant_model_registry_passed_idx ON public.quant_model_registry(passed, created_at DESC);

COMMENT ON TABLE  public.quant_model_registry IS '模型版本登记表：归档路径、是否通过验证、当前是否活动；回滚的唯一依据';
COMMENT ON COLUMN public.quant_model_registry.passed IS '该次产出的模型是否至少有一个通过验证；未通过的也登记，但不参与回滚候选';
COMMENT ON COLUMN public.quant_model_registry.base_rate IS '该次产出中通过验证的模型比例（通过数 / 总数）';
COMMENT ON COLUMN public.quant_model_registry.archive_path IS '归档目录；缺失即表示该版本不可回滚，回滚前必须校验存在';

-- 降级状态：模型维度的当前健康状态。显式记原因与操作人，避免「谁降的、为什么、怎么解除」全靠翻日志。
CREATE TABLE IF NOT EXISTS public.quant_model_health_state (
    model_version TEXT PRIMARY KEY,
    degraded      BOOLEAN     NOT NULL DEFAULT FALSE,
    reason        TEXT,
    since         TIMESTAMPTZ,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_by    TEXT NOT NULL DEFAULT ''
);

COMMENT ON TABLE  public.quant_model_health_state IS '模型降级状态；degraded 为真时页面暂停输出评分';
COMMENT ON COLUMN public.quant_model_health_state.updated_by IS '置位或解除的操作者：scheduled（自动）或管理员用户名';

-- 告警消息：写库后管理员页可读，Webhook/邮件只是同一条消息的额外通道。
CREATE TABLE IF NOT EXISTS public.quant_model_alerts (
    id            BIGSERIAL PRIMARY KEY,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    level         TEXT NOT NULL,
    category      TEXT NOT NULL,
    model_version TEXT,
    title         TEXT NOT NULL,
    message       TEXT NOT NULL,
    details       JSONB NOT NULL DEFAULT '{}'::jsonb,
    acknowledged  BOOLEAN NOT NULL DEFAULT FALSE,
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by TEXT
);
CREATE INDEX IF NOT EXISTS quant_model_alerts_time_idx ON public.quant_model_alerts(created_at DESC, id DESC);
CREATE INDEX IF NOT EXISTS quant_model_alerts_open_idx ON public.quant_model_alerts(acknowledged, created_at DESC);

COMMENT ON TABLE  public.quant_model_alerts IS '模型治理告警；写库是必达通道，Webhook 与邮件为可选通道';
COMMENT ON COLUMN public.quant_model_alerts.level IS 'warning（关注）或 critical（严重）';
COMMENT ON COLUMN public.quant_model_alerts.category IS 'pass_rate / drift / lifecycle';

-- 管理员操作审计：回滚、强制评估、解除降级等都留痕。
CREATE TABLE IF NOT EXISTS public.quant_model_governance_events (
    id            BIGSERIAL PRIMARY KEY,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    operation     TEXT NOT NULL,
    operator      TEXT NOT NULL DEFAULT '',
    model_version TEXT,
    context_id    TEXT,
    before_value  JSONB,
    after_value   JSONB,
    note          TEXT
);
CREATE INDEX IF NOT EXISTS quant_model_governance_events_time_idx
    ON public.quant_model_governance_events(created_at DESC, id DESC);

COMMENT ON TABLE  public.quant_model_governance_events IS '模型治理操作审计：谁在什么时候把哪个模型换成了什么';

INSERT INTO public.quant_schema_versions(version) VALUES (5) ON CONFLICT DO NOTHING;
