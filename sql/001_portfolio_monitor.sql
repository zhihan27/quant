-- 业务数据保存在用户指定的 public 模式，行情快照不冒充逐笔成交。
CREATE TABLE IF NOT EXISTS public.quant_schema_versions (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.quant_watchlist (
    code VARCHAR(6) PRIMARY KEY CHECK (code ~ '^[0-9]{6}$'),
    name VARCHAR(80) NOT NULL DEFAULT '',
    note TEXT NOT NULL DEFAULT '',
    monitor_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.quant_positions (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    code VARCHAR(6) NOT NULL CHECK (code ~ '^[0-9]{6}$'),
    name VARCHAR(80) NOT NULL DEFAULT '',
    buy_date DATE NOT NULL,
    buy_price NUMERIC(18,6) NOT NULL CHECK (buy_price > 0 AND buy_price < 'Infinity'::numeric),
    shares INTEGER NOT NULL CHECK (shares > 0),
    available_shares INTEGER NOT NULL CHECK (available_shares >= 0 AND available_shares <= shares),
    fees NUMERIC(18,4) NOT NULL DEFAULT 0 CHECK (fees >= 0 AND fees < 'Infinity'::numeric),
    note TEXT NOT NULL DEFAULT '',
    version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS quant_positions_code_idx ON public.quant_positions(code);

CREATE TABLE IF NOT EXISTS public.quant_position_audit (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    position_id BIGINT NOT NULL,
    operation VARCHAR(12) NOT NULL CHECK (operation IN ('新增', '编辑', '删除')),
    before_value JSONB,
    after_value JSONB,
    changed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.quant_monitor_config (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    enabled BOOLEAN NOT NULL DEFAULT FALSE,
    config JSONB NOT NULL DEFAULT '{}'::jsonb,
    last_cycle_at TIMESTAMPTZ,
    last_error TEXT,
    cycle_seconds DOUBLE PRECISION,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
INSERT INTO public.quant_monitor_config(id) VALUES (1) ON CONFLICT DO NOTHING;

CREATE TABLE IF NOT EXISTS public.quant_quote_snapshots (
    code VARCHAR(6) NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL,
    source_time TIMESTAMPTZ,
    price NUMERIC(18,6) NOT NULL CHECK (price > 0 AND price < 'Infinity'::numeric),
    volume_lots DOUBLE PRECISION,
    amount_yuan DOUBLE PRECISION,
    source TEXT NOT NULL,
    data_status TEXT NOT NULL,
    payload JSONB NOT NULL,
    PRIMARY KEY (code, observed_at)
);
CREATE INDEX IF NOT EXISTS quant_quote_source_idx ON public.quant_quote_snapshots(code, source_time DESC);
CREATE INDEX IF NOT EXISTS quant_quote_observed_idx ON public.quant_quote_snapshots(observed_at);

CREATE TABLE IF NOT EXISTS public.quant_minute_bars (
    code VARCHAR(6) NOT NULL,
    bar_time TIMESTAMPTZ NOT NULL,
    open DOUBLE PRECISION NOT NULL CHECK (open > 0 AND open < 'Infinity'::float8),
    high DOUBLE PRECISION NOT NULL CHECK (high >= open AND high < 'Infinity'::float8),
    low DOUBLE PRECISION NOT NULL CHECK (low > 0 AND low <= open),
    close DOUBLE PRECISION NOT NULL CHECK (close >= low AND close <= high),
    volume_lots DOUBLE PRECISION NOT NULL CHECK (volume_lots >= 0 AND volume_lots < 'Infinity'::float8),
    amount_yuan DOUBLE PRECISION,
    source TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (code, bar_time)
);

CREATE TABLE IF NOT EXISTS public.quant_intraday_signals (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    code VARCHAR(6) NOT NULL,
    peer_code VARCHAR(6) NOT NULL DEFAULT '',
    strategy VARCHAR(40) NOT NULL,
    bar_time TIMESTAMPTZ NOT NULL,
    action TEXT NOT NULL,
    active BOOLEAN NOT NULL DEFAULT FALSE,
    reason TEXT NOT NULL,
    reference_price NUMERIC(18,6),
    details JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (code, peer_code, strategy, bar_time)
);
CREATE INDEX IF NOT EXISTS quant_minute_time_idx ON public.quant_minute_bars(bar_time);
CREATE INDEX IF NOT EXISTS quant_signals_time_idx ON public.quant_intraday_signals(bar_time DESC);

CREATE TABLE IF NOT EXISTS public.quant_saved_backtests (
    run_id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    parameters JSONB NOT NULL,
    model_version TEXT NOT NULL,
    result JSONB NOT NULL
);

COMMENT ON TABLE public.quant_watchlist IS '持久化自选股票及监控开关';
COMMENT ON TABLE public.quant_positions IS '手工维护的持仓批次，不代表券商成交回报';
COMMENT ON TABLE public.quant_position_audit IS '持仓新增、编辑和删除的事务内修改记录';
COMMENT ON TABLE public.quant_quote_snapshots IS '定时采集的行情快照，非逐笔行情';
COMMENT ON TABLE public.quant_minute_bars IS '不复权且已完成的分钟线';
COMMENT ON TABLE public.quant_intraday_signals IS '分钟策略研究提醒，不执行真实委托';
INSERT INTO public.quant_schema_versions(version) VALUES (1) ON CONFLICT DO NOTHING;
