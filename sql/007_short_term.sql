-- 短线研究结果与用户账户执行账本分离；研究分数不会自动变成订单。
CREATE TABLE IF NOT EXISTS public.quant_short_runs (
    run_id UUID PRIMARY KEY,
    strategy_id VARCHAR(80) NOT NULL,
    strategy_version VARCHAR(40) NOT NULL,
    horizon SMALLINT NOT NULL CHECK (horizon IN (1, 3, 5)),
    status VARCHAR(24) NOT NULL CHECK (status IN ('queued', 'running', 'research_only', 'insufficient_data', 'failed')),
    signal_date DATE,
    snapshot_path TEXT,
    snapshot_hash CHAR(64),
    model_path TEXT,
    model_hash CHAR(64),
    manifest JSONB NOT NULL DEFAULT '{}'::jsonb,
    validation JSONB NOT NULL DEFAULT '{}'::jsonb,
    result JSONB NOT NULL DEFAULT '{}'::jsonb,
    error TEXT,
    created_by BIGINT REFERENCES public.quant_users(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS quant_short_runs_created_idx ON public.quant_short_runs(created_at DESC);

CREATE TABLE IF NOT EXISTS public.quant_short_predictions (
    run_id UUID NOT NULL REFERENCES public.quant_short_runs(run_id) ON DELETE CASCADE,
    code VARCHAR(6) NOT NULL CHECK (code ~ '^[0-9]{6}$'),
    signal_time TIMESTAMPTZ NOT NULL,
    ranking_score DOUBLE PRECISION NOT NULL CHECK (ranking_score > '-Infinity'::float8 AND ranking_score < 'Infinity'::float8),
    rank INTEGER NOT NULL CHECK (rank > 0),
    percentile DOUBLE PRECISION NOT NULL CHECK (percentile >= 0 AND percentile <= 1),
    status VARCHAR(24) NOT NULL DEFAULT 'research_only',
    features JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, code)
);
CREATE INDEX IF NOT EXISTS quant_short_predictions_signal_idx ON public.quant_short_predictions(signal_time DESC, rank);

CREATE TABLE IF NOT EXISTS public.quant_short_accounts (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES public.quant_users(id) ON DELETE CASCADE,
    name VARCHAR(80) NOT NULL,
    initial_cash NUMERIC(20,4) CHECK (initial_cash IS NULL OR (initial_cash >= 0 AND initial_cash < 'Infinity'::numeric)),
    cash_balance NUMERIC(20,4) CHECK (cash_balance IS NULL OR (cash_balance >= 0 AND cash_balance < 'Infinity'::numeric)),
    max_total_exposure NUMERIC(20,4) CHECK (max_total_exposure IS NULL OR (max_total_exposure > 0 AND max_total_exposure < 'Infinity'::numeric)),
    max_single_exposure NUMERIC(20,4) CHECK (max_single_exposure IS NULL OR (max_single_exposure > 0 AND max_single_exposure < 'Infinity'::numeric)),
    cost_profile JSONB NOT NULL DEFAULT '{}'::jsonb,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, name),
    UNIQUE (id, user_id)
);

CREATE TABLE IF NOT EXISTS public.quant_short_intents (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES public.quant_users(id) ON DELETE CASCADE,
    account_id BIGINT NOT NULL REFERENCES public.quant_short_accounts(id) ON DELETE CASCADE,
    strategy_id VARCHAR(80) NOT NULL,
    run_id UUID REFERENCES public.quant_short_runs(run_id) ON DELETE SET NULL,
    code VARCHAR(6) NOT NULL CHECK (code ~ '^[0-9]{6}$'),
    side VARCHAR(4) NOT NULL CHECK (side IN ('BUY', 'SELL')),
    status VARCHAR(24) NOT NULL CHECK (status IN ('pending', 'triggered', 'partially_filled', 'filled', 'expired', 'cancelled', 'rejected', 'blocked')),
    trigger_rule TEXT NOT NULL,
    reference_price NUMERIC(18,6) CHECK (reference_price IS NULL OR (reference_price > 0 AND reference_price < 'Infinity'::numeric)),
    limit_price NUMERIC(18,6) CHECK (limit_price IS NULL OR (limit_price > 0 AND limit_price < 'Infinity'::numeric)),
    intended_shares INTEGER CHECK (intended_shares IS NULL OR intended_shares > 0),
    valid_until TIMESTAMPTZ,
    idempotency_key UUID NOT NULL UNIQUE,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (account_id, user_id) REFERENCES public.quant_short_accounts(id, user_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS quant_short_intents_queue_idx ON public.quant_short_intents(user_id, account_id, status, created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS quant_short_intent_run_signal_idx
    ON public.quant_short_intents(user_id, account_id, run_id, code, side)
    WHERE run_id IS NOT NULL AND status NOT IN ('cancelled', 'expired', 'rejected');

CREATE TABLE IF NOT EXISTS public.quant_short_positions (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES public.quant_users(id) ON DELETE CASCADE,
    account_id BIGINT NOT NULL REFERENCES public.quant_short_accounts(id) ON DELETE CASCADE,
    strategy_id VARCHAR(80) NOT NULL,
    entry_intent_id BIGINT REFERENCES public.quant_short_intents(id) ON DELETE SET NULL,
    code VARCHAR(6) NOT NULL CHECK (code ~ '^[0-9]{6}$'),
    name VARCHAR(80) NOT NULL DEFAULT '',
    buy_date DATE NOT NULL,
    average_buy_price NUMERIC(18,6) NOT NULL CHECK (average_buy_price > 0 AND average_buy_price < 'Infinity'::numeric),
    original_shares INTEGER NOT NULL CHECK (original_shares > 0),
    remaining_shares INTEGER NOT NULL CHECK (remaining_shares >= 0 AND remaining_shares <= original_shares),
    available_shares INTEGER NOT NULL DEFAULT 0 CHECK (available_shares >= 0 AND available_shares <= remaining_shares),
    buy_fees NUMERIC(18,4) NOT NULL DEFAULT 0 CHECK (buy_fees >= 0 AND buy_fees < 'Infinity'::numeric),
    planned_exit_date DATE,
    stop_loss_price NUMERIC(18,6) CHECK (stop_loss_price IS NULL OR (stop_loss_price > 0 AND stop_loss_price < 'Infinity'::numeric)),
    take_profit_price NUMERIC(18,6) CHECK (take_profit_price IS NULL OR (take_profit_price > 0 AND take_profit_price < 'Infinity'::numeric)),
    status VARCHAR(20) NOT NULL DEFAULT 'holding' CHECK (status IN ('holding', 'pending_exit', 'closed')),
    exit_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (account_id, user_id) REFERENCES public.quant_short_accounts(id, user_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS quant_short_positions_open_idx ON public.quant_short_positions(user_id, account_id, status, code);

CREATE TABLE IF NOT EXISTS public.quant_short_fills (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES public.quant_users(id) ON DELETE CASCADE,
    account_id BIGINT NOT NULL REFERENCES public.quant_short_accounts(id) ON DELETE CASCADE,
    intent_id BIGINT REFERENCES public.quant_short_intents(id) ON DELETE SET NULL,
    position_id BIGINT NOT NULL REFERENCES public.quant_short_positions(id) ON DELETE RESTRICT,
    code VARCHAR(6) NOT NULL CHECK (code ~ '^[0-9]{6}$'),
    side VARCHAR(4) NOT NULL CHECK (side IN ('BUY', 'SELL')),
    traded_at TIMESTAMPTZ NOT NULL,
    price NUMERIC(18,6) NOT NULL CHECK (price > 0 AND price < 'Infinity'::numeric),
    shares INTEGER NOT NULL CHECK (shares > 0),
    fee NUMERIC(18,4) NOT NULL DEFAULT 0 CHECK (fee >= 0 AND fee < 'Infinity'::numeric),
    external_trade_id TEXT,
    idempotency_key UUID NOT NULL UNIQUE,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (account_id, user_id) REFERENCES public.quant_short_accounts(id, user_id) ON DELETE CASCADE
);
CREATE UNIQUE INDEX IF NOT EXISTS quant_short_fill_external_idx
    ON public.quant_short_fills(account_id, external_trade_id)
    WHERE external_trade_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS quant_short_fills_trade_idx ON public.quant_short_fills(user_id, account_id, traded_at DESC);

CREATE TABLE IF NOT EXISTS public.quant_short_events (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES public.quant_users(id) ON DELETE CASCADE,
    account_id BIGINT NOT NULL REFERENCES public.quant_short_accounts(id) ON DELETE CASCADE,
    strategy_id VARCHAR(80) NOT NULL,
    code VARCHAR(6) NOT NULL CHECK (code ~ '^[0-9]{6}$'),
    position_id BIGINT REFERENCES public.quant_short_positions(id) ON DELETE SET NULL,
    intent_id BIGINT REFERENCES public.quant_short_intents(id) ON DELETE SET NULL,
    event_type VARCHAR(40) NOT NULL,
    event_status VARCHAR(24) NOT NULL,
    event_key VARCHAR(240) NOT NULL UNIQUE,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (account_id, user_id) REFERENCES public.quant_short_accounts(id, user_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS quant_short_events_recent_idx ON public.quant_short_events(user_id, account_id, created_at DESC);

COMMENT ON TABLE public.quant_short_runs IS '独立短线日频研究运行台账；模型仅研究态，未经审批不产生交易指令';
COMMENT ON TABLE public.quant_short_predictions IS '冻结的短线截面排序分数，不代表概率或实际成交';
COMMENT ON TABLE public.quant_short_accounts IS '按用户隔离的短线账户预算与成本配置';
COMMENT ON TABLE public.quant_short_positions IS '按策略与账户区分的真实成交持仓批次';
COMMENT ON TABLE public.quant_short_fills IS '用户人工确认或导入的真实成交；唯一键防止重复入账';

INSERT INTO public.quant_schema_versions(version) VALUES (7) ON CONFLICT DO NOTHING;
