CREATE TABLE IF NOT EXISTS public.quant_short_forward_observations (
    run_id UUID NOT NULL,
    code VARCHAR(6) NOT NULL CHECK (code ~ '^[0-9]{6}$'),
    horizon SMALLINT NOT NULL CHECK (horizon IN (1, 3, 5)),
    signal_time TIMESTAMPTZ NOT NULL,
    entry_date DATE NOT NULL,
    entry_open DOUBLE PRECISION NOT NULL CHECK (entry_open > 0 AND entry_open < 'Infinity'::float8),
    label_end_date DATE NOT NULL,
    exit_open DOUBLE PRECISION NOT NULL CHECK (exit_open > 0 AND exit_open < 'Infinity'::float8),
    gross_return DOUBLE PRECISION NOT NULL CHECK (gross_return > '-Infinity'::float8 AND gross_return < 'Infinity'::float8),
    target_net_return DOUBLE PRECISION NOT NULL CHECK (target_net_return > '-Infinity'::float8 AND target_net_return < 'Infinity'::float8),
    label_basis VARCHAR(80) NOT NULL,
    source_hash CHAR(64) NOT NULL CHECK (source_hash ~ '^[0-9a-f]{64}$'),
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    observed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, code, horizon),
    FOREIGN KEY (run_id, code) REFERENCES public.quant_short_predictions(run_id, code) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS quant_short_forward_observations_date_idx
    ON public.quant_short_forward_observations(label_end_date DESC, horizon);

COMMENT ON TABLE public.quant_short_forward_observations IS '短线预测的前向复权开盘价观察；不代表实际成交或生产晋级证据';

INSERT INTO public.quant_schema_versions(version) VALUES (10) ON CONFLICT DO NOTHING;
