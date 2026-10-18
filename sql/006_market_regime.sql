-- 每轮扫描保存独立市场状态，历史日期重训不会覆盖更新交易日。
CREATE TABLE IF NOT EXISTS public.quant_market_regime (
    id BIGSERIAL PRIMARY KEY,
    reference_date DATE NOT NULL,
    context_id TEXT,
    model_version TEXT NOT NULL,
    trigger TEXT NOT NULL,
    label TEXT NOT NULL,
    details JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS quant_market_regime_date_idx
    ON public.quant_market_regime(reference_date DESC, created_at DESC);

INSERT INTO public.quant_schema_versions(version) VALUES (6) ON CONFLICT DO NOTHING;
