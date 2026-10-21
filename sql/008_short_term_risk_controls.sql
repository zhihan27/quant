ALTER TABLE public.quant_short_positions
    ADD COLUMN IF NOT EXISTS trailing_stop_pct NUMERIC(8,4)
        CHECK (trailing_stop_pct IS NULL OR (trailing_stop_pct > 0 AND trailing_stop_pct <= 90)),
    ADD COLUMN IF NOT EXISTS strategy_invalidation_price NUMERIC(18,6)
        CHECK (strategy_invalidation_price IS NULL OR (strategy_invalidation_price > 0 AND strategy_invalidation_price < 'Infinity'::numeric)),
    ADD COLUMN IF NOT EXISTS high_watermark_price NUMERIC(18,6)
        CHECK (high_watermark_price IS NULL OR (high_watermark_price > 0 AND high_watermark_price < 'Infinity'::numeric));

UPDATE public.quant_short_positions
SET high_watermark_price = average_buy_price
WHERE high_watermark_price IS NULL;

INSERT INTO public.quant_schema_versions(version) VALUES (8) ON CONFLICT DO NOTHING;
