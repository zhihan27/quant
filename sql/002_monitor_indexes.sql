-- 监控累计数据的最新报价查询和过期清理索引。
CREATE INDEX IF NOT EXISTS quant_quote_observed_idx ON public.quant_quote_snapshots(observed_at);
CREATE INDEX IF NOT EXISTS quant_minute_time_idx ON public.quant_minute_bars(bar_time);
INSERT INTO public.quant_schema_versions(version) VALUES (2) ON CONFLICT DO NOTHING;
