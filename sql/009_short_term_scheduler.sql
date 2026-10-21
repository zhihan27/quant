CREATE TABLE IF NOT EXISTS public.quant_short_schedule_runs (
    schedule_date DATE NOT NULL,
    horizon SMALLINT NOT NULL CHECK (horizon IN (1, 3, 5)),
    run_id UUID NOT NULL UNIQUE REFERENCES public.quant_short_runs(run_id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (schedule_date, horizon)
);

INSERT INTO public.quant_schema_versions(version) VALUES (9) ON CONFLICT DO NOTHING;
