-- 账号认证表仅保存密码摘要，不保存明文密码。
CREATE TABLE IF NOT EXISTS public.quant_users (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    username VARCHAR(32) NOT NULL UNIQUE CHECK (username ~ '^[A-Za-z0-9_.-]{3,32}$'),
    password_hash TEXT NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    must_change_password BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_login_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS quant_users_enabled_idx ON public.quant_users(enabled);

CREATE TABLE IF NOT EXISTS public.quant_auth_sessions (
    token_hash VARCHAR(64) PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES public.quant_users(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS quant_auth_sessions_user_idx ON public.quant_auth_sessions(user_id);
CREATE INDEX IF NOT EXISTS quant_auth_sessions_expiry_idx ON public.quant_auth_sessions(expires_at);

-- 只保存来源和账号的摘要；固定窗口原子计数，不依赖进程内存。
CREATE TABLE IF NOT EXISTS public.quant_login_limits (
    bucket VARCHAR(80) PRIMARY KEY,
    attempts INTEGER NOT NULL,
    window_started TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS quant_login_limits_window_idx ON public.quant_login_limits(window_started);
INSERT INTO public.quant_schema_versions(version) VALUES (3) ON CONFLICT DO NOTHING;
