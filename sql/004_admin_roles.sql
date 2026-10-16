-- 管理员角色与待激活状态：新建账号由使用者本人首次设置密码，不经过管理员之手。
-- 已有账号保持可用：activated 默认 TRUE，存量行不受影响。
ALTER TABLE public.quant_users ADD COLUMN IF NOT EXISTS is_admin BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE public.quant_users ADD COLUMN IF NOT EXISTS activated BOOLEAN NOT NULL DEFAULT TRUE;
ALTER TABLE public.quant_users ADD COLUMN IF NOT EXISTS activated_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS quant_users_admin_idx ON public.quant_users(is_admin) WHERE is_admin;

COMMENT ON COLUMN public.quant_users.is_admin IS '管理员可管理账号，普通账号看不到账号管理页';
COMMENT ON COLUMN public.quant_users.activated IS 'FALSE 表示待激活：密码由使用者本人首次设置，期间无法登录';
INSERT INTO public.quant_schema_versions(version) VALUES (4) ON CONFLICT DO NOTHING;
