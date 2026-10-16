# Linux + Nginx 部署

使用 Python 3.11及以上版本，将工程放到 `/opt/quantify`，网页工作目录为 `/opt/quantify/quant`。

```bash
sudo apt-get install python3-venv libgomp1
sudo useradd --system --home /opt/quantify --shell /usr/sbin/nologin quant
cd /opt/quantify/quant
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
sudo install -d -o quant -g quant data logs models
sudo chown -R quant:quant /opt/quantify
sudo install -m 600 deploy/quantify.env.example /etc/quantify.env
sudoedit /etc/quantify.env
```

在 `/etc/quantify.env` 填入数据库密码及真实域名。当前数据库为 `127.0.0.1:5432/quantify`，模式为 `public`。服务器上的127.0.0.1表示服务器本身，需先准备该数据库。仅部署源码不会带走本机数据库中的账号、持仓；迁移数据请使用 `pg_dump`/`pg_restore`。账号和密码摘要也在 PostgreSQL 中。

postgres 超级账号与示例口令不适合作为正式公网服务的长期凭据，正式部署建议创建仅可访问 quantify/public 的业务账号，通过环境文件覆盖。项目本地的 `config/database.local.json` 是私有文件（已加入忽略列表），勿提交或对外提供下载；仓库内只保留占位模板 `config/database.example.json`。

服务器新数据库通过终端创建账号，不提供网页注册：

```bash
sudo bash -c 'set -a; source /etc/quantify.env; set +a; cd /opt/quantify/quant; .venv/bin/python manage_users.py create admin --temporary'
```

按要求可输入初始密码 `admin`，首次登录强制设置至少12位新密码。正式密码可直接省略 `--temporary` 创建。重置和禁用命令分别为 `manage_users.py reset admin`、`manage_users.py disable admin`。重置或禁用立即撤销全部会话。本机已创建的账号不会由应用启动覆盖。

登录令牌只保存在当前 Streamlit 会话，刷新整页或服务重启需要重新登录；会话最多12小时、30分钟无业务执行后失效，每个账号最多5个会话。监控页自动刷新会持续更新活动时间，12小时绝对期限仍有效。所有授权账号共用现有自选、持仓与监控工作台，不提供多租户数据隔离。

将 `deploy/nginx.conf` 放到 Nginx 的 `http` 块（常见路径为 `/etc/nginx/conf.d/quantify.conf`），替换所有 `quant.example.com` 和证书路径。提前安装有效TLS证书，不要关闭 Streamlit 的来源校验或 XSRF 保护。

```bash
sudo install -m 644 deploy/quant-web.service deploy/quant-monitor.service /etc/systemd/system/
sudo nginx -t
sudo systemctl daemon-reload
sudo systemctl enable --now quant-web quant-monitor
sudo systemctl reload nginx
```

防火墙只对外开放需要的80/443端口，8501仅监听127.0.0.1，5432仅允许本机或受信任内网。网页服务与监控进程分开，监控沿用数据库内的开启状态，页面关闭或服务重启不影响独立任务。内存及CPU限制按实际扫描规模调整，超过限制可能终止正在运行的任务。

Nginx 默认每IP请求20次/秒（允许静态资源突发150次）、每IP最多10条活跃请求，WebSocket每IP最多5条连接和每分钟5次握手（突发5次），总连接100条。登录在计算Argon2id摘要前按数据库固定窗口限流：每账号15分钟5次、每来源15分钟30次、全局每分钟120次，成功后仅清除该账号计数。进程内最多并发2次密码计算。可按合法用户数量调整参数，共享出口用户会共享IP额度。

这些措施用于降低应用层滥用和暴力破解，无法消除链路带宽被打满的DDoS风险。公网使用还需云厂商防DDoS或支持WebSocket的CDN/WAF，源站仅放行防护节点；如果增加CDN，先配置Nginx的 `set_real_ip_from` 仅信任该服务商公布的网段，再使用真实IP限流，不能无条件信任客户端的 X-Forwarded-For。

模型分析和扫描通过数据库咨询锁限制为同一数据库一个重计算任务，重复提交会立即提示繁忙。完整市场截面训练的内存需求随股票数量和历史长度增长，服务模板中的2GB内存上限仅是初始值，部署前须按真实股票池容量调整。不要以扩大请求并发的方式突破行情源限流。

配置依据：[Nginx请求限流](https://nginx.org/en/docs/http/ngx_http_limit_req_module.html)、[WebSocket代理](https://nginx.org/en/docs/http/websocket.html)、[OWASP密码存储](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html)。

验证命令：`curl -I https://真实域名` 应可访问；未登录直接访问 `/portfolio`、`/monitor` 只显示登录页。查看日志使用 `journalctl -u quant-web -u quant-monitor -f`。本地Windows验证不能替代服务器上的 `nginx -t`、TLS与防火墙检查。
