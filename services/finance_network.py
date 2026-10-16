"""公开行情域名的代理故障回退，不修改进程环境或系统代理。"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from typing import Any
from urllib.parse import urlsplit

import requests


class FinanceNetwork:
    """优先沿用代理，在公开行情连接断开时尝试限定域名直连。"""

    def __init__(self, session: requests.Session, logger: logging.Logger | None = None) -> None:
        self.session = session
        self.direct_session = requests.Session()
        self.direct_session.trust_env = False
        self.logger = logger or logging.getLogger("quant.network")
        self.mode = os.environ.get("QUANT_NETWORK_MODE", "auto").strip().lower()
        if self.mode not in ("auto", "proxy", "direct"):
            raise ValueError("QUANT_NETWORK_MODE 仅支持 auto、proxy、direct。")
        self._direct_until: dict[str, float] = {}

    @staticmethod
    def _public_host(host: str) -> bool:
        domains = ("eastmoney.com", "sina.com.cn", "sina.cn", "sinajs.cn", "finance.qq.com", "ifzq.gtimg.cn")
        return any(host == domain or host.endswith("." + domain) for domain in domains)

    def _direct_get(self, url: str, **arguments: Any) -> requests.Response:
        arguments["headers"] = {**self.session.headers, **(arguments.get("headers") or {})}
        return self.direct_session.get(url, **arguments)

    def get(self, url: str, **arguments: Any) -> requests.Response:
        """缓存已成功的域名直连路由五分钟，避免每页重复撞上同一故障代理。"""
        host = urlsplit(url).hostname or ""
        eligible = self._public_host(host)
        if eligible and (self.mode == "direct" or (self.mode == "auto" and self._direct_until.get(host, 0) > time.monotonic())):
            return self._direct_get(url, **arguments)
        try:
            return self.session.get(url, **arguments)
        except requests.exceptions.ConnectionError as exc:
            disconnected = isinstance(exc, requests.exceptions.ProxyError) or "RemoteDisconnected" in str(exc)
            if self.mode != "auto" or not eligible or not disconnected or isinstance(exc, requests.exceptions.SSLError):
                raise
            self.logger.warning("行情连接异常断开，尝试公开行情域名直连 host=%s；系统代理设置保持不变", host)
            response = self._direct_get(url, **arguments)
            response.raise_for_status()
            self._direct_until[host] = time.monotonic() + 300
            self.logger.info("行情直连恢复成功 host=%s；后续五分钟复用直连", host)
            return response

    def status(self) -> dict[str, Any]:
        """返回不包含凭据的实际网络路由状态。"""
        hosts = [host for host, until in self._direct_until.items() if until > time.monotonic()]
        return {"mode": self.mode, "direct_fallback_hosts": sorted(hosts)}

    def close(self) -> None:
        """关闭额外的直连连接池，原会话由调用方管理。"""
        self.direct_session.close()


def brief_network_error(error: Exception | str, limit: int = 280) -> str:
    """页面只展示故障类别和末尾原因，完整堆栈保留在日志中。"""
    detail = str(error).strip()
    if isinstance(error, subprocess.TimeoutExpired):
        return f"备用历史行情处理超时（{float(error.timeout):g} 秒），请稍后重试"
    if isinstance(error, subprocess.CalledProcessError):
        output = error.stderr or error.stdout
        if output:
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            return brief_network_error(output, limit)
        return f"备用历史行情进程异常退出（代码 {error.returncode}），详细原因见日志"
    if isinstance(error, requests.exceptions.Timeout):
        return "行情接口请求超时，请稍后重试或检查网络"
    if "ProxyError" in detail or "Unable to connect to proxy" in detail:
        return "代理连接失败（ProxyError），请检查本地代理或切换行情连接模式"
    lines = [line.strip() for line in detail.splitlines() if line.strip() and "Traceback" not in line and "it/s]" not in line and "s/it]" not in line]
    result = lines[-1] if lines else type(error).__name__
    return result if len(result) <= limit else result[:limit] + "…"
