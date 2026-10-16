"""模型治理告警的分发：写库、写日志，以及可选的 Webhook 与邮件。

## 通道的主次

**写库是必达通道**，Webhook 与邮件是可选通道。顺序上先写库再发外部通知：
外部通道失败不能把告警弄丢——管理员页面读的就是库里的那几条。
反过来的话，一次 Webhook 超时就会让一条「模型连续三次未通过」的告警凭空消失。

## 为什么必须去重

每日扫描每次都会重新判定一次同样的条件。不去重的话，一个持续存在的问题
每天都会追加一条措辞完全相同的告警，几天之后页面上的告警列表就只剩同一条消息的副本，
真正的**新**问题反而被淹掉。去重的键是（类别，标题，模型版本），窗口由
``settings.ALERT_DEDUP_HOURS`` 决定。

去重窗口内被抑制的告警会返回 ``suppressed=True``，并且**不计入**返回条数统计，
免得页面上把「今天没有新问题」显示成「今天有 6 条告警」。

## 敏感信息

Webhook 地址与 SMTP 凭据只从环境变量读，不进 ``config/settings.py``，
沿用 ``QUANT_PG_*`` 的先例。

本模块刻意不引入 Streamlit：定时任务要在没有网页的进程里跑它。
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import urllib.error
import urllib.request
from email.message import EmailMessage
from typing import Any, Iterable, Sequence

from config import settings
from services.database import QuantRepository

LOGGER = logging.getLogger("quant.alerting")


class AlertDeliveryError(RuntimeError):
    """某个外部告警通道投递失败。写库通道不受影响。"""


def _db_dedup_clause() -> str:
    return "created_at >= now() - make_interval(hours => %s)"


def _recent_titles(repository: QuantRepository, titles: Sequence[str]) -> set[tuple[str, str, str]]:
    """取去重窗口内已存在的（类别，标题，模型版本）三元组。"""
    if not titles:
        return set()
    with repository.connect() as connection:
        rows = connection.execute(
            "SELECT category, title, COALESCE(model_version, '') AS model_version "
            "FROM public.quant_model_alerts "
            f"WHERE {_db_dedup_clause()} AND title = ANY(%s)",
            (int(settings.ALERT_DEDUP_HOURS), list(titles)),
        ).fetchall()
    return {(row["category"], row["title"], row["model_version"]) for row in rows}


def log_alerts(repository: QuantRepository, alerts: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """把告警写进 ``quant_model_alerts``，返回真正落库的那些。

    返回落库成功的记录而不是全部入参：调用方（扫描管线）要据此决定是否需要触发降级，
    被去重抑制的告警不应该再触发一次降级动作。
    """
    from psycopg.types.json import Jsonb

    candidates = [alert for alert in alerts if alert.get("title")]
    if not candidates:
        return []
    seen = _recent_titles(repository, [str(alert["title"]) for alert in candidates])
    stored: list[dict[str, Any]] = []
    with repository.connect() as connection:
        for alert in candidates:
            model_version = alert.get("model_version") or settings.MODEL_VERSION
            key = (str(alert.get("category") or ""), str(alert["title"]), str(model_version))
            if key in seen:
                LOGGER.info("告警在 %s 小时去重窗口内，已抑制：%s", settings.ALERT_DEDUP_HOURS, alert["title"])
                continue
            row = connection.execute(
                "INSERT INTO public.quant_model_alerts"
                "(level,category,model_version,title,message,details) VALUES (%s,%s,%s,%s,%s,%s) RETURNING *",
                (str(alert.get("level") or "warning"), str(alert.get("category") or "drift"), model_version,
                 str(alert["title"]), str(alert.get("message") or ""), Jsonb(alert.get("details") or {})),
            ).fetchone()
            # 同一次调用里若有重复标题，靠这一个 seen 也挡掉，不必再查库。
            seen.add(key)
            stored.append(dict(row))
    return stored


def _webhook_payload(alerts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Webhook 的请求体。

    只发结构化字段、不发整条记录：这里的 details 可能含大段 JSON，
    而接收端（企业微信机器人、飞书、自建网关）能用的通常只有标题与正文两段文本。
    """
    return {
        "model_version": settings.MODEL_VERSION,
        "count": len(alerts),
        "alerts": [
            {"level": alert["level"], "category": alert["category"],
             "title": alert["title"], "message": alert["message"]}
            for alert in alerts
        ],
    }


def send_webhook(alerts: Sequence[dict[str, Any]], url: str | None = None) -> dict[str, Any]:
    """投递到 Webhook。地址未配置时返回 ``configured=False`` 而不是抛错。

    未配置和投递失败是两件事：前者是正常状态（默认就不配），报错会让定时任务
    每天记一条无意义的异常；后者必须让调用方知道。
    """
    address = url or os.environ.get(settings.ALERT_WEBHOOK_ENV, "").strip()
    if not address:
        return {"configured": False, "delivered": False,
                "note": f"未配置环境变量 {settings.ALERT_WEBHOOK_ENV}，Webhook 通道未启用"}
    request = urllib.request.Request(
        address, data=json.dumps(_webhook_payload(alerts), ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=settings.ALERT_WEBHOOK_TIMEOUT_SECONDS) as response:
            return {"configured": True, "delivered": 200 <= response.status < 300,
                    "status": response.status, "note": ""}
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise AlertDeliveryError(f"Webhook 投递失败：{type(exc).__name__}: {exc}") from exc


def send_email(alerts: Sequence[dict[str, Any]], recipients: Sequence[str] | None = None) -> dict[str, Any]:
    """投递邮件。SMTP 凭据只从环境变量读，未配置时返回 ``configured=False``。"""
    targets = list(recipients or settings.ALERT_EMAIL_RECIPIENTS)
    host = os.environ.get(settings.ALERT_SMTP_HOST_ENV, "").strip()
    sender = os.environ.get(settings.ALERT_SMTP_SENDER_ENV, "").strip()
    if not host or not sender or not targets:
        missing = []
        if not host:
            missing.append(settings.ALERT_SMTP_HOST_ENV)
        if not sender:
            missing.append(settings.ALERT_SMTP_SENDER_ENV)
        if not targets:
            missing.append("settings.ALERT_EMAIL_RECIPIENTS")
        return {"configured": False, "delivered": False,
                "note": f"邮件通道未配置完整，缺少 {'、'.join(missing)}"}
    try:
        port = int(os.environ.get(settings.ALERT_SMTP_PORT_ENV, "465"))
    except ValueError:
        return {"configured": False, "delivered": False,
                "note": f"{settings.ALERT_SMTP_PORT_ENV} 不是合法端口号，邮件通道未启用"}
    message = EmailMessage()
    message["Subject"] = f"[量化系统] 模型治理告警 {len(alerts)} 条"
    message["From"] = sender
    message["To"] = ", ".join(targets)
    message.set_content("\n\n".join(f"[{alert['level']}] {alert['title']}\n{alert['message']}" for alert in alerts))
    user = os.environ.get(settings.ALERT_SMTP_USER_ENV, "").strip()
    password = os.environ.get(settings.ALERT_SMTP_PASSWORD_ENV, "")
    try:
        if port == 465:
            server = smtplib.SMTP_SSL(host, port, timeout=settings.ALERT_SMTP_TIMEOUT_SECONDS)
        else:
            server = smtplib.SMTP(host, port, timeout=settings.ALERT_SMTP_TIMEOUT_SECONDS)
            server.starttls()
        with server:
            if user:
                server.login(user, password)
            server.send_message(message, to_addrs=targets)
        return {"configured": True, "delivered": True, "recipients": len(targets), "note": ""}
    except (smtplib.SMTPException, OSError, ValueError) as exc:
        raise AlertDeliveryError(f"邮件投递失败：{type(exc).__name__}: {exc}") from exc


def dispatch_alerts(repository: QuantRepository, alerts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """告警的总入口：写日志 → 写库 → 可选外部通道。

    :func:`dispatch_alerts` 是扫描管线唯一需要调用的函数。
    外部通道的失败只记日志并体现在返回值里，**不向上抛**：
    一条「模型连续三次未通过」的告警已经落库、已经在页面上，不该因为企业微信挂了
    就让整轮扫描失败、连带着当天的评分一起没有。
    """
    result: dict[str, Any] = {
        "received": len(alerts), "stored": 0, "log_only": False,
        "channels": {"database": {"delivered": False}}, "errors": [],
    }
    if not alerts:
        return result
    if not settings.ALERT_ENABLED:
        result["log_only"] = True
        for alert in alerts:
            LOGGER.warning("[告警未落库，仅日志] %s：%s", alert.get("title"), alert.get("message"))
        return result
    for alert in alerts:
        LOGGER.warning("[%s][%s] %s：%s", alert.get("level"), alert.get("category"),
                       alert.get("title"), alert.get("message"))
    stored = log_alerts(repository, alerts)
    result["stored"] = len(stored)
    result["channels"]["database"] = {"delivered": True, "rows": len(stored)}
    if not stored:
        # 全部被去重抑制：外部通道也没必要再发一遍同样的消息。
        result["suppressed"] = len(alerts)
        return result
    if settings.ALERT_WEBHOOK_ENABLED:
        try:
            result["channels"]["webhook"] = send_webhook(stored)
        except AlertDeliveryError as exc:
            result["errors"].append(str(exc))
            LOGGER.error("%s", exc)
    if settings.ALERT_EMAIL_ENABLED:
        try:
            result["channels"]["email"] = send_email(stored)
        except AlertDeliveryError as exc:
            result["errors"].append(str(exc))
            LOGGER.error("%s", exc)
    return result


def critical_alerts(alerts: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """筛出严重级告警。降级判定只看这一档。"""
    return [alert for alert in alerts if str(alert.get("level")) == "critical"]


def degradation_reason(alerts: Sequence[dict[str, Any]]) -> str | None:
    """把触发降级的告警拼成一句可读的原因；没有严重级告警时返回 None。

    原因要写进 ``quant_model_health_state.reason`` 并在页面上显示，
    所以要能让人一眼看出是「连续失败」还是「严重漂移」，而不是一句「模型异常」。
    """
    severe = critical_alerts(alerts)
    if not severe:
        return None
    return "；".join(str(alert.get("title")) for alert in severe)
