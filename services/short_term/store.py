"""短线研究运行、账户任务、成交批次与事件台账。"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from psycopg.types.json import Jsonb

from config.short_term import SHORT_TERM_STRATEGY_ID
from services.database import QuantRepository, stock_code

SHANGHAI = ZoneInfo("Asia/Shanghai")
ACTIVE_INTENT_STATUSES = ("pending", "triggered", "partially_filled")


def _json(value: Any) -> Jsonb:
    """将业务快照规范成有限 JSON，避免 NaN 或无穷值写入 PostgreSQL。"""
    cleaned = json.loads(json.dumps(value, ensure_ascii=False, default=str, allow_nan=False))
    return Jsonb(cleaned)


def _decimal(value: Any, field: str, positive: bool = False, optional: bool = False) -> Decimal | None:
    if optional and value is None:
        return None
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{field}必须是有效数字。") from exc
    if not number.is_finite() or (number <= 0 if positive else number < 0):
        raise ValueError(f"{field}超出有效范围。")
    return number


def _shares(value: Any, field: str, optional: bool = False) -> int | None:
    if optional and value is None:
        return None
    number = _decimal(value, field)
    if number != number.to_integral_value() or number <= 0 or number > 2_000_000_000:
        raise ValueError(f"{field}必须是有效的正整数股数。")
    return int(number)


def _timestamp(value: Any) -> datetime:
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError("成交时间无效。") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=SHANGHAI)
    if parsed > datetime.now(SHANGHAI):
        raise ValueError("成交时间不能晚于当前时间。")
    return parsed


def _uuid(value: Any, field: str) -> uuid.UUID:
    try:
        return uuid.UUID(str(value))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{field}必须是有效 UUID。") from exc


class ShortTermStore:
    """通过短线专用表隔离研究结果和用户执行账本。"""

    def __init__(self, repository: QuantRepository) -> None:
        self.repository = repository

    def queue_run(self, user_id: int, horizon: int, run_id: str | None = None) -> str:
        """登记一个后台研究任务；重复启动由唯一 run_id 拦截。"""
        if int(horizon) not in (1, 3, 5):
            raise ValueError("短线研究周期只能为 1、3 或 5 个交易间隔。")
        identifier = _uuid(run_id or uuid.uuid4(), "运行编号")
        with self.repository.connect() as connection:
            connection.execute(
                "INSERT INTO public.quant_short_runs(run_id,strategy_id,strategy_version,horizon,status,created_by) "
                "VALUES (%s,%s,'1.0.0',%s,'queued',%s)",
                (identifier, SHORT_TERM_STRATEGY_ID, int(horizon), int(user_id)),
            )
        return str(identifier)

    def queue_scheduled_run(self, schedule_date: date, horizon: int = 3) -> str | None:
        """按日期与周期幂等登记收盘研究任务，归属当前有效管理员。"""
        if int(horizon) not in (1, 3, 5):
            raise ValueError("短线研究周期只能为 1、3 或 5 个交易间隔。")
        identifier = uuid.uuid4()
        with self.repository.connect() as connection:
            admin = connection.execute(
                "SELECT id FROM public.quant_users WHERE enabled AND is_admin ORDER BY id LIMIT 1"
            ).fetchone()
            if admin is None:
                return None
            connection.execute(
                "INSERT INTO public.quant_short_runs(run_id,strategy_id,strategy_version,horizon,status,created_by) "
                "VALUES (%s,%s,'1.0.0',%s,'queued',%s)",
                (identifier, SHORT_TERM_STRATEGY_ID, int(horizon), int(admin["id"])),
            )
            claim = connection.execute(
                "INSERT INTO public.quant_short_schedule_runs(schedule_date,horizon,run_id) "
                "VALUES (%s,%s,%s) ON CONFLICT(schedule_date,horizon) DO NOTHING RETURNING run_id",
                (schedule_date, int(horizon), identifier),
            ).fetchone()
            if claim is None:
                connection.execute("DELETE FROM public.quant_short_runs WHERE run_id=%s", (identifier,))
                return None
        return str(identifier)

    def mark_run_running(self, run_id: str) -> None:
        """原子领取排队任务，阻止同一运行编号被并发执行。"""
        identifier = _uuid(run_id, "运行编号")
        with self.repository.connect() as connection:
            row = connection.execute(
                "UPDATE public.quant_short_runs SET status='running',started_at=now(),error=NULL "
                "WHERE run_id=%s AND status='queued' RETURNING run_id",
                (identifier,),
            ).fetchone()
            if row is None:
                raise ValueError("研究任务不存在或已被其他进程领取。")

    def finish_run(self, run_id: str, result: dict[str, Any]) -> None:
        """保存研究摘要和冻结预测，不把任务状态改成已验收。"""
        identifier = _uuid(run_id, "运行编号")
        predictions = result.get("predictions") or []
        manifest_path = result.get("manifest_path")
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8")) if manifest_path else {}
        validation = result.get("validation") or {}
        status = result.get("status")
        if status not in ("research_only", "insufficient_data"):
            raise ValueError("研究运行只允许保存为研究态或数据不足。")
        signal_date = date.fromisoformat(result["signal_date"]) if result.get("signal_date") else None
        summary = {key: value for key, value in result.items() if key != "predictions"}
        with self.repository.connect() as connection:
            connection.execute(
                "UPDATE public.quant_short_runs SET status=%s,signal_date=%s,snapshot_path=%s,snapshot_hash=%s, "
                "model_path=%s,model_hash=%s,manifest=%s,validation=%s,result=%s,finished_at=now() "
                "WHERE run_id=%s AND status='running'",
                (status, signal_date, result.get("snapshot_path"), result.get("snapshot_hash"), result.get("model_path"),
                 result.get("model_hash"), _json(manifest), _json(validation), _json(summary), identifier),
            )
            if predictions:
                connection.cursor().executemany(
                    "INSERT INTO public.quant_short_predictions(run_id,code,signal_time,ranking_score,rank,percentile,status,features) "
                    "VALUES (%(run_id)s,%(code)s,%(signal_time)s,%(ranking_score)s,%(rank)s,%(percentile)s,%(status)s,%(features)s) "
                    "ON CONFLICT(run_id,code) DO NOTHING",
                    [
                        {
                            **row,
                            "run_id": identifier,
                            "signal_time": datetime.fromisoformat(row["signal_time"]),
                            "features": _json(row.get("features", {})),
                        }
                        for row in predictions
                    ],
                )

    def fail_run(self, run_id: str, error: str) -> None:
        """记录失败原因，避免后台任务异常后一直显示运行中。"""
        with self.repository.connect() as connection:
            connection.execute(
                "UPDATE public.quant_short_runs SET status='failed',error=%s,finished_at=now() "
                "WHERE run_id=%s AND status IN ('queued','running')",
                (str(error)[:4000], _uuid(run_id, "运行编号")),
            )

    def release_failed_schedule_claim(self, schedule_date: date, horizon: int, run_id: str) -> None:
        """仅在后台进程未能启动时释放调度占位，保留失败运行记录。"""
        if int(horizon) not in (1, 3, 5):
            raise ValueError("短线研究周期只能为 1、3 或 5 个交易间隔。")
        identifier = _uuid(run_id, "运行编号")
        with self.repository.connect() as connection:
            connection.execute(
                "DELETE FROM public.quant_short_schedule_runs schedule "
                "USING public.quant_short_runs run "
                "WHERE schedule.schedule_date=%s AND schedule.horizon=%s AND schedule.run_id=%s "
                "AND run.run_id=schedule.run_id AND run.status='failed'",
                (schedule_date, int(horizon), identifier),
            )

    def recent_runs(self, limit: int = 30) -> list[dict[str, Any]]:
        """读取最新共享研究运行，不返回任何账户数据。"""
        with self.repository.connect() as connection:
            return connection.execute(
                "SELECT run_id,strategy_id,strategy_version,horizon,status,signal_date,snapshot_hash,model_hash, "
                "validation,result,error,created_at,finished_at FROM public.quant_short_runs "
                "ORDER BY created_at DESC LIMIT %s",
                (min(max(int(limit), 1), 100),),
            ).fetchall()

    def predictions(self, run_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """读取单次运行的排序结果。"""
        with self.repository.connect() as connection:
            return connection.execute(
                "SELECT code,signal_time,ranking_score,rank,percentile,status,features FROM public.quant_short_predictions "
                "WHERE run_id=%s ORDER BY rank LIMIT %s",
                (_uuid(run_id, "运行编号"), min(max(int(limit), 1), 1000)),
            ).fetchall()

    def pending_forward_observations(self, limit: int = 10000) -> list[dict[str, Any]]:
        """读取尚无成熟前向观察的预测，按原信号时间顺序处理。"""
        with self.repository.connect() as connection:
            return connection.execute(
                "SELECT p.run_id,p.code,p.signal_time,r.horizon "
                "FROM public.quant_short_predictions p "
                "JOIN public.quant_short_runs r ON r.run_id=p.run_id "
                "LEFT JOIN public.quant_short_forward_observations o "
                "ON o.run_id=p.run_id AND o.code=p.code AND o.horizon=r.horizon "
                "WHERE o.run_id IS NULL AND r.status='research_only' "
                "ORDER BY p.signal_time,p.rank LIMIT %s",
                (min(max(int(limit), 1), 50000),),
            ).fetchall()

    def save_forward_observations(self, rows: list[dict[str, Any]]) -> int:
        """以幂等插入保存已成熟的价格代理观察。"""
        if not rows:
            return 0
        with self.repository.connect() as connection:
            cursor = connection.cursor()
            cursor.executemany(
                "INSERT INTO public.quant_short_forward_observations "
                "(run_id,code,horizon,signal_time,entry_date,entry_open,label_end_date,exit_open,"
                "gross_return,target_net_return,label_basis,source_hash,details) "
                "VALUES (%(run_id)s,%(code)s,%(horizon)s,%(signal_time)s,%(entry_date)s,%(entry_open)s,"
                "%(label_end_date)s,%(exit_open)s,%(gross_return)s,%(target_net_return)s,%(label_basis)s,"
                "%(source_hash)s,%(details)s) ON CONFLICT(run_id,code,horizon) DO NOTHING",
                [
                    {
                        **row,
                        "run_id": _uuid(row["run_id"], "运行编号"),
                        "signal_time": _timestamp(row["signal_time"]),
                        "details": _json(row.get("details", {})),
                    }
                    for row in rows
                ],
            )
            return max(int(cursor.rowcount), 0)

    def forward_observations(self, run_id: str, limit: int = 1000) -> list[dict[str, Any]]:
        """读取单次运行已冻结的前向价格代理观察。"""
        with self.repository.connect() as connection:
            return connection.execute(
                "SELECT code,horizon,signal_time,entry_date,entry_open,label_end_date,exit_open,"
                "gross_return,target_net_return,label_basis,source_hash,details,observed_at "
                "FROM public.quant_short_forward_observations WHERE run_id=%s "
                "ORDER BY signal_time,code LIMIT %s",
                (_uuid(run_id, "运行编号"), min(max(int(limit), 1), 5000)),
            ).fetchall()

    def create_account(
        self,
        user_id: int,
        name: str,
        initial_cash: Any = None,
        max_total_exposure: Any = None,
        max_single_exposure: Any = None,
        cost_profile: dict[str, Any] | None = None,
    ) -> int:
        """创建仅当前登录用户可访问的短线执行账户。"""
        name = str(name).strip()[:80]
        if not name:
            raise ValueError("账户名称不能为空。")
        cash = _decimal(initial_cash, "初始资金", optional=True)
        total_limit = _decimal(max_total_exposure, "总仓位上限", positive=True, optional=True)
        single_limit = _decimal(max_single_exposure, "单股仓位上限", positive=True, optional=True)
        if total_limit is not None and single_limit is not None and single_limit > total_limit:
            raise ValueError("单股仓位上限不能超过总仓位上限。")
        profile = dict(cost_profile or {})
        for key in ("commission_rate", "stamp_duty_rate", "slippage_rate", "minimum_commission"):
            if key in profile:
                value = _decimal(profile[key], key)
                if key.endswith("_rate") and value >= 1:
                    raise ValueError("费率必须小于 100%。")
                profile[key] = str(value)
        with self.repository.connect() as connection:
            row = connection.execute(
                "INSERT INTO public.quant_short_accounts(user_id,name,initial_cash,cash_balance,max_total_exposure, "
                "max_single_exposure,cost_profile) VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (int(user_id), name, cash, cash, total_limit, single_limit, _json(profile)),
            ).fetchone()
            return int(row["id"])

    def accounts(self, user_id: int) -> list[dict[str, Any]]:
        """仅列出当前用户拥有的账户。"""
        with self.repository.connect() as connection:
            return connection.execute(
                "SELECT * FROM public.quant_short_accounts WHERE user_id=%s AND enabled ORDER BY created_at,id",
                (int(user_id),),
            ).fetchall()

    def reconcile_cash_balance(self, user_id: int, account_id: int, cash_balance: Any) -> None:
        """按券商核对结果更正账户可用资金，并记录对账事件。"""
        balance = _decimal(cash_balance, "可用资金")
        event_key = f"cash-reconcile:{account_id}:{uuid.uuid4()}"
        with self.repository.connect() as connection:
            account = connection.execute(
                "SELECT * FROM public.quant_short_accounts WHERE id=%s AND user_id=%s AND enabled FOR UPDATE",
                (int(account_id), int(user_id)),
            ).fetchone()
            if account is None:
                raise ValueError("账户不存在或无权访问。")
            connection.execute(
                "UPDATE public.quant_short_accounts SET cash_balance=%s,updated_at=now() WHERE id=%s",
                (balance, int(account_id)),
            )
            connection.execute(
                "INSERT INTO public.quant_short_events(user_id,account_id,strategy_id,code,event_type,event_status,event_key,details) "
                "VALUES (%s,%s,%s,'000000','cash_reconciliation','reconciled',%s,%s)",
                (int(user_id), int(account_id), SHORT_TERM_STRATEGY_ID, event_key,
                 _json({"before": str(account["cash_balance"]) if account["cash_balance"] is not None else None,
                        "after": str(balance), "source": "用户核对券商账户"})),
            )

    def create_manual_intent(
        self,
        user_id: int,
        account_id: int,
        code: str,
        side: str,
        trigger_rule: str,
        idempotency_key: str,
        strategy_id: str = SHORT_TERM_STRATEGY_ID,
        run_id: str | None = None,
        reference_price: Any = None,
        limit_price: Any = None,
        intended_shares: Any = None,
        valid_until: Any = None,
        details: dict[str, Any] | None = None,
    ) -> int:
        """创建用户确认的待处理任务，不由研究排名自动下单。"""
        code = stock_code(code)
        side = str(side).upper()
        if side not in ("BUY", "SELL"):
            raise ValueError("任务方向只能是买入或卖出。")
        if not str(trigger_rule).strip():
            raise ValueError("必须记录触发规则或人工决策原因。")
        reference = _decimal(reference_price, "参考价格", positive=True, optional=True)
        limit = _decimal(limit_price, "限价", positive=True, optional=True)
        quantity = _shares(intended_shares, "计划股数", optional=True)
        expires = _timestamp(valid_until) if valid_until else None
        key = _uuid(idempotency_key, "幂等键")
        resolved_run_id = _uuid(run_id, "运行编号") if run_id else None
        metadata = dict(details or {})
        target_position_id = int(metadata["position_id"]) if metadata.get("position_id") is not None else None
        with self.repository.connect() as connection:
            account = connection.execute(
                "SELECT * FROM public.quant_short_accounts WHERE id=%s AND user_id=%s AND enabled FOR UPDATE",
                (int(account_id), int(user_id)),
            ).fetchone()
            if account is None:
                raise ValueError("账户不存在或无权访问。")
            if side == "BUY" and resolved_run_id is not None:
                run = connection.execute(
                    "SELECT status,result->'promotion_gate'->>'status' AS promotion_status "
                    "FROM public.quant_short_runs WHERE run_id=%s",
                    (resolved_run_id,),
                ).fetchone()
                if run is None:
                    raise ValueError("关联的短线研究运行不存在。")
                if run["status"] != "research_only" or run["promotion_status"] != "eligible_for_review":
                    raise ValueError("该候选仍处于研究态或未达到冻结验收门槛，不能关联创建买入任务。")
            if side == "BUY" and metadata.get("m3_enabled"):
                if limit is None or quantity is None:
                    raise ValueError("启用分钟确认时必须填写限价和计划股数。")
                cost_profile = account["cost_profile"] or {}
                required_costs = ("commission_rate", "minimum_commission", "stamp_duty_rate", "slippage_rate")
                if any(cost_profile.get(field) is None for field in required_costs):
                    raise ValueError("启用分钟确认前必须填写完整的账户费率配置。")
                metadata["cost_profile_snapshot"] = dict(cost_profile)
            existing = connection.execute(
                "SELECT id FROM public.quant_short_intents WHERE idempotency_key=%s AND user_id=%s AND account_id=%s",
                (key, int(user_id), int(account_id)),
            ).fetchone()
            if existing:
                return int(existing["id"])
            if resolved_run_id is not None:
                existing_signal = connection.execute(
                    "SELECT id FROM public.quant_short_intents WHERE user_id=%s AND account_id=%s AND run_id=%s "
                    "AND code=%s AND side=%s AND status NOT IN ('cancelled','expired','rejected') LIMIT 1",
                    (int(user_id), int(account_id), resolved_run_id, code, side),
                ).fetchone()
                if existing_signal:
                    raise ValueError("该运行信号已经关联任务；请查看原任务或使用新的研究运行。")
            if side == "BUY" and quantity is not None:
                if account["cash_balance"] is None or limit is None:
                    raise ValueError("账户资金或限价未确认，不得创建带股数的买入任务。")
                rates = account["cost_profile"] or {}
                if any(rates.get(field) is None for field in ("commission_rate", "slippage_rate", "minimum_commission")):
                    raise ValueError("买入任务带股数时，必须先录入账户佣金、最低佣金和滑点费率。")
                notional = limit * quantity
                commission = max(
                    notional * Decimal(str(rates["commission_rate"])),
                    Decimal(str(rates["minimum_commission"])),
                )
                estimated_cost = commission + notional * Decimal(str(rates["slippage_rate"]))
                pending_buys = connection.execute(
                    "SELECT intent.intended_shares, intent.limit_price, COALESCE(sum(fill.shares),0) AS filled_shares "
                    "FROM public.quant_short_intents intent LEFT JOIN public.quant_short_fills fill ON fill.intent_id=intent.id "
                    "WHERE intent.user_id=%s AND intent.account_id=%s AND intent.side='BUY' "
                    "AND intent.status=ANY(%s) AND (intent.valid_until IS NULL OR intent.valid_until>now()) "
                    "AND intent.intended_shares IS NOT NULL AND intent.limit_price IS NOT NULL "
                    "GROUP BY intent.id",
                    (int(user_id), int(account_id), list(ACTIVE_INTENT_STATUSES)),
                ).fetchall()
                reserved_cash = Decimal("0")
                reserved_exposure = Decimal("0")
                for pending in pending_buys:
                    remaining_shares = max(int(pending["intended_shares"]) - int(pending["filled_shares"]), 0)
                    if remaining_shares == 0:
                        continue
                    remaining_notional = Decimal(str(pending["limit_price"])) * remaining_shares
                    remaining_commission = max(
                        remaining_notional * Decimal(str(rates["commission_rate"])),
                        Decimal(str(rates["minimum_commission"])),
                    )
                    reserved_cash += remaining_notional + remaining_commission
                    reserved_cash += remaining_notional * Decimal(str(rates["slippage_rate"]))
                    reserved_exposure += remaining_notional
                if account["cash_balance"] - reserved_cash < notional + estimated_cost:
                    raise ValueError("买入金额超过账户可用资金。")
                if account["max_single_exposure"] is not None and notional > account["max_single_exposure"]:
                    raise ValueError("买入金额超过单股仓位上限。")
                current_exposure = connection.execute(
                    "SELECT COALESCE(sum(remaining_shares * average_buy_price),0) AS exposure "
                    "FROM public.quant_short_positions WHERE user_id=%s AND account_id=%s AND status<>'closed'",
                    (int(user_id), int(account_id)),
                ).fetchone()["exposure"]
                if account["max_total_exposure"] is not None and current_exposure + reserved_exposure + notional > account["max_total_exposure"]:
                    raise ValueError("买入金额会超过账户总仓位上限。")
            if side == "SELL":
                if target_position_id is None:
                    raise ValueError("卖出任务必须指定所属持仓批次。")
                position = connection.execute(
                    "SELECT * FROM public.quant_short_positions WHERE id=%s AND user_id=%s AND account_id=%s "
                    "AND code=%s AND status<>'closed' FOR UPDATE",
                    (target_position_id, int(user_id), int(account_id), code),
                ).fetchone()
                if position is None:
                    raise ValueError("指定的卖出批次不存在或不属于当前账户。")
                active_sell_intents = connection.execute(
                    "SELECT intent.intended_shares,COALESCE(sum(fill.shares),0) AS filled_shares "
                    "FROM public.quant_short_intents intent "
                    "LEFT JOIN public.quant_short_fills fill ON fill.intent_id=intent.id "
                    "WHERE intent.user_id=%s AND intent.account_id=%s AND intent.side='SELL' "
                    "AND intent.status=ANY(%s) AND intent.details->>'position_id'=%s "
                    "GROUP BY intent.id",
                    (int(user_id), int(account_id), list(ACTIVE_INTENT_STATUSES), str(target_position_id)),
                ).fetchall()
                reserved_shares = sum(
                    max(int(intent["intended_shares"]) - int(intent["filled_shares"]), 0)
                    for intent in active_sell_intents
                    if intent["intended_shares"] is not None
                )
                available_shares = int(position["available_shares"])
                if quantity is not None and quantity > available_shares - reserved_shares:
                    raise ValueError("计划卖出股数超过扣除其他待处理卖出任务后的可卖数量。")
                metadata["strategy_id"] = position["strategy_id"]
                strategy_id = position["strategy_id"]
            row = connection.execute(
                "INSERT INTO public.quant_short_intents(user_id,account_id,strategy_id,run_id,code,side,status,trigger_rule, "
                "reference_price,limit_price,intended_shares,valid_until,idempotency_key,details) "
                "VALUES (%s,%s,%s,%s,%s,%s,'pending',%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (int(user_id), int(account_id), str(strategy_id)[:80], resolved_run_id,
                 code, side, str(trigger_rule)[:1000], reference, limit, quantity, expires, key, _json(metadata)),
            ).fetchone()
            intent_id = int(row["id"])
            if side == "SELL":
                connection.execute(
                    "UPDATE public.quant_short_positions SET status='pending_exit',exit_reason=%s,updated_at=now() "
                    "WHERE id=%s AND status<>'closed'",
                    (str(trigger_rule)[:500], target_position_id),
                )
            connection.execute(
                "INSERT INTO public.quant_short_events(user_id,account_id,strategy_id,code,intent_id,event_type,event_status,event_key,details) "
                "VALUES (%s,%s,%s,%s,%s,'intent','pending',%s,%s)",
                (int(user_id), int(account_id), str(strategy_id)[:80], code, intent_id, f"intent:{key}", _json({"side": side, "trigger_rule": trigger_rule})),
            )
            conflicts = connection.execute(
                "SELECT id,strategy_id,side FROM public.quant_short_intents WHERE user_id=%s AND account_id=%s AND code=%s "
                "AND id<>%s AND strategy_id<>%s AND side<>%s AND status=ANY(%s)",
                (int(user_id), int(account_id), code, intent_id, str(strategy_id), side, list(ACTIVE_INTENT_STATUSES)),
            ).fetchall()
            for conflict in conflicts:
                conflict_ids = sorted([intent_id, int(conflict["id"])])
                connection.execute(
                    "INSERT INTO public.quant_short_events(user_id,account_id,strategy_id,code,intent_id,event_type,event_status,event_key,details) "
                    "VALUES (%s,%s,%s,%s,%s,'strategy_conflict','open',%s,%s) ON CONFLICT(event_key) DO NOTHING",
                    (int(user_id), int(account_id), str(strategy_id)[:80], code, intent_id,
                     f"conflict:{account_id}:{conflict_ids[0]}:{conflict_ids[1]}",
                     _json({"other_intent_id": int(conflict["id"]), "other_strategy_id": conflict["strategy_id"], "other_side": conflict["side"]})),
                )
            return intent_id

    def cancel_intent(self, user_id: int, account_id: int, intent_id: int, reason: str = "用户取消") -> None:
        """取消尚未完成的任务并保留事件记录。"""
        with self.repository.connect() as connection:
            row = connection.execute(
                "UPDATE public.quant_short_intents SET status='cancelled',updated_at=now() "
                "WHERE id=%s AND user_id=%s AND account_id=%s AND status=ANY(%s) RETURNING *",
                (int(intent_id), int(user_id), int(account_id), list(ACTIVE_INTENT_STATUSES)),
            ).fetchone()
            if row is None:
                raise ValueError("任务不存在、已完成或无权取消。")
            connection.execute(
                "INSERT INTO public.quant_short_events(user_id,account_id,strategy_id,code,intent_id,event_type,event_status,event_key,details) "
                "VALUES (%s,%s,%s,%s,%s,'intent','cancelled',%s,%s)",
                (int(user_id), int(account_id), row["strategy_id"], row["code"], row["id"], f"intent-cancel:{row['id']}", _json({"reason": reason[:1000]})),
            )
            self._resolve_intent_conflicts(connection, int(user_id), int(account_id), int(intent_id), "任务已取消")

    def intents(self, user_id: int, account_id: int, limit: int = 100) -> list[dict[str, Any]]:
        """读取指定账户的任务，不跨用户返回。"""
        with self.repository.connect() as connection:
            return connection.execute(
                "SELECT * FROM public.quant_short_intents WHERE user_id=%s AND account_id=%s ORDER BY created_at DESC,id DESC LIMIT %s",
                (int(user_id), int(account_id), min(max(int(limit), 1), 500)),
            ).fetchall()

    def positions(self, user_id: int, account_id: int, include_closed: bool = False) -> list[dict[str, Any]]:
        """读取指定账户的策略批次持仓。"""
        with self.repository.connect() as connection:
            return connection.execute(
                "SELECT * FROM public.quant_short_positions WHERE user_id=%s AND account_id=%s "
                + ("" if include_closed else "AND status<>'closed' ")
                + "ORDER BY code,buy_date,id",
                (int(user_id), int(account_id)),
            ).fetchall()

    def monitor_positions(self) -> list[dict[str, Any]]:
        """读取后台风险监控需要的批次及所属账户，不返回给前端共享展示。"""
        with self.repository.connect() as connection:
            return connection.execute(
                "SELECT * FROM public.quant_short_positions WHERE remaining_shares>0 AND status<>'closed' "
                "ORDER BY user_id,account_id,code,id"
            ).fetchall()

    def update_high_watermarks(self, user_id: int, account_id: int, quotes: dict[str, dict[str, Any]]) -> None:
        """用已校验的新鲜报价持久化持仓批次高点。"""
        if not quotes:
            return
        with self.repository.connect() as connection:
            for code, quote in quotes.items():
                price = _decimal(quote.get("price"), "最新报价", positive=True)
                connection.execute(
                    "UPDATE public.quant_short_positions SET high_watermark_price=GREATEST("
                    "COALESCE(high_watermark_price,average_buy_price),%s),updated_at=now() "
                    "WHERE user_id=%s AND account_id=%s AND code=%s AND remaining_shares>0 AND status<>'closed'",
                    (price, int(user_id), int(account_id), stock_code(code)),
                )

    def monitor_codes(self) -> list[str]:
        """读取所有用户仍持有的短线股票代码，合并后供共享行情订阅。"""
        with self.repository.connect() as connection:
            rows = connection.execute(
                "SELECT code FROM public.quant_short_positions WHERE remaining_shares>0 AND status<>'closed' "
                "UNION SELECT code FROM public.quant_short_intents WHERE status='pending' AND "
                "(limit_price IS NOT NULL OR details->>'m3_enabled'='true') ORDER BY code"
            ).fetchall()
        return [row["code"] for row in rows]

    def monitor_intents(self) -> list[dict[str, Any]]:
        """读取待监控任务及账户归属，供后台按用户隔离触发提醒。"""
        with self.repository.connect() as connection:
            return connection.execute(
                "SELECT * FROM public.quant_short_intents WHERE status='pending' ORDER BY user_id,account_id,id"
            ).fetchall()

    def save_intent_monitor_events(self, user_id: int, account_id: int, events: list[dict[str, Any]]) -> None:
        """持久化限价触发、行情缺失与到期事件，并推进任务状态。"""
        if not events:
            return
        with self.repository.connect() as connection:
            for event in events:
                intent_id = int(event["intent_id"])
                intent = connection.execute(
                    "SELECT * FROM public.quant_short_intents WHERE id=%s AND user_id=%s AND account_id=%s FOR UPDATE",
                    (intent_id, int(user_id), int(account_id)),
                ).fetchone()
                if intent is None:
                    continue
                status = event["status"]
                if status in ("triggered", "triggered_no_quantity", "expired"):
                    next_status = "expired" if status == "expired" else "triggered"
                    connection.execute(
                        "UPDATE public.quant_short_intents SET status=%s,updated_at=now() WHERE id=%s AND status='pending'",
                        (next_status, intent_id),
                    )
                event_key = f"monitor-intent:{account_id}:{intent_id}:{event['type']}:{status}"
                connection.execute(
                    "INSERT INTO public.quant_short_events(user_id,account_id,strategy_id,code,intent_id,event_type,event_status,event_key,details) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(event_key) DO NOTHING",
                    (int(user_id), int(account_id), intent["strategy_id"], intent["code"], intent_id,
                     str(event["type"])[:40], str(status)[:24], event_key, _json(event)),
                )

    def save_monitor_events(self, user_id: int, account_id: int, events: list[dict[str, Any]]) -> None:
        """持久化基础风控事件；重复状态去重，风险升级保留独立事件。"""
        if not events:
            return
        with self.repository.connect() as connection:
            for event in events:
                position_id = int(event["position_id"])
                position = connection.execute(
                    "SELECT * FROM public.quant_short_positions WHERE id=%s AND user_id=%s AND account_id=%s FOR UPDATE",
                    (position_id, int(user_id), int(account_id)),
                ).fetchone()
                if position is None:
                    continue
                if event["type"] == "exit_trigger":
                    connection.execute(
                        "UPDATE public.quant_short_positions SET status='pending_exit',exit_reason=%s,updated_at=now() "
                        "WHERE id=%s AND status<>'closed'",
                        (str(event["rule"])[:500], position_id),
                    )
                event_key = f"monitor:{account_id}:{position_id}:{event['type']}:{event['rule']}:{event['status']}"
                connection.execute(
                    "INSERT INTO public.quant_short_events(user_id,account_id,strategy_id,code,position_id,event_type,event_status,event_key,details) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(event_key) DO NOTHING",
                    (int(user_id), int(account_id), position["strategy_id"], position["code"], position_id,
                     str(event["type"])[:40], str(event["status"])[:24], event_key, _json(event)),
                )

    def reconcile_available_shares(self, user_id: int, account_id: int, position_id: int, available_shares: Any) -> None:
        """按券商账户核对结果更新某一持仓批次的可卖数量。"""
        quantity = _shares(available_shares, "可卖股数", optional=True)
        quantity = int(quantity or 0)
        with self.repository.connect() as connection:
            row = connection.execute(
                "SELECT * FROM public.quant_short_positions WHERE id=%s AND user_id=%s AND account_id=%s FOR UPDATE",
                (int(position_id), int(user_id), int(account_id)),
            ).fetchone()
            if row is None or quantity > int(row["remaining_shares"]):
                raise ValueError("持仓批次不存在，或核对的可卖数超过剩余持仓。")
            connection.execute(
                "UPDATE public.quant_short_positions SET available_shares=%s,updated_at=now() WHERE id=%s",
                (quantity, int(position_id)),
            )
            key = f"available:{position_id}:{quantity}:{datetime.now(SHANGHAI).strftime('%Y%m%d%H%M')}"
            connection.execute(
                "INSERT INTO public.quant_short_events(user_id,account_id,strategy_id,code,position_id,event_type,event_status,event_key,details) "
                "VALUES (%s,%s,%s,%s,%s,'available_shares','reconciled',%s,%s)",
                (int(user_id), int(account_id), row["strategy_id"], row["code"], int(position_id), key,
                 _json({"available_shares": quantity, "source": "用户核对券商账户"})),
            )

    def record_fill(
        self,
        user_id: int,
        account_id: int,
        code: str,
        side: str,
        traded_at: Any,
        price: Any,
        shares: Any,
        fee: Any,
        idempotency_key: str,
        position_id: int | None = None,
        intent_id: int | None = None,
        external_trade_id: str | None = None,
        name: str = "",
        strategy_id: str = SHORT_TERM_STRATEGY_ID,
    ) -> int:
        """依据真实成交记录更新现金和持仓；重复回填以幂等键拒绝。"""
        code = stock_code(code)
        side = str(side).upper()
        if side not in ("BUY", "SELL"):
            raise ValueError("成交方向只能是买入或卖出。")
        when = _timestamp(traded_at)
        fill_price = _decimal(price, "成交价格", positive=True)
        quantity = _shares(shares, "成交股数")
        actual_fee = _decimal(fee, "实际费用")
        key = _uuid(idempotency_key, "成交幂等键")
        external_id = str(external_trade_id).strip()[:160] if external_trade_id else None
        with self.repository.connect() as connection:
            existing = connection.execute(
                "SELECT id FROM public.quant_short_fills WHERE user_id=%s AND account_id=%s AND (idempotency_key=%s OR (%s IS NOT NULL AND external_trade_id=%s)) LIMIT 1",
                (int(user_id), int(account_id), key, external_id, external_id),
            ).fetchone()
            if existing:
                return int(existing["id"])
            account = connection.execute(
                "SELECT * FROM public.quant_short_accounts WHERE id=%s AND user_id=%s AND enabled FOR UPDATE",
                (int(account_id), int(user_id)),
            ).fetchone()
            if account is None:
                raise ValueError("账户不存在或无权访问。")
            intent = None
            prior_filled = 0
            if intent_id is not None:
                intent = connection.execute(
                    "SELECT * FROM public.quant_short_intents WHERE id=%s AND user_id=%s AND account_id=%s FOR UPDATE",
                    (int(intent_id), int(user_id), int(account_id)),
                ).fetchone()
                if intent is None or intent["code"] != code or intent["side"] != side:
                    raise ValueError("成交与任务的账户、股票或方向不一致。")
                strategy_id = intent["strategy_id"]
                if intent["intended_shares"] is not None:
                    prior_filled = int(connection.execute(
                        "SELECT COALESCE(sum(shares),0) AS shares FROM public.quant_short_fills WHERE intent_id=%s",
                        (int(intent_id),),
                    ).fetchone()["shares"])
                    if prior_filled + quantity > int(intent["intended_shares"]):
                        raise ValueError("累计成交股数超过任务计划股数。")
            notional = fill_price * quantity
            if side == "BUY":
                reserved_rows = connection.execute(
                    "SELECT intent.id,intent.intended_shares,intent.limit_price,COALESCE(sum(fill.shares),0) AS filled_shares "
                    "FROM public.quant_short_intents intent LEFT JOIN public.quant_short_fills fill ON fill.intent_id=intent.id "
                    "WHERE intent.user_id=%s AND intent.account_id=%s AND intent.side='BUY' AND intent.status=ANY(%s) "
                    "AND (intent.valid_until IS NULL OR intent.valid_until>now()) AND intent.intended_shares IS NOT NULL "
                    "AND intent.limit_price IS NOT NULL AND (%s IS NULL OR intent.id<>%s) GROUP BY intent.id",
                    (int(user_id), int(account_id), list(ACTIVE_INTENT_STATUSES), int(intent_id) if intent_id is not None else None,
                     int(intent_id) if intent_id is not None else None),
                ).fetchall()
                policy_costs = account["cost_profile"] or {}
                pending_remaining = int(intent["intended_shares"]) - prior_filled - quantity if (
                    intent is not None and intent["side"] == "BUY" and intent["status"] in ACTIVE_INTENT_STATUSES
                    and intent["intended_shares"] is not None
                ) else 0
                if (reserved_rows or pending_remaining > 0) and any(
                    policy_costs.get(field) is None for field in ("commission_rate", "minimum_commission", "slippage_rate")
                ):
                    raise ValueError("待处理买入预留费用无法核算；请先补齐账户费率。")
                reserved_cash = Decimal("0")
                for reserved in reserved_rows:
                    remaining_shares = max(int(reserved["intended_shares"]) - int(reserved["filled_shares"]), 0)
                    if remaining_shares <= 0:
                        continue
                    reserved_notional = Decimal(str(reserved["limit_price"])) * remaining_shares
                    reserved_commission = max(
                        reserved_notional * Decimal(str(policy_costs["commission_rate"])),
                        Decimal(str(policy_costs["minimum_commission"])),
                    )
                    reserved_cash += reserved_notional + reserved_commission
                    reserved_cash += reserved_notional * Decimal(str(policy_costs["slippage_rate"]))
                if pending_remaining > 0:
                    remaining_notional = Decimal(str(intent["limit_price"])) * pending_remaining
                    remaining_commission = max(
                        remaining_notional * Decimal(str(policy_costs["commission_rate"])),
                        Decimal(str(policy_costs["minimum_commission"])),
                    )
                    reserved_cash += remaining_notional + remaining_commission
                    reserved_cash += remaining_notional * Decimal(str(policy_costs["slippage_rate"]))
                if account["cash_balance"] is not None and account["cash_balance"] < notional + actual_fee + reserved_cash:
                    raise ValueError("买入成交会占用其他待处理买入已预留资金。")
                policy = (intent["details"] or {}) if intent else {}
                planned_exit = date.fromisoformat(str(policy["planned_exit_date"])[:10]) if policy.get("planned_exit_date") else None
                stop_loss = _decimal(policy.get("stop_loss_price"), "止损价", positive=True, optional=True)
                take_profit = _decimal(policy.get("take_profit_price"), "止盈价", positive=True, optional=True)
                trailing_stop_pct = _decimal(policy.get("trailing_stop_pct"), "移动止盈比例", positive=True, optional=True)
                invalidation_price = _decimal(policy.get("strategy_invalidation_price"), "策略失效价", positive=True, optional=True)
                position = connection.execute(
                    "INSERT INTO public.quant_short_positions(user_id,account_id,strategy_id,entry_intent_id,code,name,buy_date, "
                    "average_buy_price,original_shares,remaining_shares,available_shares,buy_fees,planned_exit_date,stop_loss_price, "
                    "take_profit_price,trailing_stop_pct,strategy_invalidation_price,high_watermark_price) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,0,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                    (int(user_id), int(account_id), str(strategy_id)[:80], int(intent_id) if intent else None, code, str(name)[:80],
                     when.date(), fill_price, quantity, quantity, actual_fee, planned_exit, stop_loss, take_profit,
                     trailing_stop_pct, invalidation_price, fill_price),
                ).fetchone()
                resolved_position_id = int(position["id"])
            else:
                if position_id is None:
                    raise ValueError("卖出成交必须关联实际持仓批次。")
                position = connection.execute(
                    "SELECT * FROM public.quant_short_positions WHERE id=%s AND user_id=%s AND account_id=%s AND code=%s "
                    "AND status<>'closed' FOR UPDATE",
                    (int(position_id), int(user_id), int(account_id), code),
                ).fetchone()
                if position is None or quantity > int(position["available_shares"]):
                    raise ValueError("卖出股数超过该批次已核对的可卖数量。")
                strategy_id = position["strategy_id"]
                remaining = int(position["remaining_shares"]) - quantity
                available = int(position["available_shares"]) - quantity
                connection.execute(
                    "UPDATE public.quant_short_positions SET remaining_shares=%s,available_shares=%s, "
                    "status=%s,exit_reason=CASE WHEN %s=0 THEN '成交回填平仓' ELSE exit_reason END,updated_at=now() WHERE id=%s",
                    (
                        remaining,
                        available,
                        "closed" if remaining == 0 else ("pending_exit" if intent else position["status"]),
                        remaining,
                        int(position_id),
                    ),
                )
                resolved_position_id = int(position_id)
                if actual_fee > notional:
                    raise ValueError("卖出费用不能超过成交金额。")
            fill = connection.execute(
                "INSERT INTO public.quant_short_fills(user_id,account_id,intent_id,position_id,code,side,traded_at,price,shares,fee,external_trade_id,idempotency_key) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (int(user_id), int(account_id), int(intent_id) if intent else None, resolved_position_id, code, side,
                 when, fill_price, quantity, actual_fee, external_id, key),
            ).fetchone()
            if account["cash_balance"] is not None:
                cash_delta = -(notional + actual_fee) if side == "BUY" else notional - actual_fee
                connection.execute(
                    "UPDATE public.quant_short_accounts SET cash_balance=cash_balance+%s,updated_at=now() WHERE id=%s",
                    (cash_delta, int(account_id)),
                )
            if intent:
                filled = int(connection.execute(
                    "SELECT COALESCE(sum(shares),0) AS shares FROM public.quant_short_fills WHERE intent_id=%s",
                    (int(intent_id),),
                ).fetchone()["shares"])
                intent_status = "filled" if intent["intended_shares"] is None or filled >= int(intent["intended_shares"]) else "partially_filled"
                connection.execute(
                    "UPDATE public.quant_short_intents SET status=%s,updated_at=now() WHERE id=%s",
                    (intent_status, int(intent_id)),
                )
                if intent_status == "filled":
                    self._resolve_intent_conflicts(connection, int(user_id), int(account_id), int(intent_id), "任务已完成成交")
            connection.execute(
                "INSERT INTO public.quant_short_events(user_id,account_id,strategy_id,code,position_id,intent_id,event_type,event_status,event_key,details) "
                "VALUES (%s,%s,%s,%s,%s,%s,'fill','recorded',%s,%s)",
                (int(user_id), int(account_id), str(strategy_id)[:80], code, resolved_position_id,
                 int(intent_id) if intent else None, f"fill:{key}",
                 _json({"side": side, "traded_at": when.isoformat(), "price": str(fill_price), "shares": quantity, "fee": str(actual_fee)})),
            )
            return int(fill["id"])

    @staticmethod
    def _resolve_intent_conflicts(connection: Any, user_id: int, account_id: int, intent_id: int, reason: str) -> None:
        """将不再同时有效的相反任务冲突标记为已处理。"""
        connection.execute(
            "UPDATE public.quant_short_events SET event_status='resolved', "
            "details=details || jsonb_build_object('resolution',%s,'resolved_at',now()) "
            "WHERE user_id=%s AND account_id=%s AND event_type='strategy_conflict' AND event_status='open' "
            "AND (intent_id=%s OR details->>'other_intent_id'=%s)",
            (str(reason)[:500], int(user_id), int(account_id), int(intent_id), str(int(intent_id))),
        )

    def events(self, user_id: int, account_id: int, limit: int = 200) -> list[dict[str, Any]]:
        """读取当前用户账户的最新风险、冲突及成交事件。"""
        with self.repository.connect() as connection:
            return connection.execute(
                "SELECT * FROM public.quant_short_events WHERE user_id=%s AND account_id=%s ORDER BY created_at DESC,id DESC LIMIT %s",
                (int(user_id), int(account_id), min(max(int(limit), 1), 1000)),
            ).fetchall()
