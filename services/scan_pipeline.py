"""无界面的扫描编排：网页端与每日定时任务共用同一条代码路径。

这段编排原先内联在 pages/2_量化选股.py 里，只有点按钮的人才能触发。定时任务需要
同一套流程，若在别处再抄一份，两边迟早各改一半——本仓库刚因为创业板号段在十处
各写一份而翻过车。因此把编排收在这里，页面退化成收集输入、渲染输出的薄壳。

本模块刻意不引入 Streamlit：scan_daily.py 要在没有网页的进程里跑同一套逻辑。
tests/test_scan_pipeline.py 会用一条断言把这个约束钉住。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from config import settings
from services.backtest import run_backtest
from services.data_cleaning import price_limit_rate
from services.exposures import load_exposures
from services.extra_factors import build_extra_factors
from services.market_data import MarketDataError, completed_daily_history, history_is_current
from services.ml_pipeline import load_context, train_market_context
from services.portfolio_backtest import run_portfolio_strategy
from services.rank_model import objective_label
from services.scanning import candidate_codes, rank_candidates, validated_return
from services.stock_scoring import analyze_stock

LOGGER = logging.getLogger("quant.scan_pipeline")
SHANGHAI = ZoneInfo("Asia/Shanghai")

# 股票池范围标签的唯一出处。services/scanning.py 用 .get(mode, "all") 兜底，
# 写错一个标签会静默退化成全市场扫描，而不是报错，因此这里必须做一次校验。
# 全市场是要落盘、给所有账号共用的那一个口径，单独起个名字，免得页面靠下标取它。
FULL_MARKET_POOL_MODE = "沪深京全部 A 股"
POOL_MODES = (FULL_MARKET_POOL_MODE, "沪深主板", "创业板", "科创板", "北交所", "用户自定义")

ProgressCallback = Callable[[str, int | None, int | None], None]


def _zone_text(zone: Any) -> str:
    """买入区间是元组，直接用会连括号和引号一起显示，这里转成「下沿 ~ 上沿」。"""
    if not zone:
        return "—"
    try:
        return f"{float(zone[0]):.2f} ~ {float(zone[1]):.2f}"
    except (TypeError, ValueError, IndexError):
        return "—"


@dataclass(frozen=True)
class ScanOptions:
    """一次扫描的全部输入，字段与页面筛选表单一一对应。"""

    pool_mode: str
    custom_codes: str = ""
    scan_limit: int = 20
    objective_mode: str = settings.DEFAULT_OBJECTIVE_MODE
    training_window: int = settings.TRAINING_WINDOW_DAYS
    enable_neutralization: bool = True
    use_extra_factors: bool = True
    run_portfolio: bool = False
    portfolio_top_n: int = settings.PORTFOLIO_TOP_N
    max_position_weight: float = settings.MAX_POSITION_WEIGHT
    rebalance_every: int = 1
    # 逐股回测在全市场下不可接受，默认关闭，只有手动扫描时才可能打开。
    include_backtest: bool = False
    trigger: str = "manual"  # "manual"（管理员手动）| "scheduled"（每日定时任务）

    def __post_init__(self) -> None:
        if self.pool_mode not in POOL_MODES:
            raise ValueError(f"未知的股票范围 {self.pool_mode!r}；可选：{'、'.join(POOL_MODES)}")


@dataclass
class ScanResult:
    """扫描产物。页面只负责把它渲染出来，不再自己拼字段。"""

    options: ScanOptions
    source_meta: dict[str, Any] = field(default_factory=dict)
    scan_stats: dict[str, int] = field(default_factory=dict)
    rows: list[dict[str, Any]] = field(default_factory=list)
    detail_rows: list[dict[str, Any]] = field(default_factory=list)
    failed_rows: list[dict[str, Any]] = field(default_factory=list)
    usable: list[dict[str, str]] = field(default_factory=list)
    notes: list[dict[str, str]] = field(default_factory=list)
    recommendations: list[dict[str, Any]] = field(default_factory=list)
    portfolio_result: dict[str, Any] | None = None
    context_id: str | None = None
    reference_date: str | None = None
    finished_at: str = ""

    def note(self, level: str, text: str) -> None:
        """记一条要显示给用户的数据血缘说明，level 取 Streamlit 的展示通道名。"""
        self.notes.append({"level": level, "text": text})


def _report(progress: ProgressCallback | None, message: str, completed: int | None = None, total: int | None = None) -> None:
    if progress is not None:
        progress(message, completed, total)


def _message_only(progress: ProgressCallback | None) -> Callable[[str], None] | None:
    """把三参数进度回调适配成子服务需要的单参数回调。

    train_market_context 与 run_portfolio_strategy 的 progress 只吃一个 message；
    页面直接传 task.step 能工作，靠的是后两个参数有默认值。这里把这个巧合显式化。
    """
    if progress is None:
        return None
    return lambda message: progress(message, None, None)


def collect_candidates(
    service: Any, options: ScanOptions, progress: ProgressCallback | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """读取股票池候选；空池时返回的 source_meta 里带 error，由调用方决定怎么展示。"""
    _report(progress, "正在读取缓存或逐页获取股票目录，并核对完整性…")
    return candidate_codes(options.pool_mode, options.custom_codes, int(options.scan_limit), service, progress)


def _fetch_benchmark(service: Any, result: ScanResult, progress: ProgressCallback | None) -> pd.DataFrame | None:
    """沪深 300 基准。取不到时返回 None 并如实说明，不编造相对市场因子。"""
    _report(progress, "正在准备沪深 300 基准行情…")
    try:
        benchmark, benchmark_meta = service.get_history(
            settings.MARKET_INDEX_CODE, count=1200, adjusted=True, market_id=1
        )
        if "过期" in benchmark_meta.get("status", ""):
            raise MarketDataError("沪深300历史已过期，不能训练超额收益标签。")
        return completed_daily_history(benchmark)
    except MarketDataError as exc:
        result.note("warning", f"沪深 300 指数不可用；相对市场因子未计算：{exc}")
        return None


def _collect_one(service: Any, candidate: dict[str, Any], source_meta: dict[str, Any]) -> dict[str, Any]:
    """采集单只股票并做进入截面的前置判断。

    每一层判断都对应一种"看起来像数据、其实不可用"的情况，宁可如实排除，
    也不让停牌、ST 或未更新的日线混进截面参与训练。
    """
    code = candidate["code"]
    quote = service.get_scan_quote(candidate, source_meta)
    if not quote.get("quote_valid", True):
        return {"code": code, "name": candidate.get("name", code), "error": "当前仅有过期行情缓存，未生成模型预测"}
    name = quote.get("name", candidate.get("name", code))
    if re.search(r"ST|退", str(name), flags=re.IGNORECASE):
        return {"code": code, "name": name, "error": "ST 或退市整理股票已排除"}
    volume = quote.get("volume_lots")
    if volume is None or volume <= 0:
        return {"code": code, "name": name, "error": "停牌或无成交量，已排除"}
    history, history_meta = service.get_history(code, count=1200, adjusted=True)
    history = completed_daily_history(history)
    if not history_is_current(history, quote, history_meta):
        return {"code": code, "name": name, "error": "历史行情未更新至报价对应的完整交易日，未生成模型预测"}
    if len(history) < settings.MINIMUM_SCAN_HISTORY_BARS:
        return {
            "code": code, "name": name,
            "error": f"历史样本不足 {settings.MINIMUM_SCAN_HISTORY_BARS} 根日线，已排除",
        }
    limit_rate = price_limit_rate(code)
    change_pct = quote.get("change_pct")
    entry_trade_status = "可正常评估"
    if change_pct is not None and abs(change_pct) >= limit_rate * 100 * 0.98:
        entry_trade_status = "接近涨跌停，成交可行性需单独确认"
    return {
        "code": code, "name": name, "quote": quote, "history_meta": history_meta,
        "history": history, "entry_trade_status": entry_trade_status,
    }


def _default_row(item: dict[str, Any]) -> dict[str, Any]:
    """默认表只留看得懂的部分；技术字段统一进明细表，避免首屏堆满行话。"""
    quote = item["quote"]
    models = item["analysis"]["models"].get(20, [])
    plan = item["analysis"].get("trade_plan", {})
    return {
        "综合评分": item.get("cross_section_score"),
        "代码": item["code"],
        "名称": item["name"],
        "最新价": quote.get("price"),
        "涨跌幅": quote.get("change_pct"),
        "参考买入区间": _zone_text(plan.get("buy_zone")),
        "止损参考": plan.get("initial_stop"),
        "交易状态": item.get("entry_trade_status", "未评估"),
        "推荐理由": item.get("reason"),
        "主要风险": item.get("risk"),
    }


def _detail_row(item: dict[str, Any]) -> dict[str, Any]:
    """技术明细表：给懂行的人复核模型到底给了什么数、过没过验证。"""
    quote = item["quote"]
    models = item["analysis"]["models"].get(20, [])
    lightgbm = next((model for model in models if model["model"] == "LightGBM"), {})
    xgboost = next((model for model in models if model["model"] == "XGBoost"), {})
    plan = item["analysis"].get("trade_plan", {})
    backtest_metrics = (item.get("backtest") or {}).get("metrics") or {}
    return {
        "代码": item["code"],
        "名称": item["name"],
        "行情时间": quote.get("quote_time") or quote.get("fetched_at"),
        "数据状态": quote.get("data_status"),
        "时序因子分": item["analysis"].get("factor_score"),
        "LightGBM 20日预测值": validated_return(lightgbm),
        "LightGBM 截面Z分数": lightgbm.get("prediction_zscore"),
        "LightGBM Rank IC": (lightgbm.get("oos_metrics") or {}).get("rank_ic"),
        "LightGBM 排名": item.get("lightgbm_rank"),
        "XGBoost 20日预测值": validated_return(xgboost),
        "XGBoost 截面Z分数": xgboost.get("prediction_zscore"),
        "XGBoost Rank IC": (xgboost.get("oos_metrics") or {}).get("rank_ic"),
        "XGBoost 排名": item.get("xgboost_rank"),
        "退出条件": plan.get("trailing_exit"),
        "样本外回测收益": backtest_metrics.get("total_return"),
        "回测状态": (item.get("backtest") or {}).get("status", "未计算"),
        "信号生成时间": plan.get("generated_at"),
        "截面日期": item["analysis"].get("prediction_reference_date"),
        "模型上下文": item["analysis"].get("market_context_id"),
    }


def _build_rows(records: Sequence[dict[str, Any]]) -> tuple[list, list, list, list]:
    """把排序结果拆成三张表与一份可选股票清单，字段口径只在这里定义一次。"""
    rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    usable: list[dict[str, str]] = []
    for item in records:
        if item.get("error"):
            failed_rows.append({"代码": item["code"], "名称": item["name"], "状态": item["error"]})
            continue
        usable.append({"code": item["code"], "name": item["name"]})
        rows.append(_default_row(item))
        detail_rows.append(_detail_row(item))
    return rows, detail_rows, failed_rows, usable


def run_scan_pipeline(
    service: Any,
    options: ScanOptions,
    candidates: list[dict[str, Any]],
    source_meta: dict[str, Any],
    progress: ProgressCallback | None = None,
    governance: Any | None = None,
) -> ScanResult:
    """按 options 扫描 candidates 并返回可直接渲染的结果。

    顺序约束（原页面靠代码恰好写对才成立，这里写成结构并由单测钉住）：
    组合回测在逐股评分循环之前跑完，逐股历史在全部计算结束之后才释放。

    ``governance`` 是模型治理钩子（``services/model_governance.py::ScanGovernance``），
    按鸭子类型使用它的 ``scoring_paused()`` 与 ``after_scan()`` 两个方法。
    **默认 None 表示完全不启用治理**——这不是为了省事，而是因为本函数的单测
    在不连任何数据库的进程里跑（tests/test_scan_pipeline.py），
    治理必须是可以摘掉的一层，而不是一个「默默去连库、失败就吞掉」的暗桩。
    """
    result = ScanResult(options=options, source_meta=source_meta)
    # 治理状态在循环开始前只读一次：逐股评分时再查库既无意义又慢，
    # 而扫描中途被降级这种情况本来就该由下一轮扫描来处理。
    scoring_paused = bool(governance.scoring_paused()) if governance is not None else False
    if scoring_paused:
        result.note("warning", f"{settings.DEGRADED_SCORING_MESSAGE}：本轮扫描只输出传统因子观察，不输出模型评分。")
    benchmark = _fetch_benchmark(service, result, progress)

    records: list[dict[str, Any]] = []
    total = len(candidates)
    for index, candidate in enumerate(candidates):
        _report(progress, f"正在分析 {candidate['code']} · 已完成 {index:,} / {total:,} 只", index, total)
        try:
            records.append(_collect_one(service, candidate, source_meta))
        except Exception as exc:  # 单只股票的任何失败都不该中断整轮扫描
            records.append({
                "code": candidate["code"],
                "name": candidate.get("name", candidate["code"]),
                "error": f"{type(exc).__name__}: {exc}",
            })
        finally:
            _report(progress, f"已完成 {index + 1:,} / {total:,} 只", index + 1, total)
    ready = [item for item in records if not item.get("error")]

    exposures = None
    if options.enable_neutralization:
        _report(progress, "正在获取流通市值与申万一级行业暴露…")
        try:
            exposures, exposure_meta = load_exposures(service, scope="all", progress=_message_only(progress))
            result.note("caption", f"暴露数据：{exposure_meta.get('status')}；{exposure_meta.get('limitation', '')}")
        except (OSError, ValueError, ImportError, MarketDataError) as exc:
            exposures = None
            result.note("warning", f"市值与行业暴露不可用，本次不做中性化：{type(exc).__name__}: {exc}")

    extra_factors: dict[str, Any] = {}
    extra_store = None
    if options.use_extra_factors:
        _report(progress, "正在装配筹码、日内与 PIT 财报因子…")
        extra_store = service.panels
        extra_factors, extra_meta = build_extra_factors(
            extra_store, [item["code"] for item in ready],
            histories={item["code"]: item["history"] for item in ready}, progress=_message_only(progress),
        )
        result.note("caption", (
            f"新增因子：{extra_meta.get('status')}；"
            f"日内 {extra_meta.get('intraday', {}).get('codes_computed', 0)} 只、"
            f"财报 {extra_meta.get('financial', {}).get('codes', 0)} 只。"
        ))

    context: dict[str, Any] = {}
    try:
        context = train_market_context(
            {item["code"]: item["history"] for item in ready}, benchmark,
            metadata={**source_meta, "pool_mode": options.pool_mode}, progress=_message_only(progress),
            exposures=exposures, objective=options.objective_mode, window=int(options.training_window),
            extra_factors=extra_factors or None, store=extra_store,
        )
        result.context_id = context.get("context_id")
        result.reference_date = context.get("reference_date")
        result.note("info", (
            f"共享模型上下文 {context['context_id'][:12]} · 截面日期 {context['reference_date']} · {len(ready)}只有效股票"
        ))
        result.note("caption", (
            f"训练目标 {objective_label(context['objective_mode'])}；滚动窗口 "
            f"{context['training_window_days'] or '不限'}个交易日；中性化状态：{context['neutralization']['status']}。"
        ))
        if (context.get("extra_factors") or {}).get("families"):
            result.note("caption", f"新增因子并入结果：{context['extra_factors'].get('status')}")
        result.note("caption", context["limitation"])
    except (ValueError, OSError, ImportError) as exc:
        result.note("warning", f"截面模型不可用：{exc}；本次只展示传统因子观察。")

    # 面板只读一次，组合回测与治理的漂移监控共用同一份：两次读盘会读到内存里两份
    # 几十兆的 DataFrame，而它们本来就该是同一次扫描的同一个截面。
    # 两个用途都没开时干脆不读——这份 parquet 有二十多兆，白读一次是纯浪费。
    panel: pd.DataFrame | None = None
    if context and (options.run_portfolio or governance is not None):
        _, panel = load_context(include_panel=True)

    if options.run_portfolio and context:
        _report(progress, "正在滚动重训截面排序模型并逐日调仓回测…")
        portfolio_panel = panel
        if portfolio_panel is None or portfolio_panel.empty:
            result.note("warning", "未能读取本次截面面板，组合回测已跳过。")
        else:
            try:
                result.portfolio_result = run_portfolio_strategy(
                    {item["code"]: item["history"] for item in ready},
                    portfolio_panel,
                    benchmark=benchmark,
                    horizon=20,
                    top_n=int(options.portfolio_top_n),
                    max_weight=float(options.max_position_weight),
                    rebalance_every=int(options.rebalance_every),
                    objective=options.objective_mode,
                    window=int(options.training_window),
                    universe_limit=settings.PORTFOLIO_BACKTEST_MAX_STOCKS,
                    progress=_message_only(progress),
                )
            except (ValueError, OSError, ImportError) as exc:
                result.note("warning", f"组合回测不可用：{type(exc).__name__}: {exc}")

    for item in ready:
        code, history, quote = item["code"], item["history"], item["quote"]
        item["analysis"] = analyze_stock(
            history, current_price=quote.get("price"), benchmark=benchmark,
            horizons=(5, 20, 60), code=code,
            as_of=quote.get("quote_time") or quote.get("fetched_at"),
            quote_status=quote.get("data_status"), market_context=context,
            scoring_paused=scoring_paused,
        )
        item["features"] = item["analysis"]["latest_features"]
        item["analysis"].pop("features", None)
        item["backtest"] = None
        if options.include_backtest and context:
            try:
                _report(progress, f"正在重新训练历史截面并回测 {code}，此步骤可能较久…")
                raw_history, _ = service.get_history(code, count=1200, adjusted=False)
                item["backtest"] = run_backtest(
                    completed_daily_history(raw_history), code=code, benchmark=benchmark, adjusted_history=history,
                )
            except (MarketDataError, ValueError) as exc:
                item["backtest"] = {"status": f"回测不可用：{exc}", "metrics": None}

    # 到这里组合回测与逐股评分都已用完 history，可以释放扫描期间收集的逐股历史。
    for item in ready:
        item.pop("history", None)

    if governance is not None and context:
        # 治理放在最后：它不影响本次扫描的任何结果，只负责把结果记下来并判定告警。
        # 因此排在评分汇总之前、逐股历史释放之后，不占用任何数据依赖。
        _report(progress, "正在写入模型通过率与漂移记录…")
        try:
            summary = governance.after_scan(context, panel)
        except Exception as exc:  # 治理的任何意外都不该让已经算完的扫描结果丢掉
            LOGGER.exception("模型治理收尾失败：%s", exc)
            result.note("warning", f"模型治理未完成：{type(exc).__name__}: {exc}")
        else:
            for level, text in governance.notes(summary):
                result.note(level, text)

    _report(progress, "所有股票处理结束，正在汇总评分与排序…", total, total)
    result.recommendations = rank_candidates(records)
    result.rows, result.detail_rows, result.failed_rows, result.usable = _build_rows(result.recommendations)
    result.scan_stats = {
        "scanned": total,
        "scored": len(ready),
        "failed": len(records) - len(ready),
    }
    result.finished_at = datetime.now(SHANGHAI).isoformat(timespec="seconds")
    LOGGER.info(
        "扫描完成 trigger=%s pool=%s 扫描=%d 有效=%d 失败=%d 截面日期=%s",
        options.trigger, options.pool_mode, total, len(ready), len(records) - len(ready), result.reference_date,
    )
    return result
