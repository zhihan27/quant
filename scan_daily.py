"""每日收盘后的全市场扫描入口，供 systemd timer 或 Windows 计划任务调用。

用法（在项目根目录）：

    .venv\\Scripts\\python.exe -X utf8 scan_daily.py --dry-run   # 只看参数与跳过原因
    .venv\\Scripts\\python.exe -X utf8 scan_daily.py             # 正常执行，已跑过就跳过
    .venv\\Scripts\\python.exe -X utf8 scan_daily.py --force     # 忽略跳过判定，立即扫描

扫描口径取自 settings.DAILY_SCAN，与管理员在页面上手动的全市场扫描完全一致：
两者训练的是同一份 models/market_context_v{版本}.joblib，都通过 settings.COMPUTE_LOCK_KEY
互斥，因此不会同时训练并互相覆盖。

退出码：0 表示成功或按判定正常跳过，1 表示失败需要人工查看日志。
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from zoneinfo import ZoneInfo

import pandas as pd

from config import settings
from services.market_data import MarketDataError, MarketDataService, completed_daily_history
from services.ml_pipeline import load_context
from services.scan_pipeline import ScanOptions, collect_candidates, run_scan_pipeline
from services.scan_store import save_scan_result
from services.workload import ComputeBusy, compute_lock

LOGGER = logging.getLogger("quant.daily_scan")
SHANGHAI = ZoneInfo("Asia/Shanghai")


def _setup_logging() -> None:
    """同时写文件与 stdout：文件供事后追溯，stdout 供 journalctl 与计划任务查看。

    与 services/market_data.py 一样用日志器属性做幂等标记，重复调用不会叠加 handler。
    """
    if getattr(LOGGER, "_quant_daily_scan", False):
        return
    LOGGER.setLevel(logging.INFO)
    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = RotatingFileHandler(
        settings.LOG_DIR / "daily_scan.log",
        maxBytes=settings.DAILY_SCAN_LOG_MAX_BYTES,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)
    LOGGER._quant_daily_scan = True


def _today() -> str:
    return datetime.now(SHANGHAI).date().isoformat()


def _skip_reason(service: MarketDataService) -> str | None:
    """判断今天要不要跑；返回 None 表示应当继续。

    这是本脚本最关键的一道保护。收盘后行情源的当日 K 线尚未更新时，逐股采集会把
    整池判成「历史行情未更新至报价对应的完整交易日」而大批排除，训练出一个几乎是
    空截面的模型，然后静默覆盖掉原本可用的好模型——全程没有任何报错。因此宁可
    什么都不做，也不在数据没就绪时写盘。
    """
    history, meta = service.get_history(settings.MARKET_INDEX_CODE, count=1200, adjusted=True, market_id=1)
    history = completed_daily_history(history)
    if history.empty:
        return f"沪深 300 日线为空，无法判定当日行情是否就绪（数据源状态：{meta.get('status')}）"
    latest = str(pd.Timestamp(history["date"].iloc[-1]).date())
    if latest != _today():
        return f"当日 K 线尚未更新（沪深 300 已完成日线最新为 {latest}，今天是 {_today()}）"
    return None


def _already_done_reason() -> str | None:
    """本地模型上下文的截面日期已经是今天就不必再跑；周末与节假日靠这条自然覆盖。"""
    context, _ = load_context()
    if not context:
        return None
    reference_date = context.get("reference_date")
    if reference_date == _today():
        return f"今日截面模型已存在（context_id={str(context.get('context_id'))[:12]}），无需重复训练"
    return None


def _describe(options: ScanOptions) -> str:
    return (
        f"股票范围={options.pool_mode} 训练目标={options.objective_mode} "
        f"滚动窗口={options.training_window or '不限'} 中性化={options.enable_neutralization} "
        f"扩展因子={options.use_extra_factors} 组合回测={options.run_portfolio} "
        f"逐股回测={options.include_backtest} 落盘行数={options.scan_limit}"
    )


def _progress(message: str, completed: int | None = None, total: int | None = None) -> None:
    """扫描各阶段的进度回调。页面把同样的回调渲染成进度条，这里只落日志。"""
    LOGGER.info("%s", message)


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="收盘后按与管理端相同的口径执行一次全市场扫描。")
    parser.add_argument("--force", action="store_true", help="跳过数据就绪与幂等判定，立即扫描。")
    parser.add_argument("--dry-run", action="store_true", help="只打印本次使用的参数与跳过原因，不联网、不落盘。")
    args = parser.parse_args(argv)

    _setup_logging()
    options = ScanOptions(scan_limit=settings.DAILY_SCAN_TOP_ROWS, trigger="scheduled", **settings.DAILY_SCAN)

    if args.dry_run:
        # 干跑不建行情连接：跳过原因里的数据就绪判定要联网，这里只做纯本地的幂等判定。
        done = _already_done_reason()
        print(f"扫描参数：{_describe(options)}")
        print(f"幂等判定：{done or '今日尚未生成截面模型，正常执行时会继续联网检查行情是否就绪'}")
        print("（--dry-run 不联网、不落盘；数据就绪判定需访问行情源，故此处未执行。）")
        return 0

    LOGGER.info("定时扫描启动 %s force=%s", _describe(options), args.force)
    service = MarketDataService()
    try:
        if not args.force:
            reason = _already_done_reason()
            if reason:
                LOGGER.info("跳过：%s", reason)
                return 0
            reason = _skip_reason(service)
            if reason:
                LOGGER.info("跳过：%s", reason)
                return 0

        with compute_lock():
            candidates, source_meta = collect_candidates(service, options, _progress)
            if not candidates:
                LOGGER.error("股票池为空：%s", source_meta.get("error", "未知原因"))
                return 1
            LOGGER.info("股票池就绪：%d 只（来源 %s）", len(candidates), source_meta.get("source", source_meta.get("status")))
            result = run_scan_pipeline(service, options, candidates, source_meta, _progress)

        if not result.rows:
            # 一行都没产出说明整池都被排除，多半是数据侧出了问题；此时不落盘，
            # 免得把只读视图上原本可看的排名换成一张空表。
            LOGGER.error("扫描未产出任何候选（扫描 %d 只，失败 %d 只），保留原有结果不覆盖",
                         result.scan_stats.get("scanned", 0), result.scan_stats.get("failed", 0))
            return 1
        target = save_scan_result(result)
        LOGGER.info("扫描完成：候选 %d 只，截面日期 %s，结果已写入 %s",
                    len(result.rows), result.reference_date, target)
        for note in result.notes:
            LOGGER.info("血缘说明：%s", note.get("text"))
        return 0
    except ComputeBusy as exc:
        LOGGER.warning("未能取得计算锁：%s", exc)
        return 1
    except (MarketDataError, OSError, ValueError, ImportError) as exc:
        LOGGER.exception("定时扫描失败：%s", exc)
        return 1
    finally:
        service.session.close()


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
