"""历史估值采集、规范数据导入、四组对照与独立验收命令行入口。"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from services.acquisition_store import _atomic_json, load_snapshot
from services.panel_store import PanelStore
from services.style_experiments import compare_variants, experiment_panels, independent_acceptance
from services.style_factors import save_history
from services.valuation_data import fetch_daily_basic


def main() -> None:
    """按子命令采集或评估；离线实验不改写生产模型文件。"""
    parser = argparse.ArgumentParser(description="历史估值与显式风格研究")
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect", help="使用 TUSHARE_TOKEN 采集日市值，估值修订口径未核验")
    collect.add_argument("--codes", required=True, help="逗号分隔的六位代码")
    collect.add_argument("--start", required=True)
    collect.add_argument("--end", required=True)
    ingest = commands.add_parser("import", help="导入带完整版本血缘的规范 Parquet")
    ingest.add_argument("--file", required=True)
    compare = commands.add_parser("compare", help="同一采集快照的四组样本外对照")
    compare.add_argument("--snapshot", required=True)
    compare.add_argument("--start", required=True)
    compare.add_argument("--end", required=True)
    compare.add_argument("--output", required=True)
    accept = commands.add_parser("accept", help="使用新区间执行一次性独立验收")
    accept.add_argument("--snapshot", required=True)
    accept.add_argument("--comparison", required=True)
    accept.add_argument("--start", required=True)
    accept.add_argument("--end", required=True)
    args = parser.parse_args()
    store = PanelStore()
    if args.command == "collect":
        for code in args.codes.split(","):
            frame = fetch_daily_basic(code.strip(), args.start, args.end)
            if frame.empty:
                print(f"{code}：接口没有可用历史，未接入")
            else:
                print(json.dumps(save_history(store, frame), ensure_ascii=False))
        return
    if args.command == "import":
        try:
            print(json.dumps(save_history(store, pd.read_parquet(args.file)), ensure_ascii=False))
        except (OSError, ValueError, ImportError):
            logging.exception("历史估值导入失败")
            raise
        return
    snapshot = load_snapshot(args.snapshot, store=store)
    history = snapshot.extra_factors.get("valuation")
    if history is None or history.empty:
        raise ValueError("采集快照没有冻结估值历史，请先导入历史并启用风格采集后重新采集快照。")
    comparison = json.loads(Path(args.comparison).read_text(encoding="utf-8")) if args.command == "accept" else None
    # 可使用追加了新区间的快照；历史股票、特征和成熟标签由对照指纹逐项校验。
    calibration_end = comparison["validation_start"] if comparison else args.start
    panels = experiment_panels(snapshot.histories, snapshot.benchmark, history, calibration_end,
                               exposures=snapshot.exposures, extra_factors=snapshot.extra_factors)
    if comparison:
        report = independent_acceptance(panels, comparison, args.start, args.end)
        print("独立验收：" + ("通过，可按配置启用" if report["passed"] else "未通过，新增特征保持关闭"))
    else:
        report = compare_variants(panels, args.start, args.end)
        report["snapshot_id"] = snapshot.snapshot_id
        _atomic_json(Path(args.output), report)
        print(f"四组对照已保存：{args.output}；验证期选择：{report['selected_variant'] or '无可用组合'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
