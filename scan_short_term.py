"""后台短线日频研究入口；训练不运行在 Streamlit 页面主线程。"""

from __future__ import annotations

import argparse
import logging

from config.short_term import SHORT_TERM_MAIN_HORIZON
from services.database import QuantRepository
from services.short_term.pipeline import run_short_term_research
from services.short_term.forward_observation import observe_pending_predictions
from services.short_term.store import ShortTermStore
from services.workload import compute_lock


def main() -> int:
    """领取数据库任务锁，完成短线研究并保存运行状态。"""
    parser = argparse.ArgumentParser(description="运行短线日频研究")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--horizon", type=int, choices=(1, 3, 5), default=SHORT_TERM_MAIN_HORIZON)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    repository = QuantRepository()
    store = ShortTermStore(repository)
    try:
        store.mark_run_running(arguments.run_id)
        with compute_lock(repository):
            result = run_short_term_research(arguments.run_id, horizon=arguments.horizon)
        try:
            result["forward_observation"] = observe_pending_predictions(store)
        except Exception as exc:
            logging.getLogger("quant.short_term.worker").exception("前向观察记录失败")
            result["forward_observation"] = {
                "status": "observation_failed",
                "observed_count": 0,
                "promotion_evidence": False,
                "blockers": [f"前向观察写入失败：{type(exc).__name__}"],
            }
        store.finish_run(arguments.run_id, result)
        logging.getLogger("quant.short_term.worker").info(
            "短线研究结束 run_id=%s status=%s predictions=%s",
            arguments.run_id,
            result.get("status"),
            result.get("prediction_count", 0),
        )
        return 0
    except Exception as exc:
        store.fail_run(arguments.run_id, f"{type(exc).__name__}: {exc}")
        logging.getLogger("quant.short_term.worker").exception("短线研究失败 run_id=%s", arguments.run_id)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
