"""逐只循环的进度节流：整倍数上报，且最后一只一定上报。

页面上的一次进度更新是一次 Streamlit 元素写入，逐只上报既刷屏又拖慢计算，
所以扫描链路里的逐股循环（分钟线汇总、筹码并入）都按固定间隔上报。

但只按整倍数上报会造成一个稳定的假象：循环结束时字幕恰好停在最后一个整倍数上，
剩下的零头被无声吞掉。5174 只的股票池停在「5151/5174」，而且**每一轮都停在这个数字**，
因为 5150 是 5174 以内 50 的最后一个整倍数——看页面的人只会以为是卡死，
而不是「还剩 23 只」。补上末条即可让计数走到 N/N，并让「还剩多少」一眼可见。
"""

from __future__ import annotations

from typing import Callable

# 逐只循环的上报间隔。5174 只的筹码并入约每只 46 毫秒，即约 2.3 秒一条，
# 既看得见进展，也不会把页面刷爆。
PROGRESS_EVERY = 50


def throttled(total: int, every: int = PROGRESS_EVERY) -> Callable[[int], bool]:
    """返回判定器：第 index 只（从 0 起）这次是否需要上报。

    只负责「该不该报」，不关心报什么——调用方保留自己的文案格式。
    """
    def should_report(index: int) -> bool:
        # 末条与整倍数落在同一次迭代时也只会进一次，不会重复上报。
        return index % every == 0 or index == total - 1

    return should_report
