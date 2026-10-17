"""逐只循环的进度节流：整倍数上报、末条必报、且不重复。

这条契约直接对应「扫描卡在 5151/5174」的故障：原实现按 ``index % 50 == 0`` 上报，
5174 只的池子在 5174 以内最后一个整倍数是 5150，字幕就永远停在 5151/5174——
剩下的 23 只被无声吞掉，而且**每一轮都停在同一个数字**，看页面的人只会以为是卡死。
"""

from __future__ import annotations

import unittest

from services.progress import PROGRESS_EVERY, throttled


def _fired(total: int) -> list[int]:
    """按循环的真实次序取出所有会上报的 index。"""
    should_report = throttled(total)
    return [index for index in range(total) if should_report(index)]


class ThrottledTests(unittest.TestCase):
    def test_reports_on_multiples_of_the_interval(self) -> None:
        fired = _fired(200)
        self.assertEqual(fired, [0, 50, 100, 150, 199])
        self.assertEqual(PROGRESS_EVERY, 50)

    def test_the_pool_size_from_the_incident_reaches_its_own_last_code(self) -> None:
        """本仓库的股票池规模：末条必须是 5174，而不是 5151。"""
        fired = _fired(5174)
        self.assertEqual(fired[-1] + 1, 5174)
        self.assertNotEqual(fired[-1] + 1, 5151)

    def test_last_item_already_on_a_multiple_is_not_reported_twice(self) -> None:
        """总数恰好是间隔整倍数时，末条与整倍数是同一次迭代，只能上报一次。"""
        for total in (1, 50, 51, 100, 101):
            with self.subTest(total=total):
                fired = _fired(total)
                self.assertEqual(len(fired), len(set(fired)))
                self.assertEqual(fired[-1], total - 1)

    def test_pool_smaller_than_the_interval_still_reaches_the_last_code(self) -> None:
        """池子小于一个间隔时只剩首条与末条，末条仍是 N/N。"""
        self.assertEqual(_fired(3), [0, 2])
        self.assertEqual(_fired(2), [0, 1])

    def test_throttling_keeps_the_message_count_far_below_the_pool(self) -> None:
        """节流本身不能失效：5174 只的回调必须远少于 5174 次。"""
        self.assertLess(len(_fired(5174)), 110)


if __name__ == "__main__":
    unittest.main()
