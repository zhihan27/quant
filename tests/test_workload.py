"""验证计算互斥锁：无界面可用、繁忙时抛异常而不排队、以及任何路径都不漏槽位。"""

import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from config import settings
from services.workload import ComputeBusy, WORK_SLOTS, compute_lock


class _FakeConnection:
    def __init__(self, owner, acquired=True):
        self.owner = owner
        self.acquired = acquired

    def execute(self, query, parameters=None):
        self.owner.queries.append((query, parameters))
        return self

    def fetchone(self):
        return {"acquired": self.acquired}

    def commit(self):
        pass


class _FakeRepository:
    """记录传入的锁编号并在退出时关连接，不接触真实数据库。"""

    def __init__(self, acquired=True):
        self.acquired = acquired
        self.queries = []
        self.closed = 0

    @contextmanager
    def connect(self):
        connection = _FakeConnection(self, self.acquired)
        try:
            yield connection
        finally:
            self.closed += 1


class ComputeLockTests(unittest.TestCase):
    def setUp(self):
        # 用例之间不能互相污染：每个用例开始前先把唯一的计算槽位腾出来。
        self.assertTrue(WORK_SLOTS.acquire(blocking=False), "上一个用例漏还了计算槽位")
        WORK_SLOTS.release()

    def test_lock_uses_the_shared_configuration_key(self):
        """定时扫描进程与网页端必须抢同一个编号，各自另起一个等于放弃互斥。"""
        repository = _FakeRepository()
        with compute_lock(repository):
            pass
        query, parameters = repository.queries[0]
        self.assertIn("pg_try_advisory_lock", query)
        self.assertEqual(parameters, (settings.COMPUTE_LOCK_KEY,))

    def test_compute_lock_key_does_not_collide_with_other_locks(self):
        # 建表锁 706821024、盘中监控锁 706821025 都还占着各自的事务/会话；
        # 撞号会让扫描在建表瞬间被误判为「已有计算任务」。
        self.assertNotIn(settings.COMPUTE_LOCK_KEY, (706821024, 706821025))

    def test_busy_process_slot_raises_instead_of_queueing(self):
        with patch.object(WORK_SLOTS, "acquire", return_value=False), self.assertRaises(ComputeBusy) as caught:
            with compute_lock(_FakeRepository()):
                self.fail("槽位被占用时不应进入临界区")
        self.assertIn("已有计算任务", str(caught.exception))

    def test_busy_server_lock_raises_and_releases_the_slot(self):
        repository = _FakeRepository(acquired=False)
        with self.assertRaises(ComputeBusy) as caught:
            with compute_lock(repository):
                self.fail("数据库锁被别的进程持有时不应进入临界区")
        self.assertIn("服务器已有计算任务", str(caught.exception))
        self.assertEqual(repository.closed, 1)
        # 关键：抛异常后槽位必须还回去，否则后续所有任务都会被自己的锁永久挡住。
        self.assertTrue(WORK_SLOTS.acquire(blocking=False), "数据库锁失败时漏还了进程槽位")
        WORK_SLOTS.release()

    def test_slot_is_released_when_the_body_fails(self):
        repository = _FakeRepository()
        with self.assertRaises(RuntimeError):
            with compute_lock(repository):
                raise RuntimeError("扫描中途失败")
        self.assertEqual(repository.closed, 1)
        self.assertTrue(WORK_SLOTS.acquire(blocking=False), "任务抛异常时漏还了进程槽位")
        WORK_SLOTS.release()

    def test_connect_failure_also_releases_the_slot(self):
        """数据库连不上时同样不能把槽位吞掉——那会让整个部署在重启前一直报「已有计算任务」。"""

        class _BrokenRepository:
            @contextmanager
            def connect(self):
                raise OSError("连接失败")
                yield

        with self.assertRaises(OSError):
            with compute_lock(_BrokenRepository()):
                self.fail("连接失败时不应进入临界区")
        self.assertTrue(WORK_SLOTS.acquire(blocking=False), "连接失败时漏还了进程槽位")
        WORK_SLOTS.release()

    def test_only_one_task_enters_at_a_time_in_process(self):
        repository = _FakeRepository()
        with compute_lock(repository):
            with self.assertRaises(ComputeBusy):
                with compute_lock(_FakeRepository()):
                    self.fail("同一进程内不应有两个任务同时持锁")


class ExplicitExclusivityTests(unittest.TestCase):
    """loading_task 曾按标题字符串猜任务轻重，漏登记一个标题就静默失去并发保护。

    「Top-N 组合构建与摩擦成本回测」就是这么被漏掉的——它是全流程里最耗 CPU 的一段，
    却因为标题没进白名单而完全没有并发保护。改成调用方显式声明后，标题不再参与判定。
    """

    def _run(self, title, exclusive):
        from services.ui_helpers import loading_task

        # 这里只问「有没有加锁」，不渲染界面：_loading_display 需要真实的 Streamlit 脚本上下文，
        # 裸跑会在 st.status(...).update() 上抛错，与本次要验证的并发判定无关。
        stub = SimpleNamespace(step=lambda *args, **kwargs: None)
        with patch("services.workload.compute_lock") as lock, patch(
            "services.runtime.get_repository", return_value=object()
        ), patch("services.ui_helpers._loading_display", return_value=nullcontext(stub)):
            with loading_task(title, exclusive=exclusive) as task:
                task.step("进行中")
            return lock.call_count

    def test_title_does_not_decide_whether_the_lock_is_taken(self):
        self.assertEqual(self._run("扫描与模型分析", exclusive=False), 0, "标题命中旧白名单不该自动加锁")
        self.assertEqual(self._run("任意别的新标题", exclusive=True), 1, "显式声明的独占任务必须加锁")

    def test_every_heavy_call_site_declares_exclusivity(self):
        """页面里真正会重算模型的环节必须显式声明；漏声明就是漏保护。"""
        expected = {
            "1_个股分析.py": {"滚动样本外回测", "配对统计分析"},
            "2_量化选股.py": {"采集股票池数据", "训练模型并生成排名"},
        }
        for name, titles in expected.items():
            source = (settings.PROJECT_ROOT / "pages" / name).read_text(encoding="utf-8")
            for title in titles:
                marker = f'loading_task("{title}"'
                self.assertIn(marker, source, f"{name} 中的「{title}」环节不见了")
                call = source[source.index(marker):source.index(marker) + 200]
                self.assertIn("exclusive=True", call, f"{name} 中的「{title}」没有声明 exclusive=True")
            # 个股分析只读已落盘的模型、不写共享产物，必须留在锁外，免得收盘后的扫描把它挡在门外。
        analysis = (settings.PROJECT_ROOT / "pages" / "1_个股分析.py").read_text(encoding="utf-8")
        marker = 'loading_task("个股分析"'
        self.assertIn(marker, analysis)
        self.assertNotIn("exclusive=True", analysis[analysis.index(marker):analysis.index(marker) + 200])


if __name__ == "__main__":
    unittest.main()
