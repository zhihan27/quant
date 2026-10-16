"""不打开网页也可运行的持久化监控入口，参数沿用数据库设置。"""

from services.database import QuantRepository
from services.intraday_monitor import IntradayMonitor


if __name__ == "__main__":
    monitor = IntradayMonitor(QuantRepository())
    print("监控任务已启动，沿用数据库中的开关及自选；按 Ctrl+C 停止。")
    try:
        monitor.run()
    except KeyboardInterrupt:
        monitor.stop()
        print("监控任务已停止。")
