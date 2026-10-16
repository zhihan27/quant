@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo 请先运行 run.bat 安装环境。
    pause
    exit /b 1
)
rem 与 run_monitor.bat 不同，这里不 pause：计划任务在无人值守下运行，pause 会让任务挂住不退出。
.venv\Scripts\python.exe -X utf8 scan_daily.py %*
exit /b %ERRORLEVEL%
