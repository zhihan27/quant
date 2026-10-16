@echo off
setlocal
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
    py -3 -m venv .venv
    if errorlevel 1 (
        echo Failed to create virtual environment. Install Python 3.11+ and try again.
        exit /b 1
    )
)

.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -m streamlit run app.py
endlocal

