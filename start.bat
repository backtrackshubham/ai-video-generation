@echo off
:: AI Video Generation — Windows Launcher
:: Delegates to start.py (cross-platform launcher)
::
:: Usage:  start.bat [port]    (default: 7860)

SET "ROOT=%~dp0"
IF "%ROOT:~-1%"=="\" SET "ROOT=%ROOT:~0,-1%"

SET "PORT=7860"
IF NOT "%~1"=="" SET "PORT=%~1"

IF NOT EXIST "%ROOT%\venv\Scripts\python.exe" (
    echo.
    echo  ERROR: Virtual environment not found.
    echo  Please run:  python setup.py
    echo.
    pause
    exit /b 1
)

"%ROOT%\venv\Scripts\python.exe" "%ROOT%\start.py" %PORT%
