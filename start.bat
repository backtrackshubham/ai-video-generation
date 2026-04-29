@echo off
setlocal EnableDelayedExpansion

:: ============================================================
::  AI Video Generation — Windows Launcher
::  Usage:  start.bat [port]    (default: 7860)
:: ============================================================

SET "ROOT=%~dp0"
IF "%ROOT:~-1%"=="\" SET "ROOT=%ROOT:~0,-1%"

SET "PORT=7860"
IF NOT "%~1"=="" SET "PORT=%~1"

:: Check setup has been run
IF NOT EXIST "%ROOT%\venv\Scripts\activate.bat" (
    echo.
    echo  ERROR: Virtual environment not found.
    echo  Please run setup_windows.bat first.
    echo.
    pause
    exit /b 1
)

IF NOT EXIST "%ROOT%\t2m_gpt" (
    echo.
    echo  ERROR: t2m_gpt directory not found.
    echo  Please run setup_windows.bat first.
    echo.
    pause
    exit /b 1
)

:: Point all caches inside the repo
SET "HF_HOME=%ROOT%\models\hf_cache"
SET "TRANSFORMERS_CACHE=%ROOT%\models\hf_cache"
SET "DIFFUSERS_CACHE=%ROOT%\models\hf_cache"
SET "TORCH_HOME=%ROOT%\models\torch_cache"
SET "XDG_CACHE_HOME=%ROOT%\models\xdg_cache"

CALL "%ROOT%\venv\Scripts\activate.bat"

echo.
echo  Starting AI Video Generation server...
echo  URL:  http://localhost:%PORT%
echo  Press Ctrl+C to stop.
echo.

python "%ROOT%\app.py" --port %PORT%
