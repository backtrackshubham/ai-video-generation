<#
.SYNOPSIS
    AI Video Generation — PowerShell Launcher
.DESCRIPTION
    Delegates to start.py (cross-platform launcher).
.PARAMETER Port
    Port to listen on (default: 7860)
.EXAMPLE
    .\start.ps1
    .\start.ps1 -Port 8080
#>
param(
    [int]$Port = 7860
)

$Root = $PSScriptRoot

if (-not (Test-Path "$Root\venv\Scripts\python.exe")) {
    Write-Host ""
    Write-Host "ERROR: Virtual environment not found." -ForegroundColor Red
    Write-Host "Please run:  python setup.py" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

& "$Root\venv\Scripts\python.exe" "$Root\start.py" $Port
