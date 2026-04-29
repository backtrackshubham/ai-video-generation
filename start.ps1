<#
.SYNOPSIS
    AI Video Generation — PowerShell Launcher
.DESCRIPTION
    Starts the Flask server. Run this from Windows Terminal or PowerShell.
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

# Check setup has been run
if (-not (Test-Path "$Root\venv\Scripts\Activate.ps1")) {
    Write-Host ""
    Write-Host "ERROR: Virtual environment not found." -ForegroundColor Red
    Write-Host "Please run setup_windows.bat first." -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

if (-not (Test-Path "$Root\t2m_gpt")) {
    Write-Host ""
    Write-Host "ERROR: t2m_gpt directory not found." -ForegroundColor Red
    Write-Host "Please run setup_windows.bat first." -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# Point all caches inside the repo
$env:HF_HOME             = "$Root\models\hf_cache"
$env:TRANSFORMERS_CACHE  = "$Root\models\hf_cache"
$env:DIFFUSERS_CACHE     = "$Root\models\hf_cache"
$env:TORCH_HOME          = "$Root\models\torch_cache"
$env:XDG_CACHE_HOME      = "$Root\models\xdg_cache"

# Activate virtual environment
& "$Root\venv\Scripts\Activate.ps1"

Write-Host ""
Write-Host "Starting AI Video Generation server..." -ForegroundColor Cyan
Write-Host "URL:  http://localhost:$Port" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop."
Write-Host ""

python "$Root\app.py" --port $Port
