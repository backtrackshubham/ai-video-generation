@echo off
setlocal EnableDelayedExpansion

:: ============================================================
::  AI Video Generation — Windows Setup Script
::  Branch: feature/cuda-windows
::
::  Run this once after cloning the repo.
::  Everything is downloaded into the repo directory.
::  No files are written outside of it.
::
::  Requirements before running:
::    - Python 3.10 or 3.11  (python.org/downloads)
::    - Git                   (git-scm.com)
::    - NVIDIA GPU driver 525+ with CUDA 12.x support
::    - Internet connection (~5 GB download total)
::
::  Usage:
::    setup_windows.bat
:: ============================================================

:: Resolve the repo root to the directory containing this script
SET "ROOT=%~dp0"
:: Strip trailing backslash
IF "%ROOT:~-1%"=="\" SET "ROOT=%ROOT:~0,-1%"

echo.
echo ==============================================================
echo  AI Video Generation - Windows CUDA Setup
echo  Repo root: %ROOT%
echo ==============================================================
echo.

:: ── Redirect all caches inside the repo ──────────────────────
SET "HF_HOME=%ROOT%\models\hf_cache"
SET "TRANSFORMERS_CACHE=%ROOT%\models\hf_cache"
SET "DIFFUSERS_CACHE=%ROOT%\models\hf_cache"
SET "TORCH_HOME=%ROOT%\models\torch_cache"
SET "XDG_CACHE_HOME=%ROOT%\models\xdg_cache"

mkdir "%ROOT%\models\hf_cache"   2>nul
mkdir "%ROOT%\models\torch_cache" 2>nul
mkdir "%ROOT%\outputs"            2>nul
mkdir "%ROOT%\logs"               2>nul

:: ════════════════════════════════════════════════════════════
:: STEP 1 — Check Python
:: ════════════════════════════════════════════════════════════
echo [1/10] Checking Python version...
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo.
    echo  ERROR: Python not found on PATH.
    echo  Please install Python 3.10 or 3.11 from https://python.org/downloads
    echo  Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

:: Extract major.minor version
FOR /F "tokens=2 delims= " %%V IN ('python --version 2^>^&1') DO SET "PYVER=%%V"
FOR /F "tokens=1,2 delims=." %%A IN ("%PYVER%") DO (
    SET "PYMAJ=%%A"
    SET "PYMIN=%%B"
)
IF %PYMAJ% LSS 3 (
    echo  ERROR: Python 3.10+ required. Found: %PYVER%
    pause & exit /b 1
)
IF %PYMAJ% EQU 3 IF %PYMIN% LSS 10 (
    echo  ERROR: Python 3.10+ required. Found: %PYVER%
    pause & exit /b 1
)
echo  OK — Python %PYVER%

:: ════════════════════════════════════════════════════════════
:: STEP 2 — Check Git
:: ════════════════════════════════════════════════════════════
echo [2/10] Checking Git...
git --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo.
    echo  ERROR: Git not found on PATH.
    echo  Please install Git from https://git-scm.com/download/win
    echo.
    pause
    exit /b 1
)
echo  OK — Git found

:: ════════════════════════════════════════════════════════════
:: STEP 3 — Create virtual environment
:: ════════════════════════════════════════════════════════════
echo [3/10] Creating Python virtual environment in venv\...
IF EXIST "%ROOT%\venv\Scripts\activate.bat" (
    echo  venv already exists, skipping creation.
) ELSE (
    python -m venv "%ROOT%\venv"
    IF ERRORLEVEL 1 (
        echo  ERROR: Failed to create virtual environment.
        pause & exit /b 1
    )
    echo  venv created.
)

:: Activate venv for remainder of script
CALL "%ROOT%\venv\Scripts\activate.bat"
IF ERRORLEVEL 1 (
    echo  ERROR: Failed to activate virtual environment.
    pause & exit /b 1
)

:: ════════════════════════════════════════════════════════════
:: STEP 4 — Upgrade pip
:: ════════════════════════════════════════════════════════════
echo [4/10] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo  pip upgraded.

:: ════════════════════════════════════════════════════════════
:: STEP 5 — Install PyTorch with CUDA 12.1
:: ════════════════════════════════════════════════════════════
echo [5/10] Installing PyTorch 2.3.1 with CUDA 12.1 support...
echo  (This downloads ~2.5 GB — may take several minutes)
pip install torch==2.3.1 torchvision==0.18.1 ^
    --index-url https://download.pytorch.org/whl/cu121 ^
    --quiet
IF ERRORLEVEL 1 (
    echo  ERROR: PyTorch installation failed.
    pause & exit /b 1
)

:: Quick sanity check
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available after install'" 2>nul
IF ERRORLEVEL 1 (
    echo.
    echo  WARNING: PyTorch installed but CUDA is not detected.
    echo  This may mean your GPU driver is outdated or your GPU does not support CUDA 12.1.
    echo  The app will fall back to CPU mode but will be much slower.
    echo  To update your driver: https://www.nvidia.com/Download/index.aspx
    echo.
) ELSE (
    FOR /F "delims=" %%G IN ('python -c "import torch; print(torch.cuda.get_device_name(0))"') DO SET "GPU_NAME=%%G"
    echo  CUDA OK — GPU: !GPU_NAME!
)

:: ════════════════════════════════════════════════════════════
:: STEP 6 — Install remaining Python dependencies
:: ════════════════════════════════════════════════════════════
echo [6/10] Installing Python dependencies...
pip install ^
    diffusers==0.27.2 ^
    transformers==4.40.0 ^
    accelerate ^
    flask ^
    flask-cors ^
    imageio ^
    imageio-ffmpeg ^
    moviepy ^
    opencv-python ^
    scipy ^
    scikit-learn ^
    matplotlib ^
    smplx ^
    ftfy ^
    regex ^
    tqdm ^
    gdown ^
    --quiet
IF ERRORLEVEL 1 (
    echo  ERROR: Dependency installation failed.
    pause & exit /b 1
)

:: Install CLIP from GitHub
pip install git+https://github.com/openai/CLIP.git --quiet
IF ERRORLEVEL 1 (
    echo  ERROR: CLIP installation failed.
    pause & exit /b 1
)
echo  All Python dependencies installed.

:: ════════════════════════════════════════════════════════════
:: STEP 7 — Clone T2M-GPT
:: ════════════════════════════════════════════════════════════
echo [7/10] Setting up T2M-GPT...
IF EXIST "%ROOT%\t2m_gpt\.git" (
    echo  t2m_gpt already exists, skipping clone.
) ELSE (
    echo  Cloning T2M-GPT repository...
    git clone https://github.com/Mael-zys/T2M-GPT.git "%ROOT%\t2m_gpt"
    IF ERRORLEVEL 1 (
        echo  ERROR: Failed to clone T2M-GPT.
        pause & exit /b 1
    )
    echo  T2M-GPT cloned.
)

:: Patch utils\quaternion.py — fix np.float removed in numpy 1.24+
echo  Patching T2M-GPT for numpy 1.24+ compatibility...
python -c ^
    "import re, pathlib; ^
     f = pathlib.Path(r'%ROOT%\t2m_gpt\utils\quaternion.py'); ^
     txt = f.read_text(); ^
     txt = txt.replace('np.finfo(np.float).eps', 'np.finfo(float).eps'); ^
     f.write_text(txt); ^
     print('  quaternion.py patched OK')"

:: Note: quantize_cnn.py is NOT patched on this branch.
:: The .cuda() calls in register_buffer() work correctly with CUDA available.

:: ════════════════════════════════════════════════════════════
:: STEP 8 — Download T2M-GPT pretrained checkpoints
:: ════════════════════════════════════════════════════════════
echo [8/10] Downloading T2M-GPT pretrained checkpoints (~994 MB)...
IF EXIST "%ROOT%\t2m_gpt\pretrained\VQVAE\net_last.pth" (
    echo  Checkpoints already downloaded, skipping.
) ELSE (
    mkdir "%ROOT%\t2m_gpt\pretrained" 2>nul
    pushd "%ROOT%\t2m_gpt\pretrained"
    gdown 1LaOvwypF-jM2Axnq5dc-Iuvv3w_G-WDE -O VQTrans_pretrained.zip
    IF ERRORLEVEL 1 (
        echo  ERROR: Failed to download checkpoints.
        popd & pause & exit /b 1
    )
    python -c "import zipfile; zipfile.ZipFile('VQTrans_pretrained.zip').extractall('.')"
    del VQTrans_pretrained.zip
    popd
    echo  Checkpoints downloaded and extracted.
)

:: ════════════════════════════════════════════════════════════
:: STEP 9 — Download GloVe vectors
:: ════════════════════════════════════════════════════════════
echo [9/10] Downloading GloVe word vectors (~6 MB)...
IF EXIST "%ROOT%\t2m_gpt\glove\our_vab_data.npy" (
    echo  GloVe already downloaded, skipping.
) ELSE (
    pushd "%ROOT%\t2m_gpt"
    gdown --fuzzy https://drive.google.com/file/d/1bCeS6Sh_mLVTebxIgiUHgdPrroW06mb6/view?usp=sharing -O glove.zip
    IF ERRORLEVEL 1 (
        echo  ERROR: Failed to download GloVe vectors.
        popd & pause & exit /b 1
    )
    python -c "import zipfile; zipfile.ZipFile('glove.zip').extractall('.')"
    del glove.zip
    popd
    echo  GloVe vectors downloaded and extracted.
)

:: ════════════════════════════════════════════════════════════
:: STEP 10 — Pre-download text-to-video model
:: ════════════════════════════════════════════════════════════
echo [10/10] Pre-downloading text-to-video model (damo-vilab, ~3.5 GB)...
echo  (This may take 5-15 minutes depending on connection speed)
IF EXIST "%ROOT%\models\hf_cache\models--damo-vilab--text-to-video-ms-1.7b" (
    echo  T2V model already downloaded, skipping.
) ELSE (
    python -c ^
        "import torch; ^
         from diffusers import DiffusionPipeline; ^
         import os; ^
         cache = r'%ROOT%\models\hf_cache'; ^
         print('  Downloading... (this is the large one, be patient)'); ^
         DiffusionPipeline.from_pretrained( ^
             'damo-vilab/text-to-video-ms-1.7b', ^
             cache_dir=cache, ^
             torch_dtype=torch.float16, ^
             trust_remote_code=True ^
         ); ^
         print('  T2V model downloaded.')"
    IF ERRORLEVEL 1 (
        echo  ERROR: Failed to download T2V model.
        pause & exit /b 1
    )
)

:: ════════════════════════════════════════════════════════════
:: Done!
:: ════════════════════════════════════════════════════════════
echo.
echo ==============================================================
echo  Setup complete!
echo ==============================================================
echo.
echo  To start the server:
echo    start.bat
echo  or with PowerShell:
echo    powershell -ExecutionPolicy Bypass -File start.ps1
echo.
echo  Then open your browser at:
echo    http://localhost:7860
echo.
pause
