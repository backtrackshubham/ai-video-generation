@echo off
setlocal EnableDelayedExpansion

:: ============================================================
::  AI Video Generation — Windows One-Shot Setup Script
::  Branch: feature/cuda-windows
::
::  Run once after cloning the repo. Downloads everything:
::    - Python venv + all dependencies (PyTorch CUDA 12.1)
::    - MDM repo + 50-step checkpoint (~1.3 GB)
::    - MDM SMPL body model (~100 MB)
::    - MDM GloVe / HumanML3D normalization files
::    - Wan2.1-T2V-1.3B model (~3 GB, pre-downloaded)
::    - CogVideoX-5B model (~22 GB, lazy — downloaded on first use)
::    - SVD 1.1 model (~8 GB, lazy — downloaded on first use)
::
::  Requirements before running:
::    - Python 3.10 or 3.11  (python.org/downloads — check "Add to PATH")
::    - Git                   (git-scm.com)
::    - NVIDIA GPU driver 525+ with CUDA 12.x support
::    - ~35 GB free disk space (models downloaded on first use)
::    - Internet connection
::
::  Usage:
::    setup_windows.bat
:: ============================================================

SET "ROOT=%~dp0"
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

for %%D in (
    "%ROOT%\models\hf_cache"
    "%ROOT%\models\torch_cache"
    "%ROOT%\outputs\normal-videos"
    "%ROOT%\outputs\i2v-videos"
    "%ROOT%\outputs\stickman-videos"
    "%ROOT%\outputs\wan-videos"
    "%ROOT%\gen-logs"
    "%ROOT%\gen-logs\wan-videos"
    "%ROOT%\uploads"
    "%ROOT%\cloned-repos"
) do mkdir %%D 2>nul

:: ════════════════════════════════════════════════════════════
:: STEP 1 — Check Python 3.10+
:: ════════════════════════════════════════════════════════════
echo [1/9] Checking Python version...
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo.
    echo  ERROR: Python not found on PATH.
    echo  Install Python 3.10 or 3.11 from https://python.org/downloads
    echo  Make sure to check "Add Python to PATH" during installation.
    echo.
    pause & exit /b 1
)
FOR /F "tokens=2 delims= " %%V IN ('python --version 2^>^&1') DO SET "PYVER=%%V"
FOR /F "tokens=1,2 delims=." %%A IN ("%PYVER%") DO (SET "PYMAJ=%%A" & SET "PYMIN=%%B")
IF %PYMAJ% LSS 3 (echo  ERROR: Python 3.10+ required. Found %PYVER% & pause & exit /b 1)
IF %PYMAJ% EQU 3 IF %PYMIN% LSS 10 (echo  ERROR: Python 3.10+ required. Found %PYVER% & pause & exit /b 1)
echo  OK — Python %PYVER%

:: ════════════════════════════════════════════════════════════
:: STEP 2 — Check Git
:: ════════════════════════════════════════════════════════════
echo [2/9] Checking Git...
git --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo.
    echo  ERROR: Git not found. Install from https://git-scm.com/download/win
    echo.
    pause & exit /b 1
)
echo  OK — Git found

:: ════════════════════════════════════════════════════════════
:: STEP 3 — Create / activate virtual environment
:: ════════════════════════════════════════════════════════════
echo [3/9] Setting up Python virtual environment...
IF EXIST "%ROOT%\venv\Scripts\activate.bat" (
    echo  venv already exists, skipping creation.
) ELSE (
    python -m venv "%ROOT%\venv"
    IF ERRORLEVEL 1 (echo  ERROR: Failed to create venv. & pause & exit /b 1)
    echo  venv created.
)
CALL "%ROOT%\venv\Scripts\activate.bat"
IF ERRORLEVEL 1 (echo  ERROR: Failed to activate venv. & pause & exit /b 1)

:: ════════════════════════════════════════════════════════════
:: STEP 4 — Upgrade pip
:: ════════════════════════════════════════════════════════════
echo [4/9] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo  pip upgraded.

:: ════════════════════════════════════════════════════════════
:: STEP 5 — Install PyTorch 2.3.1 with CUDA 12.1
:: ════════════════════════════════════════════════════════════
echo [5/10] Installing PyTorch 2.4.0 + CUDA 12.1 (~2.5 GB)...
pip install torch==2.4.0 torchvision==0.19.0 ^
    --index-url https://download.pytorch.org/whl/cu121 ^
    --quiet
IF ERRORLEVEL 1 (echo  ERROR: PyTorch install failed. & pause & exit /b 1)

python -c "import torch; assert torch.cuda.is_available(), 'no cuda'" 2>nul
IF ERRORLEVEL 1 (
    echo.
    echo  WARNING: CUDA not detected after install.
    echo  App will fall back to CPU (very slow for large models).
    echo  Update your NVIDIA driver: https://www.nvidia.com/Download/index.aspx
    echo.
) ELSE (
    FOR /F "delims=" %%G IN ('python -c "import torch; print(torch.cuda.get_device_name(0))"') DO SET "GPU_NAME=%%G"
    echo  CUDA OK — GPU: !GPU_NAME!
)

:: ════════════════════════════════════════════════════════════
:: STEP 6 — Install Python dependencies
:: ════════════════════════════════════════════════════════════
echo [6/10] Installing Python dependencies...
pip install ^
    "numpy>=1.24,<2.0" ^
    "diffusers>=0.32.0" ^
    "transformers>=4.40.0" ^
    "accelerate>=0.30.0" ^
    "huggingface_hub>=0.23.0" ^
    "tokenizers>=0.19.0" ^
    sentencepiece ^
    flask ^
    flask-cors ^
    "imageio>=2.28.0" ^
    imageio-ffmpeg ^
    "moviepy<2.0" ^
    "opencv-python>=4.8.0" ^
    "scipy>=1.11.0" ^
    scikit-learn ^
    "matplotlib>=3.7.0" ^
    smplx ^
    ftfy ^
    regex ^
    tqdm ^
    gdown ^
    chumpy ^
    einops ^
    Pillow ^
    --quiet
IF ERRORLEVEL 1 (echo  ERROR: Dependency install failed. & pause & exit /b 1)

:: Install CLIP from GitHub (required by MDM)
pip install git+https://github.com/openai/CLIP.git --quiet
IF ERRORLEVEL 1 (echo  ERROR: CLIP install failed. & pause & exit /b 1)

:: Patch chumpy for numpy 1.24+ compatibility
echo  Patching chumpy for numpy 1.24+ compatibility...
python -c ^
    "import pathlib, site; ^
     sp = site.getsitepackages()[0]; ^
     f = pathlib.Path(sp) / 'chumpy' / '__init__.py'; ^
     txt = f.read_text(); ^
     replacements = [('from numpy import bool,', 'from numpy import bool_,'), ^
                     ('np.bool,', 'np.bool_,'), ('np.int,', 'np.int_,'), ^
                     ('np.float,', 'np.float64,'), ('np.complex,', 'np.complex128,'), ^
                     ('np.object,', 'np.object_,'), ('np.str,', 'np.str_,')]; ^
     [txt.__setitem__(0, txt[0].replace(a, b)) for a, b in replacements]; ^
     print('skipping — manual patch may be needed if chumpy errors appear')" 2>nul
:: Safer inline patch via Python script
python -c "
import pathlib, site, sys
sp = site.getsitepackages()[0]
f = pathlib.Path(sp) / 'chumpy' / '__init__.py'
if not f.exists():
    print('  chumpy not found at', f)
    sys.exit(0)
txt = f.read_text()
pairs = [
    ('from numpy import bool, int, float, complex, object, str, ', 'from numpy import '),
    ('np.bool,',    'bool,'),
    ('np.int,',     'int,'),
    ('np.float,',   'float,'),
    ('np.complex,', 'complex,'),
    ('np.object,',  'object,'),
    ('np.str,',     'str,'),
]
changed = False
for old, new in pairs:
    if old in txt:
        txt = txt.replace(old, new)
        changed = True
if changed:
    f.write_text(txt)
    print('  chumpy patched OK')
else:
    print('  chumpy already compatible, no patch needed')
"
echo  All Python dependencies installed.

:: ════════════════════════════════════════════════════════════
:: STEP 7 — Clone MDM and T2M-GPT into cloned-repos\
:: ════════════════════════════════════════════════════════════
echo [7/10] Setting up cloned repos (MDM + T2M-GPT)...

:: ── MDM ──────────────────────────────────────────────────────
IF EXIST "%ROOT%\cloned-repos\mdm\.git" (
    echo  cloned-repos\mdm already cloned, skipping.
) ELSE (
    git clone --depth 1 https://github.com/GuyTevet/motion-diffusion-model.git "%ROOT%\cloned-repos\mdm"
    IF ERRORLEVEL 1 (echo  ERROR: Failed to clone MDM. & pause & exit /b 1)
    echo  MDM cloned to cloned-repos\mdm.
)

:: ── T2M-GPT ──────────────────────────────────────────────────
IF EXIST "%ROOT%\cloned-repos\t2m_gpt\.git" (
    echo  cloned-repos\t2m_gpt already cloned, skipping.
) ELSE (
    git clone --depth 1 https://github.com/Mael-zys/T2M-GPT.git "%ROOT%\cloned-repos\t2m_gpt"
    IF ERRORLEVEL 1 (
        echo  WARNING: Failed to clone T2M-GPT. Some features may not work.
    ) ELSE (
        echo  T2M-GPT cloned to cloned-repos\t2m_gpt.
    )
)

:: ── Download MDM 50-step checkpoint (~1.3 GB) ─────────────────
echo  Downloading MDM 50-step checkpoint (~1.3 GB)...
IF EXIST "%ROOT%\cloned-repos\mdm\save\humanml_enc_512_50steps\model000750000.pt" (
    echo  Checkpoint already present, skipping.
) ELSE (
    mkdir "%ROOT%\cloned-repos\mdm\save\humanml_enc_512_50steps" 2>nul
    python -c ^
        "import gdown, pathlib; ^
         out = r'%ROOT%\cloned-repos\mdm\save\humanml_enc_512_50steps\model000750000.pt'; ^
         gdown.download('https://drive.google.com/uc?id=1PE0PK8e5a5j6yYkaSi17NpvWAHqiGHLr', out, quiet=False)"
    IF ERRORLEVEL 1 (
        echo  ERROR: Failed to download MDM checkpoint.
        echo  Download manually from https://github.com/GuyTevet/motion-diffusion-model
        echo  and place at: cloned-repos\mdm\save\humanml_enc_512_50steps\model000750000.pt
        pause & exit /b 1
    )
    echo  MDM checkpoint downloaded.
)

:: ── Download SMPL body model ──────────────────────────────────
echo  Downloading SMPL body model...
IF EXIST "%ROOT%\cloned-repos\mdm\body_models\smpl\SMPL_NEUTRAL.pkl" (
    echo  SMPL already present, skipping.
) ELSE (
    mkdir "%ROOT%\cloned-repos\mdm\body_models\smpl" 2>nul
    python -c ^
        "import gdown, pathlib; ^
         out = r'%ROOT%\cloned-repos\mdm\body_models\smpl\SMPL_NEUTRAL.pkl'; ^
         gdown.download('https://drive.google.com/uc?id=1INYlGA76ak_cKGzvpOV2Pe6X4oLCDg7n', out, quiet=False)"
    IF ERRORLEVEL 1 (
        echo  ERROR: Failed to download SMPL. Manual download required.
        echo  See: https://smpl.is.tue.mpg.de/ (free registration required)
        echo  Place SMPL_NEUTRAL.pkl at: cloned-repos\mdm\body_models\smpl\SMPL_NEUTRAL.pkl
        echo  Continuing without SMPL — stickman tab may not work.
    )
)

:: ── Download HumanML3D mean/std normalization files ───────────
echo  Downloading HumanML3D normalization files...
IF EXIST "%ROOT%\cloned-repos\mdm\dataset\t2m_mean.npy" (
    echo  Normalization files already present, skipping.
) ELSE (
    mkdir "%ROOT%\cloned-repos\mdm\dataset" 2>nul
    python -c ^
        "import gdown; ^
         gdown.download('https://drive.google.com/uc?id=1tX79xk0fflp07EZ660Xz1RAFE33iEyJR', r'%ROOT%\cloned-repos\mdm\dataset\t2m_mean.npy', quiet=False); ^
         gdown.download('https://drive.google.com/uc?id=1tX79xk0fflp07EZ660Xz1RAFE33iEyJR', r'%ROOT%\cloned-repos\mdm\dataset\t2m_std.npy',  quiet=False)"
    IF ERRORLEVEL 1 (
        echo  WARNING: Failed to download normalization files. Stickman tab may fail.
    )
)

:: ── Download GloVe embeddings for MDM ────────────────────────
echo  Downloading GloVe / clip embeddings for MDM...
IF EXIST "%ROOT%\cloned-repos\mdm\glove\our_vab_data.npy" (
    echo  GloVe already present, skipping.
) ELSE (
    pushd "%ROOT%\cloned-repos\mdm"
    python -c ^
        "import gdown; gdown.download_folder('https://drive.google.com/drive/folders/1bCeS6Sh_mLVTebxIgiUHgdPrroW06mb6', output='glove', quiet=False)"
    IF ERRORLEVEL 1 (
        echo  WARNING: GloVe download failed. MDM text encoding may not work.
    )
    popd
)

:: ════════════════════════════════════════════════════════════
:: STEP 8 — Pre-download Wan2.1-T2V-1.3B (~3 GB)
:: ════════════════════════════════════════════════════════════
echo [8/10] Downloading Wan2.1-T2V-1.3B model (~3 GB)...
echo  This may take several minutes depending on your connection.
IF EXIST "%ROOT%\models\hf_cache\models--Wan-AI--Wan2.1-T2V-1.3B\snapshots" (
    echo  Wan2.1-T2V-1.3B already downloaded, skipping.
) ELSE (
    python -c "
from huggingface_hub import snapshot_download
import os
os.environ['HF_HOME'] = r'%ROOT%\models\hf_cache'
print('  Downloading Wan2.1-T2V-1.3B...')
snapshot_download('Wan-AI/Wan2.1-T2V-1.3B', cache_dir=r'%ROOT%\models\hf_cache')
print('  Wan2.1-T2V-1.3B downloaded successfully.')
"
    IF ERRORLEVEL 1 (
        echo  WARNING: Wan2.1-T2V-1.3B download failed.
        echo  The model will be downloaded automatically on first generation.
        echo  Ensure you have ~3 GB free disk space and a working internet connection.
    ) ELSE (
        echo  Wan2.1-T2V-1.3B downloaded successfully.
    )
)

:: ════════════════════════════════════════════════════════════
:: STEP 9 — HuggingFace token check (optional)
:: ════════════════════════════════════════════════════════════
echo [9/10] HuggingFace token check...
echo  CogVideoX-5B (~22 GB) and SVD 1.1 (~8 GB) are downloaded on first use.
echo  They do NOT require a HuggingFace token (both are public).
echo  First generation will take extra time for the download.
echo.
echo  If you want to pre-download now, uncomment the block in this script
echo  or run:  python -c "from diffusers import CogVideoXPipeline; CogVideoXPipeline.from_pretrained('THUDM/CogVideoX-5b')"
echo  (Requires ~22 GB free disk space and a fast internet connection)

:: ════════════════════════════════════════════════════════════
:: STEP 10 — Verify installation
:: ════════════════════════════════════════════════════════════
echo [10/10] Verifying installation...
python -c ^
    "import torch, flask, diffusers, transformers, clip, scipy, smplx; ^
     print('  torch:', torch.__version__); ^
     print('  diffusers:', diffusers.__version__); ^
     print('  transformers:', transformers.__version__); ^
     print('  CUDA available:', torch.cuda.is_available()); ^
     print('  All imports OK')"
IF ERRORLEVEL 1 (
    echo  WARNING: Some imports failed. Check the output above.
)

:: ════════════════════════════════════════════════════════════
:: Done!
:: ════════════════════════════════════════════════════════════
echo.
echo ==============================================================
echo  Setup complete!
echo ==============================================================
echo.
echo  Models downloaded on first use:
echo    Wan2.1-T2V-1.3B — ~3 GB   (Tab 4: Wan2.1 — pre-downloaded above)
echo    CogVideoX-5B    — ~22 GB  (Tab 1: Text to Video)
echo    SVD 1.1         — ~8 GB   (Tab 2: Image to Video)
echo    MDM checkpoint  — already downloaded above
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
