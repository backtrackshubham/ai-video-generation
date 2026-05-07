#!/usr/bin/env bash
# ============================================================
# AI Video Generation — Linux Setup Script
#
# What this does:
#   1. Creates a Python 3.8+ virtual environment
#   2. Installs all Python dependencies
#   3. Clones third-party repos into cloned-repos/
#   4. Downloads MDM checkpoint, SMPL body model, GloVe embeddings
#   5. Creates symlinks so app.py can find everything
#
# Usage:
#   chmod +x setup_linux.sh
#   ./setup_linux.sh
#
# Re-running is safe — each step is idempotent.
# ============================================================
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLONED_DIR="$BASE_DIR/cloned-repos"
MDM_DIR="$CLONED_DIR/mdm"
MODELS_DIR="$BASE_DIR/models"
VENV="$BASE_DIR/venv"

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "${GREEN}[setup]${NC} $*"; }
warn()    { echo -e "${YELLOW}[warn]${NC}  $*"; }
section() { echo -e "\n${GREEN}══════════════════════════════════════════${NC}"; \
            echo -e "${GREEN} $*${NC}"; \
            echo -e "${GREEN}══════════════════════════════════════════${NC}"; }

# ── Check required tools ──────────────────────────────────────────────────────
section "Checking prerequisites"
for cmd in python3 pip git wget; do
    if command -v "$cmd" &>/dev/null; then
        info "$cmd found: $(command -v $cmd)"
    else
        echo -e "${RED}[error]${NC} '$cmd' is required but not installed. Aborting." >&2
        exit 1
    fi
done

# Check Python version >= 3.8
PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYOK=$(python3 -c "import sys; print(int(sys.version_info >= (3,8)))")
if [[ "$PYOK" != "1" ]]; then
    echo -e "${RED}[error]${NC} Python 3.8+ required, found $PYVER" >&2
    exit 1
fi
info "Python $PYVER OK"

# ── Virtual environment ───────────────────────────────────────────────────────
section "Virtual environment"
if [[ ! -d "$VENV" ]]; then
    info "Creating venv at $VENV"
    python3 -m venv "$VENV"
else
    info "venv already exists — skipping creation"
fi

# shellcheck disable=SC1090
source "$VENV/bin/activate"
info "Activated: $VIRTUAL_ENV"

# ── Python dependencies ───────────────────────────────────────────────────────
section "Installing Python dependencies"
pip install --upgrade pip --quiet

pip install \
    "numpy>=1.24,<2.0" \
    "flask>=2.3.0" \
    "flask-cors>=4.0.0" \
    "torch>=2.4.0" torchvision torchaudio \
    "diffusers>=0.32.0" \
    "accelerate>=0.30.0" \
    "transformers>=4.40.0" \
    "huggingface_hub>=0.23.0" \
    "tokenizers>=0.19.0" \
    sentencepiece \
    "imageio>=2.28.0" imageio-ffmpeg \
    "moviepy<2.0" \
    "scipy>=1.11.0" \
    scikit-learn \
    "matplotlib>=3.7.0" \
    smplx \
    chumpy \
    ftfy \
    regex \
    gdown \
    Pillow \
    tqdm \
    einops \
    opencv-python \
    --quiet

pip install git+https://github.com/openai/CLIP.git --quiet

info "Python dependencies installed"

# Patch chumpy for numpy >= 1.24 compatibility
CHUMPY_INIT="$VENV/lib64/python3.8/site-packages/chumpy/__init__.py"
if [[ ! -f "$CHUMPY_INIT" ]]; then
    # Try lib (non-lib64 fallback)
    CHUMPY_INIT="$VENV/lib/python3.8/site-packages/chumpy/__init__.py"
fi
if [[ -f "$CHUMPY_INIT" ]]; then
    if grep -q "from numpy import bool," "$CHUMPY_INIT" 2>/dev/null; then
        info "Patching chumpy for numpy >= 1.24 compatibility"
        sed -i 's/from numpy import bool, int, float, complex, object, str, nan, inf/import numpy as _np; bool=_np.bool_; int=_np.int_; float=_np.float64; complex=_np.complex128; object=_np.object_; str=_np.str_; nan=_np.nan; inf=_np.inf/' "$CHUMPY_INIT"
        info "chumpy patched"
    else
        info "chumpy already patched or uses newer API"
    fi
else
    warn "chumpy not found at expected path — skipping patch"
fi

# ── Clone third-party repos ───────────────────────────────────────────────────
section "Cloning third-party repos → cloned-repos/"
mkdir -p "$CLONED_DIR"

# MDM — Motion Diffusion Model
if [[ ! -d "$MDM_DIR/.git" ]]; then
    info "Cloning MDM..."
    git clone --depth 1 https://github.com/GuyTevet/motion-diffusion-model.git "$MDM_DIR"
    info "MDM cloned"
else
    info "MDM already cloned at $MDM_DIR — skipping"
fi

# T2M-GPT
T2M_DIR="$CLONED_DIR/t2m_gpt"
if [[ ! -d "$T2M_DIR/.git" ]]; then
    info "Cloning T2M-GPT..."
    git clone --depth 1 https://github.com/Mael-zys/T2M-GPT.git "$T2M_DIR"
    info "T2M-GPT cloned"
else
    info "T2M-GPT already cloned at $T2M_DIR — skipping"
fi

# ── MDM model checkpoint ──────────────────────────────────────────────────────
section "Downloading MDM model checkpoint"
MDM_CKPT_DIR="$MDM_DIR/save/humanml_enc_512_50steps"
MDM_CKPT="$MDM_CKPT_DIR/model000750000.pt"
mkdir -p "$MDM_CKPT_DIR"

if [[ ! -f "$MDM_CKPT" ]]; then
    info "Downloading MDM checkpoint (~400 MB)..."
    # Google Drive file ID for humanml_enc_512_50steps checkpoint
    gdown --id 1PE0PK8e5a5j6yYl0jmJL4NV7jkMn6UGE -O "$MDM_CKPT_DIR/checkpoints.zip" || {
        warn "gdown failed. Trying direct wget fallback..."
        wget -q --show-progress \
            "https://drive.google.com/uc?export=download&id=1PE0PK8e5a5j6yYl0jmJL4NV7jkMn6UGE" \
            -O "$MDM_CKPT_DIR/checkpoints.zip"
    }
    info "Extracting checkpoint..."
    unzip -q "$MDM_CKPT_DIR/checkpoints.zip" -d "$MDM_CKPT_DIR/"
    rm "$MDM_CKPT_DIR/checkpoints.zip"
    info "MDM checkpoint extracted"
else
    info "MDM checkpoint already present — skipping"
fi

# args.json is required alongside the checkpoint
if [[ ! -f "$MDM_CKPT_DIR/args.json" ]]; then
    info "Downloading MDM args.json..."
    gdown --id 1PE0PK8e5a5j6yYl0jmJL4NV7jkMn6UGE --fuzzy -O "$MDM_CKPT_DIR/" 2>/dev/null || \
        warn "args.json download failed — you may need to download it manually from the MDM repo."
fi

# ── SMPL body model ───────────────────────────────────────────────────────────
section "SMPL body model"
SMPL_DIR="$MDM_DIR/body_models/smpl"
SMPL_FILE="$SMPL_DIR/SMPL_NEUTRAL.pkl"
mkdir -p "$SMPL_DIR"

if [[ ! -f "$SMPL_FILE" ]]; then
    warn "SMPL_NEUTRAL.pkl not found."
    echo ""
    echo "  SMPL requires a free registration at https://smpl.is.tue.mpg.de/"
    echo "  After registering, download 'SMPL for Python Users' and place:"
    echo "    SMPL_NEUTRAL.pkl → $SMPL_DIR/"
    echo ""
    echo "  Alternatively, copy from an existing download:"
    echo "    cp /path/to/SMPL_NEUTRAL.pkl $SMPL_DIR/"
    echo ""
    warn "Skipping SMPL — Tab 3 (Motion Stickman) will not work until this is in place."
else
    info "SMPL_NEUTRAL.pkl already present"
fi

# ── GloVe embeddings ──────────────────────────────────────────────────────────
section "GloVe embeddings for MDM"
GLOVE_DIR="$MDM_DIR/glove"
mkdir -p "$GLOVE_DIR"

if [[ ! -f "$GLOVE_DIR/our_vab_data.npy" ]]; then
    info "Downloading GloVe embeddings (~10 MB)..."
    gdown --id 1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n -O "$GLOVE_DIR/glove.zip" || {
        warn "gdown failed for GloVe. Trying wget..."
        wget -q --show-progress \
            "https://drive.google.com/uc?export=download&id=1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n" \
            -O "$GLOVE_DIR/glove.zip"
    }
    unzip -q "$GLOVE_DIR/glove.zip" -d "$GLOVE_DIR/"
    rm "$GLOVE_DIR/glove.zip"
    info "GloVe extracted"
else
    info "GloVe embeddings already present"
fi

# ── MDM dataset normalisation stats ──────────────────────────────────────────
section "MDM normalisation stats (t2m mean/std)"
DATASET_DIR="$MDM_DIR/dataset"
mkdir -p "$DATASET_DIR"

if [[ ! -f "$DATASET_DIR/t2m_mean.npy" ]]; then
    info "Downloading t2m mean/std (~1 MB)..."
    gdown --id 1metMirQ-lCJBsUjEPAfzUOjKAQxWwUxN -O "$DATASET_DIR/t2m_mean.npy" 2>/dev/null || \
        warn "t2m_mean.npy download failed — copy manually if Tab 3 errors on normalization."
    gdown --id 1metMirQ-lCJBsUjEPAfzUOjKAQxWwUxN -O "$DATASET_DIR/t2m_std.npy" 2>/dev/null || \
        warn "t2m_std.npy download failed."
else
    info "t2m mean/std already present"
fi

# ── HuggingFace model cache dirs ─────────────────────────────────────────────
section "Creating model cache directories"
mkdir -p "$MODELS_DIR/hf_cache"
mkdir -p "$MODELS_DIR/torch_cache"
info "Model cache dirs ready at $MODELS_DIR"
info "HuggingFace models are downloaded on first use (~36 GB total):"
info "  - THUDM/CogVideoX-5b           ~21 GB  (Tab 1 — Text-to-Video)"
info "  - stabilityai/stable-video-diffusion-img2vid-xt  ~8.4 GB  (Tab 2 — Image-to-Video)"
info "  - damo-vilab/text-to-video-ms-1.7b  ~6.9 GB  (legacy, may be removed)"

# ── Output / upload dirs ──────────────────────────────────────────────────────
section "Creating runtime directories"
mkdir -p "$BASE_DIR/outputs/normal-videos"
mkdir -p "$BASE_DIR/outputs/i2v-videos"
mkdir -p "$BASE_DIR/outputs/stickman-videos"
mkdir -p "$BASE_DIR/outputs/wan-videos"
mkdir -p "$BASE_DIR/uploads"
mkdir -p "$BASE_DIR/gen-logs"
mkdir -p "$BASE_DIR/gen-logs/normal-videos"
mkdir -p "$BASE_DIR/gen-logs/i2v-videos"
mkdir -p "$BASE_DIR/gen-logs/stickman-videos"
mkdir -p "$BASE_DIR/gen-logs/wan-videos"
info "Runtime directories created"

# ── Done ──────────────────────────────────────────────────────────────────────
section "Setup complete"
echo ""
echo "  Start the server with:"
echo "    ./start.sh"
echo ""
echo "  Open browser at:"
echo "    http://$(hostname -f):8083   (from other machines)"
echo "    http://localhost:8083        (local)"
echo ""
if [[ ! -f "$SMPL_FILE" ]]; then
    warn "SMPL_NEUTRAL.pkl is missing — Tab 3 (Motion Stickman) won't work."
    warn "See instructions above for how to get it."
fi
