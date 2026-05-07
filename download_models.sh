#!/usr/bin/env bash
# ============================================================
# AI Video Generation — Pre-download all HuggingFace models
#
# Run this once after setup_linux.sh to cache all models
# before starting the server. Each model is downloaded once
# and cached in models/hf_cache/ for all future runs.
#
# Usage:
#   ./download_models.sh [model]
#
# Examples:
#   ./download_models.sh              # download all recommended models
#   ./download_models.sh wan13b       # Wan2.1-T2V-1.3B only  (~3 GB)
#   ./download_models.sh wan14b       # Wan2.1-T2V-14B only   (~28 GB)
#   ./download_models.sh cogvideox    # CogVideoX-5B T2V only
#   ./download_models.sh cogvideox15  # CogVideoX-1.5-5B T2V only
#   ./download_models.sh i2v          # CogVideoX-5B-I2V only
#   ./download_models.sh i2v15        # CogVideoX-1.5-5B-I2V only
#   ./download_models.sh svd          # Stable Video Diffusion only
#   ./download_models.sh modelscope   # ModelScope (has watermarks)
#
# Model sizes (approximate):
#   Wan2.1-T2V-1.3B       ~3 GB    Tab 4 — Wan2.1 (consumer GPU, 6 GB VRAM)
#   Wan2.1-T2V-14B        ~28 GB   Tab 4 — Wan2.1 (high quality, 16+ GB VRAM)
#   CogVideoX-5B          ~21 GB   Tab 1 — Text-to-Video
#   CogVideoX-1.5-5B      ~20 GB   Tab 1 — Text-to-Video (up to 81 frames / ~10s)
#   CogVideoX-5B-I2V      ~20 GB   Tab 1 — Image+Text-to-Video
#   CogVideoX-1.5-5B-I2V  ~20 GB   Tab 1 — Image+Text-to-Video (~10s/clip)
#   SVD XT 1.1            ~8.4 GB  Tab 2 — Image-to-Video (no prompt)
#   ModelScope 1.7B       ~6.9 GB  Legacy (Shutterstock watermarks)
# ============================================================
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$BASE_DIR/venv"
CACHE_DIR="$BASE_DIR/models/hf_cache"
LOG_DIR="$BASE_DIR/gen-logs"
TARGET="${1:-all}"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[download]${NC} $*"; }
warn()    { echo -e "${YELLOW}[warn]${NC}    $*"; }
section() { echo -e "\n${GREEN}── $* ──${NC}"; }

if [[ ! -d "$VENV" ]]; then
    echo "venv not found at $VENV — run ./setup_linux.sh first." >&2
    exit 1
fi

source "$VENV/bin/activate"
mkdir -p "$CACHE_DIR" "$LOG_DIR"

download_hf_model() {
    local name="$1"
    local model_id="$2"
    local pipeline_class="$3"
    local extra_args="${4:-}"
    local log_file="$LOG_DIR/download-${name}.log"

    section "Downloading $name ($model_id)"
    info "Log: $log_file"

    python3 - <<PYEOF 2>&1 | tee "$log_file"
import torch, sys
from diffusers import $pipeline_class

print(f"Downloading $name ...")
try:
    $pipeline_class.from_pretrained(
        "$model_id",
        cache_dir="$CACHE_DIR",
        torch_dtype=torch.float32,
        $extra_args
    )
    print("✓ $name download complete")
except Exception as e:
    print(f"✗ $name download failed: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
    info "$name cached."
}

download_hf_snapshot() {
    local name="$1"
    local model_id="$2"
    local log_file="$LOG_DIR/download-${name}.log"

    section "Downloading $name ($model_id)"
    info "Log: $log_file"

    python3 - <<PYEOF 2>&1 | tee "$log_file"
import sys
from huggingface_hub import snapshot_download

print(f"Downloading $name via snapshot_download...")
try:
    snapshot_download("$model_id", cache_dir="$CACHE_DIR")
    print("✓ $name download complete")
except Exception as e:
    print(f"✗ $name download failed: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
    info "$name cached."
}

do_wan13b()      { download_hf_snapshot "wan2.1-t2v-1.3b" "Wan-AI/Wan2.1-T2V-1.3B"; }
do_wan14b()      { download_hf_snapshot "wan2.1-t2v-14b"  "Wan-AI/Wan2.1-T2V-14B"; }
do_cogvideox()   { download_hf_model "cogvideox-5b"        "THUDM/CogVideoX-5b"           "CogVideoXPipeline"; }
do_cogvideox15() { download_hf_model "cogvideox-1.5-5b"    "THUDM/CogVideoX1.5-5B"        "CogVideoXPipeline"; }
do_i2v()         { download_hf_model "cogvideox-5b-i2v"    "THUDM/CogVideoX-5b-I2V"       "CogVideoXImageToVideoPipeline"; }
do_i2v15()       { download_hf_model "cogvideox-1.5-5b-i2v" "THUDM/CogVideoX1.5-5B-I2V"  "CogVideoXImageToVideoPipeline"; }
do_svd()         { download_hf_model "svd-xt-1-1"          "stabilityai/stable-video-diffusion-img2vid-xt" "StableVideoDiffusionPipeline"; }
do_modelscope()  {
    warn "ModelScope 1.7B has Shutterstock watermarks baked into the weights."
    download_hf_model "modelscope-1.7b" "damo-vilab/text-to-video-ms-1.7b" "DiffusionPipeline" 'trust_remote_code=True,'
}

case "$TARGET" in
    all)
        do_wan13b
        do_cogvideox
        do_cogvideox15
        do_i2v
        do_i2v15
        do_svd
        warn "Skipping ModelScope (watermarks). Run './download_models.sh modelscope' to force."
        warn "Skipping Wan2.1-14B (28 GB). Run './download_models.sh wan14b' if you have 16+ GB VRAM."
        ;;
    wan13b)      do_wan13b ;;
    wan14b)      do_wan14b ;;
    cogvideox)   do_cogvideox ;;
    cogvideox15) do_cogvideox15 ;;
    i2v)         do_i2v ;;
    i2v15)       do_i2v15 ;;
    svd)         do_svd ;;
    modelscope)  do_modelscope ;;
    *)
        echo "Unknown target '$TARGET'." >&2
        echo "Valid: all, wan13b, wan14b, cogvideox, cogvideox15, i2v, i2v15, svd, modelscope" >&2
        exit 1
        ;;
esac

section "Done"
info "Models cached at: $CACHE_DIR"
info "Start the server: ./start.sh"
