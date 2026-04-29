#!/usr/bin/env bash
# ============================================================
# AI Video Generation Service — Start Script
# Usage:  ./start.sh [port]   (default port: 7860)
# ============================================================
set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$BASE_DIR/venv"
APP="$BASE_DIR/app.py"
LOG="$BASE_DIR/logs/server.log"
PORT="${1:-7860}"

echo "=============================================="
echo " AI Video Generation Service"
echo " Base : $BASE_DIR"
echo " Port : $PORT"
echo "=============================================="

# Activate venv
source "$VENV/bin/activate"

# Ensure HuggingFace cache is on localdisk, not ~/.cache
export HF_HOME="$BASE_DIR/models/hf_cache"
export TRANSFORMERS_CACHE="$BASE_DIR/models/hf_cache"
export DIFFUSERS_CACHE="$BASE_DIR/models/hf_cache"
export TORCH_HOME="$BASE_DIR/models/torch_cache"
mkdir -p "$HF_HOME" "$TORCH_HOME"

echo ""
echo "Logs: $LOG"
echo "Open browser at: http://$(hostname -f):$PORT"
echo "(or http://localhost:$PORT if running locally)"
echo ""
echo "Press Ctrl+C to stop."
echo "----------------------------------------------"

exec python "$APP" --port "$PORT" 2>&1 | tee -a "$LOG"
