#!/usr/bin/env bash
# AI Video Generation — Linux/macOS Launcher
# Delegates to start.py (cross-platform launcher)
#
# Usage:  ./start.sh [port]   (default: 7860)

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${1:-7860}"

if [[ ! -f "$BASE_DIR/venv/bin/python" ]]; then
    echo ""
    echo "  ERROR: Virtual environment not found."
    echo "  Please run:  python3 setup.py"
    echo ""
    exit 1
fi

exec "$BASE_DIR/venv/bin/python" "$BASE_DIR/start.py" "$PORT"
