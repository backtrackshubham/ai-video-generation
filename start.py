#!/usr/bin/env python3
"""
AI Video Generation — Unified Start Script
Works on Windows and Linux/macOS.

Usage:
    python start.py          # default port 7860
    python start.py 8080     # custom port
"""

import os
import platform
import subprocess
import sys
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent
VENV_DIR  = BASE_DIR / "venv"
APP       = BASE_DIR / "app.py"
LOG_DIR   = BASE_DIR / "gen-logs"
LOG_FILE  = LOG_DIR / "server.log"

_IS_WIN = platform.system() == "Windows"

if _IS_WIN:
    PYTHON = VENV_DIR / "Scripts" / "python.exe"
else:
    PYTHON = VENV_DIR / "bin" / "python"

# ── Venv isolation guard ──────────────────────────────────────────────────────
# If start.py is invoked from outside the repo-local venv, re-exec using
# the repo-local venv's Python so the correct packages are always used.
if __name__ == "__main__":
    active_venv = os.environ.get("VIRTUAL_ENV", "")
    in_repo_venv = PYTHON.exists() and (
        Path(active_venv).resolve() == VENV_DIR.resolve() if active_venv else False
    )
    using_repo_python = Path(sys.executable).resolve() == PYTHON.resolve()

    if not using_repo_python and PYTHON.exists():
        # Re-exec with the repo-local venv Python
        if active_venv and Path(active_venv).resolve() != VENV_DIR.resolve():
            print(f"[start] Foreign venv detected: {active_venv}")
        print(f"[start] Using repo-local venv: {PYTHON}")
        clean_env = {k: v for k, v in os.environ.items()
                     if k not in ("VIRTUAL_ENV", "PYTHONHOME")}
        if active_venv:
            path_parts = clean_env.get("PATH", "").split(os.pathsep)
            path_parts = [p for p in path_parts if not p.startswith(active_venv)]
            clean_env["PATH"] = os.pathsep.join(path_parts)
        result = subprocess.run([str(PYTHON)] + sys.argv, env=clean_env)
        sys.exit(result.returncode)


def fail(msg):
    print(f"\n  ERROR: {msg}\n")
    sys.exit(1)


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "7860"

    # ── Sanity checks ─────────────────────────────────────────────────────────
    if not PYTHON.exists():
        fail(
            "Virtual environment not found.\n"
            "  Please run:  python setup.py"
        )

    cloned_mdm = BASE_DIR / "cloned-repos" / "mdm"
    if not cloned_mdm.exists():
        fail(
            "cloned-repos/mdm not found.\n"
            "  Please run:  python setup.py"
        )

    # ── Set cache env vars ────────────────────────────────────────────────────
    hf_cache    = BASE_DIR / "models" / "hf_cache"
    torch_cache = BASE_DIR / "models" / "torch_cache"
    xdg_cache   = BASE_DIR / "models" / "xdg_cache"

    env = os.environ.copy()
    env["HF_HOME"]            = str(hf_cache)
    env["TRANSFORMERS_CACHE"] = str(hf_cache)
    env["DIFFUSERS_CACHE"]    = str(hf_cache)
    env["TORCH_HOME"]         = str(torch_cache)
    env["XDG_CACHE_HOME"]     = str(xdg_cache)

    hf_cache.mkdir(parents=True, exist_ok=True)
    torch_cache.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Print banner ──────────────────────────────────────────────────────────
    print()
    print("=" * 50)
    print("  AI Video Generation Service")
    print(f"  Base : {BASE_DIR}")
    print(f"  Port : {port}")
    print(f"  Log  : {LOG_FILE}")
    print("=" * 50)

    if _IS_WIN:
        print(f"  URL  : http://localhost:{port}")
    else:
        hostname = subprocess.run(
            ["hostname", "-f"], capture_output=True, text=True
        ).stdout.strip() or "localhost"
        print(f"  URL  : http://{hostname}:{port}")
        print(f"  URL  : http://localhost:{port}  (local)")

    print()
    print("  Press Ctrl+C to stop.")
    print("-" * 50)
    print()

    # ── Launch app ────────────────────────────────────────────────────────────
    cmd = [str(PYTHON), str(APP), "--port", str(port)]

    if _IS_WIN:
        # On Windows just run directly; tee isn't available in CMD by default
        subprocess.run(cmd, env=env)
    else:
        # On Linux/macOS, tee to log file
        with open(LOG_FILE, "a") as log:
            proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
            try:
                for line in proc.stdout:
                    decoded = line.decode(errors="replace")
                    sys.stdout.write(decoded)
                    sys.stdout.flush()
                    log.write(decoded)
                    log.flush()
            except KeyboardInterrupt:
                proc.terminate()
            proc.wait()


if __name__ == "__main__":
    main()
