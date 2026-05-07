#!/usr/bin/env python3
"""
AI Video Generation — Standalone Model Downloader

Downloads HuggingFace models into the correct cache directory.

Usage:
    python download_models.py                  # interactive menu
    python download_models.py all              # download all models
    python download_models.py cogvideox svd    # download specific models
    python download_models.py --list           # list status only

Available model keys:
    cogvideox           CogVideoX-5B             (~21 GB)
    cogvideox15         CogVideoX-1.5-5B         (~20 GB)
    cogvideox-i2v       CogVideoX-5B-I2V         (~20 GB)
    cogvideox15-i2v     CogVideoX-1.5-5B-I2V     (~20 GB)
    modelscope          ModelScope 1.7B           (~4 GB)
    svd                 Stable Video Diffusion 1.1 (~8 GB)
    wan-1.3b            Wan2.1-T2V-1.3B          (~3 GB)
    wan-14b             Wan2.1-T2V-14B            (~28 GB)
    all                 All models above          (~100+ GB total)
"""

import os
import sys
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent
HF_CACHE  = BASE_DIR / "models" / "hf_cache"

os.environ["HF_HOME"]            = str(HF_CACHE)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE)
os.environ["DIFFUSERS_CACHE"]    = str(HF_CACHE)

HF_CACHE.mkdir(parents=True, exist_ok=True)

# ── Colours ───────────────────────────────────────────────────────────────────
import platform
_ANSI = platform.system() != "Windows" or os.environ.get("TERM") == "xterm"
def green(t):  return f"\033[0;32m{t}\033[0m" if _ANSI else t
def yellow(t): return f"\033[1;33m{t}\033[0m" if _ANSI else t
def cyan(t):   return f"\033[0;36m{t}\033[0m" if _ANSI else t
def red(t):    return f"\033[0;31m{t}\033[0m" if _ANSI else t

# ── Model registry ────────────────────────────────────────────────────────────
MODELS = {
    "cogvideox": {
        "hf_repo": "THUDM/CogVideoX-5b",
        "size_gb": 21,
        "label":   "CogVideoX-5B",
    },
    "cogvideox15": {
        "hf_repo": "THUDM/CogVideoX1.5-5B",
        "size_gb": 20,
        "label":   "CogVideoX-1.5-5B",
    },
    "cogvideox-i2v": {
        "hf_repo": "THUDM/CogVideoX-5b-I2V",
        "size_gb": 20,
        "label":   "CogVideoX-5B-I2V",
    },
    "cogvideox15-i2v": {
        "hf_repo": "THUDM/CogVideoX1.5-5B-I2V",
        "size_gb": 20,
        "label":   "CogVideoX-1.5-5B-I2V",
    },
    "modelscope": {
        "hf_repo": "damo-vilab/text-to-video-ms-1.7b",
        "size_gb": 4,
        "label":   "ModelScope 1.7B",
    },
    "svd": {
        "hf_repo": "stabilityai/stable-video-diffusion-img2vid-xt",
        "size_gb": 8,
        "label":   "Stable Video Diffusion 1.1",
    },
    "wan-1.3b": {
        "hf_repo": "Wan-AI/Wan2.1-T2V-1.3B",
        "size_gb": 3,
        "label":   "Wan2.1-T2V-1.3B",
    },
    "wan-14b": {
        "hf_repo": "Wan-AI/Wan2.1-T2V-14B",
        "size_gb": 28,
        "label":   "Wan2.1-T2V-14B",
    },
}


def _cache_name(hf_repo: str) -> str:
    return "models--" + hf_repo.replace("/", "--")


def is_downloaded(hf_repo: str) -> bool:
    snap_dir = HF_CACHE / _cache_name(hf_repo) / "snapshots"
    if not snap_dir.exists():
        return False
    for s in snap_dir.iterdir():
        if s.is_dir() and any(s.iterdir()):
            return True
    return False


def print_status():
    print()
    print(f"  {'Key':<20} {'Label':<30} {'Size':>8}  {'Status'}")
    print(f"  {'-'*20} {'-'*30} {'-'*8}  {'-'*12}")
    for key, info in MODELS.items():
        status = green("downloaded") if is_downloaded(info["hf_repo"]) else yellow("not downloaded")
        print(f"  {key:<20} {info['label']:<30} {info['size_gb']:>6} GB  {status}")
    print()


def download_model(key: str):
    from huggingface_hub import snapshot_download
    info = MODELS[key]
    if is_downloaded(info["hf_repo"]):
        print(green(f"  ✓ {info['label']} already downloaded"))
        return

    print(cyan(f"\n  Downloading {info['label']} (~{info['size_gb']} GB)…"))
    try:
        snapshot_download(info["hf_repo"], cache_dir=str(HF_CACHE))
        print(green(f"  ✓ {info['label']} downloaded successfully"))
    except Exception as e:
        print(red(f"  ✗ Failed to download {info['label']}: {e}"))
        raise


def interactive_menu():
    print()
    print(cyan("=" * 56))
    print(cyan("  AI Video Generation — Model Downloader"))
    print(cyan("=" * 56))
    print_status()

    keys_available = [k for k, v in MODELS.items() if not is_downloaded(v["hf_repo"])]
    if not keys_available:
        print(green("  All models already downloaded!"))
        return

    print("  Select models to download (space-separated numbers, or 'all'):")
    not_downloaded = list(keys_available)
    for i, k in enumerate(not_downloaded, 1):
        info = MODELS[k]
        print(f"    [{i}] {info['label']} (~{info['size_gb']} GB)")
    print()

    raw = input(cyan("  Your choice: ")).strip()
    if not raw:
        print("  Aborted.")
        return

    if raw.lower() == "all":
        selected = not_downloaded
    else:
        selected = []
        for tok in raw.split():
            if tok.isdigit() and 1 <= int(tok) <= len(not_downloaded):
                selected.append(not_downloaded[int(tok) - 1])
            elif tok in MODELS:
                selected.append(tok)

    if not selected:
        print("  No valid selection. Aborted.")
        return

    total_gb = sum(MODELS[k]["size_gb"] for k in selected)
    print()
    print(f"  Will download {len(selected)} model(s), ~{total_gb} GB total.")
    confirm = input(cyan("  Proceed? [Y/n] ")).strip().lower()
    if confirm not in ("", "y", "yes"):
        print("  Aborted.")
        return

    for key in selected:
        download_model(key)

    print()
    print(green("  Done!"))
    print_status()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Download AI video generation models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("models", nargs="*",
                        help="Model keys to download (or 'all'). If omitted, shows interactive menu.")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List model download status and exit")
    args = parser.parse_args()

    if args.list:
        print_status()
        return

    if not args.models:
        interactive_menu()
        return

    keys_to_download = []
    if "all" in args.models:
        keys_to_download = list(MODELS.keys())
    else:
        for k in args.models:
            if k not in MODELS:
                print(red(f"  Unknown model key: {k}"))
                print(f"  Valid keys: {', '.join(MODELS.keys())}")
                sys.exit(1)
            keys_to_download.append(k)

    print()
    for key in keys_to_download:
        download_model(key)
    print()
    print_status()


if __name__ == "__main__":
    main()
