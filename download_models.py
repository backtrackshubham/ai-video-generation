#!/usr/bin/env python3
"""
AI Video Generation — Standalone Model Downloader

Downloads HuggingFace models into the correct cache directory.

Usage:
    python download_models.py                        # interactive menu
    python download_models.py all                    # download all models
    python download_models.py cogvideox svd          # download specific models
    python download_models.py --list                 # list status only
    python download_models.py --validate             # validate all downloaded models
    python download_models.py --validate cogvideox   # validate specific model(s)

Available model keys:
    cogvideox           CogVideoX-5B              (~22 GB)
    cogvideox15         CogVideoX-1.5-5B          (~31 GB)
    cogvideox-i2v       CogVideoX-5B-I2V          (~22 GB)
    cogvideox15-i2v     CogVideoX-1.5-5B-I2V      (~31 GB)
    modelscope          ModelScope 1.7B            (~33 GB)
    svd                 Stable Video Diffusion 1.1 (~30 GB)
    wan-1.3b            Wan2.1-T2V-1.3B            (~45 GB)
    wan-14b             Wan2.1-T2V-14B             (~80 GB)
    all                 All models above           (~300+ GB total)
"""

import os
import sys
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent
HF_CACHE  = BASE_DIR / "models" / "hf_cache"
LORA_DIR  = BASE_DIR / "models" / "loras"   # flat directory for LoRA files (avoids long paths)

os.environ["HF_HOME"]            = str(HF_CACHE)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE)
os.environ["DIFFUSERS_CACHE"]    = str(HF_CACHE)

HF_CACHE.mkdir(parents=True, exist_ok=True)
LORA_DIR.mkdir(parents=True, exist_ok=True)

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
        "size_gb": 22,
        "label":   "CogVideoX-5B",
    },
    "cogvideox15": {
        "hf_repo": "THUDM/CogVideoX1.5-5B",
        "size_gb": 31,
        "label":   "CogVideoX-1.5-5B",
    },
    "cogvideox-i2v": {
        "hf_repo": "THUDM/CogVideoX-5b-I2V",
        "size_gb": 22,
        "label":   "CogVideoX-5B-I2V",
    },
    "cogvideox15-i2v": {
        "hf_repo": "THUDM/CogVideoX1.5-5B-I2V",
        "size_gb": 31,
        "label":   "CogVideoX-1.5-5B-I2V",
    },
    "modelscope": {
        "hf_repo": "damo-vilab/text-to-video-ms-1.7b",
        "size_gb": 33,
        "label":   "ModelScope 1.7B",
    },
    "svd": {
        "hf_repo": "stabilityai/stable-video-diffusion-img2vid-xt",
        "size_gb": 30,
        "label":   "Stable Video Diffusion 1.1",
    },
    "wan-1.3b": {
        "hf_repo": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "size_gb": 45,
        "label":   "Wan2.1-T2V-1.3B",
    },
    "wan-14b": {
        "hf_repo": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "size_gb": 80,
        "label":   "Wan2.1-T2V-14B",
    },
    # ── Story Video pipeline ──────────────────────────────────────
    "sd15": {
        "hf_repo": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "size_gb": 4,
        "label":   "Stable Diffusion 1.5 (Story image gen)",
    },
    "qwen-7b": {
        "hf_repo": "Qwen/Qwen2.5-7B-Instruct",
        "size_gb": 15,
        "label":   "Qwen2.5-7B-Instruct FP16 (Story scene LLM)",
    },
    "qwen-7b-gguf": {
        "hf_repo": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "size_gb": 5,
        "label":   "Qwen2.5-7B-Instruct GGUF Q4 (Story scene LLM — fast on CPU)",
        "filename": "qwen2.5-7b-instruct-q4_k_m.gguf",
    },
    "phi3-mini": {
        "hf_repo": "microsoft/Phi-3.5-mini-instruct",
        "size_gb": 8,
        "label":   "Phi-3.5-Mini-Instruct (Story scene LLM, lightweight)",
    },
    "indic-f5": {
        "hf_repo": "ai4bharat/IndicF5",
        "size_gb": 2,
        "label":   "IndicF5 (Hindi TTS — requires HF login)",
    },
    "mms-tts-hindi": {
        "hf_repo": "facebook/mms-tts-hin",
        "size_gb": 0.4,
        "label":   "MMS-TTS Hindi (Hindi TTS — no login needed)",
    },
    # Style LoRAs for SD 1.5 — downloaded flat to models/loras/ to avoid Windows long path issues
    "lora-ghibli": {
        "hf_repo":  "artificialguybr/studioghibli-redmond-1-5v-studio-ghibli-lora-for-liberteredmond-sd-1-5",
        "filename": "StudioGhibliRedmond-15V-LiberteRedmond-StdGBRedmAF-StudioGhibli.safetensors",
        "size_gb":  0.1,
        "label":    "LoRA: Ghibli style",
        "lora":     True,
    },
    "lora-cartoon": {
        "hf_repo":  "artificialguybr/cutecartoon-redmond-1-5v-cute-cartoon-lora-for-liberteredmond-sd-1-5",
        "filename": "CuteCartoon15V-LiberteRedmodModel-Cartoon-CuteCartoonAF.safetensors",
        "size_gb":  0.1,
        "label":    "LoRA: Anime/Cartoon style",
        "lora":     True,
    },
    "lora-3d": {
        "hf_repo":  "artificialguybr/3d-redmond-1-5v-3d-render-style-for-liberte-redmond-sd-1-5",
        "filename": "3DRedmond-3DRenderStyle-3DRenderAF.safetensors",
        "size_gb":  0.1,
        "label":    "LoRA: Futuristic/Sci-Fi (3D render) style",
        "lora":     True,
    },
}


def _cache_name(hf_repo: str) -> str:
    return "models--" + hf_repo.replace("/", "--")


def _local_model_dir(hf_repo: str) -> Path:
    """Flat download dir for a model: models/hf_cache/<cache-name>/"""
    return HF_CACHE / _cache_name(hf_repo)


def is_downloaded(key: str) -> bool:
    info = MODELS[key]
    if info.get("lora"):
        # LoRAs stored flat in models/loras/<filename>
        return (LORA_DIR / info["filename"]).exists()
    # Non-LoRA models: downloaded with local_dir into models/hf_cache/<cache-name>/
    # Consider downloaded if directory is non-empty
    local_dir = _local_model_dir(info["hf_repo"])
    if not local_dir.exists():
        return False
    # Check for any real file (not just .cache metadata)
    return any(
        f for f in local_dir.rglob("*")
        if f.is_file() and ".cache" not in f.parts
    )


def print_status():
    print()
    print(f"  {'Key':<20} {'Label':<30} {'Size':>8}  {'Status'}")
    print(f"  {'-'*20} {'-'*30} {'-'*8}  {'-'*12}")
    for key, info in MODELS.items():
        status = green("downloaded") if is_downloaded(key) else yellow("not downloaded")
        print(f"  {key:<20} {info['label']:<30} {info['size_gb']:>6} GB  {status}")
    print()


def download_model(key: str):
    info = MODELS[key]
    if is_downloaded(key):
        print(green(f"  ✓ {info['label']} already downloaded"))
        return

    print(cyan(f"\n  Downloading {info['label']} (~{info['size_gb']} GB)…"))
    try:
        if info.get("lora"):
            # ── LoRA: single file, flat into models/loras/ ──
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id=info["hf_repo"],
                filename=info["filename"],
                local_dir=str(LORA_DIR),
            )
            # Ensure the file ended up at the expected flat location
            dest = LORA_DIR / info["filename"]
            if not dest.exists():
                for f in LORA_DIR.rglob(info["filename"]):
                    import shutil
                    shutil.copy2(str(f), str(dest))
                    break
        else:
            # ── Regular model: use local_dir (no symlinks, no deep cache) ──
            # This avoids WinError 1314 (symlink privilege) and long-path issues.
            from huggingface_hub import snapshot_download
            local_dir = _local_model_dir(info["hf_repo"])
            local_dir.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                info["hf_repo"],
                local_dir=str(local_dir),
            )

        print(green(f"  ✓ {info['label']} downloaded successfully"))
    except Exception as e:
        print(red(f"  ✗ Failed to download {info['label']}: {e}"))
        raise


# ── Validation ────────────────────────────────────────────────────────────────

def validate_model(key: str) -> bool:
    """Check a downloaded model for completeness by comparing local files
    against the HuggingFace Hub file listing (name + size).

    Falls back to a local-only size heuristic when Hub is unreachable.
    Returns True if valid, False if missing or corrupt files found.
    """
    info = MODELS[key]
    label = info["label"]

    if not is_downloaded(key):
        print(yellow(f"  ⚠  {label}: not downloaded — skipping validation"))
        return False

    print(cyan(f"\n  Validating {label}…"))

    hf_repo = info["hf_repo"]

    # ── LoRA: single-file check ───────────────────────────────────────────────
    if info.get("lora"):
        local_file = LORA_DIR / info["filename"]
        if not local_file.exists():
            print(red(f"  ✗ Missing: {info['filename']}"))
            return False
        local_size = local_file.stat().st_size
        try:
            from huggingface_hub import get_hf_file_metadata, hf_hub_url
            url = hf_hub_url(repo_id=hf_repo, filename=info["filename"])
            meta = get_hf_file_metadata(url)
            expected = meta.size
            if expected and local_size != expected:
                print(red(
                    f"  ✗ Size mismatch: {info['filename']} "
                    f"(local {local_size:,} B vs expected {expected:,} B)"
                ))
                return False
            print(green(f"  ✓ {info['filename']} — OK ({local_size / 1024**2:.1f} MB)"))
        except Exception as e:
            print(yellow(f"  ⚠  Hub unreachable ({e}); presence-only check passed"))
        return True

    # ── Regular model: try Hub comparison, fall back to size heuristic ────────
    local_dir = _local_model_dir(hf_repo)

    local_files: dict[str, int] = {}
    for f in local_dir.rglob("*"):
        if f.is_file() and ".cache" not in f.parts:
            rel = f.relative_to(local_dir).as_posix()
            local_files[rel] = f.stat().st_size

    if not local_files:
        print(red(f"  ✗ No files found in {local_dir}"))
        return False

    total_local_gb = sum(local_files.values()) / 1024**3

    # Try Hub comparison
    try:
        hub_files = {
            entry.rfilename: entry.size
            for entry in _list_repo_tree(hf_repo)
        }
    except Exception as e:
        # Offline / Hub unreachable — fall back to size heuristic
        expected_gb = info["size_gb"]
        ratio = total_local_gb / expected_gb if expected_gb else 1
        print(yellow(f"  ⚠  Hub unreachable ({e})"))
        print(yellow(f"     Offline heuristic: {total_local_gb:.2f} GB local vs ~{expected_gb} GB expected ({ratio*100:.0f}%)"))
        if ratio < 0.90:
            print(red(f"  ✗ Local data is only {ratio*100:.0f}% of expected size — likely incomplete"))
            return False
        print(green(f"  ✓ Size looks reasonable ({len(local_files)} files, {total_local_gb:.2f} GB)"))
        return True

    missing   = []
    corrupted = []
    ok_count  = 0

    for hub_path, hub_size in hub_files.items():
        if hub_path not in local_files:
            missing.append(hub_path)
        elif hub_size is not None and local_files[hub_path] != hub_size:
            corrupted.append((hub_path, local_files[hub_path], hub_size))
        else:
            ok_count += 1

    if missing:
        print(red(f"  ✗ Missing {len(missing)} file(s):"))
        for p in missing[:10]:
            print(red(f"      {p}"))
        if len(missing) > 10:
            print(red(f"      … and {len(missing) - 10} more"))

    if corrupted:
        print(red(f"  ✗ Size mismatch in {len(corrupted)} file(s):"))
        for p, got, exp in corrupted[:10]:
            print(red(f"      {p}  (local {got:,} B vs expected {exp:,} B)"))
        if len(corrupted) > 10:
            print(red(f"      … and {len(corrupted) - 10} more"))

    if not missing and not corrupted:
        print(green(
            f"  ✓ All {ok_count} file(s) present and correct "
            f"({total_local_gb:.2f} GB on disk)"
        ))
        return True

    return False


def _list_repo_tree(hf_repo: str):
    """Return an iterable of RepoFile objects with .rfilename and .size."""
    from huggingface_hub import list_repo_tree
    try:
        # list_repo_tree available in huggingface_hub >= 0.22
        return list(list_repo_tree(hf_repo, recursive=True))
    except AttributeError:
        pass
    # Fallback for older versions: list_repo_files gives names only (no size)
    from huggingface_hub import list_repo_files

    class _FakeEntry:
        def __init__(self, name):
            self.rfilename = name
            self.size = None   # can't check sizes without metadata

    return [_FakeEntry(n) for n in list_repo_files(hf_repo)]


def validate_all(keys=None) -> None:
    """Validate all (or specified) downloaded models and print a summary."""
    targets = keys if keys else list(MODELS.keys())
    downloaded = [k for k in targets if is_downloaded(k)]

    if not downloaded:
        print(yellow("  No downloaded models to validate."))
        return

    results = {}
    for key in downloaded:
        results[key] = validate_model(key)

    # Summary table
    print()
    print(cyan("  Validation Summary"))
    print(cyan("  " + "─" * 54))
    for key, ok in results.items():
        label = MODELS[key]["label"]
        status = green("✓ OK") if ok else red("✗ INCOMPLETE / CORRUPT")
        print(f"  {key:<20} {label:<30}  {status}")
    print()

    bad = [k for k, ok in results.items() if not ok]
    if bad:
        print(red(f"  {len(bad)} model(s) need re-downloading:"))
        for k in bad:
            print(red(f"    python download_models.py {k}"))
    else:
        print(green("  All validated models are complete."))
    print()


def interactive_menu():
    print()
    print(cyan("=" * 56))
    print(cyan("  AI Video Generation — Model Downloader"))
    print(cyan("=" * 56))
    print_status()

    print(cyan("  Actions:"))
    print("    [d] Download models")
    print("    [v] Validate downloaded models")
    print("    [q] Quit")
    print()
    action = input(cyan("  Your choice [d/v/q]: ")).strip().lower()

    if action in ("q", ""):
        return

    if action == "v":
        validate_all()
        return

    # ── Download flow ─────────────────────────────────────────────────────────
    keys_available = [k for k in MODELS if not is_downloaded(k)]
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
                        help="Model keys to download/validate (or 'all'). If omitted, shows interactive menu.")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List model download status and exit")
    parser.add_argument("--validate", "-v", action="store_true",
                        help="Validate downloaded models for completeness (checks file list + sizes against HF Hub)")
    args = parser.parse_args()

    if args.list:
        print_status()
        return

    if args.validate:
        # --validate [key1 key2 ...]  or  --validate (all downloaded)
        if args.models and "all" not in args.models:
            for k in args.models:
                if k not in MODELS:
                    print(red(f"  Unknown model key: {k}"))
                    sys.exit(1)
            validate_all(keys=args.models)
        else:
            validate_all()
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
