#!/usr/bin/env python3
"""
AI Video Generation — Unified Setup Script
Works on Windows and Linux/macOS.

Usage:
    python setup.py           # auto-detects platform, asks questions
    python setup.py --yes     # non-interactive, accept all defaults
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# ── Colours (disabled on Windows unless ANSI is supported) ────────────────────
_IS_WIN = platform.system() == "Windows"
_ANSI   = not _IS_WIN or os.environ.get("TERM") == "xterm"

def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if _ANSI else text

def green(t):  return _c("0;32", t)
def yellow(t): return _c("1;33", t)
def red(t):    return _c("0;31", t)
def cyan(t):   return _c("0;36", t)

def info(msg):    print(green("[setup] ") + msg)
def warn(msg):    print(yellow("[warn]  ") + msg)
def error(msg):   print(red("[error] ") + msg); sys.exit(1)
def section(msg): print("\n" + green("=" * 52) + "\n" + green(f"  {msg}") + "\n" + green("=" * 52))

# ── Helpers ───────────────────────────────────────────────────────────────────

def run(cmd, check=True, capture=False, **kwargs):
    """Run a shell command. cmd is a list or string."""
    if isinstance(cmd, str):
        kwargs["shell"] = True
    stdout = subprocess.PIPE if capture else None
    result = subprocess.run(cmd, stdout=stdout, **kwargs)
    if check and result.returncode != 0:
        error(f"Command failed: {cmd}")
    return result


def pip(*args):
    """Run pip inside the venv."""
    run([str(PYTHON), "-m", "pip"] + list(args))


def ask(question, default=None, yes_all=False):
    """Prompt the user for yes/no. Returns True for yes."""
    if yes_all:
        return True
    opts = " [Y/n]" if default else " [y/N]"
    answer = input(cyan(question) + opts + " ").strip().lower()
    if answer == "":
        return default if default is not None else False
    return answer in ("y", "yes")


def ask_choice(prompt, choices, default=0):
    """
    Ask the user to pick from a numbered list.
    Returns the index of the chosen item.
    """
    print(cyan(prompt))
    for i, c in enumerate(choices):
        marker = green("  →") if i == default else "   "
        print(f"{marker} [{i+1}] {c}")
    while True:
        raw = input(cyan(f"Enter choice [1-{len(choices)}] (default {default+1}): ")).strip()
        if raw == "":
            return default
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return int(raw) - 1
        print("  Invalid choice, try again.")

# ── Base paths ────────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).resolve().parent
CLONED_DIR  = BASE_DIR / "cloned-repos"
MDM_DIR     = CLONED_DIR / "mdm"
T2M_DIR     = CLONED_DIR / "t2m_gpt"
MODELS_DIR  = BASE_DIR / "models"
VENV_DIR    = BASE_DIR / "venv"

if _IS_WIN:
    PYTHON     = VENV_DIR / "Scripts" / "python.exe"
    PIP_BIN    = VENV_DIR / "Scripts" / "pip.exe"
    ACTIVATE   = VENV_DIR / "Scripts" / "activate.bat"
else:
    PYTHON     = VENV_DIR / "bin" / "python"
    PIP_BIN    = VENV_DIR / "bin" / "pip"
    ACTIVATE   = VENV_DIR / "bin" / "activate"

# ── Step functions ────────────────────────────────────────────────────────────

def check_prerequisites(on_windows):
    section("Checking prerequisites")

    # Python version
    major, minor = sys.version_info[:2]
    min_minor = 10 if on_windows else 8
    if (major, minor) < (3, min_minor):
        error(f"Python 3.{min_minor}+ required. Found {major}.{minor}")
    info(f"Python {major}.{minor} OK")

    # git
    if shutil.which("git") is None:
        error("'git' is required but not found. Install Git and add it to PATH.")
    info(f"git found: {shutil.which('git')}")

    # wget / curl (Linux only — used as fallback for gdown)
    if not on_windows:
        for tool in ("wget", "curl"):
            if shutil.which(tool):
                info(f"{tool} found")
                break
        else:
            warn("Neither wget nor curl found — gdown fallback downloads may fail.")


def create_venv():
    section("Virtual environment")
    if VENV_DIR.exists() and PYTHON.exists():
        info("venv already exists — skipping creation")
    else:
        info(f"Creating venv at {VENV_DIR} ...")
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
        info("venv created")


def install_pytorch(on_windows, use_cuda):
    section("Installing PyTorch")
    if use_cuda:
        if on_windows:
            info("Installing PyTorch 2.4.0 + CUDA 12.1 (~2.5 GB) ...")
            pip("install", "torch==2.4.0", "torchvision==0.19.0",
                "--index-url", "https://download.pytorch.org/whl/cu121")
        else:
            info("Installing PyTorch (latest stable) + CUDA ...")
            pip("install", "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cu121")
    else:
        info("Installing PyTorch (CPU-only) ...")
        pip("install", "torch>=2.4.0", "torchvision", "torchaudio")

    # Verify
    result = run([str(PYTHON), "-c",
                  "import torch; print('torch', torch.__version__); "
                  "print('CUDA:', torch.cuda.is_available())"],
                 capture=True, check=False)
    if result.returncode == 0:
        for line in result.stdout.decode().splitlines():
            info(line)
    else:
        warn("PyTorch import check failed — continuing anyway.")


def install_dependencies():
    section("Installing Python dependencies")
    req = BASE_DIR / "requirements.txt"
    if not req.exists():
        error(f"requirements.txt not found at {req}")
    pip("install", "--upgrade", "pip", "--quiet")
    pip("install", "-r", str(req), "--quiet")
    pip("install", "git+https://github.com/openai/CLIP.git", "--quiet")
    info("All dependencies installed")


def patch_chumpy():
    """Fix numpy 1.24+ incompatibility in chumpy."""
    import site as _site
    # Find site-packages under our venv
    sp_dirs = [
        VENV_DIR / "lib64" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
        VENV_DIR / "lib"   / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
    ]
    if _IS_WIN:
        sp_dirs = [VENV_DIR / "Lib" / "site-packages"]

    chumpy_init = None
    for sp in sp_dirs:
        candidate = sp / "chumpy" / "__init__.py"
        if candidate.exists():
            chumpy_init = candidate
            break

    if chumpy_init is None:
        warn("chumpy not found — skipping patch")
        return

    txt = chumpy_init.read_text()
    old = "from numpy import bool, int, float, complex, object, str, nan, inf"
    new = ("import numpy as _np; "
           "bool=_np.bool_; int=_np.int_; float=_np.float64; "
           "complex=_np.complex128; object=_np.object_; str=_np.str_; "
           "nan=_np.nan; inf=_np.inf")
    if old in txt:
        chumpy_init.write_text(txt.replace(old, new))
        info("chumpy patched for numpy >= 1.24")
    else:
        info("chumpy already compatible — no patch needed")


def clone_repos():
    section("Cloning third-party repos → cloned-repos/")
    CLONED_DIR.mkdir(exist_ok=True)

    repos = [
        ("MDM",     "https://github.com/GuyTevet/motion-diffusion-model.git", MDM_DIR),
        ("T2M-GPT", "https://github.com/Mael-zys/T2M-GPT.git",               T2M_DIR),
    ]
    for name, url, dest in repos:
        if (dest / ".git").exists():
            info(f"{name} already cloned at {dest} — skipping")
        else:
            info(f"Cloning {name} ...")
            run(["git", "clone", "--depth", "1", url, str(dest)])
            info(f"{name} cloned")


def download_mdm_assets():
    section("Downloading MDM model assets")

    # Try to import gdown (installed as part of requirements.txt)
    try:
        import gdown
    except ImportError:
        warn("gdown not available — MDM asset downloads may fail. Install it: pip install gdown")
        return

    # ── Checkpoint ──────────────────────────────────────────────────────────
    ckpt_dir  = MDM_DIR / "save" / "humanml_enc_512_50steps"
    ckpt_file = ckpt_dir / "model000750000.pt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_file.exists():
        info("Downloading MDM 50-step checkpoint (~1.3 GB) ...")
        try:
            gdown.download(
                "https://drive.google.com/uc?id=1PE0PK8e5a5j6yYl0jmJL4NV7jkMn6UGE",
                str(ckpt_file), quiet=False
            )
        except Exception as e:
            warn(f"MDM checkpoint download failed: {e}")
            warn("Download manually and place at: " + str(ckpt_file))
    else:
        info("MDM checkpoint already present — skipping")

    # ── SMPL body model ──────────────────────────────────────────────────────
    smpl_dir  = MDM_DIR / "body_models" / "smpl"
    smpl_file = smpl_dir / "SMPL_NEUTRAL.pkl"
    smpl_dir.mkdir(parents=True, exist_ok=True)

    if not smpl_file.exists():
        info("Attempting SMPL download via gdown ...")
        try:
            gdown.download(
                "https://drive.google.com/uc?id=1INYlGA76ak_cKGzvpOV2Pe6X4oLCDg7n",
                str(smpl_file), quiet=False
            )
            info("SMPL downloaded")
        except Exception:
            warn("SMPL download failed (requires free registration).")
            print()
            print("  Register at: https://smpl.is.tue.mpg.de/")
            print("  Download 'SMPL for Python Users' and place:")
            print(f"    SMPL_NEUTRAL.pkl → {smpl_dir}/")
            print()
            warn("Tab 3 (Motion Stickman) will not work until this file is in place.")
    else:
        info("SMPL_NEUTRAL.pkl already present — skipping")

    # ── GloVe embeddings ─────────────────────────────────────────────────────
    glove_dir   = MDM_DIR / "glove"
    glove_check = glove_dir / "our_vab_data.npy"
    glove_dir.mkdir(exist_ok=True)

    if not glove_check.exists():
        info("Downloading GloVe embeddings (~10 MB) ...")
        try:
            glove_zip = glove_dir / "glove.zip"
            gdown.download(
                "https://drive.google.com/uc?id=1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n",
                str(glove_zip), quiet=False
            )
            with zipfile.ZipFile(glove_zip, "r") as zf:
                zf.extractall(glove_dir)
            glove_zip.unlink()
            info("GloVe extracted")
        except Exception as e:
            warn(f"GloVe download failed: {e}")
    else:
        info("GloVe embeddings already present — skipping")

    # ── HumanML3D mean/std normalization stats ───────────────────────────────
    dataset_dir = MDM_DIR / "dataset"
    dataset_dir.mkdir(exist_ok=True)

    if not (dataset_dir / "t2m_mean.npy").exists():
        info("Downloading HumanML3D normalization stats (~1 MB) ...")
        for fname, fid in [
            ("t2m_mean.npy", "1metMirQ-lCJBsUjEPAfzUOjKAQxWwUxN"),
            ("t2m_std.npy",  "1metMirQ-lCJBsUjEPAfzUOjKAQxWwUxN"),
        ]:
            try:
                gdown.download(
                    f"https://drive.google.com/uc?id={fid}",
                    str(dataset_dir / fname), quiet=False
                )
            except Exception as e:
                warn(f"{fname} download failed: {e}")
    else:
        info("Normalization stats already present — skipping")


def pre_download_wan():
    section("Pre-downloading Wan2.1-T2V-1.3B (~3 GB)")
    snap_dir = MODELS_DIR / "hf_cache" / "models--Wan-AI--Wan2.1-T2V-1.3B" / "snapshots"
    if snap_dir.exists():
        info("Wan2.1-T2V-1.3B already downloaded — skipping")
        return
    info("Downloading Wan2.1-T2V-1.3B (this may take several minutes) ...")
    script = (
        "from huggingface_hub import snapshot_download; "
        "import os; "
        f"os.environ['HF_HOME'] = {str(MODELS_DIR / 'hf_cache')!r}; "
        "snapshot_download('Wan-AI/Wan2.1-T2V-1.3B', "
        f"cache_dir={str(MODELS_DIR / 'hf_cache')!r}); "
        "print('Wan2.1-T2V-1.3B downloaded.')"
    )
    result = run([str(PYTHON), "-c", script], check=False)
    if result.returncode != 0:
        warn("Wan2.1-T2V-1.3B download failed — it will be downloaded on first use.")


def create_dirs():
    section("Creating runtime directories")
    dirs = [
        MODELS_DIR / "hf_cache",
        MODELS_DIR / "torch_cache",
        BASE_DIR / "outputs" / "normal-videos",
        BASE_DIR / "outputs" / "i2v-videos",
        BASE_DIR / "outputs" / "stickman-videos",
        BASE_DIR / "outputs" / "wan-videos",
        BASE_DIR / "uploads",
        BASE_DIR / "gen-logs",
        BASE_DIR / "gen-logs" / "normal-videos",
        BASE_DIR / "gen-logs" / "i2v-videos",
        BASE_DIR / "gen-logs" / "stickman-videos",
        BASE_DIR / "gen-logs" / "wan-videos",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    info("Runtime directories ready")


def verify_install():
    section("Verifying installation")
    script = (
        "import torch, flask, diffusers, transformers, scipy; "
        "print('torch:', torch.__version__); "
        "print('diffusers:', diffusers.__version__); "
        "print('transformers:', transformers.__version__); "
        "print('CUDA available:', torch.cuda.is_available()); "
        "print('All core imports OK')"
    )
    result = run([str(PYTHON), "-c", script], check=False, capture=True)
    for line in result.stdout.decode().splitlines():
        info(line)
    if result.returncode != 0:
        warn("Some imports failed — check the output above.")


def print_done(on_windows):
    section("Setup complete!")
    print()
    if on_windows:
        print("  Start the server:")
        print("    python start.py")
        print("    -or- start.bat  (CMD)")
        print("    -or- start.ps1  (PowerShell)")
        print()
        print("  Open browser at: http://localhost:7860")
    else:
        print("  Start the server:")
        print("    python start.py")
        print("    -or- ./start.sh")
        print()
        print(f"  Open browser at: http://localhost:7860")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AI Video Generation — setup script")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Non-interactive: accept all defaults")
    parser.add_argument("--skip-wan", action="store_true",
                        help="Skip pre-downloading Wan2.1 model")
    parser.add_argument("--skip-mdm", action="store_true",
                        help="Skip cloning repos and downloading MDM assets")
    args = parser.parse_args()

    yes = args.yes

    print()
    print(green("=" * 52))
    print(green("  AI Video Generation — Unified Setup"))
    print(green("=" * 52))
    print(f"  Repo: {BASE_DIR}")
    print()

    # ── 1. Detect / confirm platform ─────────────────────────────────────────
    detected_os = platform.system()   # "Windows", "Linux", "Darwin"
    if detected_os == "Windows":
        detected_label = "Windows"
    else:
        detected_label = "Linux / macOS"

    if yes:
        on_windows = _IS_WIN
        info(f"Platform auto-detected: {detected_label}")
    else:
        choice = ask_choice(
            f"Platform detected as '{detected_label}'. Which setup should be run?",
            ["Windows (PyTorch CUDA 12.1, Python 3.10+)", "Linux / macOS (PyTorch CPU or CUDA)"],
            default=0 if _IS_WIN else 1
        )
        on_windows = (choice == 0)

    # ── 2. CUDA preference ────────────────────────────────────────────────────
    if yes:
        # Default: use CUDA on Windows, skip on Linux unless GPU present
        use_cuda = on_windows
    else:
        use_cuda = ask(
            "Install PyTorch with CUDA support? (requires NVIDIA GPU + driver 525+)",
            default=on_windows,
            yes_all=yes
        )

    # ── 3. Wan2.1 pre-download ────────────────────────────────────────────────
    if not args.skip_wan and not yes:
        pre_dl_wan = ask(
            "Pre-download Wan2.1-T2V-1.3B model now? (~3 GB, can be skipped and downloaded on first use)",
            default=True
        )
    else:
        pre_dl_wan = not args.skip_wan and yes

    print()
    info(f"Platform : {'Windows' if on_windows else 'Linux / macOS'}")
    info(f"CUDA     : {'yes' if use_cuda else 'no (CPU-only)'}")
    info(f"Pre-dl Wan2.1 : {'yes' if pre_dl_wan else 'no'}")
    print()

    if not yes and not ask("Proceed with setup?", default=True):
        print("Aborted.")
        sys.exit(0)

    # ── Run steps ─────────────────────────────────────────────────────────────
    check_prerequisites(on_windows)
    create_dirs()
    create_venv()
    install_pytorch(on_windows, use_cuda)
    install_dependencies()
    patch_chumpy()

    if not args.skip_mdm:
        clone_repos()
        download_mdm_assets()

    if pre_dl_wan:
        pre_download_wan()

    verify_install()
    print_done(on_windows)


if __name__ == "__main__":
    main()
