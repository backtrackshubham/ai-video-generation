# AI Video Generation

A Flask-based web app for AI video generation with four generation modes, each on its own tab.

**Access:** `http://<host>:7860`

---

## Table of Contents

- [Tabs & Generation Modes](#tabs--generation-modes)
- [Model Performance & Recommendations](#model-performance--recommendations)
- [Setup](#setup)
- [Starting the Server](#starting-the-server)
- [Downloading Models](#downloading-models)
- [Repository Structure](#repository-structure)
- [Git Workflow](#git-workflow)
- [API Reference](#api-reference)

---

## Tabs & Generation Modes

| Tab | Colour | Model(s) | Input |
|-----|--------|----------|-------|
| 1 — Text to Video | Purple | CogVideoX-5B, CogVideoX-1.5-5B, CogVideoX-5B-I2V, CogVideoX-1.5-5B-I2V, ModelScope 1.7B | Text prompt (+ optional seed image for I2V) |
| 2 — Image to Video | Orange | Stable Video Diffusion 1.1 | Image + settings |
| 3 — Motion Stickman | Green | MDM (Motion Diffusion Model) | Text motion description |
| 4 — Wan2.1 | Blue | Wan2.1-T2V-1.3B, Wan2.1-T2V-14B | Text prompt |

---

## Model Performance & Recommendations

### By Machine

| Machine | GPU | VRAM | RAM | Best Models | Notes |
|---------|-----|------|-----|-------------|-------|
| **Alienware m15 R7** | RTX 3060 | 6 GB | 64 GB | Wan2.1-T2V-1.3B ✦, SVD, ModelScope | CogVideoX-5B will OOM unless offloaded; use 480P for Wan |
| **High-end workstation** | RTX 3090 / 4090 | 24 GB | 32+ GB | CogVideoX-5B, CogVideoX-1.5-5B, SVD, Wan2.1-T2V-14B | All models run well; 720P Wan viable |
| **Server / A100** | A100 80 GB | 80 GB | 256+ GB | All models | Fastest generation; all resolutions |
| **Linux VM (no GPU)** | VirtIO / none | — | 16+ GB | ModelScope, MDM | CPU-only; Wan/CogVideoX very slow (30–120 min/clip) |
| **Business laptop (no GPU)** | None | — | 16 GB | ModelScope, MDM | Not recommended for video models |

### By Model

| Model | Size | Min VRAM | Speed (GPU) | Speed (CPU) | Quality | Recommended For |
|-------|------|----------|-------------|-------------|---------|-----------------|
| **CogVideoX-5B** | ~21 GB | 16 GB | ~4–12 min/clip | Not practical | ⭐⭐⭐⭐ | High-end GPU workstation |
| **CogVideoX-1.5-5B** | ~20 GB | 16 GB | ~5–15 min/clip | Not practical | ⭐⭐⭐⭐⭐ | High-end GPU; longer clips (10s) |
| **CogVideoX-5B-I2V** | ~20 GB | 16 GB | ~4–12 min/clip | Not practical | ⭐⭐⭐⭐ | Image-guided generation, high-end GPU |
| **CogVideoX-1.5-5B-I2V** | ~20 GB | 16 GB | ~5–15 min/clip | Not practical | ⭐⭐⭐⭐⭐ | Best I2V quality, high-end GPU |
| **ModelScope 1.7B** | ~4 GB | 4 GB | ~1–2 min/clip | ~15–40 min/clip | ⭐⭐ | CPU fallback; has watermarks |
| **SVD 1.1** | ~8 GB | 8 GB | ~2–5 min/clip | ~60 min/clip | ⭐⭐⭐⭐ | Image-to-video on mid-range GPU |
| **Wan2.1-T2V-1.3B** | ~3 GB | 6 GB* | ~2–4 min/clip | ~30 min/clip | ⭐⭐⭐⭐⭐ | Best quality/size ratio; RTX 3060 friendly |
| **Wan2.1-T2V-14B** | ~28 GB | 16 GB | ~8–20 min/clip | Not practical | ⭐⭐⭐⭐⭐ | Highest quality; high-end GPU only |

> \* Wan2.1-T2V-1.3B uses `enable_model_cpu_offload()` on CUDA — leverages system RAM to stay within 6 GB VRAM.

### Quick Decision Guide

```
Do you have a GPU?
├── No  → Use ModelScope (Tab 1) or MDM (Tab 3)
└── Yes → How much VRAM?
    ├── 4–6 GB  → Wan2.1-T2V-1.3B @ 480P (Tab 4)  ← best option
    │             SVD (Tab 2)  ← image-to-video
    ├── 8–12 GB → All of the above + SVD comfortably
    ├── 16–24 GB→ Add CogVideoX-5B, Wan2.1-T2V-14B
    └── 40+ GB  → All models at full quality
```

---

## Setup

### Requirements

- Python 3.8+ (Linux) / 3.10+ (Windows)
- Git
- NVIDIA driver 525+ (for CUDA; CPU-only also works)
- ~35 GB free disk space for a typical model set

### First-time setup (any platform)

```bash
python setup.py
```

The script will:
1. Auto-detect your OS (Windows or Linux/macOS) — confirm or override
2. Ask whether to install PyTorch with CUDA or CPU-only
3. Ask whether to pre-download Wan2.1-T2V-1.3B (~3 GB)
4. Create a virtual environment (`venv/`)
5. Install all Python dependencies from `requirements.txt`
6. Clone MDM and T2M-GPT into `cloned-repos/`
7. Download MDM checkpoint, SMPL body model, GloVe embeddings
8. Create all output and log directories
9. Verify the installation

**Non-interactive (CI / scripted):**
```bash
python setup.py --yes              # accept all defaults
python setup.py --yes --skip-wan   # skip Wan2.1 pre-download
python setup.py --yes --skip-mdm   # skip MDM/stickman setup
```

### SMPL Body Model (Tab 3 — Motion Stickman)

SMPL requires a free registration (cannot be auto-downloaded):

1. Register at https://smpl.is.tue.mpg.de/
2. Download **SMPL for Python Users**
3. Place `SMPL_NEUTRAL.pkl` at:
   ```
   cloned-repos/mdm/body_models/smpl/SMPL_NEUTRAL.pkl
   ```

Tab 3 will not work until this file is in place. All other tabs are unaffected.

---

## Starting the Server

### Any platform
```bash
python start.py           # default port 7860
python start.py 8080      # custom port
```

### Linux/macOS (shell wrapper)
```bash
./start.sh
./start.sh 8080
```

### Windows CMD
```bat
start.bat
start.bat 8080
```

### Windows PowerShell
```powershell
.\start.ps1
.\start.ps1 -Port 8080
```

Open browser at:
- `http://localhost:7860` (local)
- `http://<hostname>:7860` (from other machines on the network)

---

## Downloading Models

Models are lazy-loaded on first use. The UI will show a download banner if a selected model is not yet on disk.

### Via the UI

1. Open any tab
2. Select a model from the dropdown
3. If not downloaded, a yellow banner appears showing the size
4. Click **Download model** — a progress bar tracks the download
5. Once complete, the banner hides and generation is enabled

### Via CLI

```bash
python download_models.py --list                   # show status of all models
python download_models.py wan-1.3b                 # download one model
python download_models.py cogvideox svd            # download multiple
python download_models.py all                      # download everything (~100+ GB)
python download_models.py                          # interactive menu
```

Available model keys:

| Key | Model | Size |
|-----|-------|------|
| `cogvideox` | CogVideoX-5B | ~21 GB |
| `cogvideox15` | CogVideoX-1.5-5B | ~20 GB |
| `cogvideox-i2v` | CogVideoX-5B-I2V | ~20 GB |
| `cogvideox15-i2v` | CogVideoX-1.5-5B-I2V | ~20 GB |
| `modelscope` | ModelScope 1.7B | ~4 GB |
| `svd` | Stable Video Diffusion 1.1 | ~8 GB |
| `wan-1.3b` | Wan2.1-T2V-1.3B | ~3 GB |
| `wan-14b` | Wan2.1-T2V-14B | ~28 GB |

All models are cached under `models/hf_cache/` inside the repo.

---

## Repository Structure

```
ai-video-generation/
├── app.py                        ← Flask backend (all 4 generation modes + download API)
├── setup.py                      ← Unified cross-platform setup script
├── start.py                      ← Unified cross-platform server launcher
├── start.sh                      ← Linux/macOS shell wrapper → delegates to start.py
├── start.bat                     ← Windows CMD wrapper → delegates to start.py
├── start.ps1                     ← Windows PowerShell wrapper → delegates to start.py
├── download_models.py            ← Standalone model downloader CLI
├── requirements.txt              ← All Python dependencies (pinned versions)
├── templates/
│   └── index.html                ← Web UI (4-tab layout)
├── static/                       ← Static assets
├── outputs/
│   ├── normal-videos/            ← CogVideoX / ModelScope outputs
│   ├── i2v-videos/               ← SVD image-to-video outputs
│   ├── stickman-videos/          ← MDM stickman outputs
│   └── wan-videos/               ← Wan2.1 outputs
├── gen-logs/                     ← Per-job and server logs
├── uploads/                      ← Uploaded images for I2V (gitignored)
├── models/
│   ├── hf_cache/                 ← HuggingFace model cache (gitignored)
│   └── torch_cache/              ← PyTorch model cache (gitignored)
├── cloned-repos/
│   ├── mdm/                      ← Motion Diffusion Model repo (gitignored)
│   └── t2m_gpt/                  ← T2M-GPT repo (gitignored)
└── venv/                         ← Python virtual environment (gitignored)
```

---

## Git Workflow

### Remote (bare repo on Linux)

The bare repository lives at:
```
hac-sverma2-1.ciena.com:~/Ciena/repositories/bare-repos/ai-video-generation.git
```

### Clone on Windows (first time)

```powershell
git clone sverma2@hac-sverma2-1.ciena.com:Ciena/repositories/bare-repos/ai-video-generation.git
cd ai-video-generation
python setup.py
```

### Push changes from Linux

```bash
git add .
git commit -m "your message"
git push
```

### Pull updates on Windows

```powershell
git pull
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/generate` | Text-to-video (CogVideoX / ModelScope) |
| `POST` | `/api/generate_cogvx_i2v` | Image-to-video via CogVideoX I2V |
| `POST` | `/api/generate_i2v` | Image-to-video via SVD |
| `POST` | `/api/generate_stickman` | Motion stickman via MDM |
| `POST` | `/api/generate_wan` | Text-to-video via Wan2.1 |
| `GET`  | `/api/status/<job_id>` | Poll generation job status |
| `GET`  | `/api/jobs` | List recent jobs |
| `GET`  | `/api/video/<filename>` | Serve output video |
| `GET`  | `/api/model_status` | Download status for all models |
| `POST` | `/api/download_model/<key>` | Start background model download |
| `GET`  | `/api/download_status/<key>` | Poll model download progress |
| `POST` | `/api/estimate` | Estimate generation time |
