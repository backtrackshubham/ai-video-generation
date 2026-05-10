"""
AI Video Generation Backend
Flask service supporting three generation modes:

1. Text-to-Video  — CogVideoX-5B  (GPU recommended, ~22 GB)
                  — ModelScope 1.7B (CPU-friendly, ~3.5 GB)
   POST /api/generate          { prompt, title, duration_seconds, fps, num_steps, guidance_scale, model }
   model: "cogvideox" | "modelscope"

2. Image-to-Video — Stable Video Diffusion 1.1
   POST /api/generate_i2v      multipart/form-data: image, title, motion_bucket_id, num_steps, fps

3. Motion Stickman — MDM (Motion Diffusion Model)
   POST /api/generate_stickman { prompt, title }
"""

import os
import sys
import uuid
import time
import threading
import json
import logging
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR             = Path(__file__).parent.resolve()
MODEL_DIR            = BASE_DIR / "models"
OUTPUT_DIR           = BASE_DIR / "outputs"
OUTPUT_VIDEO_DIR     = OUTPUT_DIR / "normal-videos"
OUTPUT_STICKMAN_DIR  = OUTPUT_DIR / "stickman-videos"
OUTPUT_I2V_DIR       = OUTPUT_DIR / "i2v-videos"
OUTPUT_STORY_DIR     = OUTPUT_DIR / "story-videos"
OUTPUT_WAN_DIR       = OUTPUT_DIR / "wan-videos"
LOG_DIR              = BASE_DIR / "gen-logs"
LOG_VIDEO_DIR        = LOG_DIR / "normal-videos"
LOG_STICKMAN_DIR     = LOG_DIR / "stickman-videos"
LOG_I2V_DIR          = LOG_DIR / "i2v-videos"
LOG_WAN_DIR          = LOG_DIR / "wan-videos"
LOG_STORY_DIR        = LOG_DIR / "story-videos"
T2M_REPO             = BASE_DIR / "cloned-repos" / "t2m_gpt"
MDM_REPO             = BASE_DIR / "cloned-repos" / "mdm"
UPLOAD_DIR           = BASE_DIR / "uploads"

for d in (OUTPUT_VIDEO_DIR, OUTPUT_STICKMAN_DIR, OUTPUT_I2V_DIR, OUTPUT_WAN_DIR,
          OUTPUT_STORY_DIR,
          LOG_VIDEO_DIR, LOG_STICKMAN_DIR, LOG_I2V_DIR, LOG_WAN_DIR, LOG_STORY_DIR,
          UPLOAD_DIR):
    os.makedirs(d, exist_ok=True)

# ── Model caches inside repo ──────────────────────────────────────────────────
HF_CACHE = MODEL_DIR / "hf_cache"
LORA_DIR = MODEL_DIR / "loras"
os.environ.setdefault("HF_HOME",           str(HF_CACHE))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE))
os.environ.setdefault("DIFFUSERS_CACHE",    str(HF_CACHE))
os.environ.setdefault("TORCH_HOME",         str(MODEL_DIR / "torch_cache"))


def _hf_cache_name(hf_repo: str) -> str:
    return "models--" + hf_repo.replace("/", "--")


def _pretrained_source(hf_repo: str):
    """Return (path_or_id, extra_kwargs) for from_pretrained.

    If the model was downloaded with download_models.py (local_dir layout),
    return the local path directly — no symlinks, no HF network call.
    Otherwise fall back to repo-id + cache_dir so HF downloads on demand.
    """
    local = HF_CACHE / _hf_cache_name(hf_repo)
    if local.exists() and any(
        f for f in local.rglob("*")
        if f.is_file() and ".cache" not in str(f)
    ):
        return str(local), {}
    return hf_repo, {"cache_dir": str(HF_CACHE)}

# ── Device ────────────────────────────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "server.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)
log.info(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    log.info(f"GPU: {torch.cuda.get_device_name(0)}  VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")


# ── Helpers ───────────────────────────────────────────────────────────────────
def slugify(title: str) -> str:
    import re
    slug = title.strip().lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-") or "untitled"


def make_job_logger(slug: str, subdir: Path) -> logging.Logger:
    logger = logging.getLogger(f"job.{slug}")
    if not logger.handlers:
        fh = logging.FileHandler(subdir / f"{slug}.log")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)
        logger.propagate = True
    return logger


# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# ── Job persistence ───────────────────────────────────────────────────────────
JOBS_FILE = BASE_DIR / "jobs_history.json"
_jobs_lock = threading.Lock()


class JobStore:
    """Thread-safe in-memory job store that persists to jobs_history.json.

    Each value is a plain dict.  Callers use store[job_id] = {...} to create
    a job, then store[job_id].update(...) to mutate it — .update() is patched
    onto the dict so it also triggers a save.
    """

    def __init__(self):
        self._data: dict = {}
        self._load()

    # ── persistence ──────────────────────────────────────────────────────────
    def _load(self):
        if JOBS_FILE.exists():
            try:
                with open(JOBS_FILE, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                # Patch update() onto every loaded dict
                for job_id, job in raw.items():
                    self._patch(job)
                self._data = raw
                log.info(f"Loaded {len(raw)} job(s) from {JOBS_FILE}")
            except Exception as e:
                log.warning(f"Could not load {JOBS_FILE}: {e} — starting fresh")

    def _save(self):
        try:
            with _jobs_lock:
                tmp = JOBS_FILE.with_suffix(".tmp")
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(self._data, f, indent=2, default=str)
                tmp.replace(JOBS_FILE)
        except Exception as e:
            log.warning(f"Could not save jobs_history.json: {e}")

    def _patch(self, job: dict):
        """Replace job.update() with a version that also persists."""
        store_ref = self
        original_update = dict.update

        def _update(self_dict, *args, **kwargs):
            original_update(self_dict, *args, **kwargs)
            store_ref._save()

        job.update = lambda *a, **kw: _update(job, *a, **kw)

    # ── dict-like interface ──────────────────────────────────────────────────
    def __setitem__(self, key, value):
        self._patch(value)
        self._data[key] = value
        self._save()

    def __getitem__(self, key):
        return self._data[key]

    def __contains__(self, key):
        return key in self._data

    def get(self, key, default=None):
        return self._data.get(key, default)

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()


jobs = JobStore()

# ── Time estimation ───────────────────────────────────────────────────────────
# CogVideoX-5B:   ~3-4 s/step on A100 (80GB), ~8-12 s/step on RTX 3090 (24GB)
# ModelScope 1.7B: ~2 s/frame on GPU, ~40 s/frame on CPU @ 20 steps
SECS_PER_STEP_COG         = 8  if DEVICE == "cuda" else 120   # per diffusion step
SECS_PER_FRAME_MODELSCOPE = 2  if DEVICE == "cuda" else 40    # per frame @ 20 steps


def estimate_time(num_frames: int, num_steps: int, model: str = "cogvideox") -> dict:
    if model == "cogvideox":
        # CogVideoX denoises across all frames simultaneously (not per-frame)
        total_seconds = num_steps * SECS_PER_STEP_COG
    elif model == "modelscope":
        # ModelScope generates frame by frame, scaled by actual step count
        total_seconds = num_frames * num_steps * (SECS_PER_FRAME_MODELSCOPE / 20)
    else:
        total_seconds = num_frames * num_steps * (2 / 20) if DEVICE == "cuda" else num_frames * num_steps * 2

    low  = int(total_seconds * 0.8)
    high = int(total_seconds * 1.3)

    def fmt(s):
        if s < 60:   return f"{s}s"
        m, sec = divmod(s, 60)
        return f"{m}m {sec}s" if sec else f"{m}m"

    return {
        "low_seconds":  low,
        "high_seconds": high,
        "display":      f"{fmt(low)} – {fmt(high)}",
        "frames":       num_frames,
        "steps":        num_steps,
        "device":       DEVICE,
        "model":        model,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# TEXT-TO-VIDEO — CogVideoX-5B
# ═══════════════════════════════════════════════════════════════════════════════
_cogvx_pipeline = None
_cogvx_lock     = threading.Lock()

COG_MODEL_ID      = "THUDM/CogVideoX-5b"
COG15_MODEL_ID    = "THUDM/CogVideoX1.5-5B"
COG_I2V_MODEL_ID  = "THUDM/CogVideoX-5b-I2V"
COG15_I2V_MODEL_ID= "THUDM/CogVideoX1.5-5B-I2V"

# Max native frames per model (8k+1 constraint for CogVideoX)
COG_MAX_FRAMES = {
    "cogvideox":      49,   # ~6s @8fps
    "cogvideox-i2v":  49,
    "cogvideox15":    81,   # ~10s @8fps
    "cogvideox15-i2v":81,
}

def _load_cogvx_t2v(model_id: str, label: str):
    from diffusers import CogVideoXPipeline
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    log.info(f"Loading {label} ({DEVICE.upper()})…")
    _src, _kw = _pretrained_source(model_id)
    pipe = CogVideoXPipeline.from_pretrained(
        _src, torch_dtype=dtype, **_kw,
    )
    pipe = pipe.to(DEVICE)
    if DEVICE == "cuda":
        pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    log.info(f"{label} loaded.")
    return pipe

def _load_cogvx_i2v(model_id: str, label: str):
    from diffusers import CogVideoXImageToVideoPipeline
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    log.info(f"Loading {label} ({DEVICE.upper()})…")
    _src, _kw = _pretrained_source(model_id)
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        _src, torch_dtype=dtype, **_kw,
    )
    pipe = pipe.to(DEVICE)
    if DEVICE == "cuda":
        pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    log.info(f"{label} loaded.")
    return pipe

# Pipeline cache per model key
_cog_pipelines: dict = {}
_cog_locks: dict = {k: threading.Lock() for k in COG_MAX_FRAMES}

def get_cog_pipeline(model: str):
    """Return cached pipeline for the given model key, loading on first call."""
    if model in _cog_pipelines:
        return _cog_pipelines[model]
    with _cog_locks[model]:
        if model in _cog_pipelines:
            return _cog_pipelines[model]
        try:
            if model == "cogvideox":
                pipe = _load_cogvx_t2v(COG_MODEL_ID, "CogVideoX-5B")
            elif model == "cogvideox15":
                pipe = _load_cogvx_t2v(COG15_MODEL_ID, "CogVideoX-1.5-5B")
            elif model == "cogvideox-i2v":
                pipe = _load_cogvx_i2v(COG_I2V_MODEL_ID, "CogVideoX-5B-I2V")
            elif model == "cogvideox15-i2v":
                pipe = _load_cogvx_i2v(COG15_I2V_MODEL_ID, "CogVideoX-1.5-5B-I2V")
            else:
                raise ValueError(f"Unknown model key: {model}")
            _cog_pipelines[model] = pipe
        except Exception as e:
            log.error(f"Failed to load pipeline for {model}: {e}")
            raise
    return _cog_pipelines[model]


# ── Shared helpers ────────────────────────────────────────────────────────────

def _frames_to_numpy(frames):
    """Convert PIL or tensor frame list to list of uint8 numpy arrays."""
    import numpy as np
    result = []
    for f in frames:
        if hasattr(f, "numpy"):
            result.append(f.numpy())
        elif hasattr(f, "tobytes"):
            result.append(np.array(f))
        else:
            result.append(f)
    return result


def stitch_clips(clips: list, fps: int, crossfade_frames: int = 8) -> list:
    """
    Concatenate a list of frame-lists into one, with a linear crossfade
    between adjacent clips to smooth transitions.
    clips: list of lists of numpy uint8 arrays (H, W, 3)
    Returns: single flat list of numpy frames.
    """
    import numpy as np
    if len(clips) == 1:
        return clips[0]
    out = list(clips[0])
    for next_clip in clips[1:]:
        cf = min(crossfade_frames, len(out), len(next_clip))
        # blend tail of current output with head of next clip
        tail   = out[-cf:]
        head   = next_clip[:cf]
        for i in range(cf):
            alpha = (i + 1) / (cf + 1)
            blended = (tail[i].astype(np.float32) * (1 - alpha) +
                       head[i].astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
            out[-cf + i] = blended
        out.extend(next_clip[cf:])
    return out


def _make_step_callback(job_id, num_steps, start_time, clip_idx, total_clips, jlog,
                         base_pct=5, range_pct=85, preview_dir=None):
    """
    Returns a step callback that maps steps to a progress slice for one clip
    within a multi-clip job.  If preview_dir is given, saves a decoded mid-frame
    PNG at each diffusion step.
    """
    step_ref = [0]
    clip_range = range_pct / total_clips
    clip_base  = base_pct + clip_idx * clip_range

    def step_callback(pipe_obj, step, timestep, callback_kwargs):
        """callback_on_step_end signature used by CogVideoX pipelines."""
        step_ref[0] = step + 1
        pct = int(clip_base + (step_ref[0] / num_steps) * clip_range)
        elapsed   = time.time() - start_time
        per_step  = elapsed / (clip_idx * num_steps + step_ref[0])
        total_steps = total_clips * num_steps
        done_steps  = clip_idx * num_steps + step_ref[0]
        remaining = per_step * (total_steps - done_steps)
        clip_label = f"clip {clip_idx+1}/{total_clips} " if total_clips > 1 else ""
        jobs[job_id].update(
            progress=pct,
            elapsed_seconds=int(elapsed),
            remaining_seconds=int(remaining),
            message=f"{clip_label}step {step_ref[0]}/{num_steps} — {int(elapsed)}s elapsed, ~{int(remaining)}s remaining",
        )
        jlog.info(f"[{job_id}] {clip_label}step {step_ref[0]}/{num_steps} elapsed={int(elapsed)}s")

        # Save decoded mid-frame preview if requested
        if preview_dir is not None:
            try:
                latents = callback_kwargs.get("latents")
                if latents is not None:
                    with torch.no_grad():
                        frames = pipe_obj.decode_latents(latents)   # (B, C, T, H, W)
                    # pick middle temporal frame, first batch item
                    mid_t = frames.shape[2] // 2
                    frame = frames[0, :, mid_t, :, :]               # (C, H, W)
                    frame = (frame.float().cpu().clamp(-1, 1) + 1) / 2  # → [0,1]
                    frame = (frame.permute(1, 2, 0).numpy() * 255).astype("uint8")
                    from PIL import Image as PILImg
                    fname = f"clip{clip_idx+1:02d}_step_{step_ref[0]:03d}.png"
                    PILImg.fromarray(frame).save(str(preview_dir / fname))
            except Exception as e:
                jlog.warning(f"[{job_id}] Step preview save failed at step {step_ref[0]}: {e}")

        return callback_kwargs  # required by diffusers
    return step_callback


def run_generation(job_id: str, prompt: str, num_frames: int, num_steps: int,
                   guidance_scale: float, slug: str, model: str = "cogvideox",
                   num_clips: int = 1, image_path: str = None, prompts: list = None):
    jlog = make_job_logger(slug, LOG_VIDEO_DIR)
    is_i2v = model.endswith("-i2v")
    jlog.info(f"[{job_id}] Starting {model} | prompt='{prompt}' frames={num_frames} "
              f"steps={num_steps} cfg={guidance_scale} clips={num_clips} i2v={is_i2v}")

    # Build per-clip prompt list
    if not prompts or not isinstance(prompts, list):
        prompts = [prompt] * num_clips
    else:
        while len(prompts) < num_clips:
            prompts.append(prompts[-1])

    try:
        import imageio
        import numpy as np
        from PIL import Image as PILImage
    except Exception as e:
        jobs[job_id].update(status="failed", error=str(e), message=f"Import failed: {e}")
        return

    model_labels = {
        "cogvideox": "CogVideoX-5B", "cogvideox15": "CogVideoX-1.5-5B",
        "cogvideox-i2v": "CogVideoX-5B-I2V", "cogvideox15-i2v": "CogVideoX-1.5-5B-I2V",
    }
    label = model_labels.get(model, model)
    jobs[job_id].update(status="loading_model", message=f"Loading {label}…", progress=2)

    try:
        pipe = get_cog_pipeline(model)
    except Exception as e:
        jobs[job_id].update(status="failed", error=str(e), message=f"Model load failed: {e}")
        jlog.error(f"[{job_id}] Model load failed: {e}")
        return

    # Prepare seed image for I2V models
    seed_image = None
    if is_i2v:
        if not image_path:
            jobs[job_id].update(status="failed", error="image_path required for I2V",
                                message="No seed image provided")
            return
        seed_image = PILImage.open(image_path).convert("RGB")
        seed_image = seed_image.resize((720, 480))

    start_time = time.time()
    all_clips = []

    # Create per-job preview directory for step frames
    preview_dir = OUTPUT_DIR / "normal-videos" / f"{slug}-steps"
    preview_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"[{job_id}] Step previews → {preview_dir}")

    try:
        for clip_idx in range(num_clips):
            clip_prompt = prompts[clip_idx]
            clip_label = f"Clip {clip_idx+1}/{num_clips} — " if num_clips > 1 else ""
            jobs[job_id].update(
                status="generating",
                message=f"{clip_label}generating frames…",
                progress=5 + int((clip_idx / num_clips) * 85),
            )

            cb = _make_step_callback(job_id, num_steps, start_time, clip_idx, num_clips, jlog,
                                     preview_dir=preview_dir)

            with torch.no_grad():
                if is_i2v:
                    output = pipe(
                        image=seed_image,
                        prompt=clip_prompt,
                        num_frames=num_frames,
                        num_inference_steps=num_steps,
                        guidance_scale=guidance_scale,
                        callback_on_step_end=cb,
                    )
                else:
                    output = pipe(
                        prompt=clip_prompt,
                        num_frames=num_frames,
                        num_inference_steps=num_steps,
                        guidance_scale=guidance_scale,
                        callback_on_step_end=cb,
                    )

            clip_frames = _frames_to_numpy(output.frames[0])
            all_clips.append(clip_frames)
            jlog.info(f"[{job_id}] Clip {clip_idx+1} done — {len(clip_frames)} frames")

        # Stitch clips with crossfade
        if num_clips > 1:
            jobs[job_id].update(status="stitching", message="Stitching clips…", progress=92)
            final_frames = stitch_clips(all_clips, fps=8, crossfade_frames=8)
        else:
            final_frames = all_clips[0]

        video_filename = f"{slug}.mp4"
        out_path = OUTPUT_VIDEO_DIR / video_filename
        imageio.mimwrite(str(out_path), final_frames, fps=8, codec="libx264", quality=8)

        elapsed_total = int(time.time() - start_time)
        jobs[job_id].update(
            status="done", progress=100,
            video_file=video_filename,
            message=f"Done in {elapsed_total}s — {len(final_frames)} frames ({len(final_frames)//8}s)",
            elapsed_seconds=elapsed_total,
        )
        jlog.info(f"[{job_id}] Complete in {elapsed_total}s: {out_path} ({len(final_frames)} frames)")

    except Exception as e:
        jlog.error(f"[{job_id}] Generation error: {e}", exc_info=True)
        jobs[job_id].update(status="failed", error=str(e), message=f"Generation failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT-TO-VIDEO — ModelScope 1.7B  (CPU-friendly fallback)
# ═══════════════════════════════════════════════════════════════════════════════
_modelscope_pipeline = None
_modelscope_lock     = threading.Lock()

MODELSCOPE_MODEL_ID = "damo-vilab/text-to-video-ms-1.7b"


def get_modelscope_pipeline():
    global _modelscope_pipeline
    if _modelscope_pipeline is not None:
        return _modelscope_pipeline
    with _modelscope_lock:
        if _modelscope_pipeline is not None:
            return _modelscope_pipeline
        log.info(f"Loading ModelScope 1.7B pipeline ({DEVICE.upper()})…")
        try:
            from diffusers import DiffusionPipeline
            from diffusers.utils import export_to_video

            dtype = torch.float16 if DEVICE == "cuda" else torch.float32
            _src, _kw = _pretrained_source(MODELSCOPE_MODEL_ID)
            pipe = DiffusionPipeline.from_pretrained(
                _src,
                torch_dtype=dtype,
                trust_remote_code=True,
                **_kw,
            )
            pipe = pipe.to(DEVICE)
            _modelscope_pipeline = pipe
            log.info("ModelScope 1.7B loaded successfully.")
        except Exception as e:
            log.error(f"Failed to load ModelScope pipeline: {e}")
            raise
    return _modelscope_pipeline


def run_modelscope_generation(job_id: str, prompt: str, num_frames: int,
                               num_steps: int, guidance_scale: float, slug: str):
    jlog = make_job_logger(slug, LOG_VIDEO_DIR)
    jlog.info(f"[{job_id}] Starting ModelScope T2V | prompt='{prompt}' frames={num_frames} steps={num_steps}")

    try:
        import imageio
        import numpy as np
    except Exception as e:
        jobs[job_id].update(status="failed", error=str(e), message=f"Import failed: {e}")
        return

    jobs[job_id].update(status="loading_model", message="Loading ModelScope 1.7B…", progress=2)

    try:
        pipe = get_modelscope_pipeline()
    except Exception as e:
        jobs[job_id].update(status="failed", error=str(e), message=f"Model load failed: {e}")
        jlog.error(f"[{job_id}] Model load failed: {e}")
        return

    jobs[job_id].update(status="generating", message="Generating frames…", progress=5)
    start_time = time.time()
    step_ref = [0]

    def step_callback(step, timestep, latents):
        step_ref[0] = step + 1
        pct = 5 + int((step_ref[0] / num_steps) * 90)
        elapsed   = time.time() - start_time
        per_step  = elapsed / step_ref[0] if step_ref[0] else 0
        remaining = per_step * (num_steps - step_ref[0])
        jobs[job_id].update(
            progress=pct,
            elapsed_seconds=int(elapsed),
            remaining_seconds=int(remaining),
            message=f"Step {step_ref[0]}/{num_steps} — elapsed {int(elapsed)}s, ~{int(remaining)}s remaining",
        )
        jlog.info(f"[{job_id}] Step {step_ref[0]}/{num_steps} elapsed={int(elapsed)}s")

    try:
        with torch.no_grad():
            output = pipe(
                prompt,
                num_frames=num_frames,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                callback=step_callback,
                callback_steps=1,
            )

        frames = output.frames[0]
        frames_np = [np.array(f) if hasattr(f, "tobytes") else f for f in frames]

        video_filename = f"{slug}.mp4"
        out_path = OUTPUT_VIDEO_DIR / video_filename
        imageio.mimwrite(str(out_path), frames_np, fps=8, codec="libx264", quality=8)

        elapsed_total = int(time.time() - start_time)
        jobs[job_id].update(
            status="done", progress=100,
            video_file=video_filename,
            message=f"Done in {elapsed_total}s — {len(frames_np)} frames",
            elapsed_seconds=elapsed_total,
        )
        jlog.info(f"[{job_id}] ModelScope complete in {elapsed_total}s: {out_path}")

    except Exception as e:
        jlog.error(f"[{job_id}] ModelScope generation error: {e}", exc_info=True)
        jobs[job_id].update(status="failed", error=str(e), message=f"Generation failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE-TO-VIDEO — Stable Video Diffusion 1.1
# ═══════════════════════════════════════════════════════════════════════════════
_svd_pipeline = None
_svd_lock     = threading.Lock()

SVD_MODEL_ID = "stabilityai/stable-video-diffusion-img2vid-xt"


def get_svd_pipeline():
    global _svd_pipeline
    if _svd_pipeline is not None:
        return _svd_pipeline
    with _svd_lock:
        if _svd_pipeline is not None:
            return _svd_pipeline
        log.info(f"Loading Stable Video Diffusion 1.1 ({DEVICE.upper()})…")
        try:
            from diffusers import StableVideoDiffusionPipeline

            dtype = torch.float16 if DEVICE == "cuda" else torch.float32
            _src, _kw = _pretrained_source(SVD_MODEL_ID)
            pipe = StableVideoDiffusionPipeline.from_pretrained(
                _src,
                torch_dtype=dtype,
                **_kw,
            )
            pipe = pipe.to(DEVICE)
            if DEVICE == "cuda":
                pipe.enable_model_cpu_offload()
            _svd_pipeline = pipe
            log.info("SVD 1.1 loaded successfully.")
        except Exception as e:
            log.error(f"Failed to load SVD pipeline: {e}")
            raise
    return _svd_pipeline


def run_i2v_generation(job_id: str, image_path: str, slug: str,
                       motion_bucket_id: int, num_steps: int, fps: int):
    jlog = make_job_logger(slug, LOG_I2V_DIR)
    jlog.info(f"[{job_id}] Starting SVD I2V | image={image_path} motion={motion_bucket_id} steps={num_steps}")

    try:
        import imageio
        import numpy as np
        from PIL import Image as PILImg
    except Exception as e:
        jobs[job_id].update(status="failed", error=str(e), message=f"Import failed: {e}")
        return

    jobs[job_id].update(status="loading_model", message="Loading Stable Video Diffusion 1.1…", progress=2)

    try:
        pipe = get_svd_pipeline()
    except Exception as e:
        jobs[job_id].update(status="failed", error=str(e), message=f"Model load failed: {e}")
        jlog.error(f"[{job_id}] Model load failed: {e}")
        return

    jobs[job_id].update(status="generating", message="Animating image…", progress=5)
    start_time = time.time()

    try:
        # Load and resize image to SVD native resolution
        image = Image.open(image_path).convert("RGB")
        image = image.resize((1024, 576))

        # SVD doesn't support step callbacks — pulse progress while running
        def _progress_pulse():
            step = 0
            while not _svd_done[0]:
                time.sleep(2)
                step = min(step + 1, num_steps - 1)
                elapsed = time.time() - start_time
                per_step = elapsed / step if step else 0
                remaining = per_step * (num_steps - step)
                pct = 5 + int((step / num_steps) * 85)
                jobs[job_id].update(
                    progress=pct,
                    elapsed_seconds=int(elapsed),
                    remaining_seconds=int(remaining),
                    message=f"Generating… elapsed {int(elapsed)}s",
                )

        _svd_done = [False]
        pulse = threading.Thread(target=_progress_pulse, daemon=True)
        pulse.start()

        with torch.no_grad():
            output = pipe(
                image,
                num_inference_steps=num_steps,
                motion_bucket_id=motion_bucket_id,
                fps=fps,
                decode_chunk_size=8,
            )

        _svd_done[0] = True

        frames = output.frames[0]
        frames_np = [np.array(f) for f in frames]

        # Save all output frames for inspection
        preview_dir = OUTPUT_I2V_DIR / f"{slug}-frames"
        preview_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames_np):
            PILImg.fromarray(frame).save(str(preview_dir / f"frame{i:04d}.jpg"))
        jlog.info(f"[{job_id}] Saved {len(frames_np)} frames → {preview_dir}")

        video_filename = f"{slug}.mp4"
        out_path = OUTPUT_I2V_DIR / video_filename
        imageio.mimwrite(str(out_path), frames_np, fps=fps, codec="libx264", quality=8)

        elapsed_total = int(time.time() - start_time)
        jobs[job_id].update(
            status="done", progress=100,
            video_file=video_filename,
            message=f"Done in {elapsed_total}s — {len(frames_np)} frames",
            elapsed_seconds=elapsed_total,
        )
        jlog.info(f"[{job_id}] I2V complete in {elapsed_total}s: {out_path}")

    except Exception as e:
        jlog.error(f"[{job_id}] I2V error: {e}", exc_info=True)
        jobs[job_id].update(status="failed", error=str(e), message=f"Generation failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# MOTION STICKMAN — MDM  (Motion Diffusion Model)
# ═══════════════════════════════════════════════════════════════════════════════
_mdm_models = None
_mdm_lock   = threading.Lock()


def get_mdm_models():
    """Lazy-load MDM model."""
    global _mdm_models
    if _mdm_models is not None:
        return _mdm_models

    with _mdm_lock:
        if _mdm_models is not None:
            return _mdm_models

        if not MDM_REPO.exists():
            raise RuntimeError(
                f"MDM not found at {MDM_REPO}. "
                "Run setup to clone https://github.com/GuyTevet/motion-diffusion-model into mdm/"
            )

        if str(MDM_REPO) not in sys.path:
            sys.path.insert(0, str(MDM_REPO))

        log.info(f"Loading MDM models ({DEVICE.upper()})…")

        # MDM uses its own argument parser; we inject required args via sys.argv
        _mdm_model_path = str(MDM_REPO / "save" / "humanml_enc_512_50steps" / "model000750000.pt")
        old_argv = sys.argv[:]
        sys.argv = ["app", "--model_path", _mdm_model_path]

        try:
            from utils.parser_util import generate_args
            from utils.model_util import load_model_wo_clip, create_model_and_diffusion
            from model.cfg_sampler import ClassifierFreeSampleModel
            import data_loaders.humanml.utils.paramUtil as paramUtil
            from data_loaders.humanml.utils.plot_script import plot_3d_motion
            args = generate_args()
        finally:
            sys.argv = old_argv

        args.model_path = _mdm_model_path
        args.device = 0 if DEVICE == "cuda" else "cpu"

        from diffusion import logger
        from utils.fixseed import fixseed
        fixseed(args.seed)

        # Load model — must chdir to MDM_REPO so relative paths (body_models/smpl) resolve
        from utils.model_util import create_model_and_diffusion
        _orig_cwd = os.getcwd()
        os.chdir(str(MDM_REPO))
        try:
            # MDM's create_model_and_diffusion only uses data.dataset.num_actions (optional)
            class _FakeData:
                class dataset: pass
            model, diffusion = create_model_and_diffusion(args, _FakeData())
        finally:
            os.chdir(_orig_cwd)
        log.info(f"Loading MDM checkpoint: {args.model_path}")
        state_dict = torch.load(args.model_path, map_location="cpu")
        load_model_wo_clip(model, state_dict)
        model = ClassifierFreeSampleModel(model)
        model.to(DEVICE)
        model.eval()

        _mdm_models = (model, diffusion, args, paramUtil, plot_3d_motion)
        log.info("MDM loaded successfully.")

    return _mdm_models


def run_stickman_generation(job_id: str, prompt: str, slug: str,
                             motion_length: float = 6.0,
                             guidance_scale: float = 2.5,
                             seed: int = 10):
    jlog = make_job_logger(slug, LOG_STICKMAN_DIR)
    jlog.info(f"[{job_id}] Starting MDM stickman | prompt='{prompt}'")

    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
    except Exception as e:
        jobs[job_id].update(status="failed", error=str(e), message=f"Import failed: {e}")
        return

    jobs[job_id].update(status="loading_model", message="Loading MDM models…", progress=5)
    gen_start = time.time()

    try:
        model, diffusion, args, paramUtil, plot_3d_motion = get_mdm_models()
    except Exception as e:
        jobs[job_id].update(status="failed", error=str(e), message=f"Model load failed: {e}",
                            elapsed_seconds=int(time.time() - gen_start))
        jlog.error(f"[{job_id}] MDM load failed: {e}", exc_info=True)
        return

    jobs[job_id].update(status="generating", message="Generating motion…", progress=20,
                        elapsed_seconds=int(time.time() - gen_start))

    try:
        if str(MDM_REPO) not in sys.path:
            sys.path.insert(0, str(MDM_REPO))

        import numpy as np
        from utils.fixseed import fixseed
        from data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain
        from data_loaders.humanml.scripts.motion_process import recover_from_ric

        fixseed(seed)
        n_frames = min(196, int(motion_length * 20))  # MDM runs at 20fps, max 196 frames

        # Pre-encode text once (MDM optimization — avoids re-encoding every diffusion step)
        texts = [prompt]
        with torch.no_grad():
            text_embed = model.encode_text(texts)

        model_kwargs = {
            "y": {
                "mask":       torch.ones(1, 1, 1, n_frames, device=DEVICE),
                "lengths":    torch.tensor([n_frames], device=DEVICE),
                "text":       texts,
                "tokens":     [""],
                "scale":      torch.ones(1, device=DEVICE) * guidance_scale,
                "text_embed": text_embed,
            }
        }

        # Un-normalize helpers (needed for step previews)
        mean = np.load(str(MDM_REPO / "dataset" / "t2m_mean.npy"))   # (263,)
        std  = np.load(str(MDM_REPO / "dataset" / "t2m_std.npy"))    # (263,)

        num_diff_steps = args.diffusion_steps   # 50
        preview_dir    = OUTPUT_STICKMAN_DIR / f"{slug}-steps"
        preview_dir.mkdir(parents=True, exist_ok=True)

        def _sample_to_joints(s_tensor):
            """(1,263,1,T) → (T,22,3) joint positions."""
            s = s_tensor.cpu().permute(0, 2, 3, 1).squeeze(1).numpy()
            s = s * std + mean
            return recover_from_ric(torch.tensor(s).float(), 22).squeeze(0).numpy()

        def _save_step_preview(step_idx, joints_np, out_dir):
            import matplotlib.pyplot as _plt
            mid = joints_np.shape[0] // 2
            j   = joints_np.copy() * 1.3
            j[:, :, 1] -= j[:, :, 1].min()
            fig2, ax2 = _plt.subplots(1, 1, figsize=(3, 3),
                                      subplot_kw={'projection': '3d'})
            colors = ["#DD5A37","#D69E00","#B75A39","#FF6D00","#DDB50E"]
            ax2.set_xlim3d([-1.5, 1.5]); ax2.set_ylim3d([0, 3]); ax2.set_zlim3d([-1, 2])
            ax2.view_init(elev=120, azim=-90); ax2.dist = 7.5
            ax2.set_axis_off()
            fig2.suptitle(f"step {step_idx+1}/{num_diff_steps}", fontsize=8)
            for ci, (chain, color) in enumerate(zip(t2m_kinematic_chain, colors)):
                lw = 4.0 if ci < 5 else 2.0
                ax2.plot3D(j[mid, chain, 0], j[mid, chain, 1], j[mid, chain, 2],
                           linewidth=lw, color=color)
            _plt.tight_layout()
            fig2.savefig(str(out_dir / f"step_{step_idx+1:03d}.png"), dpi=80)
            _plt.close(fig2)

        # Sample — collect all intermediate steps via dump_steps
        jlog.info(f"[{job_id}] Running diffusion ({num_diff_steps} steps), saving previews → {preview_dir}")
        with torch.no_grad():
            dump = diffusion.p_sample_loop(
                model,
                (1, model.njoints, model.nfeats, n_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,
                init_image=None,
                progress=True,
                dump_steps=list(range(num_diff_steps)),
                noise=None,
                const_noise=False,
            )  # list of (1,263,1,n_frames) tensors

        # Save per-step skeleton previews (mid-sequence frame)
        for step_idx, s_tensor in enumerate(dump):
            try:
                _save_step_preview(step_idx, _sample_to_joints(s_tensor), preview_dir)
            except Exception as pe:
                jlog.warning(f"[{job_id}] Step preview {step_idx} failed: {pe}")
            pct = 70 + int(((step_idx + 1) / num_diff_steps) * 20)
            jobs[job_id].update(progress=pct,
                                message=f"Saving step previews… {step_idx+1}/{num_diff_steps}",
                                elapsed_seconds=int(time.time() - gen_start))

        jlog.info(f"[{job_id}] Saved {len(dump)} step previews → {preview_dir}")
        sample = dump[-1]  # final denoised sample

        jobs[job_id].update(progress=90, message="Rendering video…",
                            elapsed_seconds=int(time.time() - gen_start))

        # sample: (1, 263, 1, n_frames) → (1, n_frames, 263)
        sample_np = sample.cpu().permute(0, 2, 3, 1).squeeze(1).numpy()
        sample_np = sample_np * std + mean  # un-normalize

        # recover_from_ric: (B, T, 263) → (B, T, 22, 3)
        joints_tensor = recover_from_ric(torch.tensor(sample_np).float(), 22)
        joints = joints_tensor.squeeze(0).numpy()  # (n_frames, 22, 3)

        T = joints.shape[0]
        jlog.info(f"[{job_id}] Generated {T} motion frames")

        # Render
        video_filename = f"{slug}.mp4"
        out_path = str(OUTPUT_STICKMAN_DIR / video_filename)

        import matplotlib.pyplot as plt
        ani = plot_3d_motion(
            out_path,
            t2m_kinematic_chain,
            joints,
            title=prompt,
            dataset="humanml",
            fps=20,
            radius=3,
        )
        # plot_3d_motion closes the figure before VideoClip renders (lazy) — reopen is
        # not possible, so we patch: set duration then write before plt cleans up.
        # plot_3d_motion already called plt.close() — we need to recreate the clip
        # by calling write_videofile while fig is still valid.
        # Since plt.close() already ran inside plot_3d_motion, we must re-render
        # using our own matplotlib figure instead.
        from moviepy.editor import VideoClip
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain as kinematic_chain

        fig = plt.figure(figsize=(4, 4))
        ax  = fig.add_subplot(111, projection='3d')
        colors = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]

        j = joints.copy()
        j *= 1.3  # humanml scale
        j[:, :, 1] -= j[:, :, 1].min()  # floor

        def make_frame(t):
            idx = min(T - 1, int(t * 20))
            ax.clear()
            ax.set_xlim3d([-1.5, 1.5]); ax.set_ylim3d([0, 3]); ax.set_zlim3d([-1, 2])
            ax.view_init(elev=120, azim=-90); ax.dist = 7.5
            ax.set_axis_off()
            fig.suptitle(prompt[:40], fontsize=9)
            for i, (chain, color) in enumerate(zip(kinematic_chain, colors)):
                lw = 4.0 if i < 5 else 2.0
                ax.plot3D(j[idx, chain, 0], j[idx, chain, 1], j[idx, chain, 2],
                          linewidth=lw, color=color)
            from moviepy.video.io.bindings import mplfig_to_npimage
            return mplfig_to_npimage(fig)

        clip = VideoClip(make_frame, duration=T / 20.0)
        clip.write_videofile(out_path, fps=20, logger=None)
        plt.close(fig)

        elapsed = int(time.time() - gen_start)
        jobs[job_id].update(
            status="done", progress=100,
            video_file=video_filename,
            elapsed_seconds=elapsed,
            remaining_seconds=0,
            message=f"Done — {T} frames · {elapsed}s",
        )
        jlog.info(f"[{job_id}] Stickman complete: {out_path} ({elapsed}s)")

    except Exception as e:
        jlog.error(f"[{job_id}] Stickman error: {e}", exc_info=True)
        jobs[job_id].update(status="failed", error=str(e), message=f"Generation failed: {e}",
                            elapsed_seconds=int(time.time() - gen_start))


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT-TO-VIDEO — Wan2.1 (T2V-1.3B, consumer GPU friendly)
# ═══════════════════════════════════════════════════════════════════════════════
_wan_pipeline = None
_wan_lock     = threading.Lock()

WAN_MODEL_ID_13B = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
WAN_MODEL_ID_14B = "Wan-AI/Wan2.1-T2V-14B-Diffusers"

# Seconds-per-step estimate for Wan2.1 (RTX 3060 6GB with offload)
SECS_PER_STEP_WAN = 6 if DEVICE == "cuda" else 180


def get_wan_pipeline(model_id: str = WAN_MODEL_ID_13B):
    global _wan_pipeline
    if _wan_pipeline is not None:
        return _wan_pipeline
    with _wan_lock:
        if _wan_pipeline is not None:
            return _wan_pipeline
        log.info(f"Loading Wan2.1 pipeline: {model_id} ({DEVICE.upper()})…")
        try:
            from diffusers import WanPipeline

            dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32

            _src, _kw = _pretrained_source(model_id)
            pipe = WanPipeline.from_pretrained(
                _src,
                torch_dtype=dtype,
                **_kw,
            )
            if DEVICE == "cuda":
                if model_id == WAN_MODEL_ID_14B:
                    # 14B needs aggressive layer-by-layer offload; enable_model_cpu_offload
                    # is not sufficient for 6 GB cards — sequential offload is slower but safer
                    log.info("Wan2.1 14B: using sequential CPU offload (16+ GB VRAM recommended)")
                    pipe.enable_sequential_cpu_offload()
                else:
                    pipe.enable_model_cpu_offload()
            else:
                pipe = pipe.to(DEVICE)

            _wan_pipeline = pipe
            log.info("Wan2.1 pipeline loaded successfully.")
        except Exception as e:
            log.error(f"Failed to load Wan2.1 pipeline: {e}")
            raise
    return _wan_pipeline


def run_wan_generation(job_id: str, prompt: str, num_frames: int, num_steps: int,
                       guidance_scale: float, slug: str, resolution: str = "480p",
                       num_clips: int = 1, model_id: str = WAN_MODEL_ID_13B,
                       prompts: list = None):
    jlog = make_job_logger(slug, LOG_WAN_DIR)
    jlog.info(f"[{job_id}] Starting Wan2.1 | prompt='{prompt}' frames={num_frames} "
              f"steps={num_steps} cfg={guidance_scale} clips={num_clips} res={resolution}")

    # Build per-clip prompt list
    if not prompts or not isinstance(prompts, list):
        prompts = [prompt] * num_clips
    else:
        while len(prompts) < num_clips:
            prompts.append(prompts[-1])

    try:
        import imageio
        import numpy as np
    except Exception as e:
        jobs[job_id].update(status="failed", error=str(e), message=f"Import failed: {e}")
        return

    jobs[job_id].update(status="loading_model", message="Loading Wan2.1…", progress=2)

    try:
        pipe = get_wan_pipeline(model_id)
    except Exception as e:
        jobs[job_id].update(status="failed", error=str(e), message=f"Model load failed: {e}")
        jlog.error(f"[{job_id}] Wan2.1 model load failed: {e}")
        return

    # Resolution mapping
    res_map = {"480p": (832, 480), "720p": (1280, 720)}
    width, height = res_map.get(resolution, (832, 480))

    start_time = time.time()
    all_clips = []

    # Per-job preview directory for step frames
    preview_dir = OUTPUT_WAN_DIR / f"{slug}-steps"
    preview_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"[{job_id}] Step previews → {preview_dir}")

    try:
        for clip_idx in range(num_clips):
            clip_prompt = prompts[clip_idx]
            clip_label = f"Clip {clip_idx+1}/{num_clips} — " if num_clips > 1 else ""
            jobs[job_id].update(
                status="generating",
                message=f"{clip_label}generating frames…",
                progress=5 + int((clip_idx / num_clips) * 85),
            )

            step_ref = [0]
            clip_range = 85 / num_clips
            clip_base  = 5 + clip_idx * clip_range

            def step_callback(pipe_obj, step, timestep, callback_kwargs):
                step_ref[0] = step + 1
                pct = int(clip_base + (step_ref[0] / num_steps) * clip_range)
                elapsed   = time.time() - start_time
                per_step  = elapsed / (clip_idx * num_steps + step_ref[0])
                done_steps  = clip_idx * num_steps + step_ref[0]
                remaining = per_step * (num_clips * num_steps - done_steps)
                jobs[job_id].update(
                    progress=pct,
                    elapsed_seconds=int(elapsed),
                    remaining_seconds=int(remaining),
                    message=f"{clip_label}step {step_ref[0]}/{num_steps} — "
                            f"{int(elapsed)}s elapsed, ~{int(remaining)}s remaining",
                )
                jlog.info(f"[{job_id}] {clip_label}step {step_ref[0]}/{num_steps} elapsed={int(elapsed)}s")
                # Save mid-frame preview from latents if available
                try:
                    latents = callback_kwargs.get("latents")
                    if latents is not None and hasattr(pipe_obj, "vae"):
                        with torch.no_grad():
                            decoded = pipe_obj.vae.decode(
                                latents[:, :, latents.shape[2] // 2:latents.shape[2] // 2 + 1] / pipe_obj.vae.config.scaling_factor
                            ).sample  # (B, C, 1, H, W)
                        frame = decoded[0, :, 0].float().cpu()
                        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
                        frame = (frame.permute(1, 2, 0).numpy() * 255).astype("uint8")
                        fname = f"clip{clip_idx+1:02d}_step{step_ref[0]:04d}.jpg"
                        PILImg.fromarray(frame).save(str(preview_dir / fname))
                except Exception:
                    pass  # preview is best-effort, never block generation
                return callback_kwargs

            with torch.no_grad():
                output = pipe(
                    prompt=clip_prompt,
                    num_frames=num_frames,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    callback_on_step_end=step_callback,
                )

            clip_frames = _frames_to_numpy(output.frames[0])
            all_clips.append(clip_frames)
            jlog.info(f"[{job_id}] Clip {clip_idx+1} done — {len(clip_frames)} frames")

        # Stitch clips with crossfade
        if num_clips > 1:
            jobs[job_id].update(status="stitching", message="Stitching clips…", progress=92)
            final_frames = stitch_clips(all_clips, fps=16, crossfade_frames=8)
        else:
            final_frames = all_clips[0]

        video_filename = f"{slug}.mp4"
        out_path = OUTPUT_WAN_DIR / video_filename
        fps_out = 16  # Wan2.1 native fps
        imageio.mimwrite(str(out_path), final_frames, fps=fps_out, codec="libx264", quality=8)

        elapsed_total = int(time.time() - start_time)
        jobs[job_id].update(
            status="done", progress=100,
            video_file=video_filename,
            message=f"Done in {elapsed_total}s — {len(final_frames)} frames ({len(final_frames)//fps_out}s)",
            elapsed_seconds=elapsed_total,
        )
        jlog.info(f"[{job_id}] Wan2.1 complete in {elapsed_total}s: {out_path} ({len(final_frames)} frames)")

    except Exception as e:
        jlog.error(f"[{job_id}] Wan2.1 generation error: {e}", exc_info=True)
        jobs[job_id].update(status="failed", error=str(e), message=f"Generation failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


@app.route("/api/estimate", methods=["POST"])
def api_estimate():
    data      = request.get_json(force=True)
    duration  = float(data.get("duration_seconds", 5))
    fps       = int(data.get("fps", 8))
    num_steps = int(data.get("num_steps", 50))
    model     = data.get("model", "cogvideox").lower().strip()
    if model not in ("cogvideox", "modelscope"):
        model = "cogvideox"

    if model == "cogvideox":
        raw = int(duration * fps)
        num_frames = min(max(raw, 8), 49)
        num_frames = ((num_frames - 1) // 8) * 8 + 1
    else:
        num_frames = min(max(int(duration * fps), 8), 24)

    est = estimate_time(num_frames, num_steps, model)
    est["num_frames"] = num_frames
    return jsonify(est)


@app.route("/api/generate", methods=["POST"])
def api_generate():
    data           = request.get_json(force=True)
    prompt         = data.get("prompt", "").strip()
    prompts        = data.get("prompts", [])
    title          = data.get("title", "").strip()
    duration       = float(data.get("duration_seconds", 5))
    fps            = int(data.get("fps", 8))
    num_steps      = int(data.get("num_steps", 50))
    guidance_scale = float(data.get("guidance_scale", 6.0))
    model          = data.get("model", "cogvideox").lower().strip()
    num_clips      = max(1, int(data.get("num_clips", 1)))

    # Normalise prompts list — fill missing clips with the first prompt
    if not isinstance(prompts, list) or not prompts:
        prompts = [prompt] * num_clips
    else:
        prompts = [p.strip() for p in prompts if p.strip()]
        while len(prompts) < num_clips:
            prompts.append(prompts[-1])

    valid_models = ("cogvideox", "cogvideox15", "modelscope")
    if model not in valid_models:
        model = "cogvideox"

    if not prompts[0]:
        return jsonify({"error": "prompt is required"}), 400

    if model in ("cogvideox", "cogvideox15"):
        max_frames = COG_MAX_FRAMES[model]
        raw_frames = int(duration * fps)
        num_frames = min(max(raw_frames, 9), max_frames)
        num_frames = ((num_frames - 1) // 8) * 8 + 1
    else:
        num_clips  = 1  # ModelScope doesn't support stitching
        num_frames = min(max(int(duration * fps), 8), 24)

    job_id = str(uuid.uuid4())
    slug   = slugify(title) if title else f"video-{job_id[:8]}"
    est    = estimate_time(num_frames * num_clips, num_steps, model)

    jobs[job_id] = {
        "job_id":            job_id,
        "type":              "t2v",
        "model":             model,
        "status":            "queued",
        "progress":          0,
        "message":           "Queued…",
        "prompt":            prompt,
        "title":             title or slug,
        "slug":              slug,
        "num_frames":        num_frames,
        "num_clips":         num_clips,
        "num_steps":         num_steps,
        "video_file":        None,
        "error":             None,
        "elapsed_seconds":   0,
        "remaining_seconds": est["high_seconds"],
        "estimate":          est,
        "created_at":        time.time(),
        "intermediate_files": [],
    }

    if model in ("cogvideox", "cogvideox15"):
        t = threading.Thread(
            target=run_generation,
            args=(job_id, prompts[0], num_frames, num_steps, guidance_scale, slug, model, num_clips),
            kwargs={"prompts": prompts},
            daemon=True,
        )
    else:
        t = threading.Thread(
            target=run_modelscope_generation,
            args=(job_id, prompts[0], num_frames, num_steps, guidance_scale, slug),
            daemon=True,
        )
    t.start()

    log.info(f"[{job_id}] T2V: model={model} clips={num_clips} frames={num_frames} steps={num_steps}")
    return jsonify({"job_id": job_id, "estimate": est}), 202


@app.route("/api/generate_cogvx_i2v", methods=["POST"])
def api_generate_cogvx_i2v():
    if "image" not in request.files:
        return jsonify({"error": "image file is required"}), 400

    file           = request.files["image"]
    prompt         = request.form.get("prompt", "").strip()
    title          = request.form.get("title", "").strip()
    duration       = float(request.form.get("duration_seconds", 5))
    fps            = int(request.form.get("fps", 8))
    num_steps      = int(request.form.get("num_steps", 50))
    guidance_scale = float(request.form.get("guidance_scale", 6.0))
    model          = request.form.get("model", "cogvideox-i2v").lower().strip()
    num_clips      = max(1, int(request.form.get("num_clips", 1)))

    if model not in ("cogvideox-i2v", "cogvideox15-i2v"):
        model = "cogvideox-i2v"

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    max_frames = COG_MAX_FRAMES[model]
    raw_frames = int(duration * fps)
    num_frames = min(max(raw_frames, 9), max_frames)
    num_frames = ((num_frames - 1) // 8) * 8 + 1

    job_id = str(uuid.uuid4())
    slug   = slugify(title) if title else f"cogvx-i2v-{job_id[:8]}"

    ext        = Path(file.filename).suffix or ".jpg"
    image_path = str(UPLOAD_DIR / f"{job_id}{ext}")
    file.save(image_path)

    est = estimate_time(num_frames * num_clips, num_steps, "cogvideox")

    jobs[job_id] = {
        "job_id":            job_id,
        "type":              "cogvx-i2v",
        "model":             model,
        "status":            "queued",
        "progress":          0,
        "message":           "Queued…",
        "prompt":            prompt,
        "title":             title or slug,
        "slug":              slug,
        "num_frames":        num_frames,
        "num_clips":         num_clips,
        "num_steps":         num_steps,
        "video_file":        None,
        "error":             None,
        "elapsed_seconds":   0,
        "remaining_seconds": est["high_seconds"],
        "estimate":          est,
        "created_at":        time.time(),
        "intermediate_files": [],
    }

    t = threading.Thread(
        target=run_generation,
        args=(job_id, prompt, num_frames, num_steps, guidance_scale, slug, model, num_clips, image_path),
        daemon=True,
    )
    t.start()

    log.info(f"[{job_id}] CogVideoX-I2V: model={model} clips={num_clips} frames={num_frames} steps={num_steps}")
    return jsonify({"job_id": job_id, "estimate": est}), 202


@app.route("/api/generate_i2v", methods=["POST"])
def api_generate_i2v():
    if "image" not in request.files:
        return jsonify({"error": "image file is required"}), 400

    file             = request.files["image"]
    title            = request.form.get("title", "").strip()
    motion_bucket_id = int(request.form.get("motion_bucket_id", 127))
    num_steps        = int(request.form.get("num_steps", 25))
    fps              = int(request.form.get("fps", 8))

    job_id = str(uuid.uuid4())
    slug   = slugify(title) if title else f"i2v-{job_id[:8]}"

    # Save uploaded image
    ext        = Path(file.filename).suffix or ".jpg"
    image_path = str(UPLOAD_DIR / f"{job_id}{ext}")
    file.save(image_path)

    jobs[job_id] = {
        "job_id":            job_id,
        "type":              "i2v",
        "status":            "queued",
        "progress":          0,
        "message":           "Queued…",
        "prompt":            title or slug,
        "title":             title or slug,
        "slug":              slug,
        "video_file":        None,
        "error":             None,
        "elapsed_seconds":   0,
        "remaining_seconds": num_steps * (8 if DEVICE == "cuda" else 120),
        "created_at":        time.time(),
        "intermediate_files": [],
    }

    t = threading.Thread(
        target=run_i2v_generation,
        args=(job_id, image_path, slug, motion_bucket_id, num_steps, fps),
        daemon=True,
    )
    t.start()

    log.info(f"[{job_id}] I2V started: slug='{slug}' motion={motion_bucket_id} steps={num_steps}")
    return jsonify({"job_id": job_id}), 202


@app.route("/api/generate_stickman", methods=["POST"])
def api_generate_stickman():
    data   = request.get_json(force=True)
    prompt = data.get("prompt", "").strip()
    title  = data.get("title", "").strip()
    motion_length   = float(data.get("motion_length", 6.0))
    guidance_scale  = float(data.get("guidance_scale", 2.5))
    seed            = int(data.get("seed", 10))

    motion_length = max(1.0, min(9.8, motion_length))  # clamp to MDM max (196 frames @ 20fps)

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    job_id = str(uuid.uuid4())
    slug   = slugify(title) if title else f"stickman-{job_id[:8]}"
    jobs[job_id] = {
        "job_id":            job_id,
        "type":              "stickman",
        "status":            "queued",
        "progress":          0,
        "message":           "Queued…",
        "prompt":            prompt,
        "title":             title or slug,
        "slug":              slug,
        "video_file":        None,
        "error":             None,
        "elapsed_seconds":   0,
        "remaining_seconds": 60,
        "created_at":        time.time(),
        "intermediate_files": [],
    }

    t = threading.Thread(
        target=run_stickman_generation,
        args=(job_id, prompt, slug, motion_length, guidance_scale, seed),
        daemon=True,
    )
    t.start()

    log.info(f"[{job_id}] Stickman started: slug='{slug}' prompt='{prompt}'")
    return jsonify({"job_id": job_id}), 202


@app.route("/api/generate_wan", methods=["POST"])
def api_generate_wan():
    data           = request.get_json(force=True)
    prompt         = data.get("prompt", "").strip()
    prompts        = data.get("prompts", [])
    title          = data.get("title", "").strip()
    num_steps      = int(data.get("num_steps", 30))
    guidance_scale = float(data.get("guidance_scale", 5.0))
    resolution     = data.get("resolution", "480p").lower().strip()
    num_clips      = max(1, int(data.get("num_clips", 1)))
    model_variant  = data.get("model_variant", "1.3b").lower().strip()

    # Normalise prompts list
    if not isinstance(prompts, list) or not prompts:
        prompts = [prompt] * num_clips
    else:
        prompts = [p.strip() for p in prompts if p.strip()]
        while len(prompts) < num_clips:
            prompts.append(prompts[-1])

    if resolution not in ("480p", "720p"):
        resolution = "480p"

    model_id = WAN_MODEL_ID_14B if model_variant == "14b" else WAN_MODEL_ID_13B

    if not prompts[0]:
        return jsonify({"error": "prompt is required"}), 400

    # Wan2.1 generates 81 frames at 16fps (~5s) per clip
    num_frames = 81

    job_id = str(uuid.uuid4())
    slug   = slugify(title) if title else f"wan-{job_id[:8]}"

    # Time estimate
    total_steps = num_steps * num_clips
    est_low  = int(total_steps * SECS_PER_STEP_WAN * 0.8)
    est_high = int(total_steps * SECS_PER_STEP_WAN * 1.3)

    def _fmt(s):
        if s < 60: return f"{s}s"
        m, sec = divmod(s, 60)
        return f"{m}m {sec}s" if sec else f"{m}m"

    est = {
        "low_seconds": est_low, "high_seconds": est_high,
        "display": f"{_fmt(est_low)} – {_fmt(est_high)}",
        "frames": num_frames * num_clips, "steps": num_steps,
        "device": DEVICE, "model": f"wan2.1-{model_variant}",
    }

    jobs[job_id] = {
        "job_id":            job_id,
        "type":              "wan",
        "model":             f"wan2.1-{model_variant}",
        "status":            "queued",
        "progress":          0,
        "message":           "Queued…",
        "prompt":            prompt,
        "title":             title or slug,
        "slug":              slug,
        "num_frames":        num_frames,
        "num_clips":         num_clips,
        "num_steps":         num_steps,
        "resolution":        resolution,
        "video_file":        None,
        "error":             None,
        "elapsed_seconds":   0,
        "remaining_seconds": est_high,
        "estimate":          est,
        "created_at":        time.time(),
        "intermediate_files": [],
    }

    t = threading.Thread(
        target=run_wan_generation,
        args=(job_id, prompts[0], num_frames, num_steps, guidance_scale, slug, resolution, num_clips, model_id),
        kwargs={"prompts": prompts},
        daemon=True,
    )
    t.start()

    log.info(f"[{job_id}] Wan2.1: variant={model_variant} res={resolution} clips={num_clips} steps={num_steps}")
    return jsonify({"job_id": job_id, "estimate": est}), 202



# ═══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY — download status & on-demand download
# ═══════════════════════════════════════════════════════════════════════════════

# Registry: model_key → { hf_repo, size_gb, label, cache_subdir_pattern }
MODEL_REGISTRY = {
    "cogvideox": {
        "hf_repo":  COG_MODEL_ID,
        "size_gb":  22,
        "label":    "CogVideoX-5B",
        "tab":      "t2v",
    },
    "cogvideox15": {
        "hf_repo":  COG15_MODEL_ID,
        "size_gb":  31,
        "label":    "CogVideoX-1.5-5B",
        "tab":      "t2v",
    },
    "cogvideox-i2v": {
        "hf_repo":  COG_I2V_MODEL_ID,
        "size_gb":  22,
        "label":    "CogVideoX-5B-I2V",
        "tab":      "t2v",
    },
    "cogvideox15-i2v": {
        "hf_repo":  COG15_I2V_MODEL_ID,
        "size_gb":  31,
        "label":    "CogVideoX-1.5-5B-I2V",
        "tab":      "t2v",
    },
    "modelscope": {
        "hf_repo":  MODELSCOPE_MODEL_ID,
        "size_gb":  33,
        "label":    "ModelScope 1.7B",
        "tab":      "t2v",
    },
    "svd": {
        "hf_repo":         SVD_MODEL_ID,
        "size_gb":         30,
        "label":           "Stable Video Diffusion 1.1",
        "tab":             "i2v",
        "ignore_patterns": ["*.fp16.safetensors"],
    },
    "wan-1.3b": {
        "hf_repo":  WAN_MODEL_ID_13B,
        "size_gb":  45,
        "label":    "Wan2.1-T2V-1.3B",
        "tab":      "wan",
    },
    "wan-14b": {
        "hf_repo":  WAN_MODEL_ID_14B,
        "size_gb":  80,
        "label":    "Wan2.1-T2V-14B",
        "tab":      "wan",
    },
}


def _is_model_downloaded(hf_repo: str) -> bool:
    """Return True if the model is available locally (flat local_dir or HF cache snapshot)."""
    local = HF_CACHE / _hf_cache_name(hf_repo)
    if local.exists() and any(f for f in local.rglob("*") if f.is_file() and ".cache" not in str(f)):
        return True
    # Fallback: old-style HF cache with snapshots/ (symlink-based, Linux)
    snap_dir = local / "snapshots"
    if snap_dir.exists():
        for snap in snap_dir.iterdir():
            if snap.is_dir() and any(snap.iterdir()):
                return True
    return False


# Active download jobs keyed by model_key
_download_jobs: dict = {}
_download_lock = threading.Lock()


@app.route("/api/gpu_info", methods=["GET"])
def api_gpu_info():
    """Return GPU VRAM info so the frontend can warn about low-VRAM models."""
    if DEVICE != "cuda" or not torch.cuda.is_available():
        return jsonify({"available": False, "vram_gb": 0, "name": "CPU"})
    props = torch.cuda.get_device_properties(0)
    vram_gb = round(props.total_memory / (1024 ** 3), 1)
    return jsonify({"available": True, "vram_gb": vram_gb, "name": props.name})


@app.route("/api/model_status", methods=["GET"])
def api_model_status():
    """Return download status for all known models."""
    result = {}
    for key, info in MODEL_REGISTRY.items():
        result[key] = {
            "label":       info["label"],
            "size_gb":     info["size_gb"],
            "tab":         info["tab"],
            "downloaded":  _is_model_downloaded(info["hf_repo"]),
            "downloading": _download_jobs.get(key, {}).get("active", False),
            "progress":    _download_jobs.get(key, {}).get("progress", 0),
        }
    return jsonify(result)


@app.route("/api/download_model/<model_key>", methods=["POST"])
def api_download_model(model_key):
    """
    Start downloading a model in the background.
    Returns a job token; poll /api/download_status/<model_key> for progress.
    """
    if model_key not in MODEL_REGISTRY:
        return jsonify({"error": f"Unknown model key: {model_key}"}), 400

    info = MODEL_REGISTRY[model_key]

    if _is_model_downloaded(info["hf_repo"]):
        return jsonify({"status": "already_downloaded"}), 200

    with _download_lock:
        if _download_jobs.get(model_key, {}).get("active"):
            return jsonify({"status": "already_downloading"}), 200

        _download_jobs[model_key] = {
            "active":    True,
            "progress":  0,
            "message":   "Starting download…",
            "error":     None,
            "done":      False,
        }

    def _do_download():
        job = _download_jobs[model_key]
        try:
            from huggingface_hub import snapshot_download
            log.info(f"Downloading model {model_key} ({info['hf_repo']}) …")

            # huggingface_hub doesn't expose per-file progress easily,
            # so we use a background thread to approximate progress by
            # watching the cache directory size vs expected size.
            hf_repo = info["hf_repo"]
            expected_bytes = info["size_gb"] * 1024 ** 3

            stop_event = threading.Event()

            def _progress_watcher():
                # HF stores files under models--org--name/blobs/ during download.
                # The directory may not exist yet when the watcher starts, so we
                # must handle that gracefully and keep retrying.
                model_cache_dir = HF_CACHE / _hf_cache_name(hf_repo)
                while not stop_event.is_set():
                    try:
                        if model_cache_dir.exists():
                            total = sum(
                                f.stat().st_size
                                for f in model_cache_dir.rglob("*")
                                if f.is_file() and not f.name.endswith(".lock")
                                and not f.name.endswith(".incomplete")
                            )
                            pct = min(int(total / expected_bytes * 100), 99)
                            job["progress"] = pct
                            gb_done = total / 1024 ** 3
                            job["message"] = f"Downloading… {gb_done:.1f} / {info['size_gb']} GB ({pct}%)"
                        else:
                            job["message"] = "Connecting to HuggingFace…"
                    except Exception:
                        pass
                    stop_event.wait(2)

            watcher = threading.Thread(target=_progress_watcher, daemon=True)
            watcher.start()

            local_dir = HF_CACHE / _hf_cache_name(hf_repo)
            local_dir.mkdir(parents=True, exist_ok=True)
            snapshot_download(hf_repo, local_dir=str(local_dir),
                              ignore_patterns=info.get("ignore_patterns"))

            stop_event.set()
            job["progress"] = 100
            job["message"]  = "Download complete"
            job["done"]     = True
            job["active"]   = False
            log.info(f"Model {model_key} downloaded successfully.")
        except Exception as e:
            job["error"]   = str(e)
            job["message"] = f"Download failed: {e}"
            job["active"]  = False
            log.error(f"Model {model_key} download failed: {e}")

    t = threading.Thread(target=_do_download, daemon=True)
    t.start()

    return jsonify({"status": "started", "model_key": model_key}), 202


@app.route("/api/download_status/<model_key>", methods=["GET"])
def api_download_status(model_key):
    """Poll download progress for a model."""
    if model_key not in MODEL_REGISTRY:
        return jsonify({"error": f"Unknown model key: {model_key}"}), 400
    info = MODEL_REGISTRY[model_key]
    job  = _download_jobs.get(model_key, {})
    return jsonify({
        "model_key":   model_key,
        "label":       info["label"],
        "size_gb":     info["size_gb"],
        "downloaded":  _is_model_downloaded(info["hf_repo"]),
        "downloading": job.get("active", False),
        "progress":    job.get("progress", 0),
        "message":     job.get("message", ""),
        "error":       job.get("error"),
        "done":        job.get("done", False),
    })


@app.route("/api/status/<job_id>", methods=["GET"])
def api_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    return jsonify(job)


@app.route("/api/video/<filename>", methods=["GET"])
def api_video(filename):
    for d in (OUTPUT_VIDEO_DIR, OUTPUT_I2V_DIR, OUTPUT_STICKMAN_DIR, OUTPUT_WAN_DIR):
        if (d / filename).exists():
            return send_from_directory(str(d), filename)
    return jsonify({"error": "file not found"}), 404


@app.route("/api/jobs", methods=["GET"])
def api_jobs():
    recent = sorted(jobs.values(), key=lambda j: j["created_at"], reverse=True)[:20]
    return jsonify(recent)


@app.route("/api/jobs/history", methods=["GET"])
def api_jobs_history():
    """Return all persisted jobs, newest first, for the Generated Media tab."""
    all_jobs = sorted(jobs.values(), key=lambda j: j["created_at"], reverse=True)
    return jsonify(all_jobs)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Story Video Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

# ── Constants ─────────────────────────────────────────────────────────────────
STORY_LLM_MODELS = {
    "qwen":      "Qwen/Qwen2.5-7B-Instruct",
    "qwen-gguf": "Qwen/Qwen2.5-7B-Instruct-GGUF",   # loaded via llama-cpp-python
    "phi3":      "microsoft/Phi-3.5-mini-instruct",
}

# GGUF filename to download from the repo
STORY_LLM_GGUF_FILE = {
    "qwen-gguf": "qwen2.5-7b-instruct-q4_k_m.gguf",
}

SD15_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"

# Style → (LoRA repo, LoRA filename, trigger words prepended to prompt)
STYLE_LORAS = {
    "realistic":  (None, None, "photorealistic, highly detailed, 8k"),
    "ghibli":     ("artificialguybr/studioghibli-redmond-1-5v-studio-ghibli-lora-for-liberteredmond-sd-1-5",
                   "StudioGhibliRedmond-15V-LiberteRedmond-StdGBRedmAF-StudioGhibli.safetensors",
                   "StdGBRedmAF, Studio Ghibli style"),
    "anime":      ("artificialguybr/cutecartoon-redmond-1-5v-cute-cartoon-lora-for-liberteredmond-sd-1-5",
                   "CuteCartoon15V-LiberteRedmodModel-Cartoon-CuteCartoonAF.safetensors",
                   "CuteCartoonAF, anime style"),
    "cartoon":    ("artificialguybr/cutecartoon-redmond-1-5v-cute-cartoon-lora-for-liberteredmond-sd-1-5",
                   "CuteCartoon15V-LiberteRedmodModel-Cartoon-CuteCartoonAF.safetensors",
                   "CuteCartoonAF, Cartoon style"),
    "futuristic": ("artificialguybr/3d-redmond-1-5v-3d-render-style-for-liberte-redmond-sd-1-5",
                   "3DRedmond15V-LiberteRedmond-3DRenderStyle-3DRenderAF.safetensors",
                   "3DRenderAF, futuristic, sleek, neon, cinematic"),
    "scifi":      ("artificialguybr/3d-redmond-1-5v-3d-render-style-for-liberte-redmond-sd-1-5",
                   "3DRedmond15V-LiberteRedmond-3DRenderStyle-3DRenderAF.safetensors",
                   "3DRenderAF, sci-fi, space, technology, cinematic lighting"),
}

# IndicF5 reference clips bundled in HF repo
INDIC_VOICE_REFS = {
    "default_female": ("MAR_F_WIKI_00001.wav", "A female narrator speaking in a calm, clear voice."),
    "default_male":   ("MAR_M_WIKI_00001.wav", "A male narrator speaking in a calm, clear voice."),
}
INDIC_MODEL_ID = "ai4bharat/IndicF5"
INDIC_REF_DIR  = MODEL_DIR / "hf_cache" / "indic_refs"
MMS_HINDI_MODEL_ID = "facebook/mms-tts-hin"

# TTS voices that use IndicF5 (gated); all others use MMS-TTS
INDIC_VOICES = set(INDIC_VOICE_REFS.keys())

# Pipeline caches
_sd_pipe_cache    = {}   # key: style
_llm_cache        = {}   # key: llm name
_indic_tts_cache  = {}   # key: "pipe"


# ── LLM: scene breakdown ──────────────────────────────────────────────────────

def _get_llm_pipe(llm_key: str):
    """Load LLM pipeline, cached. GGUF variant uses llama-cpp-python."""
    if llm_key in _llm_cache:
        return _llm_cache[llm_key]

    if llm_key in STORY_LLM_GGUF_FILE:
        # ── GGUF path via llama-cpp-python ────────────────────────
        from llama_cpp import Llama
        from huggingface_hub import hf_hub_download

        repo_id  = STORY_LLM_MODELS[llm_key]
        filename = STORY_LLM_GGUF_FILE[llm_key]
        log.info(f"Loading GGUF {repo_id}/{filename}…")

        # Check local download first (download_models.py saves to hf_cache/<cache-name>/<filename>)
        local_gguf = HF_CACHE / _hf_cache_name(repo_id) / filename
        if local_gguf.exists():
            gguf_path = str(local_gguf)
            log.info(f"Using local GGUF: {gguf_path}")
        else:
            from huggingface_hub import hf_hub_download
            log.info("GGUF not found locally — downloading from HuggingFace…")
            gguf_path = hf_hub_download(repo_id=repo_id, filename=filename)
        n_gpu = -1 if DEVICE == "cuda" else 0   # -1 = all layers on GPU
        llm = Llama(
            model_path=gguf_path,
            n_ctx=2048,
            n_gpu_layers=n_gpu,
            verbose=False,
        )
        _llm_cache[llm_key] = ("gguf", llm)
        log.info(f"GGUF model loaded: {filename}")
        return _llm_cache[llm_key]

    # ── Transformers path ─────────────────────────────────────────
    from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForCausalLM

    model_id = STORY_LLM_MODELS.get(llm_key, STORY_LLM_MODELS["qwen"])
    log.info(f"Loading LLM {model_id} on {DEVICE}…")
    _src, _kw = _pretrained_source(model_id)

    tokenizer = AutoTokenizer.from_pretrained(_src, **_kw)
    model = AutoModelForCausalLM.from_pretrained(
        _src,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        low_cpu_mem_usage=True,
        **_kw,
    )
    if DEVICE == "cpu":
        model = model.to("cpu")

    pipe = hf_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        do_sample=False,
        temperature=None,
        top_p=None,
    )
    _llm_cache[llm_key] = ("transformers", pipe)
    log.info(f"LLM {model_id} loaded.")
    return _llm_cache[llm_key]


def llm_break_into_scenes(script: str, num_scenes: int, llm_key: str) -> list:
    """Call LLM to split script into scenes. Returns list of dicts."""
    import re as _re
    backend, pipe = _get_llm_pipe(llm_key)

    system_prompt = (
        "You are a creative director. Given a story script, break it into exactly "
        f"{num_scenes} scenes. For each scene output a JSON object with two keys: "
        '"narration" (the Hindi narration text the voiceover actor will read — write this in Hindi Devanagari script) '
        'and "image_prompt" (a detailed English prompt for an AI image generator describing the visual). '
        f'Output a JSON array of exactly {num_scenes} objects and nothing else. No markdown, no explanation.'
    )

    if backend == "gguf":
        # llama-cpp-python chat completion
        response = pipe.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"Script:\n{script}"},
            ],
            max_tokens=1024,
            temperature=0.0,
        )
        result = response["choices"][0]["message"]["content"]
    else:
        # transformers pipeline
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Script:\n{script}"},
        ]
        tokenizer = pipe.tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt_text = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{script} [/INST]"

        raw = pipe(prompt_text)[0]["generated_text"]
        result = raw[len(prompt_text):] if raw.startswith(prompt_text) else raw

    match = _re.search(r'\[.*\]', result, _re.DOTALL)
    if not match:
        raise ValueError(f"LLM did not return a JSON array. Raw:\n{result[:500]}")

    scenes = json.loads(match.group(0))
    if not isinstance(scenes, list):
        raise ValueError("LLM output is not a JSON array.")

    out = []
    for s in scenes[:num_scenes]:
        out.append({
            "narration":    str(s.get("narration", "")),
            "image_prompt": str(s.get("image_prompt", "")),
        })
    while len(out) < num_scenes:
        out.append({"narration": "", "image_prompt": ""})

    return out


# ── Image generation: SD 1.5 + LoRA ──────────────────────────────────────────

def _get_sd_pipe(style: str):
    """Load SD 1.5 pipeline with style LoRA, cached per style."""
    if style in _sd_pipe_cache:
        return _sd_pipe_cache[style]

    from diffusers import StableDiffusionPipeline

    lora_repo, lora_file, _ = STYLE_LORAS.get(style, STYLE_LORAS["realistic"])
    log.info(f"Loading SD 1.5 for style={style} on {DEVICE}…")
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    _src, _kw = _pretrained_source(SD15_MODEL_ID)
    pipe = StableDiffusionPipeline.from_pretrained(
        _src,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        **_kw,
    )

    if lora_repo and lora_file:
        lora_local = MODEL_DIR / "loras" / lora_file
        if lora_local.exists():
            log.info(f"Loading LoRA from local path {lora_local}…")
            pipe.load_lora_weights(str(lora_local.parent), weight_name=lora_file)
        else:
            log.info(f"Loading LoRA from HuggingFace {lora_repo}/{lora_file}…")
            pipe.load_lora_weights(lora_repo, weight_name=lora_file)

    if DEVICE == "cuda":
        pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()

    _sd_pipe_cache[style] = pipe
    log.info(f"SD 1.5 ({style}) ready.")
    return pipe


def generate_scene_image(image_prompt: str, style: str, out_path: Path,
                         job_id: str = None, scene_idx: int = 0,
                         num_scenes: int = 1) -> Path:
    """Generate a still image for one scene with live step logging."""
    from PIL import Image as PILImg  # noqa — ensure available

    lora_repo, lora_file, trigger = STYLE_LORAS.get(style, STYLE_LORAS["realistic"])
    full_prompt = f"{trigger}, {image_prompt}" if trigger else image_prompt
    negative    = "blurry, low quality, watermark, signature, text, ugly, deformed"
    num_steps   = 25 if DEVICE == "cuda" else 8

    log.info(f"  [scene {scene_idx+1}/{num_scenes}] Generating image: '{image_prompt[:60]}…'")

    pipe = _get_sd_pipe(style)

    def _step_cb(pipe_obj, step_idx, timestep, callback_kwargs):
        pct = int((step_idx + 1) / num_steps * 100)
        log.info(f"    SD step {step_idx+1}/{num_steps} ({pct}%)")
        if job_id and job_id in jobs:
            base = int((scene_idx / num_scenes) * 40)
            slice_size = int(40 / num_scenes)
            step_pct = base + int((step_idx + 1) / num_steps * slice_size)
            jobs[job_id].update(
                progress=step_pct,
                message=f"Scene {scene_idx+1}/{num_scenes} — image step {step_idx+1}/{num_steps}",
            )
        return callback_kwargs

    with torch.no_grad():
        result = pipe(
            prompt=full_prompt,
            negative_prompt=negative,
            num_inference_steps=num_steps,
            guidance_scale=7.5,
            width=512,
            height=512,
            callback_on_step_end=_step_cb,
        )

    result.images[0].save(str(out_path))
    log.info(f"  [scene {scene_idx+1}/{num_scenes}] Image saved → {out_path}")
    return out_path


# ── Hindi TTS: MMS-TTS (no login) + IndicF5 (gated, better quality) ──────────

_mms_tts_cache = {}  # key: "pipe"

def _get_mms_pipe():
    """Load facebook/mms-tts-hin, cached. No HF login required."""
    if "pipe" in _mms_tts_cache:
        return _mms_tts_cache["pipe"]

    log.info(f"Loading MMS-TTS Hindi on {DEVICE}…")
    from transformers import pipeline as hf_pipeline

    _src, _kw = _pretrained_source(MMS_HINDI_MODEL_ID)
    tts = hf_pipeline(
        "text-to-speech",
        model=_src,
        device=0 if DEVICE == "cuda" else -1,
        **_kw,
    )
    _mms_tts_cache["pipe"] = tts
    log.info("MMS-TTS Hindi ready.")
    return tts


def _ensure_indic_ref(voice_key: str) -> Path:
    """Download reference WAV from HuggingFace if not already cached."""
    fname, _ = INDIC_VOICE_REFS[voice_key]
    local_path = INDIC_REF_DIR / fname
    if local_path.exists():
        return local_path

    INDIC_REF_DIR.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    log.info(f"Downloading IndicF5 reference clip {fname}…")
    downloaded = hf_hub_download(
        repo_id=INDIC_MODEL_ID,
        filename=f"prompts/{fname}",
        local_dir=str(INDIC_REF_DIR),
    )
    src  = Path(downloaded)
    dest = INDIC_REF_DIR / fname
    if src != dest:
        import shutil
        shutil.copy2(str(src), str(dest))
    log.info(f"Reference clip saved → {dest}")
    return dest


def _get_indic_pipe():
    """Load IndicF5 model, cached. Requires HF login."""
    if "pipe" in _indic_tts_cache:
        return _indic_tts_cache["pipe"]

    log.info(f"Loading IndicF5 TTS on {DEVICE}…")
    from transformers import pipeline as hf_pipeline

    tts = hf_pipeline(
        "text-to-speech",
        model=INDIC_MODEL_ID,
        device=0 if DEVICE == "cuda" else -1,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    )
    _indic_tts_cache["pipe"] = tts
    log.info("IndicF5 TTS ready.")
    return tts


def generate_hindi_tts(text: str, voice_key: str, out_path: Path,
                       scene_idx: int = 0, num_scenes: int = 1) -> Path:
    """Generate Hindi audio — routes to MMS-TTS or IndicF5 based on voice_key."""
    import soundfile as sf
    import numpy as np

    log.info(f"  [scene {scene_idx+1}/{num_scenes}] TTS: '{text[:60]}…'")

    if voice_key in INDIC_VOICES:
        # ── IndicF5 path (gated, better quality) ─────────────────
        ref_path = _ensure_indic_ref(voice_key)
        _, ref_text = INDIC_VOICE_REFS[voice_key]
        tts = _get_indic_pipe()
        result = tts(
            text,
            forward_params={
                "ref_audio_path": str(ref_path),
                "ref_text": ref_text,
            },
        )
    else:
        # ── MMS-TTS Hindi path (no login required) ────────────────
        tts = _get_mms_pipe()
        result = tts(text)

    audio = np.array(result["audio"])
    if audio.ndim > 1:
        audio = audio[0]
    sf.write(str(out_path), audio, result["sampling_rate"])
    log.info(f"  [scene {scene_idx+1}/{num_scenes}] Audio saved → {out_path}")
    return out_path


# ── FFmpeg stitch ─────────────────────────────────────────────────────────────

def stitch_story_video(scene_images: list, scene_audios: list, out_path: Path,
                       transition: str = "fade", transition_duration: float = 0.8) -> Path:
    """
    Hold each still image for its audio duration, then stitch all scenes
    with optional xfade transitions between them.
    """
    import subprocess
    import tempfile

    tmp_dir = Path(tempfile.mkdtemp())
    scene_videos = []
    durations = []
    n_scenes = len(scene_images)
    log.info(f"[stitch] Building {n_scenes} scene clip(s) in {tmp_dir}…")

    # ── Step 1: render one video clip per scene (image + audio) ──
    for i, (img_path, audio_path) in enumerate(zip(scene_images, scene_audios)):
        log.info(f"[stitch] Scene {i+1}/{n_scenes} — probing audio {audio_path}")
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(audio_path)],
            capture_output=True, text=True,
        )
        duration = 5.0
        try:
            info = json.loads(probe.stdout)
            for stream in info.get("streams", []):
                if stream.get("codec_type") == "audio":
                    duration = float(stream.get("duration", 5.0))
                    break
        except Exception:
            pass
        durations.append(duration)
        log.info(f"[stitch] Scene {i+1}/{n_scenes} — audio duration={duration:.2f}s, encoding clip…")

        scene_mp4 = tmp_dir / f"scene_{i:03d}.mp4"
        subprocess.run([
            "ffmpeg", "-y",
            "-loop", "1", "-i", str(img_path),
            "-i", str(audio_path),
            "-c:v", "libx264", "-tune", "stillimage",
            "-c:a", "aac", "-b:a", "128k",
            "-pix_fmt", "yuv420p",
            "-vf", "scale=512:512:force_original_aspect_ratio=decrease,pad=512:512:(ow-iw)/2:(oh-ih)/2",
            "-r", "25",
            "-shortest",
            str(scene_mp4),
        ], check=True)
        scene_videos.append(scene_mp4)
        log.info(f"[stitch] Scene {i+1}/{n_scenes} — clip done → {scene_mp4}")

    # ── Step 2: stitch with xfade or plain concat ─────────────────
    log.info(f"[stitch] Joining {n_scenes} clips with transition={transition}…")
    if transition == "none" or len(scene_videos) == 1:
        # Simple concat — no re-encode needed
        concat_list = tmp_dir / "concat.txt"
        with open(concat_list, "w") as f:
            for sv in scene_videos:
                f.write(f"file '{sv}'\n")
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            str(out_path),
        ], check=True)
    else:
        # xfade filter chain — requires re-encoding
        # Build complex filtergraph:
        #   [0:v][1:v]xfade=transition=fade:duration=0.8:offset=<d0-dur>[v01];
        #   [v01][2:v]xfade=...offset=<d0+d1-dur>[v012]; ...
        # Audio: acrossfade each pair
        n = len(scene_videos)
        td = transition_duration

        # Input args
        inputs = []
        for sv in scene_videos:
            inputs += ["-i", str(sv)]

        # Build video filter chain
        vfilter_parts = []
        afilter_parts = []
        offset = 0.0
        prev_v = "0:v"
        prev_a = "0:a"

        for i in range(1, n):
            offset += durations[i - 1] - td
            out_v = f"v{i}" if i < n - 1 else "vout"
            out_a = f"a{i}" if i < n - 1 else "aout"

            vfilter_parts.append(
                f"[{prev_v}][{i}:v]xfade=transition={transition}:"
                f"duration={td}:offset={offset:.3f}[{out_v}]"
            )
            afilter_parts.append(
                f"[{prev_a}][{i}:a]acrossfade=d={td}[{out_a}]"
            )
            prev_v = out_v
            prev_a = out_a

        filter_complex = ";".join(vfilter_parts + afilter_parts)

        subprocess.run([
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            str(out_path),
        ], check=True)

    log.info(f"[stitch] Video written → {out_path}")
    return out_path


# ── Story generation worker ───────────────────────────────────────────────────

def run_story_generation(job_id: str, scenes: list, style: str, voice: str, slug: str,
                         transition: str = "fade", transition_duration: float = 0.8):
    jlog = make_job_logger(slug, LOG_STORY_DIR)
    jlog.info(f"[{job_id}] Story gen | scenes={len(scenes)} style={style} voice={voice}")
    log.info(f"[story:{slug}] Starting — {len(scenes)} scenes, style={style}, voice={voice}")

    job_dir = OUTPUT_STORY_DIR / slug
    job_dir.mkdir(parents=True, exist_ok=True)

    n = len(scenes)
    scene_image_paths = []
    scene_audio_paths = []
    scene_image_urls  = []
    start_time = time.time()

    try:
        # ── Phase 1: Generate images ──────────────────────────────────────────
        log.info(f"[story:{slug}] Phase 1/3 — generating {n} scene image(s) with SD 1.5 ({style})…")
        for i, scene in enumerate(scenes):
            jobs[job_id].update(
                status="generating_images",
                progress=int((i / n) * 40),
                message=f"Generating image for scene {i+1}/{n}…",
            )
            log.info(f"[story:{slug}] Scene {i+1}/{n} — image start")
            img_path = job_dir / f"scene_{i:03d}.png"
            generate_scene_image(
                scene.get("image_prompt", ""), style, img_path,
                job_id=job_id, scene_idx=i, num_scenes=n,
            )
            scene_image_paths.append(img_path)
            scene_image_urls.append(f"/outputs/story/{slug}/scene_{i:03d}.png")
            # Track intermediate files for Generated Media tab
            intermediates = jobs[job_id].get("intermediate_files", [])
            intermediates.append({"type": "image", "label": f"Scene {i+1} image",
                                   "url": f"/outputs/story/{slug}/scene_{i:03d}.png"})
            jobs[job_id].update(
                progress=int(((i + 1) / n) * 40),
                message=f"Scene {i+1}/{n} image done",
                scene_images=list(scene_image_urls),
                intermediate_files=intermediates,
            )
            log.info(f"[story:{slug}] Scene {i+1}/{n} — image DONE → {img_path}")
            jlog.info(f"[{job_id}] Scene {i+1} image done")

        # ── Phase 2: Generate TTS audio ───────────────────────────────────────
        log.info(f"[story:{slug}] Phase 2/3 — generating {n} Hindi voiceover(s)…")
        for i, scene in enumerate(scenes):
            jobs[job_id].update(
                status="generating_audio",
                progress=40 + int((i / n) * 40),
                message=f"Generating Hindi voiceover for scene {i+1}/{n}…",
            )
            log.info(f"[story:{slug}] Scene {i+1}/{n} — TTS start")
            narration = scene.get("narration", "").strip() or "दृश्य।"
            audio_path = job_dir / f"scene_{i:03d}.wav"
            generate_hindi_tts(narration, voice, audio_path,
                               scene_idx=i, num_scenes=n)
            scene_audio_paths.append(audio_path)
            intermediates = jobs[job_id].get("intermediate_files", [])
            intermediates.append({"type": "audio", "label": f"Scene {i+1} audio",
                                   "url": f"/outputs/story/{slug}/scene_{i:03d}.wav"})
            jobs[job_id].update(
                progress=40 + int(((i + 1) / n) * 40),
                message=f"Scene {i+1}/{n} audio done",
                intermediate_files=intermediates,
            )
            log.info(f"[story:{slug}] Scene {i+1}/{n} — TTS DONE → {audio_path}")
            jlog.info(f"[{job_id}] Scene {i+1} audio done")

        # ── Phase 3: Stitch video ─────────────────────────────────────────────
        log.info(f"[story:{slug}] Phase 3/3 — stitching video (transition={transition})…")
        jobs[job_id].update(status="stitching", progress=85, message="Stitching scenes into video…")
        video_filename = f"{slug}.mp4"
        out_path = OUTPUT_STORY_DIR / video_filename
        stitch_story_video(scene_image_paths, scene_audio_paths, out_path,
                           transition=transition, transition_duration=transition_duration)

        elapsed = int(time.time() - start_time)
        jobs[job_id].update(
            status="done", progress=100,
            video_file=video_filename,
            scene_count=n,
            message=f"Done in {elapsed}s — {n} scenes",
            elapsed_seconds=elapsed,
        )
        log.info(f"[story:{slug}] DONE → {out_path} ({elapsed}s)")
        jlog.info(f"[{job_id}] Story done → {out_path} ({elapsed}s)")

    except Exception as e:
        log.error(f"[story:{slug}] FAILED: {e}", exc_info=True)
        jlog.error(f"[{job_id}] Story failed: {e}", exc_info=True)
        jobs[job_id].update(status="failed", error=str(e), message=f"Failed: {e}")


# ── Story API routes ──────────────────────────────────────────────────────────

@app.route("/api/story_breakdown", methods=["POST"])
def api_story_breakdown():
    data       = request.get_json(force=True)
    script     = data.get("script", "").strip()
    num_scenes = max(2, min(8, int(data.get("num_scenes", 4))))
    llm_key    = data.get("llm", "qwen")

    if not script:
        return jsonify({"error": "script is required"}), 400
    if llm_key not in STORY_LLM_MODELS:
        llm_key = "qwen"

    try:
        scenes = llm_break_into_scenes(script, num_scenes, llm_key)
        return jsonify({"scenes": scenes})
    except Exception as e:
        log.error(f"Scene breakdown failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate_story", methods=["POST"])
def api_generate_story():
    data                = request.get_json(force=True)
    scenes              = data.get("scenes", [])
    style               = data.get("style", "realistic").lower().strip()
    voice               = data.get("voice", "mms_hindi").strip()
    title               = data.get("title", "").strip()
    transition          = data.get("transition", "fade").lower().strip()
    transition_duration = max(0.3, min(2.0, float(data.get("transition_duration", 0.8))))

    valid_transitions = {"fade","fadeblack","fadewhite","dissolve","wipeleft","wiperight",
                         "wipeup","wipedown","slideleft","slideright","circleopen",
                         "circleclose","pixelize","none"}

    if not scenes:
        return jsonify({"error": "scenes are required"}), 400
    if style not in STYLE_LORAS:
        style = "realistic"
    if voice not in INDIC_VOICE_REFS and voice != "mms_hindi":
        voice = "mms_hindi"
    if transition not in valid_transitions:
        transition = "fade"

    job_id = str(uuid.uuid4())
    slug   = slugify(title) if title else f"story-{job_id[:8]}"

    jobs[job_id] = {
        "job_id":          job_id,
        "type":            "story",
        "status":          "queued",
        "progress":        0,
        "message":         "Queued…",
        "scene_count":     len(scenes),
        "scene_images":    [],
        "video_file":      None,
        "error":           None,
        "elapsed_seconds": 0,
        "created_at":      time.time(),
        "intermediate_files": [],
        "slug":            slug,
        "title":           title or slug,
        "style":           style,
        "voice":           voice,
        "llm":             data.get("llm", ""),
    }

    threading.Thread(
        target=run_story_generation,
        args=(job_id, scenes, style, voice, slug),
        kwargs={"transition": transition, "transition_duration": transition_duration},
        daemon=True,
    ).start()

    log.info(f"[{job_id}] Story job: {len(scenes)} scenes, style={style}, voice={voice}, transition={transition}")
    return jsonify({"job_id": job_id}), 202


@app.route("/outputs/story/<path:filename>")
def serve_story_output(filename):
    return send_from_directory(str(OUTPUT_STORY_DIR), filename)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI Video Generation Server")
    parser.add_argument("--port", type=int, default=7860)
    cli_args = parser.parse_args()
    log.info(f"Starting AI Video Generation server on http://0.0.0.0:{cli_args.port}")
    app.run(host="0.0.0.0", port=cli_args.port, debug=False, threaded=True)
