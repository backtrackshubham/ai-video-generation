"""
AI Video Generation Backend
Flask service supporting three generation modes:

1. Text-to-Video  — CogVideoX-5B
   POST /api/generate          { prompt, title, duration_seconds, fps, num_steps, guidance_scale }

2. Image-to-Video — Stable Video Diffusion 1.1
   POST /api/generate_i2v      multipart/form-data: image, title, motion_bucket_id, num_steps, fps

3. Motion Stickman — MDM (Motion Diffusion Model)
   POST /api/generate_stickman { prompt, title }

Branches:
  master              — CPU-only, Linux, hardcoded /localdisk/ciena/ paths
  feature/cuda-windows — CUDA-enabled, Windows, fully portable (this file)
"""

import os
import sys
import uuid
import time
import threading
import json
import logging
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR             = Path(__file__).parent.resolve()
MODEL_DIR            = BASE_DIR / "models"
OUTPUT_DIR           = BASE_DIR / "outputs"
OUTPUT_VIDEO_DIR     = OUTPUT_DIR / "normal-videos"
OUTPUT_STICKMAN_DIR  = OUTPUT_DIR / "stickman-videos"
OUTPUT_I2V_DIR       = OUTPUT_DIR / "i2v-videos"
LOG_DIR              = BASE_DIR / "gen-logs"
LOG_VIDEO_DIR        = LOG_DIR / "normal-videos"
LOG_STICKMAN_DIR     = LOG_DIR / "stickman-videos"
LOG_I2V_DIR          = LOG_DIR / "i2v-videos"
T2M_REPO             = BASE_DIR / "t2m_gpt"
MDM_REPO             = BASE_DIR / "mdm"
UPLOAD_DIR           = BASE_DIR / "uploads"

for d in (OUTPUT_VIDEO_DIR, OUTPUT_STICKMAN_DIR, OUTPUT_I2V_DIR,
          LOG_VIDEO_DIR, LOG_STICKMAN_DIR, LOG_I2V_DIR, UPLOAD_DIR):
    os.makedirs(d, exist_ok=True)

# ── Model caches inside repo ──────────────────────────────────────────────────
os.environ.setdefault("HF_HOME",           str(MODEL_DIR / "hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(MODEL_DIR / "hf_cache"))
os.environ.setdefault("DIFFUSERS_CACHE",    str(MODEL_DIR / "hf_cache"))
os.environ.setdefault("TORCH_HOME",         str(MODEL_DIR / "torch_cache"))

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

jobs: dict = {}

# ── Time estimation ───────────────────────────────────────────────────────────
# CogVideoX-5B: ~3-4 s/step on A100 (80GB), ~8-12 s/step on RTX 3090 (24GB)
# CPU: very slow, not really practical
SECS_PER_STEP_COG = 8 if DEVICE == "cuda" else 120   # per diffusion step


def estimate_time(num_frames: int, num_steps: int, model: str = "cogvideox") -> dict:
    if model == "cogvideox":
        # CogVideoX denoises across all frames simultaneously (not per-frame)
        total_seconds = num_steps * SECS_PER_STEP_COG
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
# TEXT-TO-VIDEO — CogVideoX-5B
# ═══════════════════════════════════════════════════════════════════════════════
_cogvx_pipeline = None
_cogvx_lock     = threading.Lock()

COG_MODEL_ID = "THUDM/CogVideoX-5b"


def get_cogvx_pipeline():
    global _cogvx_pipeline
    if _cogvx_pipeline is not None:
        return _cogvx_pipeline
    with _cogvx_lock:
        if _cogvx_pipeline is not None:
            return _cogvx_pipeline
        log.info(f"Loading CogVideoX-5B pipeline ({DEVICE.upper()})…")
        try:
            from diffusers import CogVideoXPipeline

            dtype = torch.float16 if DEVICE == "cuda" else torch.float32
            pipe = CogVideoXPipeline.from_pretrained(
                COG_MODEL_ID,
                cache_dir=str(MODEL_DIR / "hf_cache"),
                torch_dtype=dtype,
            )
            pipe = pipe.to(DEVICE)
            # Memory optimisations
            pipe.enable_model_cpu_offload()
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
            _cogvx_pipeline = pipe
            log.info("CogVideoX-5B loaded successfully.")
        except Exception as e:
            log.error(f"Failed to load CogVideoX pipeline: {e}")
            raise
    return _cogvx_pipeline


def run_generation(job_id: str, prompt: str, num_frames: int, num_steps: int,
                   guidance_scale: float, slug: str):
    jlog = make_job_logger(slug, LOG_VIDEO_DIR)
    jlog.info(f"[{job_id}] Starting CogVideoX T2V | prompt='{prompt}' frames={num_frames} steps={num_steps} cfg={guidance_scale}")

    try:
        import imageio
        import numpy as np
    except Exception as e:
        jobs[job_id].update(status="failed", error=str(e), message=f"Import failed: {e}")
        return

    jobs[job_id].update(status="loading_model", message="Loading CogVideoX-5B…", progress=2)

    try:
        pipe = get_cogvx_pipeline()
    except Exception as e:
        jobs[job_id].update(status="failed", error=str(e), message=f"Model load failed: {e}")
        jlog.error(f"[{job_id}] Model load failed: {e}")
        return

    jobs[job_id].update(status="generating", message="Generating frames…", progress=5)

    start_time = time.time()
    step_ref = [0]

    def step_callback(pipe_obj, step, timestep, callback_kwargs):
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
        jlog.info(f"[{job_id}] Step {step_ref[0]}/{num_steps} elapsed={int(elapsed)}s remaining={int(remaining)}s")
        return callback_kwargs

    try:
        with torch.no_grad():
            output = pipe(
                prompt=prompt,
                num_frames=num_frames,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                callback_on_step_end=step_callback,
                callback_on_step_end_tensor_inputs=["latents"],
            )

        # output.frames is a list of PIL images or numpy arrays
        frames = output.frames[0]  # first (only) video

        video_filename = f"{slug}.mp4"
        out_path = OUTPUT_VIDEO_DIR / video_filename

        # Convert PIL frames to numpy if needed
        if hasattr(frames[0], "numpy"):
            frames_np = [f.numpy() for f in frames]
        elif hasattr(frames[0], "tobytes"):
            frames_np = [np.array(f) for f in frames]
        else:
            frames_np = frames

        imageio.mimwrite(str(out_path), frames_np, fps=8, codec="libx264", quality=8)

        elapsed_total = int(time.time() - start_time)
        jobs[job_id].update(
            status="done", progress=100,
            video_file=video_filename,
            message=f"Done in {elapsed_total}s — {len(frames_np)} frames",
            elapsed_seconds=elapsed_total,
        )
        jlog.info(f"[{job_id}] Generation complete in {elapsed_total}s: {out_path}")

    except Exception as e:
        jlog.error(f"[{job_id}] Generation error: {e}", exc_info=True)
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
            pipe = StableVideoDiffusionPipeline.from_pretrained(
                SVD_MODEL_ID,
                cache_dir=str(MODEL_DIR / "hf_cache"),
                torch_dtype=dtype,
            )
            pipe = pipe.to(DEVICE)
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
        from PIL import Image
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

        step_ref = [0]

        def step_callback(pipe_obj, step, timestep, callback_kwargs):
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
            jlog.info(f"[{job_id}] Step {step_ref[0]}/{num_steps}")
            return callback_kwargs

        with torch.no_grad():
            output = pipe(
                image,
                num_inference_steps=num_steps,
                motion_bucket_id=motion_bucket_id,
                fps=fps,
                decode_chunk_size=8,
                callback_on_step_end=step_callback,
                callback_on_step_end_tensor_inputs=["latents"],
            )

        frames = output.frames[0]
        frames_np = [np.array(f) for f in frames]

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

        # Load model
        from diffusion.gaussian_diffusion import create_gaussian_diffusion
        from utils.model_util import create_model_and_diffusion

        # MDM's create_model_and_diffusion only uses data.dataset.num_actions (optional)
        class _FakeData:
            class dataset: pass
        model, diffusion = create_model_and_diffusion(args, _FakeData())
        log.info(f"Loading MDM checkpoint: {args.model_path}")
        state_dict = torch.load(args.model_path, map_location="cpu")
        load_model_wo_clip(model, state_dict)
        model = ClassifierFreeSampleModel(model)
        model.to(DEVICE)
        model.eval()

        _mdm_models = (model, diffusion, args, paramUtil, plot_3d_motion)
        log.info("MDM loaded successfully.")

    return _mdm_models


def run_stickman_generation(job_id: str, prompt: str, slug: str):
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

    try:
        model, diffusion, args, paramUtil, plot_3d_motion = get_mdm_models()
    except Exception as e:
        jobs[job_id].update(status="failed", error=str(e), message=f"Model load failed: {e}")
        jlog.error(f"[{job_id}] MDM load failed: {e}", exc_info=True)
        return

    jobs[job_id].update(status="generating", message="Generating motion…", progress=20)

    try:
        if str(MDM_REPO) not in sys.path:
            sys.path.insert(0, str(MDM_REPO))

        import numpy as np
        from data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain
        from data_loaders.humanml.scripts.motion_process import recover_from_ric

        n_frames = 196   # max HumanML3D length

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
                "scale":      torch.ones(1, device=DEVICE) * args.guidance_param,
                "text_embed": text_embed,
            }
        }

        # Sample
        with torch.no_grad():
            sample = diffusion.p_sample_loop(
                model,
                (1, model.njoints, model.nfeats, n_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )  # (1, njoints=263, nfeats=1, nframes)

        jobs[job_id].update(progress=70, message="Rendering animation…")

        # Un-normalize using pre-downloaded mean/std
        mean = np.load(str(MDM_REPO / "dataset" / "t2m_mean.npy"))   # (263,)
        std  = np.load(str(MDM_REPO / "dataset" / "t2m_std.npy"))    # (263,)

        # sample: (1, 263, 1, 196) → (1, 196, 1, 263) → (1, 196, 263)
        sample_np = sample.cpu().permute(0, 2, 3, 1).squeeze(2).numpy()  # (1, 196, 263)
        sample_np = sample_np * std + mean  # un-normalize

        # recover_from_ric: (B, T, 263) → (B, T, 22, 3)
        joints_tensor = recover_from_ric(torch.tensor(sample_np).float(), 22)
        joints = joints_tensor.squeeze(0).numpy()  # (T, 22, 3)

        T = joints.shape[0]
        jlog.info(f"[{job_id}] Generated {T} motion frames")

        # Render
        video_filename = f"{slug}.mp4"
        out_path = str(OUTPUT_STICKMAN_DIR / video_filename)

        plot_3d_motion(
            out_path,
            t2m_kinematic_chain,
            joints,
            title=prompt,
            dataset="humanml",
            fps=20,
            radius=3,
        )

        jobs[job_id].update(
            status="done", progress=100,
            video_file=video_filename,
            message=f"Done — {T} frames",
        )
        jlog.info(f"[{job_id}] Stickman complete: {out_path}")

    except Exception as e:
        jlog.error(f"[{job_id}] Stickman error: {e}", exc_info=True)
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
    model     = data.get("model", "cogvideox")
    # CogVideoX generates at ~8fps natively (49 frames for 6s)
    num_frames = max(8, min(int(duration * fps), 49))
    est = estimate_time(num_frames, num_steps, model)
    est["num_frames"] = num_frames
    return jsonify(est)


@app.route("/api/generate", methods=["POST"])
def api_generate():
    data           = request.get_json(force=True)
    prompt         = data.get("prompt", "").strip()
    title          = data.get("title", "").strip()
    duration       = float(data.get("duration_seconds", 5))
    fps            = int(data.get("fps", 8))
    num_steps      = int(data.get("num_steps", 50))
    guidance_scale = float(data.get("guidance_scale", 6.0))

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    # CogVideoX: must be multiple of 8 + 1 (e.g. 49 frames)
    raw_frames = int(duration * fps)
    num_frames = min(max(raw_frames, 8), 49)
    # snap to valid CogVideoX frame count (8k+1)
    num_frames = ((num_frames - 1) // 8) * 8 + 1

    job_id = str(uuid.uuid4())
    slug   = slugify(title) if title else f"video-{job_id[:8]}"
    est    = estimate_time(num_frames, num_steps, "cogvideox")

    jobs[job_id] = {
        "job_id":            job_id,
        "type":              "t2v",
        "status":            "queued",
        "progress":          0,
        "message":           "Queued…",
        "prompt":            prompt,
        "title":             title or slug,
        "slug":              slug,
        "num_frames":        num_frames,
        "num_steps":         num_steps,
        "video_file":        None,
        "error":             None,
        "elapsed_seconds":   0,
        "remaining_seconds": est["high_seconds"],
        "estimate":          est,
        "created_at":        time.time(),
    }

    t = threading.Thread(
        target=run_generation,
        args=(job_id, prompt, num_frames, num_steps, guidance_scale, slug),
        daemon=True,
    )
    t.start()

    log.info(f"[{job_id}] T2V started: slug='{slug}' frames={num_frames} steps={num_steps} cfg={guidance_scale}")
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
    }

    t = threading.Thread(target=run_stickman_generation, args=(job_id, prompt, slug), daemon=True)
    t.start()

    log.info(f"[{job_id}] Stickman started: slug='{slug}' prompt='{prompt}'")
    return jsonify({"job_id": job_id}), 202


@app.route("/api/status/<job_id>", methods=["GET"])
def api_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    return jsonify(job)


@app.route("/api/video/<filename>", methods=["GET"])
def api_video(filename):
    for d in (OUTPUT_VIDEO_DIR, OUTPUT_I2V_DIR, OUTPUT_STICKMAN_DIR):
        if (d / filename).exists():
            return send_from_directory(str(d), filename)
    return jsonify({"error": "file not found"}), 404


@app.route("/api/jobs", methods=["GET"])
def api_jobs():
    recent = sorted(jobs.values(), key=lambda j: j["created_at"], reverse=True)[:20]
    return jsonify(recent)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    cli_args = parser.parse_args()
    log.info(f"Starting AI Video Generation server on http://0.0.0.0:{cli_args.port}")
    app.run(host="0.0.0.0", port=cli_args.port, debug=False, threaded=True)
