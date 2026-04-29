"""
AI Video Generation Backend
Flask service that accepts a text prompt and generates a video using
a lightweight pretrained diffusion model (CPU-compatible).

Model used: damo-vilab/text-to-video-ms-1.7b (ModelScope)
- Smallest widely-used text-to-video model (~3.5GB)
- Works on CPU (slow but functional)
- Outputs short video clips frame by frame

Also supports stickman motion video generation via T2M-GPT:
- Text-to-motion transformer (CPU-compatible)
- Generates 3D human skeleton animations as MP4
"""

import os
import sys
import uuid
import time
import threading
import json
import logging
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path("/localdisk/ciena/ai-video-generation")
MODEL_DIR  = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
LOG_DIR    = BASE_DIR / "logs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,    exist_ok=True)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "server.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# ── Job state store ──────────────────────────────────────────────────────────
jobs: dict = {}   # job_id -> { status, progress, message, video_file, error }

# ── Model (lazy-loaded once) ─────────────────────────────────────────────────
_pipeline = None
_pipeline_lock = threading.Lock()

MODEL_ID = "damo-vilab/text-to-video-ms-1.7b"

# ── Time estimation (CPU, based on empirical benchmarks) ─────────────────────
# ModelScope 1.7B on CPU: ~90-150 sec/frame at default inference steps (50)
# We use 20 steps for speed; ~30-50 sec/frame on a 10-core Xeon
SECS_PER_FRAME_CPU = 40  # conservative estimate per frame


def estimate_time(num_frames: int, num_inference_steps: int) -> dict:
    """Return human-readable time estimate."""
    total_seconds = num_frames * num_inference_steps * (SECS_PER_FRAME_CPU / 20)
    low  = int(total_seconds * 0.8)
    high = int(total_seconds * 1.3)

    def fmt(s):
        if s < 60:
            return f"{s}s"
        m, sec = divmod(s, 60)
        return f"{m}m {sec}s" if sec else f"{m}m"

    return {
        "low_seconds":  low,
        "high_seconds": high,
        "display":      f"{fmt(low)} – {fmt(high)}",
        "frames":       num_frames,
        "steps":        num_inference_steps,
    }


def get_pipeline():
    """Lazy-load the ModelScope text-to-video pipeline."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    with _pipeline_lock:
        if _pipeline is not None:  # double-checked
            return _pipeline

        log.info("Loading ModelScope text-to-video pipeline (CPU)…")
        try:
            from diffusers import DiffusionPipeline
            import torch

            pipe = DiffusionPipeline.from_pretrained(
                MODEL_ID,
                cache_dir=str(MODEL_DIR),
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            pipe = pipe.to("cpu")
            # Reduce memory footprint
            pipe.enable_attention_slicing()
            _pipeline = pipe
            log.info("Pipeline loaded successfully.")
        except Exception as e:
            log.error(f"Failed to load pipeline: {e}")
            raise
    return _pipeline


def run_generation(job_id: str, prompt: str, num_frames: int, num_steps: int):
    """Background thread: runs video generation and updates job state."""
    import torch
    import imageio

    jobs[job_id]["status"]  = "loading_model"
    jobs[job_id]["message"] = "Loading model into memory…"
    jobs[job_id]["progress"] = 2

    try:
        pipe = get_pipeline()
    except Exception as e:
        jobs[job_id]["status"]  = "failed"
        jobs[job_id]["error"]   = str(e)
        jobs[job_id]["message"] = f"Model load failed: {e}"
        return

    jobs[job_id]["status"]   = "generating"
    jobs[job_id]["message"]  = "Generating frames…"
    jobs[job_id]["progress"] = 5

    frame_count = [0]
    start_time  = time.time()

    def step_callback(pipe_obj, step, timestep, callback_kwargs):
        """Called after each diffusion step to update progress."""
        completed_steps = step + 1
        pct = 5 + int((completed_steps / num_steps) * 90)
        elapsed  = time.time() - start_time
        per_step = elapsed / completed_steps if completed_steps else 0
        remaining = per_step * (num_steps - completed_steps)

        jobs[job_id]["progress"] = pct
        jobs[job_id]["elapsed_seconds"]   = int(elapsed)
        jobs[job_id]["remaining_seconds"] = int(remaining)
        jobs[job_id]["message"] = (
            f"Step {completed_steps}/{num_steps} — "
            f"elapsed {int(elapsed)}s, ~{int(remaining)}s remaining"
        )
        return callback_kwargs

    try:
        with torch.no_grad():
            output = pipe(
                prompt,
                num_frames=num_frames,
                num_inference_steps=num_steps,
                callback_on_step_end=step_callback,
                callback_on_step_end_tensor_inputs=["latents"],
            )

        frames = output.frames[0]  # list of PIL images

        out_path = OUTPUT_DIR / f"{job_id}.mp4"
        imageio.mimwrite(
            str(out_path),
            [frame for frame in frames],
            fps=8,
            codec="libx264",
            quality=8,
        )

        elapsed_total = int(time.time() - start_time)
        jobs[job_id]["status"]    = "done"
        jobs[job_id]["progress"]  = 100
        jobs[job_id]["video_file"] = f"{job_id}.mp4"
        jobs[job_id]["message"]   = f"Done in {elapsed_total}s — {len(frames)} frames"
        jobs[job_id]["elapsed_seconds"] = elapsed_total
        log.info(f"[{job_id}] Generation complete: {out_path}")

    except Exception as e:
        log.error(f"[{job_id}] Generation error: {e}", exc_info=True)
        jobs[job_id]["status"]  = "failed"
        jobs[job_id]["error"]   = str(e)
        jobs[job_id]["message"] = f"Generation failed: {e}"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


@app.route("/api/estimate", methods=["POST"])
def api_estimate():
    """Return time estimate before starting generation."""
    data       = request.get_json(force=True)
    duration   = float(data.get("duration_seconds", 5))   # desired video length
    fps        = int(data.get("fps", 8))
    num_steps  = int(data.get("num_steps", 20))
    num_frames = max(8, min(int(duration * fps), 64))      # cap at 64 frames
    est        = estimate_time(num_frames, num_steps)
    est["num_frames"] = num_frames
    return jsonify(est)


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """Start a video generation job."""
    data       = request.get_json(force=True)
    prompt     = data.get("prompt", "").strip()
    duration   = float(data.get("duration_seconds", 5))
    fps        = int(data.get("fps", 8))
    num_steps  = int(data.get("num_steps", 20))

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    num_frames = max(8, min(int(duration * fps), 64))

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id":            job_id,
        "status":            "queued",
        "progress":          0,
        "message":           "Queued…",
        "prompt":            prompt,
        "num_frames":        num_frames,
        "num_steps":         num_steps,
        "video_file":        None,
        "error":             None,
        "elapsed_seconds":   0,
        "remaining_seconds": estimate_time(num_frames, num_steps)["high_seconds"],
        "estimate":          estimate_time(num_frames, num_steps),
        "created_at":        time.time(),
    }

    t = threading.Thread(
        target=run_generation,
        args=(job_id, prompt, num_frames, num_steps),
        daemon=True,
    )
    t.start()

    log.info(f"[{job_id}] Started: '{prompt}' | frames={num_frames} steps={num_steps}")
    return jsonify({"job_id": job_id, "estimate": jobs[job_id]["estimate"]}), 202


@app.route("/api/status/<job_id>", methods=["GET"])
def api_status(job_id):
    """Poll job status."""
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    return jsonify(job)


@app.route("/api/video/<filename>", methods=["GET"])
def api_video(filename):
    """Stream the generated video file."""
    return send_from_directory(str(OUTPUT_DIR), filename)


@app.route("/api/jobs", methods=["GET"])
def api_jobs():
    """List recent jobs."""
    recent = sorted(jobs.values(), key=lambda j: j["created_at"], reverse=True)[:20]
    return jsonify(recent)


# ── Stickman (T2M-GPT) generation ────────────────────────────────────────────

T2M_REPO = Path("/localdisk/ciena/T2M-GPT")

_stickman_models = None
_stickman_lock = threading.Lock()


def get_stickman_models():
    """Lazy-load T2M-GPT models onto CPU (once)."""
    global _stickman_models
    if _stickman_models is not None:
        return _stickman_models

    with _stickman_lock:
        if _stickman_models is not None:
            return _stickman_models

        log.info("Loading T2M-GPT stickman models (CPU)…")
        import torch

        # Monkey-patch .cuda() -> .cpu()
        torch.Tensor.cuda = lambda self, *a, **k: self.cpu()
        torch.nn.Module.cuda = lambda self, *a, **k: self.cpu()

        # Add T2M-GPT to path
        if str(T2M_REPO) not in sys.path:
            sys.path.insert(0, str(T2M_REPO))

        import clip
        import options.option_transformer as option_trans
        import models.vqvae as vqvae
        import models.t2m_trans as trans

        device = torch.device("cpu")

        # Parse args with correct checkpoint dimensions
        old_argv = sys.argv[:]
        sys.argv = ["flask"]
        args = option_trans.get_args_parser()
        sys.argv = old_argv
        args.dataname = "t2m"
        args.resume_pth   = str(T2M_REPO / "pretrained/VQVAE/net_last.pth")
        args.resume_trans = str(T2M_REPO / "pretrained/VQTransformer_corruption05/net_best_fid.pth")
        args.down_t = 2; args.depth = 3; args.block_size = 51
        args.embed_dim_gpt = 1024; args.num_layers = 9
        args.n_head_gpt = 16; args.clip_dim = 512

        clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
        clip_model.float()
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        net = vqvae.HumanVQVAE(args, args.nb_code, args.code_dim, args.output_emb_width,
                                args.down_t, args.stride_t, args.width, args.depth,
                                args.dilation_growth_rate)
        net.load_state_dict(torch.load(args.resume_pth, map_location="cpu")["net"], strict=True)
        net.eval()

        te = trans.Text2Motion_Transformer(num_vq=args.nb_code, embed_dim=args.embed_dim_gpt,
                                           clip_dim=args.clip_dim, block_size=args.block_size,
                                           num_layers=args.num_layers, n_head=args.n_head_gpt,
                                           drop_out_rate=args.drop_out_rate, fc_rate=args.ff_rate)
        te.load_state_dict(torch.load(args.resume_trans, map_location="cpu")["trans"], strict=True)
        te.eval()

        _stickman_models = (clip_model, net, te, args)
        log.info("T2M-GPT models loaded successfully.")
    return _stickman_models


def run_stickman_generation(job_id: str, prompt: str):
    """Background thread: runs T2M-GPT stickman generation."""
    import torch
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from mpl_toolkits.mplot3d import Axes3D
    import imageio_ffmpeg
    import clip

    T2M_KINEMATIC_CHAIN = [
        [0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15],
        [9, 14, 17, 19, 21], [9, 13, 16, 18, 20],
    ]
    CHAIN_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

    if str(T2M_REPO) not in sys.path:
        sys.path.insert(0, str(T2M_REPO))
    from utils.motion_process import recover_from_ric

    jobs[job_id]["status"]   = "loading_model"
    jobs[job_id]["message"]  = "Loading stickman models…"
    jobs[job_id]["progress"] = 5

    try:
        clip_model, net, te, args = get_stickman_models()
    except Exception as e:
        jobs[job_id]["status"]  = "failed"
        jobs[job_id]["error"]   = str(e)
        jobs[job_id]["message"] = f"Model load failed: {e}"
        return

    jobs[job_id]["status"]   = "generating"
    jobs[job_id]["message"]  = "Generating motion tokens…"
    jobs[job_id]["progress"] = 20

    try:
        text_tokens = clip.tokenize([prompt], truncate=True).to(torch.device("cpu"))
        with torch.no_grad():
            text_feat = clip_model.encode_text(text_tokens).float()
            index_motion = te.sample(text_feat, if_categorial=True)
            pred_pose = net.forward_decoder(index_motion)

        pred_denorm = pred_pose.detach().cpu()
        joints = recover_from_ric(pred_denorm, 22).squeeze(0).numpy()  # (T, 22, 3)
        T = joints.shape[0]
        log.info(f"[{job_id}] Generated {T} frames")

        jobs[job_id]["progress"] = 70
        jobs[job_id]["message"]  = f"Rendering {T} frames…"

        # Render stickman
        x_min, x_max = joints[:, :, 0].min(), joints[:, :, 0].max()
        y_min, y_max = joints[:, :, 1].min(), joints[:, :, 1].max()
        z_min, z_max = joints[:, :, 2].min(), joints[:, :, 2].max()

        fig = plt.figure(figsize=(6, 6), facecolor="black")
        ax = fig.add_subplot(111, projection="3d")

        def update(frame_idx):
            ax.cla()
            ax.set_facecolor("black")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            ax.set_axis_off()
            pose = joints[frame_idx]
            for chain, color in zip(T2M_KINEMATIC_CHAIN, CHAIN_COLORS):
                xs = [pose[j, 0] for j in chain]
                ys = [pose[j, 1] for j in chain]
                zs = [pose[j, 2] for j in chain]
                ax.plot(xs, ys, zs, color=color, linewidth=2.5)
                ax.scatter(xs, ys, zs, color=color, s=20, zorder=5)

        ani = animation.FuncAnimation(fig, update, frames=T, interval=50)
        out_path = OUTPUT_DIR / f"{job_id}.mp4"
        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        writer = animation.FFMpegWriter(fps=20, bitrate=1800)
        ani.save(str(out_path), writer=writer)
        plt.close(fig)

        jobs[job_id]["status"]     = "done"
        jobs[job_id]["progress"]   = 100
        jobs[job_id]["video_file"] = f"{job_id}.mp4"
        jobs[job_id]["message"]    = f"Done — {T} frames stickman animation"
        log.info(f"[{job_id}] Stickman generation complete: {out_path}")

    except Exception as e:
        log.error(f"[{job_id}] Stickman generation error: {e}", exc_info=True)
        jobs[job_id]["status"]  = "failed"
        jobs[job_id]["error"]   = str(e)
        jobs[job_id]["message"] = f"Generation failed: {e}"


@app.route("/api/generate_stickman", methods=["POST"])
def api_generate_stickman():
    """Start a stickman motion video generation job."""
    data   = request.get_json(force=True)
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id":            job_id,
        "status":            "queued",
        "progress":          0,
        "message":           "Queued…",
        "prompt":            prompt,
        "type":              "stickman",
        "video_file":        None,
        "error":             None,
        "elapsed_seconds":   0,
        "remaining_seconds": 60,
        "created_at":        time.time(),
    }

    t = threading.Thread(target=run_stickman_generation, args=(job_id, prompt), daemon=True)
    t.start()

    log.info(f"[{job_id}] Stickman started: '{prompt}'")
    return jsonify({"job_id": job_id}), 202


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    # Redirect HF/model caches to localdisk (set by start.sh env vars too)
    os.environ.setdefault("HF_HOME",            str(MODEL_DIR / "hf_cache"))
    os.environ.setdefault("TRANSFORMERS_CACHE",  str(MODEL_DIR / "hf_cache"))
    os.environ.setdefault("DIFFUSERS_CACHE",     str(MODEL_DIR / "hf_cache"))
    os.environ.setdefault("TORCH_HOME",          str(MODEL_DIR / "torch_cache"))

    log.info(f"Starting AI Video Generation server on http://0.0.0.0:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
