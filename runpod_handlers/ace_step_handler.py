"""
runpod_handlers/ace_step_handler.py

RunPod Serverless handler for ACE-Step 1.5 music generation.

Input  JSON:  { "lyrics": "...", "style": "...", "duration": 90 }
Output JSON:  { "audio_b64": "<wav bytes base64>", "sample_rate": 44100, "duration_s": 90 }
              OR { "error": "...", "traceback": "..." }
"""
from __future__ import annotations

import base64
import io
import logging
import os
import time
import traceback

import runpod

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ace_step_handler")

# ── Model singleton ───────────────────────────────────────────────────────────
_pipeline = None


def _load_pipeline():
    global _pipeline

    if _pipeline is not None:
        return _pipeline

    import torch
    log.info(f"PyTorch version: {torch.__version__}")
    log.info(f"CUDA available: {torch.cuda.is_available()}")

    # Log installed ACE-Step version
    try:
        import importlib.metadata as _meta
        for pkg in ("ace-step", "acestep"):
            try:
                log.info(f"{pkg} version: {_meta.version(pkg)}")
            except _meta.PackageNotFoundError:
                pass
    except Exception:
        pass

    from acestep.pipeline_ace_step import ACEStepPipeline

    model_path = os.environ.get("MODEL_PATH", "ACE-Step/ACE-Step-v1.5-3.5B")
    log.info(f"Loading ACE-Step from: {model_path}")
    t0 = time.time()

    _pipeline = ACEStepPipeline(
        checkpoint_dir=model_path,
        dtype=torch.bfloat16,
        device="cuda",
    )

    log.info(f"ACE-Step pipeline ready in {time.time() - t0:.1f}s")

    # Log pipeline signature so we can verify parameter names in RunPod logs
    import inspect
    try:
        call_sig = inspect.signature(_pipeline.__call__)
        log.info(f"Pipeline.__call__ params: {list(call_sig.parameters.keys())}")
    except Exception as e:
        log.warning(f"Could not inspect pipeline signature: {e}")

    return _pipeline


# ── Handler ───────────────────────────────────────────────────────────────────

def handler(job: dict) -> dict:
    inp = job.get("input", {})

    lyrics   = inp.get("lyrics", "").strip()
    style    = inp.get("style", "cheerful children's nursery rhyme, playful female voice, ukulele, slow and clear")
    duration = float(inp.get("duration", 90))

    if not lyrics:
        return {"error": "No lyrics provided"}

    try:
        import soundfile as sf
        import numpy as np

        pipe = _load_pipeline()

        log.info(f"Generating {duration}s  |  style: {style[:80]}")
        t0 = time.time()

        result = pipe(
            task_type="text2music",
            caption=style,
            lyrics=lyrics,
            audio_duration=duration,
            infer_steps=60,
            guidance_scale=7.5,
            scheduler_type="euler",
        )

        log.info(f"Generation done in {time.time() - t0:.1f}s  |  result type: {type(result)}")
        log.info(f"Result attrs: {[a for a in dir(result) if not a.startswith('_')]}")

        # ── Extract audio array — try multiple output formats ─────────────────
        audio = None
        sr    = 44100

        if hasattr(result, "audios") and result.audios is not None:
            audio = result.audios[0] if hasattr(result.audios, "__getitem__") else result.audios
            sr    = getattr(result, "sample_rate", sr)
        elif hasattr(result, "audio"):
            audio = result.audio[0] if hasattr(result.audio, "__getitem__") else result.audio
            sr    = getattr(result, "sample_rate", sr)
        elif isinstance(result, (list, tuple)) and len(result) > 0:
            audio = result[0]
        else:
            return {"error": f"Unknown pipeline output format: {type(result)} / attrs={dir(result)}"}

        if audio is None:
            return {"error": "Pipeline returned None audio"}

        audio = np.array(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio[0]  # take first channel if stereo

        actual_dur = len(audio) / sr
        log.info(f"Audio: {len(audio)} samples @ {sr}Hz = {actual_dur:.1f}s")

        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        audio_b64 = base64.b64encode(buf.read()).decode("utf-8")

        return {
            "audio_b64":   audio_b64,
            "sample_rate": sr,
            "duration_s":  round(actual_dur, 2),
        }

    except Exception as exc:
        tb = traceback.format_exc()
        log.error(f"Handler error:\n{tb}")
        return {"error": str(exc), "traceback": tb}


runpod.serverless.start({"handler": handler})
