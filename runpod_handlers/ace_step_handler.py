"""
runpod_handlers/ace_step_handler.py

RunPod Serverless handler for ACE-Step 1.5 music generation.
Receives lyrics + style prompt, returns a base64-encoded WAV.

Deploy steps (do once):
  1. Build & push the Docker image (see Dockerfile in this folder).
  2. In RunPod dashboard → Serverless → New Endpoint:
       - Docker image: your_dockerhub/ace-step-handler:latest
       - GPU: RTX 4090 (24GB VRAM) — cheapest that fits the 3.5B model
       - Network volume: mount at /runpod-volume  (stores model weights)
       - Env var: MODEL_PATH=/runpod-volume/ace-step-1.5
  3. Copy the Endpoint ID into .env: ACE_STEP_ENDPOINT_ID=xxx

Input  JSON:  { "lyrics": "...", "style": "...", "duration": 90 }
Output JSON:  { "audio_b64": "<wav bytes base64>", "sample_rate": 44100, "duration_s": 90 }
"""
from __future__ import annotations

import base64
import io
import logging
import os
import time

import runpod

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ace_step_handler")

# ── Model (loaded once per worker, stays warm between calls) ──────────────────
_pipeline = None
_sample_rate = 44100

def _load_pipeline():
    global _pipeline, _sample_rate

    if _pipeline is not None:
        return _pipeline

    import torch
    from acestep.pipeline_ace_step import ACEStepPipeline

    model_path = os.environ.get("MODEL_PATH", "ACE-Step/ACE-Step-v1.5-3.5B")
    log.info(f"Loading ACE-Step from: {model_path}")
    t0 = time.time()

    _pipeline = ACEStepPipeline(
        checkpoint_dir=model_path,
        dtype=torch.bfloat16,
        device="cuda",
    )
    # Some versions expose .to(); call it if available
    if hasattr(_pipeline, "to"):
        _pipeline = _pipeline.to("cuda")

    # Warm-up: 1-second silent generation to pre-allocate CUDA memory
    try:
        _pipeline(
            task_type="text2music",
            caption="test",
            lyrics="[verse]\nhello world",
            audio_duration=2.0,
            infer_steps=10,
        )
    except Exception:
        pass

    log.info(f"ACE-Step ready in {time.time() - t0:.1f}s")
    return _pipeline


# ── Handler ───────────────────────────────────────────────────────────────────

def handler(job: dict) -> dict:
    inp = job.get("input", {})

    lyrics   = inp.get("lyrics", "").strip()
    style    = inp.get("style", "cheerful kids nursery rhyme, female voice, ukulele, glockenspiel")
    duration = float(inp.get("duration", 90))

    if not lyrics:
        return {"error": "No lyrics provided"}

    try:
        import soundfile as sf
        import numpy as np

        pipe = _load_pipeline()

        log.info(f"Generating {duration}s song  |  style: {style[:60]}")
        t0 = time.time()

        result = pipe(
            task_type="text2music",
            caption=style,
            lyrics=lyrics,
            audio_duration=duration,
            infer_steps=60,        # quality/speed trade-off (20 = fast draft, 100 = max quality)
            guidance_scale=7.5,
            scheduler_type="euler",
        )

        # result.audios is a list of numpy arrays (float32, shape [samples])
        audio: np.ndarray = result.audios[0]
        sr:    int        = getattr(result, "sample_rate", _sample_rate)
        actual_dur        = len(audio) / sr

        log.info(f"Generated {actual_dur:.1f}s audio in {time.time() - t0:.1f}s (sr={sr})")

        # Encode as 16-bit PCM WAV → base64
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
        log.exception("Handler error")
        return {"error": str(exc)}


runpod.serverless.start({"handler": handler})
