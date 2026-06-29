"""
runpod_handlers/ace_step_handler.py

RunPod Serverless handler for ACE-Step music generation.
API confirmed from ace-step/ACE-Step official source (ace-step.github.io).

Input:  { "lyrics": "...", "prompt": "...", "audio_duration": 90 }
Output: { "audio_b64": "...", "duration_s": 90 }
        OR { "error": "...", "traceback": "..." }
"""
from __future__ import annotations

import base64
import logging
import os
import tempfile
import time
import traceback

import runpod

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ace_step_handler")

_pipeline = None

def _load_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    import torch
    log.info(f"PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()}")

    from acestep.pipeline_ace_step import ACEStepPipeline

    # checkpoint_dir=None → auto-downloads to ~/.cache/ace-step
    # Override via MODEL_PATH env var to point at a network volume cache
    model_path = os.environ.get("MODEL_PATH", None)
    log.info(f"Loading ACE-Step | checkpoint_dir={model_path!r}")
    t0 = time.time()

    _pipeline = ACEStepPipeline(
        checkpoint_dir=model_path,
        device_id=0,
        dtype="bfloat16",
    )

    log.info(f"Pipeline ready in {time.time() - t0:.1f}s")
    return _pipeline


def handler(job: dict) -> dict:
    inp = job.get("input", {})

    lyrics        = inp.get("lyrics", "").strip()
    prompt        = inp.get("prompt", inp.get("style",
                    "cheerful children's nursery rhyme, playful female voice, ukulele, slow and clear"))
    audio_duration = float(inp.get("audio_duration", inp.get("duration", 90)))

    if not lyrics:
        return {"error": "No lyrics provided"}

    try:
        pipe = _load_pipeline()
        log.info(f"Generating {audio_duration}s | prompt: {prompt[:80]}")
        t0 = time.time()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = pipe(
                task="text2music",
                format="wav",
                audio_duration=audio_duration,
                prompt=prompt,
                lyrics=lyrics,
                infer_step=60,
                guidance_scale=15.0,
                scheduler_type="euler",
                cfg_type="apg",
                omega_scale=10.0,
                save_path=tmpdir,
                batch_size=1,
            )

            # result is a list: [audio_path_1, ..., metadata_dict]
            audio_path = result[0]
            metadata   = result[-1] if isinstance(result[-1], dict) else {}

            log.info(f"Generated in {time.time() - t0:.1f}s | file: {audio_path}")

            with open(audio_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")

        return {
            "audio_b64":  audio_b64,
            "duration_s": audio_duration,
            "metadata":   metadata,
        }

    except Exception as exc:
        tb = traceback.format_exc()
        log.error(f"Handler error:\n{tb}")
        return {"error": str(exc), "traceback": tb}


runpod.serverless.start({"handler": handler})
