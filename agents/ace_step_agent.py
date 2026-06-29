"""
agents/ace_step_agent.py — ACE-Step 1.5 music generation via RunPod Serverless.

Drop-in replacement for SunoSongAgent. Generates melodious sung audio for any
nursery rhyme or song topic without copyright restrictions.

Cost: ~$0.02/song (RTX 4090 on RunPod, ~90s generation time)
vs.   $0.06/song  (sunoapi.org) + blocks famous songs.

Activate via .env:
    RUNPOD_API_KEY=your_runpod_key           (already set for WaveSpeed)
    ACE_STEP_ENDPOINT_ID=your_endpoint_id   (from RunPod dashboard)

Pipeline:
    lyrics formatted → RunPod serverless job → WAV returned as base64
    → written to disk → faster-whisper alignment → per-scene MP3 slices
    (alignment + slicing inherited from SunoSongAgent)
"""
from __future__ import annotations

import base64
import logging
import time
from pathlib import Path

from agents.suno_agent import SunoSongAgent

logger = logging.getLogger(__name__)

_RUNPOD_RUN_URL    = "https://api.runpod.ai/v2/{endpoint_id}/run"
_RUNPOD_STATUS_URL = "https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"

# Generation timeout — ACE-Step on RTX 4090 takes ~60-90s for a 90s song
_POLL_INTERVAL_S = 8
_TIMEOUT_S       = 300  # 5 minutes


class AceStepAgent(SunoSongAgent):
    """
    Generates sung audio using ACE-Step 1.5 deployed on RunPod Serverless.
    Inherits lyrics formatting and whisper-based slicing from SunoSongAgent;
    only the API call layer is replaced.
    """

    def __init__(self):
        from core.config import config
        # Reuse the same kids-song style description as Suno
        self._style       = config.SUNO_KIDS_STYLE
        self._api_key     = config.RUNPOD_API_KEY
        self._endpoint_id = config.ACE_STEP_ENDPOINT_ID
        self._persona     = ""   # ACE-Step doesn't use persona IDs

    @classmethod
    def is_available(cls) -> bool:
        """True when both RUNPOD_API_KEY and ACE_STEP_ENDPOINT_ID are configured."""
        from core.config import config
        return bool(
            getattr(config, "RUNPOD_API_KEY", "") and
            getattr(config, "ACE_STEP_ENDPOINT_ID", "")
        )

    # ── Override: submit to RunPod instead of sunoapi.org ────────────────────

    def _call_api(self, lyrics: str, title: str, output_path: Path) -> None:
        """
        Submit a RunPod serverless job, poll until COMPLETED, decode WAV to disk.
        Called by the inherited generate_for_script() / _generate_split() methods.
        """
        import requests

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type":  "application/json",
        }

        # Estimate duration from lyrics length (roughly 120 words/minute singing)
        word_count = len(lyrics.split())
        duration_s = max(30, min(180, int(word_count / 2.0)))

        payload = {
            "input": {
                "lyrics":         lyrics,
                "prompt":         self._style,   # ACE-Step uses "prompt" not "style"
                "audio_duration": duration_s,    # ACE-Step uses "audio_duration" not "duration"
            }
        }

        run_url = _RUNPOD_RUN_URL.format(endpoint_id=self._endpoint_id)
        logger.info(f"[AceStep] Submitting job ({duration_s}s song, {len(lyrics)} chars) → RunPod ...")

        resp = requests.post(run_url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        job_id = resp.json().get("id")
        if not job_id:
            raise RuntimeError(f"[AceStep] No job ID in response: {resp.text[:200]}")

        logger.info(f"[AceStep] Job {job_id} — polling every {_POLL_INTERVAL_S}s ...")

        # ── Poll ─────────────────────────────────────────────────────────────
        status_url = _RUNPOD_STATUS_URL.format(
            endpoint_id=self._endpoint_id, job_id=job_id
        )
        deadline = time.time() + _TIMEOUT_S
        attempt  = 0

        while time.time() < deadline:
            time.sleep(_POLL_INTERVAL_S)
            attempt += 1

            try:
                poll = requests.get(status_url, headers=headers, timeout=15)
                poll.raise_for_status()
                data = poll.json()
            except Exception as exc:
                logger.warning(f"[AceStep] Poll {attempt} error: {exc} — retrying")
                continue

            status = data.get("status", "").upper()
            logger.info(f"[AceStep] Poll {attempt} ({attempt * _POLL_INTERVAL_S}s): {status}")

            if status == "COMPLETED":
                output = data.get("output", {})
                if "error" in output:
                    raise RuntimeError(f"[AceStep] Worker error: {output['error']}")

                audio_b64 = output.get("audio_b64", "")
                if not audio_b64:
                    raise RuntimeError("[AceStep] COMPLETED but no audio_b64 in output")

                wav_bytes = base64.b64decode(audio_b64)
                output_path.write_bytes(wav_bytes)
                actual_dur = output.get("duration_s", "?")
                logger.info(
                    f"[AceStep] ✅ Audio ready: {len(wav_bytes)//1024}KB, "
                    f"{actual_dur}s — saved to {output_path.name}"
                )
                return

            if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                raise RuntimeError(f"[AceStep] RunPod job {status}: {data.get('error', '')}")

        raise TimeoutError(
            f"[AceStep] Timed out after {_TIMEOUT_S}s — job {job_id} still running"
        )
