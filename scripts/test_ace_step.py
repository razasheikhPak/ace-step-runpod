"""
scripts/test_ace_step.py

Quick test for the ACE-Step RunPod endpoint.
Submits a short (15s) song and saves the output to data/test_ace_step.wav

Usage:
    python scripts/test_ace_step.py
"""
import base64
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import config

ENDPOINT_ID  = config.ACE_STEP_ENDPOINT_ID
API_KEY      = config.RUNPOD_API_KEY
RUN_URL      = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"
STATUS_URL   = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{{job_id}}"
OUT_PATH     = Path("data/test_ace_step.wav")

TEST_LYRICS = """[verse]
The wheels on the bus go round and round
Round and round, round and round
The wheels on the bus go round and round
All through the town"""

TEST_STYLE = "cheerful children's nursery rhyme, playful female voice, ukulele, slow and clear"

def main():
    import requests

    if not ENDPOINT_ID:
        print("ERROR: ACE_STEP_ENDPOINT_ID not set in .env")
        sys.exit(1)

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"input": {"lyrics": TEST_LYRICS, "style": TEST_STYLE, "duration": 20}}

    print(f"Submitting test job to endpoint {ENDPOINT_ID} ...")
    resp = requests.post(RUN_URL, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    job_id = resp.json().get("id")
    print(f"Job ID: {job_id}")

    print("Polling (this may take 3-5 min on cold start) ...")
    for attempt in range(60):
        time.sleep(8)
        poll = requests.get(STATUS_URL.format(job_id=job_id), headers=headers, timeout=15)
        data = poll.json()
        status = data.get("status", "").upper()
        elapsed = (attempt + 1) * 8
        print(f"  [{elapsed:>3}s] {status}")

        if status == "COMPLETED":
            output = data.get("output", {})
            if "error" in output:
                print(f"WORKER ERROR: {output['error']}")
                sys.exit(1)
            wav_bytes = base64.b64decode(output["audio_b64"])
            OUT_PATH.parent.mkdir(exist_ok=True)
            OUT_PATH.write_bytes(wav_bytes)
            dur = output.get("duration_s", "?")
            print(f"\nSUCCESS — {len(wav_bytes)//1024}KB, {dur}s")
            print(f"Saved to: {OUT_PATH}")
            return

        if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
            out = data.get("output") or {}
            error = data.get("error") or out.get("error", "")
            tb    = out.get("traceback", "")
            print(f"\nFAILED: {error}")
            if tb:
                print(f"\nTraceback:\n{tb}")
            sys.exit(1)

    print("\nTIMEOUT — job still running after 8 minutes")
    sys.exit(1)

if __name__ == "__main__":
    main()
