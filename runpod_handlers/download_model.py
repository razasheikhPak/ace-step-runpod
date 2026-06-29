"""
runpod_handlers/download_model.py

One-time script to download the ACE-Step 1.5 model weights onto a RunPod
network volume.  Run this before deploying the serverless endpoint:

  docker run --rm \
    -v /your/network-volume:/runpod-volume \
    -e HF_TOKEN=your_hf_token \
    your_dockerhub/ace-step-handler:latest \
    python /app/download_model.py

The model (~7GB in bfloat16) will be saved to /runpod-volume/ace-step-1.5.
Set MODEL_PATH=/runpod-volume/ace-step-1.5 as an env var on the endpoint.
"""
import os
from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID   = "ACE-Step/ACE-Step-v1.5-3.5B"
OUT_DIR   = Path(os.environ.get("MODEL_PATH", "/runpod-volume/ace-step-1.5"))
HF_TOKEN  = os.environ.get("HF_TOKEN", "")

print(f"Downloading {REPO_ID} → {OUT_DIR}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=REPO_ID,
    local_dir=str(OUT_DIR),
    token=HF_TOKEN or None,
    ignore_patterns=["*.msgpack", "flax_model*"],  # skip JAX weights
)

print(f"Done. Model saved to {OUT_DIR}  ({sum(f.stat().st_size for f in OUT_DIR.rglob('*') if f.is_file()) / 1e9:.1f} GB)")
