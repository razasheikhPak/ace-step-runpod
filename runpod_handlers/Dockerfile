# ACE-Step 1.5 RunPod Serverless Worker
# PyTorch 2.4 + CUDA 12.4 devel image — flash-attn compiled from source at build time

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Core Python deps
RUN pip install --no-cache-dir \
    "runpod==1.7.9" \
    "soundfile" \
    "librosa" \
    "huggingface_hub"

# ACE-Step — try PyPI first, fall back to GitHub
RUN pip install --no-cache-dir "ace-step" 2>/dev/null || \
    pip install --no-cache-dir "acestep" 2>/dev/null || \
    pip install --no-cache-dir "git+https://github.com/ace-step/ACE-Step.git"

# Compile flash-attn from source matching this image's exact CUDA + PyTorch versions.
# --no-build-isolation uses the system PyTorch/CUDA headers already in this image.
# This takes ~20 min but only runs once at Docker build time (not at inference time).
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# Copy handler and model downloader
COPY ace_step_handler.py /app/handler.py
COPY download_model.py /app/download_model.py

CMD ["python", "-u", "/app/handler.py"]
