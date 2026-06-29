# ACE-Step 1.5 RunPod Serverless Worker
# Uses RunPod's pre-cached PyTorch runtime image (~5GB vs 15GB for devel)
# Pre-cached on RunPod workers = fast cold starts

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# System deps (minimal)
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

# Copy handler and model downloader
COPY ace_step_handler.py /app/handler.py
COPY download_model.py /app/download_model.py

CMD ["python", "-u", "/app/handler.py"]
