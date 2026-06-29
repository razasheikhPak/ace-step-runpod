# ACE-Step 1.5 RunPod Serverless Worker

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "runpod==1.7.9" \
    "soundfile" \
    "librosa" \
    "huggingface_hub"

# Install ACE-Step, then remove flash-attn — its version has a PyTorch 2.4
# schema-inference incompatibility. PyTorch 2.4 ships excellent native SDPA
# which ACE-Step/diffusers falls back to automatically when flash-attn is absent.
RUN pip install --no-cache-dir "ace-step" 2>/dev/null || \
    pip install --no-cache-dir "acestep" 2>/dev/null || \
    pip install --no-cache-dir "git+https://github.com/ace-step/ACE-Step.git"
RUN pip uninstall -y flash-attn 2>/dev/null; true

COPY ace_step_handler.py /app/handler.py
COPY download_model.py /app/download_model.py

CMD ["python", "-u", "/app/handler.py"]
