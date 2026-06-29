# ACE-Step 1.0 RunPod Serverless Worker
# Version stack confirmed from ace-step/ACE-Step official requirements:
#   Python 3.10, CUDA 12.6, PyTorch (cu126), diffusers>=0.36.0

FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    git ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Ensure python3.10 is the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    python3 -m pip install --upgrade pip

WORKDIR /app

# PyTorch cu126 — the officially tested stack for ACE-Step
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

# Install ACE-Step from GitHub (PyPI 0.1.0 sdist is broken — missing requirements.txt)
# Install other deps first so ace-step doesn't override pinned versions
RUN pip3 install --no-cache-dir \
    "diffusers>=0.36.0" \
    "transformers==4.50.0" \
    "accelerate==1.6.0" \
    "soundfile==0.13.1" \
    "librosa" \
    "huggingface_hub" \
    "runpod==1.7.9"

RUN pip3 install --no-cache-dir "git+https://github.com/ace-step/ACE-Step.git"

COPY ace_step_handler.py /app/handler.py
COPY download_model.py /app/download_model.py

CMD ["python3", "-u", "/app/handler.py"]
