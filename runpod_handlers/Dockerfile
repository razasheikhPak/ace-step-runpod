# ACE-Step RunPod Serverless Worker
# Uses RunPod's pre-cached PyTorch 2.4 + CUDA 12.4 base (avoids CUDA driver mismatch)
# diffusers>=0.36.0 fixes the PEP 604 annotation infer_schema crash on PyTorch 2.4

FROM runpod/pytorch:2.4.0-py3.10-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

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
