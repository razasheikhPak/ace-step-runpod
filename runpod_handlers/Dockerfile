# ACE-Step 1.5 RunPod Serverless Worker
# Build:  docker build -t your_dockerhub/ace-step-handler:latest -f Dockerfile .
# Push:   docker push your_dockerhub/ace-step-handler:latest

FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Python deps — pin runpod SDK + audio libs
RUN pip install --no-cache-dir \
    runpod==1.7.9 \
    soundfile \
    librosa \
    numpy \
    huggingface_hub

# ACE-Step 1.5 — install from PyPI (preferred) or GitHub
RUN pip install --no-cache-dir acestep || \
    pip install --no-cache-dir "git+https://github.com/ace-step/ACE-Step.git"

# Copy handler
COPY ace_step_handler.py /app/handler.py

# Model download script — run this ONCE against your network volume:
#   docker run --rm -v /your/volume:/runpod-volume \
#     -e HF_TOKEN=your_token \
#     your_dockerhub/ace-step-handler:latest python /app/download_model.py
COPY download_model.py /app/download_model.py

CMD ["python", "-u", "/app/handler.py"]
