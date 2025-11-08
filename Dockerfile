# --------------------------
# Base image: RunPod PyTorch
# --------------------------
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# --------------------------
# Metadata
# --------------------------
LABEL maintainer="Your Name <you@example.com>"
LABEL description="WAN I2V RP Handler - RunPod Serverless"

# --------------------------
# Environment variables
# --------------------------
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

# --------------------------
# System dependencies
# --------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    unzip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# --------------------------
# Create working directory
# --------------------------
WORKDIR /workspace

# --------------------------
# Clone repository
# --------------------------
RUN git clone https://github.com/ModelTC/Wan2.2-Lightning.git

WORKDIR /workspace/Wan2.2-Lightning

# --------------------------
# Python dependencies
# --------------------------
RUN pip install runpod librosa decord hf_transfer
RUN pip install flash_attn --no-build-isolation
RUN pip install -r requirements.txt

# --------------------------
# Download Hugging Face models
# --------------------------
RUN huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir ./Wan2.2-I2V-A14B
RUN huggingface-cli download lightx2v/Wan2.2-Lightning --local-dir ./Wan2.2-Lightning

# --------------------------
# Clean up Lightning folder
# Keep only Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1
# --------------------------
RUN find ./Wan2.2-Lightning -mindepth 1 -maxdepth 1 -type d \
    ! -name 'Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1' \
    -exec rm -rf {} +

# --------------------------
# GPU environment (optional)
# --------------------------
ENV CUDA_VISIBLE_DEVICES=0

# --------------------------
# Serverless entrypoint
# --------------------------
CMD ["python", "rp_handler.py"]
