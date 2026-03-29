FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace/SOKE

WORKDIR /workspace/SOKE

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    grep -v '^bpy==' /tmp/requirements.txt > /tmp/requirements.docker.txt && \
    python -m pip install -r /tmp/requirements.docker.txt && \
    python -m pip install huggingface_hub hf-transfer

COPY . /workspace/SOKE
RUN chmod +x /workspace/SOKE/docker/entrypoint.sh /workspace/SOKE/scripts/download_dataset_from_hf.sh || true

ENTRYPOINT ["/workspace/SOKE/docker/entrypoint.sh"]
CMD ["train"]
