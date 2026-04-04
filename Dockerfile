FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace/SOKE

WORKDIR /workspace/SOKE

LABEL org.opencontainers.image.title="SOKE" \
      org.opencontainers.image.description="SOKE training/inference container with bundled model deps" \
      org.opencontainers.image.licenses="Custom"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
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
RUN grep -v -E '^(bpy==|chumpy$)' /tmp/requirements.txt > /tmp/requirements.docker.txt && \
    python -m pip install -r /tmp/requirements.docker.txt && \
    python -m pip install --no-build-isolation chumpy && \
    python -m pip install huggingface_hub hf-transfer

COPY configs /workspace/SOKE/configs
COPY data /workspace/SOKE/data
COPY docker /workspace/SOKE/docker
COPY mGPT /workspace/SOKE/mGPT
COPY prepare /workspace/SOKE/prepare
COPY scripts /workspace/SOKE/scripts
COPY get_motion_code.py /workspace/SOKE/get_motion_code.py
COPY test.py /workspace/SOKE/test.py
COPY train.py /workspace/SOKE/train.py
COPY vis_blender.py /workspace/SOKE/vis_blender.py
COPY vis_mesh.py /workspace/SOKE/vis_mesh.py
COPY license.txt /workspace/SOKE/license.txt

# Keep deps in separate layers so each layer stays below GHCR's per-layer limit.
COPY deps/flan-t5-base /workspace/SOKE/deps/flan-t5-base
COPY deps/t2m /workspace/SOKE/deps/t2m
COPY deps/tokenizer_ckpt /workspace/SOKE/deps/tokenizer_ckpt
COPY deps/mbart-h2s-csl-phoenix /workspace/SOKE/deps/mbart-h2s-csl-phoenix
COPY deps/smpl_models /workspace/SOKE/deps/smpl_models
COPY deps/stats /workspace/SOKE/deps/stats
COPY deps/smpl_models.zip /workspace/SOKE/deps/smpl_models.zip

RUN chmod +x /workspace/SOKE/docker/entrypoint.sh /workspace/SOKE/scripts/download_dataset_from_hf.sh || true

ENTRYPOINT ["/workspace/SOKE/docker/entrypoint.sh"]
CMD ["train"]
