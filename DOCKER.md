# SOKE Docker Guide

## 1) Runtime env file (not baked into image)

Create runtime env once:

```bash
cd /home/cirillo/Desktop/SIGNGEN/SOKE
scripts/init_docker_env.sh
# writes .env.runtime
# then edit HF_TOKEN + WANDB_API_KEY
```

Or manual setup:

```bash
cp .env.runtime.example .env.runtime
# edit values
```

Important:
- `.env` / `.env.*` are ignored by `.dockerignore` and are never copied into image layers.
- Changing `.env.runtime` does not require rebuilding the image.

## 2) Build full image (with deps + SMPL)

This Dockerfile now includes `deps/*` inside the image (for train/infer/validation on remote).

```bash
docker compose build --no-cache soke
```

Compatibility notes:
- base image: `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime`
- project pin: `transformers==4.41.2`

Quick GPU check:

```bash
docker compose run --rm soke nvidia-smi
```

## 3) Run locally with compose

```bash
docker compose run --rm soke train
docker compose run --rm soke infer
docker compose run --rm soke cycle
docker compose run --rm soke shell
```

Compose now uses `.env.runtime` and does not bind-mount source code into `/workspace/SOKE`.
This ensures the embedded code + embedded `deps` from the image are used.

## 4) Run directly from pulled image (recommended on remote)

Helper script:

```bash
scripts/run_docker_image.sh train
scripts/run_docker_image.sh infer
scripts/run_docker_image.sh cycle
```

Defaults used by script:
- image: `ghcr.io/francescocassini/soke:full-deps`
- runtime env file: `.env.runtime`
- mounts:
  - `${SOKE_DATA_ROOT_HOST}` -> `/workspace/SOKE_DATA`
  - `${SOKE_ARTIFACTS_ROOT_HOST}` -> `/workspace/SOKE_ARTIFACTS`
  - `${HOME}/.cache/huggingface` -> `/workspace/.cache/huggingface`

## 5) Publish full image to GHCR

```bash
docker tag signgen/soke:local ghcr.io/francescocassini/soke:full-deps
echo <GH_TOKEN> | docker login ghcr.io -u francescocassini --password-stdin
docker push ghcr.io/francescocassini/soke:full-deps
```

Suggested extra tag:

```bash
docker tag ghcr.io/francescocassini/soke:full-deps ghcr.io/francescocassini/soke:latest
docker push ghcr.io/francescocassini/soke:latest
```

## 6) Pull + run on multi-GPU machine

```bash
echo <GH_TOKEN> | docker login ghcr.io -u francescocassini --password-stdin
docker pull ghcr.io/francescocassini/soke:full-deps

# Option A: helper script (inside repo checkout)
SOKE_DOCKER_IMAGE=ghcr.io/francescocassini/soke:full-deps scripts/run_docker_image.sh train

# Option B: plain docker run
docker run --gpus all --rm -it \
  --env-file /path/to/.env.runtime \
  -v /path/on/server/SOKE_DATA:/workspace/SOKE_DATA \
  -v /path/on/server/SOKE_ARTIFACTS:/workspace/SOKE_ARTIFACTS \
  -v $HOME/.cache/huggingface:/workspace/.cache/huggingface \
  ghcr.io/francescocassini/soke:full-deps train
```

## Notes
- Dataset remains external (`/workspace/SOKE_DATA`) unless you choose to pre-load it in host storage.
- If dataset files are missing and `SOKE_HF_DATASET_REPO` is set, container can auto-download from private HF repo.
- Artifacts are external in `/workspace/SOKE_ARTIFACTS` (checkpoints/results/logs/gifs/run_state).
- Telegram notifications are optional via runtime env vars.
- GPU/device selection is runtime-env driven (`SOKE_*GPU*`, `SOKE_DEVICE_IDS`, `SOKE_NUM_NODES`).
