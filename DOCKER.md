# SOKE Docker Guide

## 1) Prepare env file

```bash
cd /home/cirillo/Desktop/SIGNGEN/SOKE
scripts/init_docker_env.sh
# then edit .env and set HF_TOKEN
```

Alternative manual setup is still valid:

```bash
cp .env.example .env
# Edit .env and set HF_TOKEN + LOCAL_UID + LOCAL_GID
```

## 2) Build image

```bash
docker compose build
```

Compatibility note:
- image base uses `torch 2.3.1`,
- project pins `transformers==4.41.2` to avoid runtime import breaks with newer transformers.

Quick GPU check:

```bash
docker compose run --rm soke nvidia-smi
```

## 3) Train

```bash
docker compose run --rm soke train
```

Cycle mode (recommended for long runs with periodic test/inference):

```bash
docker compose run --rm soke cycle
```

## 4) Inference

```bash
docker compose run --rm soke infer
```

## 5) Shell inside container

```bash
docker compose run --rm soke shell
```

If you prefer explicit env file:

```bash
docker compose --env-file .env run --rm soke train
```

## Notes
- Dataset is external to code repo and mounted at `/workspace/SOKE_DATA`.
- Artifacts are external to code repo and mounted at `/workspace/SOKE_ARTIFACTS`:
  - training checkpoints/logs: `/workspace/SOKE_ARTIFACTS/experiments`
  - test outputs (`.pkl`, scores): `/workspace/SOKE_ARTIFACTS/results`
  - preview GIF: `/workspace/SOKE_ARTIFACTS/gifs`
  - run markers (`STARTED/DONE/FAILED`): `/workspace/SOKE_ARTIFACTS/run_state`
- If dataset files are missing and `SOKE_HF_DATASET_REPO` is set, container downloads from private HF repo.
- If HF repo provides `How2Sign.tar.gz`, `CSL-Daily.tar.gz`, `Phoenix_2014T.tar.gz`, bootstrap auto-extracts them after download.
- Split files are auto-repaired from `data/splits/` when missing or broken symlinks are detected.
- Credentials are passed only via `.env` / runtime env vars (never hardcoded in git).
- Telegram notifications are optional:
  - `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
  - `SOKE_TELEGRAM_NOTIFY=1`
  - `SOKE_TELEGRAM_HEARTBEAT_SEC=1800` (seconds, set `0` to disable heartbeat)
- In infer/test mode, container can auto-build a GT-vs-PRED GIF and send it to Telegram.
- GPU/device selection is env-driven (in `.env`):
  - `SOKE_USE_GPUS`, `SOKE_DEVICE_IDS`
  - `SOKE_TRAIN_USE_GPUS`, `SOKE_TRAIN_DEVICE_IDS`
  - `SOKE_TEST_USE_GPUS`, `SOKE_TEST_DEVICE_IDS`
  - `SOKE_NUM_NODES`
- Cycle scheduling (in `.env`):
  - `SOKE_TOTAL_EPOCHS`, `SOKE_CYCLE_EPOCHS`
  - `SOKE_CYCLE_RUN_INFER`
  - `SOKE_CYCLE_TEST_MAX_SAMPLES`, `SOKE_CYCLE_TEST_SKIP_METRICS`

## Publish image (for remote SSH server pull)
Recommended: GitHub Container Registry (GHCR), private image.

```bash
docker tag signgen/soke:local ghcr.io/francescocassini/soke:latest
echo <GH_TOKEN> | docker login ghcr.io -u francescocassini --password-stdin
docker push ghcr.io/francescocassini/soke:latest
```

On remote server:

```bash
echo <GH_TOKEN> | docker login ghcr.io -u francescocassini --password-stdin
docker pull ghcr.io/francescocassini/soke:latest
docker run --gpus all --rm -it \\
  --env SOKE_HF_DATASET_REPO=Francesco77/soke-private-data \\
  --env HF_TOKEN=<HF_TOKEN> \\
  -v /path/on/server/SOKE_DATA:/workspace/SOKE_DATA \\
  ghcr.io/francescocassini/soke:latest train
```
