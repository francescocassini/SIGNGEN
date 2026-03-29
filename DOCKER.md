# SOKE Docker Guide

## 1) Prepare env file

```bash
cd /home/cirillo/Desktop/SIGNGEN/SOKE
cp .env.example .env
# Edit .env and set HF_TOKEN
```

## 2) Build image

```bash
docker compose build
```

Quick GPU check:

```bash
docker compose run --rm soke nvidia-smi
```

## 3) Train

```bash
docker compose run --rm soke train
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
- If dataset files are missing and `SOKE_HF_DATASET_REPO` is set, container downloads from private HF repo.
- Credentials are passed only via `.env` / runtime env vars (never hardcoded in git).

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
