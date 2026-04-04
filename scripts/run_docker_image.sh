#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${1:-train}"
if [[ $# -gt 0 ]]; then
  shift
fi

IMAGE="${SOKE_DOCKER_IMAGE:-ghcr.io/francescocassini/soke:full-deps}"
ENV_FILE="${SOKE_RUNTIME_ENV_FILE:-$ROOT_DIR/.env.runtime}"
GPU_SPEC="${SOKE_GPU_SPEC:-all}"

DATA_ROOT_HOST="${SOKE_DATA_ROOT_HOST:-/home/cirillo/Desktop/SOKE_DATA}"
ARTIFACTS_ROOT_HOST="${SOKE_ARTIFACTS_ROOT_HOST:-/home/cirillo/Desktop/SOKE_ARTIFACTS}"
CACHE_ROOT_HOST="${SOKE_HF_CACHE_ROOT_HOST:-$HOME/.cache/huggingface}"

UID_VAL="${LOCAL_UID:-$(id -u)}"
GID_VAL="${LOCAL_GID:-$(id -g)}"

ENV_ARGS=()
if [[ -f "$ENV_FILE" ]]; then
  ENV_ARGS+=(--env-file "$ENV_FILE")
else
  echo "[WARN] env file not found: $ENV_FILE"
  echo "       continuing without --env-file"
fi

mkdir -p "$DATA_ROOT_HOST" "$ARTIFACTS_ROOT_HOST" "$CACHE_ROOT_HOST"

exec docker run --rm -it \
  --gpus "$GPU_SPEC" \
  --user "$UID_VAL:$GID_VAL" \
  "${ENV_ARGS[@]}" \
  -e SOKE_DATA_ROOT=/workspace/SOKE_DATA \
  -e SOKE_ARTIFACT_ROOT=/workspace/SOKE_ARTIFACTS \
  -e SOKE_EXP_ROOT=/workspace/SOKE_ARTIFACTS/experiments \
  -e SOKE_RESULTS_ROOT=/workspace/SOKE_ARTIFACTS/results \
  -e SOKE_LOG_ROOT=/workspace/SOKE_ARTIFACTS/logs \
  -e SOKE_GIF_ROOT=/workspace/SOKE_ARTIFACTS/gifs \
  -e SOKE_RUN_STATE_ROOT=/workspace/SOKE_ARTIFACTS/run_state \
  -e PYTHONPATH=/workspace/SOKE \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -e HOME=/workspace/SOKE \
  -e TMPDIR=/tmp \
  -e XDG_CONFIG_HOME=/tmp/.config \
  -e MPLCONFIGDIR=/tmp/matplotlib \
  -e HF_HOME=/workspace/.cache/huggingface \
  -v "$DATA_ROOT_HOST:/workspace/SOKE_DATA" \
  -v "$ARTIFACTS_ROOT_HOST:/workspace/SOKE_ARTIFACTS" \
  -v "$CACHE_ROOT_HOST:/workspace/.cache/huggingface" \
  "$IMAGE" "$MODE" "$@"
