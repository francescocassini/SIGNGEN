#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${SOKE_RUNTIME_ENV_FILE:-$ROOT_DIR/.env.runtime}"

REPO_ID="${1:-Francesco77/soke-private-data}"
DATA_ROOT="${2:-/home/cirillo/Desktop/SOKE_DATA}"
AUTO_DL="${3:-1}"
ARTIFACT_ROOT="${4:-/home/cirillo/Desktop/SOKE_ARTIFACTS}"
FORCE_FLAG="${5:-}"

if [[ "$ARTIFACT_ROOT" == "--force" ]]; then
  ARTIFACT_ROOT="/home/cirillo/Desktop/SOKE_ARTIFACTS"
  FORCE_FLAG="--force"
fi

if [[ -f "$ENV_FILE" && "$FORCE_FLAG" != "--force" ]]; then
  echo "[ERROR] $ENV_FILE already exists."
  echo "Use --force as 5th argument to overwrite."
  echo "Example: $0 $REPO_ID $DATA_ROOT $AUTO_DL $ARTIFACT_ROOT --force"
  exit 1
fi

UID_VAL="$(id -u)"
GID_VAL="$(id -g)"

cat > "$ENV_FILE" <<EOF
SOKE_HF_DATASET_REPO=$REPO_ID
HF_TOKEN=hf_xxx_replace_me
WANDB_API_KEY=wandb_xxx_replace_me
SOKE_DATA_ROOT_HOST=$DATA_ROOT
SOKE_ARTIFACTS_ROOT_HOST=$ARTIFACT_ROOT
SOKE_AUTO_DOWNLOAD_DATASET=$AUTO_DL
SOKE_MODE=train
SOKE_TRAIN_CFG=configs/soke.yaml
SOKE_TEST_CFG=configs/soke_infer_complete.yaml
SOKE_USE_GPUS=0
SOKE_DEVICE_IDS=0
SOKE_TRAIN_USE_GPUS=0
SOKE_TRAIN_DEVICE_IDS=0
SOKE_TEST_USE_GPUS=0
SOKE_TEST_DEVICE_IDS=0
SOKE_NUM_NODES=1
SOKE_TRAIN_END_EPOCH=150
SOKE_VAL_EVERY_EPOCHS=4
SOKE_TRAIN_BATCH_SIZE=32
SOKE_TEST_BATCH_SIZE=8
SOKE_TEST_MAX_SAMPLES=
SOKE_TEST_SKIP_METRICS=0
SOKE_DEFAULT_TEST_CKPT=/workspace/SOKE_ARTIFACTS/experiments/mgpt/SOKE/checkpoints/last.ckpt
SOKE_TOTAL_EPOCHS=150
SOKE_CYCLE_EPOCHS=50
SOKE_CYCLE_RUN_INFER=1
SOKE_CYCLE_TEST_MAX_SAMPLES=32
SOKE_CYCLE_TEST_SKIP_METRICS=0
SOKE_PERIODIC_INFER_EVERY_N_EPOCHS=0
SOKE_PERIODIC_INFER_MAX_SAMPLES=32
SOKE_PERIODIC_INFER_SKIP_METRICS=1
SOKE_PERIODIC_INFER_KEEP_CKPT=0
SOKE_TELEGRAM_NOTIFY=1
SOKE_TELEGRAM_HEARTBEAT_SEC=1800
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
LOCAL_UID=$UID_VAL
LOCAL_GID=$GID_VAL
EOF

chmod 600 "$ENV_FILE"

echo "[OK] Wrote $ENV_FILE"
echo "     LOCAL_UID=$UID_VAL LOCAL_GID=$GID_VAL"
echo "Next steps:"
echo "  1) edit HF_TOKEN and WANDB_API_KEY in $ENV_FILE"
echo "  2) optionally set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID"
echo "  3) optionally tune SOKE_USE_GPUS/SOKE_DEVICE_IDS and cycle vars"
echo "  4) docker compose build --no-cache soke"
echo "  5) docker compose run --rm soke train   # single run"
echo "     or: docker compose run --rm soke cycle # train->infer->resume"
echo "Note: changing $ENV_FILE does not require rebuilding the image."
