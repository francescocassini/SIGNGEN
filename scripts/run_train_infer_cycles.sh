#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TRAIN_CFG="${SOKE_TRAIN_CFG:-configs/soke.yaml}"
TEST_CFG="${SOKE_TEST_CFG:-configs/soke_infer_complete.yaml}"

TRAIN_USE_GPUS="${SOKE_TRAIN_USE_GPUS:-${SOKE_USE_GPUS:-0}}"
TRAIN_DEVICE_IDS="${SOKE_TRAIN_DEVICE_IDS:-${SOKE_DEVICE_IDS:-0}}"
TEST_USE_GPUS="${SOKE_TEST_USE_GPUS:-${SOKE_USE_GPUS:-0}}"
TEST_DEVICE_IDS="${SOKE_TEST_DEVICE_IDS:-${SOKE_DEVICE_IDS:-0}}"
NUM_NODES="${SOKE_NUM_NODES:-1}"

TOTAL_EPOCHS="${SOKE_TOTAL_EPOCHS:-${SOKE_TRAIN_END_EPOCH:-150}}"
CYCLE_EPOCHS="${SOKE_CYCLE_EPOCHS:-50}"
RUN_INFER="${SOKE_CYCLE_RUN_INFER:-1}"
INFER_MAX_SAMPLES="${SOKE_CYCLE_TEST_MAX_SAMPLES:-}"
INFER_SKIP_METRICS="${SOKE_CYCLE_TEST_SKIP_METRICS:-0}"

EXP_ROOT="${SOKE_EXP_ROOT:-experiments}"
MODEL_NAME="${SOKE_MODEL_NAME:-mgpt}"
EXP_NAME="${SOKE_EXP_NAME:-SOKE}"
EXP_DIR="${EXP_ROOT}/${MODEL_NAME}/${EXP_NAME}"
LAST_CKPT="${EXP_DIR}/checkpoints/last.ckpt"

if [[ ! -f "$TRAIN_CFG" ]]; then
  echo "[ERROR] Missing train cfg: $TRAIN_CFG" >&2
  exit 1
fi
if [[ ! -f "$TEST_CFG" ]]; then
  echo "[ERROR] Missing test cfg: $TEST_CFG" >&2
  exit 1
fi

if ! [[ "$TOTAL_EPOCHS" =~ ^[0-9]+$ ]] || ! [[ "$CYCLE_EPOCHS" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] SOKE_TOTAL_EPOCHS and SOKE_CYCLE_EPOCHS must be integers." >&2
  exit 1
fi
if [[ "$TOTAL_EPOCHS" -le 0 || "$CYCLE_EPOCHS" -le 0 ]]; then
  echo "[ERROR] SOKE_TOTAL_EPOCHS and SOKE_CYCLE_EPOCHS must be > 0." >&2
  exit 1
fi

read -r -a TRAIN_DEVICE_ARR <<<"$(echo "$TRAIN_DEVICE_IDS" | tr ',' ' ')"
if [[ "${#TRAIN_DEVICE_ARR[@]}" -eq 0 ]]; then
  TRAIN_DEVICE_ARR=(0)
fi

echo "[INFO] Cycle runner started"
echo "[INFO] total_epochs=$TOTAL_EPOCHS cycle_epochs=$CYCLE_EPOCHS"
echo "[INFO] train_use_gpus=$TRAIN_USE_GPUS train_devices=${TRAIN_DEVICE_ARR[*]} num_nodes=$NUM_NODES"
echo "[INFO] test_use_gpus=$TEST_USE_GPUS test_devices=$TEST_DEVICE_IDS run_infer=$RUN_INFER"
echo "[INFO] exp_dir=$EXP_DIR"

completed=0
while [[ "$completed" -lt "$TOTAL_EPOCHS" ]]; do
  target=$((completed + CYCLE_EPOCHS))
  if [[ "$target" -gt "$TOTAL_EPOCHS" ]]; then
    target="$TOTAL_EPOCHS"
  fi

  export SOKE_PERIODIC_INFER_EVERY_N_EPOCHS=0
  export SOKE_TRAIN_END_EPOCH="$target"
  if [[ -f "$LAST_CKPT" ]]; then
    export SOKE_TRAIN_RESUME="$EXP_DIR"
    echo "[INFO] Resuming training from: $LAST_CKPT (target_epoch=$target)"
  else
    unset SOKE_TRAIN_RESUME || true
    echo "[INFO] Starting fresh training (target_epoch=$target)"
  fi

  train_cmd=(
    python -u -m train
    --cfg "$TRAIN_CFG"
    --nodebug
    --use_gpus "$TRAIN_USE_GPUS"
    --device "${TRAIN_DEVICE_ARR[@]}"
    --num_nodes "$NUM_NODES"
  )

  PYTHONPATH=. CUDA_VISIBLE_DEVICES="$TRAIN_USE_GPUS" "${train_cmd[@]}"

  if [[ ! -f "$LAST_CKPT" ]]; then
    echo "[ERROR] last.ckpt not found after training: $LAST_CKPT" >&2
    exit 1
  fi

  if [[ "$RUN_INFER" == "1" || "$RUN_INFER" == "true" || "$RUN_INFER" == "yes" || "$RUN_INFER" == "on" ]]; then
    echo "[INFO] Running post-train inference on checkpoint: $LAST_CKPT"
    export SOKE_TEST_CFG="$TEST_CFG"
    export SOKE_TEST_USE_GPUS="$TEST_USE_GPUS"
    export SOKE_TEST_DEVICE_IDS="$TEST_DEVICE_IDS"
    export MAX_SAMPLES="$INFER_MAX_SAMPLES"
    export SKIP_METRICS="$INFER_SKIP_METRICS"
    bash scripts/run_inference_complete.sh "$LAST_CKPT"
  fi

  completed="$target"
  echo "[INFO] Cycle completed up to epoch=$completed"
done

echo "[INFO] All cycles completed."
