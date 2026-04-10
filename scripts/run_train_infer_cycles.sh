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
SCHEDULE_EPOCHS="${SOKE_SCHEDULE_EPOCHS:-}"
SCHEDULE_VAL_EPOCHS="${SOKE_SCHEDULE_VAL_EPOCHS:-}"
SCHEDULE_TEST_EPOCHS="${SOKE_SCHEDULE_TEST_EPOCHS:-}"

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

parse_epoch_list() {
  local raw="$1"
  local -n out_ref="$2"
  out_ref=()
  [[ -z "$raw" ]] && return 0
  local item trimmed
  IFS=',' read -r -a _items <<<"$raw"
  for item in "${_items[@]}"; do
    trimmed="$(echo "$item" | xargs)"
    [[ -z "$trimmed" ]] && continue
    if ! [[ "$trimmed" =~ ^[0-9]+$ ]]; then
      echo "[ERROR] Invalid epoch value: '$trimmed'" >&2
      exit 1
    fi
    if [[ "$trimmed" -le 0 ]]; then
      echo "[ERROR] Epoch values must be > 0. Got: '$trimmed'" >&2
      exit 1
    fi
    out_ref+=("$trimmed")
  done
}

list_contains() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

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

declare -a targets=()
declare -a val_epochs=()
declare -a test_epochs=()

if [[ -n "$SCHEDULE_EPOCHS" || -n "$SCHEDULE_VAL_EPOCHS" || -n "$SCHEDULE_TEST_EPOCHS" ]]; then
  parse_epoch_list "$SCHEDULE_EPOCHS" targets
  parse_epoch_list "$SCHEDULE_VAL_EPOCHS" val_epochs
  parse_epoch_list "$SCHEDULE_TEST_EPOCHS" test_epochs

  # Shorthand mode: one schedule controls both validation and test.
  if [[ "${#targets[@]}" -gt 0 ]]; then
    if [[ "${#val_epochs[@]}" -eq 0 ]]; then
      val_epochs=("${targets[@]}")
    fi
    if [[ "${#test_epochs[@]}" -eq 0 ]]; then
      test_epochs=("${targets[@]}")
    fi
  fi

  # Build sorted unique target epochs from val/test schedules.
  targets=("${val_epochs[@]}" "${test_epochs[@]}")
  if [[ "${#targets[@]}" -eq 0 ]]; then
    echo "[ERROR] Schedule mode enabled but no epochs provided." >&2
    exit 1
  fi

  mapfile -t targets < <(printf '%s\n' "${targets[@]}" | sort -n -u)
  for target in "${targets[@]}"; do
    if [[ "$target" -gt "$TOTAL_EPOCHS" ]]; then
      echo "[ERROR] Scheduled epoch $target exceeds TOTAL_EPOCHS=$TOTAL_EPOCHS" >&2
      exit 1
    fi
  done
  echo "[INFO] schedule mode enabled"
  echo "[INFO] val_epochs=${val_epochs[*]:-none}"
  echo "[INFO] test_epochs=${test_epochs[*]:-none}"
else
  completed=0
  while [[ "$completed" -lt "$TOTAL_EPOCHS" ]]; do
    target=$((completed + CYCLE_EPOCHS))
    if [[ "$target" -gt "$TOTAL_EPOCHS" ]]; then
      target="$TOTAL_EPOCHS"
    fi
    targets+=("$target")
    completed="$target"
  done
  val_epochs=("${targets[@]}")
  test_epochs=("${targets[@]}")
fi

for target in "${targets[@]}"; do
  export SOKE_PERIODIC_INFER_EVERY_N_EPOCHS=0
  export SOKE_TRAIN_END_EPOCH="$target"
  if [[ -f "$LAST_CKPT" ]]; then
    export SOKE_TRAIN_RESUME="$EXP_DIR"
    echo "[INFO] Resuming training from: $LAST_CKPT (target_epoch=$target)"
  else
    unset SOKE_TRAIN_RESUME || true
    echo "[INFO] Starting fresh training (target_epoch=$target)"
  fi

  # In schedule mode, force validation exactly at selected target epochs.
  # We do this by generating a temporary cfg with VAL_EVERY_STEPS=target
  # only when target is in val schedule; otherwise set a huge value to skip.
  VAL_EVERY_OVERRIDE=$((TOTAL_EPOCHS + 100000))
  if list_contains "$target" "${val_epochs[@]}"; then
    VAL_EVERY_OVERRIDE="$target"
  fi

  TMP_CFG="/tmp/soke_cycle_cfg_${target}_$$.yaml"
  cp "$TRAIN_CFG" "$TMP_CFG"
  sed -i -E "s/^([[:space:]]*VAL_EVERY_STEPS:).*/\1 ${VAL_EVERY_OVERRIDE}/" "$TMP_CFG"

  train_cmd=(
    python -u -m train
    --cfg "$TMP_CFG"
    --nodebug
    --use_gpus "$TRAIN_USE_GPUS"
    --device "${TRAIN_DEVICE_ARR[@]}"
    --num_nodes "$NUM_NODES"
  )
  PYTHONPATH=. CUDA_VISIBLE_DEVICES="$TRAIN_USE_GPUS" "${train_cmd[@]}"
  rm -f "$TMP_CFG"

  if [[ ! -f "$LAST_CKPT" ]]; then
    echo "[ERROR] last.ckpt not found after training: $LAST_CKPT" >&2
    exit 1
  fi

  if list_contains "$target" "${test_epochs[@]}" && [[ "$RUN_INFER" == "1" || "$RUN_INFER" == "true" || "$RUN_INFER" == "yes" || "$RUN_INFER" == "on" ]]; then
    echo "[INFO] Running post-train inference on checkpoint: $LAST_CKPT (epoch=$target)"
    export SOKE_TEST_CFG="$TEST_CFG"
    export SOKE_TEST_USE_GPUS="$TEST_USE_GPUS"
    export SOKE_TEST_DEVICE_IDS="$TEST_DEVICE_IDS"
    export MAX_SAMPLES="$INFER_MAX_SAMPLES"
    export SKIP_METRICS="$INFER_SKIP_METRICS"
    bash scripts/run_inference_complete.sh "$LAST_CKPT"
  fi

  echo "[INFO] Milestone completed at epoch=$target"
done

echo "[INFO] All cycles completed."
