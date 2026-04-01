#!/usr/bin/env bash
set -euo pipefail

cd /workspace/SOKE
umask "${SOKE_UMASK:-0002}"
echo "[docker-entrypoint] running as uid=$(id -u) gid=$(id -g) umask=$(umask)"

mkdir -p "${TMPDIR:-/tmp}" "${XDG_CONFIG_HOME:-/tmp/.config}" "${MPLCONFIGDIR:-/tmp/matplotlib}" "${HF_HOME:-/workspace/.cache/huggingface}" || true

notify_text() {
  /workspace/SOKE/scripts/telegram_notify.sh text "$*" || true
}

notify_gif() {
  local gif_path="$1"
  local caption="${2:-SOKE inference preview}"
  /workspace/SOKE/scripts/telegram_notify.sh gif "$gif_path" "$caption" || true
}

start_heartbeat() {
  local interval="${SOKE_TELEGRAM_HEARTBEAT_SEC:-0}"
  if [[ -z "${TELEGRAM_BOT_TOKEN:-}" || -z "${TELEGRAM_CHAT_ID:-}" ]]; then
    return 0
  fi
  if [[ ! "$interval" =~ ^[0-9]+$ ]] || [[ "$interval" -le 0 ]]; then
    return 0
  fi

  (
    while true; do
      sleep "$interval"
      now=""
      now="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
      notify_text "[SOKE][${RUN_ID}] heartbeat mode=${MODE} ts=${now} host=$(hostname)"
    done
  ) &
  HEARTBEAT_PID="$!"
}

stop_heartbeat() {
  if [[ -n "${HEARTBEAT_PID:-}" ]]; then
    kill "${HEARTBEAT_PID}" >/dev/null 2>&1 || true
  fi
}

export SOKE_DATA_ROOT="${SOKE_DATA_ROOT:-/workspace/SOKE_DATA}"
export SOKE_H2S_ROOT="${SOKE_H2S_ROOT:-$SOKE_DATA_ROOT/How2Sign}"
export SOKE_CSL_ROOT="${SOKE_CSL_ROOT:-$SOKE_DATA_ROOT/CSL-Daily}"
export SOKE_PHOENIX_ROOT="${SOKE_PHOENIX_ROOT:-$SOKE_DATA_ROOT/Phoenix_2014T}"
export SOKE_CSL_MEAN_PATH="${SOKE_CSL_MEAN_PATH:-$SOKE_CSL_ROOT/mean.pt}"
export SOKE_CSL_STD_PATH="${SOKE_CSL_STD_PATH:-$SOKE_CSL_ROOT/std.pt}"
export SOKE_ARTIFACT_ROOT="${SOKE_ARTIFACT_ROOT:-/workspace/SOKE_ARTIFACTS}"
export SOKE_EXP_ROOT="${SOKE_EXP_ROOT:-$SOKE_ARTIFACT_ROOT/experiments}"
export SOKE_RESULTS_ROOT="${SOKE_RESULTS_ROOT:-$SOKE_ARTIFACT_ROOT/results}"
export SOKE_LOG_ROOT="${SOKE_LOG_ROOT:-$SOKE_ARTIFACT_ROOT/logs}"
export SOKE_GIF_ROOT="${SOKE_GIF_ROOT:-$SOKE_ARTIFACT_ROOT/gifs}"
export SOKE_RUN_STATE_ROOT="${SOKE_RUN_STATE_ROOT:-$SOKE_ARTIFACT_ROOT/run_state}"
mkdir -p "$SOKE_EXP_ROOT" "$SOKE_RESULTS_ROOT" "$SOKE_LOG_ROOT" "$SOKE_GIF_ROOT" "$SOKE_RUN_STATE_ROOT"

AUTO_DOWNLOAD="${SOKE_AUTO_DOWNLOAD_DATASET:-1}"
AUTO_DOWNLOAD_LC="$(echo "$AUTO_DOWNLOAD" | tr '[:upper:]' '[:lower:]')"
REPO_ID="${SOKE_HF_DATASET_REPO:-}"
REPO_ID_TRIMMED="$(echo "$REPO_ID" | xargs || true)"

is_true=0
case "$AUTO_DOWNLOAD_LC" in
  1|true|yes|on) is_true=1 ;;
esac

if [[ "$is_true" == "1" ]] && [[ -n "$REPO_ID_TRIMMED" ]]; then
  echo "[docker-entrypoint] dataset sync from HF repo: ${REPO_ID_TRIMMED}"
  /workspace/SOKE/scripts/download_dataset_from_hf.sh "${REPO_ID_TRIMMED}" "${SOKE_DATA_ROOT}"
else
  if [[ "$is_true" != "1" ]]; then
    echo "[docker-entrypoint] dataset auto-download skipped (SOKE_AUTO_DOWNLOAD_DATASET=${AUTO_DOWNLOAD})"
  else
    echo "[docker-entrypoint] dataset auto-download skipped (SOKE_HF_DATASET_REPO not set)"
  fi
fi

MODE="${1:-train}"
shift || true
RUN_TS="$(date +"%Y%m%d_%H%M%S")"
RUN_ID="${SOKE_RUN_ID:-${MODE}_${RUN_TS}}"
STATE_DIR="${SOKE_RUN_STATE_ROOT}/${RUN_ID}"
mkdir -p "$STATE_DIR"
RUN_MANIFEST="${STATE_DIR}/manifest.txt"
RUN_STARTED_FILE="${STATE_DIR}/STARTED.txt"
RUN_DONE_FILE="${STATE_DIR}/DONE.txt"
RUN_FAILED_FILE="${STATE_DIR}/FAILED.txt"
touch "$RUN_STARTED_FILE"
{
  echo "run_id=$RUN_ID"
  echo "mode=$MODE"
  echo "started_at_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "hostname=$(hostname)"
  echo "exp_root=$SOKE_EXP_ROOT"
  echo "results_root=$SOKE_RESULTS_ROOT"
  echo "gif_root=$SOKE_GIF_ROOT"
  echo "data_root=$SOKE_DATA_ROOT"
  echo "repo_commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
} >"$RUN_MANIFEST"

case "$MODE" in
  train)
    TRAIN_USE_GPUS="${SOKE_TRAIN_USE_GPUS:-${SOKE_USE_GPUS:-0}}"
    TRAIN_DEVICE_IDS="${SOKE_TRAIN_DEVICE_IDS:-${SOKE_DEVICE_IDS:-0}}"
    TRAIN_NUM_NODES="${SOKE_NUM_NODES:-1}"
    read -r -a TRAIN_DEVICE_ARR <<<"$(echo "$TRAIN_DEVICE_IDS" | tr ',' ' ')"
    if [[ "${#TRAIN_DEVICE_ARR[@]}" -eq 0 ]]; then
      TRAIN_DEVICE_ARR=(0)
    fi
    CMD=(
      python -u -m train
      --cfg "${SOKE_TRAIN_CFG:-configs/soke.yaml}"
      --nodebug
      --use_gpus "$TRAIN_USE_GPUS"
      --device "${TRAIN_DEVICE_ARR[@]}"
      --num_nodes "$TRAIN_NUM_NODES"
      "$@"
    )
    export CUDA_VISIBLE_DEVICES="$TRAIN_USE_GPUS"
    ;;
  infer|test)
    TEST_USE_GPUS="${SOKE_TEST_USE_GPUS:-${SOKE_USE_GPUS:-0}}"
    TEST_DEVICE_IDS="${SOKE_TEST_DEVICE_IDS:-${SOKE_DEVICE_IDS:-0}}"
    TEST_NUM_NODES="${SOKE_NUM_NODES:-1}"
    read -r -a TEST_DEVICE_ARR <<<"$(echo "$TEST_DEVICE_IDS" | tr ',' ' ')"
    if [[ "${#TEST_DEVICE_ARR[@]}" -eq 0 ]]; then
      TEST_DEVICE_ARR=(0)
    fi
    CMD=(
      python -u -m test
      --cfg "${SOKE_TEST_CFG:-configs/soke_infer_complete.yaml}"
      --task t2m
      --nodebug
      --use_gpus "$TEST_USE_GPUS"
      --device "${TEST_DEVICE_ARR[@]}"
      --num_nodes "$TEST_NUM_NODES"
      "$@"
    )
    export CUDA_VISIBLE_DEVICES="$TEST_USE_GPUS"
    ;;
  cycle)
    CMD=(bash /workspace/SOKE/scripts/run_train_infer_cycles.sh "$@")
    ;;
  bash|shell)
    exec bash "$@"
    ;;
  *)
    exec "$MODE" "$@"
    ;;
esac

echo "[docker-entrypoint] run_id=${RUN_ID} mode=${MODE}"
notify_text "[SOKE][${RUN_ID}] START mode=${MODE} host=$(hostname)"
start_heartbeat

set +e
"${CMD[@]}"
STATUS=$?
set -e

stop_heartbeat

if [[ "$STATUS" -eq 0 ]]; then
  touch "$RUN_DONE_FILE"
  echo "finished_at_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")" >>"$RUN_MANIFEST"
  echo "status=success" >>"$RUN_MANIFEST"
  notify_text "[SOKE][${RUN_ID}] DONE mode=${MODE}"
else
  touch "$RUN_FAILED_FILE"
  echo "finished_at_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")" >>"$RUN_MANIFEST"
  echo "status=failed" >>"$RUN_MANIFEST"
  echo "exit_code=$STATUS" >>"$RUN_MANIFEST"
  notify_text "[SOKE][${RUN_ID}] FAILED mode=${MODE} exit_code=${STATUS}"
fi

if [[ "$STATUS" -eq 0 ]] && [[ "$MODE" == "infer" || "$MODE" == "test" ]]; then
  PREVIEW_ENABLED="${SOKE_PREVIEW_GIF_ON_INFER:-1}"
  PREVIEW_ENABLED_LC="$(echo "$PREVIEW_ENABLED" | tr '[:upper:]' '[:lower:]')"
  if [[ "$PREVIEW_ENABLED_LC" == "1" || "$PREVIEW_ENABLED_LC" == "true" || "$PREVIEW_ENABLED_LC" == "yes" || "$PREVIEW_ENABLED_LC" == "on" ]]; then
    PRED_DIR="${SOKE_PREVIEW_PRED_DIR:-${SOKE_RESULTS_ROOT}/mgpt/SOKE_INFER/test_rank_0}"
    PREVIEW_OUT_DIR="${SOKE_GIF_ROOT}/${RUN_ID}"
    PREVIEW_INDEX="${SOKE_PREVIEW_INDEX:-0}"
    mkdir -p "$PREVIEW_OUT_DIR"
    if [[ -d "$PRED_DIR" ]]; then
      set +e
      python scripts/preview_test_sample.py \
        --pred_dir "$PRED_DIR" \
        --index "$PREVIEW_INDEX" \
        --out_dir "$PREVIEW_OUT_DIR" \
        --fps "${SOKE_PREVIEW_FPS:-20}"
      GIF_STATUS=$?
      set -e
      if [[ "$GIF_STATUS" -eq 0 ]]; then
        GIF_FILE="$(ls -1 "$PREVIEW_OUT_DIR"/*_compare_ref_pred.gif 2>/dev/null | head -n1 || true)"
        if [[ -n "$GIF_FILE" ]]; then
          notify_gif "$GIF_FILE" "[SOKE][${RUN_ID}] GT vs PRED"
        fi
      fi
    fi
  fi
fi

exit "$STATUS"
