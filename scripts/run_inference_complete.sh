#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CFG="configs/soke_infer_complete.yaml"
GPU_ID="${GPU_ID:-0}"
MODE="${1:-last}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SKIP_METRICS="${SKIP_METRICS:-0}"
LOG_ROOT="${SOKE_LOG_ROOT:-logs}"
ENABLE_TG="${SOKE_TELEGRAM_NOTIFY:-1}"

notify_text() {
  if [[ "$ENABLE_TG" != "1" ]]; then
    return 0
  fi
  "${ROOT_DIR}/scripts/telegram_notify.sh" text "$*" || true
}

notify_gif() {
  if [[ "$ENABLE_TG" != "1" ]]; then
    return 0
  fi
  local gif_path="$1"
  local caption="${2:-SOKE inference preview}"
  "${ROOT_DIR}/scripts/telegram_notify.sh" gif "$gif_path" "$caption" || true
}

if [[ ! -f "$CFG" ]]; then
  echo "Missing config: $CFG" >&2
  exit 1
fi

run_one() {
  local ckpt="$1"
  if [[ ! -f "$ckpt" ]]; then
    echo "Checkpoint not found: $ckpt" >&2
    return 1
  fi

  local stamp name log_file
  stamp="$(date +%Y%m%d_%H%M%S)"
  name="$(basename "$ckpt" .ckpt)"
  mkdir -p "$LOG_ROOT"
  log_file="${LOG_ROOT}/infer_${name}_${stamp}.log"

  echo "[INFO] Running inference on: $ckpt"
  echo "[INFO] Log file: $log_file"
  notify_text "[SOKE][infer] START ckpt=$(basename "$ckpt")"

  cmd=(
    python -u -m test
    --cfg "$CFG"
    --task t2m
    --nodebug
    --use_gpus "$GPU_ID"
    --device "$GPU_ID"
    --checkpoint "$ckpt"
  )

  if [[ -n "$MAX_SAMPLES" ]]; then
    cmd+=(--test_max_samples "$MAX_SAMPLES")
    echo "[INFO] TEST.MAX_SAMPLES override: $MAX_SAMPLES"
  fi
  if [[ "$SKIP_METRICS" == "1" ]]; then
    cmd+=(--skip_metrics)
    echo "[INFO] SKIP_METRICS enabled"
  fi

  set +e
  PYTHONPATH=. CUDA_VISIBLE_DEVICES="$GPU_ID" "${cmd[@]}" | tee "$log_file"
  local status=$?
  set -e

  if [[ "$status" -eq 0 ]]; then
    echo "[INFO] Done: $ckpt"
    notify_text "[SOKE][infer] DONE ckpt=$(basename "$ckpt")"
  else
    echo "[ERROR] Failed: $ckpt (exit=$status)" >&2
    notify_text "[SOKE][infer] FAILED ckpt=$(basename "$ckpt") exit=$status"
    return "$status"
  fi

  local pred_dir="${SOKE_PREVIEW_PRED_DIR:-${SOKE_RESULTS_ROOT:-results}/mgpt/SOKE_INFER/test_rank_0}"
  local preview_root="${SOKE_GIF_ROOT:-visualize/preview_samples}"
  local preview_out="${preview_root}/${name}_${stamp}"
  if [[ -d "$pred_dir" ]]; then
    mkdir -p "$preview_out"
    set +e
    python scripts/preview_test_sample.py \
      --pred_dir "$pred_dir" \
      --index "${SOKE_PREVIEW_INDEX:-0}" \
      --out_dir "$preview_out" \
      --fps "${SOKE_PREVIEW_FPS:-20}"
    local gif_status=$?
    set -e
    if [[ "$gif_status" -eq 0 ]]; then
      local gif_file
      gif_file="$(ls -1 "$preview_out"/*_compare_ref_pred.gif 2>/dev/null | head -n1 || true)"
      if [[ -n "$gif_file" ]]; then
        notify_gif "$gif_file" "[SOKE][infer] $(basename "$ckpt") GT vs PRED"
      fi
    fi
  fi
}

if [[ "$MODE" == "last" ]]; then
  run_one "experiments/mgpt/SOKE/checkpoints/last.ckpt"
elif [[ "$MODE" == "all" ]]; then
  run_one "experiments/mgpt/SOKE/checkpoints/last.ckpt"
  for ckpt in experiments/mgpt/SOKE/checkpoints/min-*.ckpt; do
    run_one "$ckpt"
  done
elif [[ -f "$MODE" ]]; then
  run_one "$MODE"
else
  echo "Usage:"
  echo "  scripts/run_inference_complete.sh last"
  echo "  scripts/run_inference_complete.sh all"
  echo "  scripts/run_inference_complete.sh /abs/or/relative/path/to/checkpoint.ckpt"
  echo
  echo "Optional env vars:"
  echo "  MAX_SAMPLES=64          # run only first N test samples"
  echo "  SKIP_METRICS=1          # skip expensive DTW/SMPL-X metrics (faster preview)"
  echo "Example:"
  echo "  MAX_SAMPLES=32 SKIP_METRICS=1 scripts/run_inference_complete.sh last"
  exit 1
fi
