#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CFG="configs/soke_infer_complete.yaml"
GPU_ID="${GPU_ID:-0}"
MODE="${1:-last}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SKIP_METRICS="${SKIP_METRICS:-0}"

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
  mkdir -p logs
  log_file="logs/infer_${name}_${stamp}.log"

  echo "[INFO] Running inference on: $ckpt"
  echo "[INFO] Log file: $log_file"

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

  PYTHONPATH=. CUDA_VISIBLE_DEVICES="$GPU_ID" "${cmd[@]}" | tee "$log_file"

  echo "[INFO] Done: $ckpt"
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
