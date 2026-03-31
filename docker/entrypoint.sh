#!/usr/bin/env bash
set -euo pipefail

cd /workspace/SOKE
umask "${SOKE_UMASK:-0002}"
echo "[docker-entrypoint] running as uid=$(id -u) gid=$(id -g) umask=$(umask)"

mkdir -p "${TMPDIR:-/tmp}" "${XDG_CONFIG_HOME:-/tmp/.config}" "${MPLCONFIGDIR:-/tmp/matplotlib}" "${HF_HOME:-/workspace/.cache/huggingface}" || true

export SOKE_DATA_ROOT="${SOKE_DATA_ROOT:-/workspace/SOKE_DATA}"
export SOKE_H2S_ROOT="${SOKE_H2S_ROOT:-$SOKE_DATA_ROOT/How2Sign}"
export SOKE_CSL_ROOT="${SOKE_CSL_ROOT:-$SOKE_DATA_ROOT/CSL-Daily}"
export SOKE_PHOENIX_ROOT="${SOKE_PHOENIX_ROOT:-$SOKE_DATA_ROOT/Phoenix_2014T}"
export SOKE_CSL_MEAN_PATH="${SOKE_CSL_MEAN_PATH:-$SOKE_CSL_ROOT/mean.pt}"
export SOKE_CSL_STD_PATH="${SOKE_CSL_STD_PATH:-$SOKE_CSL_ROOT/std.pt}"

AUTO_DOWNLOAD="${SOKE_AUTO_DOWNLOAD_DATASET:-1}"
if [[ "$AUTO_DOWNLOAD" == "1" ]] && [[ -n "${SOKE_HF_DATASET_REPO:-}" ]]; then
  echo "[docker-entrypoint] dataset sync from HF repo: ${SOKE_HF_DATASET_REPO}"
  /workspace/SOKE/scripts/download_dataset_from_hf.sh "${SOKE_HF_DATASET_REPO}" "${SOKE_DATA_ROOT}"
else
  echo "[docker-entrypoint] dataset auto-download skipped"
fi

MODE="${1:-train}"
shift || true

case "$MODE" in
  train)
    exec python -u -m train --cfg "${SOKE_TRAIN_CFG:-configs/soke.yaml}" --nodebug --use_gpus 0 --device 0 "$@"
    ;;
  infer|test)
    exec python -u -m test --cfg "${SOKE_TEST_CFG:-configs/soke_infer_complete.yaml}" --task t2m --nodebug --use_gpus 0 --device 0 "$@"
    ;;
  bash|shell)
    exec bash "$@"
    ;;
  *)
    exec "$MODE" "$@"
    ;;
esac
