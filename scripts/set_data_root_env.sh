#!/usr/bin/env bash
# Usage:
#   source scripts/set_data_root_env.sh /absolute/path/to/SOKE_DATA
#
# This exports dataset env vars used by SOKE configs/scripts.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Run with: source scripts/set_data_root_env.sh /abs/path/to/SOKE_DATA"
  exit 1
fi

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: source scripts/set_data_root_env.sh /abs/path/to/SOKE_DATA"
  return 1
fi

DATA_ROOT="$1"
if [[ ! -d "$DATA_ROOT" ]]; then
  echo "Directory not found: $DATA_ROOT"
  return 1
fi

export SOKE_DATA_ROOT="$DATA_ROOT"
export SOKE_H2S_ROOT="$DATA_ROOT/How2Sign"
export SOKE_CSL_ROOT="$DATA_ROOT/CSL-Daily"
export SOKE_PHOENIX_ROOT="$DATA_ROOT/Phoenix_2014T"
export SOKE_CSL_MEAN_PATH="$SOKE_CSL_ROOT/mean.pt"
export SOKE_CSL_STD_PATH="$SOKE_CSL_ROOT/std.pt"

echo "SOKE dataset env configured:"
echo "  SOKE_DATA_ROOT=$SOKE_DATA_ROOT"
echo "  SOKE_H2S_ROOT=$SOKE_H2S_ROOT"
echo "  SOKE_CSL_ROOT=$SOKE_CSL_ROOT"
echo "  SOKE_PHOENIX_ROOT=$SOKE_PHOENIX_ROOT"
