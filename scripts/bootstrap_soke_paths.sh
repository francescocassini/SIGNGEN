#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${SOKE_DATA_ROOT:-../data}"
H2S_ROOT="${SOKE_H2S_ROOT:-$DATA_ROOT/How2Sign}"
CSL_ROOT="${SOKE_CSL_ROOT:-$DATA_ROOT/CSL-Daily}"
PHO_ROOT="${SOKE_PHOENIX_ROOT:-$DATA_ROOT/Phoenix_2014T}"

echo "== Bootstrapping SOKE dataset paths =="
echo "Resolved roots:"
echo "  H2S: $H2S_ROOT"
echo "  CSL: $CSL_ROOT"
echo "  PHO: $PHO_ROOT"

# Create expected dataset directories
mkdir -p "$CSL_ROOT"
mkdir -p "$PHO_ROOT"
mkdir -p "$H2S_ROOT/train/re_aligned" "$H2S_ROOT/val/re_aligned" "$H2S_ROOT/test/re_aligned"
mkdir -p "$H2S_ROOT/train/poses" "$H2S_ROOT/val/poses" "$H2S_ROOT/test/poses"

# Link split files if present
for f in train val test; do
  src="$DATA_ROOT/splits/csl_daily/csl_clean.${f}"
  dst="$CSL_ROOT/csl_clean.${f}"
  if [[ -f "$src" && ! -e "$dst" ]]; then
    ln -s "$(realpath "$src")" "$dst"
    echo "[LINK] $dst -> $src"
  fi
done

for f in train dev test; do
  src="$DATA_ROOT/splits/phoenix/phoenix14t.${f}"
  dst="$PHO_ROOT/phoenix14t.${f}"
  if [[ -f "$src" && ! -e "$dst" ]]; then
    ln -s "$(realpath "$src")" "$dst"
    echo "[LINK] $dst -> $src"
  fi
done

for f in train val test; do
  src="$DATA_ROOT/splits/how2sign/how2sign_realigned_${f}_preprocessed_fps.csv"
  dst="$H2S_ROOT/${f}/re_aligned/how2sign_realigned_${f}_preprocessed_fps.csv"
  if [[ -f "$src" && ! -e "$dst" ]]; then
    ln -s "$(realpath "$src")" "$dst"
    echo "[LINK] $dst -> $src"
  fi
done

echo
echo "Bootstrap done."
echo "Note: this script does NOT download/convert pose files."
echo "You still need:"
echo "  - $CSL_ROOT/poses/*"
echo "  - $PHO_ROOT/<utterance>/*"
echo "  - $H2S_ROOT/{train,val,test}/poses/<sample>/*"
