#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${SOKE_DATA_ROOT:-../data}"
H2S_ROOT="${SOKE_H2S_ROOT:-$DATA_ROOT/How2Sign}"
CSL_ROOT="${SOKE_CSL_ROOT:-$DATA_ROOT/CSL-Daily}"
PHO_ROOT="${SOKE_PHOENIX_ROOT:-$DATA_ROOT/Phoenix_2014T}"
CSL_MEAN_PATH="${SOKE_CSL_MEAN_PATH:-$CSL_ROOT/mean.pt}"
CSL_STD_PATH="${SOKE_CSL_STD_PATH:-$CSL_ROOT/std.pt}"

missing=0

check_file() {
  local p="$1"
  local label="$2"
  if [[ -f "$p" ]]; then
    echo "[OK]    $label: $p"
  else
    echo "[MISS]  $label: $p"
    missing=1
  fi
}

check_dir() {
  local p="$1"
  local label="$2"
  if [[ -d "$p" ]]; then
    echo "[OK]    $label: $p"
  else
    echo "[MISS]  $label: $p"
    missing=1
  fi
}

check_nonempty_dir() {
  local p="$1"
  local label="$2"
  if [[ -d "$p" ]] && [[ -n "$(find -L "$p" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "[OK]    $label: $p"
  else
    echo "[MISS]  $label: $p (missing or empty)"
    missing=1
  fi
}

check_nonempty_subdir() {
  local p="$1"
  local label="$2"
  if [[ -d "$p" ]] && [[ -n "$(find -L "$p" -mindepth 1 -maxdepth 1 -type d -print -quit 2>/dev/null)" ]]; then
    echo "[OK]    $label: $p"
  else
    echo "[MISS]  $label: $p (no utterance subfolders found)"
    missing=1
  fi
}

echo "== SOKE setup check =="
echo "repo: $ROOT_DIR"
echo

check_file "deps/tokenizer_ckpt/tokenizer.ckpt" "Tokenizer checkpoint"
check_file "deps/mbart-h2s-csl-phoenix/pytorch_model.bin" "mBART weights"
check_file "deps/flan-t5-base/pytorch_model.bin" "flan-t5-base weights"
check_dir "deps/t2m" "t2m evaluators"
check_dir "deps/smpl_models" "SMPL models"

echo
echo "== Dataset roots required by configs/soke.yaml =="
echo "Resolved roots:"
echo "  H2S: $H2S_ROOT"
echo "  CSL: $CSL_ROOT"
echo "  PHO: $PHO_ROOT"
check_dir "$H2S_ROOT" "How2Sign root"
check_dir "$CSL_ROOT" "CSL root"
check_dir "$PHO_ROOT" "Phoenix root"
check_file "$CSL_MEAN_PATH" "CSL mean"
check_file "$CSL_STD_PATH" "CSL std"

echo
echo "== Split/index files =="
check_file "$H2S_ROOT/train/re_aligned/how2sign_realigned_train_preprocessed_fps.csv" "How2Sign train CSV"
check_file "$H2S_ROOT/val/re_aligned/how2sign_realigned_val_preprocessed_fps.csv" "How2Sign val CSV"
check_file "$H2S_ROOT/test/re_aligned/how2sign_realigned_test_preprocessed_fps.csv" "How2Sign test CSV"
check_file "$CSL_ROOT/csl_clean.train" "CSL train split"
check_file "$CSL_ROOT/csl_clean.val" "CSL val split"
check_file "$CSL_ROOT/csl_clean.test" "CSL test split"
check_file "$PHO_ROOT/phoenix14t.train" "Phoenix train split"
check_file "$PHO_ROOT/phoenix14t.dev" "Phoenix dev split"
check_file "$PHO_ROOT/phoenix14t.test" "Phoenix test split"

echo
echo "== Pose folders (expected by dataloader) =="
check_nonempty_dir "$CSL_ROOT/poses" "CSL poses"
check_nonempty_subdir "$PHO_ROOT" "Phoenix utterance folders"
check_nonempty_dir "$H2S_ROOT/train/poses" "How2Sign train poses"
check_nonempty_dir "$H2S_ROOT/val/poses" "How2Sign val poses"
check_nonempty_dir "$H2S_ROOT/test/poses" "How2Sign test poses"

echo
echo "== Checkpoints =="
check_file "experiments/mgpt/SOKE/checkpoints/last.ckpt" "AMG checkpoint (for test.py)"

echo
if [[ "$missing" -eq 0 ]]; then
  echo "All required checks passed."
  exit 0
else
  echo "Some required items are missing."
  exit 1
fi
