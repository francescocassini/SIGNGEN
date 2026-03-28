#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage:
  scripts/link_external_pose_roots.sh \
    --how2sign-root /abs/path/how2sign \
    --csl-root /abs/path/csl_daily \
    --phoenix-root /abs/path/phoenix_2014t \
    [--data-root /abs/path/soke_data_root]

Expected source layouts:
  how2sign:
    <root>/train/poses/<SENTENCE_NAME>/*.pkl
    <root>/val/poses/<SENTENCE_NAME>/*.pkl
    <root>/test/poses/<SENTENCE_NAME>/*.pkl
    Optional (if already present): <root>/*/re_aligned/*.csv

  csl:
    <root>/poses/<NAME>/*.pkl
    Optional (if already present): <root>/csl_clean.{train,val,test}

  phoenix:
    <root>/<name-from-split>/*.pkl
    Optional (if already present): <root>/phoenix14t.{train,dev,test}
EOF
}

H2S_ROOT=""
CSL_ROOT=""
PHO_ROOT=""
DATA_ROOT="${SOKE_DATA_ROOT:-../data}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --how2sign-root) H2S_ROOT="$2"; shift 2 ;;
    --csl-root) CSL_ROOT="$2"; shift 2 ;;
    --phoenix-root) PHO_ROOT="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$H2S_ROOT" || -z "$CSL_ROOT" || -z "$PHO_ROOT" ]]; then
  echo "Missing required args."
  usage
  exit 1
fi

H2S_DST="${SOKE_H2S_ROOT:-$DATA_ROOT/How2Sign}"
CSL_DST="${SOKE_CSL_ROOT:-$DATA_ROOT/CSL-Daily}"
PHO_DST="${SOKE_PHOENIX_ROOT:-$DATA_ROOT/Phoenix_2014T}"

mkdir -p "$H2S_DST" "$CSL_DST" "$PHO_DST"

link_if_missing() {
  local src="$1"
  local dst="$2"
  if [[ -e "$dst" || -L "$dst" ]]; then
    echo "[SKIP] $dst already exists"
  else
    ln -s "$src" "$dst"
    echo "[LINK] $dst -> $src"
  fi
}

for split in train val test; do
  mkdir -p "${H2S_DST}/${split}"
  if [[ -d "${H2S_ROOT}/${split}/poses" ]]; then
    link_if_missing "${H2S_ROOT}/${split}/poses" "${H2S_DST}/${split}/poses"
  fi
  mkdir -p "${H2S_DST}/${split}/re_aligned"
  if [[ -f "${H2S_ROOT}/${split}/re_aligned/how2sign_realigned_${split}_preprocessed_fps.csv" ]]; then
    link_if_missing "${H2S_ROOT}/${split}/re_aligned/how2sign_realigned_${split}_preprocessed_fps.csv" \
      "${H2S_DST}/${split}/re_aligned/how2sign_realigned_${split}_preprocessed_fps.csv"
  fi
done

if [[ -d "${CSL_ROOT}/poses" ]]; then
  link_if_missing "${CSL_ROOT}/poses" "${CSL_DST}/poses"
fi
for split in train val test; do
  if [[ -f "${CSL_ROOT}/csl_clean.${split}" ]]; then
    link_if_missing "${CSL_ROOT}/csl_clean.${split}" "${CSL_DST}/csl_clean.${split}"
  fi
done

if [[ -d "${PHO_ROOT}" ]]; then
  # Phoenix is used directly as root containing utterance folders.
  # We only symlink split files individually to avoid replacing root dir.
  for split in train dev test; do
    if [[ -f "${PHO_ROOT}/phoenix14t.${split}" ]]; then
      link_if_missing "${PHO_ROOT}/phoenix14t.${split}" "${PHO_DST}/phoenix14t.${split}"
    fi
  done
fi

echo
echo "Done. Running setup check..."
./scripts/check_soke_setup.sh || true
