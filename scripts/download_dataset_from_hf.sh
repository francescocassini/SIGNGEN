#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${1:-${SOKE_HF_DATASET_REPO:-}}"
TARGET_DIR="${2:-${SOKE_DATA_ROOT:-$HOME/Desktop/SOKE_DATA}}"

if [[ -z "$REPO_ID" ]]; then
  echo "[ERROR] Missing repo id."
  echo "Usage: $0 <HF_USER/DATASET_REPO> [TARGET_DIR]"
  echo "or set SOKE_HF_DATASET_REPO in environment."
  exit 1
fi

echo "[INFO] Downloading private dataset: $REPO_ID"
echo "[INFO] Target dir: $TARGET_DIR"

mkdir -p "$TARGET_DIR"
export REPO_ID TARGET_DIR

python - <<'PY'
import os
import tarfile
import zipfile
from pathlib import Path
from huggingface_hub import snapshot_download

repo_id = os.environ.get("REPO_ID")
target_dir = os.environ.get("TARGET_DIR")
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

if not repo_id:
    raise SystemExit("REPO_ID not provided")

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=target_dir,
    local_dir_use_symlinks=False,
    token=token,
    resume_download=True,
)

root = Path(target_dir)
archive_names = [
    "How2Sign.tar.gz",
    "CSL-Daily.tar.gz",
    "Phoenix_2014T.tar.gz",
    "How2Sign.zip",
    "CSL-Daily.zip",
    "Phoenix_2014T.zip",
]

for name in archive_names:
    p = root / name
    if not p.exists():
        continue
    print(f"[INFO] Extracting {p.name} ...")
    if p.suffix == ".zip":
        with zipfile.ZipFile(p, "r") as zf:
            zf.extractall(root)
    else:
        with tarfile.open(p, "r:*") as tf:
            tf.extractall(root)

print("[OK] Dataset sync completed.")
PY
