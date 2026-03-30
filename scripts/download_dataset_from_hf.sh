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
archive_specs = [
    ("How2Sign.tar.gz", root / "How2Sign" / "train" / "re_aligned" / "how2sign_realigned_train_preprocessed_fps.csv"),
    ("CSL-Daily.tar.gz", root / "CSL-Daily" / "csl_clean.train"),
    ("Phoenix_2014T.tar.gz", root / "Phoenix_2014T" / "phoenix14t.train"),
    ("How2Sign.zip", root / "How2Sign" / "train" / "re_aligned" / "how2sign_realigned_train_preprocessed_fps.csv"),
    ("CSL-Daily.zip", root / "CSL-Daily" / "csl_clean.train"),
    ("Phoenix_2014T.zip", root / "Phoenix_2014T" / "phoenix14t.train"),
]

for name, marker in archive_specs:
    p = root / name
    if not p.exists():
        continue
    if marker.exists():
        print(f"[INFO] Skipping extract for {p.name} (already present: {marker})")
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
