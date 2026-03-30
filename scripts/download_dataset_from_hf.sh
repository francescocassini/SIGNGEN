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
export SOKE_SPLITS_ROOT="${SOKE_SPLITS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)/data/splits}"

python - <<'PY'
import os
import tarfile
import zipfile
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

repo_id = os.environ.get("REPO_ID")
target_dir = os.environ.get("TARGET_DIR")
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
splits_root = Path(os.environ.get("SOKE_SPLITS_ROOT", "")).resolve()

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

mapping = [
    (
        splits_root / "how2sign" / "how2sign_realigned_train_preprocessed_fps.csv",
        root / "How2Sign" / "train" / "re_aligned" / "how2sign_realigned_train_preprocessed_fps.csv",
    ),
    (
        splits_root / "how2sign" / "how2sign_realigned_val_preprocessed_fps.csv",
        root / "How2Sign" / "val" / "re_aligned" / "how2sign_realigned_val_preprocessed_fps.csv",
    ),
    (
        splits_root / "how2sign" / "how2sign_realigned_test_preprocessed_fps.csv",
        root / "How2Sign" / "test" / "re_aligned" / "how2sign_realigned_test_preprocessed_fps.csv",
    ),
    (
        splits_root / "csl_daily" / "csl_clean.train",
        root / "CSL-Daily" / "csl_clean.train",
    ),
    (
        splits_root / "csl_daily" / "csl_clean.val",
        root / "CSL-Daily" / "csl_clean.val",
    ),
    (
        splits_root / "csl_daily" / "csl_clean.test",
        root / "CSL-Daily" / "csl_clean.test",
    ),
    (
        splits_root / "phoenix" / "phoenix14t.train",
        root / "Phoenix_2014T" / "phoenix14t.train",
    ),
    (
        splits_root / "phoenix" / "phoenix14t.dev",
        root / "Phoenix_2014T" / "phoenix14t.dev",
    ),
    (
        splits_root / "phoenix" / "phoenix14t.test",
        root / "Phoenix_2014T" / "phoenix14t.test",
    ),
]

if splits_root.exists():
    repaired = 0
    for src, dst in mapping:
        if not src.exists():
            continue
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        shutil.copy2(src, dst)
        repaired += 1
    if repaired:
        print(f"[INFO] Repaired split files from repo: {repaired}")

print("[OK] Dataset sync completed.")
PY
