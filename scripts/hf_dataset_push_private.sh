#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <HF_USER/DATASET_REPO> [LOCAL_DATA_DIR]"
  echo "Example: $0 francescocassini/soke-private-data /home/cirillo/Desktop/SOKE_DATA"
  exit 1
fi

REPO_ID="$1"
LOCAL_DATA_DIR="${2:-$HOME/Desktop/SOKE_DATA}"

if [[ ! -d "$LOCAL_DATA_DIR" ]]; then
  echo "[ERROR] Local data dir not found: $LOCAL_DATA_DIR"
  exit 1
fi

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "[ERROR] huggingface-cli not found. Install with: pip install -U huggingface_hub"
  exit 1
fi

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "[ERROR] git-lfs not found. Install git-lfs first."
  exit 1
fi

echo "[INFO] Checking Hugging Face login..."
if ! huggingface-cli whoami >/dev/null 2>&1; then
  echo "[ERROR] Not logged in. Run: huggingface-cli login"
  exit 1
fi

git lfs install

cd "$LOCAL_DATA_DIR"

if [[ ! -d .git ]]; then
  echo "[INFO] Initializing git repo in $LOCAL_DATA_DIR"
  git init -b main
fi

if ! git remote get-url origin >/dev/null 2>&1; then
  git remote add origin "https://huggingface.co/datasets/$REPO_ID"
else
  git remote set-url origin "https://huggingface.co/datasets/$REPO_ID"
fi

# Track common large file types used in SOKE datasets.
git lfs track "*.pt" "*.pth" "*.ckpt" "*.bin" "*.npy" "*.npz" "*.pkl" "*.zip" "*.tar" "*.gz" "*.mp4" "*.avi" "*.mov"

git add .gitattributes
git add .

if git diff --cached --quiet; then
  echo "[INFO] No changes to commit."
else
  git commit -m "Update private SOKE dataset"
fi

echo "[INFO] Pushing to https://huggingface.co/datasets/$REPO_ID"
git push -u origin main

echo "[OK] Dataset push completed."
