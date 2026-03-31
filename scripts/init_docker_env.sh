#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"

REPO_ID="${1:-Francesco77/soke-private-data}"
DATA_ROOT="${2:-/home/cirillo/Desktop/SOKE_DATA}"
AUTO_DL="${3:-1}"

if [[ -f "$ENV_FILE" && "${4:-}" != "--force" ]]; then
  echo "[ERROR] $ENV_FILE already exists."
  echo "Use --force as 4th argument to overwrite."
  echo "Example: $0 $REPO_ID $DATA_ROOT $AUTO_DL --force"
  exit 1
fi

UID_VAL="$(id -u)"
GID_VAL="$(id -g)"

cat > "$ENV_FILE" <<EOF
SOKE_HF_DATASET_REPO=$REPO_ID
HF_TOKEN=hf_xxx_replace_me
SOKE_DATA_ROOT_HOST=$DATA_ROOT
SOKE_AUTO_DOWNLOAD_DATASET=$AUTO_DL
SOKE_TRAIN_CFG=configs/soke.yaml
SOKE_TEST_CFG=configs/soke_infer_complete.yaml
LOCAL_UID=$UID_VAL
LOCAL_GID=$GID_VAL
EOF

chmod 600 "$ENV_FILE"

echo "[OK] Wrote $ENV_FILE"
echo "     LOCAL_UID=$UID_VAL LOCAL_GID=$GID_VAL"
echo "Next steps:"
echo "  1) edit HF_TOKEN in .env"
echo "  2) docker compose build --no-cache soke"
echo "  3) docker compose run --rm soke train"
