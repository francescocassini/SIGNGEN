#!/usr/bin/env bash
set -euo pipefail

# Telegram helper for SSH-only training/inference monitoring.
# Reads credentials from env:
#   TELEGRAM_BOT_TOKEN
#   TELEGRAM_CHAT_ID

MODE="${1:-}"
shift || true

TOKEN="${TELEGRAM_BOT_TOKEN:-}"
CHAT_ID="${TELEGRAM_CHAT_ID:-}"
API_BASE="https://api.telegram.org/bot${TOKEN}"

if [[ -z "$TOKEN" || -z "$CHAT_ID" ]]; then
  exit 0
fi

case "$MODE" in
  text)
    MSG="${*:-}"
    if [[ -z "$MSG" ]]; then
      exit 0
    fi
    curl -sS -X POST "${API_BASE}/sendMessage" \
      --data-urlencode "chat_id=${CHAT_ID}" \
      --data-urlencode "text=${MSG}" \
      >/dev/null || true
    ;;
  gif)
    GIF_PATH="${1:-}"
    CAPTION="${2:-SOKE inference preview}"
    if [[ -z "$GIF_PATH" || ! -f "$GIF_PATH" ]]; then
      exit 0
    fi
    curl -sS -X POST "${API_BASE}/sendAnimation" \
      -F "chat_id=${CHAT_ID}" \
      -F "caption=${CAPTION}" \
      -F "animation=@${GIF_PATH}" \
      >/dev/null || true
    ;;
  *)
    exit 0
    ;;
esac
