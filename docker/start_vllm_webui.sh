#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/root/docker-data/indextts2-vllm}"
WEBUI_PORT="${WEBUI_PORT:-7860}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$REPO_ROOT/docker/compose.yml"

MODEL_DIR="$WORK_ROOT/checkpoints/IndexTTS-2-vLLM"
if [ ! -f "$MODEL_DIR/config.yaml" ] || [ ! -f "$MODEL_DIR/gpt/pytorch_model.bin" ]; then
  echo "Model directory is not ready: $MODEL_DIR" >&2
  echo "Expected the vLLM model at: $MODEL_DIR" >&2
  exit 2
fi

mkdir -p "$WORK_ROOT/cache" "$WORK_ROOT/logs" "$WORK_ROOT/outputs"

cd "$REPO_ROOT"
docker compose -f "$COMPOSE_FILE" --profile api --profile webui down --remove-orphans >/dev/null 2>&1 || true
WORK_ROOT="$WORK_ROOT" WEBUI_PORT="$WEBUI_PORT" \
  docker compose -f "$COMPOSE_FILE" --profile webui up -d --force-recreate webui

cat <<EOF
Started IndexTTS2 vLLM WebUI.
URL: http://127.0.0.1:$WEBUI_PORT
Logs: docker logs -f indextts2-vllm-webui
EOF
