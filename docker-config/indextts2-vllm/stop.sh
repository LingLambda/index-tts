#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
CONFIG_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
docker compose -f "$CONFIG_DIR/compose.yml" --profile api --profile webui down --remove-orphans
