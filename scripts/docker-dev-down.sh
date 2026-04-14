#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${ROOT_DIR}/docker-compose.dev.yml"

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker is not installed."
  exit 1
fi

if [[ "${1:-}" == "--volumes" ]]; then
  docker compose -f "${COMPOSE_FILE}" down -v
else
  docker compose -f "${COMPOSE_FILE}" down
fi

echo "[figaro-dev] stopped"
