#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${ROOT_DIR}/docker-compose.dist.yml"
USE_VOLUMES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --volumes)
      USE_VOLUMES=1
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker is not installed."
  exit 1
fi

COMPOSE_ARGS=(-f "${COMPOSE_FILE}")

if [[ ${USE_VOLUMES} -eq 1 ]]; then
  docker compose "${COMPOSE_ARGS[@]}" down -v
else
  docker compose "${COMPOSE_ARGS[@]}" down
fi

echo "[figaro-dist] stopped"
