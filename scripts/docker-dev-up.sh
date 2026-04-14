#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${ROOT_DIR}/docker-compose.dev.yml"
COMPOSE_FILE_REL="docker-compose.dev.yml"

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker is not installed."
  exit 1
fi

echo "[figaro-dev] starting postgres + backend + frontend..."
(
  cd "${ROOT_DIR}"
  sudo docker compose -f "${COMPOSE_FILE_REL}" up -d
)

echo
echo "[figaro-dev] started"
echo "  Frontend: http://127.0.0.1:5173"
echo "  Backend : http://127.0.0.1:8000"
echo "  API Docs: http://127.0.0.1:8000/docs"
echo
echo "Logs:"
echo "  docker compose -f ${COMPOSE_FILE} logs -f --tail=200"
