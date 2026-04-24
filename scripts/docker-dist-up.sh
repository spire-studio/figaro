#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${ROOT_DIR}/docker-compose.dist.yml"

SERVER_WEB_PORT="${DIST_SERVER_WEB_PORT:-5273}"
SERVER_API_PORT="${DIST_SERVER_API_PORT:-8100}"
CLIENT0_WEB_PORT="${DIST_CLIENT0_WEB_PORT:-5274}"
CLIENT0_API_PORT="${DIST_CLIENT0_API_PORT:-8101}"
CLIENT1_WEB_PORT="${DIST_CLIENT1_WEB_PORT:-5275}"
CLIENT1_API_PORT="${DIST_CLIENT1_API_PORT:-8102}"
USE_BUILD=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build)
      USE_BUILD=1
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker is not installed."
  exit 1
fi

COMPOSE_ARGS=(-f "${COMPOSE_FILE}")

echo "[figaro-dist] starting independent server/client0/client1 stacks..."
UP_ARGS=(up -d)
if [[ ${USE_BUILD} -eq 1 ]]; then
  UP_ARGS+=(--build)
fi
docker compose "${COMPOSE_ARGS[@]}" "${UP_ARGS[@]}" "${EXTRA_ARGS[@]}"

echo
echo "[figaro-dist] started"
echo "  Network mode: host (backend services)"
echo "  Server:"
echo "    Frontend: http://localhost:${SERVER_WEB_PORT}"
echo "    Backend : http://localhost:${SERVER_API_PORT}"
echo "    OpenAPI : http://localhost:${SERVER_API_PORT}/docs"
echo "  Client0:"
echo "    Frontend: http://localhost:${CLIENT0_WEB_PORT}"
echo "    Backend : http://localhost:${CLIENT0_API_PORT}"
echo "    OpenAPI : http://localhost:${CLIENT0_API_PORT}/docs"
echo "  Client1:"
echo "    Frontend: http://localhost:${CLIENT1_WEB_PORT}"
echo "    Backend : http://localhost:${CLIENT1_API_PORT}"
echo "    OpenAPI : http://localhost:${CLIENT1_API_PORT}/docs"
echo
echo "Logs:"
echo "  docker compose -f ${COMPOSE_FILE} logs -f --tail=200"
