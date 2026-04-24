#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE_REL="docker-compose.dev.yml"

SERVICE="${SERVICE:-backend}"
PROXY_HOSTPORT="${PROXY_HOSTPORT:-localhost:8888}"
HTTP_PROXY_VALUE="${HTTP_PROXY:-http://${PROXY_HOSTPORT}}"
HTTPS_PROXY_VALUE="${HTTPS_PROXY:-${HTTP_PROXY_VALUE}}"
ALL_PROXY_VALUE="${ALL_PROXY-${HTTP_PROXY_VALUE}}"
NO_PROXY_VALUE="${NO_PROXY:-localhost,localhost,postgres}"

USE_NO_CACHE=0
USE_PULL=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-cache)
      USE_NO_CACHE=1
      shift
      ;;
    --pull)
      USE_PULL=1
      shift
      ;;
    --service)
      SERVICE="${2:-}"
      if [[ -z "${SERVICE}" ]]; then
        echo "Error: --service requires a value"
        exit 1
      fi
      shift 2
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

if [[ -z "${SERVICE}" ]]; then
  echo "Error: service name cannot be empty."
  exit 1
fi

BUILD_ARGS=(build)
if [[ ${USE_NO_CACHE} -eq 1 ]]; then
  BUILD_ARGS+=(--no-cache)
fi
if [[ ${USE_PULL} -eq 1 ]]; then
  BUILD_ARGS+=(--pull)
fi
BUILD_ARGS+=("${SERVICE}")
BUILD_ARGS+=("${EXTRA_ARGS[@]}")

echo "[figaro-dev] building service='${SERVICE}' with proxy"
echo "  HTTP_PROXY=${HTTP_PROXY_VALUE}"
echo "  HTTPS_PROXY=${HTTPS_PROXY_VALUE}"
echo "  ALL_PROXY=${ALL_PROXY_VALUE}"
echo "  NO_PROXY=${NO_PROXY_VALUE}"

(
  cd "${ROOT_DIR}"
  sudo env \
    HTTP_PROXY="${HTTP_PROXY_VALUE}" \
    HTTPS_PROXY="${HTTPS_PROXY_VALUE}" \
    ALL_PROXY="${ALL_PROXY_VALUE}" \
    NO_PROXY="${NO_PROXY_VALUE}" \
    http_proxy="${HTTP_PROXY_VALUE}" \
    https_proxy="${HTTPS_PROXY_VALUE}" \
    all_proxy="${ALL_PROXY_VALUE}" \
    no_proxy="${NO_PROXY_VALUE}" \
    docker compose -f "${COMPOSE_FILE_REL}" "${BUILD_ARGS[@]}"
)

echo

echo "[figaro-dev] build completed"
echo "next: sudo docker compose -f ${ROOT_DIR}/${COMPOSE_FILE_REL} up -d --force-recreate ${SERVICE}"
