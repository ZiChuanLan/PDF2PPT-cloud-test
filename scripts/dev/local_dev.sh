#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-}"
API_BIND_HOST="${API_BIND_HOST:-127.0.0.1}"

WEB_PORT="${WEB_PORT:-3000}"

PYTHON="${PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
  if [[ -x "${ROOT_DIR}/api/.venv/bin/python" ]]; then
    PYTHON="${ROOT_DIR}/api/.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

port_is_free() {
  local port="$1"
  "$PYTHON" - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
s = socket.socket()
try:
    s.bind(("127.0.0.1", port))
except OSError:
    raise SystemExit(1)
finally:
    s.close()
raise SystemExit(0)
PY
}

health_ok() {
  local port="$1"
  curl -x "" --max-time 2 -fsS "http://${API_HOST}:${port}/health" >/dev/null 2>&1
}

if [[ -z "${API_PORT}" ]]; then
  if health_ok 8000; then
    API_PORT=8000
  elif health_ok 8001; then
    API_PORT=8001
  elif port_is_free 8000; then
    API_PORT=8000
  else
    API_PORT=8001
  fi
fi

API_URL="http://${API_HOST}:${API_PORT}"

export PYTHONPATH="${ROOT_DIR}/api"
export REDIS_URL="${REDIS_URL:-memory://}"
export CORS_ALLOW_ORIGINS="${CORS_ALLOW_ORIGINS:-http://localhost:${WEB_PORT},http://127.0.0.1:${WEB_PORT}}"

API_PID=""
cleanup() {
  if [[ -n "${API_PID}" ]]; then
    kill "${API_PID}" >/dev/null 2>&1 || true
    wait "${API_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

if ! health_ok "${API_PORT}"; then
  echo "[dev-local] Starting API: ${API_URL}" >&2
  (
    cd "${ROOT_DIR}"
    exec "$PYTHON" -m uvicorn app.main:app --host "${API_BIND_HOST}" --port "${API_PORT}"
  ) &
  API_PID=$!

  for _ in $(seq 1 80); do
    if health_ok "${API_PORT}"; then
      break
    fi
    sleep 0.2
  done

  if ! health_ok "${API_PORT}"; then
    echo "[dev-local] ERROR: API health check failed: ${API_URL}/health" >&2
    exit 1
  fi
else
  echo "[dev-local] Reusing running API: ${API_URL}" >&2
fi

echo "[dev-local] Starting Web: http://localhost:${WEB_PORT} (API -> ${API_URL})" >&2
cd "${ROOT_DIR}/web"
NEXT_PUBLIC_API_URL="${API_URL}" npm run dev -- -p "${WEB_PORT}"
