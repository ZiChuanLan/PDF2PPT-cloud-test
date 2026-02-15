#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

API_HOST=${API_HOST:-127.0.0.1}
API_PORT=${API_PORT:-}
API_URL=${API_URL:-}

PYTHON=${PYTHON:-}
if [[ -z "${PYTHON}" ]]; then
  if [[ -x "api/.venv/bin/python3" ]]; then
    PYTHON="api/.venv/bin/python3"
  elif [[ -x ".venv/bin/python3" ]]; then
    PYTHON=".venv/bin/python3"
  else
    PYTHON="python3"
  fi
fi

FIXTURE_PDF=${FIXTURE_PDF:-fixtures/pdfs/text.pdf}
OUT_PPTX=${OUT_PPTX:-.sisyphus/tmp/qa/output.pptx}

START_SERVER=${START_SERVER:-auto}

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: python not found/executable at: $PYTHON" >&2
  exit 2
fi

if [[ ! -f "$FIXTURE_PDF" ]]; then
  echo "ERROR: fixture not found: $FIXTURE_PDF" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUT_PPTX")"

SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

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
  curl -x "" --max-time 2 -sS "$API_URL/health" | "$PYTHON" -c "import json,sys; p=json.load(sys.stdin); sys.exit(0 if p.get('status')=='ok' else 1)"
}

if [[ -z "${API_URL}" ]]; then
  if [[ -z "${API_PORT}" ]]; then
    if curl -x "" --max-time 2 -fsS "http://${API_HOST}:8000/health" >/dev/null 2>&1; then
      API_PORT=8000
    elif curl -x "" --max-time 2 -fsS "http://${API_HOST}:8001/health" >/dev/null 2>&1; then
      API_PORT=8001
    elif port_is_free 8000; then
      API_PORT=8000
    else
      API_PORT=8001
    fi
  fi

  API_URL="http://${API_HOST}:${API_PORT}"
fi

if [[ "$START_SERVER" == "true" ]]; then
  NEED_START=1
elif [[ "$START_SERVER" == "false" ]]; then
  NEED_START=0
else
  if health_ok >/dev/null 2>&1; then
    NEED_START=0
  else
    NEED_START=1
  fi
fi

if [[ "$NEED_START" == "1" ]]; then
  echo "Starting API server in background..." >&2
  # Use in-memory job store mode for local QA runs (no Redis dependency).
  # The backend falls back to Redis when REDIS_URL is a real redis:// URL.
  (
    export PYTHONPATH=api
    export REDIS_URL=${REDIS_URL:-memory://}
    exec "$PYTHON" -m uvicorn app.main:app --host "$API_HOST" --port "${API_PORT:-8000}"
  ) &
  SERVER_PID=$!

  # Wait for health
  for _ in $(seq 1 60); do
    if health_ok >/dev/null 2>&1; then
      break
    fi
    sleep 0.2
  done

  if ! health_ok >/dev/null 2>&1; then
    echo "ERROR: API health check failed at $API_URL/health" >&2
    exit 1
  fi
fi

echo "Health OK" >&2

JOB_ID=$(
  curl -x "" --max-time 30 -sS -X POST \
    -F "file=@${FIXTURE_PDF}" \
    "$API_URL/api/v1/jobs" \
    | "$PYTHON" -c "import json,sys; p=json.load(sys.stdin); j=p.get('job_id'); sys.exit('Missing job_id in response') if not j else print(j)"
)

echo "Created job: $JOB_ID" >&2

STATUS=""
for _ in $(seq 1 200); do
  STATUS=$(
    curl -x "" --max-time 30 -sS "$API_URL/api/v1/jobs/$JOB_ID" \
      | "$PYTHON" -c "import json,sys; p=json.load(sys.stdin); print(p.get('status') or '')"
  )

  if [[ "$STATUS" == "completed" ]]; then
    break
  fi
  if [[ "$STATUS" == "failed" || "$STATUS" == "cancelled" ]]; then
    echo "ERROR: job ended with status=$STATUS" >&2
    exit 1
  fi
  sleep 0.25
done

if [[ "$STATUS" != "completed" ]]; then
  echo "ERROR: job did not complete in time (last status=$STATUS)" >&2
  exit 1
fi

curl -x "" --max-time 60 -sS -f -o "$OUT_PPTX" "$API_URL/api/v1/jobs/$JOB_ID/download"

"$PYTHON" -m zipfile -t "$OUT_PPTX" >/dev/null

"$(dirname "$0")/check_pptx_quality.sh" "$OUT_PPTX" "PDF_TO_PPT_TEST" >/dev/null

echo "OK: API smoke test passed" >&2
