#!/usr/bin/env bash
set -euo pipefail

PPTX_PATH=${1:-}
shift || true

if [[ -z "$PPTX_PATH" ]]; then
  echo "Usage: $0 <path-to.pptx> [expected-string ...]" >&2
  exit 2
fi

if [[ ! -f "$PPTX_PATH" ]]; then
  echo "ERROR: pptx not found: $PPTX_PATH" >&2
  exit 2
fi

PYTHON=${PYTHON:-}
if [[ -z "$PYTHON" ]]; then
  if [[ -x "api/.venv/bin/python3" ]]; then
    PYTHON="api/.venv/bin/python3"
  elif [[ -x ".venv/bin/python3" ]]; then
    PYTHON=".venv/bin/python3"
  else
    PYTHON="python3"
  fi
fi

if [[ "$#" -eq 0 ]]; then
  set -- "PDF_TO_PPT_TEST"
fi

"$PYTHON" - <<'PY' "$PPTX_PATH" "$@"
from __future__ import annotations

import sys
import zipfile


def main() -> int:
    pptx_path = sys.argv[1]
    expected = sys.argv[2:]
    if not expected:
        expected = ["PDF_TO_PPT_TEST"]

    with zipfile.ZipFile(pptx_path) as zf:
        xml_blob_parts: list[str] = []
        for name in zf.namelist():
            if not name.startswith("ppt/"):
                continue
            if not name.endswith(".xml"):
                continue
            data = zf.read(name)
            xml_blob_parts.append(data.decode("utf-8", errors="ignore"))

    xml_blob = "\n".join(xml_blob_parts)
    missing = [s for s in expected if s not in xml_blob]
    if missing:
        print(f"ERROR: missing expected strings in PPTX XML: {missing}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
