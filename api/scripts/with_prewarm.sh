#!/bin/sh
set -eu

python -m app.services.paddle_prewarm
exec "$@"
