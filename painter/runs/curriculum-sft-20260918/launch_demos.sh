#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(realpath "${1:?node workspace required}")
cd "$ROOT";source env.sh
exec 8>demonstrations.lock;flock -n 8
trap 'echo "$?" > demonstrations.exit' EXIT
WORKERS=$(python3 -c 'import json;print(json.load(open("hardware-plan.json"))["render_workers"])')
renderer-env/bin/python render_demos.py --data data --renderer vendor/integrations/watercolour/renderer.py --python renderer-env/bin/python --workers "$WORKERS"
