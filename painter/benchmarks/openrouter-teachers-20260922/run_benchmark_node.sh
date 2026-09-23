#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd -P)"
NODE_ROOT="$(cd -- "$REPO_ROOT/.." && pwd -P)"
PY="$NODE_ROOT/renderer-env/bin/python"
RENDERER="$REPO_ROOT/painter/vendor/integrations/watercolour/renderer.py"
BROWSERS="$NODE_ROOT/browsers"

[[ -x "$PY" ]] || { echo "renderer Python is missing: $PY" >&2; exit 2; }
[[ -f "$RENDERER" ]] || { echo "renderer is missing: $RENDERER" >&2; exit 2; }
[[ -d "$BROWSERS" ]] || { echo "Playwright browsers are missing: $BROWSERS" >&2; exit 2; }

export PLAYWRIGHT_BROWSERS_PATH="$BROWSERS"
mode="--run"
for argument in "$@"; do
    if [[ "$argument" == "--dry-run" ]]; then
        mode="--dry-run"
        break
    fi
done

exec "$PY" "$SCRIPT_DIR/run.py" \
    --root "$SCRIPT_DIR" \
    "$mode" \
    --renderer "$RENDERER" \
    --renderer-python "$PY" \
    --browser-path "$BROWSERS" \
    --run-as-user painter \
    "$@"
