#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
WORKSPACE_DIR="$(cd -- "$SCRIPT_DIR/../../.." && pwd -P)"
VENV_DIR="${PAINTER_BENCHMARK_VENV:-$SCRIPT_DIR/.local-venv}"
PYTHON_BIN="${PAINTER_BENCHMARK_PYTHON:-python3}"
VENV_PYTHON="$VENV_DIR/bin/python"
BROWSER_DIR="${PLAYWRIGHT_BROWSERS_PATH:-$WORKSPACE_DIR/work/playwright-browsers}"
RENDERER="$SCRIPT_DIR/../../vendor/integrations/watercolour/renderer.py"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "--local is only supported on macOS; use the default launcher on Linux." >&2
    exit 2
fi

# Dry runs validate benchmark inputs without creating an environment or
# touching a key/browser.  This makes the command safe on a fresh checkout.
for argument in "$@"; do
    if [[ "$argument" == "--dry-run" ]]; then
        exec "$PYTHON_BIN" "$SCRIPT_DIR/run.py" --root "$SCRIPT_DIR" --dry-run "${@:1}"
    fi
done

if [[ ! -x "$VENV_PYTHON" ]]; then
    "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
fi
if ! "$VENV_PYTHON" -c 'import PIL, playwright, tqdm' >/dev/null 2>&1; then
    "$VENV_PYTHON" -m pip install --disable-pip-version-check --no-input -r "$SCRIPT_DIR/requirements-local.txt"
fi

mkdir -p "$BROWSER_DIR"
if ! "$VENV_PYTHON" -c 'from pathlib import Path; import sys; root=Path(sys.argv[1]); raise SystemExit(0 if any(p.name.startswith("chromium-") for p in root.iterdir()) else 1)' "$BROWSER_DIR"; then
    PLAYWRIGHT_BROWSERS_PATH="$BROWSER_DIR" "$VENV_PYTHON" -m playwright install chromium
fi

export PLAYWRIGHT_BROWSERS_PATH="$BROWSER_DIR"
exec "$VENV_PYTHON" "$SCRIPT_DIR/run.py" \
    --root "$SCRIPT_DIR" \
    --run --local \
    --renderer "$RENDERER" \
    --renderer-python "$VENV_PYTHON" \
    --browser-path "$BROWSER_DIR" \
    "$@"
