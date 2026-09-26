#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(git rev-parse --show-toplevel)"
RUN="$ROOT/painter/runs/quality-curriculum-20260926"
BENCH="$ROOT/painter/benchmarks/openrouter-teachers-20260922"
RUNTIME="$ROOT/.painter-cloud-runtime"
RUN_ID="${1:-sol-brush-translation-pilot-20260927}"
[[ "$(uname -s)" == Linux ]] || { echo 'Codex Cloud Linux required' >&2; exit 2; }
[[ "$RUN_ID" =~ ^[a-z][a-z0-9-]{1,63}$ ]] || { echo 'invalid run ID' >&2; exit 2; }
[[ -n "${HF_TOKEN:-}" ]] || { echo 'HF_TOKEN needed for public evidence upload' >&2; exit 2; }
export PATH="/usr/bin:/bin:$PATH"
export PLAYWRIGHT_BROWSERS_PATH="$RUNTIME/browsers" HF_HUB_DISABLE_XET=1
if [[ ! -x "$RUNTIME/renderer-env/bin/python" || ! -d "$PLAYWRIGHT_BROWSERS_PATH" ]]; then
  bash "$BENCH/cloud/setup.sh"
fi
PY="$RUNTIME/renderer-env/bin/python"
"$PY" "$BENCH/renderer_smoke.py" --renderer "$ROOT/painter/vendor/integrations/watercolour/renderer.py" \
  --renderer-python "$PY" --browser-path "$PLAYWRIGHT_BROWSERS_PATH" --run-as-user painter \
  --timeout 600 --output-dir "$RUNTIME/sol-brush-smoke-$RUN_ID" >"$RUNTIME/sol-brush-smoke-$RUN_ID.json"
"$PY" "$RUN/render_sol_translation_pilot_cloud.py" --run-id "$RUN_ID" --workers 2 --timeout 300
"$PY" "$BENCH/cloud/publish_results.py" \
  --root "$ROOT/painter/collected/quality-curriculum-20260926/rendered-cloud/$RUN_ID" --run-id "$RUN_ID"
