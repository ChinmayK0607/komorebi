#!/usr/bin/env bash
# Finite Cloud CPU render of 12 programs with exact photos from prior public bundles.
set -Eeuo pipefail
ROOT="$(git rev-parse --show-toplevel)"
RUN="$ROOT/painter/runs/quality-curriculum-20260926"
BENCH="$ROOT/painter/benchmarks/openrouter-teachers-20260922"
RUNTIME="$ROOT/.painter-cloud-runtime"
RUN_ID="teacher600-astra-coco128-overlay-20260927"
[[ -n "${HF_TOKEN:-}" ]] || { echo 'HF_TOKEN is required for result publication' >&2; exit 2; }
export PATH="/usr/bin:/bin:$PATH"
export PLAYWRIGHT_BROWSERS_PATH="$RUNTIME/browsers"
export HF_HUB_DISABLE_XET=1
if [[ ! -x "$RUNTIME/renderer-env/bin/python" || ! -d "$PLAYWRIGHT_BROWSERS_PATH" ]]; then
  bash "$BENCH/cloud/setup.sh"
fi
PY="$RUNTIME/renderer-env/bin/python"
"$PY" "$BENCH/renderer_smoke.py" \
  --renderer "$ROOT/painter/vendor/integrations/watercolour/renderer.py" \
  --renderer-python "$PY" --browser-path "$PLAYWRIGHT_BROWSERS_PATH" \
  --run-as-user painter --timeout 600 \
  --output-dir "$RUNTIME/teacher600-overlay-smoke" >"$RUNTIME/teacher600-overlay-smoke.json"
"$PY" "$RUN/render_coco128_overlay_cloud.py" "$RUN_ID" --workers 2
"$PY" "$BENCH/cloud/publish_results.py" \
  --root "$ROOT/painter/collected/quality-curriculum-20260926/rendered-cloud/$RUN_ID" \
  --run-id "$RUN_ID"
