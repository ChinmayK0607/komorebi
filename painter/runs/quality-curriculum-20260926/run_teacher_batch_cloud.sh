#!/usr/bin/env bash
# Finite Codex Cloud CPU render of a public teacher500 source batch.
set -Eeuo pipefail

ROOT="$(git rev-parse --show-toplevel)"
RUN="$ROOT/painter/runs/quality-curriculum-20260926"
BENCH="$ROOT/painter/benchmarks/openrouter-teachers-20260922"
RUNTIME="$ROOT/.painter-cloud-runtime"
BATCH="${1:?pass a teacher batch name}"
RUN_ID="${2:?pass a unique run ID}"
WORKERS="${3:-2}"
[[ "$BATCH" =~ ^[a-z][a-z0-9-]{1,63}$ && "$RUN_ID" =~ ^[a-z][a-z0-9-]{1,63}$ ]] || {
  echo 'invalid batch or run ID' >&2; exit 2;
}
[[ "$WORKERS" =~ ^[1-8]$ ]] || { echo 'workers must be 1..8' >&2; exit 2; }
[[ -n "${HF_TOKEN:-}" ]] || { echo 'HF_TOKEN is required to publish render evidence' >&2; exit 2; }
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
  --output-dir "$RUNTIME/teacher500-smoke-$RUN_ID" >"$RUNTIME/teacher500-smoke-$RUN_ID.json"
"$PY" "$RUN/render_teacher_batch_cloud.py" "$BATCH" "$RUN_ID" --timeout 600 --workers "$WORKERS"
"$PY" "$BENCH/cloud/publish_results.py" \
  --root "$ROOT/painter/collected/quality-curriculum-20260926/rendered-cloud/$RUN_ID" \
  --run-id "$RUN_ID"
