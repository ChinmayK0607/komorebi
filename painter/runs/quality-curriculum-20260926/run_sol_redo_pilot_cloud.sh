#!/usr/bin/env bash
# Finite two-scene MiMo render-inspect-revise pilot; no GPU or scheduler.
set -Eeuo pipefail
ROOT="$(git rev-parse --show-toplevel)"
RUN="$ROOT/painter/runs/quality-curriculum-20260926"
BENCH="$ROOT/painter/benchmarks/openrouter-teachers-20260922"
RUNTIME="$ROOT/.painter-cloud-runtime"
RUN_ID="${1:-sol-redo-pilot-20260927}"
[[ "$(uname -s)" == Linux ]] || { echo 'Codex Cloud Linux required' >&2; exit 2; }
[[ "$RUN_ID" =~ ^sol-redo-[a-z0-9-]{1,50}$ ]] || { echo 'invalid run ID' >&2; exit 2; }
[[ -n "${AI_GATEWAY_API_KEY:-}" && -n "${HF_TOKEN:-}" ]] || { echo 'Gateway and HF credentials required in Cloud environment' >&2; exit 2; }
export PATH="/usr/bin:/bin:$PATH"
export NODE_USE_ENV_PROXY=1 PLAYWRIGHT_BROWSERS_PATH="$RUNTIME/browsers" HF_HUB_DISABLE_XET=1
if [[ ! -x "$RUNTIME/renderer-env/bin/python" || ! -d "$PLAYWRIGHT_BROWSERS_PATH" ]]; then
  bash "$BENCH/cloud/setup.sh"
fi
PY="$RUNTIME/renderer-env/bin/python"
(cd "$BENCH" && env -u AI_GATEWAY_API_KEY -u HF_TOKEN pnpm install --frozen-lockfile --ignore-scripts --silent)
"$PY" "$RUN/collect_sol_translation_pilot.py" sol-brush-translation-pilot-20260927

run_one() {
  local audit_id="$1" stage="$ROOT/painter/collected/quality-curriculum-20260926/rendered-cloud/$RUN_ID/$1"
  "$PY" "$RUN/stage_sol_redo_pilot.py" "$audit_id" "$RUN_ID"
  ln -s "$BENCH/node_modules" "$stage/node_modules"
  "$PY" "$BENCH/run.py" --root "$stage" --dry-run --track quality --limit-episodes 1 > "$stage/dry-run.json"
  "$PY" "$BENCH/run.py" --root "$stage" --run --track quality --limit-episodes 1 \
    --renderer "$ROOT/painter/vendor/integrations/watercolour/renderer.py" \
    --renderer-python "$PY" --browser-path "$PLAYWRIGHT_BROWSERS_PATH" --run-as-user painter
  "$PY" "$RUN/stage_sol_redo_pilot.py" "$audit_id" "$RUN_ID" --finalize
  "$PY" "$BENCH/cloud/publish_results.py" --root "$stage" --run-id "$RUN_ID-$audit_id" --split
}

run_one t600-326 &
flash_pid=$!
run_one t600-492 &
pro_pid=$!
flash_status=0
pro_status=0
wait "$flash_pid" || flash_status=$?
wait "$pro_pid" || pro_status=$?
if (( flash_status != 0 || pro_status != 0 )); then
  echo "Finite redo pilot failed: flash=$flash_status pro=$pro_status" >&2
  exit 1
fi
echo 'Both teacher episodes and public hash-verified evidence are complete; visual selection remains pending.'
