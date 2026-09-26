#!/usr/bin/env bash
# Ready-to-launch finite mixed correction wave. Do not call until the pilot is reviewed.
set -Eeuo pipefail
ROOT="$(git rev-parse --show-toplevel)"
RUN="$ROOT/painter/runs/quality-curriculum-20260926"
BENCH="$ROOT/painter/benchmarks/openrouter-teachers-20260922"
RUNTIME="$ROOT/.painter-cloud-runtime"
RUN_ID="${1:-teacher600-repair-eight-20260927}"
[[ "$(uname -s)" == Linux ]] || { echo 'Codex Cloud Linux required' >&2; exit 2; }
[[ "$RUN_ID" =~ ^teacher600-repair-[a-z0-9-]{1,50}$ ]] || { echo 'invalid run ID' >&2; exit 2; }
[[ -n "${AI_GATEWAY_API_KEY:-}" && -n "${HF_TOKEN:-}" ]] || { echo 'Gateway and HF credentials required in Cloud environment' >&2; exit 2; }
export PATH="/usr/bin:/bin:$PATH"
export NODE_USE_ENV_PROXY=1 PLAYWRIGHT_BROWSERS_PATH="$RUNTIME/browsers" HF_HUB_DISABLE_XET=1
if [[ ! -x "$RUNTIME/renderer-env/bin/python" || ! -d "$PLAYWRIGHT_BROWSERS_PATH" ]]; then
  bash "$BENCH/cloud/setup.sh"
fi
PY="$RUNTIME/renderer-env/bin/python"
(cd "$BENCH" && env -u AI_GATEWAY_API_KEY -u HF_TOKEN pnpm install --frozen-lockfile --ignore-scripts --silent)

run_tier() {
  local tier="$1" stage="$ROOT/painter/collected/quality-curriculum-20260926/rendered-cloud/$RUN_ID/$1"
  "$PY" "$RUN/stage_next_repair_wave_cloud.py" "$RUN_ID" "$tier"
  ln -s "$BENCH/node_modules" "$stage/node_modules"
  "$PY" "$BENCH/run.py" --root "$stage" --dry-run --track quality > "$stage/dry-run.json"
  if ! "$PY" "$BENCH/run.py" --root "$stage" --run --track quality \
      --renderer "$ROOT/painter/vendor/integrations/watercolour/renderer.py" \
      --renderer-python "$PY" --browser-path "$PLAYWRIGHT_BROWSERS_PATH" --run-as-user painter; then
    if [[ -d "$stage/episodes" ]]; then
      "$PY" "$BENCH/cloud/publish_results.py" --root "$stage" --run-id "$RUN_ID-$tier-partial" --split || true
    fi
    return 1
  fi
  "$PY" "$RUN/stage_next_repair_wave_cloud.py" "$RUN_ID" "$tier" --finalize
  "$PY" "$BENCH/cloud/publish_results.py" --root "$stage" --run-id "$RUN_ID-$tier" --split
}

run_tier flash & flash_pid=$!
run_tier pro & pro_pid=$!
flash_status=0
pro_status=0
wait "$flash_pid" || flash_status=$?
wait "$pro_pid" || pro_status=$?
if (( flash_status != 0 || pro_status != 0 )); then
  echo "Mixed repair wave failed: flash=$flash_status pro=$pro_status" >&2
  exit 1
fi
echo 'Eight finite teacher episodes published as candidates; visual admission remains pending.'
