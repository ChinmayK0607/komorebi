#!/usr/bin/env bash
# Finite 12-scene CPU teacher shard. No GPU or scheduled worker.
set -Eeuo pipefail

ROOT="$(git rev-parse --show-toplevel)"
RUN="$ROOT/painter/runs/quality-curriculum-20260924"
BENCH="$ROOT/painter/benchmarks/openrouter-teachers-20260922"
RUNTIME="$ROOT/.painter-cloud-runtime"
PY="$RUNTIME/renderer-env/bin/python"
export PATH="/usr/bin:/bin:$PATH"
export NODE_USE_ENV_PROXY=1
export PLAYWRIGHT_BROWSERS_PATH="$RUNTIME/browsers"
export HF_HUB_DISABLE_XET=1

SHARD="${1:?pass easy-a, easy-b, hard-a or hard-b}"
case "$SHARD" in easy-a|easy-b|hard-a|hard-b) ;; *) echo 'unexpected shard' >&2; exit 2;; esac
MODE="${2:-fresh}"
case "$MODE" in fresh|resume-n4|resume-n12) ;; *) echo 'second argument must be fresh, resume-n4 or resume-n12' >&2; exit 2;; esac
[[ -n "${AI_GATEWAY_API_KEY:-}" && -n "${HF_TOKEN:-}" ]] || {
  echo 'Cloud Gateway and HF environment credentials are required' >&2; exit 2;
}
if [[ ! -x "$PY" || ! -d "$PLAYWRIGHT_BROWSERS_PATH" ]]; then
  bash "$BENCH/cloud/setup.sh"
fi
[[ -x "$PY" && -d "$PLAYWRIGHT_BROWSERS_PATH" ]] || {
  echo 'Cloud renderer setup is incomplete' >&2; exit 2;
}
STAGE="$($PY "$RUN/stage_wave5_cloud.py" --shard "$SHARD")"
# The staged config retains 240 seconds so completed Gateway responses keep
# their exact request hashes. The first repaired pass used the older renderer's
# 180-second maximum. The long-render pass uses the upgraded 900-second-cap
# renderer to recover censored complex programs without re-paying for replies.
RENDER_TIMEOUT=180
if [[ "$MODE" == resume-n12 ]]; then RENDER_TIMEOUT=600; fi
"$PY" "$BENCH/renderer_smoke.py" \
  --renderer "$ROOT/painter/vendor/integrations/watercolour/renderer.py" \
  --renderer-python "$PY" --browser-path "$PLAYWRIGHT_BROWSERS_PATH" \
  --run-as-user painter --timeout "$RENDER_TIMEOUT" \
  --output-dir "$STAGE/renderer-smoke" >"$STAGE/renderer-smoke-result.json"
if [[ "$MODE" == resume-n4 ]]; then
  "$PY" "$BENCH/cloud/restore_public_shard.py" \
    --source-run-id "mimo-wave5-$SHARD-20260925-n4" --root "$STAGE"
elif [[ "$MODE" == resume-n12 ]]; then
  "$PY" "$BENCH/cloud/restore_public_shard.py" \
    --source-run-id "mimo-wave5-$SHARD-repaired-20260925-n12" --root "$STAGE"
fi
installed=false
for attempt in 1 2 3; do
  if (cd "$STAGE" && env -u AI_GATEWAY_API_KEY -u HF_TOKEN pnpm install --frozen-lockfile --ignore-scripts --silent); then
    installed=true
    break
  fi
  echo "Dependency install attempt $attempt failed; retrying" >&2
  sleep $((attempt * 5))
done
[[ "$installed" == true ]] || { echo 'Dependency install failed' >&2; exit 1; }
unset PAINTER_RUN_ID
RUN_ID="mimo-wave5-$SHARD-20260925"
if [[ "$MODE" == resume-n4 ]]; then RUN_ID="mimo-wave5-$SHARD-repaired-20260925"; fi
if [[ "$MODE" == resume-n12 ]]; then RUN_ID="mimo-wave5-$SHARD-longrender-20260925"; fi
LIMITS=(4 8 12)
if [[ "$MODE" == resume-n12 ]]; then LIMITS=(12); fi
for LIMIT in "${LIMITS[@]}"; do
  echo "Generating $SHARD prefix $LIMIT/12" >&2
  "$PY" "$BENCH/run.py" --root "$STAGE" --dry-run --track quality --limit-episodes "$LIMIT" >/dev/null
  "$PY" "$BENCH/run.py" --root "$STAGE" --run --track quality --limit-episodes "$LIMIT" \
    --renderer "$ROOT/painter/vendor/integrations/watercolour/renderer.py" \
    --renderer-python "$PY" --browser-path "$PLAYWRIGHT_BROWSERS_PATH" \
    --run-as-user painter --renderer-timeout-override "$RENDER_TIMEOUT"
  published=false
  for attempt in 1 2 3; do
    if "$PY" "$BENCH/cloud/publish_results.py" --root "$STAGE" --run-id "$RUN_ID-n$LIMIT"; then
      published=true
      break
    fi
    echo "Public checkpoint upload attempt $attempt failed; retrying" >&2
    sleep $((attempt * 5))
  done
  [[ "$published" == true ]] || { echo 'Public evidence upload failed' >&2; exit 1; }
  # Infrastructure failure is not a teacher-quality outcome. Preserve the
  # published receipt, then stop before starting another paid prefix.
  "$PY" - "$STAGE/run-summary.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
count = int(summary['episodes_selected'])
failed = int(summary['status_counts'].get('renderer_error', 0))
if count and failed == count:
    raise SystemExit('all episodes failed in renderer; stopping after public receipt')
PY
done
echo "Completed $SHARD: 12 candidate trajectories, 4 or 6 turns maximum; visual review still required." >&2
