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
for LIMIT in 4 8 12; do
  echo "Generating $SHARD prefix $LIMIT/12" >&2
  "$PY" "$BENCH/run.py" --root "$STAGE" --dry-run --track quality --limit-episodes "$LIMIT" >/dev/null
  "$PY" "$BENCH/run.py" --root "$STAGE" --run --track quality --limit-episodes "$LIMIT" \
    --renderer "$ROOT/painter/vendor/integrations/watercolour/renderer.py" \
    --renderer-python "$PY" --browser-path "$PLAYWRIGHT_BROWSERS_PATH" \
    --run-as-user painter
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
done
echo "Completed $SHARD: 12 candidate trajectories, 4 or 6 turns maximum; visual review still required." >&2
