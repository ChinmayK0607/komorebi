#!/usr/bin/env bash
# One finite ten-scene CPU shard. Invoke separately for flash-00..06 or pro-00..02.
set -Eeuo pipefail
ROOT="$(git rev-parse --show-toplevel)"
RUN="$ROOT/painter/runs/quality-curriculum-20260926"
BENCH="$ROOT/painter/benchmarks/openrouter-teachers-20260922"
RUNTIME="$ROOT/.painter-cloud-runtime"
SHARD="${1:?pass flash-00..06 or pro-00..02}"
RUN_ID="${2:-teacher600-repair-hundred-20260927}"
[[ "$(uname -s)" == Linux ]] || { echo 'Codex Cloud Linux required' >&2; exit 2; }
[[ "$SHARD" =~ ^(flash-0[0-6]|pro-0[0-2])$ ]] || { echo 'invalid shard' >&2; exit 2; }
[[ "$RUN_ID" =~ ^teacher600-repair-[a-z0-9-]{1,50}$ ]] || { echo 'invalid run ID' >&2; exit 2; }
[[ -n "${AI_GATEWAY_API_KEY:-}" && -n "${HF_TOKEN:-}" ]] || { echo 'Gateway and HF credentials required in Cloud environment' >&2; exit 2; }
export PATH="/usr/bin:/bin:$PATH"
export NODE_USE_ENV_PROXY=1 PLAYWRIGHT_BROWSERS_PATH="$RUNTIME/browsers" HF_HUB_DISABLE_XET=1
if [[ ! -x "$RUNTIME/renderer-env/bin/python" || ! -d "$PLAYWRIGHT_BROWSERS_PATH" ]]; then
  bash "$BENCH/cloud/setup.sh"
fi
PY="$RUNTIME/renderer-env/bin/python"
(cd "$BENCH" && env -u AI_GATEWAY_API_KEY -u HF_TOKEN pnpm install --frozen-lockfile --ignore-scripts --silent)
STAGE="$ROOT/painter/collected/quality-curriculum-20260926/rendered-cloud/$RUN_ID/$SHARD"
"$PY" "$RUN/stage_repair_hundred_cloud.py" "$RUN_ID" "$SHARD"
ln -s "$BENCH/node_modules" "$STAGE/node_modules"
"$PY" "$BENCH/run.py" --root "$STAGE" --dry-run --track quality > "$STAGE/dry-run.json"
if ! "$PY" "$BENCH/run.py" --root "$STAGE" --run --track quality \
    --renderer "$ROOT/painter/vendor/integrations/watercolour/renderer.py" \
    --renderer-python "$PY" --browser-path "$PLAYWRIGHT_BROWSERS_PATH" --run-as-user painter; then
  if [[ -d "$STAGE/episodes" ]]; then
    "$PY" "$BENCH/cloud/publish_results.py" --root "$STAGE" --run-id "$RUN_ID-$SHARD-partial" --split || true
  fi
  exit 1
fi
"$PY" "$RUN/stage_repair_hundred_cloud.py" "$RUN_ID" "$SHARD" --finalize
"$PY" "$BENCH/cloud/publish_results.py" --root "$STAGE" --run-id "$RUN_ID-$SHARD" --split
echo "Published ten candidate episodes for $SHARD; visual admission remains pending."
