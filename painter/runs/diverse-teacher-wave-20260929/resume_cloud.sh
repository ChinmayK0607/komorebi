#!/usr/bin/env bash
# Finite continuation from a verified public prefix; no completed API turns are resampled.
set -Eeuo pipefail
ROOT="$(git rev-parse --show-toplevel)"
RUN="$ROOT/painter/runs/diverse-teacher-wave-20260929"
BENCH="$ROOT/painter/benchmarks/openrouter-teachers-20260922"
RUNTIME="$ROOT/.painter-cloud-runtime"
SHARD="${1:?pass a shard from source-receipt.json}"
[[ "$(uname -s)" == Linux ]] || { echo 'Linux CPU environment required' >&2; exit 2; }
[[ "$SHARD" =~ ^(text|photo)-(flash|pro)-[0-9][0-9]$ ]] || { echo 'invalid shard' >&2; exit 2; }
[[ -n "${AI_GATEWAY_API_KEY:-}" && -n "${HF_TOKEN:-}" ]] || {
  echo 'Cloud environment needs Gateway and HF credentials' >&2; exit 2;
}
export PATH="/usr/bin:/bin:$PATH"
export NODE_USE_ENV_PROXY=1 PLAYWRIGHT_BROWSERS_PATH="$RUNTIME/browsers" HF_HUB_DISABLE_XET=1
if [[ ! -x "$RUNTIME/renderer-env/bin/python" || ! -d "$PLAYWRIGHT_BROWSERS_PATH" ]]; then
  bash "$BENCH/cloud/setup.sh"
fi
PY="$RUNTIME/renderer-env/bin/python"
(cd "$BENCH" && env -u AI_GATEWAY_API_KEY -u HF_TOKEN pnpm install --frozen-lockfile --ignore-scripts --silent)
STAGE="$($PY "$RUN/stage_cloud.py" "$SHARD")"
ln -s "$BENCH/node_modules" "$STAGE/node_modules"
RESTORE="$($PY "$RUN/resume_public_shard.py" "$SHARD" "$STAGE")"
read -r PREFIX COUNT < <("$PY" -c 'import json,sys;r=json.load(sys.stdin);print(r["restored_prefix"],r["source_rows"])' <<<"$RESTORE")
if (( PREFIX == COUNT )); then
  echo "Already public and complete: $SHARD ($COUNT episodes)"
  exit 0
fi
"$PY" "$BENCH/run.py" --root "$STAGE" --dry-run --track quality --limit-episodes "$COUNT" > "$STAGE/dry-run.json"
RUN_ID="diverse-teacher-20260929-$SHARD"
for LIMIT in 4 8 "$COUNT"; do
  (( LIMIT > COUNT || LIMIT <= PREFIX )) && continue
  "$PY" "$BENCH/run.py" --root "$STAGE" --run --track quality --limit-episodes "$LIMIT" \
    --renderer "$ROOT/painter/vendor/integrations/watercolour/renderer.py" \
    --renderer-python "$PY" --browser-path "$PLAYWRIGHT_BROWSERS_PATH" --run-as-user painter
  "$PY" "$BENCH/cloud/publish_results.py" --root "$STAGE" --run-id "$RUN_ID-n$LIMIT" --split
done
echo "Finished $SHARD from public prefix $PREFIX to $COUNT; visual admission remains separate."
