#!/usr/bin/env bash
# Finite CPU teacher shard with paid one-episode canary and public checkpoints.
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
RUN_ID="diverse-teacher-20260929-$SHARD"
COUNT="$($PY - "$RUN/source-receipt.json" "$SHARD" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['shards'][sys.argv[2]])
PY
)"
"$PY" "$BENCH/run.py" --root "$STAGE" --dry-run --track quality --limit-episodes "$COUNT" > "$STAGE/dry-run.json"
run_prefix() {
  local count="$1"
  "$PY" "$BENCH/run.py" --root "$STAGE" --run --track quality --limit-episodes "$count" \
    --renderer "$ROOT/painter/vendor/integrations/watercolour/renderer.py" \
    --renderer-python "$PY" --browser-path "$PLAYWRIGHT_BROWSERS_PATH" --run-as-user painter
  "$PY" "$BENCH/cloud/publish_results.py" --root "$STAGE" --run-id "$RUN_ID-n$count" --split
}
run_prefix 1
"$PY" - "$STAGE" <<'PY'
import json,sys
from pathlib import Path
episodes=list((Path(sys.argv[1])/'episodes').glob('*/episode.json'))
if len(episodes)!=1:
    raise SystemExit('canary episode missing or duplicated')
state=json.loads(episodes[0].read_text())
if not state.get('final_valid_canvas'):
    raise SystemExit('canary has no valid canvas; stop this shard before more paid calls')
print('Canary passed: one rendered canvas and public receipt; continuing finite shard.',flush=True)
PY
for LIMIT in 4 8 "$COUNT"; do
  (( LIMIT > COUNT )) && continue
  (( LIMIT == 1 )) && continue
  run_prefix "$LIMIT"
done
echo "Finished $SHARD: $COUNT candidate episodes; visual admission is separate."
