#!/usr/bin/env bash
# One bounded, image-conditioned canary for opaque-underpaint teacher guidance.
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
[[ -x "$PY" && -d "$PLAYWRIGHT_BROWSERS_PATH" ]] || { echo 'cloud renderer setup missing' >&2; exit 2; }
[[ -n "${AI_GATEWAY_API_KEY:-}" && -n "${HF_TOKEN:-}" ]] || { echo 'Gateway and HF environment credentials are required' >&2; exit 2; }
STAGE="$($PY "$RUN/prepare_mimo_cloud.py" --tier easy --model xiaomi/mimo-v2.6-flash | $PY -c 'import json,sys; print(json.load(sys.stdin)["stage_root"])')"
"$PY" - "$STAGE" <<'PY'
import hashlib, json, pathlib, sys
root=pathlib.Path(sys.argv[1])
config_path=root/'config.json'
config=json.loads(config_path.read_text())
config['benchmark']='quality-curriculum-flash-cup-underpaint-20260924'
config['tracks']['quality'].update(max_turns=6,max_tokens=16384)
config['concurrency']=1
config['renderer_timeout']=240
config['request_timeout_seconds']=900
config_path.write_text(json.dumps(config,indent=2,sort_keys=True)+'\n')
prompt_path=root/'prompt.txt'
source_path=root/'restored-source.json'
source=json.loads(source_path.read_text())
source['probe']={'reference_id':'simple-cup','max_turns':6,'max_completion_tokens':16384,
                 'request_timeout_seconds':900,'renderer_timeout_seconds':240,
                 'prompt_sha256':hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
                 'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest()}
source_path.write_text(json.dumps(source,indent=2,sort_keys=True)+'\n')
PY
(cd "$STAGE" && env -u AI_GATEWAY_API_KEY -u HF_TOKEN pnpm install --frozen-lockfile --ignore-scripts --silent)
"$PY" "$BENCH/run.py" --root "$STAGE" --dry-run --track quality --limit-episodes 1 >/dev/null
"$PY" "$BENCH/run.py" --root "$STAGE" --run --track quality --limit-episodes 1 \
  --renderer "$ROOT/painter/vendor/integrations/watercolour/renderer.py" \
  --renderer-python "$PY" --browser-path "$PLAYWRIGHT_BROWSERS_PATH" --run-as-user painter
"$PY" "$BENCH/cloud/publish_results.py" --root "$STAGE" --run-id mimo-flash-cup-underpaint-20260924
echo 'Flash cup canary published; visual admission remains pending.' >&2
