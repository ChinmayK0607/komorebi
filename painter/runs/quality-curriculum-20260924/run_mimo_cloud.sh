#!/usr/bin/env bash
# Finite Codex Cloud teacher generation with bounded requests and resumable evidence.
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
TIER="${1:?tier: easy or hard}"
MODEL="${2:?model: xiaomi/mimo-v2.6-flash or xiaomi/mimo-v2.6-pro}"
RUN_PREFIX="${3:?stable run prefix required}"
case "$TIER" in easy|hard) ;; *) echo 'tier must be easy or hard' >&2; exit 2;; esac
case "$MODEL" in xiaomi/mimo-v2.6-flash|xiaomi/mimo-v2.6-pro) ;; *) echo 'unexpected model' >&2; exit 2;; esac
[[ "$RUN_PREFIX" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]{0,50}$ ]] || { echo 'invalid run prefix' >&2; exit 2; }
[[ -x "$PY" && -d "$PLAYWRIGHT_BROWSERS_PATH" ]] || { echo 'cloud renderer setup missing' >&2; exit 2; }
[[ -n "${AI_GATEWAY_API_KEY:-}" && -n "${HF_TOKEN:-}" ]] || { echo 'Gateway and HF environment credentials are required' >&2; exit 2; }
STAGE="$($PY "$RUN/prepare_mimo_cloud.py" --tier "$TIER" --model "$MODEL" | $PY -c 'import json,sys; print(json.load(sys.stdin)["stage_root"])')"
"$PY" - "$STAGE" "$TIER" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); tier=sys.argv[2]
path=root/'config.json'; config=json.loads(path.read_text())
quality=config['tracks']['quality']
quality.update(max_turns=6 if tier=='easy' else 8,
               max_tokens=16384 if tier=='easy' else 24576)
config.update(request_timeout_seconds=900,renderer_timeout=240 if tier=='easy' else 300)
path.write_text(json.dumps(config,indent=2,sort_keys=True)+'\n')
source_path=root/'restored-source.json'; source=json.loads(source_path.read_text())
source['campaign_profile']={'max_turns':quality['max_turns'],
                            'max_completion_tokens':quality['max_tokens'],
                            'request_timeout_seconds':config['request_timeout_seconds'],
                            'renderer_timeout_seconds':config['renderer_timeout'],
                            'prompt_sha256':hashlib.sha256((root/'prompt.txt').read_bytes()).hexdigest(),
                            'config_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
source_path.write_text(json.dumps(source,indent=2,sort_keys=True)+'\n')
PY
publish() {
  local suffix="$1"
  [[ -d "$STAGE/episodes" ]] || return 0
  "$PY" "$BENCH/cloud/publish_results.py" --root "$STAGE" --run-id "$RUN_PREFIX-$suffix"
}
cleanup() {
  local status=$?
  if (( status != 0 )) && [[ -d "$STAGE/episodes" ]]; then
    echo 'Generation stopped; attempting to preserve partial evidence on public HF.' >&2
    publish partial || echo 'Partial evidence publication failed; VM files remain until task ends.' >&2
  fi
}
trap cleanup EXIT
trap 'exit 143' TERM INT
installed=false
for attempt in 1 2 3; do
  if (cd "$STAGE" && env -u AI_GATEWAY_API_KEY -u HF_TOKEN pnpm install --frozen-lockfile --ignore-scripts --silent); then
    installed=true
    break
  fi
  echo "Stage dependency install attempt $attempt failed; retrying" >&2
  sleep $((attempt * 5))
done
[[ "$installed" == true ]] || { echo 'Stage dependency install failed after 3 attempts' >&2; exit 1; }
if [[ "$TIER" == easy ]]; then LIMITS=(2 4 7); else LIMITS=(2 5 10); fi
start=0
for limit in "${LIMITS[@]}"; do
  count=$((limit-start))
  echo "Starting candidate scenes $((start+1))-$limit/$limit for $TIER $MODEL" >&2
  "$PY" "$BENCH/run.py" --root "$STAGE" --dry-run --track quality \
    --start-episode "$start" --limit-episodes "$count" >/dev/null
  "$PY" "$BENCH/run.py" --root "$STAGE" --run --track quality \
    --start-episode "$start" --limit-episodes "$count" \
    --renderer "$ROOT/painter/vendor/integrations/watercolour/renderer.py" \
    --renderer-python "$PY" --browser-path "$PLAYWRIGHT_BROWSERS_PATH" --run-as-user painter
  publish "n$limit"
  start="$limit"
done
echo "Completed finite candidate generation for $TIER $MODEL; visual admission is still pending." >&2
