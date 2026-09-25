#!/usr/bin/env bash
# Matched simple-cup canary: stronger Pro teacher with the solid-underpaint prompt.
set -Eeuo pipefail

ROOT="$(git rev-parse --show-toplevel)"
RUN="$ROOT/painter/runs/quality-curriculum-20260924"
BENCH="$ROOT/painter/benchmarks/openrouter-teachers-20260922"
RUNTIME="$ROOT/.painter-cloud-runtime"
PY="$RUNTIME/renderer-env/bin/python"
RUN_ID="${PAINTER_RUN_ID:-mimo-pro-cup-opaque-20260925}"
STAGE_NAME="${PAINTER_STAGE_NAME:-easy-pro}"
MAX_TOKENS="${PAINTER_MAX_TOKENS:-16384}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] || { echo 'invalid PAINTER_RUN_ID' >&2; exit 2; }
export PATH="/usr/bin:/bin:$PATH"
export NODE_USE_ENV_PROXY=1
export PLAYWRIGHT_BROWSERS_PATH="$RUNTIME/browsers"
export HF_HUB_DISABLE_XET=1

[[ -x "$PY" && -d "$PLAYWRIGHT_BROWSERS_PATH" ]] || { echo 'cloud renderer setup missing' >&2; exit 2; }
[[ -n "${AI_GATEWAY_API_KEY:-}" && -n "${HF_TOKEN:-}" ]] || { echo 'Gateway and HF environment credentials are required' >&2; exit 2; }

STAGE="$($PY "$RUN/prepare_mimo_cloud.py" --tier easy --model xiaomi/mimo-v2.6-pro --stage-name "$STAGE_NAME" | $PY -c 'import json,sys; print(json.load(sys.stdin)["stage_root"])')"
"$PY" - "$STAGE" "$RUN_ID" "$MAX_TOKENS" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
run_id = sys.argv[2]
max_tokens = int(sys.argv[3])
if not (4096 <= max_tokens <= 32768):
    raise ValueError('PAINTER_MAX_TOKENS must be 4096..32768')
path = root / 'config.json'
config = json.loads(path.read_text())
config['benchmark'] = 'quality-curriculum-' + run_id
config['tracks']['quality'].update(max_turns=4, max_tokens=max_tokens, episode_timeout_seconds=2400)
config.update(concurrency=1, temperature=0.5, renderer_timeout=240,
              request_timeout_seconds=900, max_retries=0)
path.write_text(json.dumps(config, indent=2, sort_keys=True) + '\n')
prompt = root / 'prompt.txt'
source_path = root / 'restored-source.json'
source = json.loads(source_path.read_text())
source['probe'] = {'reference_id': 'simple-cup', 'comparison': 'Flash underpaint canary at turn 4',
                   'run_id': run_id, 'max_turns': 4, 'max_completion_tokens': max_tokens,
                   'request_timeout_seconds': 900, 'renderer_timeout_seconds': 240,
                   'prompt_sha256': hashlib.sha256(prompt.read_bytes()).hexdigest(),
                   'config_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
source_path.write_text(json.dumps(source, indent=2, sort_keys=True) + '\n')
PY

(cd "$STAGE" && env -u AI_GATEWAY_API_KEY -u HF_TOKEN pnpm install --frozen-lockfile --ignore-scripts --silent)
"$PY" "$BENCH/run.py" --root "$STAGE" --dry-run --track quality --limit-episodes 1 >/dev/null
"$PY" "$BENCH/run.py" --root "$STAGE" --run --track quality --limit-episodes 1 \
  --renderer "$ROOT/painter/vendor/integrations/watercolour/renderer.py" \
  --renderer-python "$PY" --browser-path "$PLAYWRIGHT_BROWSERS_PATH" --run-as-user painter
"$PY" "$BENCH/cloud/publish_results.py" --root "$STAGE" --run-id "$RUN_ID"
echo 'Pro cup canary published; visual admission remains pending.' >&2
