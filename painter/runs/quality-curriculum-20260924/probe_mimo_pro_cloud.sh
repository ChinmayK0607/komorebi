#!/usr/bin/env bash
# One bounded, photo-conditioned teacher probe after the first hard prefix.
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

STAGE="$($PY "$RUN/prepare_mimo_cloud.py" --tier hard --model xiaomi/mimo-v2.6-pro | $PY -c 'import json,sys; print(json.load(sys.stdin)["stage_root"])')"
"$PY" - "$STAGE" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
config_path = root / 'config.json'
config = json.loads(config_path.read_text())
config['benchmark'] = 'quality-curriculum-pro-cake-bounded-20260924'
config['tracks']['quality'].update(max_turns=6, max_tokens=32768)
config['concurrency'] = 1
config['renderer_timeout'] = 300
config['request_timeout_seconds'] = 900
config['max_retries']=0
config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + '\n')
prompt_path = root / 'prompt.txt'
prompt_path.write_text(prompt_path.read_text() + '\nValid brush names are pen, rotring, 2B, HB, 2H, cpencil, pastel, crayon, charcoal, spray, and marker. Do not call brush.init() or invent brush names. Keep loops finite and geometry bounded so a 600x600 canvas renders promptly.\n')
source_path = root / 'restored-source.json'
source = json.loads(source_path.read_text())
source['probe'] = {'reference_id': 'coco128-000000000092', 'max_turns': 6,
                   'max_completion_tokens': 32768, 'request_timeout_seconds': 900,
                   'renderer_timeout_seconds': 300,
                   'prompt_sha256': hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
                   'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest()}
source_path.write_text(json.dumps(source, indent=2, sort_keys=True) + '\n')
PY
(cd "$STAGE" && env -u AI_GATEWAY_API_KEY -u HF_TOKEN pnpm install --frozen-lockfile --ignore-scripts --silent)
"$PY" "$BENCH/run.py" --root "$STAGE" --dry-run --track quality --start-episode 2 --limit-episodes 1 >/dev/null
"$PY" "$BENCH/run.py" --root "$STAGE" --run --track quality --start-episode 2 --limit-episodes 1 \
  --renderer "$ROOT/painter/vendor/integrations/watercolour/renderer.py" \
  --renderer-python "$PY" --browser-path "$PLAYWRIGHT_BROWSERS_PATH" --run-as-user painter
"$PY" "$BENCH/cloud/publish_results.py" --root "$STAGE" --run-id mimo-pro-cake-bounded-20260924
echo 'Cake probe published; visual admission is still pending.' >&2
