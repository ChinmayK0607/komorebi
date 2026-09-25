#!/usr/bin/env bash
# Four finite, diverse easy-scene MiMo Pro episodes on Codex Cloud CPU.
set -Eeuo pipefail

ROOT="$(git rev-parse --show-toplevel)"
RUN="$ROOT/painter/runs/quality-curriculum-20260924"
BENCH="$ROOT/painter/benchmarks/openrouter-teachers-20260922"
RUNTIME="$ROOT/.painter-cloud-runtime"
PY="$RUNTIME/renderer-env/bin/python"
RUN_ID="${PAINTER_RUN_ID:-mimo-pro-easy-wave1-20260925}"
STAGE_NAME="${PAINTER_STAGE_NAME:-easy-pro-wave1-20260925}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] || { echo 'invalid PAINTER_RUN_ID' >&2; exit 2; }
[[ "$STAGE_NAME" =~ ^[a-z0-9][a-z0-9._-]{0,63}$ ]] || { echo 'invalid PAINTER_STAGE_NAME' >&2; exit 2; }

export PATH="/usr/bin:/bin:$PATH"
export NODE_USE_ENV_PROXY=1
export PLAYWRIGHT_BROWSERS_PATH="$RUNTIME/browsers"
export HF_HUB_DISABLE_XET=1
[[ -x "$PY" && -d "$PLAYWRIGHT_BROWSERS_PATH" ]] || { echo 'cloud renderer setup missing' >&2; exit 2; }
[[ -n "${AI_GATEWAY_API_KEY:-}" && -n "${HF_TOKEN:-}" ]] || { echo 'Gateway and HF environment credentials are required' >&2; exit 2; }
[[ ! -e "$ROOT/painter/collected/quality-curriculum-20260924/mimo-cloud/$STAGE_NAME" ]] || {
  echo 'stage already exists; use a new PAINTER_STAGE_NAME to avoid resuming old evidence' >&2; exit 2;
}

STAGE="$($PY "$RUN/prepare_mimo_cloud.py" --tier easy --model xiaomi/mimo-v2.6-pro --stage-name "$STAGE_NAME" | $PY -c 'import json,sys; print(json.load(sys.stdin)["stage_root"])')"
"$PY" - "$STAGE" "$RUN_ID" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
run_id = sys.argv[2]
selected_ids = ('simple-pear', 'simple-poppy', 'simple-sailboat', 'simple-cottage')
refs_path = root / 'refs.json'
refs = json.loads(refs_path.read_text())
by_id = {row['id']: row for row in refs['references']}
if len(by_id) != 7 or not all(ident in by_id for ident in selected_ids):
    raise ValueError('expected seven staged easy references and four selected IDs')
refs['references'] = [by_id[ident] for ident in selected_ids]
refs['count'] = len(selected_ids)
refs_path.write_text(json.dumps(refs, indent=2, sort_keys=True) + '\n')
config_path = root / 'config.json'
config = json.loads(config_path.read_text())
config['benchmark'] = 'quality-curriculum-' + run_id
config['tracks']['quality'].update(max_turns=4, max_tokens=20480, episode_timeout_seconds=2400)
config.update(concurrency=2, render_concurrency=1, temperature=0.5,
              renderer_timeout=240, request_timeout_seconds=900, max_retries=0)
config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + '\n')
source_path = root / 'restored-source.json'
source = json.loads(source_path.read_text())
source['reference_ids'] = list(selected_ids)
source['wave'] = {
    'hypothesis': 'user-liked cup quality transfers to four different easy scene types',
    'matched_cup_run': 'mimo-pro-cup-repair-v2',
    'run_id': run_id,
    'max_turns': 4, 'max_completion_tokens': 20480,
    'concurrency': 2, 'render_concurrency': 1, 'max_retries': 0,
    'prompt_sha256': hashlib.sha256((root / 'prompt.txt').read_bytes()).hexdigest(),
    'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
    'refs_sha256': hashlib.sha256(refs_path.read_bytes()).hexdigest(),
}
source_path.write_text(json.dumps(source, indent=2, sort_keys=True) + '\n')
PY

publish() {
  [[ -d "$STAGE/episodes" ]] || return 0
  "$PY" "$BENCH/cloud/publish_results.py" --root "$STAGE" --run-id "$RUN_ID"
}
cleanup() {
  local status=$?
  if (( status != 0 )) && [[ -d "$STAGE/episodes" ]]; then
    echo 'Wave stopped; attempting to preserve partial evidence on public HF.' >&2
    publish || echo 'Partial publication failed; inspect the cloud task before retrying.' >&2
  fi
}
trap cleanup EXIT
trap 'exit 143' TERM INT

(cd "$STAGE" && env -u AI_GATEWAY_API_KEY -u HF_TOKEN pnpm install --frozen-lockfile --ignore-scripts --silent)
"$PY" "$BENCH/run.py" --root "$STAGE" --dry-run --track quality --limit-episodes 4 >/dev/null
"$PY" "$BENCH/run.py" --root "$STAGE" --run --track quality --limit-episodes 4 \
  --renderer "$ROOT/painter/vendor/integrations/watercolour/renderer.py" \
  --renderer-python "$PY" --browser-path "$PLAYWRIGHT_BROWSERS_PATH" --run-as-user painter
publish
echo 'Four easy-scene candidate episodes published; retain all turns for visual selection.' >&2
