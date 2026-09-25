#!/usr/bin/env bash
# Finite CPU teacher campaign: remaining two simple subjects and nine new COCO train photos.
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
CURRENT_STAGE=""
CURRENT_RUN=""
cleanup() {
  local status=$?
  if (( status != 0 )) && [[ -n "$CURRENT_STAGE" && -d "$CURRENT_STAGE/episodes" ]]; then
    "$PY" "$BENCH/cloud/publish_results.py" --root "$CURRENT_STAGE" --run-id "$CURRENT_RUN" || true
  fi
}
trap cleanup EXIT
trap 'exit 143' TERM INT

run_tier() {
  local tier="$1" suffix="$2" count="$3"
  local run_id="mimo-pro-diverse-${suffix}-20260925"
  local stage_name="diverse-pro-${suffix}-20260925"
  local stage="$ROOT/painter/collected/quality-curriculum-20260924/mimo-cloud/$stage_name"
  CURRENT_STAGE="$stage"
  CURRENT_RUN="$run_id"
  [[ ! -e "$stage" ]] || { echo "stage already exists: $stage_name" >&2; exit 2; }
  "$PY" "$RUN/prepare_mimo_cloud.py" --tier "$tier" --model xiaomi/mimo-v2.6-pro --stage-name "$stage_name"
  "$PY" - "$stage" "$tier" "$run_id" <<'PY'
import hashlib, json, pathlib, sys
root, tier, run_id = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
if tier == 'easy':
    selected_ids = ('simple-orange', 'simple-fern')
else:
    selected_ids = (
        'coco128-000000000025', 'coco128-000000000034',
        'coco128-000000000263',
        'coco128-000000000394', 'coco128-000000000400',
        'coco128-000000000471', 'coco128-000000000491',
        'coco128-000000000605', 'coco128-000000000650',
    )
refs_path = root / 'refs.json'
refs = json.loads(refs_path.read_text())
by_id = {row['id']: row for row in refs['references']}
if len(by_id) != (7 if tier == 'easy' else 10) or not all(ident in by_id for ident in selected_ids):
    raise ValueError('staged references differ from reviewed manifest')
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
    'hypothesis': 'approved MiMo Pro teacher quality extends from five simple scenes and the existing cake probe to two unseen simple and nine new diverse train photos',
    'matched_operating_baseline': 'mimo-pro-easy-wave1-20260925',
    'run_id': run_id, 'max_turns': 4, 'max_completion_tokens': 20480,
    'concurrency': 2, 'render_concurrency': 1, 'max_retries': 0,
    'prompt_sha256': hashlib.sha256((root / 'prompt.txt').read_bytes()).hexdigest(),
    'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
    'refs_sha256': hashlib.sha256(refs_path.read_bytes()).hexdigest(),
}
source_path.write_text(json.dumps(source, indent=2, sort_keys=True) + '\n')
PY
  (cd "$stage" && env -u AI_GATEWAY_API_KEY -u HF_TOKEN pnpm install --frozen-lockfile --ignore-scripts --silent)
  "$PY" "$BENCH/run.py" --root "$stage" --dry-run --track quality --limit-episodes "$count" >/dev/null
  "$PY" "$BENCH/run.py" --root "$stage" --run --track quality --limit-episodes "$count" \
    --renderer "$ROOT/painter/vendor/integrations/watercolour/renderer.py" \
    --renderer-python "$PY" --browser-path "$PLAYWRIGHT_BROWSERS_PATH" --run-as-user painter
  "$PY" "$BENCH/cloud/publish_results.py" --root "$stage" --run-id "$run_id"
  CURRENT_STAGE=""
  CURRENT_RUN=""
}

run_tier easy easy 2
run_tier hard photo 9
echo 'Eleven finite candidate episodes published; review per-turn canvases before SFT admission.' >&2
