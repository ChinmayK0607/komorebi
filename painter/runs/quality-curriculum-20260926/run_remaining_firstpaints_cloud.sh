#!/usr/bin/env bash
# Finite, resumable Cloud CPU render group. Each batch is published separately.
set -Eeuo pipefail

ROOT="$(git rev-parse --show-toplevel)"
RUN="$ROOT/painter/runs/quality-curriculum-20260926"
BENCH="$ROOT/painter/benchmarks/openrouter-teachers-20260922"
RUNTIME="$ROOT/.painter-cloud-runtime"
GROUP="${1:?pass group ID from REMAINING_FIRSTPAINT_RENDER_PLAN.json}"
WORKERS="${2:-2}"
[[ "$GROUP" =~ ^group-[1-6]$ ]] || { echo 'unknown render group' >&2; exit 2; }
[[ "$WORKERS" =~ ^[1-8]$ ]] || { echo 'workers must be 1..8' >&2; exit 2; }
[[ -n "${HF_TOKEN:-}" ]] || { echo 'HF_TOKEN is required for result publication' >&2; exit 2; }

export PATH="/usr/bin:/bin:$PATH"
export PLAYWRIGHT_BROWSERS_PATH="$RUNTIME/browsers"
export HF_HUB_DISABLE_XET=1
if [[ ! -x "$RUNTIME/renderer-env/bin/python" || ! -d "$PLAYWRIGHT_BROWSERS_PATH" ]]; then
  bash "$BENCH/cloud/setup.sh"
fi
PY="$RUNTIME/renderer-env/bin/python"
"$PY" "$BENCH/renderer_smoke.py" \
  --renderer "$ROOT/painter/vendor/integrations/watercolour/renderer.py" \
  --renderer-python "$PY" --browser-path "$PLAYWRIGHT_BROWSERS_PATH" \
  --run-as-user painter --timeout 600 \
  --output-dir "$RUNTIME/teacher500-smoke-$GROUP" >"$RUNTIME/teacher500-smoke-$GROUP.json"

mapfile -t ROWS < <("$PY" - "$RUN/REMAINING_FIRSTPAINT_RENDER_PLAN.json" "$GROUP" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
for row in plan['groups'][sys.argv[2]]:
    print('\t'.join((row['batch'], row['run_id'], row['source_archive_sha256'])))
PY
)

failures=0
for ROW in "${ROWS[@]}"; do
  IFS=$'\t' read -r BATCH RUN_ID SOURCE_SHA <<< "$ROW"
  echo "BEGIN $GROUP $BATCH $RUN_ID" >&2
  # Existing public, hash-verified receipts make a repeated invocation cheap.
  if "$PY" - "$RUN_ID" <<'PY'
import json, sys
from urllib.error import HTTPError
from urllib.request import urlopen
run_id = sys.argv[1]
url = f'https://huggingface.co/datasets/CK0607/komorebi-painter-teachers/resolve/main/runs/{run_id}/receipt.json?download=true'
try:
    with urlopen(url, timeout=60) as response:
        receipt = json.load(response)
except HTTPError as exc:
    if exc.code == 404:
        sys.exit(1)
    raise
if receipt.get('run_id') != run_id or receipt.get('public_hash_verified') is not True:
    raise ValueError('remote result receipt is not verified')
print(f'Already published and hash-verified: {run_id}')
PY
  then
    continue
  fi
  if ! "$PY" "$RUN/render_teacher_batch_cloud.py" "$BATCH" "$RUN_ID" \
      --expected-archive-sha256 "$SOURCE_SHA" --timeout 600 --workers "$WORKERS"; then
    echo "RENDER FAILED $BATCH" >&2
    failures=$((failures + 1))
    continue
  fi
  if ! "$PY" "$BENCH/cloud/publish_results.py" \
      --root "$ROOT/painter/collected/quality-curriculum-20260926/rendered-cloud/$RUN_ID" \
      --run-id "$RUN_ID"; then
    echo "PUBLICATION FAILED $BATCH" >&2
    failures=$((failures + 1))
    continue
  fi
  echo "PUBLISHED $GROUP $BATCH $RUN_ID" >&2
done
echo "GROUP COMPLETE $GROUP failures=$failures" >&2
[[ "$failures" -eq 0 ]]
