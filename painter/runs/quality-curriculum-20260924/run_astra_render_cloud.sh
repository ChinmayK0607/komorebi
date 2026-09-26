#!/usr/bin/env bash
# Finite CPU rendering of a public Astra candidate shard; no paid model calls.
set -Eeuo pipefail

ROOT="$(git rev-parse --show-toplevel)"
RUN="$ROOT/painter/runs/quality-curriculum-20260924"
BENCH="$ROOT/painter/benchmarks/openrouter-teachers-20260922"
RUNTIME="$ROOT/.painter-cloud-runtime"
SHARD="${1:?pass easy-a, hard-a or hard-b}"
case "$SHARD" in easy-a|hard-a|hard-b) ;; *) echo 'unexpected shard' >&2; exit 2;; esac
[[ -n "${HF_TOKEN:-}" ]] || { echo 'HF_TOKEN is required to publish public render evidence' >&2; exit 2; }
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
  --output-dir "$RUNTIME/astra-render-smoke-$SHARD" >"$RUNTIME/astra-render-smoke-$SHARD.json"
if "$PY" "$RUN/render_astra_cloud.py" "$SHARD" --timeout 600; then
  echo "All candidate programs rendered; quality review remains pending" >&2
else
  echo "Some candidate programs failed rendering; publishing complete evidence anyway" >&2
fi
"$PY" "$BENCH/cloud/publish_results.py" \
  --root "$ROOT/painter/collected/quality-curriculum-20260924/astra-high-wave1-cloud/$SHARD" \
  --run-id "astra-high-wave1-$SHARD-20260926"
