#!/usr/bin/env bash
# Finite four-scene MiMo Pro brush-feel comparison on Codex Cloud CPU.
set -Eeuo pipefail
ROOT="$(git rev-parse --show-toplevel)"
RUN="$ROOT/painter/runs/quality-curriculum-20260926"
BENCH="$ROOT/painter/benchmarks/openrouter-teachers-20260922"
RUNTIME="$ROOT/.painter-cloud-runtime"
[[ "$(uname -s)" == Linux ]] || { echo 'Codex Cloud Linux required' >&2; exit 2; }
[[ -n "${AI_GATEWAY_API_KEY:-}" && -n "${HF_TOKEN:-}" ]] || { echo 'Gateway and HF credentials required in Cloud environment' >&2; exit 2; }
export PATH="/usr/bin:/bin:$PATH"
export NODE_USE_ENV_PROXY=1 PLAYWRIGHT_BROWSERS_PATH="$RUNTIME/browsers" HF_HUB_DISABLE_XET=1
if [[ ! -x "$RUNTIME/renderer-env/bin/python" || ! -d "$PLAYWRIGHT_BROWSERS_PATH" ]]; then
  bash "$BENCH/cloud/setup.sh"
fi
PY="$RUNTIME/renderer-env/bin/python"
(cd "$BENCH" && env -u AI_GATEWAY_API_KEY -u HF_TOKEN pnpm install --frozen-lockfile --ignore-scripts --silent)
STAGE="$ROOT/painter/collected/quality-curriculum-20260926/rendered-cloud/brush-polish-four-20260927"
"$PY" "$RUN/brush_polish.py" stage
ln -s "$BENCH/node_modules" "$STAGE/node_modules"
"$PY" "$BENCH/run.py" --root "$STAGE" --dry-run --track quality > "$STAGE/dry-run.json"
if ! "$PY" "$BENCH/run.py" --root "$STAGE" --run --track quality \
    --renderer "$ROOT/painter/vendor/integrations/watercolour/renderer.py" \
    --renderer-python "$PY" --browser-path "$PLAYWRIGHT_BROWSERS_PATH" --run-as-user painter; then
  if [[ -d "$STAGE/episodes" ]]; then
    "$PY" "$BENCH/cloud/publish_results.py" --root "$STAGE" --run-id brush-polish-four-20260927-partial --split || true
  fi
  exit 1
fi
"$PY" "$BENCH/cloud/publish_results.py" --root "$STAGE" --run-id brush-polish-four-20260927 --split
echo 'Published brush-polish candidates; source-matched visual review remains pending.'
