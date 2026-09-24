#!/usr/bin/env bash
# Finite, resumable-in-place Codex Cloud candidate generation. No GPU rental.
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
(cd "$STAGE" && env -u AI_GATEWAY_API_KEY -u HF_TOKEN pnpm install --frozen-lockfile --ignore-scripts --silent)
if [[ "$TIER" == easy ]]; then LIMITS=(2 4 7); else LIMITS=(2 5 10); fi

publish() {
  local count="$1"
  [[ -d "$STAGE/episodes" ]] || return 0
  "$PY" "$BENCH/cloud/publish_results.py" --root "$STAGE" --run-id "$RUN_PREFIX-n$count"
}

for limit in "${LIMITS[@]}"; do
  echo "Starting candidate episode prefix $limit/${LIMITS[-1]} for $TIER $MODEL" >&2
  "$PY" "$BENCH/run.py" --root "$STAGE" --dry-run --track quality --limit-episodes "$limit" >/dev/null
  "$PY" "$BENCH/run.py" --root "$STAGE" --run --track quality --limit-episodes "$limit" \
    --renderer "$ROOT/painter/vendor/integrations/watercolour/renderer.py" \
    --renderer-python "$PY" --browser-path "$PLAYWRIGHT_BROWSERS_PATH" \
    --run-as-user painter
  publish "$limit"
done
echo "Completed finite candidate generation for $TIER $MODEL; visual admission is still pending." >&2
