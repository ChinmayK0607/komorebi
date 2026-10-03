#!/usr/bin/env bash
set -Eeuo pipefail
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd "$RUN/../../.." && pwd -P)"
BASE="${PAINTER_BASE_RUN:-$REPO/painter/runs/text-sft512-consistent-20260930}"
source "$RUN/runtime_env.sh"
export PAINTER_EVAL_SCOPE=balanced16 PAINTER_EVAL_EXPECTED_COUNT=47
export PAINTER_RENDERER_PATH="$RUN/runtime/watercolour-cancellable-centroid-v1/cancellable_watercolour_renderer.py"
export PAINTER_RENDERER_PYTHON_PATH="${PAINTER_RENDERER_PYTHON_PATH:-$RUN/renderer-env/bin/python}"
STEP="${1:?usage: evaluate.sh STEP [scope] [--preflight]}"
MODE="${3:-}"
[[ -z "$MODE" || "$MODE" == --preflight ]] || { echo 'usage: evaluate.sh STEP [scope] [--preflight]' >&2; exit 2; }
if [[ -n "$MODE" ]]; then
  exec bash "$BASE/evaluate.sh" "$STEP" balanced16 "$MODE"
fi
exec bash "$BASE/evaluate.sh" "$STEP" balanced16
