#!/usr/bin/env bash
set -Eeuo pipefail
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd "$RUN/../../.." && pwd -P)"
BASE="${PAINTER_BASE_RUN:-$REPO/painter/runs/text-sft512-consistent-20260930}"
source "$RUN/runtime_env.sh"
export PAINTER_EVAL_SCOPE=balanced16 PAINTER_EVAL_EXPECTED_COUNT=47
export PAINTER_RENDERER_PATH="$RUN/runtime/watercolour-cancellable-centroid-v1/cancellable_watercolour_renderer.py"
export PAINTER_RENDERER_PYTHON_PATH="${PAINTER_RENDERER_PYTHON_PATH:-$RUN/renderer-env/bin/python}"
exec bash "$BASE/finalize.sh" "$@"
