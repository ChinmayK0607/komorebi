#!/usr/bin/env bash
set -Eeuo pipefail
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ARM="${1:?usage: execute_pipeline.sh ARM RENTAL_START_EPOCH PRICE_USD_PER_HOUR MAX_GPU_USD}"
shift
[[ $# == 3 ]] || { echo 'usage: execute_pipeline.sh ARM RENTAL_START_EPOCH PRICE_USD_PER_HOUR MAX_GPU_USD' >&2; exit 2; }
source "$RUN/arm_env.sh" "$ARM"
BASE="${PAINTER_BASE_RUN:-$(cd "$RUN/../text-sft512-consistent-20260930" && pwd -P)}"
# The base pipeline invokes evaluate.sh with its historical ``all`` argument.
# Keep the finite ablation on its predeclared 16-case panel unless the
# operator explicitly exports PAINTER_EVAL_SCOPE=all with a larger reserve.
export PAINTER_EVAL_SCOPE="${PAINTER_EVAL_SCOPE:-balanced16}"
exec bash "$BASE/execute_pipeline.sh" "$@"
