#!/usr/bin/env bash
set -Eeuo pipefail
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ARM="${1:?usage: finalize.sh ARM STEP [--score]}"
STEP="${2:?usage: finalize.sh ARM STEP [--score]}"
MODE="${3:-}"
[[ -z "$MODE" || "$MODE" == --score ]] || { echo 'usage: finalize.sh ARM STEP [--score]' >&2; exit 2; }
source "$RUN/arm_env.sh" "$ARM"
BASE="${PAINTER_BASE_RUN:-$(cd "$RUN/../text-sft512-consistent-20260930" && pwd -P)}"
export PAINTER_EVAL_SCOPE="${PAINTER_EVAL_SCOPE:-balanced16}"
exec bash "$BASE/finalize.sh" "$STEP" "$MODE"
