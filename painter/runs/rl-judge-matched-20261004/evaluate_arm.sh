#!/usr/bin/env bash
set -Eeuo pipefail

RUN="${PAINTER_RUN_ROOT:?arm_env must set PAINTER_RUN_ROOT}"
REPO="$(cd "$RUN/../../.." && pwd -P)"
BASE="${PAINTER_BASE_RUN:-$REPO/painter/runs/text-sft512-consistent-20260930}"
STEP="${1:?usage: evaluate_arm.sh STEP [balanced16] [--preflight]}"
SCOPE="${2:-balanced16}"
MODE="${3:-}"
[[ "$STEP" == 12 && "$SCOPE" == balanced16 ]] || {
  echo 'matched primary evaluation is fixed to step 12 and balanced16' >&2; exit 2;
}
[[ -z "$MODE" || "$MODE" == --preflight ]] || { echo 'invalid evaluation mode' >&2; exit 2; }
export PAINTER_EVAL_MANIFEST="$RUN/anchors-frozen/validation-matched16.json"
export PAINTER_EVAL_EXPECTED_COUNT=16 PAINTER_EVAL_SCOPE=balanced16 PAINTER_EVAL_SEED=918210
exec bash "$BASE/evaluate.sh" "$STEP" balanced16 "$MODE"
