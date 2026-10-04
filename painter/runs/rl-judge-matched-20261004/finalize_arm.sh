#!/usr/bin/env bash
set -Eeuo pipefail

RUN="${PAINTER_RUN_ROOT:?arm_env must set PAINTER_RUN_ROOT}"
REPO="$(cd "$RUN/../../.." && pwd -P)"
BASE="${PAINTER_BASE_RUN:-$REPO/painter/runs/text-sft512-consistent-20260930}"
STEP="${1:?usage: finalize_arm.sh STEP}"
[[ "$STEP" == 12 ]] || { echo 'matched primary finalization is fixed to step 12' >&2; exit 2; }
[[ "${2:-}" == '' || "${2:-}" == --score ]] || { echo 'invalid finalization mode' >&2; exit 2; }
export PAINTER_EVAL_SCOPE=balanced16
exec bash "$BASE/finalize.sh" "$STEP" "${2:-}"
