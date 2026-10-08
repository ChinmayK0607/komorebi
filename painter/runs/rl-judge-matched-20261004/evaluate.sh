#!/usr/bin/env bash
set -Eeuo pipefail

STEP="${1:?usage: evaluate.sh STEP [balanced16|all] [--preflight]}"
SCOPE="${2:-balanced16}"
MODE="${3:-}"
[[ "$STEP" == 12 ]] || {
  echo 'matched primary evaluation is fixed to step 12' >&2; exit 2;
}
[[ "$SCOPE" == balanced16 || "$SCOPE" == all ]] || {
  echo 'matched primary evaluation scope is fixed to balanced16' >&2; exit 2;
}
[[ -z "$MODE" || "$MODE" == --preflight ]] || {
  echo 'invalid evaluation mode' >&2; exit 2;
}

# The shared executor keeps its historical ``all`` argument, but this matched
# campaign has only one frozen 16-case panel.  Normalize before the arm
# evaluator sees the request so the expanded validation manifest is never
# selected.
[[ "$SCOPE" == all ]] && SCOPE=balanced16

ARM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd "$ARM_ROOT/../rl-judge-matched-20261004" && pwd -P)"
ARM_ID="$(basename "$ARM_ROOT")"
ARM_ID="${ARM_ID#rl-judge-matched-20261004-}"
source "$ROOT/arm_env.sh" "$ARM_ID"
exec bash "$ROOT/evaluate_arm.sh" "$STEP" "$SCOPE" "$MODE"
