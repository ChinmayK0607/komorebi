#!/usr/bin/env bash
set -Eeuo pipefail

STEP="${1:?usage: evaluate.sh STEP [balanced16|all]}"
SCOPE="${2:-balanced16}"
[[ "$STEP" =~ ^[1-9][0-9]*$ && ( "$SCOPE" == balanced16 || "$SCOPE" == all ) ]] || {
  echo 'invalid step/scope' >&2; exit 2;
}
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ "$SCOPE" == balanced16 ]]; then
  MANIFEST="$RUN/balanced16-validation-manifest.json"
  COUNT=16
else
  MANIFEST="$RUN/anchors-all/validation-manifest.json"
  COUNT=all
fi
export PAINTER_RL_RUN_NAME=text-grounded-pilot-20260930
export PAINTER_VALIDATION_MANIFEST="$MANIFEST"
export PAINTER_CHECKPOINT_OUT="$RUN/validation/step-$STEP-$SCOPE"
export PAINTER_JUDGE_PROTOCOL=text-paint-sft-pairwise-grounded-v1
bash "$RUN/evaluate_checkpoint.sh" "$STEP" "$COUNT"
