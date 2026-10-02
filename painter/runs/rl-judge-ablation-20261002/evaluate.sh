#!/usr/bin/env bash
set -Eeuo pipefail
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
STEP="${2:?usage: evaluate.sh ARM STEP [balanced16|all] [--preflight]}"
ARM="${1:?usage: evaluate.sh ARM STEP [balanced16|all] [--preflight]}"
# The finite pipeline exports its declared scope before invoking the base
# entrypoint.  This keeps the base execute/finalize pair on one matching
# output directory even though execute_pipeline.sh supplies the literal
# ``all`` argument for its historical interface.
SCOPE="${PAINTER_EVAL_SCOPE:-${3:-balanced16}}"
MODE="${4:-}"
[[ "$SCOPE" == balanced16 || "$SCOPE" == all ]] || { echo 'scope must be balanced16 or all' >&2; exit 2; }
[[ -z "$MODE" || "$MODE" == --preflight ]] || { echo 'invalid evaluation mode' >&2; exit 2; }
source "$RUN/arm_env.sh" "$ARM"
BASE="${PAINTER_BASE_RUN:-$(cd "$RUN/../text-sft512-consistent-20260930" && pwd -P)}"
if [[ "$SCOPE" == balanced16 ]]; then
  export PAINTER_EVAL_MANIFEST="$PAINTER_RUN_ROOT/evaluation/matched16-manifest.json"
  export PAINTER_EVAL_EXPECTED_COUNT=16
else
  export PAINTER_EVAL_MANIFEST="$PAINTER_RUN_ROOT/anchors-frozen/validation-manifest.json"
  export PAINTER_EVAL_EXPECTED_COUNT=47
fi
export PAINTER_EVAL_SEED=918210
export PAINTER_EVAL_SCOPE="$SCOPE"
exec bash "$BASE/evaluate.sh" "$STEP" "$SCOPE" "$MODE"
