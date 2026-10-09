#!/usr/bin/env bash
set -Eeuo pipefail
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
BASE="$(cd "$RUN/../text-sft512-consistent-20260930" && pwd -P)"
source "$RUN/runtime_env.sh"
validate_structured_bindings
export PAINTER_RUN_ROOT="$RUN" PAINTER_RUN_NAME=diverse-flow-rl-20261009
export PAINTER_EVAL_MANIFEST="$RUN/curriculum/matched32-manifest.json"
export PAINTER_EVAL_EXPECTED_COUNT=32 PAINTER_EVAL_SCOPE=all
export PAINTER_JUDGE_MODEL=xiaomi/mimo-v2.6-flash PAINTER_JUDGE_EFFORT=medium
export PAINTER_JUDGE_FORMAT=identity-joint-side-schema-v2-generic-v1
if [[ "${PAINTER_EVALUATION_ONLY:-0}" == 1 ]]; then
  RECOVERY_PY="${PRIME_ROOT:-}/.venv/bin/python"
  [[ -x "$RECOVERY_PY" ]] || RECOVERY_PY=python3
  exec "$RECOVERY_PY" "$RUN/finalize_evaluation_recovery.py" "$@"
fi
exec bash "$BASE/finalize.sh" "$@"
