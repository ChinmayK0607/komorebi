#!/usr/bin/env bash
set -Eeuo pipefail
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
BASE="$(cd "$RUN/../text-sft512-consistent-20260930" && pwd -P)"
source "$RUN/runtime_env.sh"
validate_structured_bindings
export PAINTER_RUN_ROOT="$RUN" PAINTER_RUN_NAME=structured-flow-rl-20261009
export PAINTER_EVAL_MANIFEST="$RUN/curriculum/matched16-manifest.json"
export PAINTER_EVAL_EXPECTED_COUNT=16 PAINTER_EVAL_SCOPE=all
export PAINTER_JUDGE_MODEL=xiaomi/mimo-v2.6-flash PAINTER_JUDGE_EFFORT=medium
export PAINTER_JUDGE_FORMAT=identity-joint-side-schema-v2-generic-v1
exec bash "$BASE/finalize.sh" "$@"
