#!/usr/bin/env bash
set -Eeuo pipefail
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
BASE="$(cd "$RUN/../text-sft512-consistent-20260930" && pwd -P)"
source "$RUN/runtime_env.sh"
validate_structured_bindings
export PAINTER_RUN_ROOT="$RUN" PAINTER_RUN_NAME=structured-flow-rl-20261009
export PAINTER_EXPECTED_COUNT=32 PAINTER_EVAL_EXPECTED_COUNT=16 PAINTER_EVAL_SCOPE=all
export PAINTER_RL_MAX_STEPS=8 PAINTER_MAX_UPDATES=8 PAINTER_CHECKPOINT_INTERVAL=4
export PAINTER_RESERVE_SECONDS="${PAINTER_RESERVE_SECONDS:-3600}"
exec bash "$BASE/execute_pipeline.sh" "$@"
