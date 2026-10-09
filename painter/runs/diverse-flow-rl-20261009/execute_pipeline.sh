#!/usr/bin/env bash
set -Eeuo pipefail
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
BASE="$(cd "$RUN/../text-sft512-consistent-20260930" && pwd -P)"
# Bind the registered finite target before sourcing runtime_env.  The
# controller may carry an older inherited value from a previous 8-update
# launch; this run owns the explicit 20-update target end to end.
export PAINTER_RL_MAX_STEPS=20 PAINTER_MAX_UPDATES=20 PAINTER_CHECKPOINT_INTERVAL=4 PAINTER_EVAL_PREFERRED_STEP=20
source "$RUN/runtime_env.sh"
validate_structured_bindings
export PAINTER_RUN_ROOT="$RUN" PAINTER_RUN_NAME=diverse-flow-rl-20261009
export PAINTER_EXPECTED_COUNT=96 PAINTER_EVAL_EXPECTED_COUNT=32 PAINTER_EVAL_SCOPE=all
# prepare_node.sh owns the matched pre-warmup panel, the required four-update
# SFT warmup, and the separate post-warmup 32-case evaluation before the base
# finite supervisor is allowed to enter RL.
export PAINTER_RUN_POST_WARMUP_EVAL="${PAINTER_RUN_POST_WARMUP_EVAL:-1}"
export PAINTER_RL_MAX_STEPS=20 PAINTER_MAX_UPDATES=20 PAINTER_CHECKPOINT_INTERVAL=4 PAINTER_EVAL_PREFERRED_STEP=20
export PAINTER_RESERVE_SECONDS="${PAINTER_RESERVE_SECONDS:-3600}"
exec bash "$BASE/execute_pipeline.sh" "$@"
