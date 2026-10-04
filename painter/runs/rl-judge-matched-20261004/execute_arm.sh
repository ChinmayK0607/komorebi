#!/usr/bin/env bash
set -Eeuo pipefail

RUN="${PAINTER_RUN_ROOT:?arm_env must set PAINTER_RUN_ROOT}"
REPO="$(cd "$RUN/../../.." && pwd -P)"
BASE="${PAINTER_BASE_RUN:-$REPO/painter/runs/text-sft512-consistent-20260930}"
[[ $# == 3 ]] || { echo 'usage: execute_arm.sh RENTAL_START_EPOCH PRICE_USD_PER_HOUR MAX_GPU_USD' >&2; exit 2; }
source "$RUN/runtime_env.sh"
validate_matched_bindings
apply_matched_resource_limits
[[ -f "$RUN/prepare_node.sh" && -f "$RUN/run.sh" && -f "$RUN/evaluate.sh" && -f "$RUN/finalize.sh" ]] || {
  echo 'materialized arm entrypoints are incomplete' >&2; exit 1;
}
export PAINTER_RUN_ROOT="$RUN" PAINTER_RL_MAX_STEPS=12 PAINTER_MAX_UPDATES=12
export PAINTER_EVAL_SCOPE=balanced16 PAINTER_EVAL_PREFERRED_STEP=12 PAINTER_EVAL_EXPECTED_COUNT=47
export PAINTER_RL_CHECKPOINT_INTERVAL=4 PAINTER_CHECKPOINT_INTERVAL=4 ADAPTER_ONLY_CHECKPOINTS=1
export PAINTER_ACTIVATION_CPU_OFFLOAD=1 PAINTER_REQUIRE_VALID_FINAL_ACTION=1
export PAINTER_P5_BRUSH_API_REFERENCE_V2=1 PAINTER_MAX_POLICY_AGE=1
exec bash "$BASE/execute_pipeline.sh" "$@"
