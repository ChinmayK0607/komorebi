#!/usr/bin/env bash
set -Eeuo pipefail

RUN="${PAINTER_RUN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)}"
REPO="$(cd "$RUN/../../.." && pwd -P)"
BASE="${PAINTER_BASE_RUN:-$REPO/painter/runs/text-sft512-consistent-20260930}"
source "$RUN/runtime_env.sh"
validate_recovery_bindings
[[ $# == 3 ]] || { echo 'usage: execute_single.sh RENTAL_START_EPOCH PRICE_USD_PER_HOUR MAX_GPU_USD' >&2; exit 2; }
[[ -f "$BASE/execute_pipeline.sh" ]] || { echo "base finite pipeline missing: $BASE/execute_pipeline.sh" >&2; exit 1; }

# The base pipeline owns preparation, finite supervision, preferred verified
# checkpoint selection, matched-panel evaluation, finalization, and the
# weight-free public safe archive.  Local wrappers in this run root keep every
# path parameterized to this one arm.  finite_supervisor.py rewrites the base
# invocation's historical 50-update argument to the fixed pilot value 4.
export PAINTER_RUN_ROOT="$RUN" PAINTER_RUN_NAME=throughput-recovery-20261003
export PAINTER_EXPERIMENT_NAME=throughput-recovery-flash-20261003
export PAINTER_EVAL_SCOPE=balanced16 PAINTER_EVAL_PREFERRED_STEP=4
export PAINTER_EVAL_EXPECTED_COUNT=47
export PAINTER_RL_MAX_STEPS=4 PAINTER_MAX_UPDATES=4
export PAINTER_RL_CHECKPOINT_INTERVAL=1 PAINTER_CHECKPOINT_INTERVAL=1
export ADAPTER_ONLY_CHECKPOINTS=1 PAINTER_ACTIVATION_CPU_OFFLOAD=1
export PAINTER_REQUIRE_VALID_FINAL_ACTION=1 PAINTER_P5_BRUSH_API_REFERENCE_V2=1
export PAINTER_JUDGE_MODEL=xiaomi/mimo-v2.6-flash PAINTER_JUDGE_EFFORT=medium
export PAINTER_JUDGE_FORMAT=identity-joint-side-schema-v2-generic-v1
export PAINTER_JUDGE_PROTOCOL=text-paint-sft-pairwise-consistent-v1
export PAINTER_RENDERER_PATH="$RUN/runtime/watercolour-cancellable-centroid-v1/cancellable_watercolour_renderer.py"
export PAINTER_RENDERER_PYTHON_PATH="${PAINTER_RENDERER_PYTHON_PATH:-$RUN/renderer-env/bin/python}"

exec bash "$BASE/execute_pipeline.sh" "$@"
