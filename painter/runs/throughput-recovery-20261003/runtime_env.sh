#!/usr/bin/env bash
set -Eeuo pipefail

# One finite, single-judge recovery arm.  Keep these bindings in one file so
# preparation, preflight, training, evaluation, and the parent controller use
# the same identity.
PAINTER_SINGLE_RUN_ROOT="${PAINTER_RUN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)}"
PAINTER_SINGLE_REPO_ROOT="$(cd "$PAINTER_SINGLE_RUN_ROOT/../../.." && pwd -P)"
PAINTER_SINGLE_NODE_ROOT="${PAINTER_NODE_ROOT:-/root/painter}"
PAINTER_SINGLE_PRIME_ROOT="${PRIME_ROOT:-$PAINTER_SINGLE_NODE_ROOT/prime}"

export PAINTER_RUN_ROOT="$PAINTER_SINGLE_RUN_ROOT"
export PAINTER_REPO_ROOT="$PAINTER_SINGLE_REPO_ROOT"
export PAINTER_RUN_NAME="${PAINTER_RUN_NAME:-throughput-recovery-20261003}"
export PAINTER_EXPERIMENT_NAME="${PAINTER_EXPERIMENT_NAME:-throughput-recovery-flash-20261003}"
export PAINTER_RL_RUN_NAME="$PAINTER_RUN_NAME"
export PAINTER_RL_MODE=recovered_policy

export PAINTER_BASE_MODEL_REVISION="${PAINTER_BASE_MODEL_REVISION:-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
export PAINTER_POLICY_MODE=recovered_policy
export PAINTER_POLICY_REVISION="${PAINTER_POLICY_REVISION:-ac4d3eda8e5925e2d23992cfdc1a63620ebde573}"
export PAINTER_POLICY_PREFIX="${PAINTER_POLICY_PREFIX:-text-pairwise-scaled-20260929/step_16}"
export PAINTER_POLICY_SHA256="${PAINTER_POLICY_SHA256:-e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c}"
export PAINTER_FROZEN_SFT_MODEL_REVISION="${PAINTER_FROZEN_SFT_MODEL_REVISION:-adf687e5b321514eda7be6eab050434fdc8e44ff}"
export PAINTER_FROZEN_SFT_ADAPTER_SHA256="${PAINTER_FROZEN_SFT_ADAPTER_SHA256:-bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740}"

export PAINTER_JUDGE_PROTOCOL="${PAINTER_JUDGE_PROTOCOL:-text-paint-sft-pairwise-consistent-v1}"
export PAINTER_JUDGE_FORMAT="${PAINTER_JUDGE_FORMAT:-identity-joint-side-schema-v2-generic-v1}"
export PAINTER_JUDGE_MODEL="${PAINTER_JUDGE_MODEL:-xiaomi/mimo-v2.6-flash}"
export PAINTER_JUDGE_EFFORT="${PAINTER_JUDGE_EFFORT:-medium}"

export PAINTER_GROUP_SIZE=4
export PAINTER_INITIAL_INFLIGHT=32
export PAINTER_MAX_INFLIGHT=32
export PAINTER_RL_MAX_STEPS=4
export PAINTER_RL_CHECKPOINT_INTERVAL=1
export PAINTER_CHECKPOINT_INTERVAL=1
export ADAPTER_ONLY_CHECKPOINTS=1
export CAPACITY_STATE_COMPLETE=1
export PAINTER_ACTIVATION_CPU_OFFLOAD=1
export PAINTER_REQUIRE_VALID_FINAL_ACTION=1
export PAINTER_JS_SYNTAX_PREFLIGHT=1
export PAINTER_P5_BRUSH_API_REFERENCE_V2=1
export PAINTER_INSPECT_CHOICE_FEEDBACK=1
export PAINTER_EVAL_SCOPE="${PAINTER_EVAL_SCOPE:-balanced16}"
export PAINTER_EVAL_EXPECTED_COUNT=47
export PAINTER_EVAL_PREFERRED_STEP=4

# The finite controller leaves 1,800 seconds for a real startup plus matched-
# 16 generation pass, then 360 seconds for archive publication, 180 seconds
# for parent-owned release, 120 seconds of margin, and 60 seconds of base
# pipeline safety time.
export PAINTER_EVAL_TIMEOUT_SECONDS="${PAINTER_EVAL_TIMEOUT_SECONDS:-1800}"
export PAINTER_ARCHIVE_TIMEOUT_SECONDS="${PAINTER_ARCHIVE_TIMEOUT_SECONDS:-360}"
export PAINTER_PARENT_RELEASE_RESERVE_SECONDS="${PAINTER_PARENT_RELEASE_RESERVE_SECONDS:-180}"
export PAINTER_COMPLETION_MARGIN_SECONDS="${PAINTER_COMPLETION_MARGIN_SECONDS:-120}"
export PAINTER_RESERVE_SECONDS="${PAINTER_RESERVE_SECONDS:-2520}"
export PAINTER_INITIAL_METRIC_TIMEOUT_SECONDS="${PAINTER_INITIAL_METRIC_TIMEOUT_SECONDS:-3600}"
export PAINTER_INFERENCE_READY_TIMEOUT_SECONDS="${PAINTER_INFERENCE_READY_TIMEOUT_SECONDS:-1200}"

export PAINTER_GPUS_PER_NODE="${PAINTER_GPUS_PER_NODE:-2}"
export PAINTER_NUM_TRAIN_GPUS="${PAINTER_NUM_TRAIN_GPUS:-1}"
export PAINTER_NUM_INFER_GPUS="${PAINTER_NUM_INFER_GPUS:-$((PAINTER_GPUS_PER_NODE - PAINTER_NUM_TRAIN_GPUS))}"
export PAINTER_RENDER_WORKERS="${PAINTER_RENDER_WORKERS:-8}"

export PAINTER_SFT_ADAPTER_PATH_FILE="${PAINTER_SFT_ADAPTER_PATH_FILE:-$PAINTER_SINGLE_RUN_ROOT/policy-adapter-path.txt}"
export PAINTER_FROZEN_SFT_ADAPTER_PATH_FILE="${PAINTER_FROZEN_SFT_ADAPTER_PATH_FILE:-$PAINTER_SINGLE_RUN_ROOT/../photo-curriculum-rl-20260922/initial-adapter-path.txt}"
export PAINTER_POLICY_DOWNLOAD_SCRIPT="${PAINTER_POLICY_DOWNLOAD_SCRIPT:-$PAINTER_SINGLE_REPO_ROOT/painter/runs/text-pairwise-rl-20260928/download_public_adapter.py}"
export PAINTER_POLICY_DOWNLOAD_RECEIPT="${PAINTER_POLICY_DOWNLOAD_RECEIPT:-$PAINTER_SINGLE_RUN_ROOT/policy-adapter-receipt.json}"
export PAINTER_BOOTSTRAP_RUN="${PAINTER_BOOTSTRAP_RUN:-$PAINTER_SINGLE_REPO_ROOT/painter/runs/text-pairwise-rl-20260928}"
export MODEL_DIR="${MODEL_DIR:-$PAINTER_SINGLE_NODE_ROOT/model-cache/qwen/$PAINTER_BASE_MODEL_REVISION}"
export PRIME_ROOT="$PAINTER_SINGLE_PRIME_ROOT"
export PAINTER_GATEWAY_ENV_FILE="${PAINTER_GATEWAY_ENV_FILE:-$PAINTER_SINGLE_NODE_ROOT/private/gateway.env}"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$PAINTER_SINGLE_NODE_ROOT/private/hf-token}"
export PYTHONPATH="$PAINTER_SINGLE_REPO_ROOT:$PAINTER_SINGLE_REPO_ROOT/painter:$PAINTER_SINGLE_RUN_ROOT${PYTHONPATH:+:$PYTHONPATH}"

validate_recovery_bindings() {
  [[ "$PAINTER_RUN_NAME" == throughput-recovery-20261003 ]] || {
    echo 'PAINTER_RUN_NAME is fixed to throughput-recovery-20261003' >&2
    return 1
  }
  [[ "$PAINTER_EXPERIMENT_NAME" == throughput-recovery-flash-20261003 ]] || {
    echo 'PAINTER_EXPERIMENT_NAME is fixed to throughput-recovery-flash-20261003' >&2
    return 1
  }
  [[ "$PAINTER_RL_MODE" == recovered_policy && "$PAINTER_POLICY_MODE" == recovered_policy ]] || {
    echo 'single-arm recovery requires the verified recovered-policy initializer' >&2
    return 1
  }
  [[ "$PAINTER_POLICY_REVISION" == ac4d3eda8e5925e2d23992cfdc1a63620ebde573 ]] || {
    echo 'single-arm recovery initializer revision is not the pinned RL16 value' >&2
    return 1
  }
  [[ "$PAINTER_POLICY_PREFIX" == text-pairwise-scaled-20260929/step_16 ]] || {
    echo 'single-arm recovery initializer prefix is not the pinned RL16 value' >&2
    return 1
  }
  [[ "$PAINTER_POLICY_SHA256" == e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c ]] || {
    echo 'single-arm recovery initializer hash is not the pinned RL16 value' >&2
    return 1
  }
  [[ "$PAINTER_JUDGE_MODEL" == xiaomi/mimo-v2.6-flash && "$PAINTER_JUDGE_EFFORT" == medium ]] || {
    echo 'single-arm recovery requires exactly the Flash judge at medium effort' >&2
    return 1
  }
  [[ "$PAINTER_JUDGE_FORMAT" == identity-joint-side-schema-v2-generic-v1 ]] || {
    echo 'single-arm recovery requires the v2 generic rubric format' >&2
    return 1
  }
  [[ "$PAINTER_JUDGE_PROTOCOL" == text-paint-sft-pairwise-consistent-v1 ]] || {
    echo 'single-arm recovery reward protocol is not the matched consistent protocol' >&2
    return 1
  }
  [[ "$PAINTER_GROUP_SIZE" == 4 && "$PAINTER_INITIAL_INFLIGHT" == 32 && "$PAINTER_MAX_INFLIGHT" == 32 ]] || {
    echo 'single-arm recovery requires group4 and 32 in-flight episodes' >&2
    return 1
  }
  [[ "$PAINTER_RL_MAX_STEPS" == 4 && "$PAINTER_RL_CHECKPOINT_INTERVAL" == 1 && "$PAINTER_CHECKPOINT_INTERVAL" == 1 ]] || {
    echo 'single-arm recovery requires four updates and interval-one checkpoints' >&2
    return 1
  }
  [[ "$ADAPTER_ONLY_CHECKPOINTS" == 1 && "$PAINTER_ACTIVATION_CPU_OFFLOAD" == 1 ]] || {
    echo 'adapter-only checkpoints and CPU activation offload must be enabled' >&2
    return 1
  }
  [[ "$PAINTER_REQUIRE_VALID_FINAL_ACTION" == 1 && "$PAINTER_P5_BRUSH_API_REFERENCE_V2" == 1 ]] || {
    echo 'strict final-action credit and opt-in API v2 must remain enabled' >&2
    return 1
  }
  [[ "$PAINTER_EVAL_SCOPE" == balanced16 && "$PAINTER_EVAL_EXPECTED_COUNT" == 47 ]] || {
    echo 'single-arm recovery evaluation must use the matched 16-case panel from the immutable 47-case manifest' >&2
    return 1
  }
  [[ "$PAINTER_GPUS_PER_NODE" =~ ^[1-9][0-9]*$ && "$PAINTER_NUM_TRAIN_GPUS" =~ ^[1-9][0-9]*$ && "$PAINTER_NUM_INFER_GPUS" =~ ^[1-9][0-9]*$ ]] || {
    echo 'GPU topology values must be positive integers' >&2
    return 1
  }
  (( PAINTER_NUM_TRAIN_GPUS + PAINTER_NUM_INFER_GPUS == PAINTER_GPUS_PER_NODE )) || {
    echo 'GPU topology must partition the node into one trainer and the remaining inference GPUs' >&2
    return 1
  }
  (( PAINTER_NUM_TRAIN_GPUS == 1 )) || {
    echo 'single-arm recovery requires exactly one trainer GPU' >&2
    return 1
  }
  (( PAINTER_GPUS_PER_NODE <= 4 )) || {
    echo 'single-arm recovery topology is limited to the supplied 2- or 4-GPU node' >&2
    return 1
  }
  [[ "$PAINTER_RESERVE_SECONDS" =~ ^[0-9]+$ && "$PAINTER_RESERVE_SECONDS" -ge 400 ]] || {
    echo 'finite completion reserve must be at least 400 seconds' >&2
    return 1
  }
  (( PAINTER_RESERVE_SECONDS >= PAINTER_EVAL_TIMEOUT_SECONDS + PAINTER_ARCHIVE_TIMEOUT_SECONDS + PAINTER_PARENT_RELEASE_RESERVE_SECONDS + PAINTER_COMPLETION_MARGIN_SECONDS + 60 )) || {
    echo 'finite completion reserve does not cover evaluation, archive, release, margin, and safety time' >&2
    return 1
  }
}

validate_recovery_profile_environment() {
  local python_bin="${1:-${PAINTER_RUNTIME_VALIDATE_PYTHON:-python3}}"
  "$python_bin" - <<'PY'
import os
from painter.throughput_recovery_profile import validate_runtime_environment
validate_runtime_environment(os.environ)
PY
}
