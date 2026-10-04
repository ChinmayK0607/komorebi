#!/usr/bin/env bash
set -Eeuo pipefail

ARM_ID="${1:?arm id required: flash|pro|luna}"
case "$ARM_ID" in
  flash) ARM_MODEL=xiaomi/mimo-v2.6-flash ;;
  pro) ARM_MODEL=xiaomi/mimo-v2.6-pro ;;
  luna) ARM_MODEL=openai/gpt-6-luna ;;
  *) echo "unknown arm: $ARM_ID" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ARM_ROOT="${PAINTER_MATCHED_ARM_ROOT_BASE:-$ROOT/..}/rl-judge-matched-20261004-$ARM_ID"
source "$ROOT/runtime_env.sh"
export PAINTER_ARM_ID="$ARM_ID"
export PAINTER_ARM_ROOT="$ARM_ROOT"
export PAINTER_RUN_ROOT="$ARM_ROOT"
export PAINTER_RUN_NAME="rl-judge-matched-20261004-$ARM_ID"
export PAINTER_RL_RUN_NAME="$PAINTER_RUN_NAME"
export PAINTER_EXPERIMENT_NAME="rl-judge-matched-20261004-$ARM_ID"
export PAINTER_JUDGE_MODEL="$ARM_MODEL"
export PAINTER_OUTPUT_ROOT=rl-rollouts
export PAINTER_PUBLIC_CHECKPOINT_PREFIX="text-rl-judge-matched-20261004-$ARM_ID"
export PAINTER_POLICY_MODE=recovered_policy
export PAINTER_POLICY_REVISION=ac4d3eda8e5925e2d23992cfdc1a63620ebde573
export PAINTER_POLICY_PREFIX=text-pairwise-scaled-20260929/step_16
export PAINTER_POLICY_SHA256=e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c
export PAINTER_POLICY_DOWNLOAD_SCRIPT="$PAINTER_REPO_ROOT/painter/runs/text-pairwise-rl-20260928/download_public_adapter.py"
export PAINTER_POLICY_DOWNLOAD_RECEIPT="$ARM_ROOT/policy-adapter-receipt.json"
export PAINTER_SFT_ADAPTER_PATH_FILE="$ARM_ROOT/policy-adapter-path.txt"
export PAINTER_FROZEN_SFT_ADAPTER_PATH_FILE="$ARM_ROOT/frozen-sft-adapter-path.txt"
export PAINTER_RL_MANIFEST="$ARM_ROOT/anchors-frozen/train-manifest.json"
export PAINTER_EVAL_MANIFEST="$ARM_ROOT/anchors-frozen/validation-matched16.json"
export PAINTER_RENDERER_PATH="${PAINTER_RENDERER_PATH:-$ARM_ROOT/runtime/watercolour-cancellable-centroid-v1/cancellable_watercolour_renderer.py}"
export PAINTER_RENDERER_PYTHON_PATH="${PAINTER_RENDERER_PYTHON_PATH:-$ARM_ROOT/renderer-env/bin/python}"
export PAINTER_FROZEN_SFT_ADAPTER_PATH_FILE="$ARM_ROOT/frozen-sft-adapter-path.txt"
export MODEL_DIR="${MODEL_DIR:-$PAINTER_NODE_ROOT/model-cache/qwen/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
export PAINTER_GATEWAY_ENV_FILE="${PAINTER_GATEWAY_ENV_FILE:-$PAINTER_NODE_ROOT/private/gateway.env}"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$PAINTER_NODE_ROOT/private/hf-token}"
export PYTHONPATH="$PAINTER_REPO_ROOT:$PAINTER_REPO_ROOT/painter:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

validate_matched_bindings
