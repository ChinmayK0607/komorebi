#!/usr/bin/env bash
set -Eeuo pipefail

RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
BASE="$(cd "$RUN/../text-sft512-consistent-20260930" && pwd -P)"
export PAINTER_RUN_ROOT="$RUN" PAINTER_RUN_NAME=text-rl16-consistent-20261001
export PAINTER_POLICY_MODE=recovered_policy
export PAINTER_POLICY_REVISION=ac4d3eda8e5925e2d23992cfdc1a63620ebde573
export PAINTER_POLICY_PREFIX=text-pairwise-scaled-20260929/step_16
export PAINTER_POLICY_SHA256=e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c
export PAINTER_JUDGE_FORMAT="${PAINTER_JUDGE_FORMAT:-identity-joint-side-schema-v1}"
export PAINTER_SFT_ADAPTER_PATH_FILE="$RUN/policy-adapter-path.txt"
export PAINTER_FROZEN_SFT_ADAPTER_PATH_FILE="${PAINTER_FROZEN_SFT_ADAPTER_PATH_FILE:-$RUN/../photo-curriculum-rl-20260922/initial-adapter-path.txt}"
export PAINTER_RENDERER_PATH="${PAINTER_RENDERER_PATH:-$RUN/vendor/integrations/watercolour/renderer.py}"
export PAINTER_RENDERER_PYTHON_PATH="${PAINTER_RENDERER_PYTHON_PATH:-$RUN/renderer-env/bin/python}"
exec bash "$BASE/run.sh" "$@"
