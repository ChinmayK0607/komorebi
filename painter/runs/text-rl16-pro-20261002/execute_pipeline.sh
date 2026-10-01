#!/usr/bin/env bash
set -Eeuo pipefail

RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
BASE="$(cd "$RUN/../text-sft512-consistent-20260930" && pwd -P)"
export PAINTER_RUN_ROOT="$RUN" PAINTER_RUN_NAME=text-rl16-pro-20261002
export PAINTER_GROUP_SIZE=4 PAINTER_INITIAL_INFLIGHT=32 PAINTER_MAX_INFLIGHT=32
export PAINTER_POLICY_MODE=recovered_policy
export PAINTER_POLICY_REVISION=ac4d3eda8e5925e2d23992cfdc1a63620ebde573
export PAINTER_POLICY_PREFIX=text-pairwise-scaled-20260929/step_16
export PAINTER_POLICY_SHA256=e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c
export PAINTER_JUDGE_PROTOCOL=text-paint-sft-pairwise-consistent-v1
export PAINTER_JUDGE_FORMAT=identity-joint-side-schema-v2-pro-v1
export PAINTER_JUDGE_MODEL=xiaomi/mimo-v2.6-pro
export PAINTER_RENDER_WORKERS="${PAINTER_RENDER_WORKERS:-2}"
export PAINTER_P5_BRUSH_API_REFERENCE=1 PAINTER_INSPECT_CHOICE_FEEDBACK=1
export PAINTER_REQUIRE_VALID_FINAL_ACTION=1 PAINTER_JS_SYNTAX_PREFLIGHT=1
exec bash "$BASE/execute_pipeline.sh" "$@"
