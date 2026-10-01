#!/usr/bin/env bash
set -Eeuo pipefail

RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
BASE="$(cd "$RUN/../text-sft512-consistent-20260930" && pwd -P)"
export PAINTER_RUN_ROOT="$RUN" PAINTER_RUN_NAME=text-rl16-pro-20261002
export PAINTER_JUDGE_PROTOCOL=text-paint-sft-pairwise-consistent-v1
export PAINTER_JUDGE_FORMAT=identity-joint-side-schema-v2-pro-v1
export PAINTER_JUDGE_MODEL=xiaomi/mimo-v2.6-pro
export PAINTER_RENDER_WORKERS="${PAINTER_RENDER_WORKERS:-2}"
export PAINTER_P5_BRUSH_API_REFERENCE=1 PAINTER_INSPECT_CHOICE_FEEDBACK=1
export PAINTER_REQUIRE_VALID_FINAL_ACTION=1 PAINTER_JS_SYNTAX_PREFLIGHT=1
export PAINTER_RENDERER_PATH="${PAINTER_RENDERER_PATH:-$RUN/vendor/integrations/watercolour/renderer.py}"
export PAINTER_RENDERER_PYTHON_PATH="${PAINTER_RENDERER_PYTHON_PATH:-$RUN/renderer-env/bin/python}"
exec bash "$BASE/evaluate.sh" "$@"
