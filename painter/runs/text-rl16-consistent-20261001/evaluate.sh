#!/usr/bin/env bash
set -Eeuo pipefail

RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
BASE="$(cd "$RUN/../text-sft512-consistent-20260930" && pwd -P)"
export PAINTER_RUN_ROOT="$RUN" PAINTER_RUN_NAME=text-rl16-consistent-20261001
export PAINTER_RENDERER_PATH="${PAINTER_RENDERER_PATH:-$RUN/vendor/integrations/watercolour/renderer.py}"
export PAINTER_RENDERER_PYTHON_PATH="${PAINTER_RENDERER_PYTHON_PATH:-$RUN/renderer-env/bin/python}"
export PAINTER_REQUIRE_VALID_FINAL_ACTION=1 PAINTER_JS_SYNTAX_PREFLIGHT=1
export PAINTER_JUDGE_FORMAT="${PAINTER_JUDGE_FORMAT:-identity-joint-side-schema-v1}"
exec bash "$BASE/evaluate.sh" "$@"
