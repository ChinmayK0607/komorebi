#!/usr/bin/env bash
set -Eeuo pipefail

RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
BASE="$(cd "$RUN/../text-sft512-consistent-20260930" && pwd -P)"
export PAINTER_RUN_ROOT="$RUN" PAINTER_RUN_NAME=text-rl16-pro-20261002
export PAINTER_JUDGE_PROTOCOL=text-paint-sft-pairwise-consistent-v1
export PAINTER_JUDGE_FORMAT=identity-joint-side-schema-v2-pro-v1
export PAINTER_JUDGE_MODEL=xiaomi/mimo-v2.6-pro
exec bash "$BASE/finalize.sh" "$@"
