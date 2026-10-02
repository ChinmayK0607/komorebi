#!/usr/bin/env bash
set -Eeuo pipefail
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ARM="${1:?usage: run.sh ARM [--preflight] }"
shift
source "$RUN/arm_env.sh" "$ARM"
BASE="${PAINTER_BASE_RUN:-$(cd "$RUN/../text-sft512-consistent-20260930" && pwd -P)}"
[[ $# == 0 || ( $# == 1 && "$1" == --preflight ) ]] || { echo 'usage: run.sh ARM [--preflight]' >&2; exit 2; }
exec bash "$BASE/run.sh" "$@"
