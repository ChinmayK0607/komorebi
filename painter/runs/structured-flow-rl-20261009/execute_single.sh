#!/usr/bin/env bash
set -Eeuo pipefail
RUN="${PAINTER_RUN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)}"
[[ $# == 3 ]] || { echo 'usage: execute_single.sh RENTAL_START_EPOCH PRICE_USD_PER_HOUR MAX_GPU_USD' >&2; exit 2; }
exec bash "$RUN/execute_pipeline.sh" "$@"
