#!/usr/bin/env bash
set -Eeuo pipefail
ARM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd "$ARM_ROOT/../rl-judge-matched-20261004" && pwd -P)"
ARM_ID="$(basename "$ARM_ROOT")"
ARM_ID="${ARM_ID#rl-judge-matched-20261004-}"
source "$ROOT/arm_env.sh" "$ARM_ID"
exec bash "$ROOT/evaluate_arm.sh" "$@"
