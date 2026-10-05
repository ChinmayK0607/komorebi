#!/usr/bin/env bash
set -Eeuo pipefail

RUN="${PAINTER_RUN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)}"
REPO="$(cd "$RUN/../../.." && pwd -P)"
NODE_ROOT="${PAINTER_NODE_ROOT:-/root/painter}"
source "$RUN/runtime_env.sh"
validate_recovery_bindings
PY="$PRIME_ROOT/.venv/bin/python"
[[ "$(uname -s)" == Linux ]] || { echo 'Linux GPU node required for the RL process' >&2; exit 1; }
[[ -x "$PY" ]] || { echo "pinned Prime Python is unavailable: $PY" >&2; exit 1; }
validate_recovery_profile_environment "$PY"

exec bash "$REPO/painter/policy_runner.sh" "$@"
