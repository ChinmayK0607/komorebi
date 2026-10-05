#!/usr/bin/env bash
set -Eeuo pipefail

RUN="${PAINTER_RUN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)}"
REPO="$(cd "$RUN/../../.." && pwd -P)"
NODE_ROOT="${PAINTER_NODE_ROOT:-/root/painter}"
source "$RUN/runtime_env.sh"
validate_matched_bindings
PY="$PRIME_ROOT/.venv/bin/python"
validate_matched_config "$PY" "${PAINTER_RL_CONFIG:-$RUN/native.toml}" "$RUN/rl-config-receipt.json"

exec bash "$REPO/painter/policy_runner.sh" "$@"
