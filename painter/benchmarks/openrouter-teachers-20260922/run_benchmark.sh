#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

# Keep the existing node launcher as the default.  The opt-in local path is
# deliberately selected in this small shim so an ordinary invocation cannot
# silently bypass the Linux renderer contract.
local_mode=false
remaining=()
for argument in "$@"; do
    if [[ "$argument" == "--local" ]]; then
        local_mode=true
    else
        remaining+=("$argument")
    fi
done
if [[ "$local_mode" == true ]]; then
    exec "$SCRIPT_DIR/run_local.sh" "${remaining[@]}"
fi
exec python3 "$SCRIPT_DIR/run_on_node.py" "$@"
