#!/usr/bin/env bash
set -Eeuo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/prepare_arm.sh" "$@"
