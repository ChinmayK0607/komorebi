#!/usr/bin/env bash
# One-shot, foreground preparation for staged-family-balanced-20261010.
# It never launches Prime, starts a server, trains, generates model output, or
# schedules work.  All mutable evidence is placed below --node-root.
set -Eeuo pipefail
RUN_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
exec python3 "$RUN_ROOT/bootstrap.py" "$@"
