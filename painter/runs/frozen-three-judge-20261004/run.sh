#!/usr/bin/env bash
set -Eeuo pipefail

RUN_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
EVIDENCE_DIR=${PAINTER_FROZEN_THREE_EVIDENCE_DIR:-"$RUN_DIR/local-evidence"}
PANEL=${PAINTER_FROZEN_THREE_PANEL:-"$EVIDENCE_DIR/pairs.json"}
MANIFEST=${PAINTER_FROZEN_THREE_MANIFEST:-"$EVIDENCE_DIR/input-manifest.json"}
OUTPUT=${PAINTER_FROZEN_THREE_OUTPUT:-"$EVIDENCE_DIR/results"}
OLD_GALLERY="$RUN_DIR/../text-grounded-pilot-20260930/local-evidence/gallery/index.html"
PRO_GALLERY="$RUN_DIR/../text-rl16-pro-20261002/local-evidence/review/gallery.json"

usage() {
  cat >&2 <<'EOF'
usage:
  run.sh --prepare-only [--old-gallery PATH] [--pro-gallery PATH] [--evidence-dir DIR]
  run.sh --run --key-file PATH [--max-paid-calls 288] [--concurrency N]

Preparation is offline. Running requires an existing owner-only local key file
and performs finite paid judge requests only; it never starts a provider job.
EOF
}

MODE=
KEY_FILE=
MAX_PAID_CALLS=288
CONCURRENCY=6
while (($#)); do
  case "$1" in
    --prepare-only|--run) MODE=${1#--};;
    --key-file) (($# >= 2)) || { usage; exit 2; }; KEY_FILE=$2; shift;;
    --max-paid-calls) (($# >= 2)) || { usage; exit 2; }; MAX_PAID_CALLS=$2; shift;;
    --concurrency) (($# >= 2)) || { usage; exit 2; }; CONCURRENCY=$2; shift;;
    --old-gallery) (($# >= 2)) || { usage; exit 2; }; OLD_GALLERY=$2; shift;;
    --pro-gallery) (($# >= 2)) || { usage; exit 2; }; PRO_GALLERY=$2; shift;;
    --evidence-dir) (($# >= 2)) || { usage; exit 2; }; EVIDENCE_DIR=$2; PANEL="$EVIDENCE_DIR/pairs.json"; MANIFEST="$EVIDENCE_DIR/input-manifest.json"; OUTPUT="$EVIDENCE_DIR/results"; shift;;
    *) usage; exit 2;;
  esac
  shift
done

case "$MODE" in
  prepare-only)
    mkdir -p -- "$EVIDENCE_DIR"
    python3 "$RUN_DIR/prepare_panel.py" --old-gallery "$OLD_GALLERY" \
      --pro-gallery "$PRO_GALLERY" --output "$PANEL"
    python3 "$RUN_DIR/frozen_three_judge.py" --prepare-only --panel "$PANEL" \
      --repeat-count 2 --output "$MANIFEST"
    ;;
  run)
    [[ -n "$KEY_FILE" ]] || { echo "--key-file is required for --run" >&2; exit 2; }
    [[ -f "$MANIFEST" && ! -L "$MANIFEST" ]] || { echo "prepared manifest is missing: $MANIFEST" >&2; exit 2; }
    exec python3 "$RUN_DIR/frozen_three_judge.py" --run --manifest "$MANIFEST" \
      --output "$OUTPUT" --key-file "$KEY_FILE" --max-paid-calls "$MAX_PAID_CALLS" \
      --concurrency "$CONCURRENCY"
    ;;
  *) usage; exit 2;;
esac
