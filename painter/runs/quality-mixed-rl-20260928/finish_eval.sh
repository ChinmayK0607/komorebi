#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd "$ROOT/../../.." && pwd -P)"
OLD="$ROOT/../quality-oss-rl-20260927"
STEP="${RL_STEP:-12}"
OUT="$ROOT/matched-eval-step-$STEP"
[[ -f "$OUT/completion.json" ]] || { echo 'Matched evaluation has not completed' >&2; exit 1; }
python3 - "$OUT/completion.json" <<'PY'
import json,sys
if json.load(open(sys.argv[1]))['status']!='passed':
    raise SystemExit('Matched evaluation did not pass')
PY
python3 "$ROOT/build_gallery.py" --manifest "$OUT/cases.json" \
  --rollout-root "$REPO/painter/evaluation-rollouts" \
  --outdir "$OUT/review" --step "$STEP"
PYTHONPATH="$OLD${PYTHONPATH:+:$PYTHONPATH}" python3 "$ROOT/score_gallery.py" \
  --gallery "$OUT/review" \
  --key-file "${GATEWAY_KEY_FILE:-/root/painter/private/gateway.env}" \
  --output "$OUT/review/judge.json"
