#!/usr/bin/env bash
set -Eeuo pipefail

# Independent step-0 policy sample on the frozen held-out prompts. The fixed
# anchors themselves are never reused as candidate paintings for this score.
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PY="${PRIME_ROOT:-/root/painter/prime}/.venv/bin/python"
MANIFEST="${PAINTER_VALIDATION_MANIFEST:-$RUN/anchors-frozen/validation-manifest.json}"
POOL="${PAINTER_CONTROL_POOL:-$RUN/prepared/validation-control-pool.json}"
OUT="${PAINTER_CONTROL_OUT:-$RUN/validation/sft-step512-control}"
KEY="${PAINTER_GATEWAY_ENV_FILE:-/root/painter/private/gateway.env}"
[[ -f "$MANIFEST" ]] || { echo 'frozen validation manifest missing' >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo 'step-0 evaluation already exists' >&2; exit 1; }
"$PY" - "$MANIFEST" "$POOL" <<'PY'
import json,sys
from pathlib import Path
source,target=map(Path,sys.argv[1:]);rows=json.loads(source.read_text())["cases"]
if not rows or len({row["id"] for row in rows})!=len(rows):
    raise SystemExit("independent step-0 control needs distinct held-out anchors")
target.write_text(json.dumps({"cases":rows},indent=2,sort_keys=True)+"\n")
PY
PAINTER_POOL_PATH="$POOL" PAINTER_EVAL_OUT="$OUT/eval" \
  PAINTER_EVAL_SEED=918210 PAINTER_BUILD_ANCHORS=0 \
  bash "$RUN/render_baseline.sh"
PYTHONPATH="$RUN/../../.." "$PY" "$RUN/score_validation.py" \
  --manifest "$MANIFEST" --eval-root "$OUT/eval" \
  --output "$OUT/comparison" --key-file "$KEY" \
  --protocol "${PAINTER_JUDGE_PROTOCOL:-text-paint-sft-pairwise-v1}"
