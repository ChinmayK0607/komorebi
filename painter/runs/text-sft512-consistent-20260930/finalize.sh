#!/usr/bin/env bash
set -Eeuo pipefail

STEP="${1:?usage: finalize.sh STEP [--score]}"
MODE="${2:-}"
[[ "$STEP" =~ ^[1-9][0-9]*$ && ( -z "$MODE" || "$MODE" == --score ) ]] || {
  echo 'usage: finalize.sh STEP [--score]' >&2; exit 2;
}
RUN="${PAINTER_RUN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)}"
REPO="$(cd "$RUN/../../.." && pwd -P)"
NODE_ROOT="${PAINTER_NODE_ROOT:-/root/painter}"
PRIME_ROOT="${PRIME_ROOT:-$NODE_ROOT/prime}"
NAME="${PAINTER_RUN_NAME:-text-sft512-consistent-20260930}"
EVAL_SCOPE="${PAINTER_EVAL_SCOPE:-all}"
[[ "$EVAL_SCOPE" == all || "$EVAL_SCOPE" == balanced16 ]] || {
  echo 'PAINTER_EVAL_SCOPE must be all or balanced16' >&2; exit 2;
}
STATE="$RUN/supervisor-state.json"
GEN="$RUN/validation/step-$STEP-$EVAL_SCOPE/generation-receipt.json"
ADAPTER="$RUN/async-output/$NAME/artifacts/adapters/step_$STEP"
MANIFEST="$RUN/anchors-frozen/validation-manifest.json"
if [[ "$EVAL_SCOPE" == balanced16 ]]; then
  MANIFEST="$RUN/evaluation/matched16-manifest.json"
fi
EVAL_ROOT="$RUN/validation/step-$STEP-$EVAL_SCOPE/eval"
OUT="$RUN/validation/step-$STEP-$EVAL_SCOPE/comparison"
KEY="${PAINTER_GATEWAY_ENV_FILE:-$NODE_ROOT/private/gateway.env}"
JUDGE_PROTOCOL="${PAINTER_JUDGE_PROTOCOL:-text-paint-sft-pairwise-consistent-v1}"
JUDGE_FORMAT="${PAINTER_JUDGE_FORMAT:-identity-joint-side-schema-v1}"
JUDGE_MODEL="${PAINTER_JUDGE_MODEL:-xiaomi/mimo-v2.6-flash}"

[[ -f "$STATE" ]] || { echo 'finite supervisor state is missing' >&2; exit 1; }
STATE_PY="${PRIME_ROOT}/.venv/bin/python"
[[ -x "$STATE_PY" ]] || STATE_PY=python3
"$STATE_PY" - "$STATE" <<'PY'
import json, sys
from pathlib import Path
value = json.loads(Path(sys.argv[1]).read_text())
if value.get('status') not in {'stopped', 'completed'}:
    raise SystemExit('training supervisor is still active or has no terminal state')
PY
[[ -f "$ADAPTER/adapter_model.safetensors" && -f "$ADAPTER/hf-upload.json" ]] || {
  echo 'final checkpoint or upload receipt is missing' >&2; exit 1;
}
[[ -f "$GEN" ]] || { echo 'generation-only evaluation receipt is missing; run evaluate.sh first' >&2; exit 1; }

PY="${PRIME_ROOT}/.venv/bin/python"
[[ -x "$PY" ]] || PY=python3
"$PY" - "$RUN" "$ADAPTER" "$STEP" <<'PY'
import hashlib, json, sys
from pathlib import Path
run, adapter, step = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
upload = json.loads((adapter / 'hf-upload.json').read_text())
if upload.get('verified') is not True or upload.get('private') is not False:
    raise SystemExit('final checkpoint is not a verified public upload')
expected = upload.get('files', {}).get('adapter_model.safetensors', {}).get('sha256')
actual = hashlib.sha256((adapter / 'adapter_model.safetensors').read_bytes()).hexdigest()
if not isinstance(expected, str) or expected != actual:
    raise SystemExit('final checkpoint adapter hash does not match public receipt')
if upload.get('path') and upload['path'] != f'{run.name}/step_{step}':
    raise SystemExit('public checkpoint path does not match requested run and step')
PY

if [[ "$MODE" == --score ]]; then
  [[ -r "$KEY" && ! -L "$KEY" && "$(stat -c %a "$KEY")" == 600 ]] || {
    echo 'private Gateway key file missing or insecure' >&2; exit 1;
  }
  [[ -d "$EVAL_ROOT" && -f "$MANIFEST" ]] || { echo 'evaluation traces or manifest missing' >&2; exit 1; }
  mkdir -p "$OUT"
  PAINTER_JUDGE_PROTOCOL="$JUDGE_PROTOCOL" PAINTER_JUDGE_FORMAT="$JUDGE_FORMAT" PAINTER_JUDGE_MODEL="$JUDGE_MODEL" \
  PYTHONPATH="$REPO/painter:$RUN" "$PY" "$REPO/painter/runs/text-pairwise-rl-20260928/score_validation.py" \
    --manifest "$MANIFEST" --eval-root "$EVAL_ROOT" --output "$OUT" --key-file "$KEY" \
    --protocol "$JUDGE_PROTOCOL"
fi

"$PY" - "$RUN" "$STEP" "$EVAL_SCOPE" "$MODE" "$JUDGE_PROTOCOL" "$JUDGE_FORMAT" "$JUDGE_MODEL" <<'PY'
import hashlib, json, sys, time
from pathlib import Path
run, step, eval_scope, mode = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
judge_protocol, judge_format, judge_model = sys.argv[5:8]
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
generation = json.loads((run / f'validation/step-{step}-{eval_scope}/generation-receipt.json').read_text())
if generation.get('scoring_protocol') != judge_protocol or generation.get('matched_control') != 'sft-step512':
    raise SystemExit('generation receipt is not bound to the consistent SFT512 comparison')
comparison = run / f'validation/step-{step}-{eval_scope}/comparison/summary.json'
selection = {}
selection_path = run / 'local-evidence/latest-public-checkpoint.json'
if selection_path.is_file():
    try:
        selection = json.loads(selection_path.read_text())
    except (OSError, json.JSONDecodeError):
        raise SystemExit('checkpoint selection receipt is malformed')
report = {
    'schema': 'text-paint-consistent-finalizer-v2',
    'run_name': run.name,
    'step': step,
    'generation_receipt': str(run / f'validation/step-{step}-{eval_scope}/generation-receipt.json'),
    'comparison_path': str(comparison) if comparison.is_file() else None,
    'scoring_protocol': judge_protocol,
    'judge_format': judge_format,
    'judge_model': judge_model,
    'matched_control': 'sft-step512',
    'checkpoint_selection': selection.get('selection'),
    'primary_quality_status': (selection.get('selection') or {}).get('primary_quality_status'),
    'scored_locally': mode == '--score',
    'created_epoch': time.time(),
    'source_identity_sha256': sha(run / 'source-identity.json'),
}
(run / 'finalizer-receipt.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
print(json.dumps(report, sort_keys=True))
PY
echo "TEXT_SFT512_CONSISTENT_FINALIZED: step=$STEP public receipt verified; node cleanup remains operator-owned"
