#!/usr/bin/env bash
set -Eeuo pipefail

# Invoke once after the finite trainer exits. This does not schedule work.
STEP="${1:?usage: finalize_continuation.sh STEP [all|balanced16]}"
SCOPE="${2:-all}"
[[ "$STEP" =~ ^[1-9][0-9]*$ && ( "$SCOPE" == all || "$SCOPE" == balanced16 ) ]] || {
  echo 'step must be positive; scope must be all or balanced16' >&2; exit 2;
}
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
NAME=text-pairwise-continue-20260929
PY=/root/painter/prime/.venv/bin/python
ANCHORS="$RUN/continued-anchors/validation-manifest.json"
CONTROL="$RUN/validation/scaled-sft-control"
PRIOR="$RUN/prior-step16"
PRIOR_SUMMARY="$PRIOR/validation/scaled-step-16-all/comparison/summary.json"
PRIOR_IMAGES="$PRIOR/evidence/gallery/images"
KEY="${PAINTER_GATEWAY_ENV_FILE:-/root/painter/private/gateway.env}"
ADAPTER="$RUN/async-output/$NAME/artifacts/adapters/step_$STEP"
[[ -f "$ADAPTER/hf-upload.json" ]] || { echo 'public checkpoint receipt missing' >&2; exit 1; }
[[ -f "$CONTROL/validation-summary.json" && -d "$CONTROL/eval" ]] || {
  echo 'matched SFT control is not restored' >&2; exit 1;
}
[[ -f "$PRIOR_SUMMARY" && -d "$PRIOR_IMAGES" ]] || {
  echo 'public step-16 control is not restored' >&2; exit 1;
}
STREAM="$($PY - "$RUN" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1])))
from archive_continuation import ensure_stream
print(ensure_stream(Path(sys.argv[1])))
PY
)"
"$PY" "$RUN/plot_reward_curve.py" \
  --metrics "$RUN/async-output/$NAME/monitors/file/metrics.jsonl" \
  --traces "$STREAM" --output "$RUN/evidence/reward-curve"
"$PY" "$RUN/profile_continuation.py" \
  --metrics "$RUN/async-output/$NAME/monitors/file/metrics.jsonl" \
  --traces "$STREAM" --judges "$RUN/rl-rollouts-continued" \
  --trainer-log "$RUN/async-output/$NAME/logs/attempt_1/trainer.log" \
  --gpu-csv "$RUN/logs/gpu-telemetry-20260929T163811Z.csv" \
  --output "$RUN/evidence/profile-continuation.json"
START_ADAPTER="$($PY - "$RUN/continued-step16-download.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['adapter_path'])
PY
)"
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" "$RUN/verify_adapter_delta.py" \
  --start "$START_ADAPTER" --end "$ADAPTER" \
  --output "$RUN/evidence/adapter-delta.json"
"$PY" "$RUN/build_rollout_audit.py" \
  --stream "$STREAM" --run-root "$RUN" \
  --output "$RUN/evidence/rollout-audit"
if [[ "$SCOPE" == all ]]; then
  MANIFEST="$ANCHORS"
  BASELINE="$CONTROL/validation-summary.json"
  PRIOR_BASELINE="$PRIOR_SUMMARY"
else
  MANIFEST="$RUN/prepared/continued-validation-balanced16.json"
  BASELINE="$RUN/validation/$NAME-step-$STEP-balanced16/baseline-subset.json"
  PRIOR_BASELINE="$RUN/validation/$NAME-step-$STEP-balanced16/prior-subset.json"
  mkdir -p "$(dirname "$MANIFEST")" "$(dirname "$BASELINE")"
  "$PY" - "$ANCHORS" "$MANIFEST" <<'PY'
import json,sys
from pathlib import Path
source,target=map(Path,sys.argv[1:])
rows=json.loads(source.read_text())['cases']
foundation=[r for r in rows if r['family'] in {'simple','intermediate'}][:8]
complex_cases=[r for r in rows if r['family']=='complex'][:8]
chosen=foundation+complex_cases
if len(chosen)!=16 or len({r['id'] for r in chosen})!=16:
    raise SystemExit('cannot freeze balanced16 validation subset')
payload={'schema':'text-paint-continued-balanced16-v1','cases':chosen}
target.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY
  "$PY" "$RUN/control_sets.py" subset \
    --summary "$CONTROL/validation-summary.json" \
    --manifest "$MANIFEST" --output "$BASELINE"
  "$PY" "$RUN/control_sets.py" subset \
    --summary "$PRIOR_SUMMARY" \
    --manifest "$MANIFEST" --output "$PRIOR_BASELINE"
fi
OUT="$RUN/validation/$NAME-step-$STEP-$SCOPE"
env PAINTER_RL_RUN_NAME="$NAME" PAINTER_VALIDATION_MANIFEST="$MANIFEST" \
    PAINTER_CHECKPOINT_OUT="$OUT" \
    PAINTER_JUDGE_PROTOCOL=text-paint-sft-pairwise-margin-v1 \
    PAINTER_GATEWAY_ENV_FILE="$KEY" \
    bash "$RUN/evaluate_checkpoint.sh" "$STEP" all
"$PY" "$RUN/analyze_scaled.py" --baseline "$BASELINE" \
  --candidate "$OUT/comparison/summary.json" --output "$OUT/analysis.json"
"$PY" "$RUN/analyze_scaled.py" --baseline "$PRIOR_BASELINE" \
  --candidate "$OUT/comparison/summary.json" --output "$OUT/analysis-vs-step16.json"
if [[ "$SCOPE" == all ]]; then
  "$PY" "$RUN/score_direct_static.py" --manifest "$MANIFEST" \
    --baseline-label SFT --baseline-summary "$BASELINE" \
    --baseline-images "$PRIOR_IMAGES" --final-root "$OUT/eval" \
    --output "$OUT/direct-sft-versus-final" --key-file "$KEY"
  "$PY" "$RUN/score_direct_static.py" --manifest "$MANIFEST" \
    --baseline-label RL-step-16 --baseline-summary "$PRIOR_SUMMARY" \
    --baseline-images "$PRIOR_IMAGES" --final-root "$OUT/eval" \
    --output "$OUT/direct-step16-versus-final" \
    --key-file "$KEY"
fi
"$PY" "$RUN/build_checkpoint_gallery.py" --manifest "$MANIFEST" \
  --static "SFT:$PRIOR_IMAGES" --static "RL-step-16:$PRIOR_IMAGES" \
  --eval "RL-step-$STEP:$OUT/eval" \
  --summary "SFT:$BASELINE" \
  --summary "RL-step-16:$PRIOR_BASELINE" \
  --summary "RL-step-$STEP:$OUT/comparison/summary.json" \
  --output "$RUN/evidence/continued-gallery"
"$PY" "$RUN/collect_eval_canvases.py" --eval-root "$OUT/eval" \
  --output "$RUN/evidence/continued-canvases"
mkdir -p "$RUN/evidence"
printf '%s\n' "$STEP" > "$RUN/evidence/continued-evaluated-step.txt"
printf '%s\n' "$SCOPE" > "$RUN/evidence/continued-evaluation-scope.txt"
"$PY" "$RUN/archive_continuation.py" --run-root "$RUN" \
  --evaluation "${NAME}-step-$STEP-$SCOPE" --step "$STEP" \
  --output "/root/painter/continued-step-$STEP-$SCOPE.tar.gz" \
  --remote-path "runs/text-pairwise-rl-20260928/continued-step-$STEP-$SCOPE-20260929.tar.gz" \
  --token-file "${HF_TOKEN_PATH:-/root/painter/private/hf-token}"
echo "CONTINUATION_PUBLIC_EVIDENCE_VERIFIED step=$STEP scope=$SCOPE"
