#!/usr/bin/env bash
set -Eeuo pipefail

# One finite post-training evaluation and public archive; invoke only after
# the chosen final checkpoint exists and the training process has exited.
MID="${1:?usage: finalize_pilot.sh MID_STEP FINAL_STEP}"
FINAL="${2:?usage: finalize_pilot.sh MID_STEP FINAL_STEP}"
[[ "$MID" =~ ^[1-9][0-9]*$ && "$FINAL" =~ ^[1-9][0-9]*$ && "$MID" -lt "$FINAL" ]] || {
  echo 'expected distinct positive midpoint and final step' >&2; exit 2;
}
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
OLD="$RUN/../text-pairwise-rl-20260928"
PRIME="${PRIME_ROOT:-/root/painter/prime}"
PY="$PRIME/.venv/bin/python"
KEY="${PAINTER_GATEWAY_ENV_FILE:-/root/painter/private/gateway.env}"
TOKEN="${HF_TOKEN_PATH:-/root/painter/private/hf-token}"
NAME=text-grounded-pilot-20260930
EVIDENCE="$RUN/evidence"
MID_EVAL="$RUN/validation/step-$MID-balanced16"
FINAL_EVAL="$RUN/validation/step-$FINAL-all"
METRICS="$RUN/async-output/$NAME/monitors/file/metrics.jsonl"
TRAINER="$RUN/async-output/$NAME/logs/attempt_1/trainer.log"
mkdir -p "$EVIDENCE"
[[ -f "$RUN/async-output/$NAME/artifacts/adapters/step_$MID/hf-upload.json" &&
   -f "$RUN/async-output/$NAME/artifacts/adapters/step_$FINAL/hf-upload.json" ]] || {
  echo 'both public checkpoint receipts are required before evaluation' >&2; exit 1;
}
if [[ -f /root/painter/pilot.pid ]] && kill -0 "$(cat /root/painter/pilot.pid)" 2>/dev/null; then
  echo 'training is still active; finalization requires its completed or stopped state' >&2
  exit 1
fi

"$PY" "$RUN/prepare_controls.py" --run-root "$RUN" \
  --cache /root/painter/cache/continued-step-30-all-20260929.tar.gz

if [[ ! -f "$MID_EVAL/comparison/summary.json" ]]; then
  bash "$RUN/evaluate.sh" "$MID" balanced16
fi
if [[ ! -f "$FINAL_EVAL/comparison/summary.json" ]]; then
  bash "$RUN/evaluate.sh" "$FINAL" all
fi
if [[ ! -f "$EVIDENCE/midpoint-canvases/manifest.json" ]]; then
  "$PY" "$OLD/collect_eval_canvases.py" --eval-root "$MID_EVAL/eval" \
    --output "$EVIDENCE/midpoint-canvases"
fi
if [[ ! -f "$EVIDENCE/final-canvases/manifest.json" ]]; then
  "$PY" "$OLD/collect_eval_canvases.py" --eval-root "$FINAL_EVAL/eval" \
    --output "$EVIDENCE/final-canvases"
fi

for control in SFT RL-step-16; do
  if [[ "$control" == SFT ]]; then
    label=sft; summary="$EVIDENCE/frozen-sft-summary.json"
  else
    label=step16; summary="$EVIDENCE/frozen-step16-summary.json"
  fi
  if [[ ! -f "$RUN/validation/direct-$label/summary.json" ]]; then
    "$PY" "$RUN/score_direct_pilot.py" \
      --manifest "$RUN/anchors-all/validation-manifest.json" \
      --baseline-kind "$control" --baseline-summary "$summary" \
      --baseline-images "$EVIDENCE/frozen-control-images" \
      --candidate-root "$FINAL_EVAL/eval" \
      --output "$RUN/validation/direct-$label" --key-file "$KEY"
  fi
done
if [[ ! -f "$RUN/validation/direct-midpoint/summary.json" ]]; then
  "$PY" "$RUN/score_direct_pilot.py" \
    --manifest "$RUN/balanced16-validation-manifest.json" \
    --baseline-kind midpoint --baseline-summary "$MID_EVAL/comparison/summary.json" \
    --baseline-images "$EVIDENCE/midpoint-canvases" \
    --candidate-root "$FINAL_EVAL/eval" \
    --output "$RUN/validation/direct-midpoint" --key-file "$KEY"
fi

"$PY" "$RUN/archive_pilot.py" --help >/dev/null
"$PY" - "$RUN" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1])))
from archive_pilot import ensure_stream
ensure_stream(Path(sys.argv[1]))
PY
if [[ ! -f "$EVIDENCE/reward-curve/summary.json" ]]; then
  "$PY" "$OLD/plot_reward_curve.py" --metrics "$METRICS" \
    --traces "$EVIDENCE/native-stream.jsonl" --output "$EVIDENCE/reward-curve"
fi
mapfile -t GPU_CSV < <(find "$RUN/logs" -maxdepth 1 -name 'gpu-telemetry-*.csv' -type f | sort)
(( ${#GPU_CSV[@]} > 0 )) || { echo 'GPU interval telemetry missing' >&2; exit 1; }
GPU_ARGS=()
for file in "${GPU_CSV[@]}"; do GPU_ARGS+=(--gpu-csv "$file"); done
"$PY" "$OLD/profile_continuation.py" \
  --metrics "$METRICS" --traces "$EVIDENCE/native-stream.jsonl" \
  --judges "$RUN/rl-rollouts-grounded" --trainer-log "$TRAINER" \
  "${GPU_ARGS[@]}" --output "$EVIDENCE/profile.json"

if [[ ! -f "$EVIDENCE/gallery/index.html" ]]; then
  "$PY" "$OLD/control_sets.py" subset \
    --summary "$EVIDENCE/frozen-sft-summary.json" \
    --manifest "$RUN/anchors-all/validation-manifest.json" \
    --output "$EVIDENCE/frozen-sft-47.json"
  "$PY" "$OLD/control_sets.py" subset \
    --summary "$EVIDENCE/frozen-step16-summary.json" \
    --manifest "$RUN/anchors-all/validation-manifest.json" \
    --output "$EVIDENCE/frozen-step16-47.json"
  "$PY" "$OLD/build_checkpoint_gallery.py" \
    --manifest "$RUN/anchors-all/validation-manifest.json" \
    --static "SFT:$EVIDENCE/frozen-control-images" \
    --static "RL-step-16:$EVIDENCE/frozen-control-images" \
    --eval "pilot-step-$FINAL:$FINAL_EVAL/eval" \
    --summary "SFT:$EVIDENCE/frozen-sft-47.json" \
    --summary "RL-step-16:$EVIDENCE/frozen-step16-47.json" \
    --summary "pilot-step-$FINAL:$FINAL_EVAL/comparison/summary.json" \
    --output "$EVIDENCE/gallery"
fi

"$PY" - "$RUN" "$MID" "$FINAL" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);mid=int(sys.argv[2]);final=int(sys.argv[3]);e=root/'evidence'
def read(path): return json.loads(path.read_text())
def small(summary):
    return {k:summary.get(k) for k in ('expected','comparable','censored',
       'valid_candidates','mean_score','wins','ties','losses','both_valid',
       'mean_score','bootstrap_95pct_interval','order_gap_at_least_third')
report={'schema':'painter-grounded-pilot-final-v1','mid_step':mid,'final_step':final,
  'midpoint_balanced16':small(read(root/f'validation/step-{mid}-balanced16/comparison/summary.json')),
  'final_all47':small(read(root/f'validation/step-{final}-all/comparison/summary.json')),
  'direct_vs_sft':small(read(root/'validation/direct-sft/summary.json')),
  'direct_vs_step16':small(read(root/'validation/direct-step16/summary.json')),
  'direct_final_vs_midpoint_balanced16':small(read(root/'validation/direct-midpoint/summary.json')),
  'checkpoint_receipts':{
    str(step):read(root/f'async-output/text-grounded-pilot-20260930/artifacts/adapters/step_{step}/hf-upload.json')
    for step in (mid,final)},
  'profile_path':'evidence/profile.json','reward_curve_path':'evidence/reward-curve',
  'gallery_path':'evidence/gallery/index.html'}
(e/'final-analysis.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='checkpoint_receipts'},sort_keys=True))
PY

"$PY" "$RUN/archive_pilot.py" --run-root "$RUN" \
  --mid-step "$MID" --final-step "$FINAL" \
  --output "/root/painter/cache/grounded-pilot-step-$FINAL-20260930.tar.gz" \
  --remote-path "runs/text-grounded-pilot-20260930/grounded-pilot-step-$FINAL-20260930.tar.gz" \
  --token-file "$TOKEN"
echo 'GROUNDED_PILOT_PUBLIC_ARCHIVE_VERIFIED'
