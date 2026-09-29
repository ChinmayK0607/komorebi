#!/usr/bin/env bash
set -Eeuo pipefail

# One finite continuation on the rented node. It waits for this run's training
# attempt, scores the newest publicly verified checkpoint, and writes a
# matched SFT-versus-RL report. It never starts another training attempt.
trap 'rc=$?; printf "%s\n" "$rc" > /root/painter/scaled-eval.exit' EXIT
RUN=/root/painter/repo/painter/runs/text-pairwise-rl-20260928
PY=/root/painter/prime/.venv/bin/python
NAME=text-pairwise-scaled-20260929
DEADLINE=1790677701
until [[ -f /root/painter/scaled-rl.exit ]]; do sleep 30; done
ANCHORS="$(cat "$RUN/prepared/scaled-anchors-path.txt")"
[[ -f "$ANCHORS/validation-manifest.json" && \
   -f "$RUN/validation/scaled-sft-control/validation-summary.json" ]] || {
  echo CONTROL_OR_MANIFEST_MISSING; exit 1;
}
step="$("$PY" - "$RUN/async-output/$NAME/artifacts/adapters" <<'PY'
import sys
from pathlib import Path
root=Path(sys.argv[1])
steps=[int(path.parent.name.removeprefix('step_'))
       for path in root.glob('step_*/hf-upload.json')
       if path.parent.name.removeprefix('step_').isdigit()
       and (path.parent/'adapter_model.safetensors').is_file()]
if not steps: raise SystemExit('no public, hash-verified checkpoint to evaluate')
print(max(steps))
PY
)"

remaining=$((DEADLINE-$(date -u +%s)))
(( remaining >= 2700 )) || { echo INSUFFICIENT_EVALUATION_WINDOW; exit 1; }
if (( remaining >= 6000 )); then
  manifest="$ANCHORS/validation-manifest.json"
  scope=all
else
  manifest="$RUN/prepared/scaled-validation-16.json"
  "$PY" - "$ANCHORS/validation-manifest.json" "$manifest" <<'PY'
import json,sys
from pathlib import Path
source,target=map(Path,sys.argv[1:])
rows=json.loads(source.read_text())['cases']
foundation=[r for r in rows if r['family'] in {'simple','intermediate'}][:8]
complex_cases=[r for r in rows if r['family']=='complex'][:8]
chosen=foundation+complex_cases
if len(chosen)!=16 or len({r['id'] for r in chosen})!=16:
    raise SystemExit('cannot freeze balanced 16-case evaluation subset')
target.write_text(json.dumps({'cases':chosen,'selection':'first-eight-foundation-and-complex'},
                             indent=2,sort_keys=True)+'\n')
PY
  scope=balanced16
fi
out="$RUN/validation/scaled-step-$step-$scope"
available=$((DEADLINE-$(date -u +%s)-900))
(( available >= 1800 )) || { echo INSUFFICIENT_EVALUATION_WINDOW; exit 1; }
timeout --signal=INT --kill-after=90 "${available}s" \
  env PAINTER_RL_RUN_NAME="$NAME" \
      PAINTER_VALIDATION_MANIFEST="$manifest" \
      PAINTER_CHECKPOINT_OUT="$out" \
      PAINTER_JUDGE_PROTOCOL=text-paint-sft-pairwise-margin-v1 \
      bash "$RUN/evaluate_checkpoint.sh" "$step" all
if [[ "$scope" == all ]]; then
  baseline="$RUN/validation/scaled-sft-control/validation-summary.json"
else
  baseline="$out/baseline-subset.json"
  "$PY" "$RUN/control_sets.py" subset \
    --summary "$RUN/validation/scaled-sft-control/comparison/summary.json" \
    --manifest "$manifest" --output "$baseline"
fi
"$PY" "$RUN/analyze_scaled.py" --baseline "$baseline" \
  --candidate "$out/comparison/summary.json" --output "$out/analysis.json"
"$PY" "$RUN/build_checkpoint_gallery.py" --manifest "$manifest" \
  --eval "SFT:$RUN/validation/scaled-sft-control/eval" \
  --eval "RL-step-$step:$out/eval" \
  --summary "SFT:$baseline" \
  --summary "RL-step-$step:$out/comparison/summary.json" \
  --output "$RUN/evidence/gallery"
"$PY" "$RUN/collect_eval_canvases.py" --eval-root "$out/eval" \
  --output "$RUN/evidence/canvases"
mkdir -p "$RUN/evidence"
telemetry="$(find "$RUN/logs" -maxdepth 1 -name 'gpu-telemetry-*.csv' -type f | sort | tail -n 1)"
[[ -z "$telemetry" ]] || cp "$telemetry" "$RUN/evidence/gpu-telemetry.csv"
printf '%s\n' "$step" > "$RUN/evidence/evaluated-step.txt"
printf '%s\n' "$scope" > "$RUN/evidence/evaluation-scope.txt"
echo "SCALED_EVALUATION_COMPLETE step=$step scope=$scope"
