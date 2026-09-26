#!/usr/bin/env bash
# Finite two-GPU campaign. Setup is separate; this command performs training
# and matched evaluations without scheduling a future task or renting a node.
set -Eeuo pipefail
RUN="$(cd "${1:?usage: run_astra_firstpaint_campaign.sh RUN_DIRECTORY}" && pwd -P)"
[[ "$(uname -s)" == Linux && -f "$RUN/sft.toml" && -f "$RUN/eval-preflight.json" ]] || {
  echo 'complete SFT and eval setup before launch' >&2; exit 2;
}
exec 9>"$RUN/campaign.lock"
flock -n 9 || { echo 'campaign already running' >&2; exit 2; }
TRAIN_PID=''
BASELINE_PID=''
TRAINED_PID=''
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  for pid in "$TRAIN_PID" "$BASELINE_PID" "$TRAINED_PID"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  printf '%s\n' "$status" > "$RUN/campaign.exit"
  exit "$status"
}
trap cleanup EXIT INT TERM
GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
[[ "$GPU_COUNT" =~ ^[0-9]+$ ]] && (( GPU_COUNT >= 1 )) || {
  echo 'no CUDA GPU found for campaign' >&2; exit 2;
}
if (( GPU_COUNT >= 2 )); then
  mode=parallel_two_gpu
  export EVAL_GPU=1
  bash "$RUN/launch_astra_firstpaint_eval.sh" "$RUN" baseline > "$RUN/campaign-baseline.log" 2>&1 &
  BASELINE_PID=$!
  bash "$RUN/launch_astra_firstpaint_sft.sh" "$RUN" > "$RUN/campaign-training.log" 2>&1 &
  TRAIN_PID=$!
  set +e
  wait "$TRAIN_PID"; training_status=$?
  TRAIN_PID=''
  trained_status=99
  if (( training_status == 0 )); then
    # The trainer has released GPU 0. Evaluate the trained adapter there while
    # the baseline continues on GPU 1, avoiding a long idle-GPU tail.
    export EVAL_GPU=0
    bash "$RUN/launch_astra_firstpaint_eval.sh" "$RUN" trained > "$RUN/campaign-trained.log" 2>&1 &
    TRAINED_PID=$!
  fi
  wait "$BASELINE_PID"; baseline_status=$?
  BASELINE_PID=''
  if [[ -n "$TRAINED_PID" ]]; then
    wait "$TRAINED_PID"; trained_status=$?
    TRAINED_PID=''
  fi
  set -e
else
  mode=sequential_one_gpu
  export EVAL_GPU=0
  set +e
  bash "$RUN/launch_astra_firstpaint_sft.sh" "$RUN" > "$RUN/campaign-training.log" 2>&1
  training_status=$?
  baseline_status=99
  if (( training_status == 0 )); then
    bash "$RUN/launch_astra_firstpaint_eval.sh" "$RUN" baseline > "$RUN/campaign-baseline.log" 2>&1
    baseline_status=$?
  fi
  set -e
  trained_status=99
  if (( training_status == 0 )); then
    set +e
    bash "$RUN/launch_astra_firstpaint_eval.sh" "$RUN" trained > "$RUN/campaign-trained.log" 2>&1
    trained_status=$?
    set -e
  fi
fi
python3 - "$RUN" "$training_status" "$baseline_status" "$trained_status" "$mode" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]);training,baseline,trained=map(int,sys.argv[2:5])
record={'schema':'painter.astra-firstpaint-campaign.v1','training_exit':training,
        'baseline_eval_exit':baseline,'trained_eval_exit':trained,'execution_mode':sys.argv[5],
        'training_checkpoints':'public HF receipts under train-output',
        'evaluation_limit':'CLI completion is not a visual-quality result; review both rollout sets'}
(root/'campaign-completion.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
PY
(( training_status == 0 && baseline_status == 0 && trained_status == 0 ))
