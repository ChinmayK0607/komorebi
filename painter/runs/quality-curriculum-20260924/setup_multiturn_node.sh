#!/usr/bin/env bash
# One-shot reproducible setup. It leaves launch as a separate explicit command.
set -Eeuo pipefail
RUN="$(cd "${1:?usage: setup_multiturn_node.sh RUN_DIRECTORY}" && pwd -P)"
[[ "$(uname -s)" == Linux ]] || { echo 'setup requires Linux' >&2; exit 2; }
[[ -f "$RUN/data/mix-manifest.json" && -f "$RUN/private/hf-token" ]] || {
  echo 'mixed data or protected HF credential missing' >&2; exit 2;
}
chmod 700 "$RUN/private"
chmod 600 "$RUN/private/hf-token"
export HF_TOKEN_PATH="$RUN/private/hf-token"
export BRUSH_ROOT="$RUN/source/brush-rl"
export BRUSH_MODEL_PROFILE=qwen38_27b
bash "$RUN/bootstrap.sh" "$RUN"
source "$RUN/env.sh"
"$RUN/bootstrap/bin/python" "$RUN/download_initial_adapter.py"
"$RUN/bootstrap/bin/python" "$RUN/stage_model.py"
STEPS="$(python3 - "$RUN/data/mix-manifest.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['optimizer_steps_at_batch_4'])
PY
)"
"$RUN/bootstrap/bin/python" "$RUN/make_sft_config.py" \
  --data "$RUN/data" --model-path "$(cat "$RUN/model-path.txt")" \
  --output "$RUN/sft.toml" --output-dir "$RUN/train-output" \
  --run-name multiturn-sft-20260925-v1 --steps "$STEPS" --seq-len 16384 \
  --gpus-per-node 1 --num-train-gpus 1 --dp-replicate 1 --dp-shard 1 \
  --global-batch-size 4 --micro-batch-size 1 --learning-rate 0.000003 \
  --checkpoint-interval 16 --validation-interval 16 --seed 20260925
echo "Setup complete; launch with: bash $RUN/launch_multiturn_sft.sh $RUN"
