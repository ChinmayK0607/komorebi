#!/usr/bin/env bash
# One-shot reproducible setup. It leaves launch as a separate explicit command.
set -Eeuo pipefail
RUN="$(cd "${1:?usage: setup_multiturn_node.sh RUN_DIRECTORY}" && pwd -P)"
[[ "$(uname -s)" == Linux ]] || { echo 'setup requires Linux' >&2; exit 2; }
[[ -f "$RUN/expected-mix-manifest.json" && -f "$RUN/private/hf-token" ]] || {
  echo 'mix manifest or protected HF credential missing' >&2; exit 2;
}
chmod 700 "$RUN/private"
chmod 600 "$RUN/private/hf-token"
read -r REPEATS SEED CHECKPOINT_MULTIPLE < <(python3 - "$RUN/expected-mix-manifest.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
print(x['reviewed_repeats'],x['seed'],x['checkpoint_multiple'])
PY
)
python3 "$RUN/build_multiturn_mix.py" \
  --reviewed "$RUN/data-input/reviewed.jsonl" \
  --foundation "$RUN/data-input/foundation.jsonl" \
  --foundation-validation "$RUN/data-input/foundation-validation.jsonl" \
  --output "$RUN/data" --repeats "$REPEATS" --seed "$SEED" \
  --checkpoint-multiple "$CHECKPOINT_MULTIPLE" > "$RUN/data-rebuild.json"
python3 - "$RUN" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1])
actual=json.loads((root/'data/mix-manifest.json').read_text())
expected=json.loads((root/'expected-mix-manifest.json').read_text())
if actual != expected:raise ValueError('rebuilt mix differs from reviewed expected manifest')
print('Finite mixed data rebuilt and hash-verified')
PY
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
  --checkpoint-interval "$CHECKPOINT_MULTIPLE" \
  --validation-interval "$CHECKPOINT_MULTIPLE" --seed 20260925
bash "$RUN/setup_multiturn_eval.sh" "$RUN"
echo "Setup complete; launch with: bash $RUN/run_multiturn_campaign.sh $RUN"
