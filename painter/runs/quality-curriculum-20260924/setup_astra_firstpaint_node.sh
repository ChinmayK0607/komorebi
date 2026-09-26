#!/usr/bin/env bash
# Finite, repeatable Linux setup. No optimizer step runs here.
set -Eeuo pipefail
RUN="$(cd "${1:?usage: setup_astra_firstpaint_node.sh RUN_DIRECTORY}" && pwd -P)"
[[ "$(uname -s)" == Linux ]] || { echo 'Linux node required' >&2; exit 2; }
[[ -f "$RUN/private/hf-token" ]] || { echo 'protected HF upload credential missing' >&2; exit 2; }
chmod 700 "$RUN/private"
chmod 600 "$RUN/private/hf-token"
python3 - "$RUN" <<'PY'
import hashlib,json,pathlib,sys
r=pathlib.Path(sys.argv[1]); m=json.loads((r/'data/mix-manifest.json').read_text())
assert m['stage']=='first-paint' and m['optimizer_steps']==20 and m['batch_size']==4
assert m['reviewed_exposures']==39 and m['retention_exposures']==41
for split in ('train','validation'):
    p=r/'data'/f'{split}.jsonl'
    if hashlib.sha256(p.read_bytes()).hexdigest()!=m['output_sha256'][split]:
        raise ValueError(f'{split} hash mismatch')
print('First-paint data and schedule hash verified')
PY
export HF_TOKEN_PATH="$RUN/private/hf-token"
export BRUSH_ROOT="$RUN/source/brush-rl" BRUSH_MODEL_PROFILE=qwen38_27b
bash "$RUN/bootstrap.sh" "$RUN"
source "$RUN/env.sh"
"$RUN/bootstrap/bin/python" "$RUN/download_initial_adapter.py"
"$RUN/bootstrap/bin/python" "$RUN/stage_model.py"
"$RUN/bootstrap/bin/python" "$RUN/make_sft_config.py" \
  --data "$RUN/data" --model-path "$(cat "$RUN/model-path.txt")" \
  --output "$RUN/sft.toml" --output-dir "$RUN/train-output" \
  --run-name astra-firstpaint-sft-20260926-v1 --steps 20 --seq-len 16384 \
  --gpus-per-node 1 --num-train-gpus 1 --dp-replicate 1 --dp-shard 1 \
  --global-batch-size 4 --micro-batch-size 1 --learning-rate 0.000001 \
  --checkpoint-interval 10 --validation-interval 10 --seed 20260926
bash "$RUN/setup_multiturn_eval.sh" "$RUN"
echo "Setup complete. Launch with: bash $RUN/run_astra_firstpaint_campaign.sh $RUN"
