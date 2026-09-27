#!/usr/bin/env bash
# Repeatable Linux setup; no optimizer update starts until the campaign command.
set -Eeuo pipefail
RUN="$(cd "${1:?usage: setup_brush_sft_node.sh RUN_DIRECTORY}" && pwd -P)"
[[ "$(uname -s)" == Linux && -f "$RUN/private/hf-token" ]] || exit 2
chmod 700 "$RUN/private"
chmod 600 "$RUN/private/hf-token"
python3 - "$RUN" <<'PY'
import hashlib,json,pathlib,sys
r=pathlib.Path(sys.argv[1]);m=json.loads((r/'data/mix-manifest.json').read_text())
if m['schema']!='painter.brush-sft-data.v1' or m['batch_size']!=4 or m['optimizer_steps']%40:
    raise ValueError('wrong finite brush SFT mix')
for split in ('train','validation'):
    if hashlib.sha256((r/'data'/f'{split}.jsonl').read_bytes()).hexdigest()!=m['output_sha256'][split]:
        raise ValueError(f'{split} hash mismatch')
print(json.dumps({'data_verified':True,'steps':m['optimizer_steps'],'paintings':m['counts']['painting_targets']}))
PY
export HF_TOKEN_PATH="$RUN/private/hf-token"
export BRUSH_ROOT="$RUN/source/brush-rl" BRUSH_MODEL_PROFILE=qwen38_27b
bash "$RUN/bootstrap.sh" "$RUN"
source "$RUN/env.sh"
"$RUN/bootstrap/bin/python" "$RUN/download_initial_adapter.py"
"$RUN/bootstrap/bin/python" "$RUN/stage_model.py"
STEPS="$(python3 - "$RUN/data/mix-manifest.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['optimizer_steps'])
PY
)"
"$RUN/bootstrap/bin/python" "$RUN/make_sft_config.py" \
  --data "$RUN/data" --model-path "$(cat "$RUN/model-path.txt")" \
  --output "$RUN/sft.toml" --output-dir "$RUN/train-output" \
  --run-name brush-sft-20260927-v1 --steps "$STEPS" --seq-len 16384 \
  --gpus-per-node 1 --num-train-gpus 1 --dp-replicate 1 --dp-shard 1 \
  --global-batch-size 4 --micro-batch-size 1 --learning-rate 0.000001 \
  --checkpoint-interval 40 --validation-interval 40 --seed 20260927
bash "$RUN/setup_multiturn_eval.sh" "$RUN"
echo "Ready: bash $RUN/run_brush_sft_campaign.sh $RUN"
