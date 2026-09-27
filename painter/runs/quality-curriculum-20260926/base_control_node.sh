#!/usr/bin/env bash
# Finite fresh-base LoRA control on the exact brush SFT data package.
set -Eeuo pipefail
RUN="$(cd "${1:?usage: base_control_node.sh EXTRACTED_PACKAGE}" && pwd -P)"
[[ "$(uname -s)" == Linux && -f "$RUN/private/hf-token" ]] || exit 2
export HF_TOKEN_PATH="$RUN/private/hf-token" BRUSH_ROOT="$RUN/source/brush-rl"
export BRUSH_MODEL_PROFILE=qwen38_27b
export RUN_NAME=brush-base-control-20260927-v1
export HF_CHECKPOINT_REPO=CK0607/qwen3.8-27b-brush-painting HF_CHECKPOINT_RUN="$RUN_NAME"
export CAPACITY_STATE_COMPLETE=1 ADAPTER_ONLY_CHECKPOINTS=1 FRESH_BASE=1
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export MIXED_SFT_TRACE="$RUN/base-control-batch-shapes.jsonl"
chmod 700 "$RUN/private"; chmod 600 "$HF_TOKEN_PATH"
python3 - "$RUN" <<'PY'
import hashlib,json,pathlib,sys
r=pathlib.Path(sys.argv[1]);m=json.loads((r/'data/mix-manifest.json').read_text())
assert m['schema']=='painter.brush-sft-data.v1' and m['optimizer_steps']==160
for split in ('train','validation'):
    assert hashlib.sha256((r/'data'/f'{split}.jsonl').read_bytes()).hexdigest()==m['output_sha256'][split]
print(json.dumps({'verified_data':True,'train_sha256':m['output_sha256']['train'],'steps':160}))
PY
bash "$RUN/bootstrap.sh" "$RUN"
source "$RUN/env.sh"
"$RUN/bootstrap/bin/python" "$RUN/stage_model.py"
"$RUN/bootstrap/bin/python" "$RUN/make_sft_config.py" \
  --data "$RUN/data" --model-path "$(cat "$RUN/model-path.txt")" \
  --output "$RUN/base-control.toml" --output-dir "$RUN/train-output" \
  --run-name "$RUN_NAME" --steps 160 --seq-len 16384 \
  --gpus-per-node 1 --num-train-gpus 1 --dp-replicate 1 --dp-shard 1 \
  --global-batch-size 4 --micro-batch-size 1 --learning-rate 0.000003 \
  --checkpoint-interval 40 --validation-interval 40 --seed 20260927
"$PRIME_ROOT/.venv/bin/python" - "$RUN" <<'PY'
import json,pathlib,sys,tomllib
r=pathlib.Path(sys.argv[1]); c=tomllib.loads((r/'base-control.toml').read_text())
assert c['data']['shuffle'] is False and c['max_steps']==160
assert c['data']['loss_mask']=={'system':False,'user':False,'assistant':True,'tool':False}
assert c['optim']['lr']==3e-6 and c['model']['lora']['rank']==16
setup=r/'source/brush-rl/runs/brush-base-control-20260927-v1-setup'
setup.mkdir(parents=True,exist_ok=True)
(setup/'resolved.setup.json').write_text(json.dumps({'initialization_mode':'fresh_base',
    'init_adapter':None,'step_zero_export_required':True,'optimizer':'fresh'},indent=2)+'\n')
print(json.dumps({'fresh_base':True,'rank':16,'learning_rate':3e-6,'steps':160}))
PY
"$PRIME_ROOT/.venv/bin/python" "$RUN/audit_dataset.py" --config "$RUN/base-control.toml" \
  --output "$RUN/base-control-processor-audit.json" --brush-root "$BRUSH_ROOT" \
  > "$RUN/base-control-processor-audit.log" 2>&1
"$PRIME_ROOT/.venv/bin/python" - "$RUN/base-control-processor-audit.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]));assert x['status']=='passed' and x['train_rows']==640 and x['max_tokens']<=16384
print(json.dumps({'processor_audit':'passed','train_rows':x['train_rows'],'max_tokens':x['max_tokens']}))
PY
bash "$RUN/setup_multiturn_eval.sh" "$RUN"
[[ -f "$RUN/eval-preflight.json" ]] || exit 2
"$PRIME_ROOT/.venv/bin/python" - <<'PY'
from huggingface_hub import HfApi
assert HfApi().model_info('CK0607/qwen3.8-27b-brush-painting').private is False
print('Public checkpoint repository verified')
PY
TELEMETRY_PID=''
cleanup() {
  local status=$?
  printf '%s\n' "$status" > "$RUN/base-control-training.exit"
  if [[ -n "$TELEMETRY_PID" ]]; then kill "$TELEMETRY_PID" 2>/dev/null || true; wait "$TELEMETRY_PID" 2>/dev/null || true; fi
  python3 "$RUN/summarize_telemetry.py" --input "$RUN/base-control-gpu-telemetry.csv" \
    --output "$RUN/base-control-gpu-telemetry-summary.json" 2>/dev/null || true
}
trap cleanup EXIT
python3 "$RUN/gpu_telemetry.py" --output "$RUN/base-control-gpu-telemetry.csv" --interval 10 &
TELEMETRY_PID=$!
export HF_TOKEN="$(cat "$HF_TOKEN_PATH")" PRL_OUTPUT_DIR="$RUN/train-output"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" OMP_NUM_THREADS=8
export PYTHONPATH="$BRUSH_ROOT/scripts:$BRUSH_ROOT/integrations/prime/watercolour_sft:$BRUSH_ROOT/integrations/speedpainting/mixed_sft${PYTHONPATH:+:$PYTHONPATH}"
cd "$PRIME_ROOT"
"$PRIME_ROOT/.venv/bin/torchrun" --standalone --nnodes=1 --nproc-per-node=1 \
  "$RUN/source/brush-rl/integrations/speedpainting/mixed_sft/train.py" @ "$RUN/base-control.toml" \
  > "$RUN/base-control-train.log" 2>&1
