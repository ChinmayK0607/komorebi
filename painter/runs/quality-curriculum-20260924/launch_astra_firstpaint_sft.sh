#!/usr/bin/env bash
set -Eeuo pipefail
RUN="$(cd "${1:?usage: launch_astra_firstpaint_sft.sh RUN_DIRECTORY}" && pwd -P)"
[[ "$(uname -s)" == Linux && -f "$RUN/env.sh" && -f "$RUN/sft.toml" ]] || exit 2
exec 8>"$RUN/train.lock"
flock -n 8 || { echo 'SFT already running' >&2; exit 2; }
source "$RUN/env.sh"
export BRUSH_ROOT PRIME_ROOT HF_HOME HF_TOKEN_PATH
export BRUSH_MODEL_PROFILE=qwen38_27b CAPACITY_STATE_COMPLETE=1 ADAPTER_ONLY_CHECKPOINTS=1
export WATERCOLOUR_INIT_ADAPTER="$(cat "$RUN/initial-adapter-path.txt")"
export HF_CHECKPOINT_REPO=CK0607/qwen3.8-27b-brush-painting
export HF_CHECKPOINT_RUN=astra-firstpaint-sft-20260926-v1
export PRL_OUTPUT_DIR="$RUN/train-output" WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export MIXED_SFT_TRACE="$RUN/batch-shapes.jsonl"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
[[ -f "$HF_TOKEN_PATH" ]] || exit 2
HF_TOKEN="$(cat "$HF_TOKEN_PATH")"; export HF_TOKEN
[[ -n "$HF_TOKEN" ]] || exit 2
"$PRIME_ROOT/.venv/bin/python" - "$RUN" <<'PY'
import hashlib,json,pathlib,sys,tomllib
from huggingface_hub import HfApi
r=pathlib.Path(sys.argv[1]); m=json.loads((r/'data/mix-manifest.json').read_text())
c=tomllib.loads((r/'sft.toml').read_text())
if (m['stage'],m['optimizer_steps'],m['batch_size'])!=('first-paint',20,4):raise ValueError('wrong stage')
for s in ('train','validation'):
    if hashlib.sha256((r/'data'/f'{s}.jsonl').read_bytes()).hexdigest()!=m['output_sha256'][s]:raise ValueError('data hash')
if c['max_steps']!=20 or c['data']['batch_size']!=4 or c['ckpt']['interval']!=10 or c['optim']['lr']!=1e-6:raise ValueError('config mismatch')
if c['model']['name']!=(r/'model-path.txt').read_text().strip():raise ValueError('model mismatch')
a=pathlib.Path((r/'initial-adapter-path.txt').read_text().strip())/'adapter_model.safetensors'
if hashlib.sha256(a.read_bytes()).hexdigest()!='bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740':raise ValueError('initializer mismatch')
if HfApi().model_info('CK0607/qwen3.8-27b-brush-painting').private:raise ValueError('checkpoint repo is private')
print(json.dumps({'preflight':'verified','steps':20,'checkpoint_steps':[10,20]}))
PY
"$PRIME_ROOT/.venv/bin/python" "$RUN/audit_dataset.py" --config "$RUN/sft.toml" --output "$RUN/processor-audit.json" \
  --brush-root "$BRUSH_ROOT" > "$RUN/processor-audit.log" 2>&1
"$PRIME_ROOT/.venv/bin/python" - "$RUN" <<'PY'
import json,pathlib,sys,tomllib
r=pathlib.Path(sys.argv[1]); a=json.loads((r/'processor-audit.json').read_text()); c=tomllib.loads((r/'sft.toml').read_text())
if a['status']!='passed' or a['max_tokens']>c['data']['seq_len']:raise ValueError('exact processor audit failed')
print(json.dumps({'processor_audit':'passed','max_tokens':a['max_tokens'],'train_rows':a['train_rows']}))
PY
TELEMETRY_PID=''
cleanup() {
  local status=$?
  printf '%s\n' "$status" > "$RUN/training.exit"
  if [[ -n "$TELEMETRY_PID" ]]; then kill "$TELEMETRY_PID" 2>/dev/null || true; wait "$TELEMETRY_PID" 2>/dev/null || true; fi
  python3 "$RUN/summarize_telemetry.py" --input "$RUN/gpu-telemetry.csv" --output "$RUN/gpu-telemetry-summary.json" 2>/dev/null || true
}
trap cleanup EXIT
python3 "$RUN/gpu_telemetry.py" --output "$RUN/gpu-telemetry.csv" --interval 10 &
TELEMETRY_PID=$!
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "$PRIME_ROOT"
"$PRIME_ROOT/.venv/bin/torchrun" --standalone --nnodes=1 --nproc-per-node=1 \
  "$RUN/train_multiturn.py" @ "$RUN/sft.toml" > "$RUN/train.log" 2>&1
