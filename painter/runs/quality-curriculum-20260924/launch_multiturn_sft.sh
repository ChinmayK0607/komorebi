#!/usr/bin/env bash
# Linux-only finite SFT. Run after bootstrap, data export, and config generation.
set -Eeuo pipefail

RUN="$(cd "${1:?usage: launch_multiturn_sft.sh RUN_DIRECTORY}" && pwd -P)"
[[ "$(uname -s)" == Linux ]] || { echo 'SFT runs on the Linux GPU node' >&2; exit 2; }
cd "$RUN"
exec 8>"$RUN/train.lock"
flock -n 8 || { echo 'SFT already running' >&2; exit 1; }
[[ -f env.sh && -f sft.toml && -f data/mix-manifest.json && -f initial-adapter-path.txt ]] || {
  echo 'bootstrap, audited data, or verified initializer missing' >&2; exit 2;
}
source env.sh
export BRUSH_ROOT PRIME_ROOT HF_HOME HF_TOKEN_PATH
export BRUSH_MODEL_PROFILE=qwen38_27b CAPACITY_STATE_COMPLETE=1 ADAPTER_ONLY_CHECKPOINTS=1
export WATERCOLOUR_INIT_ADAPTER="$(cat initial-adapter-path.txt)"
export HF_CHECKPOINT_REPO=CK0607/qwen3.8-27b-brush-painting
export HF_CHECKPOINT_RUN=multiturn-sft-20260925-v1
export PRL_OUTPUT_DIR="$RUN/train-output" WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export MIXED_SFT_TRACE="$RUN/batch-shapes.jsonl"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
[[ -f "$HF_TOKEN_PATH" ]] || { echo 'protected HF upload credential missing' >&2; exit 2; }
HF_TOKEN="$(cat "$HF_TOKEN_PATH")"
export HF_TOKEN
[[ -n "$HF_TOKEN" ]] || { echo 'protected HF upload credential empty' >&2; exit 2; }

"$PRIME_ROOT/.venv/bin/python" - "$RUN" <<'PY'
import hashlib,json,pathlib,sys,tomllib
from huggingface_hub import HfApi
root=pathlib.Path(sys.argv[1]); mix=json.loads((root/'data/mix-manifest.json').read_text())
for split in ('train','validation'):
    actual=hashlib.sha256((root/'data'/f'{split}.jsonl').read_bytes()).hexdigest()
    if actual!=mix['output_sha256'][split]:raise ValueError(f'{split} hash differs from mix manifest')
cfg=tomllib.loads((root/'sft.toml').read_text())
if cfg['max_steps']!=mix['optimizer_steps_at_batch_4'] or cfg['data']['batch_size']!=4:
    raise ValueError('configuration differs from finite four-row schedule')
if cfg['ckpt']['interval']!=mix['checkpoint_multiple']:
    raise ValueError('final step is not a public checkpoint boundary')
if cfg['model']['name']!=(root/'model-path.txt').read_text().strip():
    raise ValueError('configured model snapshot differs from verified bootstrap')
adapter=pathlib.Path((root/'initial-adapter-path.txt').read_text().strip())
if not adapter.is_dir() or not (adapter/'adapter_model.safetensors').is_file():
    raise ValueError('verified step-512 adapter missing')
expected='bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740'
if hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest()!=expected:
    raise ValueError('step-512 adapter SHA-256 mismatch')
info=HfApi().model_info('CK0607/qwen3.8-27b-brush-painting')
if info.private:raise ValueError('checkpoint repository is not public')
print(json.dumps({'preflight':'verified','steps':cfg['max_steps'],'checkpoint_every':cfg['ckpt']['interval'],
                  'adapter_sha256':expected,'checkpoint_repo_public':True}))
PY

"$PRIME_ROOT/.venv/bin/python" audit_dataset.py --config sft.toml --output processor-audit.json \
  --brush-root "$BRUSH_ROOT" > processor-audit.log 2>&1
"$PRIME_ROOT/.venv/bin/python" - "$RUN" <<'PY'
import json,pathlib,sys,tomllib
root=pathlib.Path(sys.argv[1]); cfg=tomllib.loads((root/'sft.toml').read_text())
audit=json.loads((root/'processor-audit.json').read_text())
if audit['status']!='passed' or audit['max_tokens']>cfg['data']['seq_len']:
    raise ValueError('exact processor length/mask audit failed')
print(json.dumps({'processor_audit':'passed','max_tokens':audit['max_tokens'],
                  'seq_len':cfg['data']['seq_len'],'train_rows':audit['train_rows']}))
PY

TELEMETRY_PID=""
cleanup() {
  local status=$?
  printf '%s\n' "$status" > "$RUN/training.exit"
  if [[ -n "$TELEMETRY_PID" ]]; then kill "$TELEMETRY_PID" 2>/dev/null || true; wait "$TELEMETRY_PID" 2>/dev/null || true; fi
  python3 "$RUN/summarize_telemetry.py" --input "$RUN/gpu-telemetry.csv" --output "$RUN/gpu-telemetry-summary.json" 2>/dev/null || true
}
trap cleanup EXIT
python3 gpu_telemetry.py --output "$RUN/gpu-telemetry.csv" --interval 10 &
TELEMETRY_PID=$!
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "$PRIME_ROOT"
"$PRIME_ROOT/.venv/bin/torchrun" --standalone --nnodes=1 --nproc-per-node=1 \
  "$RUN/train_multiturn.py" @ "$RUN/sft.toml" > "$RUN/train.log" 2>&1
