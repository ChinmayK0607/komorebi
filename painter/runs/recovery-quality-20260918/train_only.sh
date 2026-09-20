#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(realpath "${1:?workspace}");cd "$ROOT";source env.sh
exec 8>train.lock;flock -n 8
trap 'echo "$?" > training.exit' EXIT
export HF_CHECKPOINT_REPO=CK0607/qwen3.8-27b-brush-painting HF_CHECKPOINT_RUN=recovery-quality-20260918
export WATERCOLOUR_INIT_ADAPTER=$(cat initial-adapter-path.txt)
export FRESH_BASE=0 CAPACITY_STATE_COMPLETE=1 ADAPTER_ONLY_CHECKPOINTS=1 PRL_OUTPUT_DIR="$ROOT/train-output"
export MIXED_SFT_TRACE="$ROOT/batch-shapes.jsonl" CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
SETUP="$BRUSH_ROOT/runs/$HF_CHECKPOINT_RUN-setup"
cd "$PRIME_ROOT"
timeout --signal=TERM --kill-after=60 7200 .venv/bin/torchrun --standalone --nnodes=1 --nproc-per-node=1 \
 "$BRUSH_ROOT/integrations/speedpainting/mixed_sft/train.py" @ "$SETUP/resolved.toml" > "$SETUP/train.log" 2>&1
