#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(realpath "${1:?workspace}");shift;cd "$ROOT";source env.sh
MODEL_DIR=$(cat model-path.txt)
export VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6
export CUDA_HOME=$(runtime/bin/python -c 'import sysconfig;print(sysconfig.get_paths()["purelib"]+"/nvidia/cu13")')
export PATH="$ROOT/runtime/bin:$CUDA_HOME/bin:$ROOT/bootstrap/bin:$PATH" LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
exec runtime/bin/vllm serve "$MODEL_DIR" --served-model-name base --host 127.0.0.1 --port 8000 \
 --dtype bfloat16 --max-model-len 32768 --max-num-seqs 8 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90 \
 --limit-mm-per-prompt '{"image":2,"video":0}' --gdn-prefill-backend triton --kernel-config '{"enable_jit_warmup":false}' \
 --enable-lora --max-lora-rank 16 --max-loras 2 --mm-processor-kwargs '{"min_pixels":3136,"max_pixels":1048576}' "$@"
