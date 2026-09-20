#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(realpath "${1:?workspace}");cd "$ROOT";source env.sh
export VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6
export CUDA_HOME=$(runtime/bin/python -c 'import sysconfig;print(sysconfig.get_paths()["purelib"]+"/nvidia/cu13")')
export PATH="$ROOT/runtime/bin:$CUDA_HOME/bin:$ROOT/bootstrap/bin:$PATH" LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
exec runtime/bin/vllm serve "$(cat judge-path.txt)" --served-model-name judge --host 127.0.0.1 --port 8000 \
 --dtype bfloat16 --max-model-len 8192 --max-num-seqs 4 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.92 \
 --enforce-eager --limit-mm-per-prompt '{"image":3,"video":0}' --mm-processor-kwargs '{"min_pixels":3136,"max_pixels":1048576}'
