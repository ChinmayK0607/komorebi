#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(realpath "${1:?workspace}");cd "$ROOT";mkdir -p logs private
chmod 700 private
exec 7>training-setup.lock;flock -n 7
trap 'echo "$?" > training-setup.exit' EXIT
export HF_HOME="$ROOT/cache/huggingface" UV_CACHE_DIR="$ROOT/cache/uv" HF_HUB_DISABLE_TELEMETRY=1
export BRUSH_ROOT="$ROOT/source/brush-rl" PRIME_ROOT="$ROOT/prime-rl" BRUSH_MODEL_PROFILE=qwen38_27b
export HF_TOKEN_PATH="$ROOT/private/hf-token"
if [[ ! -x bootstrap/bin/uv ]];then python3 -m venv bootstrap;bootstrap/bin/pip -q install 'uv==0.11.1';fi
bootstrap/bin/uv pip install --python bootstrap/bin/python 'huggingface_hub==1.30.0' 'hf_xet==1.6.0' > logs/hub-install.log 2>&1
export PATH="$ROOT/bootstrap/bin:$PATH"
(bash setup_node.sh "$ROOT" render) > logs/render-setup.log 2>&1 &
RENDER_PID=$!
bootstrap/bin/python download_model.py > logs/download.log 2>&1 & DOWNLOAD_PID=$!
if [[ ! -d "$PRIME_ROOT/.git" ]];then
 git init "$PRIME_ROOT";git -C "$PRIME_ROOT" remote add origin https://github.com/PrimeIntellect-ai/prime-rl.git
 git -c http.version=HTTP/1.1 -C "$PRIME_ROOT" fetch --depth=1 origin 26b3131d2716a4f8210b165df584b83b4bc54f61
 git -C "$PRIME_ROOT" checkout --detach 26b3131d2716a4f8210b165df584b83b4bc54f61
fi
[[ $(git -C "$PRIME_ROOT" rev-parse HEAD) == 26b3131d2716a4f8210b165df584b83b4bc54f61 ]]
git -c http.version=HTTP/1.1 -c url.https://github.com/.insteadOf=git@github.com: -C "$PRIME_ROOT" submodule update --init --depth 1 -- deps/verifiers deps/renderers deps/prime-envs deps/pydantic-config
EXTRAS=(--extra gpu --extra flash-attn --extra kernels --extra quack --extra disagg)
if nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | grep -q '^9\.';then EXTRAS+=(--extra flash-attn-3);fi
(cd "$PRIME_ROOT";uv sync --frozen "${EXTRAS[@]}") > logs/prime-install.log 2>&1
[[ -x "$PRIME_ROOT/.venv/bin/vllm-router" ]] || { echo 'Pinned inference router is missing'; exit 1; }
wait "$DOWNLOAD_PID"
python3 stage_model.py > logs/model-staging.log 2>&1
wait "$RENDER_PID"
export MODEL_DIR=$(cat model-path.txt)
python3 - "$ROOT" <<'PY'
import os,shlex,sys
from pathlib import Path
p=Path(sys.argv[1]);keys=['HF_HOME','UV_CACHE_DIR','BRUSH_ROOT','PRIME_ROOT','MODEL_DIR','BRUSH_MODEL_PROFILE','HF_TOKEN_PATH']
values={k:os.environ[k] for k in keys};values.update(PLAYWRIGHT_BROWSERS_PATH=str(p/'browsers'),WANDB_MODE='disabled',TOKENIZERS_PARALLELISM='false')
(p/'env.sh').write_text('\n'.join('export '+k+'='+shlex.quote(v) for k,v in values.items())+'\n')
PY
python3 "$ROOT/training_handoff.py" bootstrap "$ROOT"
printf 'Environment and rendered dataset ready; launch is separate.\n'
