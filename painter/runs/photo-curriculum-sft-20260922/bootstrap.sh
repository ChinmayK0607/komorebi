#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "${1:?usage: bootstrap.sh RUN_ROOT}" && pwd -P)"
cd "$ROOT"
[[ "$(uname -s)" == Linux ]] || { echo "bootstrap runs on the Linux node" >&2; exit 1; }
mkdir -p logs cache private
chmod 700 private
exec 9>"$ROOT/bootstrap.lock"
flock -n 9 || { echo "bootstrap already running"; exit 0; }
trap 'rc=$?; printf "%s\n" "$rc" > "$ROOT/bootstrap.exit"' EXIT

export HF_HOME="${HF_HOME:-$ROOT/cache/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/cache/uv}"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$ROOT/private/hf-token}"
export BRUSH_ROOT="${BRUSH_ROOT:-$ROOT/source/brush-rl}"
export PRIME_ROOT="${PRIME_ROOT:-$ROOT/prime-rl}"
export BRUSH_MODEL_PROFILE=qwen38_27b

if [[ ! -x bootstrap/bin/uv ]]; then
  python3 -m venv bootstrap
  bootstrap/bin/pip -q install 'uv==0.11.1'
fi
bootstrap/bin/uv pip install --python bootstrap/bin/python 'huggingface_hub==1.30.0' 'hf_xet==1.6.0' > logs/hub-install.log 2>&1
export PATH="$ROOT/bootstrap/bin:$PATH"

# The base snapshot is independent of the Prime checkout and CUDA extras.
# Start it as soon as the verified Hub client exists to avoid serial startup.
bootstrap/bin/python download_model.py > logs/download.log 2>&1 &
DOWNLOAD_PID=$!

python3 - "$ROOT" <<'PY'
import json, platform, subprocess, sys
from pathlib import Path
p=Path(sys.argv[1])
gpus=[]
try:
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,compute_cap','--format=csv,noheader,nounits'],text=True)
    for line in raw.splitlines():
        index,name,memory,cc=[x.strip() for x in line.split(',',3)]
        gpus.append({'index':int(index),'name':name,'memory_mib':int(memory),'compute_capability':cc})
except (OSError, subprocess.CalledProcessError):
    pass
(p/'hardware.json').write_text(json.dumps({'platform':platform.platform(),'gpus':gpus},indent=2)+'\n')
PY

if [[ ! -d "$PRIME_ROOT/.git" ]]; then
  git init "$PRIME_ROOT"
  git -C "$PRIME_ROOT" remote add origin https://github.com/PrimeIntellect-ai/prime-rl.git
fi
git -c http.version=HTTP/1.1 -C "$PRIME_ROOT" fetch --depth=1 origin 26b3131d2716a4f8210b165df584b83b4bc54f61 > logs/prime-fetch.log 2>&1
git -C "$PRIME_ROOT" checkout --detach 26b3131d2716a4f8210b165df584b83b4bc54f61 >> logs/prime-fetch.log 2>&1
[[ "$(git -C "$PRIME_ROOT" rev-parse HEAD)" == 26b3131d2716a4f8210b165df584b83b4bc54f61 ]]
# The pinned revision records SSH submodule URLs, while the rented node only
# has authenticated HTTPS access.  Rewrite this invocation locally rather than
# changing the checkout's durable git config or requiring an SSH credential.
git -c http.version=HTTP/1.1 \
  -c url.https://github.com/.insteadOf=git@github.com: \
  -C "$PRIME_ROOT" submodule update --init --force --depth 1 \
  -- deps/verifiers deps/renderers deps/prime-envs deps/pydantic-config > logs/prime-submodules.log 2>&1
declare -A EXPECTED_SUBMODULES=(
  [deps/verifiers]=828488fffe31aa3332b9d1bd4bd9ee320e375cf1
  [deps/renderers]=f91c3e7061ce50ea405cdf54fd419a45cb51a152
  [deps/prime-envs]=1f1e050ab0cd273bca39eed5c3e5315e6a8ae9d1
  [deps/pydantic-config]=65b15dffba82d4be19efdaf8b2b9705cc1756be8
)
for submodule in "${!EXPECTED_SUBMODULES[@]}"; do
  actual="$(git -C "$PRIME_ROOT/$submodule" rev-parse HEAD 2>/dev/null || true)"
  [[ "$actual" == "${EXPECTED_SUBMODULES[$submodule]}" ]] || {
    echo "submodule $submodule is not pinned: $actual" >&2
    exit 1
  }
  if [[ "$submodule" == "deps/pydantic-config" && ! -f "$PRIME_ROOT/$submodule/pyproject.toml" ]]; then
    echo "submodule $submodule is incomplete (pyproject.toml missing)" >&2
    exit 1
  fi
done

bootstrap/bin/uv venv --allow-existing --python python3 runtime
bootstrap/bin/uv pip install --python runtime/bin/python \
  'vllm==0.28.0' 'transformers==5.6.2' 'huggingface_hub==1.30.0' > logs/inference-install.log 2>&1
if [[ -x "$PRIME_ROOT/.venv/bin/python" ]] && "$PRIME_ROOT/.venv/bin/python" -c 'import prime_rl' >/dev/null 2>&1; then
  printf '%s\n' 'Reusing existing pinned Prime environment' > logs/prime-install.log
else
  (cd "$PRIME_ROOT" && uv sync --frozen --extra gpu --extra flash-attn-3 --extra flash-attn --extra kernels --extra quack) > logs/prime-install.log 2>&1
fi

wait "$DOWNLOAD_PID"

runtime/bin/python - <<'PY' > runtime-receipt.json
import json, torch, transformers, vllm
print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'devices':torch.cuda.device_count(),
                  'transformers':transformers.__version__,'vllm':vllm.__version__},indent=2))
PY
python3 - "$ROOT" <<'PY'
import json, os, shlex, sys
from pathlib import Path
p=Path(sys.argv[1]); model=(p/'model-path.txt').read_text().strip()
values={'HF_HOME':os.environ['HF_HOME'],'UV_CACHE_DIR':os.environ['UV_CACHE_DIR'],
        'BRUSH_ROOT':os.environ['BRUSH_ROOT'],'PRIME_ROOT':os.environ['PRIME_ROOT'],
        'MODEL_DIR':model,'HF_TOKEN_PATH':os.environ['HF_TOKEN_PATH'],
        'BRUSH_MODEL_PROFILE':'qwen38_27b','WANDB_MODE':'disabled',
        'TOKENIZERS_PARALLELISM':'false'}
(p/'env.sh').write_text('\n'.join('export '+k+'='+shlex.quote(v) for k,v in values.items())+'\n')
receipt={'setup_complete':True,'training_started':False,'prime_revision':'26b3131d2716a4f8210b165df584b83b4bc54f61',
         'model_revision':'1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0','runtime_transformers':'5.6.2','runtime_vllm':'0.28.0'}
(p/'training-ready.json').write_text(json.dumps(receipt,indent=2)+'\n')
PY
echo "Bootstrap complete; no training was started."
