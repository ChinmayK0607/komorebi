#!/usr/bin/env bash
# Bootstrap a reusable Linux inference/render node for matched painter evals.
#
# Usage: bash eval_node.sh /absolute/node/workspace
# The default `minimal` mode installs vLLM plus the pinned native verifier and
# renderer packages without Prime's trainer extras.  Set EVAL_INSTALL_MODE=prime
# to use Prime's complete frozen inference environment when a native launcher
# needs prime-rl itself.  This script never starts a model server or evaluator.
set -Eeuo pipefail

ROOT=$(realpath "${1:?node workspace required}")
MODE=${EVAL_INSTALL_MODE:-minimal}
[[ $(uname -s) == Linux ]]
[[ "$MODE" == minimal || "$MODE" == prime ]]
cd "$ROOT"
[[ -x "$ROOT/setup_node.sh" ]] || { echo "setup_node.sh is missing from $ROOT" >&2; exit 1; }
[[ -f "$ROOT/eval_model_receipt.py" ]] || { echo "eval_model_receipt.py is missing from $ROOT" >&2; exit 1; }

mkdir -p "$ROOT/logs" "$ROOT/cache" "$ROOT/private"
chmod 700 "$ROOT/private"
exec 9>"$ROOT/eval-node.lock"
flock -n 9 || { echo "an eval bootstrap is already running" >&2; exit 0; }
rm -f "$ROOT/eval-node.exit" "$ROOT/eval-ready.json"
trap 'printf "%s\n" "$?" > "$ROOT/eval-node.exit"' EXIT

export HF_HOME="${HF_HOME:-$ROOT/cache/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/cache/uv}"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

if [[ ! -x "$ROOT/bootstrap/bin/uv" ]]; then
  python3 -m venv "$ROOT/bootstrap"
  "$ROOT/bootstrap/bin/pip" -q install 'uv==0.11.1'
fi
UV="$ROOT/bootstrap/bin/uv"

# Keep the small bootstrap interpreter able to verify public HF snapshots
# before the inference environment exists.  No token is needed for these
# public revisions, and no credential is written to a receipt or env file.
"$UV" pip install --python "$ROOT/bootstrap/bin/python" \
  'huggingface_hub==1.30.0' 'hf_xet==1.6.0' > "$ROOT/logs/hub-install.log" 2>&1

# Start public HF transfer/verification as soon as the bootstrap interpreter
# exists.  It is safe to overlap with dependency installation and reuses any
# cache that an operator may already have warmed on the node.
"$ROOT/bootstrap/bin/python" "$ROOT/eval_model_receipt.py" "$ROOT" \
  > "$ROOT/logs/model-verification.log" 2>&1 &
MODEL_PID=$!

# Rendering is shared with the ordinary node bootstrap and has no CUDA or
# model dependency.  Keep it concurrent with the package/model setup.
bash "$ROOT/setup_node.sh" "$ROOT" render > "$ROOT/logs/render-setup.log" 2>&1 &
RENDER_PID=$!

PRIME_ROOT=${PRIME_ROOT:-$ROOT/prime-rl}
PRIME_COMMIT=26b3131d2716a4f8210b165df584b83b4bc54f61
VERIFIERS_COMMIT=828488fffe31aa3332b9d1bd4bd9ee320e375cf1
export PRIME_ROOT

if [[ ! -d "$PRIME_ROOT/.git" ]]; then
  mkdir -p "$(dirname "$PRIME_ROOT")"
  git init "$PRIME_ROOT" >/dev/null
  git -C "$PRIME_ROOT" remote add origin https://github.com/PrimeIntellect-ai/prime-rl.git
fi
git -C "$PRIME_ROOT" fetch --depth=1 origin "$PRIME_COMMIT" >/dev/null
git -C "$PRIME_ROOT" checkout --detach "$PRIME_COMMIT" >/dev/null
[[ $(git -C "$PRIME_ROOT" rev-parse HEAD) == "$PRIME_COMMIT" ]]
git -c http.version=HTTP/1.1 -c url.https://github.com/.insteadOf=git@github.com: \
  -C "$PRIME_ROOT" submodule update --init --depth 1 -- \
  deps/verifiers deps/renderers deps/prime-envs deps/pydantic-config
[[ $(git -C "$PRIME_ROOT/deps/verifiers" rev-parse HEAD) == "$VERIFIERS_COMMIT" ]]

# This is intentionally a small inference environment.  The native verifier
# and renderer are installed from the exact Prime submodules without pulling
# the Prime trainer, optimizer, or distributed extras.
if [[ "$MODE" == minimal ]]; then
  "$UV" venv --allow-existing --python python3 "$ROOT/inference-env"
  INFERENCE_PY="$ROOT/inference-env/bin/python"
  VLLM_WHEEL="https://github.com/vllm-project/vllm/releases/download/v0.28.0/vllm-0.28.0-cp38-abi3-manylinux_2_28_x86_64.whl"
  "$UV" pip install --python "$INFERENCE_PY" \
    --index-url 'https://download.pytorch.org/whl/cu130' \
    --extra-index-url 'https://pypi.org/simple' \
    'torch==2.13.0' "$VLLM_WHEEL" 'transformers==5.6.2' 'huggingface_hub==1.28.0' \
    'pillow==12.3.0' 'numpy>=2.2.6' 'pydantic>=2.12.3' \
    'openai>=2.54.0,<3.0.0' 'httpx>=0.27.0' 'httpx2>=2.12.0' \
    'aiohttp>=3.14.1' 'msgpack>=1.1.2' 'pyzmq>=27.1.0' \
    'tenacity>=8.5.0' 'rich>=11.0.0' 'loguru>=0.7.0' \
    'setuptools>=77.0.3,<81.0.0' \
    'gepa>=0.0.6' 'math-verify>=0.8.0' 'anthropic>=1.0.0' \
    'prime-tunnel>=0.1.8' 'prime-sandboxes>=0.2.39' \
    'prime-pydantic-config>=0.4.3' 'tiktoken' 'jinja2' \
    'openai-harmony>=0.0.4' 'typing_extensions' 'tomli-w' \
    > "$ROOT/logs/inference-install.log" 2>&1
  "$UV" pip install --python "$INFERENCE_PY" --no-deps \
    "$PRIME_ROOT/deps/pydantic-config" \
    "$PRIME_ROOT/deps/renderers" \
    "$PRIME_ROOT/deps/verifiers" \
    >> "$ROOT/logs/inference-install.log" 2>&1
else
  (cd "$PRIME_ROOT"; "$UV" sync --frozen --package prime-rl --extra gpu) \
    > "$ROOT/logs/inference-install.log" 2>&1
  INFERENCE_PY="$PRIME_ROOT/.venv/bin/python"
fi
[[ -x "$INFERENCE_PY" ]]
[[ -x "${INFERENCE_PY%/python}/vllm" ]]

# Install the active taskset as a real top-level v1 plugin.  The verifier
# loader resolves `taskset.id = native_painting` through this package; merely
# leaving a loose source file beside the TOML is not sufficient when `eval`
# runs from Prime's checkout.  The project has no dependencies because the
# exact verifier/renderer pins are already in this environment.
"$UV" pip install --python "$INFERENCE_PY" --no-deps --no-build-isolation -e "$ROOT" \
  >> "$ROOT/logs/inference-install.log" 2>&1
"$INFERENCE_PY" - <<'PY'
import importlib
import verifiers.v1 as vf

module = importlib.import_module("native_painting")
if not issubclass(module.NativePaintingTaskset, vf.Taskset):
    raise SystemExit("native_painting.NativePaintingTaskset is not a v1 Taskset")
if not issubclass(module.NativePaintingEnv, vf.Env):
    raise SystemExit("native_painting.NativePaintingEnv is not a v1 Env")
PY

wait "$MODEL_PID"
wait "$RENDER_PID"

export INFERENCE_PY
export INFERENCE_ROOT="${INFERENCE_PY%/bin/python}"
export PLAYWRIGHT_BROWSERS_PATH="$ROOT/browsers"
export MODEL_DIR="$(< "$ROOT/model-path.txt")"
export SFT_ADAPTER="$(< "$ROOT/sft-adapter-path.txt")"
export RL_ADAPTER="$(< "$ROOT/rl-adapter-path.txt")"
python3 - "$ROOT" "$MODE" "$PRIME_COMMIT" "$VERIFIERS_COMMIT" "$INFERENCE_PY" <<'PY'
import json
import shlex
import sys
from pathlib import Path

root = Path(sys.argv[1])
mode, prime_commit, verifier_commit, inference_python = sys.argv[2:]
values = {
    "HF_HOME": str(root / "cache/huggingface"),
    "UV_CACHE_DIR": str(root / "cache/uv"),
    "PLAYWRIGHT_BROWSERS_PATH": str(root / "browsers"),
    "PRIME_ROOT": str(root / "prime-rl"),
    "INFERENCE_ROOT": str(Path(inference_python).parent.parent),
    "INFERENCE_PY": inference_python,
    "MODEL_DIR": (root / "model-path.txt").read_text().strip(),
    "SFT_ADAPTER": (root / "sft-adapter-path.txt").read_text().strip(),
    "RL_ADAPTER": (root / "rl-adapter-path.txt").read_text().strip(),
    "TOKENIZERS_PARALLELISM": "false",
    "WANDB_MODE": "disabled",
}
(root / "env.sh").write_text("".join(f"export {key}={shlex.quote(value)}\n" for key, value in values.items()))
ready = {
    "schema": "painter.controlled-comparison-eval-ready.v1",
    "setup_complete": True,
    "training_started": False,
    "install_mode": mode,
    "prime_commit": prime_commit,
    "verifiers_commit": verifier_commit,
    "model_receipt": "model-receipt.json",
    "server_launch": "eval_serve.sh",
}
(root / "eval-ready.json").write_text(json.dumps(ready, indent=2, sort_keys=True) + "\n")
print(json.dumps(ready, sort_keys=True))
PY
echo "Ready: eval environment. Launch remains a separate command: bash eval_serve.sh $ROOT"
