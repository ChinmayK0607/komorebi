#!/usr/bin/env bash
set -Eeuo pipefail

# Follow-up to the shared SFT bootstrap.  Prime's asynchronous local launcher
# needs the disaggregated vLLM router extra; the SFT bootstrap intentionally
# did not install it.  This script only prepares the pinned environment and
# exits before any model, server, or training process is started.
ROOT="$(cd "${1:?usage: setup_rl_runtime.sh RUN_ROOT}" && pwd -P)"
PRIME_ROOT="$(cd "${PRIME_ROOT:?set PRIME_ROOT to the pinned Prime checkout}" && pwd -P)"
UV_BIN="${UV_BIN:-uv}"
EXPECTED_COMMIT="26b3131d2716a4f8210b165df584b83b4bc54f61"
mkdir -p "$ROOT/logs" "$ROOT/private"
chmod 700 "$ROOT/private"
[[ -d "$PRIME_ROOT/.git" ]] || { echo "missing Prime checkout: $PRIME_ROOT" >&2; exit 1; }
[[ "$(git -C "$PRIME_ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || {
  echo "Prime checkout is not pinned to $EXPECTED_COMMIT" >&2
  exit 1
}

# Rendering is independent of Prime.  When the operator supplies a staged
# render root, run the repository's locked setup there and expose its exact
# renderer/vendor/browser paths through symlinks in this run root.  Leaving the
# variable unset keeps this script usable for a pre-existing ready renderer;
# launch_rl.sh still fails closed if those paths are absent.
if [[ -n "${PAINTER_RENDER_ROOT:-}" ]]; then
  RENDER_ROOT="$(cd "$PAINTER_RENDER_ROOT" && pwd -P)"
  # The staged source may be copied with mode 0644.  Invoke it explicitly
  # through bash; requiring an executable bit makes a valid renderer bundle
  # fail before any setup work begins.
  [[ -f "$RENDER_ROOT/setup_node.sh" ]] || { echo "render setup_node.sh is missing: $RENDER_ROOT" >&2; exit 1; }
  bash "$RENDER_ROOT/setup_node.sh" "$RENDER_ROOT" render >"$ROOT/logs/render-setup.log" 2>&1
  for item in vendor renderer-env browsers; do
    [[ -e "$RENDER_ROOT/$item" ]] || { echo "render asset is missing: $RENDER_ROOT/$item" >&2; exit 1; }
    if [[ ! -e "$ROOT/$item" ]]; then
      ln -s "$RENDER_ROOT/$item" "$ROOT/$item"
    elif [[ ! -L "$ROOT/$item" && "$ROOT/$item" != "$RENDER_ROOT/$item" ]]; then
      echo "run render path already exists and is not the staged asset: $ROOT/$item" >&2
      exit 1
    fi
  done
fi
# Repair a partially completed bootstrap over HTTPS when the pinned checkout's
# submodule URLs use SSH.  This does not write credentials or mutate git config.
git -c http.version=HTTP/1.1 \
  -c url.https://github.com/.insteadOf=git@github.com: \
  -C "$PRIME_ROOT" submodule update --init --force --depth 1 \
  -- deps/verifiers deps/renderers deps/prime-envs deps/pydantic-config \
  >"$ROOT/logs/prime-submodules.log" 2>&1
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

EXTRAS=(--extra gpu --extra flash-attn --extra kernels --extra quack --extra disagg)
if command -v nvidia-smi >/dev/null 2>&1 \
  && nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | grep -q '^9\.'; then
  EXTRAS+=(--extra flash-attn-3)
fi

for attempt in 1 2 3; do
  if (cd "$PRIME_ROOT" && "$UV_BIN" sync --frozen "${EXTRAS[@]}") \
      >"$ROOT/logs/prime-rl-install-attempt-${attempt}.log" 2>&1; then
    break
  fi
  if [[ "$attempt" == 3 ]]; then
    cp "$ROOT/logs/prime-rl-install-attempt-${attempt}.log" "$ROOT/logs/prime-rl-install.log"
    exit 1
  fi
  sleep $((attempt * 10))
done
cp "$ROOT/logs/prime-rl-install-attempt-${attempt}.log" "$ROOT/logs/prime-rl-install.log"
[[ -x "$PRIME_ROOT/.venv/bin/vllm-router" ]] || {
  echo "pinned Prime environment lacks vllm-router after disagg install" >&2
  exit 1
}

"$PRIME_ROOT/.venv/bin/python" - "$ROOT" "$PRIME_ROOT" <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path

root, prime = map(Path, sys.argv[1:])
versions = subprocess.check_output(
    [str(prime / ".venv/bin/python"), "-c", "import torch,transformers,vllm; print(torch.__version__); print(transformers.__version__); print(vllm.__version__)"],
    text=True,
).splitlines()
(root / "rl-runtime-receipt.json").write_text(json.dumps({
    "setup_complete": True,
    "training_started": False,
    "prime_revision": subprocess.check_output(["git", "-C", str(prime), "rev-parse", "HEAD"], text=True).strip(),
    "torch": versions[0],
    "transformers": versions[1],
    "vllm": versions[2],
    "router": str(prime / ".venv/bin/vllm-router"),
    "extras": ["gpu", "flash-attn", "kernels", "quack", "disagg"],
}, indent=2) + "\n")
PY
printf '%s\n' 'RL runtime ready; no server or training process was started.'
