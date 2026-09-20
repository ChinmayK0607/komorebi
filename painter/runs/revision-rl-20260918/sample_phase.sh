#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(realpath "${1:?workspace}")
cd "$ROOT"
exec 6>sample-phase.lock;flock -n 6
trap 'echo "$?" > sample-phase.exit' EXIT
printf '{"phase":"setup"}\n' > rl-phase.json
bash setup_training.sh "$ROOT"
source env.sh
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_HOME=$(runtime/bin/python -c 'import sysconfig;print(sysconfig.get_paths()["purelib"]+"/nvidia/cu13")')
export PATH="$ROOT/runtime/bin:$CUDA_HOME/bin:$ROOT/bootstrap/bin:$PATH" LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
printf '{"phase":"sampling"}\n' > rl-phase.json
runtime/bin/python sample_actions.py > logs/rl-sampling.log 2>&1
printf '{"phase":"rendering"}\n' > rl-phase.json
runtime/bin/python render_actions.py > logs/rl-rendering.log 2>&1
printf '{"phase":"awaiting_luna_rewards","optimizer_updates":0}\n' > rl-phase.json
