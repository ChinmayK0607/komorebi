#!/usr/bin/env bash
set -Eeuo pipefail

# Generic bindings for one arm of the matched campaign.  The caller selects
# the judge model through arm_env.sh; this file deliberately has no Flash-only
# or four-update defaults.
MATCHED_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export PAINTER_MATCHED_ROOT="$MATCHED_ROOT"
export PAINTER_REPO_ROOT="${PAINTER_REPO_ROOT:-$(cd "$MATCHED_ROOT/../../.." && pwd -P)}"
export PAINTER_BASE_RUN="${PAINTER_BASE_RUN:-$PAINTER_REPO_ROOT/painter/runs/text-sft512-consistent-20260930}"
export PAINTER_BOOTSTRAP_RUN="${PAINTER_BOOTSTRAP_RUN:-$PAINTER_REPO_ROOT/painter/runs/text-pairwise-rl-20260928}"
export PAINTER_NODE_ROOT="${PAINTER_NODE_ROOT:-/root/painter}"
export PRIME_ROOT="${PRIME_ROOT:-$PAINTER_NODE_ROOT/prime}"
export PAINTER_GROUP_SIZE=4
export PAINTER_INITIAL_INFLIGHT=32
export PAINTER_MAX_INFLIGHT=32
export PAINTER_RL_MAX_STEPS=12
export PAINTER_MAX_UPDATES=12
export PAINTER_RL_CHECKPOINT_INTERVAL=4
export PAINTER_CHECKPOINT_INTERVAL=4
export ADAPTER_ONLY_CHECKPOINTS=1
export PAINTER_ACTIVATION_CPU_OFFLOAD=1
export PAINTER_REQUIRE_VALID_FINAL_ACTION=1
export PAINTER_P5_BRUSH_API_REFERENCE_V2=1
export PAINTER_INSPECT_CHOICE_FEEDBACK=1
export PAINTER_JS_SYNTAX_PREFLIGHT=1
export PAINTER_JUDGE_PROTOCOL=text-paint-sft-pairwise-consistent-v1
export PAINTER_JUDGE_FORMAT=identity-joint-side-schema-v2-generic-v1
export PAINTER_JUDGE_EFFORT=medium
export PAINTER_EVAL_SCOPE=balanced16
export PAINTER_EVAL_EXPECTED_COUNT=47
export PAINTER_EVAL_PREFERRED_STEP=12
export PAINTER_EVAL_SEED=918210
export PAINTER_GPUS_PER_NODE=2
export PAINTER_NUM_TRAIN_GPUS=1
export PAINTER_NUM_INFER_GPUS=1
export PAINTER_RENDER_WORKERS="${PAINTER_RENDER_WORKERS:-8}"
export PAINTER_EVAL_TIMEOUT_SECONDS="${PAINTER_EVAL_TIMEOUT_SECONDS:-1800}"
export PAINTER_ARCHIVE_TIMEOUT_SECONDS="${PAINTER_ARCHIVE_TIMEOUT_SECONDS:-360}"
export PAINTER_PARENT_RELEASE_RESERVE_SECONDS="${PAINTER_PARENT_RELEASE_RESERVE_SECONDS:-180}"
export PAINTER_COMPLETION_MARGIN_SECONDS="${PAINTER_COMPLETION_MARGIN_SECONDS:-120}"
export PAINTER_RESERVE_SECONDS="${PAINTER_RESERVE_SECONDS:-2520}"
export PAINTER_INITIAL_METRIC_TIMEOUT_SECONDS="${PAINTER_INITIAL_METRIC_TIMEOUT_SECONDS:-3600}"
export PAINTER_INFERENCE_READY_TIMEOUT_SECONDS="${PAINTER_INFERENCE_READY_TIMEOUT_SECONDS:-1200}"
export PAINTER_MAX_POLICY_AGE=1
export PAINTER_PAINTER_THINKING=0
export PAINTER_ORCHESTRATOR_SEED=42
export PAINTER_INFERENCE_SEED=0

apply_matched_resource_limits() {
  local hard soft target cpu limit chosen
  hard="$(ulimit -H -n)"
  if [[ "$hard" == unlimited ]]; then hard=131072; fi
  [[ "$hard" =~ ^[0-9]+$ && "$hard" -ge 1 ]] || {
    echo "invalid hard nofile limit: $hard" >&2; return 1;
  }
  target="$hard"; (( target > 131072 )) && target=131072
  ulimit -S -n "$target" 2>/dev/null || {
    echo "could not set soft nofile limit to $target" >&2; return 1;
  }
  soft="$(ulimit -S -n)"
  [[ "$soft" =~ ^[0-9]+$ && "$soft" -ge 1 ]] || return 1
  cpu="$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)"
  [[ "$cpu" =~ ^[1-9][0-9]*$ ]] || cpu=1
  limit=$((soft / 64)); (( limit < 1 )) && limit=1
  chosen="$cpu"; (( chosen > 32 )) && chosen=32; (( chosen > limit )) && chosen="$limit"
  UV_CONCURRENT_INSTALLS="${UV_CONCURRENT_INSTALLS:-$chosen}"
  [[ "$UV_CONCURRENT_INSTALLS" =~ ^[1-9][0-9]*$ && "$UV_CONCURRENT_INSTALLS" -le "$chosen" ]] || {
    echo "UV_CONCURRENT_INSTALLS must be between 1 and $chosen" >&2; return 1;
  }
  export PAINTER_NOFILE_HARD="$hard" PAINTER_NOFILE_SOFT="$soft" PAINTER_NOFILE_TARGET="$target"
  export PAINTER_UV_CPU_COUNT="$cpu" PAINTER_UV_CONCURRENCY_LIMIT="$chosen" UV_CONCURRENT_INSTALLS
}

validate_matched_bindings() {
  [[ "${PAINTER_RUN_NAME:-}" == rl-judge-matched-20261004-* ]] || { echo 'matched run name is not bound' >&2; return 1; }
  [[ "$PAINTER_JUDGE_EFFORT" == medium && "$PAINTER_JUDGE_FORMAT" == identity-joint-side-schema-v2-generic-v1 ]] || {
    echo 'matched judge effort/format binding is invalid' >&2; return 1;
  }
  [[ "$PAINTER_JUDGE_PROTOCOL" == text-paint-sft-pairwise-consistent-v1 ]] || { echo 'matched protocol is invalid' >&2; return 1; }
  [[ "$PAINTER_GROUP_SIZE" == 4 && "$PAINTER_INITIAL_INFLIGHT" == 32 && "$PAINTER_MAX_INFLIGHT" == 32 ]] || {
    echo 'matched campaign requires group4 and inflight32' >&2; return 1;
  }
  [[ "$PAINTER_RL_MAX_STEPS" == 12 && "$PAINTER_RL_CHECKPOINT_INTERVAL" == 4 && "$PAINTER_CHECKPOINT_INTERVAL" == 4 ]] || {
    echo 'matched campaign requires twelve updates and checkpoints 4/8/12' >&2; return 1;
  }
  [[ "$PAINTER_MAX_POLICY_AGE" == 1 && "$ADAPTER_ONLY_CHECKPOINTS" == 1 && "$PAINTER_ACTIVATION_CPU_OFFLOAD" == 1 ]] || {
    echo 'matched optimizer/offload binding is invalid' >&2; return 1;
  }
  [[ "$PAINTER_REQUIRE_VALID_FINAL_ACTION" == 1 && "$PAINTER_P5_BRUSH_API_REFERENCE_V2" == 1 ]] || {
    echo 'strict final action/API v2 binding is required' >&2; return 1;
  }
  [[ "$PAINTER_GPUS_PER_NODE" == 2 && "$PAINTER_NUM_TRAIN_GPUS" == 1 && "$PAINTER_NUM_INFER_GPUS" == 1 ]] || {
    echo 'matched campaign requires one trainer and one inference GPU' >&2; return 1;
  }
  [[ "$PAINTER_EVAL_SCOPE" == balanced16 && "$PAINTER_EVAL_EXPECTED_COUNT" == 47 && "$PAINTER_EVAL_PREFERRED_STEP" == 12 ]] || {
    echo 'matched campaign evaluation binding is invalid' >&2; return 1;
  }
  [[ "$PAINTER_RESERVE_SECONDS" =~ ^[0-9]+$ ]] || return 1
  (( PAINTER_RESERVE_SECONDS >= PAINTER_EVAL_TIMEOUT_SECONDS + PAINTER_ARCHIVE_TIMEOUT_SECONDS + PAINTER_PARENT_RELEASE_RESERVE_SECONDS + PAINTER_COMPLETION_MARGIN_SECONDS + 60 )) || {
    echo 'matched completion reserve is too small' >&2; return 1;
  }
}

validate_matched_config() {
  local python_bin="${1:-python3}" config="${2:?config required}" receipt="${3:?receipt required}"
  "$python_bin" - "$config" "$receipt" <<'PY'
import json, sys, tomllib
from pathlib import Path
config = tomllib.loads(Path(sys.argv[1]).read_text())
receipt = json.loads(Path(sys.argv[2]).read_text())
actual = {
    "updates": config.get("orchestrator", {}).get("max_steps"),
    "checkpoint_interval": config.get("ckpt", {}).get("interval"),
    "batch_size": config.get("orchestrator", {}).get("batch_size"),
    "group_size": config.get("orchestrator", {}).get("group_size"),
    "initial_inflight": config.get("orchestrator", {}).get("concurrency", {}).get("initial_inflight"),
    "max_inflight": config.get("orchestrator", {}).get("concurrency", {}).get("max_inflight"),
    "max_policy_age": config.get("orchestrator", {}).get("max_off_policy_steps"),
    "activation_cpu_offload": config.get("trainer", {}).get("model", {}).get("ac_offloading") == {"pin_memory": True, "max_inflight_activations": 1},
}
expected = {"updates": 12, "checkpoint_interval": 4, "batch_size": 16, "group_size": 4,
            "initial_inflight": 32, "max_inflight": 32, "max_policy_age": 1,
            "activation_cpu_offload": True}
if actual != expected:
    raise SystemExit(f"matched native config mismatch: {actual!r}")
for key, value in (("batch_size", 16), ("group_size", 4), ("initial_inflight", 32),
                   ("max_inflight", 32), ("max_off_policy_steps", 1),
                   ("checkpoint_interval", 4)):
    if receipt.get(key) != value:
        raise SystemExit(f"matched receipt mismatch: {key}")
if receipt.get("mode") != "recovered_policy" or receipt.get("optimizer_resume") is not False:
    raise SystemExit("matched arm does not use a fresh optimizer from RL16")
if receipt.get("activation_cpu_offload") is not True:
    raise SystemExit("matched arm does not use CPU activation offload")
PY
}
