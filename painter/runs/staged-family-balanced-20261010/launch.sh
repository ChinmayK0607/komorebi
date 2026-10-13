#!/usr/bin/env bash
set -Eeuo pipefail

# Foreground-only finite wrapper. Prime owns any internal child processes;
# this shell never backgrounds, daemonizes, submits, or schedules work.
RUN_SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
NODE_ROOT="$(cd "${1:?usage: launch.sh NODE_ROOT}" && pwd -P)"
REPO_ROOT="$(cd "${PAINTER_REPO_ROOT:?set PAINTER_REPO_ROOT to the checkout}" && pwd -P)"
PRIME_ROOT="$(cd "${PRIME_ROOT:?set PRIME_ROOT to the pinned Prime checkout}" && pwd -P)"
MATERIALIZED_ROOT="$(cd "${PAINTER_MATERIALIZED_ROOT:?set PAINTER_MATERIALIZED_ROOT to materialized evidence}" && pwd -P)"
PRIME_COMMIT="26b3131d2716a4f8210b165df584b83b4bc54f61"
RUN_ID="staged-family-balanced-20261010"
BASE_MODEL_REVISION="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
POLICY_REVISION="1227a2427a8ae16014d2fdca4d1adb6855d3fa1c"
POLICY_PREFIX="rl-judge-matched-20261004-flash/step_12"
POLICY_SHA256="c70c47ec0d6d7172b3bc25c4f54178919cf7c079cd9652fff3bc76db9e1eca79"
FROZEN_SFT_SHA256="bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740"
CONFIG="${PAINTER_RL_CONFIG:-$NODE_ROOT/native.toml}"
PYTHON="${PAINTER_PRIME_PYTHON:-$PRIME_ROOT/.venv/bin/python}"
PRIME_BIN="$PRIME_ROOT/.venv/bin"
RUN_WRAPPER="${PAINTER_PRIME_RUN_WRAPPER:-$RUN_SOURCE_ROOT/run_from_sft.py}"
TRAIN_WRAPPER="${PAINTER_PRIME_TRAIN_WRAPPER:-$RUN_SOURCE_ROOT/train_from_sft.py}"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the local base snapshot}"
POLICY_ADAPTER="${WATERCOLOUR_INIT_ADAPTER:?set WATERCOLOUR_INIT_ADAPTER to the verified Flash12 adapter}"
FROZEN_ADAPTER="${PAINTER_FROZEN_SFT_ADAPTER:?set PAINTER_FROZEN_SFT_ADAPTER to the verified SFT512 adapter}"
EXPECTED_BRUSH_ROOT="$REPO_ROOT/painter/rl_helpers"
BRUSH_ROOT="${BRUSH_ROOT:-$EXPECTED_BRUSH_ROOT}"
RENDERER_ROOT="${PAINTER_RENDERER_ROOT:?set PAINTER_RENDERER_ROOT to the verified renderer root}"
GATEWAY_ENV_FILE="${PAINTER_GATEWAY_ENV_FILE:?set PAINTER_GATEWAY_ENV_FILE to the owner-only judge credential file}"
RENDERER_PATH="${PAINTER_RENDERER_PATH:?set PAINTER_RENDERER_PATH to the verified renderer}"
RENDERER_PYTHON_PATH="${PAINTER_RENDERER_PYTHON_PATH:?set PAINTER_RENDERER_PYTHON_PATH to the renderer interpreter}"
TIMEOUT_SECONDS="${PAINTER_RUN_TIMEOUT_SECONDS:-172800}"

[[ "$(uname -s)" == Linux ]] || { echo 'staged Prime launch requires Linux' >&2; exit 1; }
[[ $# == 1 ]] || { echo 'usage: launch.sh NODE_ROOT' >&2; exit 2; }
[[ "$PYTHON" == "$PRIME_BIN/python" && -x "$PYTHON" ]] || { echo "missing pinned Prime Python: $PYTHON" >&2; exit 1; }
[[ -d "$BRUSH_ROOT" && ! -L "$BRUSH_ROOT" && "$(cd "$BRUSH_ROOT" && pwd -P)" == "$EXPECTED_BRUSH_ROOT" ]] || { echo "BRUSH_ROOT must be the reviewed helper tree: $EXPECTED_BRUSH_ROOT" >&2; exit 1; }
for required in \
    "$BRUSH_ROOT/integrations/prime/watercolour_sft/init_adapter.py" \
    "$BRUSH_ROOT/integrations/prime/watercolour_sft/export_adapter.py" \
    "$BRUSH_ROOT/integrations/prime/watercolour_sft/model_profiles.py" \
    "$BRUSH_ROOT/scripts/upload_checkpoints.py"; do
    [[ -f "$required" && ! -L "$required" ]] || { echo "reviewed trainer helper is missing or symlinked: $required" >&2; exit 1; }
done
export BRUSH_MODEL_PROFILE=qwen38_27b
for executable in inference trainer orchestrator; do
    [[ -f "$PRIME_BIN/$executable" && ! -L "$PRIME_BIN/$executable" && -x "$PRIME_BIN/$executable" ]] || {
        echo "missing pinned Prime console script: $PRIME_BIN/$executable" >&2
        exit 1
    }
done
export PATH="$PRIME_BIN${PATH:+:$PATH}"
[[ -f "$NODE_ROOT/native.toml" && -f "$NODE_ROOT/prepare-receipt.json" && -f "$NODE_ROOT/canary-receipt.json" ]] || { echo 'preparation and canary receipts are required' >&2; exit 1; }
[[ -f "$MATERIALIZED_ROOT/train-manifest.json" && -f "$MATERIALIZED_ROOT/heldout-manifest.json" ]] || { echo 'materialized manifests are missing' >&2; exit 1; }
[[ -f "$RUN_WRAPPER" && -f "$TRAIN_WRAPPER" ]] || { echo 'reviewed Prime wrappers are missing' >&2; exit 1; }
[[ "$(realpath "$RUN_WRAPPER")" == "$RUN_SOURCE_ROOT/run_from_sft.py" ]] || { echo 'the staged run wrapper must remain run-owned' >&2; exit 1; }
[[ "$(realpath "$TRAIN_WRAPPER")" == "$RUN_SOURCE_ROOT/train_from_sft.py" ]] || { echo 'the staged trainer wrapper is required' >&2; exit 1; }
# Do not let an inherited operator environment silently substitute the older
# photo trainer. The run-owned wrappers are the only admissible targets.
RUN_WRAPPER="$RUN_SOURCE_ROOT/run_from_sft.py"
TRAIN_WRAPPER="$RUN_SOURCE_ROOT/train_from_sft.py"
[[ "$(git -C "$PRIME_ROOT" rev-parse HEAD)" == "$PRIME_COMMIT" ]] || { echo "Prime checkout is not pinned to $PRIME_COMMIT" >&2; exit 1; }
[[ -d "$MODEL_DIR" && "$(basename "$MODEL_DIR")" == "$BASE_MODEL_REVISION" ]] || { echo "MODEL_DIR must end in $BASE_MODEL_REVISION" >&2; exit 1; }
[[ -f "$POLICY_ADAPTER/adapter_model.safetensors" && -f "$FROZEN_ADAPTER/adapter_model.safetensors" ]] || { echo 'verified adapters are incomplete' >&2; exit 1; }
[[ "$(cd "$POLICY_ADAPTER" && pwd -P)" != "$(cd "$FROZEN_ADAPTER" && pwd -P)" ]] || { echo 'policy and frozen adapters must be distinct' >&2; exit 1; }
[[ -f "$RENDERER_PATH" && -x "$RENDERER_PYTHON_PATH" ]] || { echo 'verified renderer paths are missing' >&2; exit 1; }
[[ -d "$RENDERER_ROOT" && ! -L "$RENDERER_ROOT" ]] || { echo 'verified renderer root is missing or symlinked' >&2; exit 1; }
[[ -f "$GATEWAY_ENV_FILE" && ! -L "$GATEWAY_ENV_FILE" && "$(stat -c %a "$GATEWAY_ENV_FILE")" == 600 ]] || { echo 'judge credential file must be owner-only mode 600' >&2; exit 1; }
[[ -n "${HF_CHECKPOINT_REPO:-}" && -n "${HF_CHECKPOINT_RUN:-}" ]] || { echo 'HF_CHECKPOINT_REPO and unique HF_CHECKPOINT_RUN are required before launch' >&2; exit 1; }
[[ "$HF_CHECKPOINT_RUN" =~ ^[A-Za-z0-9._-]+$ ]] || { echo 'HF_CHECKPOINT_RUN must be a safe single path component' >&2; exit 1; }
[[ "${ADAPTER_ONLY_CHECKPOINTS:-}" == 1 ]] || { echo 'ADAPTER_ONLY_CHECKPOINTS=1 is required for public portable checkpoints' >&2; exit 1; }
HF_TOKEN_FILE="${HF_HOME:-$HOME/.cache/huggingface}/token"
[[ -f "$HF_TOKEN_FILE" && ! -L "$HF_TOKEN_FILE" && "$(stat -c %a "$HF_TOKEN_FILE")" == 600 ]] || { echo 'owner-only Hugging Face token file is required before launch' >&2; exit 1; }
[[ "$TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ && "$TIMEOUT_SECONDS" -le 604800 ]] || { echo 'PAINTER_RUN_TIMEOUT_SECONDS must be 1..604800' >&2; exit 2; }
GUARD_NAMESPACE="${PAINTER_GUARD_NAMESPACE:-}"
GUARD_PARENT="$NODE_ROOT/launch-guards"
[[ -n "$GUARD_NAMESPACE" ]] || { echo 'PAINTER_GUARD_NAMESPACE must be supplied by launch_prepared.py' >&2; exit 2; }
case "$GUARD_NAMESPACE" in
    "$GUARD_PARENT"/*) ;;
    *) echo 'PAINTER_GUARD_NAMESPACE must be a child of the prepared launch-guards directory' >&2; exit 2 ;;
esac
GUARD_ID="${GUARD_NAMESPACE#"$GUARD_PARENT/"}"
[[ "$GUARD_ID" =~ ^[0-9a-f]{32}$ ]] || { echo 'PAINTER_GUARD_NAMESPACE must contain one UUID launch identifier' >&2; exit 2; }
CANARY_MODE="${PAINTER_CANARY_MODE:-}"
if [[ "$CANARY_MODE" == "real_backward" ]]; then
    ARTIFACT_ROOT="$GUARD_NAMESPACE"
elif [[ -z "$CANARY_MODE" ]]; then
    ARTIFACT_ROOT="$NODE_ROOT"
else
    echo "unsupported PAINTER_CANARY_MODE: $CANARY_MODE" >&2
    exit 2
fi
[[ ! -e "$ARTIFACT_ROOT/launch.started" && ! -e "$ARTIFACT_ROOT/async-output" && ! -e "$ARTIFACT_ROOT/runtime-state.json" && ! -e "$ARTIFACT_ROOT/rl-rollouts" ]] || {
    echo "existing staged output in the selected launch namespace: $ARTIFACT_ROOT" >&2
    exit 1
}
CONFIG="$NODE_ROOT/native.toml"
export PYTHONPATH="$RUN_SOURCE_ROOT:$REPO_ROOT:$REPO_ROOT/painter:$(dirname "$RUN_WRAPPER")${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$CANARY_MODE" == "real_backward" ]]; then
    mkdir -p "$GUARD_NAMESPACE"
    # Prime writes rollout artifacts as root, while the renderer intentionally
    # crosses into the unprivileged painter account. Keep file contents private
    # by default but expose directory traversal to the exact run-owned paths.
    chmod 0711 "$NODE_ROOT"
    chmod 0755 "$GUARD_PARENT" "$GUARD_NAMESPACE"
    ACCESS_PROBE="$GUARD_NAMESPACE/.painter-access-probe/episode/turn-1"
    mkdir -p "$ACCESS_PROBE"
    chmod 0755 "$GUARD_NAMESPACE/.painter-access-probe" "$GUARD_NAMESPACE/.painter-access-probe/episode"
    chmod 0777 "$ACCESS_PROBE"
    printf '%s\n' 'function setup(){createCanvas(8,8);noLoop();}' > "$ACCESS_PROBE/program.js"
    chmod 0644 "$ACCESS_PROBE/program.js"
    runuser -u painter -- test -r "$ACCESS_PROBE/program.js" || { echo 'painter cannot traverse/read generated rollout paths' >&2; exit 1; }
    rm -rf "$GUARD_NAMESPACE/.painter-access-probe"
    CONFIG="$GUARD_NAMESPACE/native.toml"
    "$PYTHON" - "$NODE_ROOT/native.toml" "$CONFIG" "$ARTIFACT_ROOT" <<'PY'
import sys
import tomllib
from pathlib import Path
from prepare_config import render_toml

source, target, artifact_root = map(Path, sys.argv[1:])
config = tomllib.loads(source.read_text(encoding="utf-8"))
config["output_dir"] = str((artifact_root / "async-output").resolve())
taskset = config["orchestrator"]["train"]["source"][0]["env"]["taskset"]
taskset["output_root"] = str((artifact_root / "rl-rollouts").resolve())
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(render_toml(config), encoding="utf-8")
PY
fi
if [[ "$CANARY_MODE" == "real_backward" ]]; then
    cd "$ARTIFACT_ROOT"
fi

"$PYTHON" - "$POLICY_ADAPTER" "$POLICY_SHA256" "$FROZEN_ADAPTER" "$FROZEN_SFT_SHA256" "$NODE_ROOT/prepare-receipt.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
policy, policy_expected, frozen, frozen_expected, receipt_path = map(Path, sys.argv[1:])
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def hf_file(path, description):
    if not path.exists() and not path.is_symlink():
        raise SystemExit(f"{description} is missing: {path}")
    if not path.is_symlink():
        if not path.is_file():
            raise SystemExit(f"{description} is not a regular file: {path}")
        return path
    cache_root = None
    current = path.parent
    while current != current.parent:
        if current.name == "snapshots":
            cache_root = current.parent.absolute()
            break
        current = current.parent
    if cache_root is None or not (cache_root / "blobs").is_dir():
        raise SystemExit(f"{description} is not below a trusted Hugging Face snapshot cache: {path}")
    try:
        path.relative_to(cache_root)
        resolved_cache_root = cache_root.resolve(strict=True)
        resolved = path.resolve(strict=True)
        resolved.relative_to(resolved_cache_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"{description} symlink escapes its Hugging Face cache root: {path}") from exc
    if not resolved.is_file():
        raise SystemExit(f"{description} symlink does not resolve to a regular file: {path}")
    return path
if digest(hf_file(policy / "adapter_model.safetensors", "Flash12 policy adapter weights")) != sys.argv[2]:
    raise SystemExit("Flash12 policy adapter SHA-256 mismatch")
if digest(hf_file(frozen / "adapter_model.safetensors", "frozen SFT adapter weights")) != sys.argv[4]:
    raise SystemExit("frozen SFT512 adapter SHA-256 mismatch")
hf_file(policy / "adapter_config.json", "Flash12 policy adapter config")
hf_file(frozen / "adapter_config.json", "frozen SFT adapter config")
receipt = json.loads(receipt_path.read_text())
if receipt.get("run_id") != "staged-family-balanced-20261010" or receipt.get("status") != "prepared_no_process_started":
    raise SystemExit("node preparation receipt is not the staged finite preparation")
if Path(receipt.get("policy_initializer", {}).get("path", "")).resolve() != policy.resolve():
    raise SystemExit("Flash12 policy path does not match node preparation receipt")
if Path(receipt.get("frozen_sft", {}).get("path", "")).resolve() != frozen.resolve():
    raise SystemExit("frozen SFT path does not match node preparation receipt")
if receipt.get("policy_initializer", {}).get("weights_sha256") != sys.argv[2] or receipt.get("frozen_sft", {}).get("weights_sha256") != sys.argv[4]:
    raise SystemExit("adapter bytes do not match node preparation receipt")
PY

"$PYTHON" - "$CONFIG" "$MATERIALIZED_ROOT" "$NODE_ROOT/prepare-receipt.json" "$NODE_ROOT/canary-receipt.json" <<'PY'
import json
import sys
import tomllib
from pathlib import Path
config_path, materialized_root, prepare_path, canary_path = map(Path, sys.argv[1:])
config = tomllib.loads(config_path.read_text())
taskset = config["orchestrator"]["train"]["source"][0]["env"]["taskset"]
if config.get("max_steps") != 20 or config.get("trainer", {}).get("max_steps") != 20:
    raise SystemExit("Prime config is not bounded to 20 updates")
if config["orchestrator"].get("batch_size") != 24 or config["orchestrator"].get("group_size") != 4 or config["orchestrator"].get("constant_trainer_batch_size") is not True:
    raise SystemExit("Prime config batch/group shape is not 24/4")
if Path(taskset["root"]).resolve() != materialized_root.resolve() or taskset.get("manifest") != "train-manifest.json":
    raise SystemExit("Prime taskset is not bound to materialized train manifest")
if taskset.get("expected_count") != 96 or taskset.get("preserve_manifest_order") is not True:
    raise SystemExit("Prime taskset count/order contract is missing")
prepare = json.loads(prepare_path.read_text())
canary = json.loads(canary_path.read_text())
if prepare.get("run_id") != "staged-family-balanced-20261010" or canary.get("status") != "passed_import_hook_and_renderer":
    raise SystemExit("preparation or canary receipt is not passed")
if canary.get("training_started") is not False or canary.get("processes_started") is not False:
    raise SystemExit("canary receipt claims that work started")
if canary.get("orchestrator_setup_called") is not False:
    raise SystemExit("canary receipt claims native Orchestrator.setup was called")
renderer_probe = canary.get("renderer_probe")
if not isinstance(renderer_probe, dict) or renderer_probe.get("status") != "verified" or renderer_probe.get("valid") is not True:
    raise SystemExit("verified renderer canary is required before launch")
trainer_validation = canary.get("trainer_dependency_validation")
if not isinstance(trainer_validation, dict) or trainer_validation.get("status") != "passed" or trainer_validation.get("model_profile") != "qwen38_27b":
    raise SystemExit("reviewed Prime trainer dependency smoke is required before launch")
final_action_validation = canary.get("final_action_gate_validation")
if not isinstance(final_action_validation, dict) or final_action_validation.get("status") != "passed" or final_action_validation.get("reward_status") != "model_invalid_final_action" or final_action_validation.get("score") != 0.0:
    raise SystemExit("NativePaintingRLTask final-action zero gate is required before launch")
PY

LAUNCH_RECEIPT="$ARTIFACT_ROOT/launch.started"
"$PYTHON" - "$LAUNCH_RECEIPT" "$CONFIG" "$MATERIALIZED_ROOT" "$RUN_WRAPPER" "$TRAIN_WRAPPER" "$POLICY_ADAPTER" "$FROZEN_ADAPTER" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
receipt, config, materialized, run_wrapper, train_wrapper, policy, frozen = map(Path, sys.argv[1:])
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
receipt.write_text(json.dumps({
    "schema": "staged-family-balanced-launch-receipt-v1",
    "status": "started_foreground",
    "run_id": "staged-family-balanced-20261010",
    "config": {"path": str(config), "sha256": sha(config)},
    "materialized_root": str(materialized),
    "run_wrapper": {"path": str(run_wrapper), "sha256": sha(run_wrapper)},
    "train_wrapper": {"path": str(train_wrapper), "sha256": sha(train_wrapper)},
    "policy_initializer": {
        "path": str(policy),
        "sha256": sha(policy / "adapter_model.safetensors"),
        "revision": "1227a2427a8ae16014d2fdca4d1adb6855d3fa1c",
        "prefix": "rl-judge-matched-20261004-flash/step_12",
    },
    "frozen_sft": {
        "path": str(frozen),
        "sha256": sha(frozen / "adapter_model.safetensors"),
        "alias": "sft-step512",
        "revision": "adf687e5b321514eda7be6eab050434fdc8e44ff",
        "prefix": "photo-curriculum-sft-20260922-balanced/step_512",
    },
    "runtime_environment": {
        "WATERCOLOUR_INIT_ADAPTER": str(policy),
        "PAINTER_FROZEN_SFT_ADAPTER": str(frozen),
        "PAINTER_FROZEN_SFT_ALIAS": "sft-step512",
    },
    "command_shape": "pinned Prime Python run_from_sft.py @ native.toml",
    "background_processes_started_by_wrapper": "Prime's reviewed rl entrypoint may own child processes; this shell remains foreground",
}, indent=2, sort_keys=True) + "\n")
PY

export PRIME_ROOT MODEL_DIR BRUSH_ROOT
export WATERCOLOUR_INIT_ADAPTER="$POLICY_ADAPTER" PAINTER_FROZEN_SFT_ADAPTER="$FROZEN_ADAPTER"
export PAINTER_RL_CONFIG="$CONFIG" PAINTER_RL_MANIFEST="$MATERIALIZED_ROOT/train-manifest.json" PAINTER_EVAL_MANIFEST="$MATERIALIZED_ROOT/heldout-manifest.json"
export PAINTER_RL_RUN_NAME="$RUN_ID" PAINTER_RL_MAX_STEPS="${PAINTER_RL_MAX_STEPS:-20}" PAINTER_RL_MODE=recovered_policy
export PAINTER_RL_ADAPTER_REVISION="$POLICY_REVISION" PAINTER_RL_ADAPTER_PREFIX="$POLICY_PREFIX" PAINTER_RL_ADAPTER_SHA256="$POLICY_SHA256"
export PAINTER_FROZEN_SFT_ALIAS=sft-step512 PAINTER_FROZEN_SFT_REVISION=adf687e5b321514eda7be6eab050434fdc8e44ff PAINTER_FROZEN_SFT_PREFIX=photo-curriculum-sft-20260922-balanced/step_512 PAINTER_FROZEN_SFT_SHA256="$FROZEN_SFT_SHA256"
export PAINTER_REQUIRE_VALID_FINAL_ACTION=1
TRAIN_GPUS="$("$PYTHON" - "$CONFIG" <<'PY'
import sys
import tomllib
from pathlib import Path
config = tomllib.loads(Path(sys.argv[1]).read_text())
deployment = config.get("deployment", {})
capacity = int(deployment.get("gpus_per_node", 0))
train = int(deployment.get("num_train_gpus", 0))
infer = int(deployment.get("num_infer_gpus", 0))
if (capacity, train, infer) not in {(2, 1, 1), (4, 2, 2), (8, 4, 4)}:
    raise SystemExit("prepared config must use the exact 1+1, 2+2, or 4+4 GPU topology")
print(train)
PY
)"
export PAINTER_RL_LIVE_POLICY_ALIAS="$MODEL_DIR" PAINTER_RL_FIXED_OPPONENT=1 PAINTER_NUM_TRAIN_GPUS="$TRAIN_GPUS"
export PAINTER_RANK_GUARD_DIR="$GUARD_NAMESPACE/rank-guard"
export PAINTER_DISTRIBUTED_FAILURE_FILE="$GUARD_NAMESPACE/distributed-failure.json"
export PAINTER_LAUNCH_PHASE_FILE="${PAINTER_LAUNCH_PHASE_FILE:-$GUARD_NAMESPACE/launch-phase.json}"
export PAINTER_RESOURCE_PREFLIGHT="$NODE_ROOT/prepare-receipt.json"
export PAINTER_WHOLE_GROUP_ADMISSION=1 PAINTER_WHOLE_GROUP_SIZE=4 PAINTER_GROUP_RANK_MODULE=painter.consistent_group_admission
export PAINTER_RENDERER_PATH PAINTER_RENDERER_PYTHON_PATH PAINTER_GATEWAY_ENV_FILE="$GATEWAY_ENV_FILE"
export PAINTER_JUDGE_MODEL=xiaomi/mimo-v2.6-flash PAINTER_JUDGE_EFFORT=medium PAINTER_JUDGE_PROTOCOL=text-paint-sft-pairwise-consistent-v1 PAINTER_JUDGE_FORMAT=identity-joint-side-schema-v2-generic-v1
export PAINTER_STAGED_CURRICULUM_ENABLE=1 PAINTER_STAGED_CURRICULUM_RUN_ID="$RUN_ID" PAINTER_STAGED_CURRICULUM_PROTOCOL=staged-family-balanced-protocol-v1 PAINTER_STAGED_CURRICULUM_ADMISSION=1
export PAINTER_STAGED_CURRICULUM_ROOT="$RUN_SOURCE_ROOT" PAINTER_STAGED_CURRICULUM_STATE="$ARTIFACT_ROOT/runtime-state.json"
if [[ "$CANARY_MODE" == "real_backward" ]]; then
    export PAINTER_RUN_ARTIFACT_ROOT="$ARTIFACT_ROOT"
else
    unset PAINTER_RUN_ARTIFACT_ROOT || true
fi
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false

set +e
"$PYTHON" "$RUN_SOURCE_ROOT/launch_guard.py" \
    --failure-file "$PAINTER_DISTRIBUTED_FAILURE_FILE" \
    --rank-dir "$PAINTER_RANK_GUARD_DIR" \
    --world-size "$TRAIN_GPUS" \
    --timeout-seconds "$TIMEOUT_SECONDS" \
    --phase-file "$PAINTER_LAUNCH_PHASE_FILE" \
    -- "$PYTHON" "$RUN_WRAPPER" @ "$CONFIG"
STATUS=$?
set -e

"$PYTHON" - "$LAUNCH_RECEIPT" "$STATUS" <<'PY'
import json
import sys
from pathlib import Path
path, status = Path(sys.argv[1]), int(sys.argv[2])
value = json.loads(path.read_text())
value["status"] = "completed" if status == 0 else "failed"
value["exit_code"] = status
path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY
exit "$STATUS"
