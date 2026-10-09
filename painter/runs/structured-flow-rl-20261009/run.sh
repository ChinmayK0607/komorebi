#!/usr/bin/env bash
set -Eeuo pipefail

RUN="${PAINTER_RUN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)}"
REPO="$(cd "$RUN/../../.." && pwd -P)"
# Registered amendment: normalize stale inherited launch values before the
# run-owned runtime validator executes.
export PAINTER_RL_MAX_STEPS=20 PAINTER_MAX_UPDATES=20 PAINTER_CHECKPOINT_INTERVAL=4 PAINTER_EVAL_PREFERRED_STEP=20
source "$RUN/runtime_env.sh"
validate_structured_bindings

[[ "$(uname -s)" == Linux ]] || { echo 'Linux GPU node required' >&2; exit 1; }
CONFIG="${PAINTER_RL_CONFIG:-$RUN/native.toml}"
MANIFEST="${PAINTER_RL_MANIFEST:-$RUN/curriculum/pilot-train-manifest.json}"
for required in "$CONFIG" "$RUN/rl-config-receipt.json" "$MANIFEST" "$RUN/curriculum/matched16-manifest.json" \
  "$RUN/launch_rl.sh" "$RUN/run_from_sft.py" "$RUN/train_from_sft.py" "$RUN/renderer-env/bin/python" \
  "$RUN/sitecustomize.py" "$RUN/runtime_receipt.py" "$REPO/painter/consistent_group_admission.py"; do
  [[ -e "$required" ]] || { echo "missing prepared input: $required" >&2; exit 1; }
done
[[ -r "$PAINTER_SFT_ADAPTER_PATH_FILE" && -r "$PAINTER_FROZEN_SFT_ADAPTER_PATH_FILE" ]] || {
  echo 'public policy and frozen SFT adapter path files are missing' >&2; exit 1;
}
POLICY_ADAPTER="$(tr -d '\r\n' < "$PAINTER_SFT_ADAPTER_PATH_FILE")"
FROZEN_ADAPTER="$(tr -d '\r\n' < "$PAINTER_FROZEN_SFT_ADAPTER_PATH_FILE")"
ACTIVE_POLICY_POINTER="${PAINTER_ACTIVE_POLICY_PATH_FILE:-$RUN/active-policy-adapter-path.txt}"
export PAINTER_ACTIVE_POLICY_PATH_FILE="$ACTIVE_POLICY_POINTER"
if [[ -r "$ACTIVE_POLICY_POINTER" ]]; then
  POLICY_ADAPTER="$(tr -d '\r\n' < "$ACTIVE_POLICY_POINTER")"
fi
[[ -f "$POLICY_ADAPTER/adapter_model.safetensors" && -f "$FROZEN_ADAPTER/adapter_model.safetensors" ]] || {
  echo 'verified policy or frozen SFT adapter is incomplete' >&2; exit 1;
}
[[ "$(cd "$POLICY_ADAPTER" && pwd -P)" != "$(cd "$FROZEN_ADAPTER" && pwd -P)" ]] || {
  echo 'policy initializer and frozen SFT opponent must be distinct' >&2; exit 1;
}

export PAINTER_REPO_ROOT="$REPO" BRUSH_ROOT="$REPO/painter/rl_helpers"
export WATERCOLOUR_INIT_ADAPTER="$POLICY_ADAPTER" PAINTER_FROZEN_SFT_ADAPTER="$FROZEN_ADAPTER"
ACTIVE_POLICY_SHA256="${PAINTER_ACTIVE_POLICY_ADAPTER_SHA256:-$PAINTER_POLICY_SHA256}"
[[ "$ACTIVE_POLICY_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
  echo 'active policy adapter SHA-256 is malformed' >&2; exit 1;
}
if [[ -r "$ACTIVE_POLICY_POINTER" ]]; then
  ACTIVE_POLICY_SHA256="$(python3 - "$POLICY_ADAPTER/adapter_model.safetensors" <<'PY'
import hashlib, sys
print(hashlib.sha256(open(sys.argv[1], 'rb').read()).hexdigest())
PY
)"
else
  ACTUAL_POLICY_SHA256="$(python3 - "$POLICY_ADAPTER/adapter_model.safetensors" <<'PY'
import hashlib, sys
print(hashlib.sha256(open(sys.argv[1], 'rb').read()).hexdigest())
PY
)"
  [[ "$ACTUAL_POLICY_SHA256" == "$ACTIVE_POLICY_SHA256" ]] || {
    echo 'policy adapter bytes do not match the bound active SHA-256' >&2; exit 1;
  }
fi
export PAINTER_ACTIVE_POLICY_ADAPTER_SHA256="$ACTIVE_POLICY_SHA256"
# The prepared receipt records the public Flash12 initializer, while Prime's
# generic launcher validates the bytes it actually loads.  Normalize the
# receipt once after warmup so both identities remain auditable and the
# launcher checks the verified active adapter rather than the parent bytes.
RECEIPT_ARGS=(
  --receipt "$RUN/rl-config-receipt.json"
  --active-sha256 "$ACTIVE_POLICY_SHA256"
  --parent-sha256 "$PAINTER_POLICY_SHA256"
  --parent-revision "$PAINTER_POLICY_REVISION"
  --parent-prefix "$PAINTER_POLICY_PREFIX"
  --mode "$PAINTER_POLICY_MODE"
)
if [[ -r "$ACTIVE_POLICY_POINTER" ]]; then
  [[ -r "$RUN/active-policy.json" ]] || { echo 'active warmup policy receipt is missing' >&2; exit 1; }
  RECEIPT_ARGS+=(--active-policy-json "$RUN/active-policy.json")
fi
python3 "$RUN/runtime_receipt.py" "${RECEIPT_ARGS[@]}"
export PAINTER_RL_MODE="$PAINTER_POLICY_MODE" PAINTER_RL_FIXED_OPPONENT=1
export PAINTER_RL_CONFIG="$CONFIG" PAINTER_RL_MANIFEST="$MANIFEST"
export PAINTER_RL_RUN_NAME="$PAINTER_RUN_NAME" PAINTER_RL_MAX_STEPS=20
export PAINTER_RL_ADAPTER_REVISION="$PAINTER_POLICY_REVISION"
export PAINTER_RL_ADAPTER_PREFIX="$PAINTER_POLICY_PREFIX"
# The config receipt retains the immutable Flash12 identity under
# policy_initializer; this top-level value binds the actual active bytes to
# Prime's generic launcher's initializer check.
export PAINTER_RL_ADAPTER_SHA256="$ACTIVE_POLICY_SHA256"
export PAINTER_RL_MAX_POLICY_AGE=1
export PAINTER_FROZEN_SFT_ALIAS=sft-step512
export PAINTER_WHOLE_GROUP_ADMISSION=1 PAINTER_WHOLE_GROUP_SIZE="$PAINTER_GROUP_SIZE"
export PAINTER_GROUP_RANK_MODULE=painter.consistent_group_admission
export PAINTER_CONSISTENT_GUARD_RECEIPT="${PAINTER_CONSISTENT_GUARD_RECEIPT:-$RUN/consistent-group-admission-activation.json}"
export PAINTER_CONSISTENT_GROUP_TELEMETRY="${PAINTER_CONSISTENT_GROUP_TELEMETRY:-$RUN/consistent-group-admission.jsonl}"
# Make the RL role explicit after the group contract is present.  This keeps
# Prime's orchestrator children fail-closed even when the controller carried
# a baseline/evaluation/warmup role into the training handoff.
export PAINTER_PRIME_ROLE=orchestrator
export PAINTER_RENDERER_PATH PAINTER_RENDERER_PYTHON_PATH
export PAINTER_GATEWAY_ENV_FILE HF_TOKEN_PATH
export PYTHONPATH="$REPO:$REPO/painter:$RUN${PYTHONPATH:+:$PYTHONPATH}"
export PAINTER_RL_JUDGE_MODEL="$PAINTER_JUDGE_MODEL" PAINTER_JUDGE_EFFORT PAINTER_JUDGE_FORMAT

[[ -f "$PAINTER_GATEWAY_ENV_FILE" && ! -L "$PAINTER_GATEWAY_ENV_FILE" ]] || {
  echo 'private Gateway key file missing' >&2; exit 1;
}
[[ "$(stat -c %a "$PAINTER_GATEWAY_ENV_FILE")" == 600 ]] || {
  echo 'Gateway key file must be mode 600' >&2; exit 1;
}

if [[ "${1:-}" == --preflight ]]; then
  [[ $# == 1 ]] || { echo 'usage: run.sh [--preflight]' >&2; exit 2; }
  PAINTER_RL_PREFLIGHT_ONLY=1 PAINTER_RL_SKIP_EDITABLE_INSTALL=1 bash "$RUN/launch_rl.sh" "$RUN"
  exit 0
fi
[[ $# == 0 ]] || { echo 'usage: run.sh [--preflight]' >&2; exit 2; }
[[ "${PAINTER_RL_SUPERVISED:-0}" == 1 ]] || {
  echo 'finite supervisor required: set PAINTER_RL_SUPERVISED=1 only for its owned process group' >&2
  exit 2
}
[[ ! -e "$RUN/async-output/$PAINTER_RUN_NAME" ]] || {
  echo 'existing training output; preserve it before another attempt' >&2; exit 1;
}
exec bash "$RUN/launch_rl.sh" "$RUN"
