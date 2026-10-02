#!/usr/bin/env bash
set -Eeuo pipefail

# Prepare all three arms from one shared, hash-verified source materialization.
# This command never starts a trainer or a judge.
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd "$RUN/../../.." && pwd -P)"
BASE="${PAINTER_BASE_RUN:-$REPO/painter/runs/text-sft512-consistent-20260930}"
RETRY="${PAINTER_RETRY_RUN:-$REPO/painter/runs/text-rl16-retry-20261001}"
SHARED="$RUN/../rl-judge-ablation-shared-20261002"
FROZEN_SOURCE="${PAINTER_FROZEN_SFT_ADAPTER_SOURCE_FILE:-${PAINTER_FROZEN_SFT_ADAPTER_PATH_FILE:-$REPO/painter/runs/photo-curriculum-rl-20260922/initial-adapter-path.txt}}"
BOOTSTRAP="${PAINTER_BOOTSTRAP_RUN:-$REPO/painter/runs/text-pairwise-rl-20260928}"
PY="${PAINTER_PREPARE_PYTHON:-${PRIME_ROOT:-/root/painter/prime}/.venv/bin/python}"
[[ -x "$PY" ]] || PY=python3

usage() {
  echo 'usage: prepare.sh [all|--plan-only]' >&2
  exit 2
}
MODE="${1:-all}"
[[ "$MODE" == all || "$MODE" == --plan-only ]] || usage

PYTHONPATH="$REPO:$REPO/painter:$RUN${PYTHONPATH:+:$PYTHONPATH}" \
  "$PY" "$RUN/prepare_matrix.py" --plan >/dev/null
if [[ "$MODE" == --plan-only ]]; then
  echo 'RL_JUDGE_ABLATION_PLAN_VALIDATED'
  exit 0
fi

[[ "$(uname -s)" == Linux ]] || { echo 'Linux GPU node required for node preparation' >&2; exit 1; }
[[ -f "$BASE/prepare_node.sh" ]] || { echo "base preparation missing: $BASE/prepare_node.sh" >&2; exit 1; }
[[ -f "$RETRY/prepare_node.sh" ]] || { echo "retry preparation missing: $RETRY/prepare_node.sh" >&2; exit 1; }
if [[ -e "$SHARED/preparation-receipt.json" ]]; then
  :
elif [[ -e "$SHARED" ]]; then
  echo 'shared preparation directory exists without a receipt; preserve and inspect it' >&2
  exit 1
else
  if [[ ! -r "$FROZEN_SOURCE" ]]; then
    [[ -f "$BOOTSTRAP/setup_node.sh" ]] || {
      echo "reviewed bootstrap setup is missing: $BOOTSTRAP/setup_node.sh" >&2
      exit 1
    }
    echo "frozen SFT path is absent; running reviewed bootstrap setup: $BOOTSTRAP/setup_node.sh"
    PAINTER_BOOTSTRAP_RUN="$BOOTSTRAP" bash "$BOOTSTRAP/setup_node.sh"
  fi
  [[ -r "$FROZEN_SOURCE" ]] || {
    echo "verified frozen SFT path file is missing: $FROZEN_SOURCE" >&2
    exit 1
  }
  mkdir -p "$SHARED"
  cp "$FROZEN_SOURCE" "$SHARED/frozen-sft-adapter-path.txt"
  PAINTER_RUN_ROOT="$SHARED" \
  PAINTER_RUN_NAME=rl-judge-ablation-shared-20261002 \
  PAINTER_POLICY_MODE=recovered_policy \
  PAINTER_POLICY_REVISION=ac4d3eda8e5925e2d23992cfdc1a63620ebde573 \
  PAINTER_POLICY_PREFIX=text-pairwise-scaled-20260929/step_16 \
  PAINTER_POLICY_SHA256=e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c \
  PAINTER_JUDGE_PROTOCOL=text-paint-sft-pairwise-consistent-v1 \
  PAINTER_JUDGE_FORMAT=identity-joint-side-schema-v2-generic-v1 \
  PAINTER_JUDGE_MODEL=xiaomi/mimo-v2.6-flash \
  PAINTER_POLICY_DOWNLOAD_SCRIPT="$REPO/painter/runs/text-pairwise-rl-20260928/download_public_adapter.py" \
  PAINTER_POLICY_DOWNLOAD_RECEIPT="$SHARED/policy-adapter-receipt.json" \
  PAINTER_GROUP_SIZE=4 \
  PAINTER_INITIAL_INFLIGHT=96 \
  PAINTER_MAX_INFLIGHT=96 \
  PAINTER_SFT_ADAPTER_PATH_FILE="$SHARED/policy-adapter-path.txt" \
  PAINTER_FROZEN_SFT_ADAPTER_PATH_FILE="$SHARED/frozen-sft-adapter-path.txt" \
  bash "$RETRY/prepare_node.sh"
fi
for path_file in "$SHARED/policy-adapter-path.txt" "$SHARED/frozen-sft-adapter-path.txt"; do
  [[ -r "$path_file" ]] || {
    echo "shared preparation did not preserve verified adapter path: $path_file" >&2
    exit 1
  }
done
PYTHONPATH="$REPO:$REPO/painter:$RUN${PYTHONPATH:+:$PYTHONPATH}" \
  "$PY" "$RUN/prepare_matrix.py" --materialize "$SHARED"

MODEL_DIR="${MODEL_DIR:-/root/painter/model-cache/qwen/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
PRIME_ROOT="${PRIME_ROOT:-/root/painter/prime}"
CONFIG_TOOL="$REPO/painter/runs/text-pairwise-rl-20260928/prepare_config.py"
GPU_COUNT=8
TRAIN_GPU_COUNT=1
INFER_GPU_COUNT=7
INITIAL_INFLIGHT=96
MAX_INFLIGHT=96
for ARM in flash pro luna; do
  ARM_MODEL="$("$PY" - "$RUN/matrix.json" "$ARM" <<'PY'
import json, sys
value = json.loads(open(sys.argv[1]).read())
print(next(arm['model'] for arm in value['arms'] if arm['id'] == sys.argv[2]))
PY
)"
  ARM_NAME="$("$PY" - "$RUN/matrix.json" "$ARM" <<'PY'
import json, sys
value = json.loads(open(sys.argv[1]).read())
print(next(arm['run_name'] for arm in value['arms'] if arm['id'] == sys.argv[2]))
PY
)"
  ARM_ROOT="$RUN/../$ARM_NAME"
  OUTPUT_ROOT="$("$PY" - "$RUN/matrix.json" "$ARM" <<'PY'
import json, sys
value = json.loads(open(sys.argv[1]).read())
print(next(arm['output_root'] for arm in value['arms'] if arm['id'] == sys.argv[2]))
PY
)"
  if [[ -e "$ARM_ROOT/native.toml" || -e "$ARM_ROOT/rl-config-receipt.json" ]]; then
    [[ -f "$ARM_ROOT/native.toml" && -f "$ARM_ROOT/rl-config-receipt.json" ]] || {
      echo "arm $ARM has a partial config; preserve it and inspect before retrying" >&2
      exit 1
    }
    "$PY" - "$ARM_ROOT/rl-config-receipt.json" "$ARM_NAME" "$MODEL_DIR" <<'PY'
import json, sys
from pathlib import Path
receipt = json.loads(Path(sys.argv[1]).read_text())
expected_name, expected_model = sys.argv[2:]
if receipt.get("run_name") != expected_name or receipt.get("model_revision") != "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0":
    raise SystemExit("existing arm receipt does not match the registered run identity")
if receipt.get("adapter_revision") != "ac4d3eda8e5925e2d23992cfdc1a63620ebde573" or receipt.get("adapter_prefix") != "text-pairwise-scaled-20260929/step_16":
    raise SystemExit("existing arm receipt does not match the registered initializer")
deployment = receipt.get("deployment", {})
if deployment != {"gpus_per_node": 8, "num_train_gpus": 1, "num_infer_gpus": 7}:
    raise SystemExit("existing arm receipt has a different frozen topology")
PY
  else
    "$PY" "$CONFIG_TOOL" \
      --output "$ARM_ROOT/native.toml" \
      --model "$MODEL_DIR" --run-root "$ARM_ROOT" \
      --manifest "$ARM_ROOT/anchors-frozen/train-manifest.json" \
      --prime-root "$PRIME_ROOT" --max-steps 50 --checkpoint-interval 4 \
      --run-name "$ARM_NAME" --output-root "$OUTPUT_ROOT" \
      --judge-protocol text-paint-sft-pairwise-consistent-v1 \
      --expected-count 40 --adapter-revision ac4d3eda8e5925e2d23992cfdc1a63620ebde573 \
      --adapter-prefix text-pairwise-scaled-20260929/step_16 \
      --adapter-sha256 e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c \
      --mode recovered_policy --learning-rate 1e-6 --gpus-per-node "$GPU_COUNT" \
      --num-train-gpus "$TRAIN_GPU_COUNT" --num-infer-gpus "$INFER_GPU_COUNT" --batch-size 16 --group-size 4 \
      --initial-inflight "$INITIAL_INFLIGHT" --max-inflight "$MAX_INFLIGHT" \
      --renderer "$ARM_ROOT/vendor/integrations/watercolour/renderer.py" \
      --renderer-python "$ARM_ROOT/renderer-env/bin/python" >/dev/null
  fi
  if [[ -f "$SHARED/policy-adapter-path.txt" ]]; then
    cp "$SHARED/policy-adapter-path.txt" "$ARM_ROOT/policy-adapter-path.txt"
  fi
  if [[ -f "$SHARED/policy-adapter-receipt.json" ]]; then
    cp "$SHARED/policy-adapter-receipt.json" "$ARM_ROOT/policy-adapter-receipt.json"
  fi
  if [[ -f "$SHARED/frozen-sft-adapter-path.txt" ]]; then
    cp "$SHARED/frozen-sft-adapter-path.txt" "$ARM_ROOT/frozen-sft-adapter-path.txt"
  elif [[ -n "${PAINTER_FROZEN_SFT_ADAPTER_PATH_FILE:-}" ]]; then
    cp "$PAINTER_FROZEN_SFT_ADAPTER_PATH_FILE" "$ARM_ROOT/frozen-sft-adapter-path.txt"
  fi
done

echo 'RL_JUDGE_ABLATION_PREPARED: shared frozen source and three isolated configs ready'
