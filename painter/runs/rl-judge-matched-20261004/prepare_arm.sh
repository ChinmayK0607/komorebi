#!/usr/bin/env bash
set -Eeuo pipefail

ARM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd "$ARM_ROOT/../rl-judge-matched-20261004" && pwd -P)"
ARM_ID="$(basename "$ARM_ROOT")"
ARM_ID="${ARM_ID#rl-judge-matched-20261004-}"
source "$ROOT/arm_env.sh" "$ARM_ID"
[[ $# == 0 ]] || { echo 'usage: prepare_arm.sh' >&2; exit 2; }
for required in "$PAINTER_RUN_ROOT/native.toml" "$PAINTER_RUN_ROOT/rl-config-receipt.json" \
  "$PAINTER_RUN_ROOT/preparation-receipt.json" "$PAINTER_RUN_ROOT/anchors-frozen/train-manifest.json" \
  "$PAINTER_RUN_ROOT/anchors-frozen/validation-matched16.json" "$PAINTER_RUN_ROOT/policy-adapter-path.txt" \
  "$PAINTER_RUN_ROOT/frozen-sft-adapter-path.txt" "$PAINTER_RUN_ROOT/source-identity.json"; do
  [[ -e "$required" ]] || { echo "prepared arm input is missing: $required" >&2; exit 1; }
done
"${PAINTER_PREPARE_PYTHON:-python3}" - "$PAINTER_RUN_ROOT/rl-config-receipt.json" "$PAINTER_RUN_ROOT/native.toml" <<'PY'
import json, sys
from pathlib import Path
receipt = json.loads(Path(sys.argv[1]).read_text())
if receipt.get("mode") != "recovered_policy" or receipt.get("optimizer_resume") is not False:
    raise SystemExit("arm must start from the RL16 adapter with a fresh optimizer")
if receipt.get("batch_size") != 16 or receipt.get("group_size") != 4:
    raise SystemExit("arm config has the wrong batch/group shape")
if receipt.get("initial_inflight") != 32 or receipt.get("max_inflight") != 32:
    raise SystemExit("arm config has the wrong in-flight bounds")
if receipt.get("max_off_policy_steps") != 1:
    raise SystemExit("arm config does not bind max policy age one")
if receipt.get("activation_cpu_offload") is not True:
    raise SystemExit("arm config does not enable CPU activation offload")
PY
echo "MATCHED_ARM_PREPARED: $ARM_ID"
