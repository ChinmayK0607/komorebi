#!/usr/bin/env bash
set -Eeuo pipefail

# Run the prepared Flash -> Pro -> Luna matrix as one foreground node command.
# Provider allocation and exact-pod release remain parent-owned; this wrapper
# only consumes the parent's start/price inputs and never calls a provider CLI.
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
STATE_ROOT="${PAINTER_CAMPAIGN_STATE_ROOT:-$RUN}"
PREPARE="${PAINTER_ABLATION_PREPARE_SCRIPT:-$RUN/prepare.sh}"
EXECUTE="${PAINTER_ABLATION_EXECUTE_SCRIPT:-$RUN/execute_pipeline.sh}"
START="${1:?usage: campaign.sh RENTAL_START_EPOCH PRICE_USD_PER_HOUR MAX_CAMPAIGN_GPU_USD}"
PRICE="${2:?usage: campaign.sh RENTAL_START_EPOCH PRICE_USD_PER_HOUR MAX_CAMPAIGN_GPU_USD}"
CAMPAIGN_CAP="${3:-59.4}"
[[ $# == 2 || $# == 3 ]] || {
  echo 'usage: campaign.sh RENTAL_START_EPOCH PRICE_USD_PER_HOUR MAX_CAMPAIGN_GPU_USD' >&2
  exit 2
}
[[ "$(uname -s)" == Linux || "${PAINTER_CAMPAIGN_ALLOW_NON_LINUX:-0}" == 1 ]] || {
  echo 'Linux GPU node required for the finite campaign' >&2
  exit 1
}

PY="${PAINTER_CAMPAIGN_PYTHON:-python3}"
[[ -x "$PY" || "$PY" == python3 ]] || { echo "campaign Python is unavailable: $PY" >&2; exit 1; }
ARM_CAP="$($PY - "$PRICE" "$CAMPAIGN_CAP" <<'PY'
import math, sys
price, campaign = map(float, sys.argv[1:])
if any(not math.isfinite(x) or x <= 0 for x in (price, campaign)):
    raise SystemExit("price and campaign cap must be positive finite numbers")
if campaign > 60.0:
    raise SystemExit("campaign GPU cap must be at most $60")
arm = campaign / 3.0
if arm > 20.0:
    raise SystemExit("derived per-arm GPU cap exceeds the $20 limit")
print(f"{arm:.9f}")
PY
)"

mkdir -p "$STATE_ROOT"
if [[ "${PAINTER_CAMPAIGN_SKIP_LOCK:-0}" == 1 ]]; then
  : # Contract tests use an isolated temporary state root.
else
  command -v flock >/dev/null 2>&1 || { echo 'Linux flock is required for the campaign lock' >&2; exit 1; }
  exec 9>"$STATE_ROOT/campaign.lock"
  flock -n 9 || { echo 'judge ablation campaign already running' >&2; exit 2; }
fi
CAMPAIGN_DIR="$STATE_ROOT/campaign"
mkdir -p "$CAMPAIGN_DIR"
RELEASE_MARKER="$STATE_ROOT/campaign-release.ready"
[[ ! -e "$RELEASE_MARKER" ]] || {
  echo 'campaign release marker already exists; preserve it before retrying' >&2
  exit 1
}
STATUS_FILE="$CAMPAIGN_DIR/status.json"
ARM_STATUS_FILE="$CAMPAIGN_DIR/arms.jsonl"
: >"$ARM_STATUS_FILE"
COMMON_PREP_SHARE=0

write_status() {
  local status="$1" reason="$2"
  "$PY" - "$STATUS_FILE" "$status" "$reason" "$START" "$PRICE" "$CAMPAIGN_CAP" "$ARM_CAP" "$COMMON_PREP_SHARE" <<'PY'
import json, sys, time
from pathlib import Path
path, status, reason, start, price, campaign, arm, prep_share = sys.argv[1:]
path = Path(path)
path.write_text(json.dumps({
    "schema": "text-paint-rl-judge-ablation-campaign-v1",
    "status": status, "reason": reason,
    "started_epoch": float(start), "price_usd_per_hour": float(price),
    "campaign_gpu_cap_usd": float(campaign), "per_arm_gpu_cap_usd": float(arm),
    "common_preparation_share_seconds": float(prep_share),
    "updated_epoch": time.time(),
}, indent=2, sort_keys=True) + "\n")
PY
}

write_status running shared_preparation
prep_start="$(date +%s.%N)"
if ! "$PREPARE" all >"$CAMPAIGN_DIR/prepare.log" 2>&1; then
  write_status failed shared_preparation_failed
  exit 1
fi
prep_end="$(date +%s.%N)"
COMMON_PREP_SHARE="$($PY - "$START" "$prep_end" <<'PY'
import sys
elapsed = max(0.0, float(sys.argv[2]) - float(sys.argv[1]))
print(f"{elapsed / 3.0:.9f}")
PY
)"

group_is_gone() {
  local state="$1"
  [[ -f "$state" ]] || return 0
  "$PY" - "$state" <<'PY'
import json, os, sys
from pathlib import Path
value = json.loads(Path(sys.argv[1]).read_text())
for key in ("pid", "pgid"):
    number = value.get(key)
    if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
        raise SystemExit(0)
    try:
        os.kill(number if key == "pid" else -number, 0)
    except ProcessLookupError:
        continue
    except PermissionError:
        raise SystemExit(1)
    else:
        raise SystemExit(1)
PY
}

for ARM in flash pro luna; do
  ARM_NAME="$($PY - "$RUN/matrix.json" "$ARM" <<'PY'
import json, sys
value = json.loads(open(sys.argv[1]).read())
print(next(row["run_name"] for row in value["arms"] if row["id"] == sys.argv[2]))
PY
)"
  ARM_ROOT="$STATE_ROOT/../$ARM_NAME"
  ARM_START="$(date +%s.%N)"
  EFFECTIVE_START="$($PY - "$ARM_START" "$COMMON_PREP_SHARE" <<'PY'
import sys
print(f"{float(sys.argv[1]) - float(sys.argv[2]):.9f}")
PY
)"
  write_status running "arm_${ARM}_running"
  set +e
  "$EXECUTE" "$ARM" "$EFFECTIVE_START" "$PRICE" "$ARM_CAP" >"$CAMPAIGN_DIR/${ARM}.log" 2>&1
  ARM_RC=$?
  set -e

  TERMINAL="$ARM_ROOT/terminal-receipt.json"
  LOCAL_ARCHIVE="$ARM_ROOT/local-evidence/${ARM_NAME}-evidence.tar.gz"
  UPLOAD_RECEIPT="$LOCAL_ARCHIVE.upload.json"
  PRESERVED="$($PY - "$TERMINAL" "$UPLOAD_RECEIPT" <<'PY'
import json, sys
from pathlib import Path
terminal, upload = map(Path, sys.argv[1:])
try:
    value = json.loads(terminal.read_text())
    receipt = json.loads(upload.read_text())
except (OSError, json.JSONDecodeError):
    print("0")
    raise SystemExit
valid_status = value.get("status") in {"complete", "failed"}
public = value.get("public_archive_hash_verified") is True
verified = receipt.get("public_hash_verified") is True and receipt.get("private") is False
print("1" if valid_status and public and verified else "0")
PY
)"
  "$PY" - "$ARM_STATUS_FILE" "$ARM" "$ARM_RC" "$PRESERVED" "$TERMINAL" "$LOCAL_ARCHIVE" "$UPLOAD_RECEIPT" "$ARM_START" "$EFFECTIVE_START" "$ARM_CAP" "$COMMON_PREP_SHARE" <<'PY'
import json, sys, time
from pathlib import Path
path, arm, code, preserved, terminal, archive, upload, arm_start, effective_start, cap, prep_share = sys.argv[1:]
row = {
    "schema": "text-paint-rl-judge-ablation-arm-result-v1",
    "arm": arm, "exit_code": int(code), "preserved": preserved == "1",
    "terminal_receipt": terminal, "local_evidence_archive": archive,
    "upload_receipt": upload,
    "arm_start_epoch": float(arm_start), "effective_start_epoch": float(effective_start),
    "per_arm_gpu_cap_usd": float(cap), "common_preparation_share_seconds": float(prep_share),
    "recorded_epoch": time.time(),
}
with Path(path).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(row, sort_keys=True) + "\n")
PY
  if ! group_is_gone "$ARM_ROOT/supervisor-state.json"; then
    write_status failed "owned_process_group_remains_after_${ARM}"
    exit 1
  fi
  if [[ "$PRESERVED" != 1 ]]; then
    write_status failed "public_archive_not_verified_after_${ARM}"
    exit 1
  fi
done

ALL_PRESERVED="$($PY - "$ARM_STATUS_FILE" <<'PY'
import json, sys
from pathlib import Path
rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
print("1" if len(rows) == 3 and all(row.get("preserved") for row in rows) else "0")
PY
)"
if [[ "$ALL_PRESERVED" != 1 ]]; then
  write_status failed arm_evidence_not_preserved
  exit 1
fi

write_status complete all_arms_preserved
"$PY" - "$RELEASE_MARKER" "$STATUS_FILE" "$ARM_STATUS_FILE" <<'PY'
import hashlib, json, sys, time
from pathlib import Path
marker, status, arms = map(Path, sys.argv[1:])
digest = hashlib.sha256(arms.read_bytes()).hexdigest()
marker.write_text(json.dumps({
    "schema": "text-paint-rl-judge-ablation-campaign-release-v1",
    "status": "all_arms_preserved", "arm_status_sha256": digest,
    "status_path": str(status), "created_epoch": time.time(),
}, indent=2, sort_keys=True) + "\n")
PY
echo "RL_JUDGE_ABLATION_CAMPAIGN_COMPLETE: release marker created after all three arm receipts"
