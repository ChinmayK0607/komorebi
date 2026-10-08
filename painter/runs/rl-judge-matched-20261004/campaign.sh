#!/usr/bin/env bash
set -Eeuo pipefail

# Sequential finite campaign entrypoint.  The provider allocator and outer
# detached controller remain parent-owned; this script only runs the prepared
# node command and writes weight-free terminal receipts.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PY="${PAINTER_CAMPAIGN_PYTHON:-python3}"
STATE_ROOT="${PAINTER_CAMPAIGN_STATE_ROOT:-$ROOT}"
PREPARE="${PAINTER_MATCHED_PREPARE_SCRIPT:-$ROOT/prepare.sh}"
BASELINE="${PAINTER_MATCHED_BASELINE_SCRIPT:-$ROOT/baseline.sh}"
EXECUTE="${PAINTER_MATCHED_EXECUTE_SCRIPT:-$ROOT/execute_arm.sh}"
ARM_ENV="${PAINTER_MATCHED_ARM_ENV_SCRIPT:-$ROOT/arm_env.sh}"
RESTORE="${PAINTER_MATCHED_RESTORE_SCRIPT:-$ROOT/restore_continuation.py}"

CONTINUATION=0
if [[ "${1:-}" == --continuation ]]; then
  CONTINUATION=1
  shift
fi
if [[ "${1:-}" == --plan-only ]]; then
  "$PY" "$ROOT/prepare_campaign.py" --plan
  echo 'MATCHED_THREE_JUDGE_CAMPAIGN_PLAN_VALIDATED'
  exit 0
fi
[[ $# == 3 ]] || { echo 'usage: campaign.sh [--continuation] RENTAL_START_EPOCH PRICE_USD_PER_HOUR MAX_CAMPAIGN_GPU_USD' >&2; exit 2; }
START="$1"; PRICE="$2"; CAP="$3"
PARAMS="$($PY - "$START" "$PRICE" "$CAP" "$CONTINUATION" <<'PY'
import math, sys
start, price, cap = map(float, sys.argv[1:4])
continuation = sys.argv[4] == '1'
if any(not math.isfinite(x) or x <= 0 for x in (start, price, cap)):
    raise SystemExit('start, price, and campaign cap must be positive finite values')
limit = 43.5 if continuation else 58.5
if cap > limit:
    raise SystemExit(f'campaign cap exceeds the registered ${limit:.2f} total limit')
if continuation:
    if cap < 43.5:
        raise SystemExit('continuation requires the registered $43.50 campaign cap')
    # Flash recovery is bounded separately; Pro and Luna retain their
    # registered $19.50 training caps.
    per = 19.5
else:
    per = cap / 3.0
    if per > 19.5:
        raise SystemExit('equal per-arm allocation exceeds the registered $19.50 limit')
if cap / price * 3600 <= 2520:
    raise SystemExit('campaign cap leaves no finite completion reserve')
print(f'{per:.9f}')
PY
)"

[[ "$(uname -s)" == Linux || "${PAINTER_CAMPAIGN_ALLOW_NON_LINUX:-0}" == 1 ]] || {
  echo 'Linux GPU node required for the finite campaign' >&2; exit 1;
}
mkdir -p "$STATE_ROOT/campaign"
if command -v flock >/dev/null 2>&1; then
  exec 9>"$STATE_ROOT/campaign.lock"
  flock -n 9 || { echo 'matched campaign already running' >&2; exit 2; }
else
  LOCK_DIR="$STATE_ROOT/campaign.lock.d"
  mkdir "$LOCK_DIR" 2>/dev/null || { echo 'matched campaign already running' >&2; exit 2; }
  trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT
fi
STATUS="$STATE_ROOT/campaign/status.json"
ARMS="$STATE_ROOT/campaign/arms.jsonl"
TERMINAL="$STATE_ROOT/campaign/terminal-receipt.json"
: >"$ARMS"

write_status() {
  local state="$1" reason="$2"
  "$PY" - "$STATUS" "$state" "$reason" "$START" "$PRICE" "$CAP" "$PARAMS" "$CONTINUATION" <<'PY'
import hashlib, json, sys, time
from pathlib import Path
path, status, reason, start, price, cap, per, continuation = sys.argv[1:]
path = Path(path)
temporary = path.with_name(path.name + '.tmp')
payload = {
    'schema': 'matched-three-judge-campaign-status-v1', 'status': status,
    'reason': reason, 'started_epoch': float(start), 'price_usd_per_hour': float(price),
    'campaign_cap_usd': float(cap), 'per_arm_cap_usd': float(per),
    'updated_epoch': time.time(),
}
if continuation == '1':
    payload.update({'continuation': True, 'flash_recovery_reserved_usd': float(cap) - 2.0 * float(per)})
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
temporary.replace(path)
PY
}

campaign_on_exit() {
  local original_rc=$?
  trap - EXIT
  set +e
  if [[ ! -f "$TERMINAL" ]]; then
    "$PY" - "$TERMINAL" "$ARMS" "$original_rc" <<'PY'
import hashlib, json, sys, time
from pathlib import Path
terminal, arms, rc = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
terminal.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "schema": "matched-three-judge-campaign-terminal-v1",
    "status": "complete" if rc == 0 else "failed",
    "exit_code": rc,
    "arms": [json.loads(line) for line in arms.read_text().splitlines() if line.strip()],
    "arm_status_sha256": hashlib.sha256(arms.read_bytes()).hexdigest(),
    "created_epoch": time.time(),
}
temporary = terminal.with_name(terminal.name + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(terminal)
PY
  fi
  if [[ "$original_rc" != 0 ]]; then
    # Publish the failed state before archiving so a public evidence bundle
    # cannot retain the transient running status after a failed arm or
    # preflight.  The terminal receipt above still preserves the original
    # exit code and arm evidence.
    write_status failed campaign_failed
  fi
  local archive_rc=0
  if [[ "${PAINTER_CAMPAIGN_SKIP_ARCHIVE:-0}" != 1 ]]; then
    local archive_root="${PAINTER_MATCHED_ARCHIVE_ROOT:-$ROOT}"
    local repo
    repo="$(cd "$ROOT/../../.." && pwd -P)"
    local evidence="$archive_root/local-evidence"
    local archive="$evidence/campaign-evidence.tar.gz"
    local token="${HF_TOKEN_PATH:-${PAINTER_NODE_ROOT:-/root/painter}/private/hf-token}"
    local timeout_seconds="${PAINTER_ARCHIVE_TIMEOUT_SECONDS:-360}"
    local archive_python="${PRIME_ROOT:-${PAINTER_NODE_ROOT:-/root/painter}/prime}/.venv/bin/python"
    [[ -x "$archive_python" ]] || archive_python="$PY"
    mkdir -p "$evidence"
    if [[ "$timeout_seconds" =~ ^[1-9][0-9]*$ && -r "$token" && ! -L "$token" && "$(stat -c %a "$token")" == 600 ]]; then
      timeout --signal=TERM --kill-after=30 "$timeout_seconds" \
        "$archive_python" "$repo/painter/runs/text-sft512-consistent-20260930/safe_archive.py" \
        --run-root "$archive_root" --output "$archive" --token-file "$token" \
        --repo-id "${PAINTER_EVIDENCE_REPO_ID:-CK0607/komorebi-painter-experiments}" \
        --remote-path "${PAINTER_CAMPAIGN_EVIDENCE_REMOTE_PATH:-runs/rl-judge-matched-20261004/campaign-evidence.tar.gz}" \
        --verify-dir "$evidence/public-verify" >"$evidence/archive.log" 2>&1
      archive_rc=$?
      if [[ "$archive_rc" == 0 && -f "$archive.upload.json" ]]; then
        "$PY" - "$archive.upload.json" <<'PY'
import json, sys
from pathlib import Path
value = json.loads(Path(sys.argv[1]).read_text())
if value.get("private") is not False or value.get("public_hash_verified") is not True:
    raise SystemExit("campaign archive was not anonymously verified as public")
PY
        archive_rc=$?
      else
        archive_rc=1
      fi
    else
      archive_rc=1
    fi
  fi
  if [[ "$original_rc" == 0 && "$archive_rc" != 0 ]]; then
    write_status failed campaign_public_archive_failed
    original_rc=1
  fi
  exit "$original_rc"
}

trap campaign_on_exit EXIT

write_status running shared_preparation
prep_started="$(date +%s.%N)"
if ! "$PREPARE" all >"$STATE_ROOT/campaign/prepare.log" 2>&1; then
  write_status failed shared_preparation_failed; exit 1
fi

# Run the real launcher preflight for every materialized arm before paying for
# the baseline evaluation.  This follows the same arm_env -> run.sh ->
# run_arm.sh path used by training, but launch_rl.sh exits before model,
# renderer, or judge work when PAINTER_RL_PREFLIGHT_ONLY is set.  Keep this
# bounded so a malformed prepared arm cannot consume the campaign reserve.
if [[ "${PAINTER_CAMPAIGN_SKIP_PREFLIGHT:-0}" != 1 ]]; then
  PREFLIGHT_TIMEOUT="${PAINTER_MATCHED_PREFLIGHT_TIMEOUT_SECONDS:-180}"
  [[ "$PREFLIGHT_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || {
    write_status failed arm_preflight_invalid_timeout; exit 1
  }
  write_status running arm_preflight
  for ARM in flash pro luna; do
    ARM_ROOT="${PAINTER_MATCHED_ARM_ROOT_BASE:-$ROOT/..}/rl-judge-matched-20261004-$ARM"
    PREFLIGHT_LOG="$STATE_ROOT/campaign/preflight-$ARM.log"
    if ! timeout --signal=TERM --kill-after=30 "$PREFLIGHT_TIMEOUT" \
      bash -c 'set -Eeuo pipefail
        source "$1" "$2"
        export PAINTER_RL_SUPERVISED=1
        exec bash "$3/run.sh" --preflight
      ' _ "$ARM_ENV" "$ARM" "$ARM_ROOT" >"$PREFLIGHT_LOG" 2>&1; then
      write_status failed "arm_preflight_${ARM}_failed"
      exit 1
    fi
  done
fi
SHARED_ROOT="${PAINTER_MATCHED_SHARED_ROOT:-$ROOT/../rl-judge-matched-20261004-shared}"
BASELINE_ROOT="${PAINTER_MATCHED_BASELINE_ROOT:-$ROOT/../rl-judge-matched-20261004-baseline}"
FLASH_ROOT="${PAINTER_MATCHED_ARM_ROOT_BASE:-$ROOT/..}/rl-judge-matched-20261004-flash"

if [[ "$CONTINUATION" == 1 ]]; then
  # Recovery is a separate, explicit phase.  It restores only the pinned
  # public baseline/evidence and Flash step-12 checkpoint; it never calls the
  # Flash training launcher or regenerates the baseline.
  write_status running continuation_recovery
  [[ -f "$RESTORE" ]] || { write_status failed continuation_recovery_missing; exit 1; }
  RESTORE_TIMEOUT="${PAINTER_MATCHED_RESTORE_TIMEOUT_SECONDS:-600}"
  [[ "$RESTORE_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || {
    write_status failed continuation_recovery_invalid_timeout; exit 1;
  }
  RECOVERY_PY="${PRIME_ROOT:-${PAINTER_NODE_ROOT:-/root/painter}/prime}/.venv/bin/python"
  [[ -x "$RECOVERY_PY" ]] || RECOVERY_PY="$PY"
  if [[ "$RESTORE" == *.py ]]; then
    timeout --signal=TERM --kill-after=30 "$RESTORE_TIMEOUT" "$RECOVERY_PY" "$RESTORE" \
      --baseline-root "$BASELINE_ROOT" --flash-root "$FLASH_ROOT" \
      --shared-root "$SHARED_ROOT" --state-root "$STATE_ROOT" \
      >"$STATE_ROOT/campaign/recovery.log" 2>&1
  else
    timeout --signal=TERM --kill-after=30 "$RESTORE_TIMEOUT" "$RESTORE" \
      --baseline-root "$BASELINE_ROOT" --flash-root "$FLASH_ROOT" \
      --shared-root "$SHARED_ROOT" --state-root "$STATE_ROOT" \
      >"$STATE_ROOT/campaign/recovery.log" 2>&1
  fi || { write_status failed continuation_recovery_failed; exit 1; }
  "$PY" - "$BASELINE_ROOT" "$SHARED_ROOT" <<'PY'
import json, sys
from pathlib import Path
baseline, shared = map(Path, sys.argv[1:])
identity = json.loads((baseline / 'baseline-identity.json').read_text())
generation = json.loads((baseline / 'validation/step-1-balanced16/generation-receipt.json').read_text())
recovery = json.loads((baseline / 'baseline-recovery-receipt.json').read_text())
actual = json.loads((baseline / 'validation/step-1-balanced16/cases.json').read_text()).get('cases', [])
expected = json.loads((shared / 'anchors-frozen/validation-matched16.json').read_text()).get('cases', [])
if (identity.get('mode') != 'recovered_policy' or identity.get('display_checkpoint') != 'fresh_rl16_initializer'
        or identity.get('trained') is not False or identity.get('optimizer_resume') is not False
        or identity.get('initializer_sha256') != 'e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c'
        or identity.get('scope') != 'balanced16' or identity.get('case_count') != 16
        or identity.get('seed') != 918210):
    raise SystemExit('restored baseline is not the pinned fresh RL16 initializer evaluation')
if (generation.get('scope') != 'balanced16' or generation.get('adapter_sha256') != identity.get('initializer_sha256')
        or generation.get('checkpoint_provenance') != 'public_download' or generation.get('matched_control') != 'sft-step512'
        or generation.get('max_turns') != 3 or generation.get('max_tokens') != 8192):
    raise SystemExit('restored baseline generation receipt is not the strict balanced16 contract')
if recovery.get('regenerated') is not False or recovery.get('archive', {}).get('sha256') != '6967e1304b8b51aeda5b73bb00b05365edf72e44ae92baffaabaf93327158d41':
    raise SystemExit('baseline recovery receipt is not bound to the pinned public archive')
if [row.get('id') for row in actual] != [row.get('id') for row in expected] or len(actual) != 16:
    raise SystemExit('restored baseline did not use the exact shared matched16 case order')
PY

  # Bind finalization to the recovered public step before generation starts;
  # this writes the same primary checkpoint selection receipt used by the
  # ordinary pipeline without invoking a trainer.
  "$RECOVERY_PY" - "$FLASH_ROOT" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
sys.path.insert(0, str(run))
from safe_archive import select_preferred_public_checkpoint
selected, selection = select_preferred_public_checkpoint(
    run, 12, run_name='rl-judge-matched-20261004-flash'
)
if selected is None or selection.get('primary_quality_status') != 'primary':
    raise SystemExit('recovered Flash step-12 checkpoint is not the primary verified selection')
target = run / 'local-evidence/latest-public-checkpoint.json'
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps({**selected, 'selection': selection}, indent=2, sort_keys=True) + '\n')
PY

  # The recovered Flash adapter is already trained. Only the strict matched
  # generation and finalizer run here, followed by the normal public archive.
  write_status running arm_flash_recovery_running
  EVAL_TIMEOUT="${PAINTER_EVAL_TIMEOUT_SECONDS:-1800}"
  FINALIZE_TIMEOUT="${PAINTER_FINALIZE_TIMEOUT_SECONDS:-600}"
  ARCHIVE_TIMEOUT="${PAINTER_ARCHIVE_TIMEOUT_SECONDS:-600}"
  for timeout_value in "$EVAL_TIMEOUT" "$FINALIZE_TIMEOUT" "$ARCHIVE_TIMEOUT"; do
    [[ "$timeout_value" =~ ^[1-9][0-9]*$ ]] || { write_status failed continuation_invalid_timeout; exit 1; }
  done
  if ! timeout --signal=TERM --kill-after=60 "$EVAL_TIMEOUT" bash -c 'set -Eeuo pipefail
    source "$1" flash
    export PAINTER_RL_SUPERVISED=1 PAINTER_EVAL_SCOPE=balanced16 PAINTER_EVAL_EXPECTED_COUNT=16
    exec bash "$2/evaluate.sh" 12 balanced16
  ' _ "$ARM_ENV" "$FLASH_ROOT" >"$STATE_ROOT/campaign/flash-recovery-evaluate.log" 2>&1; then
    write_status failed flash_recovery_evaluation_failed; exit 1
  fi
  if ! timeout --signal=TERM --kill-after=30 "$FINALIZE_TIMEOUT" bash -c 'set -Eeuo pipefail
    source "$1" flash
    export PAINTER_RL_SUPERVISED=1 PAINTER_EVAL_SCOPE=balanced16
    exec bash "$2/finalize.sh" 12
  ' _ "$ARM_ENV" "$FLASH_ROOT" >"$STATE_ROOT/campaign/flash-recovery-finalize.log" 2>&1; then
    write_status failed flash_recovery_finalization_failed; exit 1
  fi
  FLASH_EVIDENCE="$FLASH_ROOT/local-evidence"
  FLASH_ARCHIVE="$FLASH_EVIDENCE/rl-judge-matched-20261004-flash-evidence.tar.gz"
  FLASH_UPLOAD="$FLASH_ARCHIVE.upload.json"
  mkdir -p "$FLASH_EVIDENCE"
  TOKEN="${HF_TOKEN_PATH:-${PAINTER_NODE_ROOT:-/root/painter}/private/hf-token}"
  REPO="$(cd "$ROOT/../../.." && pwd -P)"
  ARCHIVE_PY="${PRIME_ROOT:-${PAINTER_NODE_ROOT:-/root/painter}/prime}/.venv/bin/python"
  [[ -x "$ARCHIVE_PY" ]] || ARCHIVE_PY="$PY"
  "$PY" - "$FLASH_ROOT" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
(run / 'continuation-evaluation-receipt.json').write_text(json.dumps({
    'schema': 'matched-flash-recovered-evaluation-v1', 'status': 'complete',
    'evaluation_exit_code': 0, 'finalizer_exit_code': 0,
    'archive_public_hash_verified': False, 'checkpoint_step': 12,
    'actual_optimizer_updates': json.loads((run / 'supervisor-state.json').read_text()).get('completed_optimizer_steps'),
    'training_invoked': False, 'provenance_type': 'recovered_evaluation',
}, indent=2, sort_keys=True) + '\n')
PY
  if ! timeout --signal=TERM --kill-after=30 "$ARCHIVE_TIMEOUT" "$ARCHIVE_PY" \
      "$REPO/painter/runs/text-sft512-consistent-20260930/safe_archive.py" \
      --run-root "$FLASH_ROOT" --output "$FLASH_ARCHIVE" --token-file "$TOKEN" \
      --repo-id "${PAINTER_EVIDENCE_REPO_ID:-CK0607/komorebi-painter-experiments}" \
      --remote-path "${PAINTER_EVIDENCE_REMOTE_PATH:-runs/rl-judge-matched-20261004-flash/evidence.tar.gz}" \
      --verify-dir "$FLASH_EVIDENCE/public-verify" >"$STATE_ROOT/campaign/flash-recovery-archive.log" 2>&1; then
    write_status failed flash_recovery_archive_failed; exit 1
  fi
  "$PY" - "$FLASH_UPLOAD" "$FLASH_ROOT" <<'PY'
import json, sys
from pathlib import Path
upload, flash = Path(sys.argv[1]), Path(sys.argv[2])
value = json.loads(upload.read_text())
state = json.loads((flash / 'supervisor-state.json').read_text())
receipt = json.loads((flash / 'flash-recovery-receipt.json').read_text())
if (value.get('private') is not False or value.get('public_hash_verified') is not True
        or state.get('completed_optimizer_steps') != 12 or state.get('last_optimizer_step') != 12
        or receipt.get('training_invoked') is not False):
    raise SystemExit('recovered Flash archive is not publicly verified or bound to twelve prior updates')
receipt_path = flash / 'continuation-evaluation-receipt.json'
receipt = json.loads(receipt_path.read_text())
receipt['archive_public_hash_verified'] = True
receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
PY
  "$PY" - "$ARMS" "$FLASH_ROOT" "$FLASH_UPLOAD" <<'PY'
import json, sys, time
from pathlib import Path
arms, flash, upload = map(Path, sys.argv[1:])
state = json.loads((flash / 'supervisor-state.json').read_text())
value = json.loads(upload.read_text())
row = {
    'schema': 'matched-three-judge-arm-result-v1', 'arm': 'flash', 'exit_code': 0,
    'primary_success': True, 'completed_optimizer_steps': 12, 'checkpoint_step': 12,
    'training_invoked': False, 'provenance_type': 'recovered_evaluation',
    'training_provenance': 'public_checkpoint_and_archived_supervision_receipt',
    'evaluation_scope': 'balanced16', 'evaluation_step': 12,
    'terminal_receipt': str(flash / 'continuation-evaluation-receipt.json'),
    'checkpoint_receipt': str(flash / 'async-output/rl-judge-matched-20261004-flash/artifacts/adapters/step_12/hf-upload.json'),
    'archive_upload_receipt': str(upload), 'recorded_epoch': time.time(),
}
with arms.open('a', encoding='utf-8') as stream:
    stream.write(json.dumps(row, sort_keys=True) + '\n')
PY
  RUN_ARMS=(pro luna)
else
  write_status running fresh_rl16_baseline
  if ! "$BASELINE" >"$STATE_ROOT/campaign/baseline.log" 2>&1; then
    write_status failed fresh_rl16_baseline_failed; exit 1
  fi
"$PY" - "$BASELINE_ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
identity = root / 'baseline-identity.json'
generation = root / 'validation/step-1-balanced16/generation-receipt.json'
cases = root / 'validation/step-1-balanced16/cases.json'
manifest = root / 'anchors-frozen/validation-matched16.json'
if not identity.is_file():
    raise SystemExit('fresh baseline identity receipt is missing')
if not generation.is_file() or not cases.is_file() or not manifest.is_file():
    raise SystemExit('fresh baseline generation or matched16 manifest is missing')
value = json.loads(identity.read_text())
generation_value = json.loads(generation.read_text())
if value.get('mode') != 'recovered_policy' or value.get('display_checkpoint') != 'fresh_rl16_initializer' \
        or value.get('trained') is not False or value.get('optimizer_resume') is not False:
    raise SystemExit('baseline is not a fresh RL16 initializer evaluation')
if not value.get('initializer_sha256') or value.get('initializer_sha256') == 'bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740':
    raise SystemExit('baseline is missing the distinct RL16 initializer identity')
if generation_value.get('adapter_sha256') != value.get('initializer_sha256') or generation_value.get('scope') != 'balanced16':
    raise SystemExit('baseline generation receipt is not bound to the RL16 initializer')
actual = json.loads(cases.read_text()).get('cases', [])
expected = json.loads(manifest.read_text()).get('cases', [])
if [row.get('id') for row in actual] != [row.get('id') for row in expected] or len(actual) != 16:
    raise SystemExit('baseline did not use the exact matched16 case order')
PY
  RUN_ARMS=(flash pro luna)
fi

# Allocation/staging, shared preparation, and the baseline/recovery phase are
# common costs. Charge their complete elapsed wall time equally to the arms.
baseline_finished="$(date +%s.%N)"
SHARED_ARCHIVE_RESERVE_SECONDS="${PAINTER_CAMPAIGN_ARCHIVE_RESERVE_SECONDS:-360}"
SHARED_RELEASE_RESERVE_SECONDS="${PAINTER_CAMPAIGN_RELEASE_RESERVE_SECONDS:-180}"
COMMON_SHARE="$($PY - "$START" "$baseline_finished" "$SHARED_ARCHIVE_RESERVE_SECONDS" "$SHARED_RELEASE_RESERVE_SECONDS" <<'PY'
import sys
elapsed = max(0.0, float(sys.argv[2]) - float(sys.argv[1]))
future = float(sys.argv[3]) + float(sys.argv[4])
if elapsed < 0 or future < 0:
    raise SystemExit('shared campaign reserve must be non-negative')
print((elapsed + future) / 3.0)
PY
)"

for ARM in "${RUN_ARMS[@]}"; do
  ARM_ROOT="${PAINTER_MATCHED_ARM_ROOT_BASE:-$ROOT/..}/rl-judge-matched-20261004-$ARM"
  ARM_STARTED="$(date +%s.%N)"
  EFFECTIVE_START="$($PY - "$ARM_STARTED" "$COMMON_SHARE" <<'PY'
import sys
print(float(sys.argv[1]) - float(sys.argv[2]))
PY
)"
  write_status running "arm_${ARM}_running"
  set +e
  (
    source "$ROOT/arm_env.sh" "$ARM"
    "$EXECUTE" "$EFFECTIVE_START" "$PRICE" "$PARAMS"
  ) >"$STATE_ROOT/campaign/$ARM.log" 2>&1
  ARM_RC=$?
  set -e
  "$PY" - "$ARM_ROOT" "$ARM" "$ARM_RC" "$ARMS" "$ARM_STARTED" "$EFFECTIVE_START" "$PARAMS" "$START" "$baseline_finished" "$SHARED_ARCHIVE_RESERVE_SECONDS" "$SHARED_RELEASE_RESERVE_SECONDS" <<'PY'
import hashlib, json, sys, time
from pathlib import Path
root, arm, rc, arms, started, effective, cap, campaign_start, baseline_finished, archive_reserve, release_reserve = sys.argv[1:]
root = Path(root)
terminal_path = root / 'terminal-receipt.json'
state_path = root / 'supervisor-state.json'
checkpoint_path = root / 'local-evidence/latest-public-checkpoint.json'
def read(path):
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}
terminal, state, checkpoint = map(read, (terminal_path, state_path, checkpoint_path))
updates = state.get('completed_optimizer_steps', state.get('last_optimizer_step', 0))
try: updates = int(updates)
except (TypeError, ValueError): updates = 0
public_checkpoint = False
if checkpoint.get('step') == 12 and checkpoint.get('selection', {}).get('primary_quality_status') == 'primary':
    adapter_raw = checkpoint.get('adapter')
    receipt_raw = checkpoint.get('receipt')
    adapter = Path(adapter_raw).resolve() if isinstance(adapter_raw, str) else None
    receipt = Path(receipt_raw).resolve() if isinstance(receipt_raw, str) else None
    if adapter is not None and receipt is not None:
        weight = adapter / 'adapter_model.safetensors'
        upload_path = adapter / 'hf-upload.json'
        try:
            adapter.relative_to(root.resolve())
            receipt.relative_to(root.resolve())
            receipt.relative_to(adapter)
            upload = json.loads(upload_path.read_text())
            digest = hashlib.sha256(weight.read_bytes()).hexdigest()
            expected = upload.get('files', {}).get('adapter_model.safetensors', {}).get('sha256')
            public_checkpoint = (
                weight.is_file() and upload_path.is_file() and receipt == upload_path and
                upload.get('verified') is True and upload.get('private') is False and
                isinstance(expected, str) and digest == expected and
                checkpoint.get('adapter_sha256') == digest
            )
        except (OSError, ValueError, json.JSONDecodeError):
            public_checkpoint = False
primary = (
    int(rc) == 0 and terminal.get('status') == 'complete' and
    terminal.get('pipeline_exit_code') == 0 and terminal.get('train_exit_code') == 0 and
    terminal.get('evaluation_exit_code') == 0 and terminal.get('finalizer_exit_code') == 0 and
    terminal.get('checkpoint_step') == 12 and terminal.get('public_archive_hash_verified') is True and
    updates == 12 and public_checkpoint
)
row = {
    'schema': 'matched-three-judge-arm-result-v1', 'arm': arm, 'exit_code': int(rc),
    'primary_success': primary, 'completed_optimizer_steps': updates,
    'checkpoint_step': terminal.get('checkpoint_step'), 'terminal_receipt': str(terminal_path),
    'checkpoint_receipt': str(checkpoint_path), 'arm_started_epoch': float(started),
    'effective_start_epoch': float(effective), 'per_arm_cap_usd': float(cap),
    'shared_elapsed_seconds': max(0.0, float(baseline_finished) - float(campaign_start)),
    'shared_future_reserved_seconds': float(archive_reserve) + float(release_reserve),
    'shared_charge_seconds': (max(0.0, float(baseline_finished) - float(campaign_start)) +
                              float(archive_reserve) + float(release_reserve)) / 3.0,
    'recorded_epoch': time.time(),
}
with Path(arms).open('a', encoding='utf-8') as stream:
    stream.write(json.dumps(row, sort_keys=True) + '\n')
if not primary:
    raise SystemExit(f'arm {arm} did not produce the fixed step-12 primary receipt')
PY
done

"$PY" - "$STATUS" "$ARMS" "$TERMINAL" <<'PY'
import hashlib, json, sys, time
from pathlib import Path
status, arms, terminal = map(Path, sys.argv[1:])
rows = [json.loads(line) for line in arms.read_text().splitlines() if line.strip()]
if len(rows) != 3 or any(not row.get('primary_success') for row in rows):
    raise SystemExit('campaign primary arm receipts are incomplete')
payload = {'schema': 'matched-three-judge-campaign-terminal-v1', 'status': 'complete',
           'arms': rows, 'arm_status_sha256': hashlib.sha256(arms.read_bytes()).hexdigest(),
           'created_epoch': time.time()}
tmp = terminal.with_name(terminal.name + '.tmp')
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
tmp.replace(terminal)
value = json.loads(status.read_text())
value.update({'status': 'complete', 'reason': 'baseline_and_all_three_step12_primary_receipts_verified', 'updated_epoch': time.time()})
tmp = status.with_name(status.name + '.tmp'); tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n'); tmp.replace(status)
PY
echo 'MATCHED_THREE_JUDGE_CAMPAIGN_COMPLETE: baseline and all three fixed step-12 arm receipts verified'
