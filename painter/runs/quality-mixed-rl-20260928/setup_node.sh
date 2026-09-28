#!/usr/bin/env bash
set -Eeuo pipefail

# Reuse the pinned, already exercised two-GPU setup. Then stage a public,
# immutable mixed curriculum and validate the new finite config.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
OLD="$ROOT/../quality-oss-rl-20260927"
REPO="$(cd "$ROOT/../../.." && pwd -P)"
NODE_ROOT="${PAINTER_NODE_ROOT:-/root/painter}"
PRIME_ROOT="${PRIME_ROOT:-$NODE_ROOT/prime}"
MODEL_REVISION=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
MODEL_DIR="$NODE_ROOT/model-cache/qwen/$MODEL_REVISION"
RUN_NAME=quality-mixed-anchors-20260928
PACK_SHA256=1d4c098f6a40f82091cccf870611cc7c82b4e0a951762fd0d6c2172494a10fcd
PACK_REVISION=320337fd3bcf60d128b50787766e6e5be7ec9897
TRAIN_SHA256=83adfcee25a83badae322c0c6d39905bfd0f2de42a9c88c4d82e4b370e76bcf6

[[ "$(uname -s)" == Linux ]] || { echo 'Linux GPU node required' >&2; exit 1; }
PAINTER_RL_CONFIG="$OLD/native-rl-overfit-four.toml" \
  bash "$OLD/setup_node_once.sh"

NODE_ROOT="$NODE_ROOT" PACK_SHA256="$PACK_SHA256" \
PACK_REVISION="$PACK_REVISION" TRAIN_SHA256="$TRAIN_SHA256" \
  "$NODE_ROOT/bootstrap/bin/python" - <<'PY'
import hashlib, json, os, tarfile
from pathlib import Path
from huggingface_hub import hf_hub_download

root = Path(os.environ['NODE_ROOT'])
archive = Path(hf_hub_download(
    repo_id='CK0607/komorebi-painter-experiments', repo_type='dataset',
    filename='runs/quality-oss-rl-20260927/mixed-fixed-anchors-v1.tar.gz',
    revision=os.environ['PACK_REVISION'], local_dir=root/'mixed-pack-download',
))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
if sha(archive)!=os.environ['PACK_SHA256']:
    raise SystemExit('mixed curriculum public archive hash mismatch')
destination=root/'mixed-fixed-anchors-v1'
with tarfile.open(archive,'r:gz') as tar:
    members=tar.getmembers()
    if any(not m.isfile() or not m.name.startswith('mixed-fixed-anchors-v1/')
           or '..' in Path(m.name).parts for m in members):
        raise SystemExit('unsafe curriculum archive member')
    if not destination.exists():
        tar.extractall(root,filter='data')
manifest=destination/'train-manifest.json'
if sha(manifest)!=os.environ['TRAIN_SHA256']:
    raise SystemExit('mixed curriculum manifest hash mismatch')
rows=json.loads(manifest.read_text())['rows']
if len(rows)!=15 or len({r['id'] for r in rows})!=15:
    raise SystemExit('unexpected mixed curriculum count')
for row in rows:
    for key in ('reference','opponent'):
        image=destination/row[key]
        if sha(image)!=row[key+'_sha256']:
            raise SystemExit(f"{row['id']} {key} hash mismatch")
    if row['reference_sha256']==row['opponent_sha256']:
        raise SystemExit(f"{row['id']} uses reference as opponent")
print('MIXED_CURRICULUM_VERIFIED',len(rows),flush=True)
PY

CONFIG="$OLD/native-rl-mixed-anchors.toml"
MANIFEST="$NODE_ROOT/mixed-fixed-anchors-v1/train-manifest.json"
MANIFEST_REL="$(python3 - "$MANIFEST" "$OLD" <<'PY'
import os,sys
print(os.path.relpath(sys.argv[1],sys.argv[2]))
PY
)"
if [[ ! -f "$CONFIG" ]]; then
  PYTHONPATH="$REPO/painter:$OLD" "$PRIME_ROOT/.venv/bin/python" \
    "$OLD/prepare_rl_config.py" --output "$CONFIG" --model "$MODEL_DIR" \
    --run-root "$OLD" \
    --manifest "$MANIFEST_REL" \
    --expected-count 15 --max-steps 12 --checkpoint-interval 4 \
    --gpus-per-node 2 --num-train-gpus 1 --num-infer-gpus 1 \
    --batch-size 8 --group-size 4 --max-turns 4 \
    --max-completion-tokens 8192 --learning-rate 1e-6 \
    --initial-inflight 8 --max-inflight 8 --run-name "$RUN_NAME" \
    --adapter-revision adf687e5b321514eda7be6eab050434fdc8e44ff \
    --adapter-prefix photo-curriculum-sft-20260922-balanced/step_512 \
    --adapter-sha256 bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740 \
    --prime-root "$PRIME_ROOT" --validate-prime
fi
# The generic preflight validates the task wire format; it does not exercise
# the SFT/fixed-opponent startup branch. Check that branch before GPU warmup.
PYTHONPATH="$REPO/painter:$OLD" "$PRIME_ROOT/.venv/bin/python" - "$CONFIG" <<'PY'
from pathlib import Path
import sys
from run_from_sft import _prewarm_frozen_opponent
result = _prewarm_frozen_opponent(Path(sys.argv[1]))
if result.get('status') != 'fixed_verified':
    raise SystemExit('fixed-opponent startup check did not verify the anchor')
print('FIXED_ANCHOR_STARTUP_VERIFIED', result['sha256'], flush=True)
PY
PAINTER_RL_CONFIG="$CONFIG" \
  PAINTER_RL_MANIFEST="$MANIFEST" \
  PAINTER_RL_MAX_STEPS=12 PAINTER_RL_RUN_NAME="$RUN_NAME" \
  bash "$ROOT/run.sh" --preflight
echo 'MIXED_NODE_READY: launch with bash painter/runs/quality-mixed-rl-20260928/run.sh'
