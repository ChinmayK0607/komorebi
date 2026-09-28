#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd "$ROOT/../../.." && pwd -P)"
OLD="$ROOT/../quality-oss-rl-20260927"
NODE_ROOT="${PAINTER_NODE_ROOT:-/root/painter}"
PRIME_ROOT="${PRIME_ROOT:-$NODE_ROOT/prime}"
MODEL_REVISION=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
MODEL_DIR="$NODE_ROOT/model-cache/qwen/$MODEL_REVISION"
CURRICULUM="$NODE_ROOT/four-turn-relative-curriculum-20260928"
CURRICULUM_SHA256=105620c5998a4d4a5b49126588e7722bb1c9be80eb88caa92069fe9f97ed76ec
CURRICULUM_REVISION=0068bc95df0b2bc0d06c42f066761c939d9baffd
MANIFEST_SHA256=0d4cf5c68527a089dc118a02b05d767aafa81a2467d3a0c506983f41b6223ddf

[[ "$(uname -s)" == Linux ]] || { echo 'Linux GPU node required' >&2; exit 1; }
# The existing setup creates the exact pinned 27B, Prime, renderer, adapter,
# terminal judge and public old mixed-reference pack. Its finite preflight
# exercises the actual fixed-opponent path without starting training.
bash "$ROOT/../quality-mixed-rl-20260928/setup_node.sh"

"$NODE_ROOT/bootstrap/bin/python" - "$NODE_ROOT" "$CURRICULUM" \
  "$CURRICULUM_REVISION" "$CURRICULUM_SHA256" "$MANIFEST_SHA256" <<'PY'
import hashlib,json,sys,tarfile
from pathlib import Path
from huggingface_hub import hf_hub_download
node,out,revision,archive_sha,manifest_sha=sys.argv[1:]
node,out=Path(node),Path(out)
archive=Path(hf_hub_download('CK0607/komorebi-painter-experiments',repo_type='dataset',
    filename='runs/quality-relative-rl-20260928/four-turn-relative-curriculum-20260928.tar.gz',
    revision=revision,token=False,local_dir=node/'relative-curriculum-download'))
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
if digest(archive)!=archive_sha:raise SystemExit('public curriculum archive hash mismatch')
with tarfile.open(archive) as bundle:
    members=bundle.getmembers()
    if any(not item.isfile() or not item.name.startswith('four-turn-relative-curriculum-20260928/')
           or '..' in Path(item.name).parts for item in members):
        raise SystemExit('unsafe curriculum archive')
    if not out.exists():bundle.extractall(node,filter='data')
manifest=out/'train-manifest.json'
if digest(manifest)!=manifest_sha:raise SystemExit('curriculum manifest hash mismatch')
rows=json.loads(manifest.read_text())['rows']
if len(rows)!=9 or len({row['id'] for row in rows})!=9:raise SystemExit('wrong curriculum count')
for row in rows:
    for role in ('reference','opponent'):
        if digest(out/row[role])!=row[role+'_sha256']:
            raise SystemExit(f"{row['id']} {role} hash mismatch")
print('FOUR_TURN_OPPONENTS_VERIFIED',len(rows),flush=True)
PY

"$PRIME_ROOT/.venv/bin/python" "$ROOT/install_prime_hook.py" "$PRIME_ROOT"
CONFIG="$OLD/native-rl-relative-eight.toml"
MANIFEST_REL="$(python3 - "$CURRICULUM/train-manifest.json" "$OLD" <<'PY'
import os,sys
print(os.path.relpath(sys.argv[1],sys.argv[2]))
PY
)"
if [[ ! -f "$CONFIG" ]]; then
  PYTHONPATH="$ROOT:$REPO/painter:$OLD" "$PRIME_ROOT/.venv/bin/python" \
    "$OLD/prepare_rl_config.py" --output "$CONFIG" --model "$MODEL_DIR" \
    --run-root "$OLD" --manifest "$MANIFEST_REL" \
    --expected-count 9 --max-steps 4 --checkpoint-interval 2 \
    --gpus-per-node 2 --num-train-gpus 1 --num-infer-gpus 1 \
    --batch-size 8 --group-size 8 --max-turns 4 --max-completion-tokens 8192 \
    --learning-rate 1e-6 --initial-inflight 8 --max-inflight 8 \
    --run-name quality-relative-eight-20260928 \
    --adapter-revision adf687e5b321514eda7be6eab050434fdc8e44ff \
    --adapter-prefix photo-curriculum-sft-20260922-balanced/step_512 \
    --adapter-sha256 bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740 \
    --prime-root "$PRIME_ROOT" --validate-prime
fi
PYTHONPATH="$ROOT:$REPO/painter:$OLD" "$PRIME_ROOT/.venv/bin/python" - "$CONFIG" <<'PY'
import sys,tomllib
from pathlib import Path
from run_from_sft import _prewarm_frozen_opponent
config=tomllib.loads(Path(sys.argv[1]).read_text())
env=config['orchestrator']['train']['source'][0]['env']
assert config['orchestrator']['group_size']==8
assert config['orchestrator']['batch_size']==8
assert config['orchestrator']['max_steps']==4
assert env['taskset']['expected_count']==9
result=_prewarm_frozen_opponent(Path(sys.argv[1]))
if result.get('status')!='fixed_verified':raise SystemExit('fixed opponent startup failed')
print('GROUP_EIGHT_AND_ANCHOR_VERIFIED',flush=True)
PY
PAINTER_RL_CONFIG="$CONFIG" bash "$ROOT/run.sh" --preflight
echo 'RELATIVE_EIGHT_NODE_READY: launch with bash painter/runs/quality-relative-rl-20260928/run.sh'
