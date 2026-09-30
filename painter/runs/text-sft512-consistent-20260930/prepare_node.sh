#!/usr/bin/env bash
set -Eeuo pipefail

# Finite, idempotent Linux preparation.  This stages the known-good bootstrap,
# fixed SFT anchors, and one frozen config; it never starts the trainer.
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd "$RUN/../../.." && pwd -P)"
NODE_ROOT="${PAINTER_NODE_ROOT:-/root/painter}"
PRIME_ROOT="${PRIME_ROOT:-$NODE_ROOT/prime}"
BOOTSTRAP_RUN="${PAINTER_BOOTSTRAP_RUN:-}"
MODEL_REVISION="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
MODEL_DIR="${MODEL_DIR:-$NODE_ROOT/model-cache/qwen/$MODEL_REVISION}"
RUN_NAME="text-sft512-consistent-20260930"
MANIFEST_ROOT="$RUN/anchors-frozen"
SFT_PATH_FILE="${PAINTER_SFT_ADAPTER_PATH_FILE:-$RUN/../photo-curriculum-rl-20260922/initial-adapter-path.txt}"
ARCHIVE="$NODE_ROOT/cache/scaled-final-20260929.tar.gz"
ARCHIVE_SHA256="ba4252251a54f9eeab16cff7cbf45939ee37e57730a8fa9191fe9121f235ccd3"
ANCHORS_ALL="$RUN/anchors-all"

[[ "$(uname -s)" == Linux ]] || { echo 'Linux GPU node required' >&2; exit 1; }

if [[ -z "$BOOTSTRAP_RUN" ]]; then
  for candidate in "$RUN"/../text-pairwise-rl-*; do
    if [[ -f "$candidate/setup_node.sh" && -f "$candidate/run.sh" ]]; then
      BOOTSTRAP_RUN="$(cd "$candidate" && pwd -P)"
      break
    fi
  done
fi
[[ -n "$BOOTSTRAP_RUN" && -f "$BOOTSTRAP_RUN/setup_node.sh" ]] || {
  echo 'set PAINTER_BOOTSTRAP_RUN to the reviewed two-GPU bootstrap directory' >&2; exit 1;
}

# The bootstrap validates the pinned model, Prime checkout, renderer bundle,
# and public SFT adapter.  It exits before any server or optimizer starts.
bash "$BOOTSTRAP_RUN/setup_node.sh"

for item in launch_rl.sh run_from_sft.py train_from_sft.py; do
  [[ -f "$BOOTSTRAP_RUN/$item" ]] || { echo "bootstrap output missing: $item" >&2; exit 1; }
  cp "$BOOTSTRAP_RUN/$item" "$RUN/$item"
done
for item in vendor renderer-env browsers; do
  [[ -e "$BOOTSTRAP_RUN/$item" ]] || { echo "bootstrap renderer asset missing: $item" >&2; exit 1; }
  if [[ ! -e "$RUN/$item" ]]; then
    ln -s "$BOOTSTRAP_RUN/$item" "$RUN/$item"
  fi
done

[[ -r "$SFT_PATH_FILE" ]] || { echo "verified SFT adapter path file missing: $SFT_PATH_FILE" >&2; exit 1; }
SFT_ADAPTER="$(tr -d '\r\n' < "$SFT_PATH_FILE")"
[[ -f "$SFT_ADAPTER/adapter_model.safetensors" && -f "$SFT_ADAPTER/adapter_config.json" ]] || {
  echo 'verified SFT adapter directory is incomplete' >&2; exit 1;
}

# Reuse the public, hash-verified 184-brief evidence archive. Rehydration
# checks every prompt/canvas hash and yields the immutable 119-row source;
# curation then writes the reviewed 40-row fixed-opponent training split.
mkdir -p "$NODE_ROOT/cache"
if [[ ! -f "$ARCHIVE" ]]; then
  curl -fLsS --retry 3 --retry-delay 2 \
    'https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/resolve/e39e6e58fc830eb7be0a77a8bdde7d906943d1d3/runs/text-pairwise-rl-20260928/scaled-final-20260929.tar.gz' \
    -o "$ARCHIVE.tmp"
  mv "$ARCHIVE.tmp" "$ARCHIVE"
fi
"$NODE_ROOT/bootstrap/bin/python" - "$ARCHIVE" "$ARCHIVE_SHA256" <<'PY'
import hashlib, sys
from pathlib import Path
path, expected = Path(sys.argv[1]), sys.argv[2]
actual = hashlib.sha256(path.read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit(f"public evidence archive hash mismatch: {actual}")
PY
if [[ ! -f "$ANCHORS_ALL/reuse-receipt.json" ]]; then
  [[ ! -e "$ANCHORS_ALL" ]] || {
    echo 'partial anchor rehydration exists without receipt; preserve it and inspect before retrying' >&2
    exit 1
  }
  "$PRIME_ROOT/.venv/bin/python" "$BOOTSTRAP_RUN/rehydrate_scaled_anchors.py" \
    --archive "$ARCHIVE" --output "$ANCHORS_ALL"
fi
[[ -f "$ANCHORS_ALL/train-manifest.json" && -f "$ANCHORS_ALL/validation-manifest.json" && \
   -f "$ANCHORS_ALL/reuse-receipt.json" ]] || { echo 'rehydrated anchor source is incomplete' >&2; exit 1; }
if [[ ! -f "$MANIFEST_ROOT/train-manifest.json" ]]; then
  mkdir -p "$MANIFEST_ROOT"
  "$PRIME_ROOT/.venv/bin/python" "$BOOTSTRAP_RUN/curate_grounded_pool.py" \
    --source "$ANCHORS_ALL/train-manifest.json" \
    --output "$MANIFEST_ROOT/train-manifest.json"
  cp "$ANCHORS_ALL/validation-manifest.json" "$MANIFEST_ROOT/validation-manifest.json"
  cp "$ANCHORS_ALL/reserve-manifest.json" "$MANIFEST_ROOT/reserve-manifest.json"
fi
[[ -f "$MANIFEST_ROOT/train-manifest.json" && -f "$MANIFEST_ROOT/validation-manifest.json" ]] || {
  echo 'curated fixed-anchor manifests are incomplete' >&2; exit 1;
}
"$PRIME_ROOT/.venv/bin/python" - "$MANIFEST_ROOT" "$ANCHORS_ALL" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, source = map(Path, sys.argv[1:])
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
train = json.loads((root / 'train-manifest.json').read_text())['rows']
if len(train) != 40 or len({row['id'] for row in train}) != 40:
    raise SystemExit('curated training manifest is not the reviewed 40-row set')
semantic = [{k: row[k] for k in ('id', 'family', 'reference_sha256', 'opponent_sha256')} for row in train]
content = hashlib.sha256(json.dumps(semantic, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
if content != '13ac0eebb3595fbd931ee7864e69101abdc8b7c9c1faccf11dfcd9bd495c6501':
    raise SystemExit('curated training semantic hash mismatch')
for row in train:
    for key in ('reference', 'opponent'):
        path = Path(row[key])
        if sha(path) != row[key + '_sha256']:
            raise SystemExit(f"fixed anchor hash mismatch: {row['id']} {key}")
validation = json.loads((root / 'validation-manifest.json').read_text())['cases']
if len(validation) != 47 or len({row['id'] for row in validation}) != 47:
    raise SystemExit('rehydrated validation manifest is not the immutable 47-case set')
receipt = json.loads((source / 'reuse-receipt.json').read_text())
if receipt.get('source_archive_sha256') != 'ba4252251a54f9eeab16cff7cbf45939ee37e57730a8fa9191fe9121f235ccd3':
    raise SystemExit('rehydrated source archive identity mismatch')
print('FIXED_ANCHORS_VERIFIED', content, len(train), len(validation), flush=True)
PY

PY="$PRIME_ROOT/.venv/bin/python"
[[ -x "$PY" ]] || { echo "pinned Prime Python missing: $PY" >&2; exit 1; }
PROPOSED="$RUN/native.toml.proposed.$$"
trap 'rm -f "$PROPOSED"' EXIT
"$PY" "$REPO/painter/runs/text-pairwise-rl-20260928/prepare_config.py" \
  --model "$MODEL_DIR" --run-root "$RUN" \
  --manifest "$MANIFEST_ROOT/train-manifest.json" --output "$PROPOSED" \
  --prime-root "$PRIME_ROOT" --expected-count "${PAINTER_EXPECTED_COUNT:-40}" \
  --max-steps 50 --checkpoint-interval "${PAINTER_CHECKPOINT_INTERVAL:-4}" \
  --run-name "$RUN_NAME" --output-root rl-rollouts \
  --judge-protocol text-paint-sft-pairwise-consistent-v1 \
  --mode fresh --adapter-revision adf687e5b321514eda7be6eab050434fdc8e44ff \
  --adapter-prefix photo-curriculum-sft-20260922-balanced/step_512 \
  --adapter-sha256 bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740 \
  --gpus-per-node "${PAINTER_GPUS_PER_NODE:-2}" \
  --num-train-gpus "${PAINTER_NUM_TRAIN_GPUS:-1}" \
  --num-infer-gpus "${PAINTER_NUM_INFER_GPUS:-1}" \
  --batch-size 16 --group-size 8 \
  --initial-inflight "${PAINTER_INITIAL_INFLIGHT:-16}" \
  --max-inflight "${PAINTER_MAX_INFLIGHT:-16}" \
  --learning-rate 1e-6
if [[ -e "$RUN/native.toml" ]]; then
  cmp -s "$PROPOSED" "$RUN/native.toml" || {
    echo 'existing native.toml differs from the frozen proposal' >&2; exit 1;
  }
else
  mv "$PROPOSED" "$RUN/native.toml"
fi
rm -f "$PROPOSED"
trap - EXIT

"$PY" - "$RUN" "$SFT_ADAPTER" <<'PY'
import hashlib, json, sys
from pathlib import Path
run, adapter = Path(sys.argv[1]), Path(sys.argv[2])
weight = adapter / "adapter_model.safetensors"
digest = hashlib.sha256(weight.read_bytes()).hexdigest()
if digest != "bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740":
    raise SystemExit("SFT512 adapter hash mismatch")
manifest = run / "anchors-frozen/train-manifest.json"
manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
config = run / "native.toml"
config_sha = hashlib.sha256(config.read_bytes()).hexdigest()
(run / "preparation-receipt.json").write_text(json.dumps({
    "schema": "text-sft512-consistent-preparation-v1",
    "run_name": "text-sft512-consistent-20260930",
    "base_model_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "prime_revision": "26b3131d2716a4f8210b165df584b83b4bc54f61",
    "initializer_revision": "adf687e5b321514eda7be6eab050434fdc8e44ff",
    "initializer_prefix": "photo-curriculum-sft-20260922-balanced/step_512",
    "initializer_sha256": digest,
    "frozen_anchor_sha256": digest,
    "train_manifest_sha256": manifest_sha,
    "config_sha256": config_sha,
    "judge_protocol": "text-paint-sft-pairwise-consistent-v1",
    "optimizer_resume": False,
}, indent=2, sort_keys=True) + "\n")
PY
echo 'TEXT_SFT512_CONSISTENT_PREPARED: finite config and fixed anchors ready; launch via finite_supervisor.py'
