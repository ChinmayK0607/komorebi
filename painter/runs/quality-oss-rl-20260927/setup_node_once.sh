#!/usr/bin/env bash
set -Eeuo pipefail

# Run once after cloning the committed repository onto a Linux GPU node.
# Public model/data downloads, runtime setup, manifest construction and a
# no-GPU launch preflight are included. Actual RL launch remains separate.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd "$ROOT/../../.." && pwd -P)"
NODE_ROOT="${PAINTER_NODE_ROOT:-/root/painter}"
PRIME_ROOT="${PRIME_ROOT:-$NODE_ROOT/prime}"
BOOTSTRAP="$NODE_ROOT/bootstrap"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$NODE_ROOT/private/hf-token}"
export GATEWAY_KEY_FILE="${GATEWAY_KEY_FILE:-$NODE_ROOT/private/gateway.env}"
MODEL_REVISION=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
MODEL_DIR="$NODE_ROOT/model-cache/qwen/$MODEL_REVISION"

[[ "$(uname -s)" == Linux ]] || { echo 'Linux node required' >&2; exit 1; }
[[ -f "$REPO/painter/foundation-lock.json" ]] || { echo 'painter checkout missing' >&2; exit 1; }
for credential in "$HF_TOKEN_PATH" "$GATEWAY_KEY_FILE"; do
  [[ -f "$credential" && ! -L "$credential" ]] || {
    echo "Protected credential file missing: $credential" >&2; exit 1;
  }
  [[ "$(stat -c %a "$credential")" == 600 ]] || {
    echo "Credential file must have mode 600: $credential" >&2; exit 1;
  }
done
[[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -ge 2 ]] || {
  echo 'This profile requires two visible GPUs' >&2; exit 1;
}
mkdir -p "$NODE_ROOT" "$MODEL_DIR" "$ROOT/logs"
for name in launch_rl.sh prepare_rl_config.py run_from_sft.py train_from_sft.py aesthetic_judge.py judge_queue.py; do
  cp "$ROOT/../photo-curriculum-rl-20260922/$name" "$ROOT/$name"
done
python3 -m venv "$BOOTSTRAP"
"$BOOTSTRAP/bin/python" -m pip install -q huggingface_hub hf_xet
if command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
else
  "$BOOTSTRAP/bin/python" -m pip install -q uv
  UV_BIN="$BOOTSTRAP/bin/uv"
fi

download_public_inputs() {
  MODEL_DIR="$MODEL_DIR" ROOT="$ROOT" NODE_ROOT="$NODE_ROOT" "$BOOTSTRAP/bin/python" - <<'PY'
import hashlib
import os
import shutil
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

root = Path(os.environ["ROOT"])
node = Path(os.environ["NODE_ROOT"])
model_dir = Path(os.environ["MODEL_DIR"])
token_file = Path(os.environ["HF_TOKEN_PATH"])
token = token_file.read_text().strip() if token_file.is_file() else None
snapshot_download("Qwen/Qwen3.8-27B",
                  revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
                  local_dir=model_dir, token=token)
archive = Path(hf_hub_download(
    repo_id="CK0607/komorebi-painter-experiments", repo_type="dataset",
    filename="runs/quality-oss-rl-20260927/profile-evidence.tar.gz",
    local_dir=node / "profile-download", token=token,
))
expected = "b48659cdedae0121b7a4388658d97d7a760acf5cf7f4e0fe9d9c5d9ce8a028d0"
if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
    raise ValueError("profile archive hash mismatch")
profile = root / "profile-data"
if not profile.exists():
    extracted = node / "profile-extract"
    with tarfile.open(archive) as handle:
        handle.extractall(extracted, filter="data")
    source = extracted / "painter/runs/quality-oss-rl-20260927/profile-data"
    shutil.copytree(source, profile)
if hashlib.sha256((profile / "profile-manifest.json").read_bytes()).hexdigest() != "c974e038354d4f663e9987b0ea53b1e61be22001d9cace080ac767c62322b4e6":
    raise ValueError("profile manifest hash mismatch")
dog = root.parent / "scaled-curriculum-20260921/real-pool/raw/coco128/images/train2017/000000000490.jpg"
dog_hash = "1274c3c4010d45ddd84faf1c56f7c997c0c2533e4f25c6ccea25e8233e5684e5"
if not dog.is_file():
    coco_archive = node / "coco128.zip"
    if not coco_archive.is_file():
        url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
        for attempt in range(3):
            try:
                urllib.request.urlretrieve(url, coco_archive)
                break
            except Exception:
                coco_archive.unlink(missing_ok=True)
                if attempt == 2:
                    raise
                time.sleep(5)
    if hashlib.sha256(coco_archive.read_bytes()).hexdigest() != "61e5e3028863d8ffc3b81d6a514603954889f0edd5e4b44c4ce60b2da99aeb8e":
        raise ValueError("upstream COCO128 archive hash mismatch")
    with zipfile.ZipFile(coco_archive) as archive:
        name = "coco128/images/train2017/000000000490.jpg"
        content = archive.read(name)
    dog.parent.mkdir(parents=True, exist_ok=True)
    dog.write_bytes(content)
if hashlib.sha256(dog.read_bytes()).hexdigest() != dog_hash:
    raise ValueError("COCO128 dog reference hash mismatch")
print("PUBLIC_INPUTS_VERIFIED", flush=True)
PY
  "$BOOTSTRAP/bin/python" \
    "$ROOT/../photo-curriculum-rl-20260922/download_initial_adapter.py"
  "$BOOTSTRAP/bin/python" "$ROOT/prepare_overfit_data.py" \
    --profile "$ROOT/profile-data/profile-manifest.json" \
    --output "$ROOT/profile-data/overfit-four-manifest.json"
}

download_public_inputs >"$ROOT/logs/public-inputs-setup.log" 2>&1 &
inputs_pid=$!
trap 'kill "$inputs_pid" 2>/dev/null || true' EXIT

if [[ ! -d "$PRIME_ROOT/.git" ]]; then
  [[ ! -e "$PRIME_ROOT" ]] || { echo "Non-Git Prime path exists: $PRIME_ROOT" >&2; exit 1; }
  temporary="$PRIME_ROOT.clone.$$"
  for attempt in 1 2 3; do
    if git -c http.version=HTTP/1.1 clone -q https://github.com/PrimeIntellect-ai/prime-rl.git "$temporary"; then
      mv "$temporary" "$PRIME_ROOT"
      break
    fi
    rm -rf "$temporary"
    [[ "$attempt" -lt 3 ]] || { echo 'Prime clone failed' >&2; exit 1; }
    sleep 5
  done
fi
for attempt in 1 2 3; do
  if git -C "$PRIME_ROOT" -c http.version=HTTP/1.1 fetch -q origin 26b3131d2716a4f8210b165df584b83b4bc54f61; then
    break
  fi
  [[ "$attempt" -lt 3 ]] || { echo 'Pinned Prime fetch failed' >&2; exit 1; }
  sleep 5
done
git -C "$PRIME_ROOT" checkout -q 26b3131d2716a4f8210b165df584b83b4bc54f61
PRIME_ROOT="$PRIME_ROOT" PAINTER_RENDER_ROOT="$REPO/painter" UV_BIN="$UV_BIN" \
  bash "$ROOT/../photo-curriculum-rl-20260922/setup_rl_runtime.sh" "$ROOT"
if ! wait "$inputs_pid"; then
  tail -20 "$ROOT/logs/public-inputs-setup.log" >&2
  exit 1
fi
trap - EXIT

if [[ ! -f "$ROOT/native-rl-overfit-four.toml" ]]; then
  PYTHONPATH="$REPO/painter:$ROOT" "$PRIME_ROOT/.venv/bin/python" "$ROOT/prepare_rl_config.py" \
    --output "$ROOT/native-rl-overfit-four.toml" --model "$MODEL_DIR" \
    --run-root "$ROOT" --manifest profile-data/overfit-four-manifest.json \
    --expected-count 4 --max-steps 4 --checkpoint-interval 1 \
    --gpus-per-node 2 --num-train-gpus 1 --num-infer-gpus 1 \
    --batch-size 8 --group-size 4 --max-turns 4 \
    --max-completion-tokens 8192 --learning-rate 1e-6 \
    --run-name gateway-overfit-four-20260928 \
    --adapter-revision adf687e5b321514eda7be6eab050434fdc8e44ff \
    --adapter-prefix photo-curriculum-sft-20260922-balanced/step_512 \
    --adapter-sha256 bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740 \
    --prime-root "$PRIME_ROOT" --validate-prime
fi

MODEL_DIR="$MODEL_DIR" PRIME_ROOT="$PRIME_ROOT" PAINTER_REPO_ROOT="$REPO" \
  BRUSH_ROOT="$REPO/painter/rl_helpers" \
  bash "$ROOT/run_overfit.sh" --preflight
echo 'NODE_READY: launch with bash painter/runs/quality-oss-rl-20260927/run_overfit.sh'
