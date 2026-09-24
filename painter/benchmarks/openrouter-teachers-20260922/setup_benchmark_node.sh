#!/usr/bin/env bash
# Idempotent renderer/AI-SDK bootstrap for one fresh Linux benchmark node.
set -Eeuo pipefail
BOOTSTRAP_PHASE=preflight
trap 'printf "Renderer bootstrap failed in %s at line %s\n" "$BOOTSTRAP_PHASE" "$LINENO" >&2' ERR
ROOT=$(realpath "${1:?node workspace required}")
[[ "$ROOT" == /* && "$ROOT" != "/" && $(uname -s) == Linux ]]
[[ ${EUID:-$(id -u)} -eq 0 ]]
SOURCE_BUNDLE="${SOURCE_BUNDLE:-$ROOT/source.bundle}"
REPO_ROOT="${REPO_ROOT:-$ROOT/repo}"
REFERENCES_ARCHIVE="${REFERENCES_ARCHIVE:-$ROOT/references.tar.gz}"
mkdir -p "$ROOT/logs" "$ROOT/cache" "$ROOT/browsers"
apt_ready=0
apt_install() {
  if (( apt_ready == 0 )); then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt_ready=1
  fi
  export DEBIAN_FRONTEND=noninteractive
  apt-get install -y --no-install-recommends "$@"
}
python_ok=false
command -v python3 >/dev/null 2>&1 && python3 -c 'import venv' >/dev/null 2>&1 && python_ok=true
node_major=0
command -v node >/dev/null 2>&1 && node_major=$(node --version | sed -E 's/^v([0-9]+).*/\1/' || true)
if ! [[ "$node_major" =~ ^[0-9]+$ ]]; then node_major=0; fi
if [[ "$python_ok" != true || ! -x /usr/sbin/runuser || ! -x /usr/bin/curl || ! "$(command -v git || true)" ]]; then
  apt_install ca-certificates curl python3 python3-venv python3-pip util-linux git
fi
node_major=$(node --version | sed -E 's/^v([0-9]+).*/\1/' || true)
if ! [[ "$node_major" =~ ^[0-9]+$ ]] || (( node_major < 22 )); then
  BOOTSTRAP_PHASE=node-install
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
  apt_install nodejs
fi
node_major=$(node --version | sed -E 's/^v([0-9]+).*/\1/' || true)
if ! [[ "$node_major" =~ ^[0-9]+$ ]] || (( node_major < 22 )); then
  echo "Node.js >=22 required; found $(node --version 2>/dev/null || echo missing)" >&2
  exit 1
fi
if [[ -f "$SOURCE_BUNDLE" ]]; then
  if [[ ! -d "$REPO_ROOT/.git" ]]; then
    mkdir -p "$REPO_ROOT"
    git clone "$SOURCE_BUNDLE" "$REPO_ROOT"
  else
    [[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || { echo "refusing to replace a dirty node checkout: $REPO_ROOT" >&2; exit 1; }
    git -C "$REPO_ROOT" fetch "$SOURCE_BUNDLE" 'refs/heads/*:refs/remotes/bundle/*' >/dev/null
    current_branch=$(git -C "$REPO_ROOT" symbolic-ref --short HEAD 2>/dev/null || true)
    target="refs/remotes/bundle/$current_branch"
    if ! git -C "$REPO_ROOT" show-ref --verify --quiet "$target"; then
      for branch in main master; do
        if git -C "$REPO_ROOT" show-ref --verify --quiet "refs/remotes/bundle/$branch"; then
          target="refs/remotes/bundle/$branch"; break
        fi
      done
    fi
    git -C "$REPO_ROOT" show-ref --verify --quiet "$target" || { echo "source bundle has no matching branch" >&2; exit 1; }
    git -C "$REPO_ROOT" reset --hard "$target" >/dev/null
  fi
fi
if [[ -d "$REPO_ROOT/painter" ]]; then
  CODE_ROOT="$REPO_ROOT/painter"
else
  CODE_ROOT="$ROOT"
fi
RENDERER="$CODE_ROOT/vendor/integrations/watercolour/renderer.py"
ASSETS="$CODE_ROOT/vendor/integrations/watercolour/assets"
BENCHMARK="$CODE_ROOT/benchmarks/openrouter-teachers-20260922"
PY="$ROOT/renderer-env/bin/python"
BROWSERS="$ROOT/browsers"
TQDM_VERSION="4.67.1"
if [[ "$(pnpm --version 2>/dev/null || true)" != "10.15.1" ]]; then
  BOOTSTRAP_PHASE=pnpm-install
  npm install --global pnpm@10.15.1
fi
BOOTSTRAP_PHASE=source-check
[[ "$(pnpm --version)" == "10.15.1" ]]
if ! id painter >/dev/null 2>&1; then
  useradd --create-home --shell /bin/bash painter
fi
[[ -f "$RENDERER" && -d "$ASSETS" ]]
[[ -f "$BENCHMARK/run.py" && -f "$BENCHMARK/pnpm-lock.yaml" && -x "$BENCHMARK/run_benchmark_node.sh" ]]
if [[ -f "$REFERENCES_ARCHIVE" ]]; then
  python3 - "$REFERENCES_ARCHIVE" "$BENCHMARK" <<'PY'
import hashlib, json, pathlib, sys, tarfile
archive_path, target_root = map(pathlib.Path, sys.argv[1:])
target_root = target_root.resolve()
with tarfile.open(archive_path, "r:gz") as archive:
    members = archive.getmembers()
    manifest_member = next((member for member in members if member.name == "references-manifest.json"), None)
    if manifest_member is None or not manifest_member.isfile():
        raise SystemExit("reference archive has no manifest")
    manifest = json.loads(archive.extractfile(manifest_member).read())
    rows = manifest.get("files") if isinstance(manifest, dict) else None
    if manifest.get("count") != 40 or not isinstance(rows, list) or len(rows) != 40:
        raise SystemExit("reference manifest must contain exactly 40 files")
    by_name = {member.name: member for member in members if member.isfile()}
    for row in rows:
        name, expected = row.get("path"), row.get("sha256")
        if not isinstance(name, str) or not name.startswith("references/") or ".." in pathlib.PurePosixPath(name).parts:
            raise SystemExit("unsafe reference archive path")
        member = by_name.get(name)
        if member is None:
            raise SystemExit("reference archive member missing: " + name)
        target = (target_root / name).resolve()
        if target_root not in target.parents:
            raise SystemExit("reference archive escapes benchmark root")
        data = archive.extractfile(member).read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise SystemExit("reference hash mismatch: " + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        target.chmod(0o644)
PY
fi
BOOTSTRAP_PHASE=reference-check
if [[ "${PAINTER_SETUP_SKIP_BENCHMARK_REFERENCES:-0}" != 1 ]]; then
  [[ $(find "$BENCHMARK/references" -maxdepth 1 -type f -name '*.jpg' | wc -l) -eq 40 ]]
fi
if [[ ! -x "$PY" ]] || ! "$PY" -c 'import importlib.metadata as m; assert m.version("playwright") == "1.58.0" and m.version("Pillow") == "12.3.0" and m.version("tqdm") == "4.67.1"' >/dev/null 2>&1; then
  BOOTSTRAP_PHASE=python-packages
  python3 -m venv "$ROOT/renderer-env"
  "$PY" -m pip install --disable-pip-version-check --no-input playwright==1.58.0 pillow==12.3.0 "tqdm==$TQDM_VERSION" >"$ROOT/logs/renderer-install.log" 2>&1
fi
browser_ok=false
if [[ -d "$BROWSERS" ]] && "$PY" - "$BROWSERS" <<'PY'
import os,sys
from pathlib import Path
root=Path(sys.argv[1])
raise SystemExit(0 if any(p.is_file() and os.access(p,os.X_OK) and p.name in {"chrome","chrome-headless-shell"} for p in root.rglob("*")) else 1)
PY
then browser_ok=true; fi
if [[ "$browser_ok" != true ]]; then
  BOOTSTRAP_PHASE=chromium-install
  PLAYWRIGHT_BROWSERS_PATH="$BROWSERS" "$PY" -m playwright install --with-deps chromium >>"$ROOT/logs/renderer-install.log" 2>&1
fi

# Install the checked-in AI SDK transport dependencies in the benchmark
# directory. This is separate from the renderer environment and is safe to
# repeat because the lockfile is authoritative.
(cd "$BENCHMARK" && pnpm install --frozen-lockfile --ignore-scripts) >"$ROOT/logs/benchmark-pnpm-install.log" 2>&1
BOOTSTRAP_PHASE=renderer-smoke
cursor="$ROOT"
while [[ "$cursor" != / ]]; do chmod o+x "$cursor"; cursor=$(dirname "$cursor"); done
chmod -R a+rX "$CODE_ROOT/vendor" "$ROOT/renderer-env" "$ROOT/browsers"
mkdir -p "$ROOT/.benchmark-render-smoke"
SMOKE="$ROOT/.benchmark-render-smoke"
rm -f "$SMOKE/program.js" "$SMOKE/canvas.png" "$SMOKE/canvas.json"
mkdir -p "$SMOKE/home"; chmod 777 "$SMOKE" "$SMOKE/home"
cat >"$SMOKE/program.js" <<'JS'
function setup(){createCanvas(600,600,WEBGL);pixelDensity(1);randomSeed(17);brush.seed(17);background(248,245,237);}
function draw(){translate(-300,-300);brush.noStroke();brush.noWash();brush.fill("#c8704a",220);brush.polygon([[100,220],[300,100],[500,220],[380,430],[180,430]]);noLoop();}
JS
chmod 644 "$SMOKE/program.js"; chown -R painter:painter "$SMOKE"
runuser -u painter -- env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME="$SMOKE/home" TMPDIR="$SMOKE" LANG=C.UTF-8 PLAYWRIGHT_BROWSERS_PATH="$BROWSERS" "$PY" "$RENDERER" "$SMOKE/program.js" --output "$SMOKE/canvas.png" --timeout 180 --backend swiftshader >"$SMOKE/renderer-output.json"
"$PY" - "$SMOKE" "$RENDERER" <<'PY'
import hashlib,json,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]); renderer=Path(sys.argv[2])
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""): h.update(b)
 return h.hexdigest()
receipt=json.loads((root/"canvas.json").read_text()); image=root/"canvas.png"; source=root/"program.js"
if not receipt.get("valid") or not image.is_file(): raise SystemExit("renderer smoke invalid")
if receipt.get("source_sha256") != sha(source) or receipt.get("png_sha256") != sha(image): raise SystemExit("renderer smoke hash mismatch")
ready={"status":"ready","renderer_sha256":sha(renderer),"playwright_version":"1.58.0","pillow_version":"12.3.0","node_version":subprocess.check_output(["node","--version"],text=True).strip(),"pnpm_version":subprocess.check_output(["pnpm","--version"],text=True).strip(),"chromium_version":receipt.get("chromium_version"),"backend":receipt.get("requested_angle_backend"),"canvas_sha256":sha(image),"source_sha256":sha(source),"network_enabled":receipt.get("network_enabled"),"browser_sandbox_enabled":receipt.get("browser_sandbox_enabled")}
(root.parent/"benchmark-node-ready.json").write_text(json.dumps(ready,indent=2,sort_keys=True)+"\n")
print(json.dumps(ready,sort_keys=True))
PY
rm -rf "$SMOKE"
cat <<EOF
Ready: benchmark node environment. No benchmark request was started.
Run manually from the node (the command prompts for the Gateway key on a TTY):
cd $BENCHMARK
bash $BENCHMARK/run_benchmark_node.sh
EOF
