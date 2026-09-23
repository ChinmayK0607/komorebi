#!/usr/bin/env bash
# Codex cloud environment setup command. No paid model calls or credentials.
set -Eeuo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
BENCHMARK="$REPO_ROOT/painter/benchmarks/openrouter-teachers-20260922"
RUNTIME_ROOT="$REPO_ROOT/.painter-cloud-runtime"

# The universal image includes an LLVM apt source that this benchmark never
# needs. Codex task setup can receive a proxy 403 for it, which aborts every
# apt-get update even though the required Ubuntu/Node sources are reachable.
# Move only that source aside in this disposable container; preserve it under
# the ignored runtime directory so the change is reversible.
mkdir -p "$RUNTIME_ROOT/apt-sources-disabled"
for source in /etc/apt/sources.list.d/*; do
  [[ -f "$source" && ! -L "$source" ]] || continue
  if grep -q 'apt\.llvm\.org' "$source"; then
    if grep -Eq 'snapshot\.ubuntu\.com|archive\.ubuntu\.com|security\.ubuntu\.com' "$source"; then
      echo "Refusing to move mixed Ubuntu/LLVM apt source: $source" >&2
      exit 1
    fi
    mv "$source" "$RUNTIME_ROOT/apt-sources-disabled/$(basename "$source")"
    echo "Disabled unused LLVM apt source: $(basename "$source")"
  fi
done

python3 "$BENCHMARK/cloud/fetch_references.py" --root "$BENCHMARK"
# Codex universal prepends mise's Node 20. The node bootstrap installs
# Node 22 in /usr/bin; keep that binary first for its version checks.
export PATH="/usr/bin:/bin:$PATH"
SOURCE_BUNDLE="$RUNTIME_ROOT/no-source-bundle" \
REPO_ROOT="$REPO_ROOT" \
REFERENCES_ARCHIVE="$RUNTIME_ROOT/no-reference-archive" \
  bash "$BENCHMARK/setup_benchmark_node.sh" "$RUNTIME_ROOT"
"$RUNTIME_ROOT/renderer-env/bin/python" -m pip install \
  --disable-pip-version-check --no-input 'huggingface_hub>=1.0,<2'
"$RUNTIME_ROOT/renderer-env/bin/python" "$BENCHMARK/run.py" \
  --root "$BENCHMARK" --dry-run --track screen --limit-episodes 1
