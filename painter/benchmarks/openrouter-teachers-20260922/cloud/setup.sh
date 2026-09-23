#!/usr/bin/env bash
# Codex cloud environment setup command. No paid model calls or credentials.
set -Eeuo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
BENCHMARK="$REPO_ROOT/painter/benchmarks/openrouter-teachers-20260922"
RUNTIME_ROOT="$REPO_ROOT/.painter-cloud-runtime"

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
