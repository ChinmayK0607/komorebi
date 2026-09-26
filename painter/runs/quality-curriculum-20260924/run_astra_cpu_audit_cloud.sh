#!/usr/bin/env bash
# Finite Linux CPU preflight. It never loads weights or starts training.
set -Eeuo pipefail
[[ "$(uname -s)" == Linux ]] || { echo 'Linux is required' >&2; exit 2; }
ROOT="$(cd "$(dirname "$0")/../../.." && pwd -P)"
ENV="${TMPDIR:-/tmp}/astra-qwen-cpu-audit-venv"
python3 -m venv "$ENV"
"$ENV/bin/python" -m pip -q install 'transformers==5.6.2' 'huggingface_hub==1.30.0' 'Pillow==11.3.0'
export HF_HOME="${TMPDIR:-/tmp}/astra-qwen-cpu-audit-hf"
export HF_HUB_DISABLE_TELEMETRY=1
export ASTRA_CPU_AUDIT_OUTPUT="$ROOT/astra-cpu-audit.json"
"$ENV/bin/python" "$ROOT/painter/runs/quality-curriculum-20260924/audit_astra_cpu.py"
