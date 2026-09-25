#!/usr/bin/env bash
# Replace the provisional reviewed JSONL and expected mix after cloud review.
# Requires the pinned node bootstrap, but never starts training.
set -Eeuo pipefail
RUN="$(cd "${1:?usage: finalize_multiturn_data.sh RUN_DIRECTORY}" && pwd -P)"
[[ "$(uname -s)" == Linux && -f "$RUN/training-ready.json" && -f "$RUN/env.sh" ]] || {
  echo 'complete Linux node setup before finalizing data' >&2; exit 2;
}
exec 9>"$RUN/campaign.lock"
flock -n 9 || { echo 'campaign is running; data cannot change' >&2; exit 2; }
[[ ! -e "$RUN/training.exit" && ! -e "$RUN/campaign.exit" ]] || {
  echo 'data replacement is only allowed before the first training launch' >&2; exit 2;
}
read -r REPEATS SEED CHECKPOINT_MULTIPLE < <(python3 - "$RUN/expected-mix-manifest.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
print(x['reviewed_repeats'],x['seed'],x['checkpoint_multiple'])
PY
)
python3 "$RUN/build_multiturn_mix.py" \
  --reviewed "$RUN/data-input/reviewed.jsonl" \
  --foundation "$RUN/data-input/foundation.jsonl" \
  --foundation-validation "$RUN/data-input/foundation-validation.jsonl" \
  --output "$RUN/data" --repeats "$REPEATS" --seed "$SEED" \
  --checkpoint-multiple "$CHECKPOINT_MULTIPLE" > "$RUN/data-rebuild.json"
python3 - "$RUN" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]);actual=json.loads((root/'data/mix-manifest.json').read_text())
expected=json.loads((root/'expected-mix-manifest.json').read_text())
if actual != expected:raise ValueError('rebuilt final mix differs from expected manifest')
PY
source "$RUN/env.sh"
export BRUSH_ROOT PRIME_ROOT HF_HOME HF_TOKEN_PATH
export BRUSH_MODEL_PROFILE=qwen38_27b CAPACITY_STATE_COMPLETE=1
STEPS="$(python3 - "$RUN/data/mix-manifest.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['optimizer_steps_at_batch_4'])
PY
)"
"$RUN/bootstrap/bin/python" "$RUN/make_sft_config.py" \
  --data "$RUN/data" --model-path "$(cat "$RUN/model-path.txt")" \
  --output "$RUN/sft.toml" --output-dir "$RUN/train-output" \
  --run-name multiturn-sft-20260925-v1 --steps "$STEPS" --seq-len 16384 \
  --gpus-per-node 1 --num-train-gpus 1 --dp-replicate 1 --dp-shard 1 \
  --global-batch-size 4 --micro-batch-size 1 --learning-rate 0.000003 \
  --checkpoint-interval "$CHECKPOINT_MULTIPLE" \
  --validation-interval "$CHECKPOINT_MULTIPLE" --seed 20260925
"$PRIME_ROOT/.venv/bin/python" "$RUN/audit_dataset.py" \
  --config "$RUN/sft.toml" --output "$RUN/processor-audit.json" \
  --brush-root "$BRUSH_ROOT" > "$RUN/processor-audit.log" 2>&1
"$PRIME_ROOT/.venv/bin/python" - "$RUN" <<'PY'
import hashlib,json,pathlib,sys,tomllib
root=pathlib.Path(sys.argv[1]);mix=json.loads((root/'data/mix-manifest.json').read_text())
audit=json.loads((root/'processor-audit.json').read_text());cfg=tomllib.loads((root/'sft.toml').read_text())
if audit['status']!='passed' or audit['max_tokens']>cfg['data']['seq_len']:
    raise ValueError('final processor length/mask audit failed')
if cfg['max_steps']!=mix['optimizer_steps_at_batch_4'] or cfg['ckpt']['interval']!=mix['checkpoint_multiple']:
    raise ValueError('final configuration differs from data schedule')
record={'schema':'painter.multiturn-final-data.v1','status':'ready_not_launched',
        'reviewed_sha256':mix['input_sha256']['reviewed'],
        'train_sha256':mix['output_sha256']['train'],
        'optimizer_steps':cfg['max_steps'],'max_processor_tokens':audit['max_tokens'],
        'seq_len':cfg['data']['seq_len']}
(root/'final-data-ready.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
PY
