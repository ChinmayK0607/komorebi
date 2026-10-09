# Structured-flow launch handoff

Prepare the curriculum metadata before renting. The controller-side steps use only the local Python standard library and do not contact Git, a provider, a model, a renderer, or a judge. Real frozen-opponent anchors are generated on the Linux node after its SFT512 endpoint is loaded; they are not a local prerequisite.

```bash
RUN="$PWD/painter/runs/structured-flow-rl-20261009"
python3 "$RUN/prepare_local.py" plan --output "$RUN/local-plan"
python3 "$RUN/prepare_local.py" validate --plan "$RUN/local-plan/launch-plan.json"
```

`prepare_node.sh` builds the 240/60 prompt proposals when they are not already staged, then creates a lazy 32/16 pilot manifest. After the frozen SFT512 endpoint is available, `anchor_bank.py materialize` runs one bounded native batch and writes real PNGs, receipts, and provenance before replacing the lazy rows. It fails when an anchor is missing or its bytes do not match the manifest; it never creates placeholders.

Before bootstrap downloads the model or installs the Prime environment, the run-owned renderer helper records the node's user-namespace/setuid capability. After the renderer environment is staged, its sandbox smoke writes `browser-smoke-diagnostic.json` with both the first and last Chromium diagnostics; inspect that file and `browser-capability-preflight.json` after a failed setup. The launch keeps Chromium sandboxing enabled. Before Qwen serving, an actual `native_painting_renderer.render_action` probe runs in a nested attempt root through Prime Python with an explicit verified browser directory. Its source, PNG and renderer receipts are preserved under `runtime/native-render-probe/`. Browser startup alone is not a successful render.

On the Linux node, set `PAINTER_STRUCTURED_FLOW_POLICY=1` only when the reviewed casebrief hint should be active. Set `PAINTER_CODE_SUPERVISION_ENABLED=1` only after the separate teacher JSONL passes `validate_code_supervision.py`. Run preparation first:

```bash
export PAINTER_SOURCE_COMMIT='REVIEWED_FULL_SOURCE_COMMIT'
export PAINTER_STRUCTURED_FLOW_POLICY=1
bash "$RUN/prepare_node.sh"
```

The parent rental controller supplies the exact pod, start epoch, listed whole-node price, and source commit. It must keep the allocator in the foreground and release the exact pod in its outer cleanup path. The node command is:

```bash
PRICE_USD_PER_HOUR='REPLACE_WITH_LISTED_PRICE'
RENTAL_START_EPOCH='REPLACE_WITH_PARENT_RENTAL_START_EPOCH'
GPU_ALLOWANCE_USD=18 # current retry: $2 reserved for earlier starts
bash "$RUN/execute_single.sh" "$RENTAL_START_EPOCH" "$PRICE_USD_PER_HOUR" "$GPU_ALLOWANCE_USD"
```

This includes setup, the 20-update training bound, matched pre/post evaluation work, archive/upload, and release reserve in the current `$18` allowance; `$2` remains reserved for prior starts under the original `$20` total GPU cap. No recurring watcher or scheduled task is part of this run.
