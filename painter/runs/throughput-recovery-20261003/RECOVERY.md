# Applying the recovery changes

These are preparation tools, not an automatically launched experiment. Keep the registered three-judge campaign closed. The next measurement should use a single arm, the existing RL16 initializer and frozen SFT512 opponents, and the user's $20 GPU-rental cap. API expense is separate. No scheduled task is needed.

## Renderer

On Linux, build a distinct cancellation-safe runtime:

```bash
python3 painter/patch_brush_centroid.py \
  --renderer-overlay \
  --output-root painter/runtime/watercolour-cancellable-centroid-v1
```

Select `painter/runtime/watercolour-cancellable-centroid-v1/cancellable_watercolour_renderer.py` through the taskset/config writer's explicit `renderer` argument, using an absolute path on the node. Keep the existing renderer Python and browser installation. This directory contains an adjacent frozen renderer and the same centroid-fixed brush asset; its manifest records both versions. It does not replace the historical `renderer.py` runtime or change its hash guard. Use `validate_renderer_overlay` before reusing a materialized derivative.

Cancellation retains the semaphore slot while the synchronous worker is drained. Worker/browser cleanup uses only owned process/session identities. Normal termination escalates TERM to KILL; if an OS cannot reap its owned child after KILL, cleanup fails closed by retaining the slot rather than pretending the worker stopped. This exceptional final reap is not a guaranteed bounded wall-clock operation. Deployment still needs an outer finite process/rental deadline.

Turn traces now distinguish `render_queue_seconds` from `render_active_seconds`. Active time includes the thread dispatch and subprocess path; the renderer's own `elapsed_seconds` remains separately available. These fields make queueing visible without treating it as painting compute.

## Native configuration

`painter.throughput_recovery_profile.prepare_profile` wraps the existing text config writer and binds batch16, group4, maximum policy age1, 32 in-flight episodes, CPU activation offload, the immutable RL16 initializer, the unchanged consistent pairwise reward protocol, and checkpoint interval1. Hardware layout remains an explicit input to the existing writer; this profile does not require eight GPUs or infer a speedup from GPU count.

Required environment:

```bash
export PAINTER_RL_CHECKPOINT_INTERVAL=1
export ADAPTER_ONLY_CHECKPOINTS=1
export PAINTER_ACTIVATION_CPU_OFFLOAD=1
```

These bindings must be applied to the trainer process using the existing adapter export hook and existing public-HF upload/verification configuration. `validate_runtime_environment` checks the local bindings; it cannot prove a remote process received them. The generated receipt records the required checkpoint mode rather than claiming successful runtime export. There is no optimizer-state resume. Interval-one upload overhead must be measured, and an update stopped during publication is not yet a verified public checkpoint.

The historical writer still defaults to interval4 when no explicit interval/environment override is supplied. Historical protocols are not rewritten.

## Drawing API reference

`PAINTER_P5_BRUSH_API_REFERENCE_V2=1` explicitly selects the new default-off reference. It specifies scalar circle/rect/line arguments, spline pressure coordinates, named brush/color initialization, a complete WEBGL lifecycle, and avoidance of duplicate lifecycle functions or helpers shadowing p5 globals. The trace records the selected version and SHA. API-v1 bytes are unchanged.

Treat v2 as a separately labeled instruction change. Static signature/template tests are not evidence that the policy generates more valid paintings. Keep strict final-action scoring unchanged during infrastructure measurement; a reward on the retained last valid canvas is a separate research choice.

## Offline audit

```bash
python3 -m painter.pipeline_audit \
  painter/runs/rl-judge-ablation-20261002/private/offline-analysis \
  --output /tmp/painter-pipeline-audit.json
```

The command reads metadata and reports actual optimizer updates, admissions, canceled/stale work, render validity, completed-trace reward paths, judge usage, and GPU sample windows. It emits aggregate fields and relative evidence hashes, excluding program/response payloads. The extracted metadata directory is ignored by Git; retrieve raw evidence from the immutable public archives linked in `ANALYSIS.md`.

Next report: completed valid episodes, published updates per dollar, queue/active render timing, cancellation fraction, and a matched held-out visual panel. Offline checks do not establish throughput or visual improvement. Do not launch a full three-arm campaign merely because unit tests pass.

## Recovery bootstrap after the zero-output failure

The recovery-only wrapper places dependency caches, the new Prime virtualenv and pinned browser tree on local temporary storage. It preserves existing caches and refuses to move a live real virtualenv. Descriptor and installer concurrency settings adapt to the host hard limit and are applied by the execution parent. Historical shared setup defaults remain unchanged unless the recovery override is supplied.

The renderer helper binds the Node shipped by locked Playwright into the worker PATH and runs syntax checking. It installs hash-matching, root-owned Chromium sandbox helpers beside both matching full-browser and headless executables, rejecting symlink or untrusted destinations before mutation. Browser startup is tested as the unprivileged painter account with sandboxing enabled and a credential-free environment before GPU policy initialization; reused preparation also rechecks the runtime.

Parent metadata/lifecycle verification: 22 single-arm tests and eight profile tests passed. These Mac fixtures do not render, train, call a provider, or prove target-host Chrome compatibility. The Linux target must pass the executable checks before policy launch. [Review and remaining proof](bootstrap-review.json). The failure and source history remain in [attempt five](allocation-attempt5-result.json); the same pilot retains a $20 GPU ceiling, with a conservative $6 prior-start reserve and $14 for any retry.

## Durable remote execution after SSH loss

The single-arm controller now starts the finite pipeline in a detached session and reads an atomic completion receipt. A missing SSH exit status is transport uncertainty, not the pipeline exit. The controller reconnects without launching a second job; a hard deadline sends a stop within the release reserve and retains exact-node cleanup. Deliberate interruption still reaches cleanup. The new allocator checks both complete Git bundles before allocation, waits for RUNNING plus SSH readiness in bounded intervals, verifies Prime bytes after transfer, and limits this registered pilot to two H100 GPUs before comparing prices. Broader hardware selection remains a separate extension.

Parent verification: 28 lifecycle tests and 10 allocator tests, including real dummy-process execution of the generated wrapper on the Mac; no rendering or ML occurred locally. Linux bootstrap and actual learning remain pending. The same logical pilot has a $20 total GPU ceiling, with a $6.50 reserve for failed starts and at most $13.50 for another rental. [Review](disconnect-review.json).
