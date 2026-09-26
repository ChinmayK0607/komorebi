# Render the authored first-paint pool

## Prelaunch record, 2026-09-27

**Hypothesis.** Rendering every authored first-paint program, then inspecting its exact prompt/reference and canvas, will expose a broad set of correction states and some direct visual positives. Valid rendering alone is not a quality admission. The matched operational baseline is 82 published first-paint inputs already rendered in prior finite CPU waves; their visual screening admitted few direct positives. No student-quality improvement is claimed.

The frozen 28-photo evaluation set remains excluded from authoring. The [pinned render plan](REMAINING_FIRSTPAINT_RENDER_PLAN.json) covers all **498 as-yet-unrendered public first-paint candidates** in 24 source bundles, partitioned into six finite Cloud CPU groups of 80–88 paintings. Each row pins its public source archive SHA-256 and dataset revision. The renderer driver checks that archive hash before painting. Each group sets up the renderer once, uses two workers by default with a 600-second per-program timeout, publishes each completed bundle separately to the public Hugging Face dataset, and checks its anonymous download hash. A rerun skips only a publicly verified result receipt. Raw programs/canvases remain outside Git.

The authored count is 592, comprising 580 public first paints and 12 reused COCO128 photo first paints in `astra-reference-seed`. Those 12 were withheld from public source publication because per-image source/rights metadata is incomplete. They are **outside this cloud plan**, not silently counted as rendered. A separate eight-input Astra-high text batch is being authored to reach 600 distinct raw tasks; it will be packaged, checked for duplicates and contract/syntax, published, and rendered in its own finite job. This plan does not include the 64 static alternatives or 25 render-conditioned correction candidates; those are separate programs on existing inputs.

For each render group, record the Cloud task ID, actual source Git commit, per-bundle run summaries, renderer failures/timeouts, wall time, worker count, and anonymous publication receipt. Then collect and hash-verify the public results and build prompt/reference/canvas galleries. CPU busy fraction and billed Cloud cost are not instrumented; worker count is not utilization. Do not use GPU or a scheduled watcher for this phase.

**Decision after rendering.** Visually triage all valid first paints and select diverse exact-canvas corrections. Do not add a program to SFT merely because it rendered. Preserve failed and censored cases for debugging.
