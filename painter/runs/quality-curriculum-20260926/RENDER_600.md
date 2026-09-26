# Render the authored first-paint pool

## Prelaunch record, 2026-09-27

**Hypothesis.** Rendering every authored first-paint program, then inspecting its exact prompt/reference and canvas, will expose a broad set of correction states and some direct visual positives. Valid rendering alone is not a quality admission. The matched operational baseline is 82 published first-paint inputs already rendered in prior finite CPU waves; their visual screening admitted few direct positives. No student-quality improvement is claimed.

The frozen 28-photo evaluation set remains excluded from authoring. The [pinned render plan](REMAINING_FIRSTPAINT_RENDER_PLAN.json) covers all **498 as-yet-unrendered public first-paint candidates** in 24 source bundles, partitioned into six finite Cloud CPU groups of 80–88 paintings. Each row pins its public source archive SHA-256 and dataset revision. The renderer driver checks that archive hash before painting. Each group sets up the renderer once, uses two workers by default with a 600-second per-program timeout, publishes each completed bundle separately to the public Hugging Face dataset, and checks its anonymous download hash. A rerun skips only a publicly verified result receipt. Raw programs/canvases remain outside Git.

The authored count is 592, comprising 580 public first paints and 12 reused COCO128 photo first paints in `astra-reference-seed`. Those 12 were withheld from public source publication because per-image source/rights metadata is incomplete. They are **outside this cloud plan**, not silently counted as rendered. A separate eight-input Astra-high text batch is being authored to reach 600 distinct raw tasks; it will be packaged, checked for duplicates and contract/syntax, published, and rendered in its own finite job. This plan does not include the 64 static alternatives or 25 render-conditioned correction candidates; those are separate programs on existing inputs.

For each render group, record the Cloud task ID, actual source Git commit, per-bundle run summaries, renderer failures/timeouts, wall time, worker count, and anonymous publication receipt. Then collect and hash-verify the public results and build prompt/reference/canvas galleries. CPU busy fraction and billed Cloud cost are not instrumented; worker count is not utilization. Do not use GPU or a scheduled watcher for this phase.

**Decision after rendering.** Visually triage all valid first paints and select diverse exact-canvas corrections. Do not add a program to SFT merely because it rendered. Preserve failed and censored cases for debugging.

## Launch record

The reviewed source commit is `9bd821ff933ac3c0792bd6bb10bd9adce08f4d55` on `codex/painter-multiturn-sft`. Six finite Codex Cloud CPU tasks were submitted; these are one-shot jobs, not scheduled monitors:

| Group | Count | Task |
| --- | ---: | --- |
| 1 | 84 | [Cloud task](https://chatgpt.com/codex/tasks/task_e_6ab814c1782c832bacac80b6ee82ddfd) |
| 2 | 84 | [Cloud task](https://chatgpt.com/codex/tasks/task_e_6ab814d3eda8832bbfc9ce87a07cf4fa) |
| 3 | 82 | [Cloud task](https://chatgpt.com/codex/tasks/task_e_6ab814d274d8832b8244b0ac04e60403) |
| 4 | 88 | [Cloud task](https://chatgpt.com/codex/tasks/task_e_6ab814d3765c832bb10cddc458f6acfd) |
| 5 | 80 | [Cloud task](https://chatgpt.com/codex/tasks/task_e_6ab814d3b890832b975e9f71aec9a64a) |
| 6 | 80 | [Cloud task](https://chatgpt.com/codex/tasks/task_e_6ab814d3fcf4832b8a1699fdb1433ff0) |

The distinct eight-text batch `astra-text-eight-v1` passed contract, Node syntax, finite geometry and duplicate prompt/subject checks. Its 10,406-byte program/source archive has SHA-256 `caaa093bc11eae390e04750f36632531f2e06692f8888a438a154f42270cd91b`; the public source is at HF dataset revision `5d970aeeabb6ab407ede50c321467c2cd1da5ad6` and was anonymously hash verified. Its separate [finite CPU render task](https://chatgpt.com/codex/tasks/task_e_6ab815b63d28832bb050ac93733b2fef) was submitted. This brings raw authoring to **600 distinct first-paint candidates: 290 text and 310 photo**, of which 588 are in public source bundles. Renderer and visual outcomes for the new tasks are pending.

## Reused-photo overlay for the remaining 12

The 12 `astra-reference-seed` programs use photos that had already been published in earlier Astra-high source bundles, but this campaign's manifest lacked per-image rights metadata. `prepare_coco128_overlay.py` selects an earlier public bundle for each exact photo, anonymously verifies that bundle and the JPEG's SHA-256 against the current input hash, and packages **only the 12 new programs and hash-bound pointers**. The first local preparation passed for all 12. `render_coco128_overlay_cloud.py` downloads the older photo bytes into the temporary Linux render workspace, renders the current program, then omits the input JPEGs from the newly published result archive. The collector reconstructs the photo only in the ignored local review directory and verifies its row hash. Thus the current campaign does not republish these photos or make an unsupported per-image license assertion. The program-only archive is a distinct source format and is not counted among the 52 full teacher source bundles in `PUBLIC_SOURCE_INDEX.json`.
