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

The 12 `astra-reference-seed` programs use photos that had already been published in earlier Astra-high source bundles, but this campaign's manifest lacked per-image rights metadata. `prepare_coco128_overlay.py` selects an earlier public bundle for each exact photo, anonymously verifies that bundle and the JPEG's SHA-256 against the current input hash, and packages **only the 12 new programs and hash-bound pointers**. The first local preparation passed for all 12. `render_coco128_overlay_cloud.py` downloads the older photo bytes into the temporary Linux render workspace, renders the current program, then omits the input JPEGs from the newly published result archive. The collector reconstructs the photo only in the ignored local review directory and verifies its row hash. Thus the current campaign does not republish these photos or make an unsupported per-image license assertion. The program-only archive is a distinct source format and is not counted among the 53 full teacher source bundles in `PUBLIC_SOURCE_INDEX.json`.

The program-only archive is 13,766 bytes, SHA-256 `d521ef151de14c3410dddf9f7e7d4844fc36c75e2de853173cf59a12a961e717`, public at dataset revision `fbb431b44ab816fd94f032e1fb35ce378cbec272`. An anonymous staging test retrieved the archive, the twelve earlier public sources, and all twelve exact programs/photos with row-hash matches. The separate [finite overlay render task](https://chatgpt.com/codex/cloud/tasks/task_e_6ab818d0f5b4832ba25ba2d2e00d3922) uses Git source commit `f3e74bacd69c19355cad6d8e5cd027e3459e8679`; render results are pending.

## Eight new text paintings: first result

The [eight-text CPU task](https://chatgpt.com/codex/cloud/tasks/task_e_6ab815b63d28832bb050ac93733b2fef) completed from Git source `9bd821ff933ac3c0792bd6bb10bd9adce08f4d55`: **8/8 renderer-valid** in 273.531 seconds wall time with two workers. Its [hash-verified local gallery](../../collected/quality-curriculum-20260926/rendered-cloud/teacher600-astra-text-eight-20260927/review.html) presents the exact prompts and canvases. Parent inspected all eight. Persimmons, citrus juicer, lapwing and abacus preserve the requested primary forms; the horseshoe crab is too schematic, and the garden structures and lamp remain mechanically simple. Soft color is pleasant, but edges, volume, lighting and compositional detail fall below the direct high-quality SFT bar. **Zero of eight are directly admitted for SFT**; they remain valid correction states. This is an aesthetic judgment on eight identified prompts, not an estimate of the rest of the 600-task pool.

The twelve-photo overlay completed **12/12 renderer-valid** in 1,097.601 seconds wall time with two workers. Its public result archive has 49 members and **zero JPEGs**; the local collector reconstructed and hash-verified all twelve inputs for the [review gallery](../../collected/quality-curriculum-20260926/rendered-cloud/teacher600-astra-coco128-overlay-20260927/review.html). Spot checks show a recognizable cake slice but weak giraffe anatomy and other schematic forms; no direct SFT admission follows from this render result.

## Runtime repair wave: prelaunch

The first collected batches expose seven immediate Sol text-program failures: six use unsupported `curveVertex`, and one redefines the p5 `box` global. The [frozen failure selection](RUNTIME_REPAIR_V1.json) identifies exact prompt/program hashes and the observed renderer error for each. `prepare_runtime_repair.py` makes seven separate alternative programs: `curveVertex` becomes `vertex`, and the conflicting helper becomes `paintBox`. These are **alternative programs on existing input tasks**, not seven new distinct inputs. The originals remain in their immutable public render receipts. The hypothesis is renderer recovery on the same prompts; the matched baseline is each original failed program. Any recovered canvas still needs visual review and is not automatically better or SFT-worthy. The selection excludes 600-second render timeouts, which need separate performance diagnosis.

The 7,807-byte repair archive (SHA-256 `66e71b9c2beb3377d703d5a070c444e511a8d69a832432ec3e9c4e52da55ba76`) was published and anonymously verified at dataset revision `7228cf303cf262fc732f3d94839911e8b9b5cae2`. Its [finite Cloud CPU render task](https://chatgpt.com/codex/cloud/tasks/task_e_6ab81faf73f0832b946e49d0f037c41d) used Git source `7f71627fc97cd7e8a289520a91ff885a3f5e5bda`. The public index counts these as seven alternatives, leaving 600 distinct raw inputs unchanged.

## Runtime repair result

The seven alternative programs all rendered valid canvases in 23.675 seconds wall time with two CPU workers, versus seven invalid original programs. The [hash-verified repair gallery](../../collected/quality-curriculum-20260926/rendered-cloud/teacher600-sol-runtime-repair-20260927/review.html) preserves the exact prompts and canvases. Visual inspection of all seven found recognizable illustration subjects, including the violin, child with kite, and container crane, but also flat or awkward anatomy and schematic scenes. This establishes execution recovery only; none is admitted directly as a high-quality SFT positive. The original failed receipts remain in the first-paint sweep.
