# Manual painting v1: Linux CPU render and visual review

**Decision:** do not admit these ten v1 programs to teacher SFT. All ten are technically valid 600×600 canvases, but none passes the planned high-quality, reference-faithful bar. Retain them as inspected drafts and revise the painting method and geometry before attempting to scale manual examples.

The matched comparison is the exact source photo for each program in `reference-manifest.json`, not a model checkpoint. The finite Codex Cloud [render task](https://chatgpt.com/codex/tasks/task_e_6ab4af26bc48832b83b18e0cc69b7adf) ran source commit `43ce2ef8184e4534cccfd4bdf6bfb8f7b5f3f765`, program-manifest SHA-256 `c681f8d1b6ee5f017ba642f21eb1b4b778032cdc6b3c93236bbcc7abc4ac856b`, and renderer SHA-256 `ad440b1aa8612e293b52fa5e68c2252926722852155eb8cd145d369a862a8a63`. It made zero teacher/model calls and used no rented GPU. The 10 sequential render calls totaled 279.945 seconds; this excludes fresh cloud setup, upload and task overhead. GPU utilization and rental cost are inapplicable; Codex Cloud usage cost was not measured.

All ten render receipts are valid. The 41-file, 3,523,210-byte archive was anonymously retrieved and SHA-256 verified as `a4c83c4b23d12d0ce9525eb63cc8bc78fcbe810d462181e10ca9d4da4eac6db8` from public HF dataset commit `92cda91ac44112d4ba797b15f568da8e1ee4b78b`. The [receipt](https://huggingface.co/datasets/CK0607/komorebi-painter-teachers/resolve/main/runs/manual-quality-v1-20260924/receipt.json) and local [side-by-side gallery](../../collected/quality-curriculum-20260924/manual-cloud-render-v1/index.html) identify the raw evidence. The gallery builder independently checks each reference and canvas hash against the manifest/summary. The public archive contains paintings and programs, not the source photos; reference photos remain in the ignored local collection.

## Visual review against exact reference photos

This is a single, unblinded parent inspection, not a calibrated pairwise judge. "Readable" below means the subject can be identified; it is not SFT admission.

| COCO ID | Observation | Decision |
| --- | --- | --- |
| 092 cake | Slice, plate and fork read, but chocolate is washed out and layer texture/lighting weak. | Revise |
| 025 giraffes | Tall animal and tree read, but neck/leg transitions and reference pose are crude. | Revise |
| 034 zebra | Stripes are detached-looking bars; head, neck, legs and grazing pose lack credible anatomy. | Reject v1 geometry |
| 263 elephants | Two bodies merge into a low-contrast green-gray wash; trunks and ears barely resolve. | Reject v1 geometry/contrast |
| 394 dog/frisbee | Large white torso and purple disc read, but head and legs are ghostly and overlap incorrectly. | Reject v1 geometry/contrast |
| 400 pontoon dog | Boat and dog are identifiable, but cabin/pontoon structure is flat and too pale. | Revise |
| 471 bus | Strongest structural study; school bus reads at a glance, but the wrong three-quarter shape and generic windows miss the source bus. | Revise |
| 491 teddy | Toy reads, but translucent ellipses, no fur and simplified anatomy make it unlike the source plush. | Revise |
| 605 coffee | Cup/saucer and wood surface read, but latte art, pastry, glass and perspective are underdeveloped. | Revise |
| 650 cat/car | Cat face reads; body fades into background and roof geometry/pose are wrong. | Reject v1 geometry/contrast |

**Method issue:** the uniform semi-transparent ellipse/polygon recipe favors pale overlapping silhouettes and straight seams. Technical render validity did not predict beauty or reference fidelity. The next manual pass needs stronger subject/background separation and reference-specific contours, especially animal anatomy and occlusion. Rerender a small revised subset on Codex Cloud CPU and compare v1/v2 at equal source IDs before producing more teachers. Keep MiMo Flash/Pro for scalable demonstrations only after a visual admission bar is met; reserve Lium for training/inference requiring GPUs.
