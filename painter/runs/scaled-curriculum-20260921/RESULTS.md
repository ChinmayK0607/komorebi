# Scaled curriculum preparation — 2026-09-21

The data expansion is complete. This is preparation, not a new training or model-quality result.

- 2,640 final procedural references rendered successfully: 2,400 train / 240 development, twelve object recipes, five composition layouts and 574 ordered member-family combinations. All 2,640 image hashes and geometry hashes are distinct; there is no exact image-hash overlap across the split. This tests parameter/composition transfer, not unseen-family generalization.
- 128 real photographs were imported with exact source hashes: 100 train / 28 development, covering 71 annotation classes. These are reference inputs without canonical painting teachers; prior model exposure and per-image licensing are not established by this import.
- Only final references were bulk-rendered. The three source-program stages per procedural case are retained, but intermediate canvas rendering and SFT export were not performed.
- Parent and Luna inspected twelve procedural previews, finding readable structures with no systemic clipping, but limited texture, material variation and complex perspective. These are foundation targets, not aesthetic gold. The gallery includes twenty-four photos from the larger pool.

The first render attempt failed before drawing because the runner dereferenced the virtualenv interpreter to the system Python. The fix preserves the virtualenv path. Failed receipts remain in `corpus/scaled-data/attempt-1/`; the retry rendered all references from 16:19:29–17:02:19 UTC. The 155,836,336-byte archive and 15,888 listed extracted files were hash-verified locally before releasing the node. Archive SHA-256: `bcc805df66e2da42ec8217c2e9fd6a21e25599e1d3dcf6bb5bbb22f03c12eb35`.

The Linux preparation node was rented from approximately 16:02:50–17:07:14 UTC at $0.59/hour, an estimated rental below about $0.65, not an invoice. It ran CPU/SwiftShader rendering, with no model or trainer. GPU utilization was not measured over an interval; this is not a saturated ML pipeline. No node is intentionally retained for this preparation.

The native RL extension reuses the shared paint/inspect/revise interaction and one terminal aesthetic pairwise reward. Parent review corrected a family-prefix sampling bug; the first 512 references are now tested for family balance before source repetition. The evaluation helper accepts larger JSON/JSONL manifests. Forty-two focused pure tests passed; the new Prime runtime has not been launched.

The finite Luna-low CLI worker runs actual attached-image judgments with strict schema output, bounded concurrency, raw evidence and token usage. A hashed-filename batch exposed one candidate-role confusion. Neutral role filenames fixed that observed failure in six follow-up views over cylinder, cup and box. Cylinder/cup preferences were order-consistent; box preferences differed. This is limited input-binding evidence, not broad judge calibration. JPEG/PNG/WebP attachment naming preserves actual bytes. See `CLI_BATCH_REVIEW.json`; no test labels were used for learning.

The next learning allocation is prepared for 128 updates × 32 admitted episodes, eight rollouts per reference group, aiming for 512 distinct references. The 2,400-reference pool is larger than that first allocation. GPU role counts are configurable, while the example uses one trainer and one inference device. Remaining operations are frozen student-opponent generation, the live node queue connection, and native launch validation. No new optimizer updates have occurred and no quality improvement is claimed. Public hash-verified HF checkpoints remain required when training resumes. No scheduled task or heartbeat was created.

Evidence: `CORPUS_AUDIT.json`, `EXPERIMENT.json`, `CLI_BATCH_REVIEW.json`, `real-pool/manifest-index.json`, `preview/preview-review.json`, `review/index.html`, and the ignored `corpus/` archive and extracted files.
