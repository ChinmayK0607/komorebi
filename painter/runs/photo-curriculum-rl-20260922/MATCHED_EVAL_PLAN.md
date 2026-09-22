# Matched six-turn painting comparison

**Question.** Does the RL policy paint the same references better than its
frozen SFT initializer when both get the same opportunity to inspect and
revise? A training reward increase alone does not answer this question.

Compare the public `photo-curriculum-sft-20260922-balanced/step_512` adapter
(revision `adf687e5b321514eda7be6eab050434fdc8e44ff`) against a named,
public, hash-verified RL adapter from this run. Record that adapter's exact
commit and weight SHA before generation. The latest available RL adapter is
not automatically the best checkpoint; include intermediate snapshots if the
first comparison finds a regression.

Use two existing development splits, reported separately:

- All 28 photographs in `../photo-curriculum-sft-20260922/eval-prep/eval-manifest.json`
  (SHA-256 `9943878cc43d48f1cf9ff60e5af633089bf2bd8d45141216531363cad8f1c878`).
- All 240 `split=dev` procedural references in
  `../scaled-curriculum-20260921/corpus/scaled-data/manifest.jsonl`
  (SHA-256 `c828f5c3f5ae4004be976009790e252b8713c225c17520ef643fe7498173a218`).

These references were excluded from this RL training pool, but they have been
used in earlier development work. Call this a development comparison, not an
untouched test result. Reserve new references that have not influenced data,
checkpoint, or reward decisions for a later final generalization check.

Generate each policy's painting with the **same** reference bytes, six-turn
paint → render → inspect → revise contract, prompt, chat template, renderer,
sampling temperature, seed, 8,192 output-token ceiling per turn, 32,768-token
context, and timeouts. Run both on the Linux inference environment. Save every
turn's program, canvas, feedback, token usage, errors and terminal painting;
keep service failures distinct from invalid model programs. The prior SFT
28-photo evaluation was one-turn and cannot serve as the baseline here.

For each reference, randomize A/B identity and show Luna-low the reference plus
both terminal paintings without policy labels. Judge which **looks better as a
painting while preserving the reference's identity and structure**, and name
visible weaknesses of both. Use both A/B orders; disagreement or unclear
images are ties/uncertain cases, not forced wins. Judge every case if quota
allows and keep raw image hashes and judgments. Present a side-by-side gallery
for human review, including all failures and clear regressions.

Report paired RL wins/ties/losses with all references in the denominator,
valid-canvas counts, service exclusions, per-family outcomes, turn/token
distributions, and a paired uncertainty interval. Show photo and procedural
results separately. A checkpoint is a real improvement only if quality and
reference structure improve without a material rise in invalid outputs; a
single aggregate reward or a few favorable examples are insufficient.

**Execution timing.** Prepare manifests and comparison configuration while RL
continues. Do not load both evaluation policies on the active two-GPU training
node: its trainer is already near memory capacity and the inference GPU serves
rollouts. Run the finite evaluation after training on an inference allocation,
or on a separately costed node that does not disturb training. Use the existing
`painter/native_painting_eval.py` as the six-turn config generator; the old
`launch_matched_eval.sh` is explicitly one-turn and must not be reused without
changing that contract. Record runtime and cost before declaring the result.
