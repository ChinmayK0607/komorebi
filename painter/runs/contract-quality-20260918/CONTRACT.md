# Stable painting foundation, version 1

Owner of the interface: `contract.py`. Snapshot: `foundation-lock.json`.
This is an engineering correction, not evidence of improved model quality. Historical run directories remain unchanged. The selected model remains recovery step72.

## Fixed across a learning comparison

| Component | Contract |
|---|---|
| Coordinates | 600 × 600 pixels, top-left origin after exactly one `translate(-300,-300)`. Teacher targets contain no extra object coordinate frames. |
| Runtime | Vendored p5 2.3.2 / p5.brush 2.2.1; renderer and asset hashes recorded. Node-only rendering and inference. |
| Actions | Emit a complete replacement sketch; receive its actual canvas; then revise or finish. A newly rendered program cannot finish in that same response. |
| Existing good canvas | Finish without redrawing. Repeating an already observed unchanged program does not trigger another render. |
| Failures | Keep the last valid canvas and show the execution error. A turn-limit exit remains incomplete, even when the last render was executable. |
| SFT | Same system prompt and action format as rollout. Paint targets have no FINISHED suffix; finish targets have an observed canvas and no code. |
| Provenance | Contract identity, source hashes, model/config/reference binding, render source and PNG hashes. Old output directories cannot silently resume a changed rollout. |
| Quality scoring | Reference-conditioned blinded pairwise judgment plus absolute acceptability. Missing/uncertain/order-inconsistent judgments are masked. Validity and training loss are diagnostics. |
| Speed | No speed reward during quality training. Log tokens, actions, retries and preview calls without rewarding shorter output. |

The existing Luna reviews remain evidence about their original runs. `pairwise.py` and `reward.py` are scoring components; a continuously callable Luna reward service and end-to-end Prime RL integration are still work to do. This snapshot does not claim those exist.

The teacher source check catches our known affine-format bug; it is not a complete JavaScript parser or a proof of visible quality. Student programs are not silently corrected into the desired answer. The model still has to learn geometry and reference fidelity.

## One check, performed automatically

`setup_node.sh` checks the foundation snapshot and runs the pure/stubbed regression tests before environment setup. A successful setup records that identity in `ready.json`. This is automatic, with no human approval stage. Both the portable bundle and evidence export carry the snapshot.

```bash
python3 foundation_lock.py
python3 -m unittest discover -s . -p 'test_*.py'
```

An intentional foundation edit requires tests, an explicit `python3 foundation_lock.py --snapshot`, and a new matched baseline. Experiment data, sampling, learning rate and training duration stay in the run recipe, not in this shared contract. Hardware selection may change parallelism and microbatch/accumulation while preserving the intended effective batch. CUDA is the tested learning backend; provider independence does not imply every accelerator is supported.

## Migrating old teacher geometry

On a CPU Linux node, `compile_teacher.cjs` can lower **trusted procedural recipes** to explicit canvas-coordinate brush calls. Do not run model-written programs through this compiler; its JS VM is not a security boundary. The browser renderer retains responsibility for isolated execution of model programs.

```bash
node compile_teacher.cjs legacy-programs.jsonl canonical-programs.jsonl
```

The compiler verifies identical ordered brush-call traces and preserves full numeric precision. It writes a new file and keeps the original. It has been syntax-checked locally; execution and browser-render parity remain to be tested on the node. Evaluate the compiler on a small fixture set before the full corpus. Preserve both renders and compare them; record target token lengths too. Flattened loops/ellipse vertices may make code much longer. This is a migration experiment, not an instruction to train on a giant literal program automatically. Existing compact direct-coordinate teachers already satisfy the coordinate convention.

## Next experiment

1. Use the recovery parent under this exact preview protocol. Use the existing evaluation references only for evaluation, with four matched samples per painting reference. Preserve original A/B evidence separately because the scaffold changed.
2. Prepare compact complete paintings and useful revisions on **training references**, spanning all current shape/object families. Include finish decisions on correct canvases and corrections of real student mistakes. Compile affine examples only if geometry parity and usable target lengths hold; other compatible data can proceed independently.
3. Run a short SFT pilot from recovery step72, with intermediate public HF checkpoints. Keep the reference split, prompt, renderer, evaluation sampling and judge fixed. Measure both average preference and the tails: each family, worst regressions, absolute acceptability, and whether revisions actually improve their starting canvases. Do not select by validation NLL.
4. Continue the best visually supported checkpoint. Expand real-reference diversity and use pairwise RL on useful competing samples; keep teacher SFT replay. Speed curricula come after dependable quality.

This separates a scaffold baseline from a learning comparison. A compiler failure affects that source subset; an unavailable RL judge endpoint does not stop SFT or reference collection. One bad sample stays in the report instead of blocking the whole campaign. Actual corrupt inputs, wrong masks and unverified checkpoint transfers are technical failures to fix automatically where possible.

No GPU is currently rented, and this foundation change has not launched training. Next node work is teacher conversion/render parity and a matched parent rollout, followed by the bounded learning pilot.
