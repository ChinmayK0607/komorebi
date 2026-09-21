# Reference painter: [current experiments and evidence](../RESEARCH_LOG.md)

**Current status, 2026-09-21:** the [controlled comparison](runs/controlled-comparison-20260921/RESULTS.md) is complete and its GPU is released. The recent RL checkpoint did not beat the step-24 SFT initializer overall. The next work is the [scaled quality curriculum](runs/scaled-curriculum-20260921/PLAN.md): expand actual training coverage, reuse the corrected native environment, and address judge throughput. See [NEXT-RUN.md](NEXT-RUN.md) and the [research index](../RESEARCH_LOG.md) for current execution status. Older sections below describe earlier stages, not active launch instructions.

The [rollout audit](runs/diversity-sft-20260918/FORENSICS.md) found that none of 216 drawing/repair episodes inspected a successful render before finishing. Painting targets append FINISHED immediately, so extra turns only repair execution errors. The next correction is to teach separate paint, preview, and revise/finish decisions and isolate coordinate-representation errors. More of the same SFT is not supported by this comparison.

The shared interface is now defined in [CONTRACT.md](CONTRACT.md) and `contract.py`. The current rollout requires observing a new render before finishing; the current SFT builders emit separate paint and finish targets. `foundation-lock.json` fingerprints this interface and the pinned renderer, and node setup checks it automatically. These changes passed local stubbed regression tests; model behavior and teacher render parity still require node evaluation. The preceding findings describe the preserved historical run, whose code and evidence have not been rewritten.

The working sequence is teacher demonstrations → supervised practice → pairwise RL on the current difficulty → harder references. Speed training begins after the quality policy is consistently good. We keep the 27B student and Prime RL as the intended training backend.

| Difficulty | Practice | Data |
|---|---|---|
| 0 | Marks, curves, washes, volumes, light and shadow | Known-program synthetic references |
| 1 | Overlap, connected parts, simple compositions | Procedural objects plus teacher revisions |
| 2 | Recognizable individual objects | Real photographs with visually inspected teacher demonstrations |
| 3 | Perspective, anatomy, occlusion, scenes | Diverse real references and correction trajectories |
| 4 | Complete attractive paintings across subjects | Distinct references from all six families |
| Speed | Preserve quality while using less work | Quality-policy trajectories with progressively tighter budgets |

Quality mode has no target brush count, no token-efficiency target and zero speed reward. Infrastructure still bounds runaway episodes. An episode stopped by a technical limit is explicitly marked censored; it is not evidence of mastery. The initial operational defaults are 16,384 tokens per reply, eight turns and 30 minutes per episode; these are adjustable, not artistic targets. The interface is editable p5.brush JavaScript with functions, loops, layered washes and access to the actual rendered canvas.

## Data and sampling

`demonstrations.py` authors six procedural families with three genuine construction programs each. `render_demos.py` runs those programs on a node and writes reference/current-canvas/program SFT examples. These are curriculum seeds, not claims of realistic painting or semantic generalisation. Development holds out parameter bands. Real-photo and template-family holdouts remain necessary for later levels; procedural data must not become the whole training distribution.

`campaign.json` records initial SFT token-mixture targets (70% current level, 20% prior levels, 10% general vision/text), eight rollouts per reference, and difficulty sampling (60% current, 25% review, 15% next). Those are starting settings to be measured, not claims of optimality. Missing replay/retention pools must be disclosed in a materialized training manifest, never silently claimed as mixed. SFT should supervise complete assistant plans/programs and stopping, masking image/tool observations. The Prime complete-target preprocessing and 96-update SFT path have now been exercised. Recovery training balances repair/paint/finish/replay decisions by example count because FINISHED targets are much shorter than painting programs.

Continued supervised adaptation here means complete reference-conditioned plan/program/revision examples, not a separate raw-text pretraining phase. Alternate small SFT and RL rounds; preserve earlier successful behavior in replay.

The sampler balances available subject families and favors references with a mixture of success and failure. Pass@k distinguishes rare successful samples from consistent first-attempt performance. Easy successes stay in review; all-failure references need a better demonstration or intermediate difficulty. Mastery requires absolute acceptability on distinct development references, sustained across evaluations. Relative victories over other poor outputs do not count as mastery.

## Pairwise improvement

The user requested **Luna (`gpt-5.6-luna`) for visual judging**. Carry that preference into subsequent rounds: use blinded Luna reviews of actual reference/candidate images, record their scope and model identity, and preserve the raw labels. The recovery run also has a separate automated Qwen3-VL-32B scoring pass; those scores are not Luna judgments. A Luna agent review does not by itself establish a callable judge endpoint for online RL. Keep that distinction explicit when integrating training rewards.

`compare.py` sends **reference + candidate + same-reference anchor**, in both candidate orders, to a configurable vision endpoint. It preserves raw responses, accepts long explanations, handles individual failures independently, and resumes using image/prompt/model/task bindings. All rollouts in a reference group use the same anchor (or identical anchor set); otherwise reward differences confound painting quality with opponent difficulty. Anchor pools are frozen within an optimization round and refreshed between rounds. Compare against both the previous student and strong teachers so the task is neither trivially easy nor always impossible.

`pairwise.py` maps stable wins/ties/losses to 1/.5/0. Order disagreement and missing judgments are masked, not rewarded as ties. Groups without enough judgments or reward variation are excluded from learning while their artifacts remain. These are tested reward components; the adapter feeding rollout masks and advantages into a new Prime RL run still needs end-to-end integration. This repository does not claim RL is running.

Use a small continuing human spot-check of the actual candidate distribution. Track reference fidelity and craft separately. Introduce speed budgets only after sustained quality on the final diverse-painting level: initially 90%, 75%, 60%, then 45% of measured acceptable-policy cost, retaining unrestricted quality replay. Count planning, failed proposals, generated tokens, previews and actual brush calls. Wall-clock speed must be compared on matched hardware.

## Repeatable node workflow

Run local preparation and tests:

```bash
python3 -m unittest discover -s painter -p 'test_*.py'
python3 painter/demonstrations.py --output painter/data --variants 8
```

Copy the `painter` source/data bundle to a rented Linux node. On the node:

```bash
bash setup_node.sh /root/painter render
bash launch_demos.sh /root/painter
```

Setup is idempotent and never starts training. It pins Playwright/Pillow/browser dependencies, creates an unprivileged renderer account and records a hardware plan. `learning` also installs the pinned vLLM inference runtime. Rendering and model calls are node-only. `build_gallery.py` only displays already-rendered files and may run locally after collection.

`hardware.py` selects initial parallelism and dtype from capability/memory, not GPU marketing names. It preserves effective batch across GPU counts. The execution interface is SSH/provider-independent; no Lium SDK is embedded in painting/training logic. Native model serving is presently prepared for Linux CUDA. Rendering does not require CUDA. AMD and other accelerator runtimes are not claimed tested.

`rollout.py` uses the same image/program/canvas loop for a teacher endpoint and the student. Every response and render is persisted before the next turn. Completed cases can be resumed without paying for the same model calls again. Example:

```bash
runtime/bin/python rollout.py --reference /path/reference.png --output /path/episode \
  --endpoint http://127.0.0.1:8000/v1 --model base \
  --renderer vendor/integrations/watercolour/renderer.py --renderer-python renderer-env/bin/python
```

Portable adapters remain public on Hugging Face under versioned run paths, using the existing verified export/upload implementation. Native optimizer-state resume requires separate checkpoint storage; the prior adapter-only run does not provide exact optimizer continuation. Do not claim otherwise. Per-case data/render/model-call resume is implemented here; distributed training resume is not newly validated.

Routine PNG differences are diagnostics. Wrong inputs, unusable outputs, nonfinite training or execution failures need correction, but a failed case does not discard its neighbors. Generate galleries before asking for more labels. Keep a node through recoverable fixes and scheduled evaluation; release it after evidence collection. Historical correction checkpoints and their unfinished comparison remain a separate diagnostic, not evidence that this new curriculum succeeded.

## Executed bootstrap — 2026-09-17

The first 48 synthetic references (six procedural families, three stages each) produced 144 valid renders on a rented node, 108 training turns and 36 development turns with disjoint reference groups. Representative visual inspection fixed a dropped cube face at a wash-to-pencil transition, wedge-shaped petals, and a filled-in cup handle. A complete replay reused all 144 renders without redrawing them. Twelve unit tests pass.

Evidence collection verified 963 archived files by size and SHA-256. The gallery is `collected/evidence/data/index.html`; `visual-review.json` states the limited review scope. The node was released after collection (last reported H100 cost $0.59; the earlier unreachable spot attempt is separate). At that bootstrap milestone, no new student SFT or RL had run. `NEXT-RUN.md` now specifies the next baseline/SFT comparison and outstanding Prime integration. Input manifest status is historical; `rendered-manifest.jsonl` and `render-summary.json` contain execution results.
