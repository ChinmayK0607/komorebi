# Run the painting teacher benchmark

**Prepared, not started. No paid model calls have been made.**

Everything is stored on this Mac at:

```text
/Users/chinmay/Documents/Codex/2026-09-07/if-they-did-speed-painting-type/painter/benchmarks/openrouter-teachers-20260922
```

There are 40 bundled, visually checked reference images, five default models and two tracks: **400 episodes total**. Every painting turn is rendered on Linux, and its actual canvas is returned to the teacher for inspection, revision or finishing. Inspect the selected photos in [reference-gallery.html](reference-gallery.html).

## Launch after adding credits

From Terminal:

```bash
cd /Users/chinmay/Documents/Codex/2026-09-07/if-they-did-speed-painting-type/painter/benchmarks/openrouter-teachers-20260922
bash run_benchmark.sh --pod painter-photo-rl
```

The launcher asks for the OpenRouter API key with input hidden. Alternatively, it accepts `OPENROUTER_API_KEY` from the environment. It does not save the supplied key on this Mac or include it in artifacts. On the node, its temporary credential file is private and removed when the wrapper exits normally or through its cleanup handler. If the connection is lost during cleanup, follow the printed cleanup warning.

This uses an **existing** Lium node and its installed painter renderer; it never rents or stops GPUs. The current default node name is `painter-photo-rl`. If that pod has been released, select another prepared Linux node with `--pod NAME`. The Mac needs Python 3.10+, SSH, and the authenticated Lium CLI used by this workspace. The Linux node needs the project's pinned renderer, Playwright browser installation and unprivileged `painter` user. The wrapper creates `/root/painter/benchmarks/openrouter-teachers-20260922/.benchmark-venv` with `tqdm` if needed, without mutating the shared training or renderer environments.

For a free, offline configuration/package check that neither contacts Lium nor reads an API key:

```bash
bash run_benchmark.sh --dry-run
```

## Choose models, track or keys

Use exact OpenRouter model IDs:

```bash
bash run_benchmark.sh --pod painter-photo-rl \
  --models z-ai/glm-5.3-flash xiaomi/mimo-v2.6-pro \
  --track quality
```

Default IDs:

- `z-ai/glm-5.3-flash`
- `deepseek/deepseek-v4.1-flash`
- `stepfun/step-3.7-flash`
- `xiaomi/mimo-v2.6-flash`
- `xiaomi/mimo-v2.6-pro`

You can also edit the model list in `config.json`. New IDs must support image input. The public catalog supplies capabilities and native limits; the runner records resolved metadata rather than silently substituting another model. StepFun 3.7 Flash is the current image-capable catalog entry used for the requested latest preview.

For separate keys per model, create a JSON file **outside this benchmark folder**, such as `~/.config/painter-benchmark-keys.json`:

```json
{
  "default": "YOUR_SHARED_OPENROUTER_KEY",
  "models": {
    "z-ai/glm-5.3-flash": "YOUR_OPTIONAL_GLM_KEY",
    "xiaomi/mimo-v2.6-pro": "YOUR_OPTIONAL_XIAOMI_KEY"
  }
}
```

```bash
chmod 600 ~/.config/painter-benchmark-keys.json
bash run_benchmark.sh --pod painter-photo-rl \
  --api-key-file ~/.config/painter-benchmark-keys.json
```

Keys here are OpenRouter account keys, not direct vendor endpoint credentials. Omitted model overrides use `default`. Do not paste real keys into `config.json`, the shell script or this README.

## Track settings

| Track | Turns | Output allowance | Episode allowance |
|---|---:|---|---|
| `quality` | Up to 12 | Native model ceiling, limited only by available context | No artificial episode time cap |
| `speed` | Up to 3 | 8,192 tokens per turn, including reasoning | 900 active seconds |
| `all` (default) | Both tracks | Separate results | 400 episodes |

Quality uses the highest advertised reasoning effort; speed uses the lowest. Models without an advertised effort selector have reasoning enabled without an invented effort setting. Each mode can stop early after inspecting a valid canvas. Both request faithful, attractive paintings; there is no synthetic additive aesthetic score.

Native output ceilings can be large. The full run has at most 3,000 assistant requests before retries, but actual duration and cost depend on how much the models generate and when they finish. API and rendering times, queue waits, token breakdowns, provider identifiers and reported costs are retained. Unknown costs are marked missing, not assumed to be zero. One sample per image/model/track does not estimate sampling variance.

## Progress, resume and artifacts

`tqdm` shows episode progress while generation runs. Persistent progress and event files supplement the terminal display. Rerun the same command to resume. Completed matching responses are reused, including when only rendering needs retry. Adding another model should preserve already completed results for the existing models. Changing the prompt, renderer or generation settings is a different experiment and must not silently reuse incompatible completed episodes.

The node retains the complete output tree, and the wrapper collects available artifacts under `results/` in this folder. Open `results/gallery.html` after collection; raw trajectories are in `results/episodes/`, with `results/progress.json` and `results/events.jsonl` for progress. Keep the terminal running for the foreground launch. If disconnected, outputs remain on the node for a subsequent resume/collection; do not release the node until desired evidence has been collected.

Stored evidence includes:

- Exact configuration, model metadata, prompt, reference hashes and source/runtime identities.
- Every assistant response and any reasoning returned by the provider, separate from proposed SFT targets.
- Conversation/image provenance for each request; generated programs; every valid canvas; renderer receipts and errors.
- Finish reasons, truncation, retry and deadline records; first and final valid canvases; explicit invalid-final flags.
- API/render/queue timing, token usage, known costs and missing-cost counts.
- A visual review gallery and blind review packet. **No visual winner is invented automatically.** Use [JUDGING.md](JUDGING.md) for pairwise quality review.

All demonstrations are candidates for later review. Executable code or a self-declared FINISHED response does not automatically enter training. See [PROTOCOL.md](PROTOCOL.md) for the comparison design and limitations.

## Validation before handoff

16 offline tests pass, including the real runner CLI with multiple model IDs. The Linux node passed the fixed-sketch sandbox render with matching source/canvas hashes and the isolated `tqdm` environment bootstrap. No OpenRouter generation request was made; paid API behavior remains to be exercised after funding. Details: [VALIDATION.md](VALIDATION.md).
