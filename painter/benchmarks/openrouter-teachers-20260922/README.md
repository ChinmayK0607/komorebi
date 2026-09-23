# Run the AI Gateway painting teacher benchmark

For the GitHub branch and Codex cloud path, see [CODEX_CLOUD.md](CODEX_CLOUD.md).
The rented-node screen is a separate ongoing run; its output is not bundled in
this source branch.

Everything is stored on this Mac at:

```text
/Users/chinmay/Documents/Codex/2026-09-07/if-they-did-speed-painting-type/painter/benchmarks/openrouter-teachers-20260922
```

There are 40 bundled, visually checked reference images and five default models. Quality and speed cover all 40 references (**400 episodes**); the separate `screen` track uses one deterministic reference from each of the eight categories (**40 episodes**). The default node launch retains the quality+speed matrix. An opt-in macOS launch uses the same pinned Watercolour renderer locally, with the actual canvas returned to the teacher for inspection, revision or finishing. Inspect the selected photos in [reference-gallery.html](reference-gallery.html).

## Launch after adding credits

From Terminal:

```bash
cd /Users/chinmay/Documents/Codex/2026-09-07/if-they-did-speed-painting-type/painter/benchmarks/openrouter-teachers-20260922
bash run_benchmark.sh --pod painter-photo-rl
```

The launcher reads `AI_GATEWAY_API_KEY` from the ignored `.env.local` file or the environment and otherwise asks for it with input hidden. It does not save the supplied key on this Mac or include it in artifacts. On the node, its temporary credential file is private and removed when the wrapper exits normally or through its cleanup handler. If the connection is lost during cleanup, follow the printed cleanup warning.

This uses an **existing** Lium node and its installed painter renderer; it never rents or stops GPUs. The current default node name is `painter-photo-rl`. If that pod has been released, select another prepared Linux node with `--pod NAME`. The Mac needs Python 3.10+, SSH, and the authenticated Lium CLI used by this workspace. The Linux node needs the project's pinned renderer, Playwright browser installation, Node.js/pnpm, and unprivileged `painter` user. The wrapper creates isolated Python and Node environments under `/root/painter/benchmarks/ai-gateway-teachers-20260923`, without mutating the shared training or renderer environments.

For a fresh Lium node, commit the reviewed source first and upload the tracked Git bundle plus the 40 ignored reference JPEGs:

```bash
python3 prepare_benchmark_node.py --pod YOUR_POD
lium ssh YOUR_POD
bash /root/painter/setup_benchmark_node.sh /root/painter
cd /root/painter/repo/painter/benchmarks/openrouter-teachers-20260922
bash run_benchmark_node.sh --dry-run
```

The upload script never includes `.env.local`, API keys, results, caches or model files. The node setup clones `/root/painter/source.bundle`, extracts the separate reference archive, installs Node.js 22 and pnpm, creates `/root/painter/renderer-env` with Playwright 1.58.0, Pillow 12.3.0 and tqdm, installs Chromium, creates the unprivileged `painter` user, installs the lockfile's AI SDK packages and runs a fixed 600×600 renderer smoke. It prints the exact key-prompting `bash .../run_benchmark_node.sh` command after setup; it never starts paid generation. Add `--dry-run` to that wrapper for a key-free node-side budget check.

For a free, offline configuration/package check that neither contacts Lium nor reads an API key:

```bash
bash run_benchmark.sh --dry-run
```

For the first screening pass, inspect the exact eight-reference, no-network budget with:

```bash
python3 run.py --root . --dry-run --track screen
```

The report lists the selected reference IDs, `episodes_total: 40`, a six-turn cap, native output allowance and zero paid calls. Run this track first, review the resulting canvases, then send only the strongest finalists through the 12-turn `quality` track. A screen episode may finish before four turns when the teacher observes a faithful current canvas; four to six turns is a target range, not a minimum.

## Run entirely on this Mac

Use the local mode when the benchmark should use the Mac's CPU/GPU and browser instead of a Linux node:

```bash
cd /Users/chinmay/Documents/Codex/2026-09-07/if-they-did-speed-painting-type/painter/benchmarks/openrouter-teachers-20260922
bash run_benchmark.sh --local --models zai/glm-5.3-flash --track speed --limit-episodes 1
```

The first real local run creates the ignored `.local-venv/`, installs the pinned `playwright`, Pillow and `tqdm` dependencies, and installs Chromium under the ignored `work/playwright-browsers/` directory when it is not already available. Node.js/pnpm is still used for the checked-in Vercel AI SDK transport. macOS 12 or newer, Python 3.10+, and a working `pnpm` are required. The renderer defaults to Chromium's Metal backend; `--renderer-backend swiftshader` is available if Metal is unavailable.

Local mode reads only the `AI_GATEWAY_API_KEY` assignment from this benchmark's ignored `.env.local`; it does not source or execute the file, print the key, put it in an argv value, or pass it to the renderer/browser. The same key can be supplied through the environment or hidden prompt. Requests still go to the paid Vercel AI Gateway, so use `--dry-run` first and start with `--limit-episodes 1`.

## Choose models, track or keys

Use exact Vercel AI Gateway model IDs:

```bash
bash run_benchmark.sh --pod painter-photo-rl \
  --models zai/glm-5.3-flash xiaomi/mimo-v2.6-pro \
  --track quality
```

Default IDs:

- `zai/glm-5.3-flash`
- `deepseek/deepseek-v4.1-flash`
- `stepfun/step-5-preview`
- `xiaomi/mimo-v2.6-flash`
- `xiaomi/mimo-v2.6-pro`

You can also edit the model list in `config.json`. IDs are passed byte-for-byte to the AI SDK and Gateway. The offline catalog is provenance only; a new or unsupported model is probed at runtime and its redacted provider error is recorded without inventing capability limits. Step 5 Preview is listed as image-capable by the public Gateway model catalog; its runtime availability and billing remain provider-controlled.

For separate keys per model, create a JSON file **outside this benchmark folder**, such as `~/.config/painter-benchmark-keys.json`:

```json
{
  "default": "YOUR_SHARED_AI_GATEWAY_KEY",
  "models": {
    "zai/glm-5.3-flash": "YOUR_OPTIONAL_GLM_KEY",
    "xiaomi/mimo-v2.6-pro": "YOUR_OPTIONAL_XIAOMI_KEY"
  }
}
```

```bash
chmod 600 ~/.config/painter-benchmark-keys.json
bash run_benchmark.sh --pod painter-photo-rl \
  --api-key-file ~/.config/painter-benchmark-keys.json
```

Keys here are Vercel AI Gateway account keys. Omitted model overrides use `default`. Do not paste real keys into `config.json`, the shell script or this README.

## Track settings

| Track | Turns | Output allowance | Episode allowance |
|---|---:|---|---|
| `quality` | Up to 12 | Native model ceiling, limited only by available context | No artificial episode time cap |
| `speed` | Up to 3 | 8,192 tokens per turn, including reasoning | 900 active seconds |
| `screen` | Up to 6 (target 4–6) | Native model ceiling, limited only by available context | No artificial episode time cap |
| `all` (default) | Quality and speed | Separate results | 400 episodes |

Quality and screening use the native Gateway model behavior and highest explicitly supported reasoning effort; speed uses an explicit 8,192-token output ceiling and lowest supported effort. Models without an advertised effort selector use the provider default without an invented effort setting. Each mode can stop early after inspecting a valid canvas. All request faithful, attractive paintings; there is no synthetic additive aesthetic score.

Native output ceilings can be large. The quality+speed default has at most 3,000 assistant requests before retries; the separate `screen` track has at most 240. Actual duration and cost depend on generated reasoning/code, full conversation input, pricing changes, retries and early stopping. API and rendering times, queue waits, token breakdowns, provider identifiers and reported costs are retained. Unknown costs are marked missing, not assumed to be zero. One sample per image/model/track does not estimate sampling variance.

## Progress, resume and artifacts

`tqdm` shows episode progress while generation runs, and the runner prints phase lines to stderr for each episode and turn. While a provider request is pending, a heartbeat appears every five seconds with the elapsed time, completed turns and known cumulative tokens. The current transport uses AI SDK `generateText`, so this is not a live token stream: prompt, completion and total token usage become available only after each provider response returns. The durable `progress.json` file includes a `live` field with the latest phase snapshot and cumulative known token count.

Rerun the same command to resume. Completed matching responses are reused, including when only rendering needs retry. Adding another model should preserve already completed results for the existing models. Changing the prompt, renderer or generation settings is a different experiment and must not silently reuse incompatible completed episodes.

The node retains the complete output tree. The Mac `run_on_node.py` wrapper collects available artifacts under `results-ai-gateway-20260923/`; the direct node launcher leaves them on the node for an explicit collection. Open `results-ai-gateway-20260923/gallery.html` after collection; raw trajectories are in `results-ai-gateway-20260923/episodes/`, with `progress.json` and `events.jsonl` for progress. Keep the terminal running for the foreground launch. If disconnected, outputs remain on the node for a subsequent resume/collection; do not release the node until desired evidence has been collected.

Stored evidence includes:

- Exact configuration, model metadata, prompt, reference hashes and source/runtime identities.
- Every assistant response and any reasoning returned by the provider, separate from proposed SFT targets.
- Conversation/image provenance for each request; generated programs; every valid canvas; renderer receipts and errors.
- Finish reasons, truncation, retry and deadline records; first and final valid canvases; explicit invalid-final flags.
- API/render/queue timing, token usage, known costs and missing-cost counts.
- A visual review gallery and blind review packet. **No visual winner is invented automatically.** Use [JUDGING.md](JUDGING.md) for pairwise quality review.

All demonstrations are candidates for later review. Executable code or a self-declared FINISHED response does not automatically enter training. See [PROTOCOL.md](PROTOCOL.md) for the comparison design and limitations.

## Validation before handoff

Offline tests cover the real runner CLI with multiple model IDs, including arbitrary Gateway IDs. The Linux node passed the fixed-sketch sandbox render with matching source/canvas hashes and the isolated `tqdm` environment bootstrap. One paid quality pilot has run; it is not a model ranking. Details: [VALIDATION.md](VALIDATION.md) and the root [research log](../../../RESEARCH_LOG.md).
