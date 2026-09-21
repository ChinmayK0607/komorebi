# COCO capability comparison

The unadapted Qwen3.8-27B can produce recognizable rough paintings of some real-photo scenes, but this setup does not yet produce reliably good paintings. The current contract-quality step24 adapter transfers poorly to these references. Do not expand real-photo RL from that adapter on the strength of its earlier procedural results.

## Matched inference result

All 24 episodes completed the six-turn allowance. Each policy saw the same twelve frozen COCO development photographs, shared painting instructions, response budget of 8192 tokens, 98304-token context, temperature 1 and seed 918200. Thinking was disabled. Neither policy explicitly finished a painting; “retained” below means the last valid canvas available at the operational turn limit, not a quality pass.

| Recorded measure | Unadapted base | SFT step24 |
|---|---:|---:|
| Episodes | 12 | 12 |
| Episodes with a retained valid canvas | 11 | 3 |
| Valid paint attempts | 38 / 72 | 13 / 72 |
| Responses without an accepted JavaScript block | 2 / 72 | 35 / 72 |
| Unchanged valid attempts | 1 / 38 | 9 / 13 |
| Completion tokens | 303,491 | 68,066 |
| API responses ending with `stop` | 72 / 72 | 72 / 72 |
| API responses ending with `length` | 0 | 0 |

Five SFT episodes repeat a short no-code plan for every turn. The sink episode has one working program surrounded by five no-code replies. Other failures repeat syntax errors or unsupported brush calls. The short 9–15-token replies stop voluntarily despite the 8192-token allowance. This is not per-response token truncation. The inspected training artifacts do not establish why this behavior developed; [diagnostic notes](DIAGNOSTIC_NOTES.md) distinguish evidence from hypotheses.

## Visual review

The parent inspected all twelve reference/first/final rows, plus larger images where needed. Base café, kitchen, bed and the first bus painting retain recognizable object relationships. The cat's anatomy, clock geometry and mountain structure are weak. The lunchbox changes from a colorful rough arrangement to neon outlines on black; the bus's later washes lose the vehicle silhouette. Valid code is clearly insufficient as the quality criterion. None of the reviewed results meets the intended standard for a high-quality teacher demonstration.

Luna-low judged all eleven base first-versus-final pairs with valid images in neutral A/B order. It preferred final in **4**, first in **6**, and tied **1**. The three base/SFT pairs with canvases on both sides produced two base preferences and one SFT preference. These are single-pass judgments, not order-validated judge accuracy or a statistically decisive model ranking. Nine missing SFT canvases are counted as execution failures, not invented aesthetic votes. Parent disagreements, including overly strong judge wording about the sink subject, remain in the diagnostic record.

| Photo subject / ID suffix | Base retained | SFT retained | Luna: base first vs final |
|---|---|---|---|
| Café / 127 | Yes | No | Final |
| Lunchboxes / 009 | Yes | No | First |
| Tennis / 419 | Yes | No | First |
| Cat / 599 | Yes | No | First |
| Loft bed / 133 | Yes | Yes | Final |
| Kitchen / 089 | Yes | No | First |
| Sink and toy / 208 | Yes | Yes | First |
| Baseball / 357 | No | No | Unavailable |
| Skier / 382 | Yes | Yes | Final |
| Aircraft / 542 | Yes | No | Final |
| Bus / 641 | Yes | No | First |
| Clock / 078 | Yes | No | Tie |

Raw judge labels: [policy pairs](preview/blind-review/labels.json) and [revision pairs](preview/revision-blind/labels.json). The neutral-position key and final raw native traces are preserved with the collected evidence. [View the complete comparison](review/index.html).

## Operational limits and preservation

The primary run lasted **2026-09-21 19:44:24–20:37:06 UTC**. Its single RTX PRO 6000 Blackwell inference GPU averaged **94.41% utilization across 317 ten-second samples**, from 19:44:24.868 to 20:37:05.581 UTC. **298/317 samples (94.01%) were at least 90%**. This includes the rendering/idle tail; it is sampled inference-device activity, not proof of optimal throughput or saturated trainer/inference RL. No trainer or optimizer update ran. Browser rendering sometimes took close to the 120-second hard deadline.

The interrupted first attempt used 32k context and four serving sequences. It hit confirmed context errors on fourth requests and is preserved separately as `attempt-1-context32k`; it is excluded from all primary counts above. The primary rerun changed the operational context/concurrency/timeout envelope while retaining the frozen cases, model pins and prompts. No primary API call had a length cutoff.

The baseball base episode hit four render deadlines after two earlier program failures. A post-hoc replay of the exact last program **rendered validly in 155.652 seconds** with the supported 180-second allowance. Its source hash is unchanged (`6940292430543e72be80d523b2003bac5ec9e4a9debbde899334dde737b862a5`). The 120-second limit caused that primary failure; this successful replay must not be substituted into the matched 11/12 result. The initial replay request of 300 seconds was rejected by the renderer's 180-second maximum before any rendering, and its log is preserved. The clock is the only other terminal base render-deadline case; its exact last program also rendered validly under the same extension, in **134.797 seconds**. Both controls are displayed separately in the gallery. Thus the base generated a renderable program for every reference when these two deadline controls are considered, which still says nothing by itself about visual quality.

Model revisions, adapter hash, source commit and exact node configuration hashes are in [EXPERIMENT.json](EXPERIMENT.json). The original node manifest and corrected local replay manifest have different metadata hashes but identical frozen case records. The node TOML configurations and operational amendment describe the actual primary settings; the original node manifest retains its earlier protocol metadata.

The archive and **all 525 listed regular files** verify locally; four `logs/latest` directory aliases are recorded in the preservation manifest without duplicating their target contents. [Preservation receipt](receipt-preservation.json). The GPU node was removed and its absence confirmed at **2026-09-21 20:52:24 UTC**. Estimated rental through that confirmation is **$2.29 or less at the quoted $1.29/hour**, including setup, the interrupted attempt, primary evaluation, rendering controls and preservation. This is a rate-based estimate, not a final invoice. No new checkpoint was trained or copied locally.

The gallery has 24 matched episodes and two explicitly separate controls; all 45 image links resolve. The offline summarizer's three focused tests and the gallery/preparation suite's five tests passed. Implementations were supplied by Luna-xhigh and reviewed by the parent; no source change was applied to the completed model trajectories.

## Decision

Use the base as the comparison initializer for the next real-photo adapter, and expand stronger teacher demonstrations plus successful corrections of student-generated mistakes. An API judge can help select better rendered alternatives at scale; selecting a winner between two bad images must not turn it into an aesthetic gold target. Keep development references out of teacher training data. The concrete proposed data handoff is [DATA_HANDOFF.md](DATA_HANDOFF.md).

This experiment is one sample per photo/policy with thinking disabled. It does not establish the base model's best-of-k or reasoning-enabled ceiling, and COCO does not support a claim of unseen pretraining data. No new training, API-provider switch or scheduled task was launched.
