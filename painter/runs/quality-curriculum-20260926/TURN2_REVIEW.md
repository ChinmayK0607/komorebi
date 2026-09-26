# Actual-canvas correction pilot, 2026-09-26

**Hypothesis.** Showing a strong teacher its exact reference, prompt, complete first-turn program, and rendered canvas may produce a better second painting. The matched baseline for each of six scenes is its own first-turn canvas from `teacher500-simple-cc0-pilot-20260926`; input bytes and renderer are identical. This is a teacher-data quality probe, not student training or RL.

All six teachers actually inspected the reference and first-turn canvas before writing a complete replacement program. The revised programs passed the painting contract and JavaScript syntax checks. The two public source bundles include the prior canvas and full program by SHA-256. Their Linux second-turn renders were published and anonymously hash-verified, then the parent and an independent `gpt-5.6-luna` low reviewer displayed the exact reference, prior and new images. These are nonblind visual judgments, not calibrated human preferences.

| Scene | New versus prior | Remaining weakness | Decision |
| --- | --- | --- | --- |
| Shell `openverse-a698f10d` | Better silhouette and connected spiral | Still flat; shell ridges, highlights and mineral texture missing | Continue to a third inspected turn; no SFT admission |
| Watering can `openverse-db1a60b0` | Slightly better squat body/spout relation | Tall rough handle, disconnected shadow, little metallic reflection | Continue to a third inspected turn; no SFT admission |
| Coast `openverse-8f9803c4` | Mixed: clearer shoreline structure | Large geometric dark wedges dominate, little rock or water texture | Keep diagnostic only; no SFT admission |
| Lake `openverse-3f66008b` | Worse | Large ochre blobs obscure the sailboat and disturb the calm framing | Reject correction |
| Barn `openverse-093b0ce2` | Worse overall | Warped/floating roof and poor wall perspective despite more openings | Reject correction |
| Artist tools `openverse-99091b40` | Tie to slightly worse | Tube/tabletop artifacts; glass, lighting and brush bundles remain weak | Reject correction |

The six-program outcome is **two relative improvements, one mixed, three without improvement; zero high-quality admissions**. Relative improvement is insufficient: these remain stylized, mechanical approximations of the references. The separate 14-text and nine-photo first-paint CPU samples also rendered 14/14 and 9/9 valid, yet parent inspection found flat geometry and scratch-like artifacts in several examples. Do not use renderer validity as a training label or scale this unscreened pool into SFT.

## Immutable evidence and scope

| Run | Scope | Source bundle SHA-256 | Public data revision | Run summary SHA-256 | Code commit | Linux render wall time |
| --- | --- | --- | --- | --- | --- | ---: |
| `teacher500-simple-cc0-pilot-20260926` | 11 first turns; 11 valid | `6af60ddf17a9528b6333eed2f58edb08cd8cd081d9f033a937c7f7d802b51fd2` | `3d3ba8eb20190b8fafa23e81e881a9da9f4ab43b` | `1dcdf7196f43a872e7f778d0c4ffe9e4a2978addaa4233c3d914a5027ba85417` | `bb5dc0ef05985a9fed88fe52259d3a69d8ee0d88` | 364.003 s |
| `teacher500-astra-turn2-20260926` | 4 second turns; 4 valid | `23839247f033f180d891c1de0a236d9575ea0cbcf42d91fa0b206590c2897acb` | `2c55d7b2f426554458a11f77dcf3963fa8a833fa` | `be9321991bd6c514a59dfbd0df5fb85b6bc1e9ad44e0c7c60a64ca638bd6f5bc` | `7b3f5e8788bd5fa0546210f68aa62df311a1144d` | 165.919 s |
| `teacher500-sol-turn2-20260926` | 2 second turns; 2 valid | `e03ae6bb9424e076fbfc5fd17e5735f72efbc271073e3e16a34420d1b063a1e7` | `17942469ddbcf9a20dc7bd2a93c4002cbdce67ae` | `06becc1b4e757addb0cd21778eea6b0cae687fbccbf419ec97b3ea0b642fa9fd` | `7b3f5e8788bd5fa0546210f68aa62df311a1144d` | 71.256 s |

All three runs used renderer SHA-256 `ad440b1aa8612e293b52fa5e68c2252926722852155eb8cd145d369a862a8a63`, a 600-second per-program timeout, and two bounded CPU workers. The exact per-example reference/program/prior/new hashes are in the public render summaries and locally verified episodes. [Astra reference/prior/new gallery](../../collected/quality-curriculum-20260926/rendered-cloud/teacher500-astra-turn2-20260926/review.html) and [Sol reference/prior/new gallery](../../collected/quality-curriculum-20260926/rendered-cloud/teacher500-sol-turn2-20260926/review.html) retain visible comparisons. The public receipts are under `runs/<run-id>/receipt.json` in [the public teacher dataset](https://huggingface.co/datasets/CK0607/komorebi-painter-teachers/tree/main/runs).

No GPU was rented. Codex Cloud CPU utilization and provider cost were not instrumented; two workers and elapsed wall time are throughput observations, not utilization measurements. No case was censored by the renderer timeout. The next decision is to render one more actual-canvas correction for the shell and watering can, then admit only if the new canvases meet the aesthetic bar; rejected second turns must not become positive SFT targets.
