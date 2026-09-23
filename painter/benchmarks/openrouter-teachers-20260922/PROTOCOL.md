# Multiturn painting teacher comparison

Prepared on 2026-09-23 as the Vercel AI Gateway migration of the 2026-09-22 teacher benchmark. **Not launched. No paid Gateway requests have been made for this benchmark.** Existing RL continues independently; this benchmark neither rents nor stops GPUs.

## Question and comparison

Which image-capable teacher produces the most attractive faithful reference painting after observing and correcting its actual rendered canvas? Separately, which produces the best painting within a constrained speed budget? Compare models within each track on identical image bytes and the same versioned painting contract. Do not combine quality, latency and token counts into an arbitrary reward or rank syntax validity as beauty.

The initial exact model IDs are `zai/glm-5.3-flash`, `deepseek/deepseek-v4.1-flash`, `stepfun/step-3.7-flash`, `xiaomi/mimo-v2.6-flash`, and `xiaomi/mimo-v2.6-pro`. IDs are passed unchanged to the AI SDK Gateway. The checked-in catalog is provenance only; the runner does not invent context or image capability limits for new IDs, and records the provider's runtime probe error when unsupported. Public sources: [AI Gateway models and providers](https://vercel.com/docs/ai-gateway/models-and-providers) and [AI Gateway SDK](https://vercel.com/docs/ai-gateway/sdks-and-apis).

Gateway model IDs are provider model names, not immutable weight revisions. Actual response model/provider identifiers remain in raw receipts. Provider routing and catalog revisions can change; do not claim immutable weights or hardware-matched latency.

## References

`refs.json` freezes 40 existing COCO128 training-pool photographs and their SHA-256 hashes: eight structural categories with five images each. They cover animal anatomy and occlusion, vehicle perspective, crowds, action poses, interior geometry, food/table arrangements, people with objects, and fine environmental structure. Annotation counts help characterize complexity; they are not difficulty scores or evidence of visual quality.

The original 28 development photographs remain excluded. These references have prior project exposure; this is a matched teacher-generation comparison, **not an unseen student generalization test**. The 40 original JPEGs are bundled unchanged under `references/`. Provenance and per-image license status are retained. Images and generated artifacts stay out of Git. Do not silently upload this benchmark or its images publicly.

## Two separate tracks

| Setting | Quality | Speed |
|---|---|---|
| Maximum assistant turns | 12 | 3 |
| Completion ceiling per request | Model's catalog-native ceiling, subject to remaining physical context | 8,192 tokens including reasoning |
| Episode time budget | None beyond operational request/render deadlines | 900 active seconds |
| Reasoning selection | Highest explicitly supported effort | Lowest explicitly supported effort |
| Models without advertised effort control | Enable reasoning without inventing an effort enum | Enable reasoning without inventing an effort enum |
| Temperature | 1.0 | 1.0 |
| Samples per reference/model | 1 | 1 |

A model can finish early only after receiving an actual valid canvas. A code response is not also a finish action. Both tracks prioritize aesthetics and faithful subject structure; speed adds explicit limits. The different reasoning setting is an intentional speed strategy, so this is a practical two-mode comparison, not a controlled isolation of token budget alone. Native caps are ceilings, not requested response lengths. Do not add an arbitrary low cap to quality, truncate programs, quietly switch models, or substitute a text-only model.

The complete default matrix is **400 episodes** (40 images × 5 models × 2 tracks), with at most **3,000 assistant turns before bounded service retries**. This is not a small fixed-cost API smoke test. Actual spending depends on generated reasoning/code, full conversation input, pricing changes, retries and early stopping. Saved provider usage/cost is authoritative when available; estimates and missing values must be labeled separately.

## Interaction and evidence

The teacher receives the real reference, paints a complete p5.brush sketch, and sees the real renderer output before revising. Keep the reference visually available throughout. Rendering errors are returned as errors; a failed render must not be represented as a successful new canvas. The last valid canvas may be retained with explicit invalid/deadline flags. Preserve every earlier canvas so a worse final revision can be diagnosed; do not silently choose a judge-selected best intermediate and call it the model's final output.

Save full prompts and conversation messages, assistant replies, returned reasoning fields, complete programs, image/render hashes and receipts, turn and episode outcomes, finish/truncation reasons, timestamps, API and render durations, queue waits, retries, token breakdowns, response model/provider identifiers and costs. Credentials are the exception to 'everything': no API keys, Authorization headers, private connection metadata or secret environment values enter artifacts. Returned reasoning is separate evidence, not automatically an SFT target.

Linux performs rendering with the existing pinned p5.js 2.3.2 / p5.brush 2.2.1 runtime. The existing unprivileged painter renderer and browser isolation apply. The 180-second renderer deadline is a service timeout shared by the tracks, not a stroke budget. No model training runs locally. API generation is external, so local GPU utilization is not applicable; API/provider latency is not the same as intrinsic model speed. Rendering contention on a node also running RL must be reported rather than hidden.

## Analysis and next decision

Inspect first versus final paintings, per-turn regressions, early finishing, syntax/render failures, and image-specific model differences. Use reference-conditioned blind pairwise judgments of beauty and fidelity, with weaknesses of the preferred image also recorded; the rubric is in `JUDGING.md`. Report all 40 references, ties and uncertainty. Separate service-censored cases from model failures, retaining every case in coverage reports. Display quality, actual latency, cost and token usage separately. A valid render alone does not establish quality.

This is one sample per model/image/track and does not estimate sampling variance. Any apparent winning teacher needs an image-paired quality comparison and a cost/latency tradeoff, not just cherry-picked attractive examples. Exported demonstrations remain candidates until visual review; do not automatically admit every executable program or self-declared FINISHED output to SFT. No optimizer updates or checkpoint improvements are claimed by this preparation.
