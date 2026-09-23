# Cloud teacher screen: first ten episodes

This record was written after launch while the task was running; the hypothesis
and comparison below describe the launch decision, not observed results.

- **Hypothesis:** Under the same six-turn visual feedback scaffold, at least
  one candidate besides DeepSeek may produce valid, reference-faithful
  paintings. Successful render rate, response latency, and tokens per valid
  painting should distinguish useful teachers before the longer quality and
  constrained speed tracks.
- **Matched baseline:** The prior cloud DeepSeek screen on
  `coco128-000000000109` used the same prompt, model, reference, track config
  and renderer, but only one stochastic sample. It produced six valid canvases
  in 845.3 active seconds with 184,655 reported tokens. The earlier rented
  12-episode screen differs in transport/environment and is diagnostic context,
  not a matched visual-quality baseline.
- **Task:**
  [`task_e_6ab3db1fdeac832ba1c4a113b05d5fa7`](https://chatgpt.com/codex/cloud/tasks/task_e_6ab3db1fdeac832ba1c4a113b05d5fa7),
  run ID `cloud-screen-first10-20260923`.
- **Immutable source:** Git commit
  `ddd335adb74ea44f53c731a4191b70830133e714`; config SHA-256
  `512e2cc3a7d566cbf14ceb7fe5196c05e108ccdbe3f5a62f70b9b962aac0c5f3`;
  renderer SHA-256
  `ad7ba24d632f0f16989741560c3af0fe42e426275c1df01954b60ddd69b63395`;
  reference manifest SHA-256
  `44334c164a7f1a3b1c7c0931444eefa64886d6bd88087735992cb53f78d45b74`.
  The public 40-reference dataset is pinned at HF revision
  `811564c415aac98601754ce6133de33bf25cd69d`.
- **Selection:** `--track screen --start-episode 0 --limit-episodes 10`,
  five configured Gateway model IDs on each of
  `coco128-000000000109` and `coco128-000000000061`, one sample per pair.
  Up to six turns and native model output ceiling per episode. Three concurrent
  API episodes, one renderer at a time. No rented GPU is involved.
- **Planned readout:** Count valid final canvases, failures and censored cases
  separately by model/reference. Inspect reference/canvas pairs blind to
  model where practical; report strengths and defects of chosen paintings.
  Measure turns, tokens, provider/render latency, and cost only where the
  provider reports it. Missing cost is unknown, not zero. Do not select a
  teacher from status alone or infer statistical superiority from two images.
- **Publication:** The cloud exit hook should publish a public HF archive and
  anonymous SHA-256 receipt under
  [`runs/cloud-screen-first10-20260923/`](https://huggingface.co/datasets/CK0607/komorebi-painter-teachers/tree/main/runs/cloud-screen-first10-20260923).
  Verify it before using outputs as training data.

## Observed result and decision

All ten episodes finished in 59m12.8s wall time, with 30 provider requests and
569,986 reported tokens. The statuses were two `turn_limit`, six
`renderer_error`, and two `api_error`. Four of ten episode records retain a
valid final canvas, including two that subsequently ended in a renderer error.
The six renderer errors were **hard 180-second render timeouts**, not evidence
that the corresponding paintings were visually poor. On reference `061`, all
five models hit that deadline at some point. The two API errors yielded no
model tokens. All 28 token-bearing turns lack provider-cost metadata, so the
recorded $0.00 known cost is not total spend.

| Model | Valid final canvas | Episodes | Reported tokens |
| --- | ---: | ---: | ---: |
| DeepSeek V4.1 Flash | 1 | 2 | 187,703 |
| Step-5 Preview | 1 | 2 | 86,998 |
| MiMo V2.6 Flash | 0 | 2 | 9,640 |
| MiMo V2.6 Pro | 1 | 2 | 144,210 |
| GLM 5.3 Flash | 1 | 2 | 141,435 |

The LFS exit upload failed with HTTP 501. A follow-up reused the **existing**
9,480,549-byte archive without any paid model calls and published ten regular
Git Base64 parts plus a [public receipt](https://huggingface.co/datasets/CK0607/komorebi-painter-teachers/blob/main/runs/cloud-screen-first10-20260923/receipt.json).
I independently fetched the parts anonymously at immutable HF dataset commit
`1afad82156cc8f8c4238592db7ee08424f911072`; the reassembled SHA-256
matched the preserved archive and receipt:
`19f3f704abff83b6b143c8765cf99933349bb8287284493b19c1f884e84e9a86`.

**Decision:** Do not rank teachers from these two references or count renderer
timeouts as aesthetic losses. The user-authorized 40-reference quality and
speed benchmark is running in independent cloud shards. Evaluate visual
quality on its valid canvases and treat the timed-out programs as censored;
where useful, rerender saved programs with a longer deadline before spending
on new model calls. No blind visual labels have been assigned yet.
