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

Results, visual labels, cost and next decision are pending task completion.
