# Full cloud teacher benchmark, 40 references

This is the prelaunch record for the user-requested full benchmark. Results
will be added only after each public archive has been anonymously downloaded
and hash-verified.

- **Hypothesis:** Across 40 diverse references, the six-to-twelve-turn quality
  track will distinguish reference-faithful teachers from models that merely
  produce valid code. The constrained speed track will expose whether those
  teachers retain recognizable structure with fewer turns and tokens.
- **Matched baseline:** For reference `coco128-000000000109`, the cloud
  DeepSeek six-turn screen at source `4361dd56df26f3b25de1ebf72da0c711ae0a11d1`
  produced six valid canvases in 845.3 active seconds with 184,655 provider
  tokens. The ten-episode, two-reference cloud screen at source
  `ddd335adb74ea44f53c731a4191b70830133e714` is still running at launch.
  Neither is a matched 12-turn or three-turn visual-quality comparison; the
  full benchmark establishes those track-specific baselines.
- **Source and data:** Git branch `codex/painter-teacher-cloud` at the launch
  commit recorded below. Config SHA-256
  `512e2cc3a7d566cbf14ceb7fe5196c05e108ccdbe3f5a62f70b9b962aac0c5f3`;
  renderer SHA-256
  `ad7ba24d632f0f16989741560c3af0fe42e426275c1df01954b60ddd69b63395`;
  reference manifest SHA-256
  `44334c164a7f1a3b1c7c0931444eefa64886d6bd88087735992cb53f78d45b74`.
  The public 40-reference dataset is pinned at HF revision
  `811564c415aac98601754ce6133de33bf25cd69d`.
- **Scope:** Five configured Gateway models × 40 references × two tracks =
  400 episodes. Quality allows at most 12 visually inspected turns at native
  output ceiling. Speed allows at most three turns, 8,192 output tokens per
  call, and a 900-second episode timeout. Maximum possible requests: 3,000;
  actual calls, billed tokens, and cost will be measured from receipts.
- **Execution:** 20 independent cloud tasks, ten per track. Each task gets
  four consecutive references × five models = 20 episodes, via
  `--start-episode` offsets 0, 20, ..., 180 and `--limit-episodes 20`.
  All tasks use the same prepared Codex cloud environment, three concurrent
  API episodes and one renderer per task. A task may fail without erasing
  another task's public evidence. This is a finite launch; no scheduler or
  watcher is created.
- **Publication:** Each task has a unique `PAINTER_RUN_ID` of
  `full-{quality|speed}-{00..09}-20260923` and must publish its archive to
  the public dataset `CK0607/komorebi-painter-teachers`. A receipt is usable
  only after anonymous SHA-256 verification, including Base64-part fallback
  when direct LFS upload fails.
- **Readout:** Report final valid canvas rate, structural fidelity and
  aesthetics per category, turns, tokens, active and provider-wait time,
  failures and censored episodes separately. Provider cost that is absent
  from the API response remains unknown, not zero. The candidate for teacher
  data will be selected on painting quality and reliability, not parse status
  alone. No SFT data should be accepted without reviewing the images and
  render provenance.

| Track | Shard | Zero-based start | Episodes | Run ID | Cloud task |
| --- | ---: | ---: | ---: | --- | --- |
| quality | 00 | 0 | 20 | `full-quality-00-20260923` | [task](https://chatgpt.com/codex/cloud/tasks/task_e_6ab3e1811f60832bbc129d6e66cb03c0) |
| quality | 01 | 20 | 20 | `full-quality-01-20260923` | [task](https://chatgpt.com/codex/cloud/tasks/task_e_6ab3e198a4ec832b8f1d626c4f9583aa) |
| quality | 02 | 40 | 20 | `full-quality-02-20260923` | [task](https://chatgpt.com/codex/cloud/tasks/task_e_6ab3e19d9b0c832b9d17b004857fa8b5) |
| quality | 03 | 60 | 20 | `full-quality-03-20260923` | [task](https://chatgpt.com/codex/cloud/tasks/task_e_6ab3e1a27a64832b944b94772e7393d6) |
| quality | 04 | 80 | 20 | `full-quality-04-20260923` | [task](https://chatgpt.com/codex/cloud/tasks/task_e_6ab3e1a78f84832bb5759daf2bfd979d) |
| quality | 05 | 100 | 20 | `full-quality-05-20260923` | [task](https://chatgpt.com/codex/cloud/tasks/task_e_6ab3e1d0f2c4832ba89192303d1aeb45) |
| quality | 06 | 120 | 20 | `full-quality-06-20260923` | [task](https://chatgpt.com/codex/cloud/tasks/task_e_6ab3e1d51114832bafd2868129063d8c) |
| quality | 07 | 140 | 20 | `full-quality-07-20260923` | [task](https://chatgpt.com/codex/cloud/tasks/task_e_6ab3e1d9c14c832bb9b3b34780ee4ead) |
| quality | 08 | 160 | 20 | `full-quality-08-20260923` | [task](https://chatgpt.com/codex/cloud/tasks/task_e_6ab3e1de9884832bbe3f1445b31296fe) |
| quality | 09 | 180 | 20 | `full-quality-09-20260923` | [task](https://chatgpt.com/codex/cloud/tasks/task_e_6ab3e1e30270832ba016c4f0ddd1a46e) |
| speed | 00 | 0 | 20 | `full-speed-00-20260923` | [task](https://chatgpt.com/codex/cloud/tasks/task_e_6ab3e1f1848c832bae58b9bce16ecaf8) |
| speed | 01 | 20 | 20 | `full-speed-01-20260923` | [task](https://chatgpt.com/codex/cloud/tasks/task_e_6ab3e1f66670832b99a702e4b1212094) |
| speed | 02 | 40 | 20 | `full-speed-02-20260923` | [task](https://chatgpt.com/codex/cloud/tasks/task_e_6ab3e1fb6068832b8eaaf892291efd38) |
| speed | 03 | 60 | 20 | `full-speed-03-20260923` | [task](https://chatgpt.com/codex/cloud/tasks/task_e_6ab3e1ffd5a4832baaf23153998c96b9) |
| speed | 04 | 80 | 20 | `full-speed-04-20260923` | [task](https://chatgpt.com/codex/cloud/tasks/task_e_6ab3e2049848832ba4cc62b5cd1ad783) |
| speed | 05 | 100 | 20 | `full-speed-05-20260923` | [task](https://chatgpt.com/codex/cloud/tasks/task_e_6ab3e2145b40832ba730d3acff20d9bc) |
| speed | 06 | 120 | 20 | `full-speed-06-20260923` | [task](https://chatgpt.com/codex/cloud/tasks/task_e_6ab3e2197288832b98434c03445c6b9e) |
| speed | 07 | 140 | 20 | `full-speed-07-20260923` | [task](https://chatgpt.com/codex/cloud/tasks/task_e_6ab3e21dd6dc832bb114db8c125eb2d8) |
| speed | 08 | 160 | 20 | `full-speed-08-20260923` | [task](https://chatgpt.com/codex/cloud/tasks/task_e_6ab3e23f9cc0832bb7e53cc553963e18) |
| speed | 09 | 180 | 20 | `full-speed-09-20260923` | [task](https://chatgpt.com/codex/cloud/tasks/task_e_6ab3e257e65c832bb037b94bffadeeae) |

Launch source commit: `10f047ac91204bf1260e9325d4003f9a7d4e57a3`.
All 20 tasks were submitted on 2026-09-23 UTC. Each quality and speed track
ran separately; results below distinguish verified completion from submission.

## Speed track: complete, visual ranking pending

All ten speed shards completed and published public
[HF run directories](https://huggingface.co/datasets/CK0607/komorebi-painter-teachers/tree/main/runs)
named `full-speed-00-20260923` through `full-speed-09-20260923`.
Each has a `receipt.json` and Base64 text-part archive. I anonymously fetched
and reassembled **all ten** at their receipts' immutable dataset revisions and
independently matched each archive byte count and SHA-256. The archives contain
200 distinct model/reference episodes, 579 response turns and 5,703,928
reported tokens. There are 579 provider requests in the summaries. Cost
metadata is missing for 574 turns; known cost of $0.00 is not total spend.

| Model | Valid final canvas / 40 | Turn limit | Invalid | Render timeout | Deadline censored |
| --- | ---: | ---: | ---: | ---: | ---: |
| GLM 5.3 Flash | 31 | 22 | 6 | 11 | 1 |
| DeepSeek V4.1 Flash | 21 | 19 | 10 | 11 | 0 |
| MiMo V2.6 Pro | 21 | 15 | 17 | 8 | 0 |
| MiMo V2.6 Flash | 14 | 13 | 19 | 5 | 3 |
| Step-5 Preview | 9 | 6 | 30 | 3 | 1 |
| **Total** | **96 / 200** | **75** | **82** | **38** | **5** |

Every `renderer_error` in the speed track is a hard 180-second render timeout.
Valid-final counts include retained earlier canvases in 13 `invalid` and eight
`renderer_error` episodes. One unblinded look at reference `109` found
recognizable river, bank and grass composition in three valid paintings, but
small details and geometry remain rough. This is not a pairwise teacher ranking.
The next decision is to visually compare a diverse, category-balanced sample,
classify invalid programs, and distinguish render-time censoring from model
quality before accepting teacher demonstrations.

## Quality track: running

The ten quality shards were still running at the speed-track readout. No
quality archive had been publicly verified yet. Its outcomes, cost and visual
comparison against speed remain pending.
