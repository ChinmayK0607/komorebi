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
| quality | 00 | 0 | 20 | `full-quality-00-20260923` | pending |
| quality | 01 | 20 | 20 | `full-quality-01-20260923` | pending |
| quality | 02 | 40 | 20 | `full-quality-02-20260923` | pending |
| quality | 03 | 60 | 20 | `full-quality-03-20260923` | pending |
| quality | 04 | 80 | 20 | `full-quality-04-20260923` | pending |
| quality | 05 | 100 | 20 | `full-quality-05-20260923` | pending |
| quality | 06 | 120 | 20 | `full-quality-06-20260923` | pending |
| quality | 07 | 140 | 20 | `full-quality-07-20260923` | pending |
| quality | 08 | 160 | 20 | `full-quality-08-20260923` | pending |
| quality | 09 | 180 | 20 | `full-quality-09-20260923` | pending |
| speed | 00 | 0 | 20 | `full-speed-00-20260923` | pending |
| speed | 01 | 20 | 20 | `full-speed-01-20260923` | pending |
| speed | 02 | 40 | 20 | `full-speed-02-20260923` | pending |
| speed | 03 | 60 | 20 | `full-speed-03-20260923` | pending |
| speed | 04 | 80 | 20 | `full-speed-04-20260923` | pending |
| speed | 05 | 100 | 20 | `full-speed-05-20260923` | pending |
| speed | 06 | 120 | 20 | `full-speed-06-20260923` | pending |
| speed | 07 | 140 | 20 | `full-speed-07-20260923` | pending |
| speed | 08 | 160 | 20 | `full-speed-08-20260923` | pending |
| speed | 09 | 180 | 20 | `full-speed-09-20260923` | pending |

Launch source commit: pending. Results and next decision: pending.
