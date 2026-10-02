# Actual three-judge RL ablation — 2026-10-02

## User objective and hypothesis

The user clarified that the ablation means three actual RL runs, not merely grading frozen images. Compare which judge produces the best visual checkpoints and reinforces the intended readability, beauty, warmth and spatial precision. The offline three-model assay is supporting diagnostic evidence only; it does not satisfy this training comparison or prove a training improvement.

Register three independent continuations, all starting from the same old scaled RL16 adapter with a fresh optimizer: Flash, Pro, GPT-6 Luna. Only the judge model changes. The same current v2 holistic rubric already includes beauty, warmth, readability, spatial relationships and muddy overpainting. Do not introduce new additive reward terms or merge weights in this comparison.

## Matched recipe

- Base model `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; Prime commit `26b3131d2716a4f8210b165df584b83b4bc54f61`.
- Initial policy `CK0607/qwen3.8-27b-brush-painting@ac4d3eda8e5925e2d23992cfdc1a63620ebde573`, prefix `text-pairwise-scaled-20260929/step_16`, adapter SHA-256 `e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c`. Each arm reloads this exact initializer; no chaining one judge's result into another arm.
- Frozen SFT opponent from public revision `adf687e5b321514eda7be6eab050434fdc8e44ff`, prefix `photo-curriculum-sft-20260922-balanced/step_512`, adapter SHA-256 `bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740`.
- Same 40 interleaved training briefs, semantic manifest SHA-256 `13ac0eebb3595fbd931ee7864e69101abdc8b7c9c1faccf11dfcd9bd495c6501`, and 47 disjoint held-out briefs. Same frozen PNG opponents and brief hashes. This reproduces the grounded run's curriculum, not the distinct 68-brief original scaled curriculum. Data expansion follows judge comparison rather than changing data between arms.
- Group4, batch16, learning rate1e-6, identical optimizer/normalization/credit/admission logic, three turns with8,192 maximum completion tokens and32,768 context, painter temperature1 and thinking disabled. Prime's pinned training defaults are orchestrator seed42 and inference seed0; the matched evaluation panel uses seed918210 in every arm.
- Judges: `xiaomi/mimo-v2.6-flash`, `xiaomi/mimo-v2.6-pro`, `openai/gpt-6-luna`, explicit requested reasoning medium, equal4,096 output ceilings, omitted sampling controls. Same v2 prompt SHA-256 `d21e4644e7a3874a0694b2ad406420e733b7638222eb7717e7f5f91cbdf1613c`, exact side schema, balanced views, same recheck consensus and whole-group uncertainty censoring. Xiaomi maps non-none effort to thinking enabled, so equal requested effort does not mean equal internal compute. Preserve actual usage and model identities.

The materializer keeps the three run roots as flat siblings under `painter/runs/`: `rl-judge-ablation-flash-20261002`, `rl-judge-ablation-pro-20261002`, and `rl-judge-ablation-luna-20261002`. Each root has its own config, manifests, checkpoints, receipts, and `rl-rollouts` directory. The shared `rl-rollouts` name is required by the existing archive/supervisor contract; the sibling roots and distinct run names keep the evidence isolated.

Use [`campaign.sh`](campaign.sh) as the single foreground node command after the parent has allocated the exact pod and supplied its start epoch and whole-node price:

```bash
RUN=/root/painter/repo/painter/runs/rl-judge-ablation-20261002
bash "$RUN/campaign.sh" "$RENTAL_START_EPOCH" "$PRICE_USD_PER_HOUR" 59.4
```

The campaign prepares the shared source once, divides the elapsed setup time from the rental start equally across the three `$19.80` per-arm budgets, then runs Flash, Pro, and Luna sequentially through the bounded pipeline. It records terminal outcomes, requires a verified public archive receipt before advancing to the next arm, checks that the previous arm's owned process group is gone, and creates `campaign-release.ready` only after all three public archives are verified. A missing public archive stops the campaign and leaves the arm censored for parent review; a local archive alone is not release proof. The parent controller remains responsible for exact-pod release; this command contains no rental credentials or provider calls.

On a fresh node, preparation invokes the reviewed bootstrap setup before checking the frozen SFT path file. That bootstrap installs the pinned runtime and materializes the public SFT adapter path; the shared retry preparation then reuses that verified path for all three arms.

## Comparison, budget and stopping

Sequential independent arms reuse one node's runtime/model cache. The registered candidate topology is eight GPUs with one trainer and seven inference replicas, 96 initial/max in-flight episodes, and eight renderer workers; these values are identical in all three configs. The pinned adapter initializer supports one trainer GPU, so preparation rejects the earlier two trainer split. Prime maps inference to seven tensor-parallel-one data-parallel replicas; the parent must still verify the actual provider's memory and startup before launch. The restart opts into Prime's native activation CPU offload (`pin_memory=true`, `max_inflight_activations=1`) while leaving optimizer CPU offload disabled; the pinned implementation uses `use_streams=False`, so the matrix does not claim asynchronous stream overlap. This is a source-supported runtime setting, not proof that the 27B trainer fits a particular card or that throughput is unchanged. It may trade device memory for host-memory use and slower updates, so record measured interval utilization and update throughput. The prior measured 27B BF16 run peaked at about 92.2 GiB on a 96 GiB card, so the parent must preserve trainer-card headroom. Exact price/start/topology/source/config hashes will be recorded before training starts.

Hard GPU ceiling **$20 per arm**, including allocated preparation/evaluation/preservation overhead; total maximum across three arms **$60**, not a claim of expected cost. Gateway charges separate. This matrix records a 1,920-second per-arm completion reserve: 1,200 seconds for the matched16 evaluation, 360 for archive preservation, 240 for parent release, and 120 margin. Target up to50 updates only while the registered trend rises; first actual-optimizer trend decision at15, then20/every5, stop when latest5 mean fails to exceed prior5. Same stopping rule across arms; unequal stop lengths remain explicit. A judge's own reward curve is not a cross-judge quality score.

Public hash-verified checkpoints every4 updates. Step12 is the predeclared primary quality comparison: when the existing public-checkpoint validator confirms a public step12 adapter, the bounded pipeline evaluates that step on the same16-case panel. If an arm cannot verify step12, it evaluates its latest verified public checkpoint and records the result as exploratory and censored for the primary comparison. The training trend stop and the quality checkpoint choice are separate decisions; a judge's own reward curve never selects its quality checkpoint. A parent-controlled postpass may collect the same panel at other common steps only when the completion reserve fits. Avoid calling an arm superior because it alone was allowed more updates. Predeclare the same16-case evaluation panel: simple t600-512/516/542/543; intermediate425/433/439/445; complex309/366/375/377/413/466/478/560. These are selected diagnostic cases from the original47 held-out prompts, not a random new confirmation set. Include invalid final actions and censored cases in denominators. Final render must be the same defined action/turn selection in every arm; do not replace failed final images with earlier best-looking ones. Blinded visual comparison against RL16 and frozen SFT512 is primary, with judge agreement/censoring and observed failures supporting interpretation. Preserve complete47 evaluation option for a later independent confirmation when affordable.

The previous all47 global30-minute cutoff truncated evaluation. This matrix requires a bounded matched smaller panel and a complete/partial summary even on a cutoff; show the16/47 scope rather than inventing a47-case result. After an arm finishes, upload screened checkpoint/evidence publicly, independently hash verify, then advance from the original initializer to the next arm. Release the exact pod at completion or the finite campaign cost boundary. No scheduled task, heartbeat watcher or recurring automation.

## Supporting API wiring check

Before GPU rental, six Gateway calls (three models, both image orders on the same fan) returned exact requested model identities and valid strict schemas with explicit medium reasoning. All three models reversed their image preference between orders on this pair; the images and mappings were correct. This is actual judgment inconsistency, not a parser failure, and remains censored/rechecked under the same shared protocol in all arms. It is not evidence that any judge is generally better. Reported call times: Luna8.35/10.16s, Flash14.69/21.03s, Pro16.48/20.39s; limited to two calls each, not a general latency benchmark. Raw safe receipts are in `../judge-threeway-ablation-20261002/local-evidence/wiring-six-calls/`.

Status: implementation/preparation in progress; no GPU rental or actual training has started.

## Allocation attempt 1: setup failure, zero updates

Source `f6e9388e615186555e89b2f209f06bda0a1b0586` was cloned from the reviewed Git bundle onto eight 97,887-MiB cards. The finite campaign began at 07:24:15 UTC and failed shared preparation before any optimizer or server startup. The exact-pod controller independently verified release at 07:24:41 UTC. Request-to-release GPU-clock estimate is $1.9389 at $13.52/hour, not a provider invoice. Include this failed setup in the campaign cap and allocate it equally across all three arms.

Fresh-checkout diagnosis: the Git index records the base `prepare_node.sh` as nonexecutable while the matrix checks `-x`, despite calling it through Bash. Local working-tree permissions hid this mismatch. The remote preparation log was not collected before release, so this is a deterministic reproduced source defect and inferred explanation of the remote exit, not a preserved remote error message. Next: remove the unnecessary executable dependency, verify clean checkout semantics, preserve setup failures, and retry from the same model/data/judge recipe with the remaining GPU cap.

## Allocation attempt 2: corrected source, shared preparation

Reviewed source `a3ada6cb3c75879c6916af202f8af787d817cfe5` passed the plan check in a clean clone from the Git bundle. Retry request began 07:29:04 UTC; the finite campaign began 07:29:52 UTC. Actual node exposes eight 97,887-MiB cards and a 128-core cgroup CPU quota. One trainer/seven inference replicas, 96 in-flight episodes, and eight renderer workers are fixed for every arm. These are configuration values, not utilization measurements.

The retry budget is $57.45 at $13.52/hour, $19.15 per arm including one-third of common setup. With failed attempt1 ($1.9389 estimated), the combined GPU-clock ceiling is $59.3889, or $19.7963 per arm. Source transfer and protected credential file modes were verified without reading or logging credential values. Shared preparation is downloading the public pinned checkpoints and installing the runtime. No optimizer update or quality improvement is established yet. The finite controller will release the exact pod on terminal exit or budget deadline; no recurring automation was created.

## Allocation attempt 2 terminal outcome: wrong interpreter, no training

Shared preparation installed Prime/renderer, downloaded pinned public weights, verified the40/47 anchor split and materialized three arm roots. Arm config validation then failed with `Prime config import unavailable: ModuleNotFoundError: No module named pydantic`. The matrix captured its Python interpreter before bootstrap; system Python remained selected after the Prime environment was created. This is a preserved remote error, not a model or judge failure. All arms have zero optimizer updates and no new checkpoints.

The finite controller verified exact-node release at07:53:19 UTC (13:23:19 IST), before the user status check. The provider inventory now contains zero active pods. Retry GPU-clock estimate $5.4662; combined two failed setup attempts $7.4051, not invoices. No additional rental is being made during repair. Next: refresh the pinned interpreter after installation, exercise a test where bootstrap creates that interpreter during the shell flow, and review the resulting config validation before retrying.

## Interpreter repair reviewed

Preparation now uses system Python only for the dependency-free plan check, then resolves the pinned Prime interpreter after bootstrap and uses it for materialization and all three config validations. An explicit test interpreter override remains supported. Parent review reran26 ablation/controller tests successfully, including a temporary shell flow in which bootstrap creates the interpreter only after the plan check and all three config invocations must use it. Shell syntax and plan checks pass. This verifies the interpreter-handoff regression; it is not a successful Linux training launch or a learning result. No GPU is allocated during this repair.

## Restart plan after unavailable 96-GiB offer — 2026-10-02

The user authorized restarting and continuous monitoring. The original eight96-GiB offer is no longer available. Registered replacement: eightA10080-GiB cards, one trainer/seven inference replicas, 96 in-flight and eight renderer workers, native activation CPU offload enabled identically in all three arms. Optimizer CPU offload remains disabled. This changes memory placement and hardware, not the reward, sampling, optimizer objective, initial adapters, or datasets; the comparison is still conditional on this common runtime. Actual fit, gradient updates and throughput remain unverified until the Linux smoke step.

Candidate whole-node price $9.20/hour, advertised3,872Mbps download and1,772GiB host RAM; actual cgroup limits and GPUs must be verified after allocation. Remaining campaign cap $51.90, or$17.30 per arm. Including$7.4051 prior failed setup divided equally, each arm remains below$20 ($19.7684 ceiling); Vercel charges are separate. A finite rental expiry backs up the exact-pod controller, and finite30-second observations collect campaign stages and trainer counters; this is not a scheduled Codex task or recurring automation. Five-second GPU telemetry is retained by the existing launcher for interval utilization analysis. The campaign stops on a shared zero-update training/preflight fatal, preserving evidence instead of repeating the same failure in all three arms.

## Allocation attempt 3: eight-A100 restart

The replacement was allocated at12:20:12 UTC (17:50:12 IST), source `8ef3c32dc1adc4e9dc0f5f9fd21e449d38818605`. Clean Git-bundle plan validation and35 scoped tests passed before rental. The finite exact-pod controller started immediately after allocation with a$51.90 remaining campaign cap,335-minute provider expiry, and shared CPU activation offloading. Current status is shared preparation, not a completed optimizer update. See `allocation-attempt3.json`; actual memory fit, gradient updates and utilization will be recorded after node measurements.

## Attempt 3 terminal result: shared compiler-cache failure

Shared preparation completed in347 seconds from request; all three native configs validated with1trainer/7inference andCPU activation offload. Flash passed the33-field native rollout check and started seven data-parallel inference workers. During kernel autotuning, one worker raised `FileNotFoundError` for a Triton kernel JSON under the shared inference cache. The missing file is observed; a concurrent compiler-cache collision is the leading mechanism, pending isolated-cache replay. Trainer initialization/backward never completed; zero optimizerupdates. Therefore80-GiB training fit and gradient correctness are still untested.

The zero-update fatal guard prevented Pro/Luna startup. The pipeline preserved a47,020-byte screened archive publicly; anonymous retrieval SHA256 `b885c0785be62c8df7ee8aab22c44d9f0ef0bfab0cbfece6e525e82eb8bcdb9f` at dataset revision `bebefdd91d5333ac4895a22bd2939211fc2bdb47`. Exact-pod release verified12:35:17UTC (18:05:17IST). Attempt3 cost$2.3123; prior setup combined$9.7174. Neither is a providerinvoice. [Terminal record](allocation-attempt3-result.json). Next: isolate per-worker compiler caches in the child-startup hook, exercise spawned-child regressions, then replay the same RL recipe. No GPU is allocated during repair.

Startup profile:12:27:13–12:34:04UTC,336GPU-samples across8cards (42/card; requested5-second sampling, realized interval about10seconds). Prime mapped physical0–6 to inference,7 to trainer. Inference sample-mean utilization1.26%, nonzero-utilization samplefraction25.17% over294samples; trainer0% over42samples. Inference maximum allocatedmemory63,304MiB; trainer4MiB. These are startup-only unweighted samplemeans, not an end-to-end saturated training measurement. [Window and denominators](allocation-attempt3-profile.json).

## Compiler isolation repair and next replay registration

Inference now forces the pinned vLLM worker context to `spawn`, and the Python-startup hook gives each process distinct vLLM/Triton/Inductor namespaces usingPID plus a fresh token (rank included if present). Trainer cache behavior remains unchanged; cache reuse is limited to the process lifetime. Parent inspected vLLMv0.28.0 `v1/engine/utils.py` (EngineCoreManager uses `get_mp_context`) and `utils/system_utils.py` (reads `VLLM_WORKER_MULTIPROC_METHOD`, calls `multiprocessing.get_context`). This ensures the child hook executes rather than relying on fork inheritance.

Parent reran34 scoped tests, including actual Python child-startup tests, source hash/plan validation and existing runner contracts; all pass. These prove cache wiring, not GPU warm-up or gradients. Same RL16/model/data/rubric/optimizer protocol across all judges. Remaining GPU cap$49.50 ($16.50 perarm), with$9.7174 failed setup spread equally: combined perarm ceiling$19.7391. Proposed same eightA10080/$9.20hour offer, finite319-minute expiry; no new rental until reviewed source bundle passes its clean plan check.

## Allocation attempt4: isolated workers

Restart allocated12:44UTC (18:14IST) from reviewed source `0c011a989c9201d11f877068e5469458f4b5f2c2`; clean bundleplan and34targetedtests passed. Same eightA10080/$9.20hour, CPUactivationoffload, matched three-arm RL protocol. The finite controller started immediately; remaining campaigncap$49.50 with319-minute expiry. This is an allocation/preparation record, not a completed update. [Safe allocation record](allocation-attempt4.json).

## Attempt4 terminal result and bounded readiness repair

Inference workers completed compilation and CUDA graph capture; the earlier missingTriton file exception did not recur. The launcher cut startup at its600-second readiness deadline while API processor warm-up was still progressing. Parent termination produced a final KeyboardInterrupt; this is not an independently observed processor defect. Zero optimizer updates; Pro/Luna skipped. An attempted mid-run compiler-sidecar patch was blocked by a node-materialized tracked source-identity file and **was not applied**. The next fresh bundle includes that reviewed compatibility fix.

Public screened archive:50,745bytes, revision `e4163ae55abefe92ccd412fda4e93a86801b1f12`, SHA256 `b46df59741f920ba13eb869406aac701f61d5d6d0f93ccb0eb2b0f8588b50f7c`; anonymous retrieval verified. Node release13:02:55UTC (18:32:55IST). Attempt4 GPU-clock estimate$2.8732; cumulative failed starts$12.5907, not invoices. [Terminal record](allocation-attempt4-result.json).

Next replay retains all registered arms and opts into a bounded1200-second initial server readiness allowance, instead of600; the historical default and120-second adapter load waits stay unchanged. Seven workers warmed close to the old cutoff, motivating this allowance; readiness and80-GiB trainer backward remain unproved. Parent37 scoped checks pass, including invalid/overlarge timeout rejection. Remaining totalcap$46.50 ($15.50/arm) plus prior failures gives combined per-arm ceiling$19.6969, below$20; provider299-minute expiry also bounds the rental. No GPU allocated during repair.
