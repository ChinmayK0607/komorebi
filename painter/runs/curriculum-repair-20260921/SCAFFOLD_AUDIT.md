# Curriculum repair scaffold audit

This is a source/data audit for a possible next run. It does not report a
training result, render a new image, or make a quality claim. Historical run
directories remain unchanged.

## Findings

1. **The active native taskset admits tasks in manifest order.**
   `painter/runs/aesthetic-rl-20260921/taskset_impl.py` iterates the state list
   and then each state's anchors without family or source-group round-robin.
   The current `rl-states.json` prefix is `v2-box-000` (two anchors),
   `v2-box-001` (two), `v2-bus-000` (two), and `v2-bus-001` (one). The run's
   `RESULTS.md` records that the first bounded updates admitted only
   `v2-box-000` and `v2-bus-000`, so the available 12-family inventory was not
   actual training coverage. The new `prepare_curriculum.py` emits a
   family/group round-robin order and reports prefix coverage.

2. **Training and fresh-reference evaluation use different starts.**
   The active Env initializes every task from `state["canvas"]` and
   `state["program"]`, then sends both as the current observation. This is a
   correction task. `evaluate_matched.py` calls the rollout with
   `initial_canvas=None` and `initial_program=None`, so its baseline/candidate
   episodes start from the reference alone. The new curriculum makes `fresh`
   and `correction` explicit modes; the repaired Env supports both without
   silently changing the mode.

3. **The native Env replaces the full shared contract prompt.**
   Its `_QueueContract.system_prompt()` is a short generic instruction. It
   omits the canonical WEBGL origin/`translate(-300,-300)` rule, brush API
   guidance, no-change handling, and the explicit paint/finish protocol in
   `painter/contract.py`. The repaired next-run taskset imports `SYSTEM`,
   `NEXT_VERSION`, `NO_CHANGE`, and `FINISH_WITHOUT_CANVAS` from the run-local
   copy of the canonical contract. The contract change is versioned as
   `canvas-pixels-preview-v2`. The copied file hash is
   `0c77f0ed4bd5cad3121918d27495e3aef780dcb84ff0dd51cf61c3ffdd2ffaa0`, and
   its identity hash is
   `dd99279a5f3f6e98126e82db8b026c1ba726500cc8efb3fde53633772b5c65fb`.

4. **The native Env ignores pixel-identical/no-change observations.**
   After a valid render, the active taskset sends generic inspect/revise text
   even when `render_action()` returns `no_change=True`. The historical
   rollout uses the shared `NO_CHANGE` instruction for unchanged code or
   pixels. The repaired Env routes both unchanged-code and pixel-identical
   renders to that canonical instruction.

5. **The starting-program coordinate check exposes a wider reviewed-data issue.**
   `revision-rl-20260918/inputs/rl-states/state-03/program.js` (the archived
   `v2-bus-001` state) contains `translate(-300,-300)` in `setup()`, while the
   canonical contract requires the transform once in `draw()`. The historical
   validator only counts and normalizes the transform call; it does not inspect
   function placement. Applying the new pure check to all 24 archived starting
   programs rejects nine: states 02, 03, 05, 06, 08, 11, 14, 17, and 22.
   Five are misplaced transforms and four are missing or non-exact transforms.
   This is a data-review finding, not a reason to rewrite student programs.
   `teacher_program_checks.py` adds the placement check for reviewed teacher
   programs and never rewrites student programs.

6. **History differs by harness.**
   The native Env uses `agents.agent.interaction(task)` and sends subsequent
   `interaction.turn(...)` calls, preserving the native interaction trace.
   The historical matched rollout reconstructs a system plus user message on
   each turn; it carries the latest canvas and program, but not prior assistant
   messages. A fresh/correction comparison should therefore record its mode
   and use the same prompt/history contract before drawing conclusions.

## Scaffolded next-run files

- `contract.py`: byte-identical copy of canonical `painter/contract.py` for
  node packaging; no historical contract was edited.
- `taskset_impl.py`: next-run Env/taskset derivative with canonical prompts,
  explicit `fresh`/`correction` starts, and `NO_CHANGE` feedback.
- `render_actions.py`: same renderer boundary with optional empty starts for
  fresh tasks; no rendering is run by this scaffold.
- `teacher_program_checks.py`: canonical contract hash plus strict draw-place
  validation for reviewed teacher programs.
- `prepare_curriculum.py`: deterministic family/group-balanced ordering with
  explicit default first-cycle limits of 24 fresh tasks and 8 correction tasks,
  a recorded `max_turns` cap, source-manifest hash, and contract snapshot.

The repaired Env consumes `curriculum-plan.json` (or an explicitly configured
prepared artifact) before falling back to the legacy state manifest. It checks
the embedded contract snapshot at load time and records each task's configured
turn cap. A configured but missing curriculum path fails closed rather than
silently selecting the old order.

## Runtime dependency boundary

The scaffold is source/data ready but is not a launch receipt. On the Linux
node, the taskset still requires `verifiers.v1`, the run-local
`aesthetic_judge.py`, `judge_queue.py`, and `render_actions.py`. Rendering also
requires the node's `vendor/integrations/watercolour/renderer.py`, its
`renderer-env/bin/python`, and `/usr/sbin/runuser`; judge scoring requires the
filesystem queue worker to write receipts. Prime configuration/setup remains
owned by the run bootstrap and is not bundled or revalidated by this pure
scaffold.

The first-cycle ratio is configuration in the generated plan, not a hidden
gate. The helper fails closed when a reviewed manifest cannot supply the
requested number of rows.

The reviewed candidate manifest now produces a concrete
`curriculum-plan.json` with 24 fresh rows, 8 correction rows, and
`max_turns: 6`. The fresh prefix covers all 12 families and 24 source groups;
the eight correction rows map each reviewed `repair_id` from `input_png` to
`target_png`, use `input_png` as the pairwise anchor, and keep that anchor
distinct from the target/reference. The source manifest SHA-256 and v2
contract snapshot are embedded in the plan. The fresh rows still have no
reviewed distinct opponent in the candidate manifest, so the native loader
fails closed on their missing anchor until that separate pool is supplied;
this is a readiness boundary, not a launch claim.
