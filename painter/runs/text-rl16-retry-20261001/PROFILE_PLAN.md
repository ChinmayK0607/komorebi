# Renderer profile plan

The saved CPU recovery receipts show 19 deadline cases at the 180 second cap,
18 valid cases, and 10 completed invalid cases.  Valid receipts spend most of
their time in drawing (the valid median is about 75 seconds of wall time), and
the source programs are bounded but contain repeated filled polygons.  The
saved static characterization reports timeout cases at median 3,662 source
bytes, 20 polygon call sites, and 18 fill call sites, versus 2,893.5 bytes, 11
polygon sites, and 9 fill sites for valid cases.  These are source descriptors,
not executed counts, so the profile records both static metrics and runtime
events.

`profile_renderer.py` and `profile_worker.py` are an opt-in Linux-only path.
They verify the recovery manifest, renderer hash, asset hashes, source hashes,
and seed.  The worker preserves the frozen renderer's automatic startup order:
p5.js and p5.brush load, the saved source is appended to the body, and the
canvas-contract bootstrap follows the source.  It never calls `new p5()`.
Instrumentation wraps the existing brush methods with `original.apply(this,
args)`, preserving receiver, arguments, return value, and RNG.  It emits real
browser `performance.now()` start/end events through the Playwright console;
the parent appends those events to a durable journal.

The parent starts one worker process group per case and enforces the hard
deadline externally.  It kills the complete group on timeout and writes a
partial receipt containing phase events and any brush call whose start was
seen without an end.  Receipts include browser launch, context setup,
automatic-startup/drawing, PNG readback, image decode/write timings; wall time includes teardown.  The current reliable profile is intentionally a serial baseline.
Browser pooling is deferred until it can meet this same per-case watchdog
contract.

`--mode both` additionally runs the untouched pinned renderer
worker directly (rather than its parent launcher) on two valid controls
(`t600-342`, `t600-542`) and compares PNG hashes with the instrumented
baseline.  This keeps the same external watchdog process group.  A hash
mismatch or control/profile validity mismatch is reported as evidence that
instrumentation changed behavior; no visual-judge result is claimed.

The run summary records platform, Python, CPU count, scheduler affinity, and
available cgroup CPU quota/weight/cpuset values.  Worker receipts also record
the Chromium version.

The fixed profile subset is `t600-309`, `t600-323`, `t600-342`, `t600-542`, and
`t600-352`: two saved deadline cases, one slower valid case, one valid control,
and one completed invalid case.  Override it with repeated `--case` arguments.

Example on the Linux node or Codex Cloud, after copying the exact replay
runtime and inputs:

```sh
python painter/runs/text-rl16-retry-20261001/profile_renderer.py \
  --manifest /data/inputs/manifest.json \
  --runtime-root /data/runtime \
  --output-root /data/profile-20261001 \
  --mode both --timeout 180 --execute
```

This is a first-turn recovery profile, not a three-turn evaluation and not a
visual-judge result.  No model generation, target PNG, or network request is
part of the profile.
