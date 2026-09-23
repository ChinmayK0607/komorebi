# Bounded matched photo evaluation

`launch_matched_eval.sh` is a finite, foreground Linux launcher for twelve
photo cases. It selects the lowest reviewed `selection_rank` with one case per
`primary_family` from the staged 28-case manifest, then gives frozen SFT step
512 and public RL step 24 the same six-turn native paint → render → inspect →
revise protocol: seed `918200`, temperature `1.0`, thinking disabled, 8,192
output tokens per turn, 32,768 context, and a 180-second renderer deadline.

The GPU phase has a hard 2,700-second (45-minute) deadline covering vLLM
startup and both evaluation workers. It starts one 27B vLLM server on each GPU,
records `nvidia-smi` telemetry, preserves launcher metadata and logs under
`eval-runs/photo-curriculum-rl-20260922-step24/`, and kills only process groups
whose leaders it started. Prime's generated turn traces and canvases follow
the config's relative `output_root` and are written separately under
`evaluation-rollouts/photo-sft-step512/` and
`evaluation-rollouts/photo-rl-step24/`. Quality scoring is offline; the
launcher does not contact a judge or upload results. A timeout leaves partial
rollouts and a `completion.json` status of `worker_failed` for review.

After the workers have stopped, collect both trees and the launcher evidence
with this read/copy command (the destination is a separate evidence folder):

```bash
set -Eeuo pipefail
RUN=/root/painter/runs/photo-curriculum-rl-20260922
EVAL="$RUN/eval-runs/photo-curriculum-rl-20260922-step24"
OUT="$RUN/collected-eval-step24"
mkdir -p "$OUT/evaluation-rollouts" "$OUT/eval-runs"
cp -a /root/painter/evaluation-rollouts/photo-sft-step512 "$OUT/evaluation-rollouts/"
cp -a /root/painter/evaluation-rollouts/photo-rl-step24 "$OUT/evaluation-rollouts/"
cp -a "$EVAL" "$OUT/eval-runs/"
```

The copy preserves invalid programs, failed renders, turn records, and
launcher logs for later review; it does not delete or alter either source
tree.

On the prepared node, after staging the 28-case source manifest and its
reference files under `eval-source/`, the exact invocation is:

```bash
cd /root/painter
GPU_PHASE_TIMEOUT_SECONDS=2700 \
  bash /root/painter/runs/photo-curriculum-rl-20260922/launch_matched_eval.sh
```

The launcher expects the pinned Prime environment at
`/root/painter/prime-rl/.venv`, vLLM `0.28.0`, Transformers `5.6.2`, two CUDA
devices, the Qwen3.8-27B snapshot at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, the SFT adapter pointer in
`initial-adapter-path.txt`, and the verified RL step-24 backup under
`checkpoint-preservation/adapters/step_24`. It verifies adapter bytes, rank 16,
alpha 32, and the public step-24 receipt before starting a server. Paths and
revision/hash values can be overridden through environment variables; no
credential is read or emitted by this launcher.
