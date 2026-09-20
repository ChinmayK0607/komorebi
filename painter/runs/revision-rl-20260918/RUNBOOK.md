# Luna-low asynchronous RL pilot

User explicitly chose **gpt-5.6-luna, reasoning low** on September 20. OpenRouter is no longer used. No pending judge choice or API key is required. A real Luna-low agent graded four preserved pairs successfully before rental. All previous pods are released.

Active pod: `930789d5-94da-4f46-80a7-1c7f2a1c0366`, 2 RTX PRO 6000 Blackwell Server Edition 96 GB, $2.38/hour total. Provider TTL: **2026-09-20 08:29:39 UTC**. SSH root@154.54.100.111, port 40300, known_hosts-luna-20260920. Resolve the exact pod with Lium before any lifecycle action. Do not reconnect prior ids or rent a duplicate.

deploy_async.py packages portable inputs and current scripts, transfers only the HF checkpoint credential, starts setup_async_node.sh, then activate_async.sh waiting for setup. Activation starts supervise_async.py and the native Prime coordinator. Separate trainer and inference GPUs; filesystem LoRA broadcasts; eight samples per group, 16 samples per update, 32 episodes in flight, bounded policy staleness, eight pilot updates, lr 1e-6. Initializer is the exact contract-quality-20260918/step_24 adapter pinned in EXPERIMENT.json. No optimizer resume, speed reward or length penalty. ML and rendering run only on node.

Node output: /root/painter/async-output/revision-rl-luna-20260920. Public hash-verified HF adapters: CK0607/qwen3.8-27b-brush-painting/revision-rl-luna-20260920/step_N. Confirm native optimizer audits before claiming updates. No global promotion without held-out evaluation. sample_phase.sh is obsolete.

## Live Luna queue

Run `python3 luna_queue_ctl.py status --host 154.54.100.111 --port 40300` from this run directory. Local control data: luna-live-20260920/; latest-status.json records each status snapshot.

For unassigned rewards, run `python3 luna_queue_ctl.py pull live-NNN --host 154.54.100.111 --port 40300 --limit 12` using a NEW batch name. This exports and hash-verifies reference/A/B images and stages three blinded judge shards. Assign each luna-live-20260920/reward-batches/live-NNN/judge-0X.json to an actual Luna low agent. Reversed views go to different agents. Agents must inspect the actual images and must not read private keys, other labels, code, policy identities or metrics.

Each agent writes labels-0X.json in that batch directory, with schema:
`{"judge_model":"gpt-5.6-luna","reasoning_effort":"low","judgments":[{"id":"...","winner":"A|B|tie|uncertain","confidence":"high|medium|low","acceptable_A":"yes|no|uncertain","acceptable_B":"yes|no|uncertain","reason":"specific visual evidence","reviewed_images":["reference absolute path","A absolute path","B absolute path"]}]}`
Grade reference fidelity (subject, silhouette, placement, proportions, connected parts, completion) and brushwork. Attractive but structurally wrong is unacceptable. Uncertainty is allowed.

Run `python3 luna_queue_ctl.py push live-NNN --host 154.54.100.111 --port 40300` after labels arrive. It validates provenance, image hashes and reversed consistency, then installs completed receipts atomically on node. Partial pushes are supported. Never invent labels or turn infrastructure failures into zero rewards. Historical Ling labels are not shipped as Luna rewards.

The supervisor stops trainer/inference if pending rewards make no progress for 15 minutes. Keep Luna agents grading while work is pending. On completion/failure or a sustained stall, preserve rl-rollouts, judge-queue, reward-batches, native configs/logs/metrics/audits and checkpoint upload receipts; exclude secrets and redundant base weights. Release the exact pod and confirm provider removal before TTL. Prior restart-setup-preserved-20260920.tar.gz is historical, SHA-256 1a0b777f12d9aabd4606614e9c112e4710e2566b23fee99e20474623abda7ce1.
