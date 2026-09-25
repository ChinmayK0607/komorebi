# Pro cup canary — 2026-09-25

**Question.** Can MiMo V2.6 Pro, using the same solid-underpaint prompt and simple-cup reference as the completed MiMo Flash canary, produce a visibly better valid painting within four turns? This is a teacher-scaffold decision, not a training run.

**Matched evidence.** The Flash canary used source `7ec3c7d`, the same reference SHA-256 `c6a862a0b281fa693ceca9b6a34a092b025a8721e68696083441179cde95341b` and the same `quality-prompt.txt`. Its turn-4 canvas was valid but omitted the cup; turns 5 and 6 were invalid. It consumed 166,935 reported tokens over six turns, with missing provider cost fields. The separate Pro cake probe on a harder COCO image produced a recognizable but mechanical six-turn canvas and consumed 203,969 reported tokens. Neither trajectory is admitted to SFT.

**Canary.** `probe_mimo_pro_cup_cloud.sh` uses one verified training reference, the unchanged solid-underpaint prompt, four turns maximum, 16,384 completion tokens per turn, no automatic provider retries, a 900-second request deadline, a 240-second render deadline, and a 2,400-second episode deadline. It runs on Codex Cloud CPU and publishes a public, anonymously hash-verified archive under `mimo-pro-cup-opaque-20260925`. No Lium GPU or scheduled task is used. The source commit and final archive revision must be recorded after execution.

**Decision.** Inspect reference, all actual renders and code. Require a recognizable opaque cup, attached handle, coherent table/wall composition and pleasant painterly finish. If Pro cannot clear this easy case within the bound, stop scaling the current full-sketch-per-turn scaffold and change the action representation or teacher before buying a large curriculum. A valid program or a turn-limit status alone is not acceptance.
