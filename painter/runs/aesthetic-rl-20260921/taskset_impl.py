"""Native Prime multi-turn aesthetic painting taskset.

The task deliberately keeps the interaction loop in a v1 ``vf.Env``.  A model
turn is followed by a real renderer observation, and the next model turn is
sent through ``vf.Agent.interaction``.  This is the native path that preserves
the renderer's exact token ids, sampled-token mask and sampling logprobs for
the complete conversation.

The run-specific ``aesthetic_judge`` and ``judge_queue`` modules own the judge
schema and queue.  This module only binds a final reference/anchor/candidate
triple to those pinned APIs and records its terminal score.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib
import inspect
import json
import math
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import verifiers.v1 as vf


RUN_DIR = Path(__file__).resolve().parent
MAX_GENERATED_TURNS = 3
RENDER_LIMIT = asyncio.Semaphore(8)
JUDGE_TIMEOUT_SECONDS = 1800.0


def atomic(path: Path, value: Any) -> None:
    """Write JSON without exposing a partial result to queue readers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _image(path: Path) -> vf.ImageUrlContentPart:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return vf.ImageUrlContentPart(
        image_url=vf.ImageUrlSource(url="data:image/png;base64," + encoded)
    )


def _contract(root: Path):
    """Load the run's pinned aesthetic schema and queue facade."""
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return _QueueContract(root)


class _QueueContract:
    """Small facade over the sibling aesthetic schema and judge queue."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.judge = importlib.import_module("aesthetic_judge")
        self.queue = importlib.import_module("judge_queue")

    def system_prompt(self) -> str:
        return (
            "Paint the supplied reference as an attractive p5.brush JavaScript "
            "sketch on a 600x600 canvas. Inspect the actual current canvas after "
            "each render. Return a complete sketch in a javascript code block; "
            "return FINISHED alone only after the visible canvas is complete."
        )

    def initial_prompt(self, *, state, reference, current, program):
        return [
            vf.TextContentPart(text="REFERENCE"),
            reference,
            vf.TextContentPart(text="CURRENT CANVAS"),
            current,
            vf.TextContentPart(text="Current program:\n" + program),
            vf.TextContentPart(text="Inspect the images and paint the next complete version."),
        ]

    def observation_prompt(self, *, state, feedback, has_program):
        return feedback or "Inspect the actual canvas and revise only remaining mismatches."

    def enqueue_comparison(self, *, root, trace_id, reference, anchor, candidate):
        return self.queue.enqueue_comparison(
            root, str(trace_id), reference, anchor, candidate
        )

    async def await_receipt(self, *, root, request_id, timeout):
        receipt_path = root / "judge-queue" / str(request_id) / "receipt.json"
        started = time.monotonic()
        while not receipt_path.is_file():
            if time.monotonic() - started > timeout:
                raise TimeoutError(f"aesthetic judge receipt timed out: {request_id}")
            await asyncio.sleep(1.0)
        return json.loads(receipt_path.read_text())

    def score_receipt(self, *, receipt, request_id, reference, anchor, candidate):
        return self.queue.score_receipt(
            receipt, request_id, reference, anchor, candidate
        )


async def _contract_call(function, **kwargs):
    """Call the sibling contract without imposing an async queue API."""
    parameters = inspect.signature(function).parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        selected = kwargs
    else:
        selected = {name: value for name, value in kwargs.items() if name in parameters}
    result = function(**selected)
    if inspect.isawaitable(result):
        return await result
    return result


def _prompt_parts(root: Path, state: dict[str, Any], *, observation: Path | None = None,
                  program: str | None = None, feedback: str | None = None) -> list[Any]:
    """Build one multimodal user message; every model turn has two images."""
    contract = _contract(root)
    reference = root / state["reference"]
    current = root / (state["canvas"] if observation is None else observation)
    if hasattr(contract, "observation_prompt"):
        text = contract.observation_prompt(
            state=state, feedback=feedback, has_program=program is not None
        )
    else:
        text = feedback or "Inspect the reference and current canvas, then paint the next version."
    parts: list[Any] = [
        vf.TextContentPart(text="REFERENCE"),
        _image(reference),
        vf.TextContentPart(text="CURRENT CANVAS"),
        _image(current),
    ]
    if program is not None:
        parts.append(vf.TextContentPart(text="Current program:\n" + program))
    parts.append(vf.TextContentPart(text=text))
    return parts


def _initial_prompt(root: Path, state: dict[str, Any]) -> list[Any]:
    contract = _contract(root)
    if hasattr(contract, "initial_prompt"):
        return contract.initial_prompt(
            state=state,
            reference=_image(root / state["reference"]),
            current=_image(root / state["canvas"]),
            program=(root / state["program"]).read_text(),
        )
    return _prompt_parts(
        root,
        state,
        program=(root / state["program"]).read_text(),
        feedback="Inspect the images and paint the next complete version.",
    )


def _is_finish(reply: str) -> bool:
    """Recognize only a standalone explicit finish marker without a program."""
    # Code plus FINISHED must first render and become visible on the next turn.
    if "```" in reply:
        return False
    return bool(re.search(r"(?m)^\s*FINISHED\s*$", reply))


class PainterAestheticData(vf.TaskData):
    root: str
    state: dict[str, Any]
    anchor: str
    max_turns: int = MAX_GENERATED_TURNS


class PainterAestheticConfig(vf.TasksetConfig):
    root: str


class PainterAestheticTask(vf.Task[PainterAestheticData, vf.State, vf.TaskConfig]):
    @vf.reward(weight=1.0)
    async def final_canvas_aesthetic(self, trace: vf.Trace) -> float:
        """Score exactly one terminal final-canvas comparison.

        Operational failures are intentionally raised so Prime excludes the
        trace.  A cap after a valid visible canvas remains scoreable; the
        cap itself never becomes a reward component.
        """
        info = trace.info
        if info.get("service_failure"):
            raise RuntimeError(str(info["service_failure"]))
        final_canvas = info.get("final_canvas")
        if not final_canvas or not info.get("final_canvas_valid"):
            raise ValueError("no valid visible canvas was produced")
        root = Path(self.data.root)
        candidate = Path(final_canvas)
        if not candidate.is_absolute():
            candidate = root / candidate
        reference = root / self.data.state["reference"]
        anchor = root / self.data.anchor
        if not all(path.is_file() for path in (reference, anchor, candidate)):
            raise ValueError("final comparison image is missing")

        contract = _contract(root)
        request_id: str | None = None
        if hasattr(contract, "enqueue_comparison"):
            request_id = await _contract_call(
                contract.enqueue_comparison,
                root=root,
                trace_id=trace.id,
                reference=reference,
                anchor=anchor,
                candidate=candidate,
            )
        else:
            raise RuntimeError("aesthetic judge queue lacks enqueue_comparison")
        info["judge_request_id"] = request_id
        started = time.monotonic()
        try:
            receipt = await _contract_call(
                contract.await_receipt,
                root=root, request_id=request_id, timeout=JUDGE_TIMEOUT_SECONDS
            )
        except Exception as exc:
            info["service_failure"] = f"judge service failure: {type(exc).__name__}: {exc}"
            raise
        info["judge_wait_seconds"] = time.monotonic() - started
        if not isinstance(receipt, dict):
            raise ValueError("judge receipt must be a JSON object")
        if hasattr(contract, "score_receipt"):
            score = await _contract_call(
                contract.score_receipt,
                receipt=receipt,
                request_id=request_id,
                reference=reference,
                anchor=anchor,
                candidate=candidate,
            )
        else:
            raise RuntimeError("aesthetic judge queue lacks score_receipt")
        if score is None:
            raise ValueError("uncertain or provenance-invalid aesthetic receipt")
        score = float(score)
        if not math.isfinite(score) or not (0.0 <= score <= 1.0):
            raise ValueError(f"aesthetic score outside [0, 1]: {score}")
        info["judge_receipt"] = receipt
        info["final_canvas_score"] = score
        receipt_path = root / "judge-queue" / str(request_id) / "receipt.json"
        # training_handoff consumes this small trace-keyed record.  The native
        # interaction trace remains the source of token ids/masks/logprobs,
        # while the queue receipt remains the source of judge labels.
        atomic(
            root / "rl-rollouts" / str(trace.id) / "reward.json",
            {
                "status": "scored",
                "reward": score,
                "score_evidence": {
                    "status": "scored",
                    "request_id": request_id,
                    "receipt_sha256": sha(receipt_path),
                    "protocol_version": contract.judge.PROTOCOL_VERSION,
                },
            },
        )
        return score


class PainterAestheticEnvConfig(vf.EnvConfig):
    agent: vf.AgentConfig = vf.AgentConfig()


class PainterAestheticEnv(vf.Env[PainterAestheticEnvConfig]):
    """Drive paint -> visible canvas -> inspect/revise/finish."""

    async def run(self, task: PainterAestheticTask, agents: vf.Agents) -> None:
        root = Path(task.data.root)
        root.mkdir(parents=True, exist_ok=True)
        state = task.data.state
        current_canvas = root / state["canvas"]
        current_program = (root / state["program"]).read_text()
        folder = root / "rl-rollouts" / task.data.state["id"] / str(uuid.uuid4())
        folder.mkdir(parents=True, exist_ok=True)
        trace_id = None

        async with agents.agent.interaction(task) as interaction:
            trace_id = interaction.trace.id
            # Make the immutable starting canvas explicit for the score hook.
            interaction.trace.info.update(
                initial_canvas=str(current_canvas),
                initial_canvas_sha256=sha(current_canvas),
                anchor=task.data.anchor,
                turn_cap=task.data.max_turns,
            )
            segment = await interaction.turn()
            for turn in range(1, task.data.max_turns + 1):
                if segment.terminated:
                    break
                reply = segment.last_reply
                if _is_finish(reply):
                    # FINISHED is meaningful only after the model has observed a
                    # successful canvas.  The initial state is already visible.
                    interaction.trace.info.update(
                        finished=True,
                        finish_turn=turn,
                        final_canvas=str(current_canvas),
                        final_canvas_valid=True,
                        operational_cutoff=False,
                    )
                    break
                sample = folder / f"turn-{turn}.json"
                atomic(
                    sample,
                    {
                        "id": f"{interaction.trace.id}-turn-{turn}",
                        "text": reply,
                        "finish_reason": "stop",
                        "state": state,
                        "initial_canvas": str(current_canvas),
                    },
                )
                try:
                    from render_actions import render_action

                    async with RENDER_LIMIT:
                        result = await asyncio.to_thread(
                            render_action,
                            root=root,
                            folder=folder / f"turn-{turn}",
                            reply=reply,
                            initial_canvas=current_canvas,
                            initial_program=current_program,
                        )
                except Exception as exc:
                    # Renderer/process failures are infrastructure failures and
                    # must not be disguised as model-correctable syntax errors.
                    interaction.trace.info["service_failure"] = (
                        f"renderer service failure: {type(exc).__name__}: {exc}"
                    )
                    break

                interaction.trace.info.setdefault("turns", []).append(result)
                if result.get("valid"):
                    current_canvas = Path(result["canvas"])
                    current_program = result.get("program", current_program)
                    feedback = result.get("feedback") or "Inspect the actual canvas and revise only remaining mismatches."
                else:
                    # Invalid student code is recoverable.  Preserve the last
                    # valid canvas and expose the render error as the next user
                    # turn, so the native trace contains the repair attempt.
                    feedback = "The previous program failed to render:\n" + str(
                        result.get("error") or result.get("renderer_errors") or "unknown render error"
                    ) + "\nCorrect it and return a complete JavaScript sketch."

                if turn >= task.data.max_turns:
                    # Render the final generated turn before applying the cap.
                    # A valid canvas is scoreable without an explicit FINISHED.
                    interaction.trace.info.update(
                        final_canvas=str(current_canvas),
                        final_canvas_valid=current_canvas.is_file(),
                        operational_cutoff=True,
                        finish_turn=None,
                    )
                    break

                parts = _prompt_parts(
                    root,
                    state,
                    observation=current_canvas,
                    program=current_program,
                    feedback=feedback,
                )
                segment = await interaction.turn([vf.UserMessage(content=parts)])

            if not interaction.trace.info.get("final_canvas"):
                interaction.trace.info.update(
                    final_canvas=str(current_canvas),
                    final_canvas_valid=current_canvas.is_file(),
                    operational_cutoff=True,
                )


class PainterAestheticTaskset(vf.Taskset[PainterAestheticTask, PainterAestheticConfig]):
    """Load prepared same-reference states and optionally round-robin anchors."""

    def load(self) -> list[PainterAestheticTask]:
        root = Path(self.config.root).resolve()
        state_candidates = (
            root / "aesthetic-states.json",
            root / "rl-states.json",
            root / "inputs" / "aesthetic-states.json",
            root / "inputs" / "rl-states.json",
        )
        states_path = next((path for path in state_candidates if path.is_file()), None)
        if states_path is None:
            raise FileNotFoundError(
                "aesthetic state manifest is missing; checked: "
                + ", ".join(str(path) for path in state_candidates)
            )
        contract = _contract(root)
        manifest = json.loads(states_path.read_text())
        states = manifest.get("training_states", manifest) if isinstance(manifest, dict) else manifest
        if not isinstance(states, list):
            raise ValueError("aesthetic state manifest must contain training_states")
        tasks: list[PainterAestheticTask] = []
        for index, raw in enumerate(states):
            state = dict(raw)
            anchor_rows = list(state.get("anchors") or [])
            anchors = [
                row.get("canvas") if isinstance(row, dict) else row
                for row in anchor_rows
            ]
            if not anchors:
                anchors = [state.get("anchor") or state["canvas"]]
            for anchor_index, anchor in enumerate(anchors):
                anchor_path = Path(anchor)
                if anchor_path.is_absolute():
                    try:
                        anchor = str(anchor_path.relative_to(root))
                    except ValueError:
                        raise ValueError(f"anchor must be inside run root: {anchor}") from None
                state_copy = dict(state, anchor=anchor)
                prompt = _initial_prompt(root, state_copy)
                if hasattr(contract, "system_prompt"):
                    system_prompt = contract.system_prompt()
                else:
                    system_prompt = getattr(
                        contract, "SYSTEM", getattr(contract, "SYSTEM_PROMPT", "")
                    )
                task_data = PainterAestheticData(
                    # Anchor counts may vary by state; a monotonic index keeps
                    # each native task unique in the Prime task buffer.
                    idx=len(tasks),
                    name=f"{state['id']}:anchor-{anchor_index}",
                    prompt=[vf.UserMessage(content=prompt)],
                    system_prompt=system_prompt,
                    root=str(root),
                    state=state_copy,
                    anchor=anchor,
                    max_turns=MAX_GENERATED_TURNS,
                )
                tasks.append(PainterAestheticTask(task_data, self.config.task))
        if not tasks:
            raise ValueError("aesthetic state manifest contains no tasks")
        return tasks


# Keep the activation scripts' taskset name stable while exposing the explicit
# env id used by Prime's native v1 server.
PainterRevisionTask = PainterAestheticTask
PainterRevisionTaskset = PainterAestheticTaskset
PainterRevisionConfig = PainterAestheticConfig
PainterRevisionEnv = PainterAestheticEnv
PainterRevisionEnvConfig = PainterAestheticEnvConfig

__all__ = [
    "PainterRevisionTaskset",
    "PainterRevisionConfig",
    "PainterRevisionEnv",
    "PainterRevisionEnvConfig",
    "PainterRevisionTask",
]
