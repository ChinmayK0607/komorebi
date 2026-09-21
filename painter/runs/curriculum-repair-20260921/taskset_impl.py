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
import importlib.util
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

from contract import FINISH_WITHOUT_CANVAS, NEXT_VERSION, NO_CHANGE, SYSTEM


RUN_DIR = Path(__file__).resolve().parent
MAX_GENERATED_TURNS = 3
CURRICULUM_MODES = frozenset({"fresh", "correction", "both"})
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
        return SYSTEM

    def initial_prompt(self, *, state, reference, current=None, program=None, fresh=False):
        parts = [
            vf.TextContentPart(text="REFERENCE"),
            reference,
        ]
        if not fresh and current is not None:
            parts.extend([
                vf.TextContentPart(text="CURRENT CANVAS"),
                current,
            ])
        if not fresh and program is not None:
            parts.append(vf.TextContentPart(text="Current program:\n" + program))
        parts.append(vf.TextContentPart(text=NEXT_VERSION))
        return parts

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
                  program: str | None = None, feedback: str | None = None,
                  fresh: bool = False) -> list[Any]:
    """Build a correction or fresh-start multimodal user message."""
    contract = _contract(root)
    reference = root / state["reference"]
    if hasattr(contract, "observation_prompt"):
        text = contract.observation_prompt(
            state=state, feedback=feedback, has_program=program is not None
        )
    else:
        text = feedback or "Inspect the reference and current canvas, then paint the next version."
    parts: list[Any] = [
        vf.TextContentPart(text="REFERENCE"),
        _image(reference),
    ]
    if not fresh:
        current = root / (state["canvas"] if observation is None else observation)
        parts.extend([
            vf.TextContentPart(text="CURRENT CANVAS"),
            _image(current),
        ])
    if program is not None and not fresh:
        parts.append(vf.TextContentPart(text="Current program:\n" + program))
    parts.append(vf.TextContentPart(text=text))
    return parts


def _initial_prompt(root: Path, state: dict[str, Any], *, fresh: bool = False) -> list[Any]:
    contract = _contract(root)
    if hasattr(contract, "initial_prompt"):
        return contract.initial_prompt(
            state=state,
            reference=_image(root / state["reference"]),
            current=None if fresh else _image(root / state["canvas"]),
            program=None if fresh else (root / state["program"]).read_text(),
            fresh=fresh,
        )
    return _prompt_parts(
        root,
        state,
        program=None if fresh else (root / state["program"]).read_text(),
        feedback=NEXT_VERSION,
        fresh=fresh,
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
    mode: str = "correction"
    max_turns: int = MAX_GENERATED_TURNS


class PainterAestheticConfig(vf.TasksetConfig):
    root: str
    curriculum_path: str | None = None
    curriculum_mode: str = "both"
    max_turns: int | None = None


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
        mode = getattr(task.data, "mode", state.get("mode", "correction"))
        if mode not in {"fresh", "correction"}:
            raise ValueError(f"unsupported curriculum mode: {mode}")
        current_canvas = None if mode == "fresh" else root / state["canvas"]
        current_program = None if mode == "fresh" else (root / state["program"]).read_text()
        folder = root / "rl-rollouts" / task.data.state["id"] / str(uuid.uuid4())
        folder.mkdir(parents=True, exist_ok=True)
        trace_id = None

        async with agents.agent.interaction(task) as interaction:
            trace_id = interaction.trace.id
            # Make the immutable starting canvas explicit for the score hook.
            interaction.trace.info.update(
                initial_mode=mode,
                initial_canvas=(str(current_canvas) if current_canvas is not None else None),
                anchor=task.data.anchor,
                turn_cap=task.data.max_turns,
            )
            if current_canvas is not None:
                interaction.trace.info["initial_canvas_sha256"] = sha(current_canvas)
            segment = await interaction.turn()
            for turn in range(1, task.data.max_turns + 1):
                if segment.terminated:
                    break
                reply = segment.last_reply
                if _is_finish(reply):
                    if current_canvas is None:
                        # This is a recoverable model mistake.  Keep the
                        # native interaction alive so a remaining turn can
                        # submit code; only cap exhaustion becomes the
                        # no-valid-canvas terminal outcome.
                        interaction.trace.info.setdefault("invalid_finish_turns", []).append(turn)
                        if turn >= task.data.max_turns:
                            interaction.trace.info.update(
                                failure_kind="no_valid_canvas_at_cap",
                                no_valid_canvas_at_cap=True,
                                final_canvas=None,
                                final_canvas_valid=False,
                                operational_cutoff=True,
                                finish_turn=None,
                            )
                            break
                        parts = _prompt_parts(
                            root,
                            state,
                            observation=None,
                            program=None,
                            feedback=FINISH_WITHOUT_CANVAS,
                            fresh=True,
                        )
                        segment = await interaction.turn([vf.UserMessage(content=parts)])
                        continue
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
                        "initial_canvas": (str(current_canvas) if current_canvas is not None else None),
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
                    feedback = NO_CHANGE if result.get("no_change") else NEXT_VERSION
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
                        final_canvas=(str(current_canvas) if current_canvas is not None else None),
                        final_canvas_valid=(current_canvas is not None and current_canvas.is_file()),
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
                    fresh=current_canvas is None,
                )
                segment = await interaction.turn([vf.UserMessage(content=parts)])

            if not interaction.trace.info.get("final_canvas"):
                interaction.trace.info.update(
                    final_canvas=(str(current_canvas) if current_canvas is not None else None),
                    final_canvas_valid=(current_canvas is not None and current_canvas.is_file()),
                    operational_cutoff=True,
                )
                if current_canvas is None:
                    interaction.trace.info.setdefault("failure_kind", "no_valid_canvas_at_cap")
                    interaction.trace.info["no_valid_canvas_at_cap"] = True


def _positive_turn_cap(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _resolve_config_path(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path


def _runtime_contract_snapshot(root: Path) -> dict[str, Any]:
    path = root / "contract.py"
    if not path.is_file():
        raise FileNotFoundError(f"prepared curriculum requires run contract: {path}")
    spec = importlib.util.spec_from_file_location("curriculum_runtime_contract", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load run contract: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("SYSTEM", "NEXT_VERSION", "NO_CHANGE", "FINISH_WITHOUT_CANVAS"):
        if globals().get(name) != getattr(module, name):
            raise ValueError(
                f"taskset imported {name} from a different contract than {path}"
            )
    return {
        "path": path.name,
        "file_sha256": sha(path),
        "identity": module.identity(),
    }


def _load_prepared_curriculum(root: Path, config: PainterAestheticConfig):
    """Load the reviewed order artifact when one is present.

    The prepared artifact is the source of task order in a repair run.  The
    legacy state manifest remains a deliberate fallback for compatibility with
    the frozen aesthetic run, but a new run can select a plan explicitly via
    ``curriculum_path`` and ``curriculum_mode``.
    """
    requested = getattr(config, "curriculum_mode", "both")
    if requested not in CURRICULUM_MODES:
        raise ValueError(f"unsupported curriculum mode: {requested!r}")
    configured = _resolve_config_path(root, getattr(config, "curriculum_path", None))
    candidates = [configured] if configured is not None else [
        root / "curriculum-plan.json",
        root / "fresh-curriculum.json",
        root / "correction-curriculum.json",
    ]
    path = next((item for item in candidates if item is not None and item.is_file()), None)
    if path is None:
        if configured is not None:
            raise FileNotFoundError(f"configured curriculum_path does not exist: {configured}")
        return None
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("prepared curriculum must be a JSON object")
    expected_contract = value.get("contract_snapshot")
    if expected_contract is not None:
        actual_contract = _runtime_contract_snapshot(root)
        if expected_contract != actual_contract:
            raise ValueError("prepared curriculum contract snapshot does not match run contract.py")
    if "modes" in value:
        modes = value["modes"]
        if not isinstance(modes, dict):
            raise ValueError("prepared curriculum modes must be an object")
        selected_modes = ("fresh", "correction") if requested == "both" else (requested,)
        rows: list[dict[str, Any]] = []
        selected_cap = value.get("max_turns")
        for mode in selected_modes:
            section = modes.get(mode)
            if not isinstance(section, dict) or not isinstance(section.get("tasks"), list):
                raise ValueError(f"prepared curriculum is missing {mode} tasks")
            rows.extend(dict(row, mode=mode) for row in section["tasks"])
            if selected_cap is None:
                selected_cap = section.get("max_turns")
    else:
        rows = value.get("tasks")
        if not isinstance(rows, list) or not rows:
            raise ValueError("prepared curriculum must contain a non-empty tasks list")
        if requested != "both":
            rows = [row for row in rows if isinstance(row, dict) and row.get("mode") == requested]
        selected_cap = value.get("max_turns")
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("prepared curriculum selected no valid task rows")
    return path, rows, selected_cap


def _prepared_state(row: dict[str, Any], index: int) -> tuple[dict[str, Any], str]:
    state = dict(row.get("state") or {}) if isinstance(row.get("state"), dict) else {}
    state.update({key: value for key, value in row.items() if key != "state"})
    state["id"] = row.get("task_id") or row.get("id") or f"curriculum-{index:04d}"
    state["reference"] = row.get("reference") or row.get("reference_png") or state.get("reference")
    mode = row.get("mode") or state.get("mode")
    if mode not in {"fresh", "correction"}:
        raise ValueError(f"prepared task {state['id']} lacks fresh/correction mode")
    state["mode"] = mode
    if not isinstance(state.get("reference"), str) or not state["reference"]:
        raise ValueError(f"prepared task {state['id']} lacks a reference path")
    canvas = row.get("initial_canvas") or row.get("canvas") or state.get("canvas")
    program = row.get("initial_program") or row.get("program") or state.get("program")
    if canvas is not None:
        state["canvas"] = canvas
    if program is not None:
        state["program"] = program
    if mode == "correction" and (not isinstance(canvas, str) or not isinstance(program, str)):
        raise ValueError(f"correction task {state['id']} lacks an initial canvas/program")
    anchor = row.get("anchor") or row.get("anchor_canvas") or state.get("anchor")
    if not isinstance(anchor, str) or not anchor:
        raise ValueError(
            f"prepared task {state['id']} lacks an anchor path; pairwise reward requires a reviewed anchor"
        )
    anchor_path = Path(anchor)
    if anchor_path.is_absolute():
        state["anchor"] = str(anchor_path)
    else:
        state["anchor"] = anchor
    return state, state["anchor"]


class PainterAestheticTaskset(vf.Taskset[PainterAestheticTask, PainterAestheticConfig]):
    """Load prepared same-reference states and optionally round-robin anchors."""

    def load(self) -> list[PainterAestheticTask]:
        root = Path(self.config.root).resolve()
        prepared = _load_prepared_curriculum(root, self.config)
        if prepared is not None:
            curriculum_path, curriculum_rows, curriculum_cap = prepared
            configured_cap = getattr(self.config, "max_turns", None)
            default_cap = configured_cap if configured_cap is not None else curriculum_cap
            default_cap = MAX_GENERATED_TURNS if default_cap is None else _positive_turn_cap(
                default_cap, label="curriculum max_turns"
            )
            prepared_items = []
            for index, raw in enumerate(curriculum_rows):
                state, anchor = _prepared_state(raw, index)
                cap = _positive_turn_cap(
                    raw.get("max_turns", default_cap), label=f"task {state['id']} max_turns"
                )
                prepared_items.append((state, anchor, state["mode"], cap))
            return self._build_tasks(root, prepared_items, source=str(curriculum_path))

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
        manifest = json.loads(states_path.read_text())
        states = manifest.get("training_states", manifest) if isinstance(manifest, dict) else manifest
        if not isinstance(states, list):
            raise ValueError("aesthetic state manifest must contain training_states")
        manifest_cap = manifest.get("max_turns") if isinstance(manifest, dict) else None
        configured_cap = getattr(self.config, "max_turns", None)
        default_cap = configured_cap if configured_cap is not None else manifest_cap
        default_cap = MAX_GENERATED_TURNS if default_cap is None else _positive_turn_cap(
            default_cap, label="state manifest max_turns"
        )
        state_items = []
        for index, raw in enumerate(states):
            state = dict(raw)
            mode = state.get("mode", "correction")
            if mode not in {"fresh", "correction"}:
                raise ValueError(f"state {state.get('id', index)} has unsupported mode {mode!r}")
            cap = _positive_turn_cap(state.get("max_turns", default_cap), label=f"state {state.get('id', index)} max_turns")
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
                state_items.append((state_copy, anchor, mode, cap))
        return self._build_tasks(root, state_items, source=str(states_path))

    def _build_tasks(self, root: Path, state_items, *, source: str) -> list[PainterAestheticTask]:
        """Build native tasks from either a prepared order or legacy states."""
        contract = _contract(root)
        tasks: list[PainterAestheticTask] = []
        for task_index, (state, anchor, mode, cap) in enumerate(state_items):
            prompt = _initial_prompt(root, state, fresh=mode == "fresh")
            if hasattr(contract, "system_prompt"):
                system_prompt = contract.system_prompt()
            else:
                system_prompt = getattr(
                    contract, "SYSTEM", getattr(contract, "SYSTEM_PROMPT", "")
                )
            anchor_label = state.get("anchor_id", "anchor-0")
            task_data = PainterAestheticData(
                # Anchor counts may vary by state; a monotonic index keeps
                # each native task unique in the Prime task buffer.
                idx=task_index,
                name=f"{state['id']}:{anchor_label}",
                prompt=[vf.UserMessage(content=prompt)],
                system_prompt=system_prompt,
                root=str(root),
                state=state,
                anchor=anchor,
                mode=mode,
                max_turns=cap,
            )
            tasks.append(PainterAestheticTask(task_data, self.config.task))
        if not tasks:
            raise ValueError(f"curriculum source contains no tasks: {source}")
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
