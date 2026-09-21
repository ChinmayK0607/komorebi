"""One active native painting environment for matched SFT/RL evaluation.

This module targets the pinned current Verifiers API: a typed ``Taskset`` and
an ``Env.run(task, agents)`` interaction loop. The loop is the same full-history
``Agent.interaction`` path used by the repaired curriculum; only the task source
and the offline/no-reward policy differ. Rendering is the only operation between
model turns. Visual scoring happens later.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import uuid
from asyncio import Semaphore, to_thread
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import verifiers.v1 as vf

try:
    from painter.contract import FINISH_WITHOUT_CANVAS, NEXT_VERSION, NO_CHANGE, SYSTEM
except ImportError:
    from contract import FINISH_WITHOUT_CANVAS, NEXT_VERSION, NO_CHANGE, SYSTEM

try:
    from painter.native_painting_renderer import render_action
except ImportError:
    from native_painting_renderer import render_action


PROTOCOL_VERSION = "native-painting-eval-v1"
DEFAULT_SEED = 918200
DEFAULT_MAX_TURNS = 6
DEFAULT_MAX_OUTPUT_TOKENS = 8192
DEFAULT_CONTEXT_LENGTH = 32768
RENDER_LIMIT = Semaphore(8)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _image(path: Path) -> vf.ImageUrlContentPart:
    if not path.is_file():
        raise FileNotFoundError(f"image is missing: {path}")
    payload = path.read_bytes()
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif payload.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    else:
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(path.suffix.lower())
        if mime is None:
            raise ValueError(f"unsupported image type: {path}")
    encoded = base64.b64encode(payload).decode("ascii")
    return vf.ImageUrlContentPart(
        image_url=vf.ImageUrlSource(url=f"data:{mime};base64," + encoded)
    )


def _text(value: str) -> vf.TextContentPart:
    return vf.TextContentPart(text=value)


def _rows(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        values = raw
    elif isinstance(raw, dict):
        values = raw.get("cases") or raw.get("eval") or raw.get("tasks") or raw.get("evaluation") or []
    else:
        raise TypeError("evaluation manifest must be a list or contain cases/tasks")
    if not isinstance(values, list) or not all(isinstance(row, dict) for row in values):
        raise TypeError("evaluation manifest rows must be objects")
    return [dict(row) for row in values]


def _read_manifest_rows(path: Path) -> list[dict[str, Any]]:
    """Read the JSON and JSONL manifest forms accepted by the taskset."""

    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"manifest line {line_number} must be an object")
            rows.append(value)
        return _rows(rows)
    return _rows(json.loads(path.read_text()))


def _is_finish(reply: str) -> bool:
    return "```" not in reply and bool(re.search(r"(?m)^\s*FINISHED\s*$", reply))


class NativePaintingData(vf.TaskData):
    root: str
    task_id: str
    source_group: str
    family: str | None = None
    reference: str
    reference_sha256: str
    policy_label: str = "unknown"
    output_root: str = "evaluation-rollouts"
    renderer: str | None = None
    renderer_python: str | None = None
    max_turns: int = DEFAULT_MAX_TURNS
    mode: str = "fresh"
    initial_canvas: str | None = None
    initial_program: str | None = None
    protocol_version: str = PROTOCOL_VERSION


class NativePaintingTask(vf.Task[NativePaintingData, vf.State, vf.TaskConfig]):
    """No reward hooks: final image review is deliberately offline."""


class NativePaintingConfig(vf.TasksetConfig):
    root: str = "."
    manifest: str = "native-eval-tasks.json"
    expected_count: int = 12
    max_turns: int = DEFAULT_MAX_TURNS
    policy_label: str = "unknown"
    output_root: str = "evaluation-rollouts"
    renderer: str | None = None
    renderer_python: str | None = None


class NativePaintingTaskset(vf.Taskset[NativePaintingTask, NativePaintingConfig]):
    def load(self) -> Iterable[NativePaintingTask]:
        root = Path(self.config.root)
        manifest_path = _resolve(root, self.config.manifest)
        rows = _read_manifest_rows(manifest_path)
        if len(rows) != self.config.expected_count:
            raise ValueError(f"expected {self.config.expected_count} evaluation rows, found {len(rows)}")
        groups: set[str] = set()
        for index, row in enumerate(rows):
            task_id = row.get("id") or row.get("task_id") or row.get("case_id")
            source_group = row.get("source_group") or row.get("group") or task_id
            reference = row.get("reference_png") or row.get("reference")
            if not all(isinstance(value, str) and value for value in (task_id, source_group, reference)):
                raise ValueError("every evaluation row needs id, source_group, and reference_png/reference")
            if source_group in groups:
                raise ValueError(f"duplicate source_group: {source_group}")
            groups.add(source_group)
            # The controlled-comparison manifest declares paths relative to its
            # own directory; retain a root fallback for compact node manifests.
            reference_path = Path(reference) if Path(reference).is_absolute() else manifest_path.parent / reference
            if not reference_path.is_file():
                reference_path = _resolve(root, reference)
            mode = str(row.get("mode", "fresh"))
            if mode not in {"fresh", "correction"}:
                raise ValueError(f"unsupported evaluation mode for {task_id}: {mode}")
            initial_canvas_value = row.get("initial_canvas") or row.get("canvas")
            initial_program_value = row.get("initial_program") or row.get("program")
            initial_canvas_path = (Path(initial_canvas_value) if Path(str(initial_canvas_value)).is_absolute() else manifest_path.parent / str(initial_canvas_value)) if initial_canvas_value else None
            if initial_canvas_path is not None and not initial_canvas_path.is_file():
                initial_canvas_path = _resolve(root, str(initial_canvas_value))
            if initial_program_value:
                program_path = Path(str(initial_program_value)) if Path(str(initial_program_value)).is_absolute() else manifest_path.parent / str(initial_program_value)
                if not program_path.is_file():
                    program_path = _resolve(root, str(initial_program_value))
                if program_path.is_file():
                    initial_program_value = program_path.read_text()
            if mode == "correction" and (initial_canvas_path is None or not initial_program_value):
                raise ValueError(f"correction row {task_id} needs initial_canvas and initial_program")
            if initial_canvas_path is not None and not initial_canvas_path.is_file():
                raise FileNotFoundError(f"initial canvas is missing: {initial_canvas_path}")
            expected_sha = row.get("reference_sha256")
            if expected_sha is None and isinstance(row.get("sha256"), dict):
                expected_sha = row["sha256"].get("png")
            elif expected_sha is None:
                expected_sha = row.get("sha256")
            actual_sha = sha(reference_path)
            if expected_sha is not None and expected_sha != actual_sha:
                raise ValueError(f"reference hash mismatch for {task_id}: {actual_sha}")
            prompt = [_text("REFERENCE"), _image(reference_path)]
            if mode == "correction":
                prompt.extend([_text("CURRENT CANVAS"), _image(initial_canvas_path), _text("Current program:\n" + str(initial_program_value))])
            prompt.append(_text(NEXT_VERSION))
            data = NativePaintingData(
                idx=index,
                name=str(task_id),
                prompt=[vf.UserMessage(content=prompt)],
                system_prompt=SYSTEM,
                root=str(root),
                task_id=str(task_id),
                source_group=str(source_group),
                family=row.get("family"),
                reference=str(reference_path),
                reference_sha256=actual_sha,
                policy_label=self.config.policy_label,
                output_root=self.config.output_root,
                renderer=self.config.renderer,
                renderer_python=self.config.renderer_python,
                max_turns=int(row.get("max_turns", self.config.max_turns)),
                mode=mode,
                initial_canvas=str(initial_canvas_path) if initial_canvas_path is not None else None,
                initial_program=str(initial_program_value) if initial_program_value is not None else None,
            )
            yield NativePaintingTask(data)


class NativePaintingEnvConfig(vf.EnvConfig):
    agent: vf.AgentConfig = vf.AgentConfig()


class NativePaintingEnv(vf.Env[NativePaintingEnvConfig]):
    """Reference-only fresh painting with complete native interaction history."""

    async def run(self, task: NativePaintingTask, agents: vf.Agents) -> None:
        data = task.data
        if data.mode not in {"fresh", "correction"}:
            raise ValueError(f"unsupported evaluation mode: {data.mode}")
        root = Path(data.root)
        reference = Path(data.reference)
        folder = root / data.output_root / data.policy_label / data.task_id / str(uuid.uuid4())
        folder.mkdir(parents=True, exist_ok=True)
        current_canvas: Path | None = Path(data.initial_canvas) if data.initial_canvas else None
        current_program: str | None = data.initial_program
        info: dict[str, Any] = {
            "protocol_version": PROTOCOL_VERSION,
            "mode": data.mode,
            "task_id": data.task_id,
            "source_group": data.source_group,
            "family": data.family,
            "reference": str(reference),
            "reference_sha256": data.reference_sha256,
            "initial_canvas": str(current_canvas) if current_canvas else None,
            "turn_cap": data.max_turns,
            "turns": [],
            "final_canvas": None,
            "final_canvas_valid": False,
        }
        if current_canvas is not None:
            info["initial_canvas_sha256"] = sha(current_canvas)
        async with agents.agent.interaction(task) as interaction:
            interaction.trace.info.update(info)
            segment = await interaction.turn()
            for turn in range(1, data.max_turns + 1):
                if segment.terminated:
                    break
                reply = segment.last_reply
                turn_info: dict[str, Any] = {"turn": turn, "reply": reply}
                if _is_finish(reply):
                    if current_canvas is None:
                        turn_info.update(action="finish", valid=False, failure_kind="finish_without_canvas")
                        interaction.trace.info.setdefault("invalid_finish_turns", []).append(turn)
                        feedback = FINISH_WITHOUT_CANVAS
                    else:
                        turn_info.update(action="finish", valid=True)
                        interaction.trace.info.update(
                            finished=True,
                            finish_turn=turn,
                            final_canvas=str(current_canvas),
                            final_canvas_valid=True,
                            operational_cutoff=False,
                        )
                        interaction.trace.info.setdefault("turns", []).append(turn_info)
                        break
                else:
                    try:
                        async with _render_slot():
                            result = await to_thread(
                                render_action,
                                root=root,
                                folder=folder / f"turn-{turn}",
                                reply=reply,
                                initial_canvas=current_canvas,
                                initial_program=current_program,
                                renderer=_resolve(root, data.renderer) if data.renderer else None,
                                renderer_python=_resolve(root, data.renderer_python) if data.renderer_python else None,
                            )
                    except Exception as exc:
                        interaction.trace.info["service_failure"] = f"renderer service failure: {type(exc).__name__}: {exc}"
                        turn_info.update(action="paint", valid=False, failure_kind="service")
                        interaction.trace.info.setdefault("turns", []).append(turn_info)
                        break
                    turn_info.update({key: value for key, value in result.items() if key != "program"})
                    if result.get("valid"):
                        current_canvas = Path(str(result["canvas"])) if result.get("canvas") else current_canvas
                        current_program = result.get("program") or current_program
                        feedback = NO_CHANGE if result.get("no_change") else NEXT_VERSION
                    else:
                        feedback = "The sketch did not produce a valid canvas. Fix the reported program error and return a complete javascript code block."
                        if result.get("error"):
                            feedback += "\nRenderer feedback: " + str(result["error"])
                interaction.trace.info.setdefault("turns", []).append(turn_info)
                if turn >= data.max_turns:
                    break
                parts: list[Any] = [_text("REFERENCE"), _image(reference)]
                if current_canvas is not None and current_canvas.is_file():
                    parts.extend([_text("CURRENT CANVAS"), _image(current_canvas)])
                if current_program is not None:
                    parts.append(_text("Current program:\n" + current_program))
                parts.append(_text(feedback))
                segment = await interaction.turn([vf.UserMessage(content=parts)])
            if not interaction.trace.info.get("final_canvas_valid") and current_canvas is not None:
                interaction.trace.info.update(final_canvas=str(current_canvas), final_canvas_valid=True, operational_cutoff=True)
            if not interaction.trace.info.get("final_canvas_valid") and not interaction.trace.info.get("service_failure"):
                interaction.trace.info.update(failure_kind="no_valid_canvas_at_cap", no_valid_canvas_at_cap=True, operational_cutoff=True)


class _render_slot:
    async def __aenter__(self):
        await RENDER_LIMIT.acquire()

    async def __aexit__(self, *_exc):
        RENDER_LIMIT.release()


__all__ = ["NativePaintingTaskset", "NativePaintingEnv", "NativePaintingConfig", "NativePaintingEnvConfig"]
