"""Pure tests for the current-pin native painting environment."""

from __future__ import annotations

import asyncio
import base64
import importlib
import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Generic, TypeVar, get_args
from unittest.mock import patch

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class _TaskData(_Base):
    idx: int | None = None
    name: str | None = None
    prompt: object = None
    system_prompt: str | None = None


class _Task:
    def __init__(self, data, config=None):
        self.data = data
        self.config = config


class _TasksetConfig(_Base):
    id: str = ""
    system_prompt: object = None


class _Taskset:
    def __init__(self, config):
        self.config = config


class _EnvConfig(_Base):
    taskset: object = None


class _AgentConfig(_Base):
    pass


class _State:  # only used as a generic argument
    pass


class _Message:
    def __init__(self, content):
        self.content = content


class _Text:
    def __init__(self, text):
        self.text = text


class _ImageSource:
    def __init__(self, url):
        self.url = url


class _Image:
    def __init__(self, image_url):
        self.image_url = image_url


class _User:
    def __init__(self, content):
        self.content = content


class _VfEnv:
    def __init_subclass__(cls, **kwargs):
        return super().__init_subclass__()
    def __class_getitem__(cls, _item):
        return cls


_TaskDataT = TypeVar("_TaskDataT")
_TaskStateT = TypeVar("_TaskStateT")
_TaskConfigT = TypeVar("_TaskConfigT")


class _VfTask(Generic[_TaskDataT, _TaskStateT, _TaskConfigT]):
    def __init__(self, data, config=None):
        self.data, self.config = data, config

    @classmethod
    def data_type(cls):
        # Mirror the current VF concrete-generic lookup used by the node
        # serializer, enough for local task declaration tests.
        for base in getattr(cls, "__orig_bases__", ()):
            args = get_args(base)
            if args and isinstance(args[0], type):
                return args[0]
        return _TaskData


class _VfTaskset:
    def __class_getitem__(cls, _item):
        return cls
    def __init__(self, config):
        self.config = config


def _load_module():
    vf = types.ModuleType("verifiers.v1")
    vf.TaskData = _TaskData
    vf.Task = _VfTask
    vf.Taskset = _VfTaskset
    vf.TasksetConfig = _TasksetConfig
    vf.TaskConfig = _Base
    vf.Env = _VfEnv
    vf.EnvConfig = _EnvConfig
    vf.AgentConfig = _AgentConfig
    vf.State = _State
    vf.Agents = object
    vf.ImageUrlContentPart = _Image
    vf.ImageUrlSource = _ImageSource
    vf.TextContentPart = _Text
    vf.UserMessage = _User
    verifiers = types.ModuleType("verifiers")
    verifiers.v1 = vf
    with patch.dict(sys.modules, {"verifiers": verifiers, "verifiers.v1": vf}, clear=False):
        sys.modules.pop("painter.native_painting", None)
        return importlib.import_module("painter.native_painting")


class _Segment:
    def __init__(self, reply):
        self.last_reply = reply
        self.terminated = False


class _Trace:
    def __init__(self):
        self.info = {}


class _Interaction:
    def __init__(self, replies):
        self.replies = list(replies)
        self.trace = _Trace()
        self.messages = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def turn(self, message=None):
        self.messages.append(message)
        return _Segment(self.replies.pop(0))


class _Agent:
    def __init__(self, replies):
        self.interaction_obj = _Interaction(replies)

    def interaction(self, _task):
        return self.interaction_obj


class _Agents:
    def __init__(self, replies):
        self.agent = _Agent(replies)


class NativePaintingTests(unittest.TestCase):
    def test_text_reference_is_text_only_on_first_and_revision_turns(self):
        module = _load_module()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "brief.txt").write_text("Paint a pear left of a blue mug.\n")
            manifest = root / "tasks.json"
            manifest.write_text(json.dumps({"cases": [{
                "id": "text-pear", "source_group": "text-pear",
                "reference": "brief.txt", "reference_kind": "text",
            }]}))
            task = list(module.NativePaintingTaskset(module.NativePaintingConfig(
                root=str(root), manifest=manifest.name, expected_count=1, max_turns=2,
            )).load())[0]
            self.assertEqual(task.data.reference_kind, "text")
            self.assertIn("PAINTING BRIEF", task.data.system_prompt)
            self.assertNotIn("reference", task.data.system_prompt.lower())
            self.assertIn("exposes `brush` as a global", task.data.system_prompt)
            self.assertTrue(all(not hasattr(part, "image_url")
                                for part in task.data.prompt[0].content))
            agents = _Agents(["FINISHED", "FINISHED"])
            asyncio.run(module.NativePaintingEnv.__dict__["run"](_DummyEnv(), task, agents))
            followup = agents.agent.interaction_obj.messages[1][0].content
            self.assertTrue(all(not hasattr(part, "image_url") for part in followup))
            self.assertIn("pear left of a blue mug", " ".join(part.text for part in followup))

    def test_program_errors_get_specific_repair_feedback(self):
        module = _load_module()
        cases = (
            (["SyntaxError: unexpected token ')'"], "Fix the JavaScript syntax"),
            (["TypeError: Cannot redefine property: ellipse"], "Do not redefine p5 globals"),
            (["TypeError: brush.ellipse is not a function"], "has no brush.ellipse() method"),
            (["ReferenceError: v is not defined"], "Define every variable"),
            (["TypeError: a is not iterable"], "brush.polygon point list"),
        )
        for errors, expected in cases:
            with self.subTest(errors=errors):
                feedback = module._program_failure_feedback(errors)
                self.assertIn(expected, feedback)
                self.assertIn(errors[0], feedback)

    def test_image_preserves_bytes_and_uses_png_or_jpeg_mime(self):
        module = _load_module()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            png = root / "reference.png"
            png_bytes = b"\x89PNG\r\n\x1a\nminimal-png"
            png.write_bytes(png_bytes)
            jpeg = root / "reference.jpg"
            jpeg_bytes = b"\xff\xd8\xff\xe0minimal-jpeg"
            jpeg.write_bytes(jpeg_bytes)
            for path, expected_mime, expected_bytes in (
                (png, "image/png", png_bytes),
                (jpeg, "image/jpeg", jpeg_bytes),
            ):
                part = module._image(path)
                url = part.image_url.url
                prefix, encoded = url.split(",", 1)
                self.assertEqual(prefix, f"data:{expected_mime};base64")
                self.assertEqual(base64.b64decode(encoded), expected_bytes)

    def test_taskset_loads_cases_and_rejects_duplicate_groups(self):
        module = _load_module()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "reference.png"
            image.write_bytes(b"png")
            manifest = root / "tasks.json"
            manifest.write_text(json.dumps({"cases": [{"id": "v2-box-007", "source_group": "v2-box-007", "reference_png": "reference.png"}]}))
            config = module.NativePaintingConfig(root=str(root), manifest="tasks.json", expected_count=1)
            tasks = list(module.NativePaintingTaskset(config).load())
            self.assertEqual(tasks[0].data.mode, "fresh")
            self.assertEqual(tasks[0].data.source_group, "v2-box-007")
            self.assertEqual(len(tasks[0].data.prompt), 1)
            manifest.write_text(json.dumps({"cases": [
                {"id": "a", "source_group": "same", "reference_png": "reference.png"},
                {"id": "b", "source_group": "same", "reference_png": "reference.png"},
            ]}))
            with self.assertRaises(ValueError):
                list(module.NativePaintingTaskset(module.NativePaintingConfig(root=str(root), manifest="tasks.json", expected_count=2)).load())

    def test_taskset_loads_two_jsonl_rows_with_declared_count(self):
        module = _load_module()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "reference.jpg").write_bytes(b"\xff\xd8\xff\xe0reference")
            manifest = root / "scaled.jsonl"
            manifest.write_text("\n".join(json.dumps({
                "id": f"scaled-{index}",
                "source_group": f"group-{index}",
                "reference": "reference.jpg",
            }) for index in range(2)) + "\n")
            config = module.NativePaintingConfig(
                root=str(root), manifest=manifest.name, expected_count=2
            )
            tasks = list(module.NativePaintingTaskset(config).load())
            self.assertEqual(len(tasks), 2)
            self.assertEqual([task.data.task_id for task in tasks], ["scaled-0", "scaled-1"])

    def test_taskset_preserves_optional_correction_start(self):
        module = _load_module()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "reference.png").write_bytes(b"ref")
            (root / "start.png").write_bytes(b"start")
            (root / "start.js").write_text("function draw(){}")
            manifest = root / "tasks.json"
            manifest.write_text(json.dumps({"cases": [{
                "id": "repair-1", "source_group": "box", "reference_png": "reference.png",
                "mode": "correction", "initial_canvas": "start.png", "initial_program": "start.js",
            }]}))
            task = list(module.NativePaintingTaskset(module.NativePaintingConfig(root=str(root), manifest="tasks.json", expected_count=1)).load())[0]
            self.assertEqual(task.data.mode, "correction")
            self.assertEqual(task.data.initial_program, "function draw(){}")

    def test_finish_without_canvas_is_recoverable_and_history_is_multimodal(self):
        module = _load_module()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "reference.png"
            reference.write_bytes(b"ref")
            data = module.NativePaintingData(
                root=str(root), task_id="v2-box-007", source_group="box", reference=str(reference),
                reference_sha256=module.sha(reference), max_turns=2,
            )
            task = module.NativePaintingTask(data)
            agents = _Agents(["FINISHED", "```javascript\nfunction draw(){}\n```"])
            render = {"valid": True, "canvas": str(root / "canvas.png"), "program": "function draw(){}", "no_change": False}
            with patch.object(module, "render_action", return_value=render):
                asyncio.run(module.NativePaintingEnv.__dict__["run"](_DummyEnv(), task, agents))
            self.assertTrue(agents.agent.interaction_obj.trace.info["final_canvas_valid"])
            self.assertEqual(len(agents.agent.interaction_obj.messages), 2)
            payload_text = " ".join(part.text for part in agents.agent.interaction_obj.messages[1][0].content if hasattr(part, "text"))
            self.assertIn(module.FINISH_WITHOUT_CANVAS, payload_text)

    def test_invalid_render_preserves_canvas_and_next_turn_has_history(self):
        module = _load_module()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "reference.png"
            reference.write_bytes(b"ref")
            canvas = root / "canvas.png"
            canvas.write_bytes(b"canvas")
            data = module.NativePaintingData(
                root=str(root), task_id="v2-box-007", source_group="box", reference=str(reference),
                reference_sha256=module.sha(reference), max_turns=3,
            )
            task = module.NativePaintingTask(data)
            agents = _Agents(["```javascript\none();\n```", "```javascript\ntwo();\n```", "FINISHED"])
            results = iter([
                {"valid": False, "canvas": None, "error": "syntax", "failure_kind": "program"},
                {"valid": True, "canvas": str(canvas), "program": "two();", "no_change": False},
            ])
            with patch.object(module, "render_action", side_effect=lambda **_: next(results)):
                asyncio.run(module.NativePaintingEnv.__dict__["run"](_DummyEnv(), task, agents))
            info = agents.agent.interaction_obj.trace.info
            self.assertEqual(info["final_canvas"], str(canvas))
            self.assertTrue(info["final_canvas_valid"])
            self.assertEqual(len(agents.agent.interaction_obj.messages), 3)
            payload_text = " ".join(part.text for part in agents.agent.interaction_obj.messages[2][0].content if hasattr(part, "text"))
            self.assertIn("CURRENT CANVAS", payload_text)


class _DummyEnv:
    pass


if __name__ == "__main__":
    unittest.main()
