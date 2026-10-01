"""Pure tests for the current-pin native painting environment."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib
import json
import re
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
    def test_renderer_worker_limit_is_opt_in_and_validated(self):
        module = _load_module()
        self.assertEqual(module.render_worker_limit({}), module.DEFAULT_RENDER_WORKERS)
        self.assertEqual(module.render_worker_limit({module.RENDER_WORKERS_ENV: "2"}), 2)
        with self.assertRaises(ValueError):
            module.render_worker_limit({module.RENDER_WORKERS_ENV: "0"})
        with self.assertRaises(ValueError):
            module.render_worker_limit({module.RENDER_WORKERS_ENV: "cpu"})

    def test_finish_parser_accepts_reason_but_never_code_with_finished(self):
        module = _load_module()
        self.assertTrue(module._is_finish("The observed canvas is complete.\nFINISHED"))
        self.assertFalse(module._is_finish("FINISHED in prose"))
        self.assertFalse(module._is_finish("```javascript\nfunction draw(){}\nFINISHED\n```"))

    def test_inspect_choice_feedback_is_opt_in_and_trace_versioned(self):
        module = _load_module()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            brief = root / "brief.txt"
            brief.write_text("Paint a pear left of a blue mug.\n")
            manifest = root / "tasks.json"
            manifest.write_text(json.dumps({"cases": [{
                "id": "text-pear", "source_group": "text-pear", "reference": brief.name,
                "reference_kind": "text",
            }]}))
            def load_task():
                return list(module.NativePaintingTaskset(module.NativePaintingConfig(
                    root=str(root), manifest=manifest.name, expected_count=1, max_turns=2,
                )).load())[0]

            with patch.dict("os.environ", {}, clear=True):
                task = load_task()
                initial_text = " ".join(part.text for part in task.data.prompt[0].content if hasattr(part, "text"))
                self.assertIn(module.TEXT_NEXT_VERSION, initial_text)
                agents = _Agents(["FINISHED", "FINISHED"])
                asyncio.run(module.NativePaintingEnv.__dict__["run"](_DummyEnv(), task, agents))
            self.assertEqual(
                agents.agent.interaction_obj.trace.info["feedback_version"],
                module.LEGACY_FEEDBACK_VERSION,
            )

            with patch.dict("os.environ", {module.INSPECT_CHOICE_FEEDBACK_ENV: "1"}, clear=True):
                task = load_task()
                initial_text = " ".join(part.text for part in task.data.prompt[0].content if hasattr(part, "text"))
                self.assertIn("No CURRENT CANVAS exists yet", initial_text)
                self.assertIn("FINISHED is invalid", initial_text)
                self.assertIn("Choose exactly one response form", module._text_initial_feedback(has_canvas=True))
                agents = _Agents(["FINISHED", "FINISHED"])
                asyncio.run(module.NativePaintingEnv.__dict__["run"](_DummyEnv(), task, agents))
            self.assertEqual(
                agents.agent.interaction_obj.trace.info["feedback_version"],
                module.INSPECT_CHOICE_FEEDBACK_VERSION,
            )
            followup_text = " ".join(
                part.text
                for part in agents.agent.interaction_obj.messages[1][0].content
                if hasattr(part, "text")
            )
            self.assertIn("No CURRENT CANVAS exists yet", followup_text)
            self.assertIn("FINISHED is invalid", followup_text)

            image = root / "reference.png"
            image.write_bytes(b"\x89PNG\r\n\x1a\nreference")
            image_data = module.NativePaintingData(
                idx=0, name="image-pear", source_group="image-pear", root=str(root),
                task_id="image-pear", reference=str(image), reference_sha256=module.sha(image),
                reference_kind="image", max_turns=2,
            )
            with patch.dict("os.environ", {module.INSPECT_CHOICE_FEEDBACK_ENV: "1"}, clear=True):
                image_agents = _Agents(["FINISHED", "FINISHED"])
                asyncio.run(module.NativePaintingEnv.__dict__["run"](
                    _DummyEnv(), module.NativePaintingTask(image_data), image_agents
                ))
            self.assertEqual(
                image_agents.agent.interaction_obj.trace.info["feedback_version"],
                module.LEGACY_FEEDBACK_VERSION,
            )

    def test_inspect_choice_feedback_retains_canvas_as_explicit_finish_option(self):
        module = _load_module()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            brief = root / "brief.txt"
            brief.write_text("Paint a pear beside a blue mug.\n")
            canvas = root / "canvas.png"
            canvas.write_bytes(b"canvas")
            data = module.NativePaintingData(
                idx=0, name="text-pear", source_group="text-pear", root=str(root),
                task_id="text-pear", reference=str(brief), reference_sha256=module.sha(brief),
                reference_kind="text", max_turns=3,
            )
            agents = _Agents([
                "```javascript\nfunction draw(){}\n```",
                "```javascript\nbroken();\n```",
                "FINISHED",
            ])
            results = iter([
                {"valid": True, "canvas": str(canvas), "program": "function draw(){}", "no_change": False},
                {"valid": False, "canvas": None, "error": "SyntaxError: unexpected token ')'", "failure_kind": "program"},
            ])
            with patch.dict("os.environ", {module.INSPECT_CHOICE_FEEDBACK_ENV: "1"}, clear=True), \
                    patch.object(module, "render_action", side_effect=lambda **_: next(results)):
                asyncio.run(module.NativePaintingEnv.__dict__["run"](_DummyEnv(), module.NativePaintingTask(data), agents))
            info = agents.agent.interaction_obj.trace.info
            self.assertEqual(info["feedback_version"], module.INSPECT_CHOICE_FEEDBACK_VERSION)
            feedback = " ".join(
                part.text
                for part in agents.agent.interaction_obj.messages[2][0].content
                if hasattr(part, "text")
            )
            self.assertIn("previously valid CURRENT CANVAS is retained", feedback)
            self.assertIn("SyntaxError: unexpected token", feedback)
            self.assertNotIn("Fix the JavaScript syntax", feedback)
            self.assertIn("reply FINISHED without a code block", feedback)
            self.assertIn("complete replacement JavaScript sketch", feedback)
            self.assertIn("JavaScript fenced code block", feedback)
            self.assertTrue(info["final_canvas_valid"])

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
            self.assertNotIn("brief visual plan", task.data.system_prompt)
            self.assertIn("no prose plan", task.data.system_prompt)
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
            (["TypeError: scale is not a function"], "Do not shadow p5 global functions"),
            (["No complete JavaScript code block"], "Begin the next reply"),
            (["ReferenceError: v is not defined"], "Define every variable"),
            (["TypeError: a is not iterable"], "brush.polygon point list"),
        )
        for errors, expected in cases:
            with self.subTest(errors=errors):
                feedback = module._program_failure_feedback(errors)
                self.assertIn(expected, feedback)
                self.assertIn(errors[0], feedback)

    def test_opt_in_brush_reference_matches_pinned_exports_and_names(self):
        module = _load_module()
        asset = Path(__file__).parent / "vendor/integrations/watercolour/assets/p5.brush.js"
        source = asset.read_text()
        self.assertEqual(
            hashlib.sha256(asset.read_bytes()).hexdigest(),
            module.P5_BRUSH_EXPORT_BASIS_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(module.P5_BRUSH_API_REFERENCE.encode("utf-8")).hexdigest(),
            module.P5_BRUSH_API_REFERENCE_SHA256,
        )
        for operation in module.P5_BRUSH_SUPPORTED_OPERATIONS:
            with self.subTest(operation=operation):
                self.assertRegex(source, rf"t\.{re.escape(operation)}\s*=")
        for name in module.P5_BRUSH_DEFAULT_NAMES:
            with self.subTest(name=name):
                self.assertIn(f'["{name}",', source)
        for fictional in ("cgpen", "cwall", "cwater", "cmarker"):
            self.assertNotIn(f'"{fictional}"', module.P5_BRUSH_API_REFERENCE)

        with patch.dict("os.environ", {module.P5_BRUSH_API_REFERENCE_ENV: "1"}, clear=True):
            prompt = module._painting_system_prompt("text")
            self.assertIn(module.P5_BRUSH_API_REFERENCE, prompt)
            self.assertIn("`brush.rect` is supported", prompt)
            self.assertNotIn("use brush.rect/brush.ellipse", prompt)
            cases = (
                (["Error: Unknown brush name mysterybrush"], "brush.box() names"),
                (["Invalid color string"], "valid p5 color"),
                (["TypeError: p5.brush is not a constructor"], "already-loaded global brush singleton"),
                (["TypeError: brush.ellipse is not a function"], "brush.circle"),
                (["RangeError: Invalid array length"], "flat array of finite numeric [x, y] points"),
                (["TypeError: scale is not a function"], "shadows the global"),
            )
            for errors, expected in cases:
                with self.subTest(errors=errors):
                    feedback = module._program_failure_feedback(
                        errors, include_guidance=False, include_api_guidance=True,
                    )
                    self.assertIn(expected, feedback)
                    self.assertNotIn("Return a complete", feedback)

            legacy = module._program_failure_feedback(["Error: Brush mysterybrush not found"])
            api = module._program_failure_feedback(
                ["Error: Brush mysterybrush not found"], include_api_guidance=True,
            )
            self.assertIn(legacy, api)
            self.assertIn("brush.box() names", api)

        with patch.dict("os.environ", {}, clear=True):
            self.assertIs(module._painting_system_prompt("text"), module.TEXT_SYSTEM)

        with TemporaryDirectory() as temp:
            root = Path(temp)
            brief = root / "brief.txt"
            brief.write_text("Paint a pear.\n")
            data = module.NativePaintingData(
                idx=0, name="api-trace", source_group="api-trace", root=str(root),
                task_id="api-trace", reference=str(brief), reference_sha256=module.sha(brief),
                reference_kind="text", max_turns=2,
            )
            with patch.dict("os.environ", {module.P5_BRUSH_API_REFERENCE_ENV: "1"}, clear=True):
                agents = _Agents(["FINISHED", "FINISHED"])
                asyncio.run(module.NativePaintingEnv.__dict__["run"](_DummyEnv(), module.NativePaintingTask(data), agents))
            info = agents.agent.interaction_obj.trace.info
            self.assertEqual(info["brush_api_reference_version"], module.P5_BRUSH_API_REFERENCE_VERSION)
            self.assertEqual(info["brush_api_reference_sha256"], module.P5_BRUSH_API_REFERENCE_SHA256)
            self.assertEqual(
                info["brush_api_reference_export_basis_sha256"],
                module.P5_BRUSH_EXPORT_BASIS_SHA256,
            )

    def test_api_reference_keeps_legacy_retained_canvas_feedback(self):
        module = _load_module()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            brief = root / "brief.txt"
            brief.write_text("Paint a pear.\n")
            canvas = root / "canvas.png"
            canvas.write_bytes(b"canvas")
            data = module.NativePaintingData(
                idx=0, name="api-retained", source_group="api-retained", root=str(root),
                task_id="api-retained", reference=str(brief), reference_sha256=module.sha(brief),
                reference_kind="text", max_turns=3,
            )
            agents = _Agents([
                "```javascript\nfunction draw(){}\n```",
                "```javascript\nbroken();\n```",
                "FINISHED",
            ])
            results = iter([
                {"valid": True, "canvas": str(canvas), "program": "function draw(){}", "no_change": False},
                {"valid": False, "canvas": None, "error": "Error: Brush mysterybrush not found", "failure_kind": "program"},
            ])
            with patch.dict("os.environ", {module.P5_BRUSH_API_REFERENCE_ENV: "1"}, clear=True), \
                    patch.object(module, "render_action", side_effect=lambda **_: next(results)):
                asyncio.run(module.NativePaintingEnv.__dict__["run"](_DummyEnv(), module.NativePaintingTask(data), agents))
            feedback = " ".join(
                part.text
                for part in agents.agent.interaction_obj.messages[2][0].content
                if hasattr(part, "text")
            )
            self.assertIn("Return a complete, readable JavaScript sketch", feedback)
            self.assertIn("brush.box() names", feedback)
            self.assertNotIn("previously valid CURRENT CANVAS is retained", feedback)
            self.assertTrue(agents.agent.interaction_obj.trace.info["final_canvas_valid"])

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
