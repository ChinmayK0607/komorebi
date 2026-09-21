"""Mocked native Env tests for fresh/correction recovery behavior.

These tests exercise prompt construction and state transitions only.  They do
not invoke a renderer, model, judge worker, or network service.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from asyncio import run as run_async
from tempfile import TemporaryDirectory
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent


class _Generic:
    @classmethod
    def __class_getitem__(cls, _item):
        return cls


class _Task(_Generic):
    def __init__(self, data, config=None):
        self.data = data


class _TaskData:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _Segment:
    terminated = False

    def __init__(self, reply):
        self.last_reply = reply


class _Trace:
    id = "curriculum-repair-test"

    def __init__(self):
        self.info = {}


class _Interaction:
    def __init__(self, replies):
        self.trace = _Trace()
        self.replies = list(replies)
        self.payloads = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def turn(self, payload=None):
        self.payloads.append(payload)
        return _Segment(self.replies.pop(0))


class _Agent:
    def __init__(self, replies):
        self.interaction_obj = _Interaction(replies)

    def interaction(self, _task):
        return self.interaction_obj


class _UserMessage:
    def __init__(self, content):
        self.content = content


class _TextContentPart:
    type = "text"

    def __init__(self, text):
        self.text = text


class _ImageUrlSource:
    def __init__(self, url):
        self.url = url


class _ImageUrlContentPart:
    type = "image_url"

    def __init__(self, image_url):
        self.image_url = image_url


def _load_taskset():
    vf = types.ModuleType("verifiers.v1")
    vf.Task = _Task
    vf.TaskData = _TaskData
    vf.Taskset = _Generic
    vf.TasksetConfig = type("TasksetConfig", (), {})
    vf.TaskConfig = type("TaskConfig", (), {})
    vf.State = type("State", (), {})
    vf.Env = _Generic
    vf.EnvConfig = type("EnvConfig", (), {})
    vf.AgentConfig = type("AgentConfig", (), {})
    vf.Agents = type("Agents", (), {})
    vf.UserMessage = _UserMessage
    vf.TextContentPart = _TextContentPart
    vf.ImageUrlSource = _ImageUrlSource
    vf.ImageUrlContentPart = _ImageUrlContentPart
    vf.reward = lambda **_kwargs: (lambda fn: fn)
    verifiers = types.ModuleType("verifiers")
    verifiers.v1 = vf
    old_modules = {name: sys.modules.get(name) for name in (
        "verifiers", "verifiers.v1", "aesthetic_judge", "judge_queue", "contract"
    )}
    sys.modules["verifiers"] = verifiers
    sys.modules["verifiers.v1"] = vf
    sys.modules["aesthetic_judge"] = types.SimpleNamespace(PROTOCOL_VERSION="fixture")
    sys.modules["judge_queue"] = types.SimpleNamespace()
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("curriculum_repair_taskset", ROOT / "taskset_impl.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def restore():
        sys.path.remove(str(ROOT))
        for name, value in old_modules.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value

    return module, restore


def _texts(payload):
    if not payload:
        return []
    return [part.text for part in payload[0].content if getattr(part, "type", None) == "text"]


class TasksetEnvTests(unittest.TestCase):
    def _fixture(self, mode="correction", replies=()):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "reference.png").write_bytes(b"reference")
        (root / "canvas.png").write_bytes(b"initial")
        (root / "program.js").write_text("initial();")
        state = {"id": "s", "reference": "reference.png", "mode": mode}
        if mode == "correction":
            state.update(canvas="canvas.png", program="program.js")
        return root, state, _Agent(replies)

    def _run(self, module, root, state, agent, max_turns=3):
        data = types.SimpleNamespace(
            root=str(root), state=state, anchor="canvas.png", mode=state["mode"], max_turns=max_turns
        )
        task = types.SimpleNamespace(data=data)
        run_async(module.PainterAestheticEnv().run(task, types.SimpleNamespace(agent=agent)))
        return agent.interaction_obj

    def test_fresh_premature_finish_gets_recoverable_feedback_and_paint(self):
        module, restore = _load_taskset()
        try:
            root, state, agent = self._fixture(
                mode="fresh",
                replies=["FINISHED", "```javascript\npaint();\n```", "FINISHED"],
            )
            rendered = []

            def fake_render_action(**kwargs):
                output = kwargs["folder"] / "painting.png"
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(b"painted")
                rendered.append(kwargs)
                return {"valid": True, "canvas": str(output), "program": "paint();", "no_change": False}

            with patch.dict(sys.modules, {"render_actions": types.SimpleNamespace(render_action=fake_render_action)}):
                info = self._run(module, root, state, agent)
            self.assertEqual(info.trace.info["invalid_finish_turns"], [1])
            self.assertEqual(len(rendered), 1)
            self.assertTrue(info.trace.info["final_canvas_valid"])
            self.assertIn(module.FINISH_WITHOUT_CANVAS, _texts(info.payloads[1]))
            self.assertIsNotNone(info.payloads[2])
        finally:
            restore()

    def test_no_change_feedback_uses_shared_contract(self):
        module, restore = _load_taskset()
        try:
            root, state, agent = self._fixture(
                replies=["```javascript\nfirst();\n```", "```javascript\nfirst();\n```", "FINISHED"]
            )

            def fake_render_action(**kwargs):
                return {
                    "valid": True,
                    "canvas": str(root / "canvas.png"),
                    "program": "first();",
                    "no_change": True,
                }

            with patch.dict(sys.modules, {"render_actions": types.SimpleNamespace(render_action=fake_render_action)}):
                info = self._run(module, root, state, agent)
            self.assertIn(module.NO_CHANGE, _texts(info.payloads[2]))
        finally:
            restore()

    def test_invalid_render_preserves_last_canvas_then_recovers(self):
        module, restore = _load_taskset()
        try:
            root, state, agent = self._fixture(
                replies=[
                    "```javascript\nfirst();\n```",
                    "```javascript\nbroken();\n```",
                    "```javascript\nfixed();\n```",
                ]
            )
            calls = []

            def fake_render_action(**kwargs):
                calls.append(kwargs)
                if len(calls) == 1:
                    output = kwargs["folder"] / "painting.png"
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(b"first")
                    return {"valid": True, "canvas": str(output), "program": "first();", "no_change": False}
                if len(calls) == 2:
                    return {"valid": False, "error": "syntax error", "failure_kind": "program"}
                output = kwargs["folder"] / "painting.png"
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(b"fixed")
                return {"valid": True, "canvas": str(output), "program": "fixed();", "no_change": False}

            with patch.dict(sys.modules, {"render_actions": types.SimpleNamespace(render_action=fake_render_action)}):
                info = self._run(module, root, state, agent)
            self.assertEqual(len(calls), 3)
            self.assertEqual(calls[1]["initial_canvas"], Path(calls[0]["folder"]) / "painting.png")
            self.assertTrue(info.trace.info["final_canvas_valid"])
            self.assertNotIn("service_failure", info.trace.info)
        finally:
            restore()

    def test_prepared_curriculum_order_is_consumed_by_taskset_loader(self):
        module, restore = _load_taskset()
        try:
            with TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "contract.py").write_bytes((ROOT / "contract.py").read_bytes())
                for name in ("fresh-ref.png", "fresh-anchor.png", "corr-ref.png", "corr-canvas.png", "corr-anchor.png"):
                    (root / name).write_bytes(name.encode())
                (root / "corr.js").write_text("initial();")
                plan = {
                    "schema": "painter.curriculum-order.v1",
                    "max_turns": 6,
                    "contract_snapshot": module._runtime_contract_snapshot(root),
                    "modes": {
                        "fresh": {"max_turns": 6, "tasks": [{
                            "task_id": "fresh-first", "source_group": "g1", "family": "box",
                            "reference": "fresh-ref.png", "anchor": "fresh-anchor.png", "mode": "fresh",
                        }]},
                        "correction": {"max_turns": 6, "tasks": [{
                            "task_id": "correction-second", "source_group": "g2", "family": "bus",
                            "reference": "corr-ref.png", "initial_canvas": "corr-canvas.png",
                            "initial_program": "corr.js", "anchor": "corr-anchor.png", "mode": "correction",
                        }]},
                    },
                }
                (root / "curriculum-plan.json").write_text(json.dumps(plan))
                module.PainterAestheticData = _TaskData
                module.PainterAestheticTask = _Task
                taskset = module.PainterAestheticTaskset()
                taskset.config = types.SimpleNamespace(root=str(root), curriculum_mode="both", max_turns=None, task=None)
                tasks = taskset.load()
                self.assertEqual([task.data.name for task in tasks], ["fresh-first:anchor-0", "correction-second:anchor-0"])
                self.assertEqual([task.data.mode for task in tasks], ["fresh", "correction"])
                self.assertEqual([task.data.max_turns for task in tasks], [6, 6])
        finally:
            restore()


if __name__ == "__main__":
    unittest.main()
