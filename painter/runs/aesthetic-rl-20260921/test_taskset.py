"""Offline contract tests for the native aesthetic taskset.

These tests exercise parsing, renderer handoff, and the native Env loop with
small local doubles.  They do not start a model, launch a renderer process,
contact Luna, or require CUDA.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from asyncio import run as run_async
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent


def _load_renderer():
    spec = importlib.util.spec_from_file_location("aesthetic_render_actions", ROOT / "render_actions.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_finish_requires_a_standalone_marker_without_code() -> None:
    source = (ROOT / "taskset_impl.py").read_text()
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_is_finish")
    namespace = {"re": __import__("re")}
    exec(compile(ast.Module([function], []), str(ROOT / "taskset_impl.py"), "exec"), namespace)
    is_finish = namespace["_is_finish"]
    assert is_finish("The current canvas is done.\nFINISHED")
    assert not is_finish("```javascript\nsetup();\n```\nFINISHED")
    assert not is_finish("FINISHED with a code block")


def test_native_multiturn_and_terminal_only_reward_are_structural() -> None:
    source = (ROOT / "taskset_impl.py").read_text()
    tree = ast.parse(source)
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "PainterAestheticEnv" in classes
    env_source = ast.get_source_segment(source, classes["PainterAestheticEnv"])
    assert env_source and ".interaction(task)" in env_source
    assert env_source.count("interaction.turn") >= 2
    assert "@vf.stop" not in source
    reward_defs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "reward"
            for decorator in node.decorator_list
        )
    ]
    assert [node.name for node in reward_defs] == ["final_canvas_aesthetic"]
    assert "finish_reward" not in source
    assert "turn_reward" not in source
    assert "length_reward" not in source
    assert "acceptability_reward" not in source


def test_invalid_program_is_in_band_and_unchanged_program_keeps_canvas() -> None:
    renderer = _load_renderer()
    with TemporaryDirectory() as directory:
        root = Path(directory)
        initial = root / "initial.png"
        initial.write_bytes(b"initial canvas")
        invalid = renderer.render_action(
            root=root,
            folder=root / "invalid",
            reply="I cannot provide code.",
            initial_canvas=initial,
            initial_program="setup();",
        )
        assert invalid["valid"] is False
        assert invalid["failure_kind"] == "program"
        unchanged = renderer.render_action(
            root=root,
            folder=root / "unchanged",
            reply="```javascript\nsetup();\n```",
            initial_canvas=initial,
            initial_program="setup();",
        )
        assert unchanged["valid"] is True
        assert unchanged["no_change"] is True
        assert Path(unchanged["canvas"]) == initial


def test_valid_render_records_source_and_png_hashes() -> None:
    renderer = _load_renderer()
    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "vendor/integrations/watercolour").mkdir(parents=True)
        (root / "vendor/integrations/watercolour/renderer.py").write_text("# fixture")
        (root / "renderer-env/bin").mkdir(parents=True)
        (root / "renderer-env/bin/python").write_text("")
        initial = root / "initial.png"
        initial.write_bytes(b"initial canvas")
        program = "setup();\ndraw();"

        def fake_run(command, **kwargs):
            source = Path(command[command.index("--output") - 1])
            output = Path(command[command.index("--output") + 1])
            output.write_bytes(b"rendered canvas")
            receipt = {
                "valid": True,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "png_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "painting_seconds": 0.01,
            }
            output.with_suffix(".json").write_text(json.dumps(receipt))
            return subprocess.CompletedProcess(command, 0, "", "")

        with patch.object(renderer.subprocess, "run", side_effect=fake_run):
            result = renderer.render_action(
                root=root,
                folder=root / "turn-1",
                reply=f"```javascript\n{program}\n```",
                initial_canvas=initial,
                initial_program="setup();",
            )
        assert result["valid"] is True
        assert result["no_change"] is False
        assert Path(result["canvas"]).read_bytes() == b"rendered canvas"
        assert result["program"].strip() == program


def test_native_env_runs_three_turn_paint_inspect_finish_loop() -> None:
    """Exercise the actual Env loop with a tiny vf/agent double."""

    class Generic:
        @classmethod
        def __class_getitem__(cls, _item):
            return cls

    class Task(Generic):
        def __init__(self, data, config=None):
            self.data = data

    class Env(Generic):
        pass

    class TaskData:
        pass

    class Segment:
        terminated = False

        def __init__(self, reply):
            self.last_reply = reply

    class Trace:
        id = "trace-loop-test"

        def __init__(self):
            self.info = {}

    class Interaction:
        def __init__(self):
            self.trace = Trace()
            self.payloads = []
            self.replies = [
                "```javascript\nfirst();\n```",
                "```javascript\nsecond();\n```",
                "FINISHED",
            ]

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def turn(self, payload=None):
            self.payloads.append(payload)
            return Segment(self.replies.pop(0))

    class Agent:
        def __init__(self):
            self.interaction_obj = Interaction()

        def interaction(self, _task):
            return self.interaction_obj

    class UserMessage:
        def __init__(self, content):
            self.content = content

    class TextContentPart:
        def __init__(self, text):
            self.text = text
            self.type = "text"

    class ImageUrlSource:
        def __init__(self, url):
            self.url = url

    class ImageUrlContentPart:
        def __init__(self, image_url):
            self.image_url = image_url
            self.type = "image_url"

    vf = types.ModuleType("verifiers.v1")
    vf.Task = Task
    vf.TaskData = TaskData
    vf.Taskset = Generic
    vf.TasksetConfig = type("TasksetConfig", (), {})
    vf.TaskConfig = type("TaskConfig", (), {})
    vf.State = type("State", (), {})
    vf.Env = Env
    vf.EnvConfig = type("EnvConfig", (), {})
    vf.AgentConfig = type("AgentConfig", (), {})
    vf.Agents = type("Agents", (), {})
    vf.UserMessage = UserMessage
    vf.TextContentPart = TextContentPart
    vf.ImageUrlSource = ImageUrlSource
    vf.ImageUrlContentPart = ImageUrlContentPart
    vf.reward = lambda **_kwargs: (lambda fn: fn)
    vf.stop = lambda **_kwargs: (lambda fn: fn)
    verifiers = types.ModuleType("verifiers")
    verifiers.v1 = vf
    old_verifiers = sys.modules.get("verifiers")
    old_vf = sys.modules.get("verifiers.v1")
    sys.modules["verifiers"] = verifiers
    sys.modules["verifiers.v1"] = vf

    try:
        spec = importlib.util.spec_from_file_location("aesthetic_taskset_loop", ROOT / "taskset_impl.py")
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "reference.png").write_bytes(b"reference")
            (root / "canvas.png").write_bytes(b"canvas")
            (root / "program.js").write_text("initial();")
            state = {"id": "s", "reference": "reference.png", "canvas": "canvas.png", "program": "program.js"}
            data = types.SimpleNamespace(root=str(root), state=state, anchor="canvas.png", max_turns=3)
            task = types.SimpleNamespace(data=data)
            agent = Agent()

            def fake_render_action(**kwargs):
                turn = len(list(kwargs["folder"].parent.glob("turn-*"))) + 1
                output = kwargs["folder"] / "painting.png"
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(f"canvas-{turn}".encode())
                return {"valid": True, "canvas": str(output), "program": kwargs["reply"]}

            with patch.dict(sys.modules, {"render_actions": types.SimpleNamespace(render_action=fake_render_action)}):
                run_async(module.PainterAestheticEnv().run(task, types.SimpleNamespace(agent=agent)))

            assert agent.interaction_obj.trace.info["finished"] is True
            assert agent.interaction_obj.trace.info["finish_turn"] == 3
            assert agent.interaction_obj.trace.info["final_canvas_valid"] is True
            assert len(agent.interaction_obj.payloads) == 3
            for payload in agent.interaction_obj.payloads[1:]:
                assert len(payload) == 1
                assert sum(getattr(part, "type", "") == "image_url" for part in payload[0].content) == 2
    finally:
        if old_verifiers is None:
            sys.modules.pop("verifiers", None)
        else:
            sys.modules["verifiers"] = old_verifiers
        if old_vf is None:
            sys.modules.pop("verifiers.v1", None)
        else:
            sys.modules["verifiers.v1"] = old_vf
