import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image


from painter import native_painting_renderer as renderer_module


class _StubProcess:
    pid = 4242

    def __init__(self):
        self.returncode = None
        self.wait_calls = []

    def poll(self):
        return self.returncode

    def communicate(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired(["stub"], timeout)
        return "", ""

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        if self.returncode is None:
            raise subprocess.TimeoutExpired(["stub"], timeout)
        return self.returncode

    def send_signal(self, signum):
        self.returncode = -signum

    def kill(self):
        self.returncode = -9


class NativePaintingRendererTests(unittest.TestCase):
    def test_code_only_unfenced_sketch_is_accepted_but_plan_is_not(self):
        code = "function setup(){createCanvas(600,600,WEBGL);}\nfunction draw(){noLoop();}"
        self.assertEqual(renderer_module.extract_sketch(code), code + "\n")
        with self.assertRaisesRegex(ValueError, "No complete JavaScript code block"):
            renderer_module.extract_sketch("Visual plan: paint a blue cup.")
        with self.assertRaisesRegex(ValueError, "No complete JavaScript code block"):
            renderer_module.extract_sketch("<script>" + code + "</script>")

    def test_configured_deadline_is_forwarded_and_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "episode"
            renderer = root / "renderer.py"
            renderer_python = root / "python"
            renderer.write_text("# test renderer\n")
            renderer_python.write_text("# test python\n")

            def fake_run(command, **kwargs):
                source = Path(command[6])
                output = Path(command[8])
                image = Image.new("RGB", (600, 600), "white")
                image.paste("#315b70", (100, 100, 500, 500))
                image.save(output)
                receipt = {
                    "valid": True,
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "png_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                }
                output.with_suffix(".json").write_text(json.dumps(receipt))
                return SimpleNamespace(returncode=0, stderr="")

            with patch.dict(os.environ, {"PAINTER_RENDER_TIMEOUT_SECONDS": "57"}, clear=False):
                with patch.object(renderer_module, "_run_renderer_process", side_effect=fake_run) as run:
                    result = renderer_module.render_action(
                        root=root, folder=folder,
                        reply="```javascript\nfunction setup() {}\n```",
                        initial_canvas=None, initial_program=None,
                        renderer=renderer, renderer_python=renderer_python,
                    )
            self.assertTrue(result["valid"])
            self.assertEqual(result["render_timeout_seconds"], 57)
            self.assertEqual(
                result["render_process_timeout_seconds"],
                57 + renderer_module.RENDER_PROCESS_GRACE_SECONDS,
            )
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--timeout") + 1], "57")
            self.assertEqual(
                run.call_args.kwargs["timeout"],
                57 + renderer_module.RENDER_PROCESS_GRACE_SECONDS,
            )
            self.assertEqual(run.call_args.kwargs["env"]["PAINTER_PRIME_ROLE"], "renderer")

    def test_renderer_interpreter_starts_with_renderer_role_without_rendering(self):
        """The lightweight child startup cannot activate the RL guard."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            environment = renderer_module._renderer_child_environment(root)
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import os; print(os.environ['PAINTER_PRIME_ROLE'])",
                ],
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "renderer")

    def test_deadline_rejects_out_of_range_values(self):
        for value in ("0", "not-an-int"):
            with self.subTest(value=value), patch.dict(os.environ, {"PAINTER_RENDER_TIMEOUT_SECONDS": value}, clear=False):
                with self.assertRaises(ValueError):
                    renderer_module.render_timeout_seconds()

    def test_historical_deadline_is_clamped_to_the_new_hard_cap(self):
        with patch.dict(os.environ, {"PAINTER_RENDER_TIMEOUT_SECONDS": "180"}, clear=False):
            self.assertEqual(
                renderer_module.render_timeout_seconds(),
                renderer_module.MAX_RENDER_TIMEOUT_SECONDS,
            )

    def test_cancelled_active_child_is_terminated_as_owned_group(self):
        process = _StubProcess()
        cancel_event = threading.Event()
        cancel_event.set()
        with patch.object(renderer_module.subprocess, "Popen", return_value=process):
            with patch.object(renderer_module.os, "getpgid", return_value=process.pid):
                with patch.object(renderer_module.os, "getsid", return_value=process.pid):
                    with patch.object(renderer_module, "_session_has_member", return_value=False):
                        with patch.object(
                            renderer_module.os,
                            "killpg",
                            side_effect=lambda _pid, signum: setattr(process, "returncode", -signum),
                        ) as killpg:
                            with self.assertRaises(renderer_module.RenderCancelled):
                                renderer_module._run_renderer_process(
                                    ["stub-renderer"], env={}, timeout=10,
                                    cancel_event=cancel_event,
                                )
        self.assertEqual(killpg.call_args.args[0], process.pid)
        self.assertEqual(killpg.call_args.args[1], renderer_module.signal.SIGTERM)
        self.assertTrue(process.wait_calls)

    def test_renderer_timeout_is_cleaned_up_and_remains_operational_failure(self):
        process = _StubProcess()
        with patch.object(renderer_module.subprocess, "Popen", return_value=process):
            with patch.object(renderer_module.os, "getpgid", return_value=process.pid):
                with patch.object(renderer_module.os, "getsid", return_value=process.pid):
                    with patch.object(renderer_module, "_session_has_member", return_value=False):
                        with patch.object(
                            renderer_module.os,
                            "killpg",
                            side_effect=lambda _pid, signum: setattr(process, "returncode", -signum),
                        ) as killpg:
                            with self.assertRaises(subprocess.TimeoutExpired):
                                renderer_module._run_renderer_process(
                                    ["stub-renderer"], env={}, timeout=0,
                                    cancel_event=threading.Event(),
                                )
        self.assertEqual(killpg.call_args.args[1], renderer_module.signal.SIGTERM)

    def test_parent_watchdog_writes_an_explicit_timeout_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            renderer = root / "renderer.py"
            renderer_python = root / "python"
            renderer.write_text("# test renderer\n")
            renderer_python.write_text("# test python\n")
            with patch.dict(
                os.environ,
                {"PAINTER_RENDER_TIMEOUT_SECONDS": "90"},
                clear=False,
            ):
                with patch.object(
                    renderer_module,
                    "_run_renderer_process",
                    side_effect=subprocess.TimeoutExpired(["renderer"], 95),
                ):
                    result = renderer_module.render_action(
                        root=root,
                        folder=root / "episode",
                        reply="```javascript\nfunction draw(){noLoop();}\n```",
                        initial_canvas=None,
                        initial_program=None,
                        renderer=renderer,
                        renderer_python=renderer_python,
                    )
            self.assertFalse(result["valid"])
            self.assertEqual(result["failure_kind"], "program_timeout")
            receipt = json.loads(Path(result["renderer_receipt"]).read_text())
            self.assertTrue(receipt["timed_out"])
            self.assertEqual(receipt["error_code"], "render_timeout")
            self.assertIn("watchdog", receipt["error"])

    def test_receipt_finished_before_cancel_is_never_consumed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            renderer = root / "renderer.py"
            renderer_python = root / "python"
            renderer.write_text("# test renderer\n")
            renderer_python.write_text("# test python\n")

            def stale_renderer(command, **kwargs):
                source = Path(command[6])
                output = Path(command[8])
                Image.new("RGB", (600, 600), "#315b70").save(output)
                output.with_suffix(".json").write_text(json.dumps({
                    "valid": True,
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "png_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                }))
                kwargs["cancel_event"].set()
                return SimpleNamespace(returncode=0, stderr="", stdout="")

            cancel_event = threading.Event()
            with patch.object(renderer_module, "_run_renderer_process", side_effect=stale_renderer):
                with self.assertRaises(renderer_module.RenderCancelled):
                    renderer_module.render_action(
                        root=root, folder=root / "episode",
                        reply="```javascript\nfunction setup() {}\n```",
                        initial_canvas=None, initial_program=None,
                        renderer=renderer, renderer_python=renderer_python,
                        cancel_event=cancel_event,
                    )

    def test_syntax_error_skips_renderer_and_preserves_previous_canvas(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "episode"
            renderer = root / "renderer.py"
            renderer_python = root / "python"
            renderer.write_text("# test renderer\n")
            renderer_python.write_text("# test python\n")
            previous_canvas = root / "previous.png"
            previous_canvas.write_bytes(b"previous canvas")

            def fake_run(command, **kwargs):
                self.assertEqual(command[:2], ["node", "--check"])
                self.assertNotIn("HF_TOKEN", kwargs["env"])
                return SimpleNamespace(
                    returncode=1,
                    stdout="",
                    stderr="/episode/program.js:1\nSyntaxError: Unexpected token '}'",
                )

            with patch.dict(
                os.environ,
                {renderer_module.JS_SYNTAX_PREFLIGHT_ENV: "1", "HF_TOKEN": "secret"},
                clear=False,
            ):
                with patch.object(renderer_module.subprocess, "run", side_effect=fake_run) as run:
                    result = renderer_module.render_action(
                        root=root,
                        folder=folder,
                        reply="```javascript\nfunction setup(){\nfunction draw(){}}\n```",
                        initial_canvas=previous_canvas,
                        initial_program=None,
                        renderer=renderer,
                        renderer_python=renderer_python,
                    )

            self.assertFalse(result["valid"])
            self.assertEqual(result["failure_kind"], "program")
            self.assertEqual(result["canvas"], str(previous_canvas))
            self.assertIn("SyntaxError", result["error"])
            self.assertEqual(run.call_count, 1)

    def test_valid_browser_script_is_checked_then_forwarded_to_existing_renderer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            renderer = root / "renderer.py"
            renderer_python = root / "python"
            renderer.write_text("# test renderer\n")
            renderer_python.write_text("# test python\n")

            def fake_run(command, **kwargs):
                if command[:2] == ["node", "--check"]:
                    return SimpleNamespace(returncode=0, stdout="", stderr="")
                raise AssertionError("renderer process should use _run_renderer_process")

            def fake_renderer(command, **kwargs):
                source = Path(command[6])
                output = Path(command[8])
                image = Image.new("RGB", (600, 600), "white")
                image.paste("#315b70", (100, 100, 500, 500))
                image.save(output)
                output.with_suffix(".json").write_text(json.dumps({
                    "valid": True,
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "png_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                }))
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            # Optional chaining, nullish coalescing, destructuring, and spread
            # are valid browser Script grammar and must not be rejected by a
            # preflight implemented with a narrower ad-hoc parser.
            reply = """```javascript
function setup(){createCanvas(600,600);}
function draw(){
  const points = [{x: 1, y: 2, extra: true}];
  const copy = points?.map(({x = 0, y = 0, ...rest}) => ({x, y, ...rest})) ?? [];
  background(copy.length ? '#fff' : '#000');
  noLoop();
}
```"""
            with patch.dict(
                os.environ,
                {renderer_module.JS_SYNTAX_PREFLIGHT_ENV: "1"},
                clear=False,
            ):
                with patch.object(renderer_module.subprocess, "run", side_effect=fake_run) as run:
                    with patch.object(renderer_module, "_run_renderer_process", side_effect=fake_renderer) as renderer_run:
                        result = renderer_module.render_action(
                            root=root,
                            folder=root / "episode",
                            reply=reply,
                            initial_canvas=None,
                            initial_program=None,
                            renderer=renderer,
                            renderer_python=renderer_python,
                        )

            self.assertTrue(result["valid"])
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args_list[0].args[0][:2], ["node", "--check"])
            self.assertEqual(renderer_run.call_args.args[0][0], "/usr/sbin/runuser")

    def test_missing_node_is_an_operational_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.dict(
                os.environ,
                {renderer_module.JS_SYNTAX_PREFLIGHT_ENV: "1"},
                clear=False,
            ):
                with patch.object(
                    renderer_module.subprocess,
                    "run",
                    side_effect=FileNotFoundError("node"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "Node.js.*unavailable"):
                        renderer_module.render_action(
                            root=root,
                            folder=root / "episode",
                            reply="```javascript\nfunction setup(){}\nfunction draw(){}\n```",
                            initial_canvas=None,
                            initial_program=None,
                            renderer=root / "renderer.py",
                            renderer_python=root / "python",
                        )

    def test_syntax_preflight_timeout_is_an_operational_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            timeout = subprocess.TimeoutExpired(
                ["node", "--check", "program.js"],
                renderer_module.JS_SYNTAX_PREFLIGHT_TIMEOUT_SECONDS,
            )
            with patch.dict(
                os.environ,
                {renderer_module.JS_SYNTAX_PREFLIGHT_ENV: "1"},
                clear=False,
            ):
                with patch.object(renderer_module.subprocess, "run", side_effect=timeout):
                    with self.assertRaisesRegex(RuntimeError, "syntax preflight timed out"):
                        renderer_module.render_action(
                            root=root,
                            folder=root / "episode",
                            reply="```javascript\nfunction setup(){}\nfunction draw(){}\n```",
                            initial_canvas=None,
                            initial_program=None,
                            renderer=root / "renderer.py",
                            renderer_python=root / "python",
                        )

    def test_missing_no_loop_is_rejected_with_a_preflight_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            renderer = root / "renderer.py"
            renderer_python = root / "python"
            renderer.write_text("# test renderer\n")
            renderer_python.write_text("# test python\n")
            with patch.object(renderer_module, "_run_renderer_process") as render_process:
                result = renderer_module.render_action(
                    root=root,
                    folder=root / "episode",
                    reply="""```javascript
function setup(){createCanvas(600,600);}
function draw(){background('#fff');}
```""",
                    initial_canvas=None,
                    initial_program=None,
                    renderer=renderer,
                    renderer_python=renderer_python,
                )
            self.assertFalse(result["valid"])
            self.assertEqual(result["failure_kind"], "program_timeout")
            self.assertTrue(result["renderer_preflight"])
            self.assertIn("noLoop", result["error"])
            render_process.assert_not_called()
            receipt = json.loads(Path(result["renderer_receipt"]).read_text())
            self.assertEqual(receipt["schema"], "native-painting-render-preflight-v1")
            self.assertEqual(receipt["error_code"], "render_preflight")
            self.assertFalse(receipt["timed_out"])

    def test_no_loop_in_a_comment_does_not_satisfy_preflight(self):
        self.assertTrue(
            renderer_module._missing_draw_completion_call(
                "function draw(){/* noLoop(); */ background('#fff');}"
            )
        )

    def test_no_loop_sketch_reaches_renderer_unchanged(self):
        self.assertFalse(
            renderer_module._missing_draw_completion_call(
                "function draw(){background('#fff'); noLoop();}"
            )
        )

    def test_blank_canvas_is_model_failure_even_when_renderer_reports_valid(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            renderer = root / "renderer.py"
            renderer_python = root / "python"
            renderer.write_text("# test renderer\n")
            renderer_python.write_text("# test python\n")

            def fake_run(command, **kwargs):
                source = Path(command[6])
                output = Path(command[8])
                Image.new("RGB", (600, 600), "black").save(output)
                output.with_suffix(".json").write_text(json.dumps({
                    "valid": True,
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "png_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                }))
                return SimpleNamespace(returncode=0, stderr="")

            with patch.object(renderer_module, "_run_renderer_process", side_effect=fake_run):
                result = renderer_module.render_action(
                    root=root, folder=root / "episode",
                    reply="```javascript\nfunction setup() {}\n```",
                    initial_canvas=None, initial_program=None,
                    renderer=renderer, renderer_python=renderer_python,
                )
            self.assertFalse(result["valid"])
            self.assertEqual(result["failure_kind"], "blank_canvas")
            self.assertIsNone(result["canvas"])

    def test_runtime_receipt_failure_is_service_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "episode"
            renderer = root / "renderer.py"
            renderer_python = root / "python"
            renderer.write_text("# test renderer\n")
            renderer_python.write_text("# test python\n")

            def fake_run(command, **kwargs):
                output = Path(command[8])
                output.with_suffix(".json").write_text(json.dumps({
                    "valid": False,
                    "error_code": "renderer_error",
                    "error": "ModuleNotFoundError: No module named 'playwright'",
                }))
                return SimpleNamespace(returncode=0, stderr="")

            with patch.object(renderer_module, "_run_renderer_process", side_effect=fake_run):
                with self.assertRaisesRegex(RuntimeError, "renderer service failure.*playwright"):
                    renderer_module.render_action(
                        root=root, folder=folder,
                        reply="```javascript\nfunction setup() {}\n```",
                        initial_canvas=None, initial_program=None,
                        renderer=renderer, renderer_python=renderer_python,
                    )

    def test_completed_worker_timeout_is_invalid_model_action(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "episode"
            renderer = root / "renderer.py"
            renderer_python = root / "python"
            renderer.write_text("# test renderer\n")
            renderer_python.write_text("# test python\n")

            def fake_run(command, **kwargs):
                output = Path(command[8])
                output.with_suffix(".json").write_text(json.dumps({
                    "valid": False,
                    "timed_out": True,
                    "error_code": "render_timeout",
                    "error": "Hard render deadline exceeded (180s)",
                }))
                return SimpleNamespace(returncode=0, stderr="")

            with patch.object(renderer_module, "_run_renderer_process", side_effect=fake_run):
                result = renderer_module.render_action(
                    root=root, folder=folder,
                    reply="```javascript\nfunction draw() { noLoop(); }\n```",
                    initial_canvas=None, initial_program=None,
                    renderer=renderer, renderer_python=renderer_python,
                )
            self.assertFalse(result["valid"])
            self.assertEqual(result["failure_kind"], "program_timeout")
            self.assertIn("noLoop", result["error"])
            receipt = json.loads(Path(result["renderer_receipt"]).read_text())
            self.assertTrue(receipt["timed_out"])
            self.assertEqual(receipt["error_code"], "render_timeout")


if __name__ == "__main__":
    unittest.main()
