import json
import io
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run


def _fixture(root: Path) -> run.BenchmarkInputs:
    image = root / "ref.jpg"
    image.write_bytes(b"fixture-jpeg")
    catalog = {
        "models": [{
            "id": "test/model",
            "architecture": {"modality": "text+image->text", "input_modalities": ["text", "image"]},
            "context_length": 100000,
            "top_provider": {"context_length": 100000, "max_completion_tokens": 500},
            "reasoning": {"supported_efforts": ["low", "high"], "default_enabled": True},
        }]
    }
    (root / "model-catalog.json").write_text(json.dumps(catalog))
    (root / "config.json").write_text(json.dumps({
        "models": ["test/model"],
        "tracks": {"quality": {"max_turns": 2, "max_tokens": "native", "episode_timeout_seconds": None}, "speed": {"max_turns": 1, "max_tokens": 8, "episode_timeout_seconds": 2}},
        "temperature": 1.0, "concurrency": 1, "render_concurrency": 1, "samples_per_image": 1,
        "catalog": "model-catalog.json", "max_retries": 1,
    }))
    (root / "refs.json").write_text(json.dumps({"references": [{"id": "r1", "image": "ref.jpg", "sha256": run.sha_file(image)}]}))
    (root / "prompt.txt").write_text("SYSTEM PROMPT")
    return run.load_inputs(root)


class RunTests(unittest.TestCase):
    def test_progress_reporter_is_truthful_until_response_usage_exists(self):
        stream = io.StringIO()
        reporter = run.ProgressReporter(episodes_total=1, stream=stream)
        common = {
            "job_id": "episode-1", "episode_number": 1, "model": "test/model",
            "reference_id": "r1", "turn": 1, "max_turns": 2,
        }
        reporter.emit({**common, "phase": "waiting_for_provider", "total_tokens": 0})
        reporter.emit({
            **common, "phase": "response_received", "total_tokens": 12,
            "prompt_tokens": 5, "completion_tokens": 7, "response_total_tokens": 12,
        })
        reporter.emit({
            **common, "phase": "episode_complete", "status": "turn_limit",
            "turns_completed": 1, "total_tokens": 12,
        })
        output = stream.getvalue()
        self.assertIn("phase=waiting_for_provider", output)
        self.assertIn("current_response_tokens=unknown (provider response pending; no live token stream)", output)
        self.assertIn("response_tokens=prompt:5 completion:7 total:12", output)
        self.assertIn("known_tokens=12", output)
        self.assertNotIn("prompt", output.split("phase=waiting_for_provider", 1)[1].split("\n", 1)[0])

    def test_progress_reporter_clears_phase_and_reuse_fields(self):
        stream = io.StringIO()
        reporter = run.ProgressReporter(episodes_total=1, stream=stream)
        common = {
            "job_id": "episode-1", "episode_number": 1, "model": "test/model",
            "reference_id": "r1", "max_turns": 2,
        }
        first = reporter.emit({
            **common, "turn": 1, "phase": "waiting_for_provider",
            "phase_elapsed_seconds": 9.0, "total_tokens": 0,
        })
        reused = reporter.emit({
            **common, "turn": 1, "phase": "response_reused", "api_reused": True,
            "prompt_tokens": 2, "completion_tokens": 3, "response_total_tokens": 5,
            "total_tokens": 5,
        })
        reporter.emit({**common, "turn": 1, "phase": "rendering", "total_tokens": 5})
        reporter.emit({
            **common, "turn": 2, "phase": "waiting_for_provider",
            "phase_elapsed_seconds": 4.0, "total_tokens": 5,
        })
        reporter.emit({**common, "turn": 2, "phase": "response_received", "total_tokens": 11})
        self.assertGreater(reused["sequence"], first["sequence"])
        lines = stream.getvalue().splitlines()
        rendering = next(line for line in lines if "phase=rendering" in line)
        response = next(line for line in lines if "phase=response_received" in line)
        self.assertNotIn("api=reused", rendering)
        self.assertNotIn("phase_elapsed=00m09s", response)
        self.assertNotIn("prompt:2", response)

    def test_episode_runner_emits_provider_heartbeat_and_phases_offline(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = _fixture(root)
            ref = inputs.references[0]
            settings = run.settings_from_inputs(inputs, "quality")
            settings["max_turns"] = 1
            episode = root / "episodes" / run.job_id("test/model", "r1", track="quality")
            events = []

            class Client:
                api_key = "secret"
                timeout = 2

                def complete(self, payload, **kwargs):
                    time.sleep(0.03)
                    return {
                        "choices": [{"message": {"content": "plan\n```js\nfunction draw(){noLoop()}\n```"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10},
                    }

            def render(**kwargs):
                output = Path(kwargs["output"])
                output.write_bytes(b"canvas")
                return {"valid": True, "receipt": {"valid": True}, "canvas_sha256": run.sha_file(output)}

            result = run.EpisodeRunner(
                inputs, "test/model", ref, 1, episode_dir=episode, client=Client(),
                render_fn=render, render_lock=__import__("threading").Semaphore(1),
                renderer_options={}, settings=settings, on_progress=events.append,
                progress_heartbeat_seconds=0.01,
            ).run()
            self.assertEqual(result["status"], "turn_limit")
            phases = [event["phase"] for event in events]
            self.assertIn("waiting_for_provider", phases)
            self.assertIn("response_received", phases)
            self.assertIn("rendering", phases)
            self.assertIn("turn_complete", phases)
            response_event = next(event for event in events if event["phase"] == "response_received")
            self.assertEqual(response_event["total_tokens"], 10)

    def test_local_dotenv_reads_only_the_gateway_key(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / ".env.local").write_text(
                "UNRELATED=do-not-load\nAI_GATEWAY_API_KEY='local-secret'\n"
            )
            self.assertEqual(run.local_dotenv_api_key(root), "local-secret")

    def test_local_renderer_uses_metal_and_scrubs_key_from_child(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "program.js"
            renderer = root / "renderer.py"
            renderer_python = root / "python"
            output = root / "canvas.png"
            source.write_text("function setup() {}\n")
            renderer.write_text("# renderer\n")
            renderer_python.write_text("# python\n")
            observed = {}

            def fake_run(command, **kwargs):
                observed["command"] = command
                observed["env"] = kwargs["env"]
                output_path = Path(command[command.index("--output") + 1])
                output_path.write_bytes(b"png")
                receipt = {
                    "valid": True,
                    "source_sha256": run.sha_file(source),
                    "png_sha256": run.sha_file(output_path),
                }
                output_path.with_suffix(".json").write_text(json.dumps(receipt))
                return __import__("subprocess").CompletedProcess(command, 0, stderr="")

            with patch.object(run.sys, "platform", "darwin"):
                with patch.dict(run.os.environ, {"AI_GATEWAY_API_KEY": "secret"}, clear=False):
                    with patch.object(run.subprocess, "run", side_effect=fake_run):
                        result = run.render_program(
                            root=root, source=source, output=output, renderer=renderer,
                            renderer_python=renderer_python, local=True,
                        )
            self.assertTrue(result["valid"])
            self.assertNotIn("runuser", observed["command"])
            self.assertEqual(observed["command"][-1], "metal")
            self.assertNotIn("AI_GATEWAY_API_KEY", observed["env"])

    def test_content_extraction_and_last_complete_fence(self):
        self.assertEqual(run.extract_content([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]), "a\nb")
        code, plan = run.parse_program("plan\n```javascript\nold();\n```\nmore\n```js\nnew();\n```")
        self.assertEqual(code, "new();\n")
        self.assertEqual(plan, "plan\n\nmore")
        self.assertIsNone(run.parse_program("FINISHED")[0])
        self.assertTrue(run.is_finished_text("The observed canvas matches the reference.\nFINISHED"))
        self.assertTrue(run.is_finished_text("**Reason**: The current canvas matches the reference. FINISHED"))
        self.assertTrue(run.is_finished_text("**Reason**: The current canvas matches the reference.\nFINISHED."))
        self.assertFalse(run.is_finished_text("The explanation mentions FINISHED but does not finish."))
        self.assertFalse(run.is_finished_text("```js\n// FINISHED\n```"))

    def test_gateway_response_preserves_usage_and_redacts_errors(self):
        response = {"id": "x", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10, "cost": None}}
        self.assertEqual(run.response_record(response)["usage"]["total_tokens"], 10)
        self.assertIsNone(run.response_record(response)["usage"]["cost"])
        self.assertEqual(run.redact("Bearer secret", "secret"), "Bearer [REDACTED]")
        nested = {"usage": {"completion_tokens_details": {"reasoning_tokens": 3}}}
        self.assertEqual(run.normalize_usage(nested)["reasoning_tokens"], 3)

    def test_context_budget_is_explicit_and_can_be_exhausted(self):
        context = run.context_accounting([{"role": "system", "content": "x" * 2000}], {"provider_context_length": 100, "native_ceiling": 500}, 500)
        self.assertTrue(context["context_exhausted"])
        self.assertEqual(context["effective_max_tokens"], 0)
        self.assertEqual(context["reduction_reason"], "physical_context_remaining")

    def test_reasoning_payload_filters_provenance_and_catalog_defaults(self):
        payload = run.build_payload([], {"temperature": 1.0, "max_tokens": 8}, "test/model", max_tokens=8, reasoning={"enabled": True, "effort": "high", "source": "catalog", "_omit": False})
        self.assertEqual(payload["reasoning"], {"enabled": True, "effort": "high"})
        self.assertNotIn("source", payload["reasoning"])
        self.assertNotIn("_omit", payload["reasoning"])
        inputs = run.load_inputs(Path(__file__).resolve().parent)
        expected_quality = {
            "zai/glm-5.3-flash", "deepseek/deepseek-v4.1-flash",
            "stepfun/step-5-preview", "xiaomi/mimo-v2.6-flash",
            "xiaomi/mimo-v2.6-pro",
        }
        self.assertEqual(set(inputs.config["models"]), set(expected_quality))
        for model in expected_quality:
            resolved = run.resolve_reasoning(inputs, model, "quality")
            self.assertTrue(resolved.get("_omit"))

    def test_screen_track_uses_highest_effort_and_stratified_references(self):
        inputs = run.load_inputs(Path(__file__).resolve().parent)
        screen_refs = run.references_for_track(inputs, "screen")
        self.assertEqual(len(screen_refs), 8)
        self.assertEqual(len({str(ref.metadata["category"]) for ref in screen_refs}), 8)
        self.assertEqual(len(run.jobs(inputs, track="screen")), 40)
        self.assertEqual(len(run.jobs(inputs, track="all")), 400)
        first_slice = run.jobs(inputs, track="screen", start=0, limit=5)
        second_slice = run.jobs(inputs, track="screen", start=5, limit=5)
        self.assertEqual({item[2].id for item in first_slice}, {screen_refs[0].id})
        self.assertEqual({item[2].id for item in second_slice}, {screen_refs[1].id})
        self.assertFalse({(track, model, ref.id, sample) for track, model, ref, sample in first_slice} &
                         {(track, model, ref.id, sample) for track, model, ref, sample in second_slice})
        self.assertEqual(run.dry_run_report(inputs, track="screen", start=5, limit=5)["start_episode"], 5)
        report = run.dry_run_report(inputs, track="screen")
        self.assertEqual(report["episodes_total"], 40)
        self.assertEqual(report["tracks"]["screen"]["max_turns"], 6)
        self.assertEqual(report["tracks"]["screen"]["max_tokens"], "native")
        self.assertEqual(report["reference_ids_selected"], [ref.id for ref in screen_refs])
        self.assertNotIn("min_turns", report["tracks"]["screen"])
        all_report = run.dry_run_report(inputs, track="all")
        self.assertEqual(all_report["episodes_total"], 400)
        self.assertEqual(set(all_report["tracks"]), {"quality", "speed"})

    def test_highest_supported_reasoning_applies_to_screen(self):
        with tempfile.TemporaryDirectory() as temp:
            inputs = _fixture(Path(temp))
            inputs.catalog["profile_kind"] = "catalogued"
            inputs.config["tracks"]["screen"] = {
                "max_turns": 6, "max_tokens": "native", "episode_timeout_seconds": None,
                "reasoning_effort": "highest_supported",
            }
            resolved = run.resolve_reasoning(inputs, "test/model", "screen")
            self.assertEqual(resolved["effort"], "high")

    def test_track_paths_and_dry_report(self):
        with tempfile.TemporaryDirectory() as temp:
            inputs = _fixture(Path(temp))
            self.assertNotEqual(run.job_id("test/model", "r1", track="quality"), run.job_id("test/model", "r1", track="speed"))
            report = run.dry_run_report(inputs)
            self.assertEqual(report["episodes_total"], 2)
            self.assertEqual(report["requests_max_total"], 3)

    def test_run_benchmark_survives_generator_progress_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = _fixture(root)
            renderer = root / "renderer.py"
            renderer_python = root / "python"
            renderer.write_text("# renderer\n")
            renderer_python.write_text("# python\n")
            result = {
                "status": "turn_limit", "job_id": "job", "settings": {"track": "quality"},
                "model": "test/model", "reference_id": "r1", "turns": [],
                "total_tokens": 0, "total_cost": 0, "cost_missing_turns": 0,
                "cost_complete": True,
            }
            class FakeClient:
                timeout = 0
                max_retries = 0

            with patch.object(run, "GatewayClientPool") as pool_class:
                with patch.object(run, "EpisodeRunner") as runner_class:
                    pool_class.return_value.get.return_value = FakeClient()
                    runner_class.return_value.run.return_value = result
                    with patch.object(run, "tqdm", side_effect=lambda iterable, **_: iter(iterable)):
                        summary = run.run_benchmark(
                            inputs, track="quality", limit=1,
                            renderer_options={
                                "renderer": renderer, "renderer_python": renderer_python,
                                "browser_path": None, "run_as_user": None, "local": True,
                            },
                            api_key="test-key",
                        )
            self.assertEqual(summary["status_counts"], {"turn_limit": 1})
            self.assertEqual(json.loads((root / "progress.json").read_text())["episodes_completed"], 1)

    def test_node_setup_checks_and_installs_pinned_tqdm(self):
        setup = (Path(__file__).resolve().parent / "setup_benchmark_node.sh").read_text()
        self.assertIn('m.version("tqdm") == "4.67.1"', setup)
        self.assertIn('"tqdm==$TQDM_VERSION"', setup)
        syntax = __import__("subprocess").run(
            ["bash", "-n", str(Path(__file__).resolve().parent / "setup_benchmark_node.sh")],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)

    def test_resume_reuses_completed_response_after_renderer_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = _fixture(root)
            ref = inputs.references[0]
            settings = run.settings_from_inputs(inputs, "quality")
            settings["max_turns"] = 1
            episode = root / "episodes" / run.job_id("test/model", "r1", track="quality")
            responses = [{"choices": [{"message": {"content": "plan\n```js\nfunction draw(){noLoop()}\n```"}, "finish_reason": "stop"}], "usage": {"total_tokens": 7}}]

            class Client:
                api_key = "secret"
                timeout = 2
                def __init__(self): self.calls = 0
                def complete(self, payload, **kwargs):
                    self.calls += 1
                    return responses.pop(0)

            client = Client()
            def bad_render(**kwargs):
                return {"valid": False, "service_failure": True, "error_code": "renderer_error", "error": "offline"}

            first = run.EpisodeRunner(inputs, "test/model", ref, 1, episode_dir=episode, client=client, render_fn=bad_render, render_lock=__import__("threading").Semaphore(1), renderer_options={}, settings=settings).run()
            self.assertEqual(first["status"], "renderer_error")
            self.assertEqual(client.calls, 1)

            def good_render(**kwargs):
                output = Path(kwargs["output"])
                output.write_bytes(b"png")
                return {"valid": True, "receipt": {"valid": True}, "canvas_sha256": run.sha_file(output)}

            second_client = Client()
            second = run.EpisodeRunner(inputs, "test/model", ref, 1, episode_dir=episode, client=second_client, render_fn=good_render, render_lock=__import__("threading").Semaphore(1), renderer_options={}, settings=settings).run()
            self.assertEqual(second_client.calls, 0)
            self.assertEqual(second["status"], "turn_limit")
            self.assertTrue(second["final_valid_canvas"])

    def test_second_turn_reconstructs_full_history_and_current_image(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = _fixture(root)
            ref = inputs.references[0]
            settings = run.settings_from_inputs(inputs, "quality")
            episode = root / "episodes" / run.job_id("test/model", "r1", track="quality")
            replies = [
                "first plan\n```javascript\nfunction draw(){noLoop()}\n```",
                "**Reason**: The observed canvas matches the reference. FINISHED",
            ]
            sent = []
            class Client:
                api_key = "secret"
                timeout = 2
                def complete(self, payload, **kwargs):
                    sent.append(payload)
                    return {"choices": [{"message": {"content": replies.pop(0)}, "finish_reason": "stop"}], "usage": {"total_tokens": 5, "cost": 0.1}}
            def render(**kwargs):
                output = Path(kwargs["output"])
                output.write_bytes(b"canvas")
                return {"valid": True, "receipt": {"valid": True}, "canvas_sha256": run.sha_file(output)}
            result = run.EpisodeRunner(inputs, "test/model", ref, 1, episode_dir=episode, client=Client(), render_fn=render, render_lock=__import__("threading").Semaphore(1), renderer_options={}, settings=settings).run()
            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(result["turns"]), 2)
            self.assertTrue(result["turns"][0]["render"]["valid"])
            self.assertTrue(result["turns"][1]["finished_observed"])
            self.assertEqual(len(sent), 2)
            self.assertEqual([message["role"] for message in sent[1]["messages"]], ["system", "user", "assistant", "user"])
            self.assertIn("CURRENT CANVAS", sent[1]["messages"][-1]["content"][0]["text"])
            self.assertEqual(len(sent[1]["messages"][-1]["content"]), 3)

    def test_speed_deadline_is_censored_and_keeps_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = _fixture(root)
            ref = inputs.references[0]
            settings = run.settings_from_inputs(inputs, "speed")
            episode = root / "episodes" / run.job_id("test/model", "r1", track="speed")
            class Client:
                api_key = "secret"
                timeout = 2
                def complete(self, payload, **kwargs): raise AssertionError("deadline should preflight before API")
            runner = run.EpisodeRunner(inputs, "test/model", ref, 1, episode_dir=episode, client=Client(), render_fn=lambda **_: {}, render_lock=__import__("threading").Semaphore(1), renderer_options={}, settings=settings)
            state = runner._load_or_init()
            state["active_seconds"] = 3
            runner._save(state)
            result = runner.run()
            self.assertEqual(result["status"], "deadline_censored")
            self.assertEqual(result["turns"], [])


if __name__ == "__main__":
    unittest.main()
