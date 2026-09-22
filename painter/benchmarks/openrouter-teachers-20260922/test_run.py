import io
import json
import sys
import tempfile
import time
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run


class _Response:
    def __init__(self, payload, status=200):
        self.status = status
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self._payload).encode()

    def getcode(self):
        return self.status


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
    def test_content_extraction_and_last_complete_fence(self):
        self.assertEqual(run.extract_content([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]), "a\nb")
        code, plan = run.parse_program("plan\n```javascript\nold();\n```\nmore\n```js\nnew();\n```")
        self.assertEqual(code, "new();\n")
        self.assertEqual(plan, "plan\n\nmore")
        self.assertIsNone(run.parse_program("FINISHED")[0])

    def test_http_retry_redacts_key_and_preserves_usage(self):
        calls = []

        def opener(request, timeout):
            calls.append((request, timeout))
            if len(calls) == 1:
                raise urllib.error.HTTPError(request.full_url, 429, "busy", {"Retry-After": "0"}, io.BytesIO(b'{"error":"Bearer secret"}'))
            return _Response({"id": "x", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10, "cost": 0.25}})

        client = run.OpenRouterClient(endpoint="https://example.invalid", api_key="secret", opener=opener, sleep=lambda _: None, max_retries=1)
        response = client.complete({"model": "m", "messages": []})
        self.assertEqual(response["usage"]["total_tokens"], 10)
        self.assertEqual(len(calls), 2)
        self.assertIn("Bearer", calls[0][0].headers["Authorization"])
        self.assertEqual(run.response_record(response)["usage"]["cost"], 0.25)
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
            "z-ai/glm-5.3-flash": "max", "deepseek/deepseek-v4.1-flash": "max",
            "stepfun/step-3.7-flash": "high", "xiaomi/mimo-v2.6-flash": None,
            "xiaomi/mimo-v2.6-pro": None,
        }
        self.assertEqual(set(inputs.config["models"]), set(expected_quality))
        for model, effort in expected_quality.items():
            resolved = run.resolve_reasoning(inputs, model, "quality")
            self.assertTrue(resolved.get("enabled"))
            self.assertEqual(resolved.get("effort"), effort)

    def test_track_paths_and_dry_report(self):
        with tempfile.TemporaryDirectory() as temp:
            inputs = _fixture(Path(temp))
            self.assertNotEqual(run.job_id("test/model", "r1", track="quality"), run.job_id("test/model", "r1", track="speed"))
            report = run.dry_run_report(inputs)
            self.assertEqual(report["episodes_total"], 2)
            self.assertEqual(report["requests_max_total"], 3)

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
                "The observed canvas matches the reference.\nFINISHED",
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
