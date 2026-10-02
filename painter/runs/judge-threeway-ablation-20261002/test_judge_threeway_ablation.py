from __future__ import annotations

import io
import importlib.util
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from PIL import Image


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("threeway_runner_tests", HERE / "judge_threeway_ablation.py")
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class _Response:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body


def _body(model: str, winner: str = "LEFT", *, usage: dict | None = None) -> bytes:
    assessment = {
        "brief_fidelity": "clear",
        "visible_evidence": "The connected form is readable.",
        "weaknesses": ["Some edges are soft."],
    }
    raw = json.dumps({
        "left_assessment": assessment,
        "right_assessment": assessment,
        "reason": "The panels differ in visible connected form.",
        "winner": winner,
        "preference": "tied" if winner == "TIE" else "slight",
    })
    return json.dumps({
        "model": model,
        "choices": [{"message": {"content": raw}, "finish_reason": "stop"}],
        "usage": usage if usage is not None else {
            "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
            "completion_tokens_details": {"reasoning_tokens": 2},
        },
    }).encode()


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.left = root / "left.png"
        self.right = root / "right.png"
        Image.new("RGB", (12, 8), "#cc5544").save(self.left)
        Image.new("RGB", (8, 12), "#4477cc").save(self.right)
        self.left_id = runner._image_id(self.left)
        self.right_id = runner._image_id(self.right)

    def tearDown(self):
        self.temp.cleanup()

    def test_side_reversal_and_explicit_reasoning_payload(self):
        model = runner.MODELS[0]
        requests = []

        def open_url(request, timeout):
            requests.append(json.loads(request.data))
            return _Response(_body(model))

        with mock.patch.object(runner.urllib.request, "urlopen", side_effect=open_url):
            first = runner._view(key="secret-not-recorded", model=model, brief="Paint a form.",
                                 left=self.left, right=self.right,
                                 left_id=self.left_id, right_id=self.right_id)
            second = runner._view(key="secret-not-recorded", model=model, brief="Paint a form.",
                                  left=self.right, right=self.left,
                                  left_id=self.right_id, right_id=self.left_id)
        self.assertEqual(first["status"], "valid")
        self.assertEqual(first["assessment"]["winner_id"], self.left_id)
        self.assertEqual(second["assessment"]["winner_id"], self.right_id)
        self.assertEqual(requests[0]["reasoning"], {"effort": "medium"})
        self.assertEqual(requests[0]["max_tokens"], 4096)
        self.assertNotIn("temperature", requests[0])
        self.assertEqual(requests[0]["messages"][0]["content"], runner.PROMPT)
        self.assertEqual(requests[0]["response_format"], runner.rubric._response_format())

    def test_returned_model_identity_mismatch_is_censored(self):
        with mock.patch.object(
            runner.urllib.request, "urlopen",
            return_value=_Response(_body(runner.MODELS[0])),
        ):
            call = runner._view(key="secret", model=runner.MODELS[1], brief="Paint.",
                                left=self.left, right=self.right,
                                left_id=self.left_id, right_id=self.right_id)
        self.assertEqual(call["status"], "censored")
        self.assertEqual(call["parse_error"], "model_identity_mismatch")
        self.assertNotIn("secret", json.dumps(call))

    def test_http_error_body_is_not_recorded(self):
        error = urllib.error.HTTPError("https://gateway.invalid", 400, "Bad Request", {}, io.BytesIO(b"provider-secret"))
        with mock.patch.object(runner.urllib.request, "urlopen", side_effect=error):
            call = runner._view(key="secret", model=runner.MODELS[0], brief="Paint.",
                                left=self.left, right=self.right,
                                left_id=self.left_id, right_id=self.right_id)
        self.assertEqual(call["provider_error"], "HTTPError")
        self.assertNotIn("provider-secret", json.dumps(call))

    def test_strict_parser_and_order_conflict_are_censored(self):
        bad = {"winner": "LEFT"}
        with self.assertRaises(ValueError):
            runner.rubric._parse(json.dumps(bad), (self.left_id, self.right_id))
        manifest = {
            "manifest_content_sha256": "m", "prompt_sha256": "p",
            "rubric_source_sha256": "r", "runner_source_sha256": "s",
            "rubric_format_version": "v", "request_config_sha256": "c",
            "response_schema_sha256": "f", "reasoning": {"effort": "medium"},
            "temperature": 0, "max_tokens": 4096,
        }
        pair = {
            "pair_id": "pair-1", "case_id": "x", "brief_sha256": "b",
            "left": {"sha256": "l", "image_id": self.left_id},
            "right": {"sha256": "r", "image_id": self.right_id},
        }
        def row(order: str, winner: str):
            return {"identity": {"order": order}, "status": "valid", "call": {
                "assessment": {"winner_id": winner, "preference": "slight"},
            }}
        result = runner._round_result(
            manifest=manifest, pair=pair, model=runner.MODELS[0], round_index=0,
            calls=[row("canonical", self.left_id), row("reverse", self.right_id)],
        )
        self.assertEqual(result["status"], "censored")
        self.assertEqual(result["reason"], "order_conflict")
        self.assertIsNone(result["winner_id"])


class ManifestAndSummaryTests(unittest.TestCase):
    def test_round_resume_reuses_matching_round_receipt(self):
        panel = HERE / "local-evidence" / "pairs.json"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = runner.prepare_manifest(panel_path=panel, output=root / "manifest.json", fixed_repeat_rounds=1)
            pair = manifest["pairs"][0]
            output = root / "output"
            sem = __import__("asyncio").Semaphore(2)
            lock = __import__("threading").Lock()
            calls = 0

            def fake_view(*, key, model, brief, left, right, left_id, right_id):
                nonlocal calls
                calls += 1
                return {
                    "schema": runner.CALL_SCHEMA, "status": "valid",
                    "image_ids": {"LEFT": left_id, "RIGHT": right_id},
                    "requested_model": model, "returned_model": model,
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "reasoning_tokens": 1},
                    "elapsed_seconds": 0.01, "finish_reason": "stop",
                    "assessment": {"winner_id": pair["left"]["image_id"], "preference": "slight"},
                }

            async def run_twice():
                with mock.patch.object(runner, "_view", side_effect=fake_view):
                    first = await runner._one_round(
                        manifest=manifest, pair=pair, model=runner.MODELS[0], round_index=0,
                        output=output, key="key", progress_lock=lock, sem=sem,
                    )
                    second = await runner._one_round(
                        manifest=manifest, pair=pair, model=runner.MODELS[0], round_index=0,
                        output=output, key="key", progress_lock=lock, sem=sem,
                    )
                return first, second

            first, second = __import__("asyncio").run(run_twice())
            self.assertEqual(calls, 2)
            self.assertEqual(first["identity"], second["identity"])
            self.assertEqual(first["status"], "scored")

    def test_manifest_resume_identity_and_usage_missing(self):
        panel = HERE / "local-evidence" / "pairs.json"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.json"
            manifest = runner.prepare_manifest(panel_path=panel, output=path, fixed_repeat_rounds=1)
            self.assertEqual(manifest["planned_call_count"], 180)
            loaded = runner.load_manifest(path)
            loaded["models"][0] = "changed/model"
            path.write_text(json.dumps(loaded), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest identity"):
                runner.load_manifest(path)

        pair = manifest["pairs"][0]
        rows = []
        for order, completion, reasoning in (("canonical", 5, 2), ("reverse", None, None)):
            rows.append({
                "identity": {"pair_id": pair["pair_id"], "model": runner.MODELS[0], "round": 0},
                "status": "valid", "_pair": pair,
                "call": {"elapsed_seconds": 1.0, "usage": {
                    "prompt_tokens": 11, "completion_tokens": completion,
                    "reasoning_tokens": reasoning,
                }, "assessment": {"winner_id": pair["left"]["image_id"]}},
            })
        metrics = runner._stats_for_rows(rows, 1)
        self.assertEqual(metrics["tokens"]["reasoning"]["per_call"]["observed"], 1)
        self.assertEqual(metrics["tokens"]["reasoning"]["per_call"]["missing"], 1)
        self.assertEqual(metrics["tokens"]["completion"]["per_pair"]["observed"], 0)
        self.assertEqual(metrics["tokens"]["completion"]["per_pair"]["partial_pairs"], 1)


if __name__ == "__main__":
    unittest.main()
