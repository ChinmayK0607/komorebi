from __future__ import annotations

import asyncio
import base64
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "frozen_model_pair_repeat_runner", HERE / "calibrate_frozen_model_pair_repeats.py"
)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

from painter import text_pairwise_consistent_side_judge_v2_model as transport  # noqa: E402


def _assessment() -> dict:
    return {
        "brief_fidelity": "clear",
        "visible_evidence": "The connected subject is readable.",
        "weaknesses": ["Some edges are soft."],
    }


def _body(model: str, *, winner: str = "LEFT") -> bytes:
    preference = "tied" if winner == "TIE" else "slight"
    raw = json.dumps({
        "left_assessment": _assessment(), "right_assessment": _assessment(),
        "reason": "The LEFT panel has the stronger connected form.",
        "winner": winner, "preference": preference,
    })
    return json.dumps({
        "model": model,
        "choices": [{"message": {"content": raw}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 41, "completion_tokens": 19, "total_tokens": 60,
            "completion_tokens_details": {"reasoning_tokens": 7},
        },
    }).encode()


class _Response:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body


class ModelTransportTests(unittest.TestCase):
    def test_payload_binds_requested_model_schema_and_swapped_ids(self):
        model = "xiaomi/mimo-v2.6-pro"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            left, right = root / "left.png", root / "right.png"
            Image.new("RGB", (12, 8), "#d35f4f").save(left)
            Image.new("RGB", (8, 12), "#4f75d3").save(right)
            ids = (transport.image_id(left), transport.image_id(right))
            requests = []

            def open_url(request, timeout):
                requests.append((json.loads(request.data), timeout))
                return _Response(_body(model))

            with mock.patch("urllib.request.urlopen", side_effect=open_url):
                first = transport._view(
                    key="external", model=model, brief="Paint a connected subject.",
                    a=left, b=right, a_id=ids[0], b_id=ids[1],
                )
                second = transport._view(
                    key="external", model=model, brief="Paint a connected subject.",
                    a=right, b=left, a_id=ids[1], b_id=ids[0],
                )
            self.assertEqual(len(requests), 2)
            for payload, timeout in requests:
                self.assertEqual(payload["model"], model)
                self.assertEqual(payload["temperature"], 0)
                self.assertEqual(payload["response_format"], transport._response_format())
                self.assertNotIn("reasoning", payload)
                self.assertEqual(payload["messages"][0]["content"], transport.PROMPT)
                sheet_url = payload["messages"][1]["content"][1]["image_url"]["url"]
                sheet = Image.open(io.BytesIO(base64.b64decode(sheet_url.split(",", 1)[1])))
                self.assertEqual(sheet.size, (1248, 696))
                self.assertEqual(timeout, 240)
            self.assertEqual(first["requested_model"], model)
            self.assertEqual(first["returned_model"], model)
            self.assertEqual(first["observed_reasoning_tokens"], 7)
            self.assertEqual(first["image_ids"], {"A": ids[0], "B": ids[1]})
            self.assertEqual(second["image_ids"], {"A": ids[1], "B": ids[0]})
            self.assertEqual(first["assessment"]["winner_id"], ids[0])
            self.assertEqual(second["assessment"]["winner_id"], ids[1])

    def test_returned_model_mismatch_is_parse_failure_not_a_score(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            left, right = root / "left.png", root / "right.png"
            Image.new("RGB", (12, 8), "#d35f4f").save(left)
            Image.new("RGB", (8, 12), "#4f75d3").save(right)
            ids = (transport.image_id(left), transport.image_id(right))
            with mock.patch(
                "urllib.request.urlopen",
                return_value=_Response(_body("xiaomi/mimo-v2.6-flash")),
            ):
                call = transport._view(
                    key="external", model="xiaomi/mimo-v2.6-pro", brief="Paint a connected subject.",
                    a=left, b=right, a_id=ids[0], b_id=ids[1],
                )
            self.assertEqual(call["requested_model"], "xiaomi/mimo-v2.6-pro")
            self.assertEqual(call["returned_model"], "xiaomi/mimo-v2.6-flash")
            self.assertEqual(call["parse_error"], "model_identity_mismatch")
            self.assertNotIn("assessment", call)
            self.assertEqual(call["raw_response"] != "", True)


class ModelRunnerManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.anchor = self.root / "anchor.png"
        self.candidate = self.root / "candidate.png"
        Image.new("RGB", (12, 8), "#d35f4f").save(self.anchor)
        Image.new("RGB", (8, 12), "#4f75d3").save(self.candidate)
        self.pairs_file = self.root / "pairs.json"
        self.output = self.root / "comparison"
        self.pairs_file.write_text(json.dumps({
            "selection_note": "One calibration and one confirmation pair; no gold labels.",
            "pairs": [
                {"pair_id": "cal-1", "split": "calibration", "brief": "Paint a connected subject.",
                 "anchor_path": str(self.anchor), "candidate_path": str(self.candidate)},
                {"pair_id": "conf-1", "split": "confirmation", "brief": "Paint another connected subject.",
                 "anchor_path": str(self.anchor), "candidate_path": str(self.candidate)},
            ],
        }, indent=2), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_manifest_binds_models_and_rejects_changed_expected_model(self):
        manifest = runner.prepare_manifest(
            pairs_file=self.pairs_file, output=self.output, repeat_count=2,
            baseline_model="xiaomi/mimo-v2.6-flash",
            candidate_model="xiaomi/mimo-v2.6-pro",
        )
        self.assertEqual(manifest["models"]["candidate"], "xiaomi/mimo-v2.6-pro")
        self.assertEqual(manifest["reasoning"]["mode"], "provider_default")
        with self.assertRaisesRegex(ValueError, "manifest model configuration differs"):
            runner.read_manifest(
                self.output / "input-manifest.json",
                expected_models={
                    "baseline": "xiaomi/mimo-v2.6-flash",
                    "candidate": "xiaomi/mimo-v2.6-flash",
                },
            )

    def test_stale_model_receipt_cannot_resume(self):
        manifest = runner.prepare_manifest(
            pairs_file=self.pairs_file, output=self.output, repeat_count=1,
            baseline_model="xiaomi/mimo-v2.6-flash",
            candidate_model="xiaomi/mimo-v2.6-pro",
        )
        pair = manifest["pairs"][0]
        path = self.output / "receipts" / "baseline" / "cal-1--repeat-1.json"
        path.parent.mkdir(parents=True)
        identity = runner._identity(manifest, pair, "baseline", 1)
        identity["model"] = "xiaomi/mimo-v2.6-pro"
        path.write_text(json.dumps({"schema": runner.RECEIPT_SCHEMA, "identity": identity}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "stale receipt"):
            asyncio.run(runner._one(
                manifest=manifest, pair=pair, method="baseline", repeat=1,
                output=self.output, key_file=self.root / "unused-key",
                sem=asyncio.Semaphore(1),
            ))

    def test_mocked_repeats_report_usage_latency_order_and_scope(self):
        manifest = runner.prepare_manifest(
            pairs_file=self.pairs_file, output=self.output, repeat_count=2,
            baseline_model="xiaomi/mimo-v2.6-flash",
            candidate_model="xiaomi/mimo-v2.6-pro",
        )

        async def fake_judge(*, brief, anchor, candidate, key_file, model):
            pair = next(row for row in manifest["pairs"] if row["brief_sha256"] == transport.digest(brief))
            method = "baseline" if model == manifest["models"]["baseline"] else "candidate"
            winner = pair["image_ids"]["anchor"]
            call = {
                "requested_model": model, "returned_model": model,
                "request_config_sha256": transport._model_config_sha256(model),
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                          "completion_tokens_details": {"reasoning_tokens": 2}},
                "elapsed_seconds": 0.125,
                "image_ids": {"A": pair["image_ids"]["anchor"], "B": pair["image_ids"]["candidate"]},
            }
            view = {"assessment": {"winner_id": winner, "preference": "slight"}, "calls": [call]}
            return {
                "protocol": manifest["protocol"], "format_version": manifest["format_version"],
                "source_format_sha256": manifest["source_format_sha256"],
                "rubric_source_format_sha256": manifest["rubric_source_format_sha256"],
                "prompt_sha256": manifest["prompt_sha256"], "requested_model": model,
                "model_config_sha256": transport._model_config_sha256(model),
                "brief_sha256": pair["brief_sha256"], "image_ids": pair["image_ids"],
                "image_sha256": pair["image_sha256"], "status": "scored",
                "winner_id": winner, "preference": "slight", "score": 1 / 3,
                "initial_votes": [winner, winner], "recheck_votes": [],
                "views": [view, view], "recheck_views": [], "returned_models": [model],
            }

        with mock.patch.object(runner.model_judge, "judge_pair", side_effect=fake_judge):
            summary = asyncio.run(runner.run(
                manifest_path=self.output / "input-manifest.json", output=self.output,
                key_file=self.root / "unused-key", baseline_model="xiaomi/mimo-v2.6-flash",
                candidate_model="xiaomi/mimo-v2.6-pro", concurrency=2,
            ))
        self.assertEqual(summary["pair_count"], 2)
        self.assertEqual(summary["models"]["candidate"], "xiaomi/mimo-v2.6-pro")
        self.assertEqual(summary["metrics"]["baseline"]["total"]["api_call_count"], 8)
        self.assertEqual(summary["metrics"]["baseline"]["total"]["latency_seconds"]["mean"], 0.125)
        self.assertEqual(summary["metrics"]["baseline"]["total"]["initial_order_agreement_count"], 4)
        self.assertEqual(summary["total_repeat_consistency"]["candidate"]["winner_consistency_rate"], 1.0)
        self.assertEqual(summary["paired_model_comparison"]["total"]["winner_agreement"], 4)
        self.assertFalse(summary["gold_labels"])
        self.assertIsNotNone(summary["bootstrap_diagnostics"]["cross_model_winner_agreement_95pct"])

    def test_end_to_end_real_judge_pair_with_mock_http_and_key(self):
        runner.prepare_manifest(
            pairs_file=self.pairs_file, output=self.output, repeat_count=2,
            baseline_model="xiaomi/mimo-v2.6-flash",
            candidate_model="xiaomi/mimo-v2.6-pro",
        )
        requests = []

        def open_url(request, timeout):
            payload = json.loads(request.data)
            requests.append((payload, timeout))
            return _Response(_body(payload["model"], winner="TIE"))

        with mock.patch.object(runner.model_judge, "load_key", return_value="external"), \
                mock.patch("urllib.request.urlopen", side_effect=open_url):
            summary = asyncio.run(runner.run(
                manifest_path=self.output / "input-manifest.json", output=self.output,
                key_file=self.root / "unused-key", baseline_model="xiaomi/mimo-v2.6-flash",
                candidate_model="xiaomi/mimo-v2.6-pro", concurrency=2,
            ))
        self.assertEqual(len(requests), 16)  # 2 pairs × 2 models × 2 repeats × 2 orders
        self.assertEqual(summary["metrics"]["baseline"]["total"]["scored_count"], 4)
        self.assertEqual(summary["metrics"]["candidate"]["total"]["scored_count"], 4)
        self.assertEqual(summary["metrics"]["baseline"]["total"]["implementation_error_count"], 0)
        self.assertEqual(summary["metrics"]["candidate"]["total"]["censored_count"], 0)
        for payload, timeout in requests:
            self.assertIn(payload["model"], {"xiaomi/mimo-v2.6-flash", "xiaomi/mimo-v2.6-pro"})
            self.assertEqual(payload["temperature"], 0)
            self.assertEqual(payload["response_format"], transport._response_format())
            self.assertEqual(timeout, 240)
        receipt = json.loads((self.output / "receipts" / "candidate" / "cal-1--repeat-1.json").read_text())
        self.assertEqual(receipt["judge"]["requested_model"], "xiaomi/mimo-v2.6-pro")
        self.assertEqual(receipt["judge"]["model_config_sha256"], transport._model_config_sha256("xiaomi/mimo-v2.6-pro"))
        self.assertEqual(receipt["judge"]["returned_models"], ["xiaomi/mimo-v2.6-pro"])


if __name__ == "__main__":
    unittest.main()
