import asyncio
import importlib.util
import tempfile
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).parent


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


split = _load("split_judge_test_target", "../../text_pairwise_grounded_v2_split_judge.py")
runner = _load("split_runner_test_target", "run_split_ablation.py")


def _png(path: Path, value: bytes = b"x") -> None:
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + value)


def _pair() -> dict:
    return {
        "pair_id": "t600-413__rl-step-16__pilot-step-24",
        "case_id": "t600-413",
        "brief": "brief",
        "brief_sha256": runner.ja.sha256_text("brief"),
        "anchor_role": "RL-step-16",
        "candidate_role": "pilot-step-24",
        "images": {
            "RL-step-16": {"path": "a.png", "sha256": "a" * 64},
            "pilot-step-24": {"path": "b.png", "sha256": "b" * 64},
        },
    }


class SplitAblationTest(unittest.TestCase):
    def test_malformed_non_png_is_rejected_before_request(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.png"
            path.write_bytes(b"not an image")
            with self.assertRaises(ValueError):
                split._data_url(path)

    def test_censored_and_malformed_results_never_become_zero(self):
        with self.assertRaises((ValueError, KeyError, TypeError)):
            split._label("{\"winner\":\"A\"}")
        pair = _pair()
        censored = runner._split_flat({"status": "censored", "reason": "judge_failure_ValueError"}, pair)
        self.assertEqual(censored["status"], "censored")
        self.assertNotIn("score", censored)
        malformed = runner._split_flat({"status": "scored", "judge": {"score": None, "views": []}}, pair)
        self.assertEqual(malformed["status"], "scored")
        self.assertIsNone(malformed["score"])

    def test_malformed_winner_strength_response_is_rejected_without_api(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"choices":[{"message":{"content":"{\\"winner\\":\\"A\\"}"},"finish_reason":"stop"}]}'

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a, b = root / "a.png", root / "b.png"
            _png(a, b"a")
            _png(b, b"b")
            with mock.patch.object(split.urllib.request, "urlopen", return_value=Response()):
                with self.assertRaises(ValueError):
                    split._view(key="test-key", brief="brief", a=a, b=b)

    def test_split_two_order_score_and_cap_are_v2_compatible(self):
        calls = 0

        async def fake_view(**kwargs):
            nonlocal calls
            calls += 1
            winner = "B" if calls == 1 else "A"

            return {
                "label": {
                    "winner": winner,
                    "strength": 7 if winner == "B" else 1,
                    "brief_fidelity_A": "partial",
                    "brief_fidelity_B": "missing",
                    "visible_evidence_A": "A visible",
                    "visible_evidence_B": "B visible",
                    "reason": "Specific visible difference",
                    "weaknesses_A": ["A weakness"],
                    "weaknesses_B": ["B weakness"],
                },
                "image_sha256": {"A": "a" * 64, "B": "b" * 64},
                "format_version": split.FORMAT_VERSION,
                "usage": {"total_tokens": 2},
            }

        original_retry, original_key = split._retry_view, split.load_key
        split._retry_view = fake_view
        split.load_key = lambda path: "test-key"
        try:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                brief, anchor, candidate = root / "brief.txt", root / "a.png", root / "b.png"
                brief.write_text("brief", encoding="utf-8")
                _png(anchor, b"a")
                _png(candidate, b"b")
                result = asyncio.run(split.judge_pair(brief=brief, anchor=anchor, candidate=candidate, key_file=root / "unused"))
                self.assertEqual(result["protocol"], split.PROTOCOL)
                self.assertEqual(result["format_version"], split.FORMAT_VERSION)
                self.assertEqual(result["raw_pairwise_score"], 1.0)
                self.assertEqual(result["score"], 0.5)
                self.assertEqual(result["candidate_margin_cap"], 0.5)
        finally:
            split._retry_view, split.load_key = original_retry, original_key

    def test_identity_binds_format_prompt_manifest_and_both_images(self):
        pair = _pair()
        manifest = {
            "manifest_content_sha256": "manifest-a",
            "source_pass1_manifest_content_sha256": "pass1-a",
            "protocol": split.PROTOCOL,
            "model": split.MODEL,
            "prompt_sha256": runner.ja.sha256_text(split.PROMPT),
            "format_version": split.FORMAT_VERSION,
        }
        first = runner._identity(manifest, pair)
        changed = dict(pair, images={**pair["images"], "pilot-step-24": {"path": "b.png", "sha256": "c" * 64}})
        self.assertNotEqual(first, runner._identity(manifest, changed))
        self.assertNotEqual(first, runner._identity(dict(manifest, format_version="other"), pair))
        self.assertNotEqual(first, runner._identity(dict(manifest, prompt_sha256="changed"), pair))


if __name__ == "__main__":
    unittest.main()
