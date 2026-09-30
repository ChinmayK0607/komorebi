import asyncio
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE = Path(__file__).with_name("judge_alignment.py")
SPEC = importlib.util.spec_from_file_location("judge_alignment_test_target", MODULE)
ja = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ja)


def _entry(role: str, digest: str, old_score=None) -> dict:
    return {
        "path": f"images/t600-413-{role}.png",
        "sha256": digest,
        "exists": True,
        "status": "valid",
        "old_score": old_score,
        "old_score_protocol": "text-paint-sft-pairwise-margin-v1",
    }


def _case_and_pair() -> tuple[dict, dict]:
    case = {
        "id": "t600-413",
        "family": "complex",
        "brief": "A test brief",
        "brief_sha256": ja.sha256_text("A test brief"),
        "images": {
            "RL-step-16": _entry("RL-step-16", "a" * 64, 0.3),
            "pilot-step-24": _entry("pilot-step-24", "b" * 64, 0.7),
        },
    }
    return case, ja._make_pair(case, "RL-step-16", "pilot-step-24", "human_panel")


def _result(*, score: float, raw: float, cap: float, protocol: str) -> dict:
    def view(winner: str, strength: int) -> dict:
        return {
            "label": {
                "winner": winner,
                "strength": strength,
                "brief_fidelity_A": "partial",
                "brief_fidelity_B": "partial",
                "visible_evidence_A": "visible A",
                "visible_evidence_B": "visible B",
                "reason": "Specific visible comparison",
                "weaknesses_A": ["A weakness"],
                "weaknesses_B": ["B weakness"],
            },
            "elapsed_seconds": 0.1,
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    return {
        "protocol": protocol,
        "status": "scored",
        "score": score,
        "raw_pairwise_score": raw,
        "candidate_margin_cap": cap,
        "order_gap": 1 / 6,
        # B in forward order and A in reverse order both mean candidate.
        "views": [view("B", 5), view("A", 5)],
    }


class JudgeAlignmentTest(unittest.TestCase):
    def test_gallery_parser_drops_missing_canvas_and_freezes_image_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "images" / "t600-413-RL-step-16.png"
            image.parent.mkdir()
            image.write_bytes(b"frozen-image")
            gallery = root / "index.html"
            gallery.write_text(
                "<section><h2>t600-413 · complex</h2>"
                "<p class='brief'>A brief</p><div class='row'>"
                "<div class='cell'><div class='missing'>No valid canvas</div></div>"
                "<div class='cell'><img src='images/t600-413-RL-step-16.png'>"
                "<p><strong>RL-step-16</strong> · valid · score 0.500</p></div>"
                "</div></section>", encoding="utf-8"
            )
            cases = ja.parse_gallery(gallery)
            self.assertEqual(len(cases), 1)
            row = cases[0]["images"]["RL-step-16"]
            self.assertEqual(row["sha256"], ja.sha256_file(image))
            self.assertTrue(row["exists"])

    def test_receipt_identity_binds_both_prompt_and_image_hashes(self):
        case, pair = _case_and_pair()
        manifest = {
            "manifest_content_sha256": "manifest-a",
            "protocol": ja.judge.PROTOCOL,
            "old_protocol": ja.old_judge.PROTOCOL,
            "model": ja.MODEL,
            "prompt_sha256": "new-prompt-a",
            "old_prompt_sha256": "old-prompt-a",
        }
        first = ja._pair_identity(manifest, pair, 1)
        changed_pair = json.loads(json.dumps(pair))
        changed_pair["images"]["pilot-step-24"]["sha256"] = "c" * 64
        second = ja._pair_identity(manifest, changed_pair, 1)
        self.assertNotEqual(first, second)
        changed_manifest = dict(manifest, prompt_sha256="new-prompt-b")
        self.assertNotEqual(first, ja._pair_identity(changed_manifest, pair, 1))

    def test_failed_versions_are_censored_without_a_zero_score(self):
        case, pair = _case_and_pair()
        receipt = {
            "schema": ja.RECEIPT_SCHEMA,
            "identity": {"pair_id": pair["pair_id"], "repeat": 1},
            "status": "censored",
            "versions": {
                "old": {"status": "censored", "reason": "judge_failure_RuntimeError"},
                "new": {"status": "censored", "reason": "judge_failure_TimeoutError"},
            },
        }
        row = ja._flatten_receipt(receipt, pair)
        self.assertEqual(row["old"]["status"], "censored")
        self.assertEqual(row["new"]["status"], "censored")
        self.assertNotIn("score", row["old"])
        self.assertNotIn("score", row["new"])
        self.assertNotIn("secret", json.dumps(row).lower())

    def test_human_counts_use_explicit_labels_and_orientation_is_invertible(self):
        case, pair = _case_and_pair()
        manifest = {
            "protocol": ja.judge.PROTOCOL,
            "old_protocol": ja.old_judge.PROTOCOL,
            "model": ja.MODEL,
            "prompt_sha256": "new",
            "old_prompt_sha256": "old",
            "manifest_content_sha256": "manifest",
            "gallery_html_sha256": "gallery",
            "historical_score_note": "context only",
            "cases": [case],
            "human_feedback": ja.HUMAN_FEEDBACK,
            "pair_plan": {"requested_pair_count_by_case": {case["id"]: 1}, "missing_pairs_by_case": {}},
        }
        old = _result(score=0.2, raw=0.2, cap=0.5, protocol=ja.old_judge.PROTOCOL)
        new = _result(score=0.5, raw=0.8, cap=0.5, protocol=ja.judge.PROTOCOL)
        receipt = {
            "schema": ja.RECEIPT_SCHEMA,
            "identity": {"pair_id": pair["pair_id"], "repeat": 1},
            "status": "scored",
            "versions": {
                "old": {"status": "scored", "judge": old},
                "new": {"status": "scored", "judge": new},
            },
        }
        summary = ja.summarize(manifest, [pair], [receipt], gallery=Path("gallery.html"), output=Path("out"), repeats=1)
        self.assertEqual(summary["human_agreement"]["old"]["agreements"], 1)
        self.assertEqual(summary["human_agreement"]["new"]["agreements"], 0)
        self.assertEqual(summary["human_agreement"]["new"]["raw_direction_agreements"], 0)
        self.assertEqual(summary["human_agreement"]["new"]["two_order_unanimous_denominator"], 1)
        self.assertEqual(summary["human_agreement"]["new"]["two_order_unanimous_agreements"], 0)
        self.assertEqual(summary["versions"]["new"]["two_order_label_consensus"]["consensus_count"], 1)
        self.assertEqual(summary["versions"]["new"]["two_order_label_consensus"]["disagreement_count"], 0)
        new_row = summary["pairs"][0]["repeats"][0]["new"]
        self.assertEqual(new_row["normalized_label_votes"], ["pilot-step-24", "pilot-step-24"])
        self.assertTrue(new_row["two_order_label_consensus"])
        self.assertEqual(new_row["raw_preferred_role"], "pilot-step-24")
        self.assertEqual(new_row["capped_preferred_role"], "tie")


if __name__ == "__main__":
    unittest.main()
