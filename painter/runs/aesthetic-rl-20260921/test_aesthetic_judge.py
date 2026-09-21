"""Offline tests for the aesthetic judge contract and queue.

These tests use bytes as image stand-ins.  They do not render, call a model, or
require network access.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import aesthetic_judge as judge
import judge_queue as queue


def label(winner="A", reason="visible comparison"):
    return {
        "winner": winner,
        "reason": reason,
        "weaknesses_A": [],
        "weaknesses_B": ["less coherent visible brushwork"],
    }


class AestheticJudgeTest(unittest.TestCase):
    def test_schema_is_exact_and_empty_weaknesses_are_valid(self):
        row = {
            "winner": "tie",
            "reason": "Both paintings have comparable visible strengths.",
            "weaknesses_A": [],
            "weaknesses_B": [],
        }
        self.assertEqual(judge.validate_judgment(row), row)
        self.assertEqual(judge.parse_judgment(json.dumps(row)), row)

    def test_unknown_missing_and_malformed_schema_is_rejected(self):
        valid = label()
        for row in (
            {**valid, "extra": "no"},
            {key: value for key, value in valid.items() if key != "reason"},
            {**valid, "reason": ["not a string"]},
            {**valid, "weaknesses_A": [3]},
            {**valid, "weaknesses_B": "none"},
            {**valid, "winner": "A", "reason": ""},
        ):
            with self.assertRaises(ValueError):
                judge.validate_judgment(row)
        with self.assertRaises(ValueError):
            judge.parse_judgment("```json\n" + json.dumps(valid) + "\n```")
        with self.assertRaises(ValueError):
            judge.parse_judgment('{"winner":"A","winner":"B","reason":"x",'
                                 '"weaknesses_A":[],"weaknesses_B":[]}')

    def test_bidirectional_reward_averages_without_rejecting_disagreement(self):
        # Candidate is A in the first view and B in the reversed view.
        self.assertEqual(judge.pairwise_win_fraction(label("A"), label("B")), 1.0)
        self.assertEqual(judge.pairwise_win_fraction(label("tie"), label("tie")), 0.5)
        # Both judges selecting displayed A is a directional disagreement after
        # reversal, and therefore averages to 0.5 instead of being rejected.
        self.assertEqual(judge.pairwise_win_fraction(label("A"), label("A")), 0.5)
        self.assertIsNone(judge.pairwise_win_fraction(label("uncertain"), label("B")))
        self.assertIsNone(judge.pairwise_win_fraction(label("A"), None))

    def test_reversing_candidate_side_preserves_reward(self):
        self.assertEqual(
            judge.pairwise_win_fraction(
                label("B"), label("A"), first_candidate_side="B", reverse_candidate_side="A"
            ),
            1.0,
        )

    def test_normalized_reward_has_no_action_or_finish_bonus(self):
        normalized = judge.reconcile(label("A"), label("B"))
        self.assertEqual(normalized, {
            "protocol_version": judge.PROTOCOL_VERSION,
            "pairwise_win_fraction": 1.0,
        })
        self.assertEqual(judge.reward(normalized), 1.0)
        self.assertIsNone(judge.reward({"protocol_version": judge.PROTOCOL_VERSION,
                                        "pairwise_win_fraction": None}))


class AestheticQueueTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = {}
        for role, data in {
            "reference": b"reference-image",
            "anchor": b"anchor-image",
            "candidate": b"candidate-image",
        }.items():
            path = self.root / f"{role}.png"
            path.write_bytes(data)
            self.source[role] = path
        hashes = {role: queue.sha(path) for role, path in self.source.items()}
        request_id = queue.request_key(hashes)
        request = {
            "id": request_id,
            "protocol_version": judge.PROTOCOL_VERSION,
            "images": {role: path.name for role, path in self.source.items()},
            "image_sha256": hashes,
            "created_at": 1.0,
        }
        self.request = queue.create_or_verify_request(
            self.root / "judge-queue" / request_id / "request.json", request
        )
        self.batch = "test"
        queue.export(self.root, self.batch, limit=1)
        queue.stage(self.root, self.batch)
        self.destination = self.root / "reward-batches" / self.batch

    def _write_orientation_labels(self, winner_a="A", winner_b="B"):
        mapping = {
            row["id"]: row
            for row in json.loads((self.destination / "private-key.json").read_text())
        }
        sources = sorted(
            path for path in self.destination.glob("judge-*.json")
            if path.name != "judge-manifest.json"
        )
        for index, source in enumerate(sources):
            document = json.loads(source.read_text())
            comparison = document["comparisons"][0]
            winner = winner_a if mapping[comparison["id"]]["candidate_side"] == "A" else winner_b
            row = {"id": comparison["id"], **label(winner)}
            queue.write(self.destination / f"labels-{index:02d}.json", {
                "protocol_version": judge.PROTOCOL_VERSION,
                "judge_model": judge.JUDGE_MODEL,
                "reasoning_effort": judge.REASONING_EFFORT,
                "judgments": [row],
            })

    def test_export_has_neutral_hashes_and_private_orientation_mapping(self):
        pairs = json.loads((self.destination / "pairs.json").read_text())
        private = json.loads((self.destination / "private-key.json").read_text())
        self.assertEqual(len(pairs), 2)
        self.assertEqual({row["candidate_side"] for row in private}, {"A", "B"})
        for pair, mapping in zip(pairs, private):
            self.assertEqual(len(pair["id"]), 64)
            self.assertTrue(all(character in "0123456789abcdef" for character in pair["id"]))
            self.assertEqual(pair["protocol_version"], judge.PROTOCOL_VERSION)
            self.assertEqual(set(pair["image_sha256"]), {"reference", "A", "B"})
            self.assertEqual(pair["id"], mapping["id"])
            for role in ("reference", "A", "B"):
                self.assertEqual(queue.sha(self.destination / pair[role]), pair["image_sha256"][role])
                self.assertTrue(Path(pair[role]).name.startswith(pair["image_sha256"][role]))
        self.assertEqual(
            json.loads((self.destination / "mapping.json").read_text()), private
        )

    def test_receipt_imports_two_labels_and_installs_pairwise_reward(self):
        self._write_orientation_labels()
        summary = queue.receipts(self.root, self.batch)
        self.assertEqual(summary["ready_requests"], 1)
        receipt_path = next((self.destination / "receipts").glob("*/receipt.json"))
        receipt = json.loads(receipt_path.read_text())
        self.assertEqual(receipt["pairwise_win_fraction"], 1.0)
        self.assertEqual(receipt["reward"], 1.0)
        self.assertEqual(receipt["judgment"]["protocol_version"], judge.PROTOCOL_VERSION)
        result = queue.install(self.root, self.batch)
        self.assertEqual(result["installed"], 1)
        installed = json.loads((self.root / "judge-queue" / self.request["id"] / "receipt.json").read_text())
        self.assertEqual(installed, receipt)

    def test_uncertain_receipt_is_excluded_and_missing_orientation_stays_pending(self):
        self._write_orientation_labels(winner_a="uncertain", winner_b="B")
        summary = queue.receipts(self.root, self.batch)
        self.assertEqual(summary["ready_requests"], 1)
        self.assertEqual(summary["excluded_requests"], 1)
        receipt = json.loads(next((self.destination / "receipts").glob("*/receipt.json")).read_text())
        self.assertIsNone(receipt["reward"])

        # A fresh batch with one label has no receipt and therefore no zero.
        root = Path(self.temp.name) / "missing"
        root.mkdir()
        for role, path in self.source.items():
            (root / path.name).write_bytes(path.read_bytes())
        hashes = {role: queue.sha(root / path.name) for role, path in self.source.items()}
        ident = queue.request_key(hashes)
        queue.create_or_verify_request(root / "judge-queue" / ident / "request.json", {
            "id": ident, "protocol_version": judge.PROTOCOL_VERSION,
            "images": {role: path.name for role, path in self.source.items()},
            "image_sha256": hashes,
        })
        queue.export(root, "missing", 1)
        queue.stage(root, "missing")
        destination = root / "reward-batches" / "missing"
        source = next(
            path for path in destination.glob("judge-*.json")
            if path.name != "judge-manifest.json"
        )
        row = json.loads(source.read_text())["comparisons"][0]
        queue.write(destination / "labels-one.json", {
            "protocol_version": judge.PROTOCOL_VERSION,
            "judge_model": judge.JUDGE_MODEL,
            "reasoning_effort": judge.REASONING_EFFORT,
            "judgments": [{"id": row["id"], **label("A")}],
        })
        summary = queue.receipts(root, "missing")
        self.assertEqual(summary["ready_requests"], 0)
        self.assertFalse((destination / "receipts" / ident / "receipt.json").exists())

    def test_protocol_and_request_hash_binding_reject_old_or_changed_requests(self):
        hashes = self.request["image_sha256"]
        self.assertNotEqual(
            queue.request_key(hashes, "old_live_protocol"), self.request["id"]
        )
        changed = dict(self.request, protocol_version="old_live_protocol")
        with self.assertRaises(ValueError):
            queue.validate_request(self.root, changed)
        changed = dict(self.request, id="0" * 64)
        with self.assertRaises(ValueError):
            queue.validate_request(self.root, changed)

    def test_import_rejects_changed_receipt_protocol(self):
        self._write_orientation_labels()
        queue.receipts(self.root, self.batch)
        receipt_path = next((self.destination / "receipts").glob("*/receipt.json"))
        receipt = json.loads(receipt_path.read_text())
        receipt["protocol_version"] = "old_live_protocol"
        queue.write(receipt_path, receipt)
        with self.assertRaises(ValueError):
            queue.install(self.root, self.batch)


if __name__ == "__main__":
    unittest.main()
