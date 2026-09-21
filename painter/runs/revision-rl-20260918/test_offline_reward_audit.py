import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from offline_reward_audit import audit, request_key


def label(winner="A", confidence="high", acceptable_A="yes", acceptable_B="no",
          reason="raw label reason"):
    rolemap = {"reference": "asset-reference", "A": "asset-candidate",
               "B": "asset-anchor"}
    return {
        "id": "view",
        "rolemap": rolemap,
        "observations": {},
        "winner": winner,
        "confidence": confidence,
        "acceptable_A": acceptable_A,
        "acceptable_B": acceptable_B,
        "reason": reason,
        "displayed_asset_ids": list(rolemap.values()),
    }


class OfflineRewardAuditTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="offline-reward-audit-test-")
        self.root = Path(self.temp.name)
        state_dir = self.root / "rl-states"
        state_dir.mkdir()
        (state_dir / "reference.png").write_bytes(b"reference")
        (state_dir / "canvas.png").write_bytes(b"shared canvas")
        (state_dir / "candidate.png").write_bytes(b"candidate")
        (state_dir / "mismatch.png").write_bytes(b"mismatch candidate")
        (state_dir / "ground.png").write_bytes(b"grounding candidate")
        (state_dir / "program.js").write_text("function draw(){}\n")

    def tearDown(self):
        self.temp.cleanup()

    def add_rollout(self, rollout_id, action, canvas="rl-states/canvas.png",
                    valid=True, failure_kind=None):
        folder = self.root / "rl-rollouts" / rollout_id
        folder.mkdir(parents=True)
        sample = {
            "id": rollout_id,
            "state": {
                "id": "state-00",
                "source_group": "source-group-00",
                "reference": "rl-states/reference.png",
                "canvas": "rl-states/canvas.png",
                "program": "rl-states/program.js",
            },
            "text": "FINISHED" if action == "finish" else "```javascript\nfunction draw(){}\n```",
        }
        result = {
            "id": rollout_id,
            "state_id": "state-00",
            "valid": valid,
            "action": action,
            "canvas": canvas if valid else None,
            "no_change": canvas == "rl-states/canvas.png",
        }
        if failure_kind is not None:
            result["failure_kind"] = failure_kind
        (folder / "sample.json").write_text(json.dumps(sample))
        (folder / "result.json").write_text(json.dumps(result))

    def add_receipt(self, request_id, image_hashes, first=None, reverse=None,
                    expected_hashes=None, installed=False):
        if installed:
            batch = self.root / "judge-queue" / request_id
        else:
            batch = self.root / "reward-batches" / "fixture" / "receipts" / request_id
        batch.mkdir(parents=True)
        receipt = {
            "id": request_id,
            "image_sha256": expected_hashes or image_hashes,
            "judge_model": "gpt-5.6-luna",
            "reasoning_effort": "low",
        }
        if first is not None:
            receipt["candidate_A"] = first
        if reverse is not None:
            receipt["candidate_B"] = reverse
        (batch / "receipt.json").write_text(json.dumps(receipt))

    def hashes(self, canvas):
        paths = {
            "reference": self.root / "rl-states/reference.png",
            "anchor": self.root / "rl-states/canvas.png",
            "candidate": self.root / canvas,
        }
        return {
            key: hashlib.sha256(path.read_bytes()).hexdigest()
            for key, path in paths.items()
        }

    def test_join_preserves_shared_canvas_actions_and_duplicate_request(self):
        self.add_rollout("paint-shared", "paint")
        self.add_rollout("finish-shared", "finish")
        hashes = self.hashes("rl-states/canvas.png")
        rid = request_key(hashes)
        self.add_receipt(rid, hashes, label(), label(winner="B", acceptable_A="no",
                                                      acceptable_B="yes"))

        report = audit(self.root)
        rows = {row["rollout_id"]: row for row in report["rollouts"]}
        self.assertEqual(rows["paint-shared"]["action"], "paint")
        self.assertEqual(rows["finish-shared"]["action"], "finish")
        self.assertEqual(rows["paint-shared"]["request_id"], rid)
        self.assertEqual(rows["finish-shared"]["duplicate_request_count"], 2)
        self.assertEqual(rows["paint-shared"]["v1_reward"], 0.75)
        self.assertEqual(rows["paint-shared"]["v2_reward"], 1.0)
        self.assertEqual(rows["finish-shared"]["v1_reward"], 1.1)
        self.assertEqual(rows["finish-shared"]["v2_reward"], 1.0)

    def test_partial_labels_hash_mismatch_and_grounding_exclusion_are_masked(self):
        self.add_rollout("partial", "paint", "rl-states/candidate.png")
        partial_hashes = self.hashes("rl-states/candidate.png")
        partial_id = request_key(partial_hashes)
        self.add_receipt(partial_id, partial_hashes, label())

        self.add_rollout("hash-mismatch", "repair", "rl-states/mismatch.png")
        mismatch_hashes = self.hashes("rl-states/mismatch.png")
        mismatch_id = request_key(mismatch_hashes)
        bad_hashes = dict(mismatch_hashes, candidate="0" * 64)
        self.add_receipt(mismatch_id, mismatch_hashes, label(), label(winner="B",
                          acceptable_A="no", acceptable_B="yes"), expected_hashes=bad_hashes)

        self.add_rollout("grounding", "paint", "rl-states/ground.png")
        grounding_hashes = self.hashes("rl-states/ground.png")
        grounding_id = request_key(grounding_hashes)
        self.add_receipt(grounding_id, grounding_hashes, label(), label(winner="B",
                            acceptable_A="no", acceptable_B="yes"))
        exclusions = {"exclusions": [{"request_id": grounding_id,
                                       "reason": "reviewed image grounding error"}]}

        report = audit(self.root, exclusions=exclusions)
        rows = {row["rollout_id"]: row for row in report["rollouts"]}
        self.assertIsNone(rows["partial"]["v1_reward"])
        self.assertIsNone(rows["partial"]["v2_reward"])
        self.assertIn("v2_judgment_unavailable", rows["partial"]["v2_masked_reasons"])
        self.assertIsNone(rows["hash-mismatch"]["v1_reward"])
        self.assertIn("hash_mismatch", rows["hash-mismatch"]["v1_masked_reasons"])
        self.assertTrue(rows["grounding"]["excluded"])
        self.assertIn("grounding_exclusion", rows["grounding"]["v1_masked_reasons"])
        self.assertIn("grounding_exclusion", rows["grounding"]["v2_masked_reasons"])

    def test_invalid_program_scores_penalty_but_cutoff_is_masked(self):
        self.add_rollout("invalid-program", None, valid=False, failure_kind="program")
        self.add_rollout("cutoff", None, valid=False, failure_kind="cutoff")
        report = audit(self.root)
        rows = {row["rollout_id"]: row for row in report["rollouts"]}
        self.assertEqual(rows["invalid-program"]["v1_reward"], -2.0)
        self.assertEqual(rows["invalid-program"]["v2_reward"], -2.0)
        self.assertIsNone(rows["cutoff"]["v1_reward"])
        self.assertIsNone(rows["cutoff"]["v2_reward"])
        self.assertIn("cutoff", rows["cutoff"]["v1_masked_reasons"])

    def test_missing_native_provenance_is_explicitly_not_training_ready(self):
        self.add_rollout("one", "finish")
        report = audit(self.root)
        self.assertFalse(report["native_provenance"]["training_ready"])
        self.assertIn("missing_native_token_ids",
                      report["native_provenance"]["not_training_ready_reasons"])
        self.assertIn("missing_native_logprobs",
                      report["native_provenance"]["not_training_ready_reasons"])
        self.assertIn("missing_native_group_id",
                      report["native_provenance"]["not_training_ready_reasons"])

    def test_nested_native_trace_and_installed_receipt_layout_are_joined(self):
        rollout_id = "native-rollout"
        self.add_rollout(rollout_id, "finish")
        hashes = self.hashes("rl-states/canvas.png")
        request_id = request_key(hashes)
        self.add_receipt(request_id, hashes, label(), label(winner="B",
                            acceptable_A="no", acceptable_B="yes"), installed=True)
        trace_dir = self.root / "async-output" / "traces"
        trace_dir.mkdir(parents=True)
        trace = {
            "group": {"id": "native-group-00"},
            "traces": [{"id": rollout_id, "nodes": [{
                "token_ids": [11, 12], "mask": [True, True],
                "logprobs": [-0.1, -0.2],
            }]}],
        }
        (trace_dir / "stream.jsonl").write_text(json.dumps(trace) + "\n")

        report = audit(self.root)
        row = report["rollouts"][0]
        self.assertEqual(row["receipt_source"], f"judge-queue/{request_id}/receipt.json")
        self.assertEqual(row["native_provenance"]["token_ids"], "present")
        self.assertEqual(row["native_provenance"]["mask"], "present")
        self.assertEqual(row["native_provenance"]["logprobs"], "present")
        self.assertEqual(row["native_provenance"]["group_id_value"], "native-group-00")
        self.assertEqual(report["native_provenance"]["field_counts"], {
            "group_id": 1, "logprobs": 1, "mask": 1, "token_ids": 1,
        })
        self.assertTrue(report["native_provenance"]["replay_fields_present"]["mask"])
        self.assertFalse(report["native_provenance"]["training_ready"])

    def test_rollout_and_state_identity_mismatch_masks_visual_score(self):
        self.add_rollout("sample-id", "finish")
        folder = self.root / "rl-rollouts" / "sample-id"
        result = json.loads((folder / "result.json").read_text())
        result["id"] = "result-id"
        result["state_id"] = "state-other"
        (folder / "result.json").write_text(json.dumps(result))
        hashes = self.hashes("rl-states/canvas.png")
        self.add_receipt(request_key(hashes), hashes, label(), label(winner="B",
                            acceptable_A="no", acceptable_B="yes"))
        row = audit(self.root)["rollouts"][0]
        self.assertIsNone(row["v1_reward"])
        self.assertIn("rollout_id_mismatch", row["v1_masked_reasons"])
        self.assertIn("state_id_mismatch", row["v2_masked_reasons"])


if __name__ == "__main__":
    unittest.main()
