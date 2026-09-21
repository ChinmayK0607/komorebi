"""Offline checks for the bounded native replay preflight."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("native_replay.py")
SPEC = importlib.util.spec_from_file_location("native_replay", MODULE_PATH)
assert SPEC and SPEC.loader
native_replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native_replay)
CONFIG_PATH = Path(__file__).with_name("prepare_replay_config.py")
CONFIG_SPEC = importlib.util.spec_from_file_location("prepare_replay_config", CONFIG_PATH)
assert CONFIG_SPEC and CONFIG_SPEC.loader
prepare_replay_config = importlib.util.module_from_spec(CONFIG_SPEC)
CONFIG_SPEC.loader.exec_module(prepare_replay_config)


def _record(trace_id: str = "trace", *, valid: bool = True, failure_kind: str | None = None) -> dict:
    return {
        "id": f"episode-{trace_id}",
        "group": {"id": "group"},
        "run": {"work": {"policy": {"start": 0}}},
        "traces": [
            {
                "id": trace_id,
                "ok": True,
                "info": {"painting_result": {"id": f"paint-{trace_id}", "valid": valid, "failure_kind": failure_kind}},
                "nodes": [
                    {
                        "sampled": True,
                        "token_ids": [1, 2, 3],
                        "mask": [False, True, True],
                        "logprobs": [-0.1, -0.2],
                    }
                ],
            }
        ],
    }


class NativeReplayTests(unittest.TestCase):
    def test_score_document_mapping_and_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.json"
            path.write_text(json.dumps({"scores": {"trace-a": 1, "request-b": {"score": 0.5}}}))
            self.assertEqual(native_replay.load_score_document(path), {"trace-a": 1.0, "request-b": 0.5})
            path.write_text(json.dumps([{"trace_id": "trace-a", "reward": -1}]))
            self.assertEqual(native_replay.load_score_document(path), {"trace-a": -1.0})

    def test_validate_rejects_unaligned_logprobs(self) -> None:
        record = _record()
        record["traces"][0]["nodes"][0]["logprobs"] = [-0.1]
        with self.assertRaises(native_replay.ReplayError):
            native_replay.validate_native_record(record)

    def test_group_must_be_complete_and_policy_matched(self) -> None:
        records = [_record(f"trace-{index}") for index in range(7)]
        with self.assertRaises(native_replay.ReplayError):
            native_replay.select_complete_group(records, "group")
        records.append(_record("trace-7"))
        selected = native_replay.select_complete_group(records, "group")
        self.assertEqual(len(selected), 8)
        selected[0]["run"]["work"]["policy"]["start"] = 1
        with self.assertRaises(native_replay.ReplayError):
            native_replay.select_complete_group(selected, "group")

    def test_exclusions_are_explicit_members_of_complete_group(self) -> None:
        records = [_record(f"trace-{index}") for index in range(8)]
        kept = native_replay.exclude_trace_ids(records, ["trace-6", "trace-7"])
        self.assertEqual([item["traces"][0]["id"] for item in kept], [f"trace-{i}" for i in range(6)])
        with self.assertRaises(native_replay.ReplayError):
            native_replay.exclude_trace_ids(records, ["not-a-member"])

    def test_invalid_program_penalty_is_explicit_opt_in(self) -> None:
        record = _record("bad", valid=False, failure_kind="program")
        with self.assertRaises(native_replay.ReplayError):
            native_replay._score_for(record, {}, allow_invalid_program_penalty=False)
        score, key = native_replay._score_for(record, {}, allow_invalid_program_penalty=True)
        self.assertEqual((score, key), (-2.0, "deterministic_invalid_program_penalty"))

    def test_aliases_include_painting_id(self) -> None:
        record = _record("trace-a")
        score, key = native_replay._score_for(record, {"paint-trace-a": 1.0}, allow_invalid_program_penalty=False)
        self.assertEqual((score, key), (1.0, "paint-trace-a"))

    def test_config_derivation_keeps_root_vlm_settings(self) -> None:
        baseline = {
            "model": {"name": "/old", "vlm": {"vision_encoder_attr": "model.visual"}},
            "trainer": {"model": {"lora": {"rank": 16}}, "max_steps": 99},
        }
        config = prepare_replay_config.build_config(
            baseline, model_path="/new", output_dir="/out", seq_len=16384
        )
        self.assertEqual(config["model"]["name"], "/new")
        self.assertEqual(config["model"]["vlm"]["vision_encoder_attr"], "model.visual")
        self.assertEqual(config["model"]["lora"]["rank"], 16)
        self.assertEqual(config["max_steps"], 1)


if __name__ == "__main__":
    unittest.main()
