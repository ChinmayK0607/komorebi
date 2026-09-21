"""Offline tests for the native wire-artifact observer and queue race fix."""

import ast
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest

import training_handoff as handoff


def load_request_writer():
    """Load the queue writer without importing node-only verifier packages."""
    source = Path(__file__).with_name("taskset_impl.py").read_text()
    tree = ast.parse(source)
    wanted = {"create_or_verify_request", "sha"}
    stdlib_modules = {"hashlib", "json", "os", "time", "uuid"}
    nodes = [node for node in tree.body
             if isinstance(node, ast.Import) and all(alias.name in stdlib_modules for alias in node.names)]
    nodes.extend(node for node in tree.body
                 if isinstance(node, ast.ImportFrom) and node.module == "pathlib")
    nodes.extend(node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in wanted)
    namespace = {"__name__": "request_writer_test"}
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(Path(__file__).with_name("taskset_impl.py")), "exec"), namespace)
    return namespace["create_or_verify_request"]


def micro_batch(trace_ids=None):
    return {
        "input_ids": [101, 102, 103, 201, 202],
        # Packed/padded positions are represented by false entries.  A zero
        # advantage member is valid; Prime groups and normalisation decide its
        # eligibility upstream.
        "loss_mask": [False, True, True, False, True],
        "inference_logprobs": [-1.2, -0.8, -0.6, -1.1, -0.7],
        "temperatures": [1.0] * 5,
        "advantages": [0.0, 0.0, 1.0, 0.0, -1.0],
        "sequence_lengths": [3, 2],
        "env_names": ["painter-revision"] * 5,
        "trace_ids": trace_ids if trace_ids is not None else ["trace-0", "trace-1"],
        "branch_indices": [0, 1],
        # These are optional in the decoded native microbatch.
        "mm_kwargs": None,
        "mm_token_type_ids": None,
        "ref_logprobs": None,
    }


class TrainingHandoffTest(unittest.TestCase):
    def test_native_observation_retains_wire_bytes_and_shape_facts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rank = root / "async-output" / "batches" / "step_0" / "rank_0.bin"
            rank.parent.mkdir(parents=True)
            raw = b"native-msgpack-rank-0"
            rank.write_bytes(raw)
            destination = handoff.record_microbatches(
                root, [micro_batch()], training_step=0, native_rank_files=[rank])
            observation = json.loads(destination.read_text())
            self.assertEqual(observation["status"], "observed")
            self.assertEqual(observation["micro_batches"][0]["sequence_lengths"], [3, 2])
            self.assertEqual(observation["micro_batches"][0]["stream_lengths"]["input_ids"], 5)
            retained = Path(observation["native_rank_files"][0]["retained"])
            self.assertEqual(retained.read_bytes(), raw)
            self.assertEqual(observation["native_rank_files"][0]["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertFalse(observation["proves_cuda_backward"])
            rank.write_bytes(b"changed-after-capture")
            with self.assertRaises(ValueError):
                handoff.record_microbatches(
                    root, [micro_batch()], training_step=0, native_rank_files=[rank])
            self.assertEqual(retained.read_bytes(), raw)

    def test_missing_optional_metadata_is_observed_not_a_training_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = handoff.record_microbatches(
                root, [{"input_ids": [1], "loss_mask": [True], "advantages": [0.0],
                        "sequence_lengths": [1]}], training_step=0)
            observation = json.loads(destination.read_text())
            self.assertEqual(observation["score_observations"], [])
            self.assertEqual(observation["metadata_authority"]["policy_revision"],
                             "Prime TrainSink; not reconstructed here")

    def test_runtime_state_is_derived_from_process_and_native_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "EXPERIMENT.json").write_text(json.dumps({"run_name": "fixture"}))
            state = handoff.runtime_state(root)
            self.assertEqual(state["status"], "not_ready")
            self.assertIn("no native filesystem batch observed", state["reasons"])
            self.assertFalse(state["first_native_batch"])
            batch = root / "async-output" / "fixture" / "batches" / "step_0"
            batch.mkdir(parents=True)
            (batch / "rank_0.bin").write_bytes(b"native")
            state = handoff.runtime_state(root)
            self.assertEqual(state["status"], "first_native_batch_observed")
            self.assertTrue(state["producer_ready"])
            self.assertTrue(state["trainer_started"])
            self.assertFalse(state["proves_cuda_backward"])
            audit = root / "async-output" / "fixture" / "artifacts" / "optimizer-update-audit.jsonl"
            audit.parent.mkdir(parents=True)
            audit.write_text(json.dumps({"completed_updates": 1,
                                         "method": "exact pre/post native optimizer step CUDA tensor comparison"}) + "\n")
            state = handoff.runtime_state(root)
            self.assertEqual(state["status"], "optimizer_evidence_observed")
            self.assertEqual(state["optimizer"]["updates_observed"], 1)

    def test_concurrent_duplicate_requests_preserve_first_age_and_existing_state(self):
        create_or_verify_request = load_request_writer()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = root / "judge-queue" / ("a" * 64)
            queue.mkdir(parents=True)
            first = root / "rollout-a"
            second = root / "rollout-b"
            first.mkdir()
            second.mkdir()
            for folder in (first, second):
                (folder / "reference.png").write_bytes(b"reference")
                (folder / "anchor.png").write_bytes(b"anchor")
                (folder / "candidate.png").write_bytes(b"same-candidate")

            def hashes(folder):
                return {role: hashlib.sha256((folder / f"{role}.png").read_bytes()).hexdigest()
                        for role in ("reference", "anchor", "candidate")}

            results = []
            barrier = threading.Barrier(2)

            def writer(folder, created_at):
                images = {role: str((folder / f"{role}.png").relative_to(root))
                          for role in ("reference", "anchor", "candidate")}
                value = dict(id=queue.name, images=images, image_sha256=hashes(folder),
                             created_at=created_at, protocol="luna-ab-ba-v1")
                barrier.wait()
                results.append(create_or_verify_request(queue / "request.json", value))

            threads = [threading.Thread(target=writer, args=(first, 1.0)),
                       threading.Thread(target=writer, args=(second, 2.0))]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(len(results), 2)
            request = json.loads((queue / "request.json").read_text())
            self.assertIn(request["created_at"], (1.0, 2.0))
            self.assertEqual(results[0]["created_at"], results[1]["created_at"])
            candidate = root / request["images"]["candidate"]
            self.assertEqual(hashlib.sha256(candidate.read_bytes()).hexdigest(),
                             request["image_sha256"]["candidate"])
            (queue / "assigned.json").write_text(json.dumps({"batch": "assigned"}))
            (queue / "receipt.json").write_text(json.dumps({"id": queue.name}))
            create_or_verify_request(queue / "request.json", {
                "id": queue.name,
                "images": {role: str((second / f"{role}.png").relative_to(root))
                           for role in ("reference", "anchor", "candidate")},
                "image_sha256": hashes(second), "created_at": 99.0,
                "protocol": "luna-ab-ba-v1"})
            self.assertEqual(json.loads((queue / "request.json").read_text())["created_at"],
                             request["created_at"])
            self.assertTrue((queue / "assigned.json").is_file() and
                            (queue / "receipt.json").is_file())


if __name__ == "__main__":
    unittest.main()
