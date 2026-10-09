"""Controller-side invariants for the structured-flow SFT warmup."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


RUN = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("structured_flow_warmup", RUN / "warmup.py")
assert SPEC and SPEC.loader
warmup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(warmup)
FLOW_SPEC = importlib.util.spec_from_file_location("structured_flow_export", RUN / "flow_demonstrations.py")
assert FLOW_SPEC and FLOW_SPEC.loader
flow = importlib.util.module_from_spec(FLOW_SPEC)
FLOW_SPEC.loader.exec_module(flow)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


class WarmupTest(unittest.TestCase):
    def _pool(self, root: Path) -> tuple[Path, Path]:
        train_rows, dev_rows = [], []
        for split, count, destination in (("train", 24, train_rows), ("dev", 8, dev_rows)):
            for index in range(count):
                ident = f"{split}-{index:02d}"
                prior = root / f"{ident}-prior.png"
                target = root / f"{ident}-target.png"
                prior.write_bytes(f"prior-{ident}".encode())
                target.write_bytes(f"target-{ident}".encode())
                destination.append({
                    "id": ident,
                    "source_group": f"family-{split}-{index // 2:02d}",
                    "split": split,
                    "flow_state": warmup.FLOW_STATES[index % len(warmup.FLOW_STATES)],
                    "brief": f"Paint the approved {ident} study.",
                    "flow_guidance": "Inspect the prior canvas; revise only when a concrete improvement is needed, else finish.",
                    "prior_canvas": {"path": prior.name, "sha256": _sha(prior)},
                    "target_status": "approved",
                    "target_png": target.name,
                    "target_sha256": _sha(target),
                    "target_receipt": {"verified": True, "png_sha256": _sha(target)},
                    "messages": [
                        {"role": "system", "content": [{"type": "text", "text": "Paint with the pinned API."}]},
                        {"role": "user", "content": [
                            {"type": "text", "text": f"PAINTING BRIEF\n{ident}"},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                        ]},
                        {"role": "assistant", "content": [{"type": "text", "text": "```javascript\nfunction setup(){}\n```"}]},
                    ],
                })
        train = root / "train.jsonl"
        dev = root / "dev.jsonl"
        _write_rows(train, train_rows)
        _write_rows(dev, dev_rows)
        return train, dev

    def test_config_is_four_step_rank16_single_gpu(self) -> None:
        config = warmup.build_config(Path("/node/model/" + warmup.MODEL_REVISION), Path("/node/data"))
        self.assertEqual(config["max_steps"], 4)
        self.assertEqual(config["data"]["batch_size"], 3)
        self.assertEqual(config["data"]["micro_batch_size"], 1)
        self.assertFalse(config["data"]["shuffle"])
        self.assertEqual(config["val"]["data"]["splits"], ["validation"])
        self.assertEqual(config["model"]["lora"]["rank"], 16)
        self.assertEqual(config["optim"]["lr"], 1e-6)
        self.assertEqual(config["deployment"], {"gpus_per_node": 1, "num_train_gpus": 1})

    def test_warmup_gpu_defaults_to_one_and_bounds_inheritance(self) -> None:
        self.assertEqual(warmup.resolve_warmup_gpu({}), "1")
        self.assertEqual(warmup.resolve_warmup_gpu({"PAINTER_WARMUP_GPU": "0"}), "0")
        self.assertEqual(warmup.resolve_warmup_gpu({"PAINTER_WARMUP_GPU": "inherit", "CUDA_VISIBLE_DEVICES": "1"}), "1")
        with self.assertRaisesRegex(ValueError, "ordinal 0 or 1"):
            warmup.resolve_warmup_gpu({"PAINTER_WARMUP_GPU": "2"})
        with self.assertRaisesRegex(ValueError, "ordinal 0 or 1"):
            warmup.resolve_warmup_gpu({"PAINTER_WARMUP_GPU": "inherit", "CUDA_VISIBLE_DEVICES": "0,1"})

    def test_approved_flow_pool_is_exact_24_8_and_family_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            train, dev = self._pool(Path(temporary))
            receipt = warmup.validate_inputs(train, dev)
            self.assertEqual(receipt["counts"], {"train": 24, "dev": 8})
            self.assertTrue(set(receipt["source_groups"]["train"]).isdisjoint(receipt["source_groups"]["dev"]))
            self.assertEqual(len(receipt["rows"]["train"]), 24)
            self.assertEqual(len(receipt["rows"]["dev"]), 8)
            scheduled = warmup.build_schedule(json.loads(line) for line in train.read_text().splitlines())
            self.assertEqual(len(scheduled), 12)
            self.assertEqual(
                [row["warmup_flow_state"] for row in scheduled],
                [state for pattern in warmup.WARMUP_STATE_PATTERN for state in pattern],
            )
            self.assertTrue(all(row["task_family"] == "painting" for row in scheduled))
            self.assertTrue(all(not warmup.SCHEDULE_RESERVED_FIELDS.intersection(row) for row in scheduled))

    def test_legacy_schedule_passes_actual_pinned_validator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, _ = self._pool(root)
            rows = warmup.build_schedule(json.loads(line) for line in train.read_text().splitlines())
            data = root / "data"
            data.mkdir()
            raw = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows).encode()
            (data / "train.jsonl").write_bytes(raw)
            receipt = warmup.build_schedule_receipt(rows, hashlib.sha256(raw).hexdigest())
            receipt_path = root / "schedule.json"
            receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
            pinned = RUN / "sft_source/mixed_sft/scheduled_data.py"
            result = warmup.validate_warmup_schedule(data, receipt_path, pinned)
            self.assertEqual(result["native_schedule"], None)
            self.assertEqual(result["rows"], 12)

    def test_processor_audit_covers_runtime_capacity_mode(self) -> None:
        self.assertEqual(warmup._processor_mask_modes(), (("baseline", None), ("capacity_prefill", "1")))
        self.assertIn("CAPACITY_STATE_COMPLETE", (RUN / "sft_source/mixed_sft/train.py").read_text(encoding="utf-8"))

    def test_baseline_mask_mismatch_is_diagnostic_but_capacity_mismatch_fails(self) -> None:
        baseline = warmup._mask_observation("baseline", 5, [False, True, True, True, False], [8, 9, 10], [9, 10], 8)
        self.assertEqual(baseline["status"], "diagnostic")
        self.assertFalse(baseline["exact_target_match"])
        passing = warmup._mask_observation("capacity_prefill", 5, [False, False, True, True, False], [9, 10], [9, 10], 8)
        self.assertEqual(passing["status"], "required_pass")
        with self.assertRaisesRegex(ValueError, "capacity_prefill assistant target"):
            warmup._mask_observation("capacity_prefill", 5, [False, True, True, True, False], [8, 9, 10], [9, 10], 8)

    def test_unapproved_target_and_prior_without_hash_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, dev = self._pool(root)
            rows = [json.loads(line) for line in train.read_text().splitlines()]
            rows[0]["target_status"] = "render_valid_pending_visual_review"
            _write_rows(train, rows)
            with self.assertRaisesRegex(ValueError, "not explicitly approved"):
                warmup.validate_inputs(train, dev)
            rows[0]["target_status"] = "approved"
            rows[1]["prior_canvas"].pop("sha256")
            _write_rows(train, rows)
            with self.assertRaisesRegex(ValueError, "prior canvas path/hash"):
                warmup.validate_inputs(train, dev)

    def test_construction_can_start_without_a_prior_canvas(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, dev = self._pool(root)
            rows = [json.loads(line) for line in train.read_text().splitlines()]
            rows[0].pop("prior_canvas")
            rows[0]["messages"][1]["content"] = [{"type": "text", "text": "blank canvas"}]
            _write_rows(train, rows)
            receipt = warmup.validate_inputs(train, dev)
            self.assertFalse(receipt["rows"]["train"][0]["prior_canvas"]["available"])

    def test_pending_export_rows_need_matching_review_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, dev = self._pool(root)
            rows = [json.loads(line) for line in train.read_text().splitlines()]
            by_group = {}
            for row in rows:
                row["target_status"] = "render_valid_pending_visual_review"
                row.pop("target_receipt", None)
                by_group.setdefault(row["source_group"], {})[row["flow_state"]] = row
            candidates = []
            for group, grouped in by_group.items():
                construction = grouped.get("construction") or next(iter(grouped.values()))
                final = grouped.get("revision") or grouped.get("finish") or construction
                candidates.append({
                    "id": group, "render_valid": True, "decision": "approved", "improved_reviewed": True,
                    "initial_png_sha256": construction["target_sha256"],
                    "final_png_sha256": final["target_sha256"],
                })
            approval = root / "approval.json"
            approval.write_text(json.dumps({"status": "reviewed", "candidates": candidates}) + "\n")
            _write_rows(train, rows)
            with self.assertRaisesRegex(ValueError, "row render hash disagrees"):
                warmup.validate_inputs(train, dev, approval)
            # Make the stage-specific row hashes agree with the approval map.
            for row in rows:
                candidate = next(item for item in candidates if item["id"] == row["source_group"])
                row["target_sha256"] = candidate["initial_png_sha256"] if row["flow_state"] == "construction" else candidate["final_png_sha256"]
            # The synthetic files still have their original bytes, so the hash
            # change is deliberately rejected as stale evidence.
            _write_rows(train, rows)
            with self.assertRaisesRegex(ValueError, "target PNG hash mismatch"):
                warmup.validate_inputs(train, dev, approval)

    def test_actual_rich_flow_export_is_admitted_only_with_review_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            flow.build(root)
            render_records = []
            for candidate in flow.CANDIDATES:
                for version in ("block_in", "improved_final"):
                    source = root / "programs" / candidate.id / f"{version}.js"
                    png = root / "renders" / candidate.id / f"{version}.png"
                    png.parent.mkdir(parents=True, exist_ok=True)
                    png.write_bytes(f"{candidate.id}:{version}".encode())
                    render_records.append({
                        "candidate_id": candidate.id, "version": version, "valid": True,
                        "source_sha256": _sha(source), "png_sha256": _sha(png),
                    })
            manifest = json.loads((root / "manifest.json").read_text())
            (root / "render-manifest.json").write_text(json.dumps({
                "schema": "painter.structured-flow-render.v1", "run_id": flow.RUN_ID,
                "source_manifest_sha256": manifest["source_manifest_sha256"],
                "renderer": {"renderer_sha256": "a" * 64, "assets_sha256": "b" * 64},
                "program_count": 32, "valid_program_count": 32, "complete_hash_verified": True,
                "results": render_records,
            }))
            approval = {"status": "reviewed", "candidates": []}
            for candidate in flow.CANDIDATES:
                folder = root / "renders" / candidate.id
                approval["candidates"].append({
                    "id": candidate.id, "render_valid": True, "decision": "approved", "improved_reviewed": True,
                    "initial_png_sha256": _sha(folder / "block_in.png"),
                    "final_png_sha256": _sha(folder / "improved_final.png"),
                })
            approval_path = root / "approval.json"
            approval_path.write_text(json.dumps(approval))
            exported = flow.export_sft(root, include_finish=True, approval_manifest=approval_path)
            self.assertEqual(exported["rows"], 40)
            receipt = warmup.validate_inputs(root / "train.jsonl", root / "dev.jsonl", approval_path)
            self.assertEqual(receipt["counts"], {"train": 32, "dev": 8})
            self.assertEqual(len(warmup.build_schedule(json.loads(line) for line in (root / "train.jsonl").read_text().splitlines())), 12)

    def test_parent_adapter_requires_exact_bytes_and_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "adapter"
            root.mkdir()
            weights = root / "adapter_model.safetensors"
            weights.write_bytes(b"reviewed-test-adapter")
            (root / "adapter_config.json").write_text(json.dumps({
                "peft_type": "LORA", "r": 16, "lora_alpha": 32,
                "target_modules": list(warmup.TARGET_MODULES),
            }))
            value = warmup.verify_parent_adapter(root, expected_sha256=_sha(weights))
            self.assertEqual(value["rank"], 16)
            with self.assertRaisesRegex(ValueError, "Flash12 SHA"):
                warmup.verify_parent_adapter(root)

    def test_runtime_command_is_one_gpu_torchrun_and_no_fresh_base(self) -> None:
        runtime = Path("/node/runtime-source")
        command = warmup.build_train_command(runtime, Path("/node/setup/resolved.toml"), "/node/python")
        self.assertEqual(command[:6], ["/node/python", "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=1"])
        self.assertEqual(command[-2:], ["@", "/node/setup/resolved.toml"])
        self.assertNotIn("FRESH_BASE=1", command)

    def test_packaged_reviewed_helpers_are_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = warmup.materialize_source_layout(RUN / "sft_source", Path(temporary))
            self.assertTrue((runtime / "integrations/speedpainting/mixed_sft/train.py").is_file())
            self.assertTrue((runtime / "integrations/prime/watercolour_sft/export_adapter.py").is_file())
            self.assertTrue((runtime / "scripts/upload_checkpoints.py").is_file())

    def test_checkpoint_receipt_requires_public_finished_upload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "step_4"
            checkpoint.mkdir()
            weight = checkpoint / "adapter_model.safetensors"
            weight.write_bytes(b"adapter")
            digest = _sha(weight)
            (checkpoint / "validation.json").write_text(json.dumps({"finite": True, "sha256": digest}))
            (checkpoint / "hf-upload.json").write_text(json.dumps({"verified": True, "private": False, "repo_id": "public/x"}))
            (checkpoint / ".finished").touch()
            value = warmup.validate_checkpoint(checkpoint)
            self.assertEqual(value["step"], 4)
            (checkpoint / "hf-upload.json").write_text(json.dumps({"verified": True, "private": True}))
            with self.assertRaisesRegex(ValueError, "verified public"):
                warmup.validate_checkpoint(checkpoint)

    def test_optimizer_audit_requires_all_four_changed_finite_updates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "optimizer-update-audit.jsonl"
            path.write_text("".join(json.dumps({
                "completed_updates": step, "all_finite": True,
                "changed_elements": 3, "changed_tensors": 2,
            }) + "\n" for step in range(1, 5)))
            value = warmup.validate_optimizer_audit(path)
            self.assertEqual(value["updates"], 4)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[3]["changed_elements"] = 0
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "no changed adapter tensors"):
                warmup.validate_optimizer_audit(path)


if __name__ == "__main__":
    unittest.main()
