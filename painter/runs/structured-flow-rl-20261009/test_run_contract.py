"""Offline contract checks for the structured-flow launcher and handoff."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest import mock


RUN = Path(__file__).resolve().parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class StructuredFlowRunTests(unittest.TestCase):
    def test_lazy_manifest_is_exact_pilot_and_has_explicit_frozen_identity(self):
        curriculum = load("structured_flow_curriculum_contract", RUN / "prepare_curriculum.py")
        bank = load("structured_flow_anchor_bank_contract", RUN / "anchor_bank.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            curriculum.build_proposal(root / "proposal")
            report = bank.prepare_lazy(
                train_proposal=root / "proposal/train-proposal.json",
                heldout_proposal=root / "proposal/heldout-proposal.json",
                train_output=root / "train.json",
                heldout_output=root / "heldout.json",
            )
            self.assertEqual(report, {"status": "pending_runtime_frozen_anchors", "train_rows": 32, "heldout_cases": 16})
            train = json.loads((root / "train.json").read_text())
            self.assertTrue(all("opponent" not in row and row["lazy_opponent"]["model_alias"] == "sft-step512" for row in train["rows"]))
            self.assertTrue(all(row["max_turns"] == 3 for row in train["rows"]))
            heldout = json.loads((root / "heldout.json").read_text())
            for row in [*train["rows"], *heldout["cases"]]:
                casebrief = Path(row["casebrief"])
                self.assertTrue(casebrief.is_file())
                self.assertEqual(hashlib.sha256(casebrief.read_bytes()).hexdigest(), row["casebrief_sha256"])
                self.assertTrue(casebrief.read_text().startswith("[SYSTEM]\n"))
                self.assertIn("\n\n[USER]\n", casebrief.read_text())
            hook = load("structured_flow_lazy_manifest_hook_contract", RUN / "flow_prompt_hook.py")
            with mock.patch.dict(
                os.environ,
                {"PAINTER_RL_MANIFEST": str(root / "train.json"), "PAINTER_EVAL_MANIFEST": str(root / "heldout.json")},
                clear=False,
            ):
                train_rows, _ = hook._manifest_rows("PAINTER_RL_MANIFEST")
                eval_rows, _ = hook._manifest_rows("PAINTER_EVAL_MANIFEST")
            self.assertEqual(len(train_rows), 32)
            self.assertEqual(len(eval_rows), 16)
            self.assertTrue(all(hook._parse_casebrief(Path(row["casebrief"]))[1] for row in eval_rows.values()))
            from pydantic import BaseModel, ConfigDict

            class FrozenData(BaseModel):
                model_config = ConfigDict(frozen=True)
                task_id: str
                reference: str
                system_prompt: str

            class FrozenTask(BaseModel):
                model_config = ConfigDict(frozen=True)
                data: FrozenData

            for row in [*train["rows"], *heldout["cases"]]:
                reference = Path(str(row["reference"]))
                if not reference.is_absolute():
                    reference = (root / "proposal" / reference).resolve()
                original = FrozenTask(data=FrozenData(
                    task_id=str(row["id"]), reference=str(reference), system_prompt="native"
                ))
                patched = hook._patch_task(original, row)
                self.assertIsNot(patched, original)
                self.assertEqual(original.data.system_prompt, "native")
                system, user = hook._parse_casebrief(Path(row["casebrief"]))
                self.assertTrue(system)
                self.assertEqual(user, reference.read_text().strip())
            heldout["cases"][0].pop("casebrief")
            heldout["cases"][0].pop("casebrief_sha256")
            heldout["cases"][0]["opponent"] = "preserve-this-anchor.png"
            (root / "heldout.json").write_text(json.dumps(heldout))
            repair = bank.ensure_casebriefs(
                train_manifest=root / "train.json",
                heldout_manifest=root / "heldout.json",
                proposal_root=root / "proposal",
            )
            self.assertEqual(repair, {"train_repaired": 0, "heldout_repaired": 1})
            repaired = json.loads((root / "heldout.json").read_text())["cases"][0]
            self.assertEqual(repaired["opponent"], "preserve-this-anchor.png")
            self.assertTrue(Path(repaired["casebrief"]).is_file())

    def test_config_cli_accepts_pending_manifest_only_with_explicit_flag(self):
        config = load("structured_flow_config_contract", RUN / "prepare_structured_config.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "curriculum/train.json"
            manifest.parent.mkdir()
            row = {
                "id": "case-1", "reference_kind": "text", "reference": "brief.txt",
                "reference_sha256": hashlib.sha256(b"brief\n").hexdigest(),
                "lazy_opponent": {"model_alias": "sft-step512"},
            }
            (root / "curriculum/brief.txt").write_text("brief\n")
            manifest.write_text(json.dumps({"rows": [row] * 32}))
            # Duplicate IDs are rejected before a pending config can be built.
            with self.assertRaises(ValueError):
                config.prepare(output=root / "bad.toml", model=Path("/tmp/model"), run_root=root, manifest=manifest, expected_count=32, max_turns=3, allow_pending_anchors=True)

    def test_flow_hook_consumes_casebrief_as_system_input(self):
        hook = load("structured_flow_prompt_hook_contract", RUN / "flow_prompt_hook.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "brief.txt"
            reference.write_text("Paint one orb.\n")
            casebrief = root / "casebrief.txt"
            casebrief.write_text("[SYSTEM]\nPlan by hierarchy.\n\n[USER]\nPaint one orb.\n")
            digest = hashlib.sha256(casebrief.read_bytes()).hexdigest()
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"rows": [{"id": "case-1", "reference": str(reference), "casebrief": str(casebrief), "casebrief_sha256": digest}]}))
            os.environ["PAINTER_STRUCTURED_FLOW_POLICY"] = "1"
            os.environ["PAINTER_RL_MANIFEST"] = str(manifest)
            class Task:
                def __init__(self):
                    self.data = types.SimpleNamespace(task_id="case-1", reference=str(reference), system_prompt="native", prompt=[])
            class Taskset:
                def load(self):
                    return [Task()]
            fake = types.ModuleType("painter.native_painting_rl")
            fake.NativePaintingRLTaskset = Taskset
            old = sys.modules.get("painter.native_painting_rl")
            sys.modules["painter.native_painting_rl"] = fake
            try:
                hook.install()
                task = Taskset().load()[0]
                self.assertIn("Plan by hierarchy.", task.data.system_prompt)
                self.assertNotIn("STRUCTURED-FLOW RUN HINT", "".join(str(part) for part in task.data.prompt))
            finally:
                if old is None:
                    sys.modules.pop("painter.native_painting_rl", None)
                else:
                    sys.modules["painter.native_painting_rl"] = old
                os.environ.pop("PAINTER_STRUCTURED_FLOW_POLICY", None)
                os.environ.pop("PAINTER_RL_MANIFEST", None)

    def test_flow_hook_copies_frozen_pydantic_task_data(self):
        """The real Prime models are frozen; patching must return a new task."""
        from pydantic import BaseModel, ConfigDict

        hook = load("structured_flow_prompt_hook_frozen_contract", RUN / "flow_prompt_hook.py")

        class FrozenData(BaseModel):
            model_config = ConfigDict(frozen=True)
            task_id: str
            reference: str
            system_prompt: str

        class FrozenTask(BaseModel):
            model_config = ConfigDict(frozen=True)
            data: FrozenData

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "brief.txt"
            reference.write_text("Paint one orb.\n")
            casebrief = root / "casebrief.txt"
            casebrief.write_text("[SYSTEM]\nUse hierarchy.\n\n[USER]\nPaint one orb.\n")
            row = {
                "id": "case-1",
                "casebrief": str(casebrief),
                "casebrief_sha256": hashlib.sha256(casebrief.read_bytes()).hexdigest(),
            }
            task = FrozenTask(data=FrozenData(
                task_id="case-1", reference=str(reference), system_prompt="native"
            ))
            patched = hook._patch_task(task, row)
            self.assertIsNot(patched, task)
            self.assertIsNot(patched.data, task.data)
            self.assertEqual(task.data.system_prompt, "native")
            self.assertIn("Use hierarchy.", patched.data.system_prompt)

    def test_supervisor_source_has_arm_bound_and_no_old_checkpoint_cadence(self):
        text = (RUN / "finite_supervisor.py").read_text()
        self.assertIn("base.CHECKPOINT_STEPS = ()", text)
        self.assertIn("CHECKPOINTS = (4, 8, 12, 16, 20)", text)
        self.assertIn("max_updates=MAX_UPDATES", text)
        self.assertIn("MAX_UPDATES = 20", text)
        self.assertNotIn("max-updates 50", text)

    def test_legacy_eight_update_config_request_is_recorded_but_upgraded_to_twenty(self):
        config = load("structured_flow_twenty_config_contract", RUN / "prepare_structured_config.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "brief.txt"
            reference.write_text("Paint one orb.\n")
            manifest = root / "manifest.json"
            rows = [{
                "id": f"case-{index:02d}",
                "reference_kind": "text",
                "reference": str(reference),
                "lazy_opponent": {"model_alias": "sft-step512"},
            } for index in range(32)]
            manifest.write_text(json.dumps({"rows": rows}))
            output = root / "native.toml"
            config.prepare(
                output=output,
                model=root / "model",
                run_root=root,
                manifest=manifest,
                expected_count=32,
                max_steps=8,
                checkpoint_interval=4,
                allow_pending_anchors=True,
            )
            receipt = json.loads((root / "rl-config-receipt.json").read_text())
            self.assertEqual(receipt["requested_updates"], 8)
            self.assertEqual(receipt["updates"], 20)
            self.assertEqual(receipt["effective_updates"], 20)
            self.assertEqual(receipt["checkpoint_steps"], [4, 8, 12, 16, 20])
            self.assertIn("max_steps = 20", output.read_text())

    def test_final_config_refresh_repairs_existing_eight_update_receipt(self):
        config = load("structured_flow_refresh_config_contract", RUN / "prepare_structured_config.py")
        refresh = load("structured_flow_refresh_contract", RUN / "run_from_sft.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "brief.txt"
            reference.write_text("Paint one orb.\n")
            opponent = root / "opponent.png"
            opponent.write_bytes(b"PNG fixture")
            rows = [{
                "id": f"case-{index:02d}",
                "reference_kind": "text",
                "reference": str(reference),
                "opponent": str(opponent),
                "opponent_sha256": hashlib.sha256(opponent.read_bytes()).hexdigest(),
                "anchor_status": "verified_file",
            } for index in range(32)]
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"rows": rows}))
            output = root / "native.toml"
            config.prepare(
                output=output,
                model=root / "model",
                run_root=root,
                manifest=manifest,
                expected_count=32,
                max_steps=20,
                checkpoint_interval=4,
            )
            receipt_path = root / "rl-config-receipt.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["updates"] = 8
            receipt.pop("effective_updates", None)
            receipt["checkpoint_steps"] = [4, 8]
            receipt_path.write_text(json.dumps(receipt))
            output.write_text(output.read_text().replace("max_steps = 20", "max_steps = 8"))
            refresh._refresh_final_config(output, manifest, 32)
            repaired = json.loads(receipt_path.read_text())
            self.assertEqual(repaired["requested_updates"], 8)
            self.assertEqual(repaired["updates"], 20)
            self.assertEqual(repaired["effective_updates"], 20)
            self.assertEqual(repaired["checkpoint_steps"], [4, 8, 12, 16, 20])
            self.assertIn("max_steps = 20", output.read_text())

    def test_controller_rejects_stale_eight_update_remote_receipt(self):
        operator = load("structured_flow_twenty_controller_contract", RUN / "operate_single.py")
        base_summary = {
            "preparation": {"schema": "structured-flow-rl-preparation-v1"},
            "config": {
                "updates": 20,
                "checkpoint_steps": [4, 8, 12, 16, 20],
                "checkpoint_interval": 4,
                "batch_size": 16,
                "group_size": 4,
                "initial_inflight": 32,
                "max_inflight": 32,
                "deployment": {"type": "single_node", "gpus_per_node": 2, "num_train_gpus": 1, "num_infer_gpus": 1},
            },
        }
        operator._validate_config_summary(base_summary)
        stale = json.loads(json.dumps(base_summary))
        stale["config"]["updates"] = 8
        with self.assertRaises(Exception):
            operator._validate_config_summary(stale)

    def test_archive_selector_maps_legacy_step_eight_to_twenty_and_bounds_fallback(self):
        with mock.patch.dict(os.environ, {"PAINTER_RUN_NAME": "structured-flow-rl-20261009"}, clear=False):
            archive = load("structured_flow_twenty_archive_contract", RUN / "safe_archive.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapters = root / "async-output/structured-flow-rl-20261009/artifacts/adapters"

            def add_checkpoint(step: int) -> None:
                checkpoint = adapters / f"step_{step}"
                checkpoint.mkdir(parents=True)
                weight = checkpoint / "adapter_model.safetensors"
                weight.write_bytes(f"step-{step}".encode())
                digest = archive.digest(weight)
                (checkpoint / "hf-upload.json").write_text(json.dumps({
                    "verified": True,
                    "private": False,
                    "files": {"adapter_model.safetensors": {"sha256": digest}},
                }))

            add_checkpoint(12)
            add_checkpoint(20)
            selected, selection = archive.select_preferred_public_checkpoint(root, 8)
            self.assertEqual(selected["step"], 20)
            self.assertEqual(selection["requested_preferred_step"], 8)
            self.assertEqual(selection["preferred_step"], 20)
            self.assertEqual(selection["selected_step"], 20)
            (adapters / "step_20").rename(adapters / "step_20-unavailable")
            selected, selection = archive.select_preferred_public_checkpoint(root, 8)
            self.assertEqual(selected["step"], 12)
            self.assertEqual(selection["primary_quality_status"], "exploratory_censored")
            self.assertEqual(selection["fallback_reason"], "registered_final_checkpoint_not_verified_or_unavailable")

    def test_execute_pipeline_binds_twenty_before_inherited_runtime_values(self):
        execute = (RUN / "execute_pipeline.sh").read_text()
        bind = "export PAINTER_RL_MAX_STEPS=20 PAINTER_MAX_UPDATES=20 PAINTER_CHECKPOINT_INTERVAL=4 PAINTER_EVAL_PREFERRED_STEP=20"
        self.assertGreaterEqual(execute.count(bind), 2)
        self.assertLess(execute.index(bind), execute.index('source "$RUN/runtime_env.sh"'))

    def test_runtime_bindings_normalize_inherited_eight_update_values(self):
        result = subprocess.run(
            ["bash", "-c", 'source "$1/runtime_env.sh"; validate_structured_bindings; printf "%s %s %s" "$PAINTER_RL_MAX_STEPS" "$PAINTER_MAX_UPDATES" "$PAINTER_EVAL_PREFERRED_STEP"', "runtime", str(RUN)],
            env={
                **os.environ,
                "PAINTER_RUN_ROOT": str(RUN),
                "PAINTER_RL_MAX_STEPS": "8",
                "PAINTER_MAX_UPDATES": "8",
                "PAINTER_EVAL_PREFERRED_STEP": "8",
            },
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(result.stdout, "20 20 20")

    def test_renderer_and_anchor_handoff_are_hash_bound(self):
        prepare = (RUN / "prepare_node.sh").read_text()
        anchor = (RUN / "anchor_bank.py").read_text()
        self.assertIn("db5b0094cb223529a8c5fc3bbba7bf8f2d5b5f5a5c0a847062194feaf917d16b", prepare)
        self.assertIn("dc6b9d837ed7be382784a6c84de1b64911506b17a19b499ce054c0c572e75794", prepare)
        self.assertIn("prepare_renderer_runtime.py", prepare)
        self.assertIn('"reference_kind": "text"', anchor)
        self.assertIn("max_concurrent=workers", anchor)
        self.assertIn("_native_eval_canvas", anchor)

    def test_runtime_env_upgrades_raw_renderer_after_node_overlay_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            overlay = root / "runtime/watercolour-cancellable-centroid-v1"
            overlay.mkdir(parents=True)
            selected = overlay / "cancellable_watercolour_renderer.py"
            selected.write_text("# verified test overlay\n")
            raw = root / "vendor/integrations/watercolour/renderer.py"
            raw.parent.mkdir(parents=True)
            raw.write_text("# raw vendor\n")
            result = subprocess.run(
                ["bash", "-c", 'source "$1/runtime_env.sh"; printf "%s" "$PAINTER_RENDERER_PATH"', "_", str(RUN)],
                env={
                    **os.environ,
                    "PAINTER_RUN_ROOT": str(root),
                    "PAINTER_RENDERER_PATH": str(raw),
                    "PAINTER_RUN_NAME": "structured-flow-rl-20261009",
                },
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(result.stdout, str(selected))

    def test_runtime_env_exposes_prepared_prime_bin_to_downstream_children(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prime_bin = root / "prime/.venv/bin"
            prime_bin.mkdir(parents=True)
            result = subprocess.run(
                ["/bin/bash", "-c", 'source "$1/runtime_env.sh"; printf "%s" "$PATH"', "_", str(RUN)],
                env={
                    **os.environ,
                    "PATH": "/controller/path-without-node:/usr/local/bin:/usr/bin:/bin",
                    "PAINTER_RUN_ROOT": str(root),
                    "PAINTER_RUN_NAME": "structured-flow-rl-20261009",
                    "PRIME_ROOT": str(root / "prime"),
                },
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(result.stdout.split(":", 1)[0], str(prime_bin))

    def test_opt_in_local_scratch_moves_model_and_prime_but_not_credentials(self):
        command = (
            'source "$1/runtime_env.sh"; '
            'printf "%s\\n%s\\n%s\\n%s" "$PRIME_ROOT" "$MODEL_DIR" '
            '"$PAINTER_NODE_ROOT" "$HF_TOKEN_PATH"'
        )
        environment = os.environ.copy()
        environment.update({
            "PAINTER_RUN_ROOT": str(RUN),
            "PAINTER_NODE_ROOT": "/protected/node-root",
            "PAINTER_LOCAL_SCRATCH_ROOT": "/local/scratch/structured-flow",
        })
        environment.pop("PRIME_ROOT", None)
        environment.pop("MODEL_DIR", None)
        result = subprocess.run(
            ["bash", "-c", command, "scratch-contract", str(RUN)],
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        )
        prime, model, node_root, token = result.stdout.splitlines()
        self.assertEqual(prime, "/local/scratch/structured-flow/prime")
        self.assertEqual(model, "/local/scratch/structured-flow/model-cache/qwen/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
        self.assertEqual(node_root, "/protected/node-root")
        self.assertEqual(token, "/protected/node-root/private/hf-token")
        bootstrap = (RUN.parent / "text-pairwise-rl-20260928/setup_node.sh").read_text()
        self.assertIn('MODEL_DIR="${MODEL_DIR:-$NODE_ROOT/model-cache/qwen/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"', bootstrap)
        invalid = dict(environment, PAINTER_LOCAL_SCRATCH_ROOT="relative/scratch")
        refused = subprocess.run(
            ["bash", "-c", 'source "$1/runtime_env.sh"; validate_structured_bindings', "scratch-contract", str(RUN)],
            env=invalid,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("must be absolute", refused.stderr)

    def test_controller_forwards_opt_in_scratch_to_detached_node_job(self):
        operator = load("structured_flow_operation_scratch_contract", RUN / "operate_single.py")
        with mock.patch.dict(os.environ, {"PAINTER_LOCAL_SCRATCH_ROOT": "/tmp/structured-flow-scratch"}, clear=False):
            command = operator._detached_job_command(
                source_commit="a" * 40,
                rental_start_epoch=100.0,
                price_usd_per_hour=3.70,
                cap_usd=0.50,
            )
        self.assertIn("PAINTER_LOCAL_SCRATCH_ROOT=/tmp/structured-flow-scratch", command)
        self.assertIn("HF_TOKEN_PATH=/root/painter/private/hf-token", command)
        with mock.patch.dict(os.environ, {"PAINTER_LOCAL_SCRATCH_ROOT": "relative/scratch"}, clear=False):
            with self.assertRaises(ValueError):
                operator._detached_job_command(
                    source_commit="a" * 40,
                    rental_start_epoch=100.0,
                    price_usd_per_hour=3.70,
                    cap_usd=0.50,
                )

    def test_prime_stage_uses_remote_scratch_for_large_bundle_and_checkout(self):
        allocator = load("structured_flow_allocator_scratch_stage_contract", RUN / "allocate_single.py")

        class FakeBase:
            REMOTE_ROOT = Path("/root/painter")

        staged_roots = []

        def fake_stage(**kwargs):
            staged_roots.append(kwargs["operation"].REMOTE_ROOT)

        operation_module = SimpleNamespace(base=FakeBase)
        with tempfile.TemporaryDirectory() as directory:
            prime = Path(directory) / "prime.bundle"
            prime.write_bytes(b"bundle")
            with mock.patch.dict(os.environ, {"PAINTER_LOCAL_SCRATCH_ROOT": "/tmp/structured-flow-large"}, clear=False), \
                 mock.patch.object(allocator.alloc, "stage_prime_bundle", side_effect=fake_stage):
                result = allocator._stage_prime_bundle(
                    api=object(),
                    operation_module=operation_module,
                    window=SimpleNamespace(pod_id="pod"),
                    prime_bundle=prime,
                    prime_commit="a" * 40,
                    prime_sha256="b" * 64,
                )
        self.assertEqual(result, Path("/tmp/structured-flow-large"))
        self.assertEqual(staged_roots, [Path("/tmp/structured-flow-large")])
        self.assertEqual(FakeBase.REMOTE_ROOT, Path("/root/painter"))

    def test_controller_forwards_opt_in_pinned_anchor_restore(self):
        operator = load("structured_flow_operation_restore_contract", RUN / "operate_single.py")
        environment = {
            "PAINTER_RESTORE_ANCHOR_EVIDENCE": "1",
            "PAINTER_ANCHOR_EVIDENCE_SHA256": "a" * 64,
            "PAINTER_ANCHOR_EVIDENCE_URL": "https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/resolve/revision/runs/structured-flow-rl-20261009/evidence.tar.gz",
        }
        with mock.patch.dict(os.environ, environment, clear=False):
            command = operator._detached_job_command(
                source_commit="a" * 40,
                rental_start_epoch=100.0,
                price_usd_per_hour=3.70,
                cap_usd=0.50,
            )
        self.assertIn("PAINTER_RESTORE_ANCHOR_EVIDENCE=1", command)
        self.assertIn("PAINTER_ANCHOR_EVIDENCE_SHA256=" + "a" * 64, command)
        self.assertIn("PAINTER_ANCHOR_EVIDENCE_URL=https://huggingface.co/", command)
        with mock.patch.dict(os.environ, {"PAINTER_RESTORE_ANCHOR_EVIDENCE": "1", "PAINTER_ANCHOR_EVIDENCE_URL": "https://example.com/a"}, clear=False):
            with self.assertRaises(ValueError):
                operator._detached_job_command(
                    source_commit="a" * 40,
                    rental_start_epoch=100.0,
                    price_usd_per_hour=3.70,
                    cap_usd=0.50,
                )

    def test_controller_forwards_opt_in_pinned_supervision_restore(self):
        operator = load("structured_flow_operation_supervision_restore_contract", RUN / "operate_single.py")
        environment = {
            "PAINTER_RESTORE_SUPERVISION_EVIDENCE": "1",
            "PAINTER_SUPERVISION_EVIDENCE_SHA256": "b" * 64,
            "PAINTER_SUPERVISION_EVIDENCE_URL": "https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/resolve/revision/runs/structured-flow-rl-20261009/evidence.tar.gz",
        }
        with mock.patch.dict(os.environ, environment, clear=False):
            command = operator._detached_job_command(
                source_commit="a" * 40,
                rental_start_epoch=100.0,
                price_usd_per_hour=3.70,
                cap_usd=0.50,
            )
        self.assertIn("PAINTER_RESTORE_SUPERVISION_EVIDENCE=1", command)
        self.assertIn("PAINTER_SUPERVISION_EVIDENCE_SHA256=" + "b" * 64, command)
        self.assertIn("PAINTER_SUPERVISION_EVIDENCE_URL=https://huggingface.co/", command)
        with mock.patch.dict(os.environ, {"PAINTER_RESTORE_SUPERVISION_EVIDENCE": "1", "PAINTER_SUPERVISION_EVIDENCE_URL": "https://example.com/a"}, clear=False):
            with self.assertRaises(ValueError):
                operator._detached_job_command(
                    source_commit="a" * 40,
                    rental_start_epoch=100.0,
                    price_usd_per_hour=3.70,
                    cap_usd=0.50,
                )

    def test_controller_forwards_explicit_paired_rl_only_fallback(self):
        operator = load("structured_flow_operation_rl_only_contract", RUN / "operate_single.py")
        with mock.patch.dict(
            os.environ,
            {"PAINTER_RUN_WARMUP": "0", "PAINTER_WARMUP_REQUIRED": "0"},
            clear=False,
        ):
            command = operator._detached_job_command(
                source_commit="a" * 40,
                rental_start_epoch=100.0,
                price_usd_per_hour=3.70,
                cap_usd=0.50,
            )
        self.assertIn("PAINTER_RUN_WARMUP=0", command)
        self.assertIn("PAINTER_WARMUP_REQUIRED=0", command)

        with mock.patch.dict(os.environ, {"PAINTER_RUN_WARMUP": "0"}, clear=False):
            with self.assertRaisesRegex(ValueError, "paired"):
                operator._detached_job_command(
                    source_commit="a" * 40,
                    rental_start_epoch=100.0,
                    price_usd_per_hour=3.70,
                    cap_usd=0.50,
                )

    def test_controller_forwards_opt_in_matched_baseline_reuse_bit(self):
        operator = load("structured_flow_operation_baseline_reuse_contract", RUN / "operate_single.py")
        command = operator._detached_job_command(
            source_commit="a" * 40,
            rental_start_epoch=100.0,
            price_usd_per_hour=3.70,
            cap_usd=0.50,
        )
        self.assertIn("PAINTER_REUSE_MATCHED_BASELINE_EVIDENCE=0", command)
        with mock.patch.dict(
            os.environ,
            {"PAINTER_REUSE_MATCHED_BASELINE_EVIDENCE": "1"},
            clear=False,
        ):
            command = operator._detached_job_command(
                source_commit="a" * 40,
                rental_start_epoch=100.0,
                price_usd_per_hour=3.70,
                cap_usd=0.50,
            )
        self.assertIn("PAINTER_REUSE_MATCHED_BASELINE_EVIDENCE=1", command)
        self.assertNotIn("PAINTER_MATCHED_BASELINE_EVIDENCE_URL", command)
        self.assertNotIn("PAINTER_MATCHED_BASELINE_EVIDENCE_SHA256", command)
        with mock.patch.dict(
            os.environ,
            {"PAINTER_REUSE_MATCHED_BASELINE_EVIDENCE": "maybe"},
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
                operator._detached_job_command(
                    source_commit="a" * 40,
                    rental_start_epoch=100.0,
                    price_usd_per_hour=3.70,
                    cap_usd=0.50,
                )

    def test_remaining_restart_forwards_short_completion_reserve(self):
        operator = load("structured_flow_operation_reserve_contract", RUN / "operate_single.py")
        command = operator._detached_job_command(
            source_commit="a" * 40,
            rental_start_epoch=100.0,
            price_usd_per_hour=3.70,
            cap_usd=0.50,
        )
        self.assertIn("PAINTER_RESERVE_SECONDS=2700", command)
        runtime = (RUN / "runtime_env.sh").read_text(encoding="utf-8")
        self.assertIn('PAINTER_RESERVE_SECONDS="${PAINTER_RESERVE_SECONDS:-3600}"', runtime)

    def test_controller_forwards_opt_in_prime_seed_root(self):
        operator = load("structured_flow_operation_prime_seed_contract", RUN / "operate_single.py")
        with mock.patch.dict(os.environ, {
            "PAINTER_PRIME_SEED_ENABLED": "1",
            "PAINTER_PRIME_SEED_ROOT": "/root/painter/prime",
        }, clear=False):
            command = operator._detached_job_command(
                source_commit="a" * 40,
                rental_start_epoch=100.0,
                price_usd_per_hour=3.70,
                cap_usd=0.50,
            )
        self.assertIn("PAINTER_PRIME_SEED_ENABLED=1", command)
        self.assertIn("PAINTER_PRIME_SEED_ROOT=/root/painter/prime", command)
        with mock.patch.dict(os.environ, {
            "PAINTER_PRIME_SEED_ENABLED": "1",
            "PAINTER_PRIME_SEED_ROOT": "relative/prime",
        }, clear=False):
            with self.assertRaises(ValueError):
                operator._detached_job_command(
                    source_commit="a" * 40,
                    rental_start_epoch=100.0,
                    price_usd_per_hour=3.70,
                    cap_usd=0.50,
                )

    def test_scratch_prime_seed_handoff_matches_runtime_and_reuses_staged_checkout(self):
        operator = load("structured_flow_operation_scratch_prime_handoff_contract", RUN / "operate_single.py")
        environment = os.environ.copy()
        environment.pop("PAINTER_PRIME_SEED_ROOT", None)
        environment.update({
            "PAINTER_LOCAL_SCRATCH_ROOT": "/tmp/structured-flow-large",
            "PAINTER_PRIME_SEED_ENABLED": "1",
        })
        with mock.patch.dict(os.environ, environment, clear=True):
            command = operator._detached_job_command(
                source_commit="a" * 40,
                rental_start_epoch=100.0,
                price_usd_per_hour=3.70,
                cap_usd=0.50,
            )
        self.assertIn("PAINTER_PRIME_SEED_ROOT=/tmp/structured-flow-large/prime", command)

        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory) / "scratch"
            staged_prime = scratch / "prime"
            (staged_prime / ".git").mkdir(parents=True)
            result = subprocess.run(
                ["bash", "-c", 'source "$1/runtime_env.sh"; printf "%s\\n%s" "$PRIME_ROOT" "$PAINTER_PRIME_SEED_ROOT"', "runtime", str(RUN)],
                env={
                    **os.environ,
                    "PAINTER_RUN_ROOT": str(RUN),
                    "PAINTER_RUN_NAME": "structured-flow-rl-20261009",
                    "PAINTER_LOCAL_SCRATCH_ROOT": str(scratch),
                    "PAINTER_PRIME_SEED_ROOT": str(staged_prime),
                    "PAINTER_PRIME_SEED_ENABLED": "1",
                },
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(result.stdout.splitlines(), [str(staged_prime), str(staged_prime)])
            reuse = subprocess.run(
                ["bash", "-c", 'if [[ ! -d "$1/.git" && "${PAINTER_PRIME_SEED_ENABLED:-0}" == 1 ]]; then printf seed; else printf reuse; fi', "reuse", str(staged_prime)],
                env={"PAINTER_PRIME_SEED_ENABLED": "1"},
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(reuse.stdout, "reuse")

        setup = (RUN.parent / "text-pairwise-rl-20260928/setup_node.sh").read_text()
        self.assertIn('if [[ ! -d "$PRIME_ROOT/.git" && "${PAINTER_PRIME_SEED_ENABLED:-0}" == 1 ]]; then', setup)

    def test_node_builds_missing_proposals_and_evaluator_preserves_flow_hook(self):
        prepare = (RUN / "prepare_node.sh").read_text()
        evaluator = (RUN / "evaluate.sh").read_text()
        self.assertIn('prepare_curriculum.py" build --output "$RUN/curriculum"', prepare)
        self.assertIn('CURRICULUM_BUILD_TMP="$RUN/.curriculum-build.json.$$"', prepare)
        self.assertNotIn('>"$RUN/curriculum/build.json"', prepare)
        self.assertIn('FLOW_BUILD_TMP="$RUN/.flow-supervision-build.json.$$"', prepare)
        self.assertNotIn('>"$SUPERVISION_ROOT/build.json"', prepare)
        # RL-only preparation must not restore, render, judge, or export the
        # optional teacher cache even when its legacy supervision flags remain
        # enabled.  The required SFT path still enters this branch by default.
        supervision_gate = 'if [[ "${PAINTER_RUN_WARMUP:-1}" == 1 && "$PAINTER_CODE_SUPERVISION_ENABLED" == 1'
        self.assertIn(supervision_gate, prepare)
        self.assertLess(prepare.index(supervision_gate), prepare.index('FLOW_BUILD_TMP='))
        self.assertIn('restore_anchor_evidence.py', prepare)
        self.assertIn('PAINTER_RESTORE_SUPERVISION_EVIDENCE', prepare)
        self.assertIn('--mode supervision', prepare)
        self.assertIn('--reuse-approved', prepare)
        self.assertIn('"$RUN/anchor_bank.py" ensure-casebriefs', prepare)
        self.assertLess(prepare.index('ensure-casebriefs'), prepare.index('bash "$RUN/matched_baseline.sh"'))
        self.assertLess(prepare.index('restore_anchor_evidence.py'), prepare.index('anchor_bank.py" prepare-lazy'))
        self.assertIn('seed_prime_checkout.py', prepare)
        self.assertLess(prepare.index('seed_prime_checkout.py'), prepare.index('bash "$BOOTSTRAP_RUN/setup_node.sh"'))
        self.assertIn('run_from_sft.py" --refresh-final-config', prepare)
        self.assertLess(prepare.index('bash "$RUN/matched_baseline.sh"'), prepare.index('run_from_sft.py" --refresh-final-config'))
        self.assertNotIn('flow_demonstrations.py" render', prepare)
        self.assertIn('supervision_phase.py" --data', prepare)
        self.assertIn('PAINTER_PRIME_ROLE=evaluation', evaluator)
        self.assertNotIn('PAINTER_PRIME_ROLE=orchestrator', evaluator)
        self.assertIn('PAINTER_STRUCTURED_FLOW_POLICY=1', evaluator)
        self.assertIn('PYTHONPATH="$REPO:$REPO/painter:$RUN"', evaluator)
        self.assertIn('export PAINTER_PRIME_ROLE=warmup', (RUN / 'warmup.sh').read_text())
        run_script = (RUN / 'run.sh').read_text()
        self.assertIn('export PAINTER_PRIME_ROLE=orchestrator', run_script)
        self.assertLess(run_script.index('PAINTER_WHOLE_GROUP_ADMISSION=1'), run_script.index('export PAINTER_PRIME_ROLE=orchestrator'))

    def test_ungrouped_evaluation_role_keeps_flow_hook_without_training_guard(self):
        """Baseline/eval must patch prompts without activating RL admission."""
        import runpy

        calls = []
        flow = types.ModuleType("flow_prompt_hook")

        def install():
            calls.append("flow")
            return {"installed": True}

        flow.install = install
        group = types.ModuleType("painter.consistent_group_admission")

        def forbidden_group_install():
            raise AssertionError("baseline/eval unexpectedly installed RL group guard")

        group.install_from_environment = forbidden_group_install
        with mock.patch.dict(
            sys.modules,
            {"flow_prompt_hook": flow, "painter.consistent_group_admission": group},
            clear=False,
        ), mock.patch.dict(
            os.environ,
            {"PAINTER_PRIME_ROLE": "evaluation", "PAINTER_STRUCTURED_FLOW_POLICY": "1"},
            clear=False,
        ):
            runpy.run_path(str(RUN / "sitecustomize.py"), run_name="structured_flow_eval_sitecustomize")
        self.assertEqual(calls, ["flow"])

    def test_orchestrator_without_group_admission_fails_closed_before_training_imports(self):
        """The RL role still fails before importing/creating training state."""
        import runpy

        calls = []
        group = types.ModuleType("painter.consistent_group_admission")

        def disabled_group_install():
            calls.append("group")
            return {"installed": False, "already_installed": False, "enabled": False}

        group.install_from_environment = disabled_group_install
        with mock.patch.dict(
            sys.modules, {"painter.consistent_group_admission": group}, clear=False
        ), mock.patch.dict(
            os.environ, {"PAINTER_PRIME_ROLE": "orchestrator"}, clear=False
        ):
            os.environ.pop("PAINTER_STRUCTURED_FLOW_POLICY", None)
            with self.assertRaises(SystemExit):
                runpy.run_path(str(RUN / "sitecustomize.py"), run_name="structured_flow_rl_sitecustomize")
        self.assertEqual(calls, ["group"])

    def test_runtime_process_env_marks_actual_uv_harness_child(self):
        """The pinned runtime boundary marks uv probes and task programs.

        The test uses a tiny Runtime stand-in only to exercise the run-local
        patch, then launches a real dependency-free PEP 723 script through uv
        with the resulting environment.  Prime remains intentionally absent.
        """

        repo = RUN.parents[2]
        provenance = json.loads(
            (RUN / "prime-runtime-manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(provenance["schema"], "prime-runtime-harness-provenance-v1")
        self.assertRegex(provenance["source_revision"], r"^[0-9a-f]{40}$")
        self.assertRegex(provenance["runtime_file_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            provenance["prepare_uv_script"]["path_template"],
            "{scripts_dir}/{sha256(script)}.py",
        )
        self.assertEqual(provenance["prepare_uv_script"]["argv_tail"], ["interpreter", "path"])
        runtime_source = repo / "work" / provenance["source_repository"] / provenance["runtime_file"]
        if runtime_source.is_file():
            self.assertEqual(
                hashlib.sha256(runtime_source.read_bytes()).hexdigest(),
                provenance["runtime_file_sha256"],
            )
            runtime_text = runtime_source.read_text(encoding="utf-8")
            self.assertIn('path = f"{self.scripts_dir}/{digest}.py"', runtime_text)
            self.assertIn("interpreter, path", runtime_text)

        runtime_module = types.ModuleType("verifiers.v1.runtimes.base")

        class Runtime:
            def process_env(self, env):
                return dict(env)

        runtime_module.Runtime = Runtime
        verifiers = types.ModuleType("verifiers")
        verifiers_v1 = types.ModuleType("verifiers.v1")
        verifiers_runtimes = types.ModuleType("verifiers.v1.runtimes")
        verifiers.v1 = verifiers_v1
        verifiers_v1.runtimes = verifiers_runtimes
        verifiers_runtimes.base = runtime_module
        group = types.ModuleType("painter.consistent_group_admission")
        group.install_from_environment = lambda: {"installed": True, "already_installed": False}
        flow = types.ModuleType("flow_prompt_hook")
        flow.install = lambda: {"installed": True}
        with mock.patch.dict(
            sys.modules,
            {
                "verifiers": verifiers,
                "verifiers.v1": verifiers_v1,
                "verifiers.v1.runtimes": verifiers_runtimes,
                "verifiers.v1.runtimes.base": runtime_module,
                "painter.consistent_group_admission": group,
                "flow_prompt_hook": flow,
            },
            clear=False,
        ), mock.patch.dict(
            os.environ,
            {"PAINTER_PRIME_ROLE": "orchestrator"},
            clear=False,
        ):
            os.environ.pop("PAINTER_STRUCTURED_FLOW_POLICY", None)
            os.environ.pop("PAINTER_ANCHOR_REQUIRE_NO_LOOP", None)
            import runpy

            runpy.run_path(str(RUN / "sitecustomize.py"), run_name="structured_flow_runtime_env_contract")
            launch_env = dict(os.environ)
            launch_env["PYTHONPATH"] = os.pathsep.join((str(RUN), str(repo), str(repo / "painter")))
            launch_env = Runtime().process_env(launch_env)
        self.assertEqual(launch_env["PAINTER_PRIME_ROLE"], "harness")

        uv = shutil.which("uv")
        if not uv:
            self.skipTest("uv is unavailable for the dependency-free PEP 723 subprocess probe")
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "harness_probe.py"
            cache = Path(directory) / "uv-cache"
            script.write_text(
                '# /// script\n# requires-python = ">=3.11"\n# ///\n'
                "import os\nprint('uv-harness-started', os.environ['PAINTER_PRIME_ROLE'])\n",
                encoding="utf-8",
            )
            uv_env = dict(launch_env)
            uv_env["UV_CACHE_DIR"] = str(cache)
            sync = subprocess.run(
                [uv, "sync", "--offline", "--script", str(script), "-q", "--no-config"],
                env=uv_env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(sync.returncode, 0, sync.stderr)
            interpreter = subprocess.check_output(
                [uv, "python", "find", "--offline", "--script", str(script), "--no-config"],
                env=uv_env,
                text=True,
                timeout=30,
            ).strip().splitlines()[-1]
            result = subprocess.run(
                [interpreter, str(script)],
                env=uv_env,
                capture_output=True,
                text=True,
                timeout=30,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("uv-harness-started harness", result.stdout)
        self.assertNotIn("consistent group admission activation failed", result.stderr)

    def test_orchestrator_subprocess_still_fails_closed_without_prime(self):
        """The same inherited environment still rejects a real entrypoint."""

        repo = RUN.parents[2]
        env = os.environ.copy()
        env.update(
            {
                "PAINTER_PRIME_ROLE": "orchestrator",
                "PAINTER_WHOLE_GROUP_ADMISSION": "1",
                "PAINTER_WHOLE_GROUP_SIZE": "8",
                "PAINTER_JUDGE_PROTOCOL": "text-paint-sft-pairwise-consistent-v1",
                "PYTHONPATH": os.pathsep.join((str(RUN), str(repo), str(repo / "painter"))),
            }
        )
        env.pop("PAINTER_STRUCTURED_FLOW_POLICY", None)
        env.pop("PAINTER_ANCHOR_REQUIRE_NO_LOOP", None)
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "orchestrator-entry.py"
            script.write_text("print('should-not-start')\n")
            result = subprocess.run(
                [sys.executable, str(script)],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("consistent group admission activation failed", result.stderr)
        self.assertIn("prime_rl", result.stderr)

    def test_runtime_isolation_missing_verifiers_fails_closed_with_system_exit(self):
        """An orchestrator never proceeds when the runtime patch cannot load."""

        import runpy

        group = types.ModuleType("painter.consistent_group_admission")
        group.install_from_environment = lambda: {"installed": True, "already_installed": False}
        with mock.patch.dict(
            sys.modules,
            {
                "painter.consistent_group_admission": group,
                "verifiers": None,
            },
            clear=False,
        ), mock.patch.dict(
            os.environ,
            {"PAINTER_PRIME_ROLE": "orchestrator"},
            clear=False,
        ):
            os.environ.pop("PAINTER_STRUCTURED_FLOW_POLICY", None)
            os.environ.pop("PAINTER_ANCHOR_REQUIRE_NO_LOOP", None)
            with self.assertRaises(SystemExit) as raised:
                runpy.run_path(str(RUN / "sitecustomize.py"), run_name="structured_flow_missing_runtime")
        self.assertIn("runtime environment isolation requires", str(raised.exception))

    def test_clean_tracked_source_builds_receipts_outside_strict_output_dirs(self):
        """Exercise the actual builders with the same shell redirection shape.

        A shell creates a redirection target before Python starts.  The
        curriculum builder intentionally rejects a non-empty output directory,
        so this test catches a regression where the node script puts its own
        receipt below that directory.  The source files are read from the
        tracked run tree rather than reimplemented in a string-only fixture.
        """
        repo = Path(subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=RUN, text=True
        ).strip())
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(RUN.relative_to(repo) / "prepare_curriculum.py")],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        self.assertEqual(tracked.returncode, 0, tracked.stderr)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "clean-run"
            root.mkdir()
            script = RUN / "prepare_curriculum.py"
            flow_script = RUN / "flow_demonstrations.py"
            builder = f'''set -eu
run_root="$1"
python="$2"
curriculum_builder="$3"
flow_builder="$4"
mkdir -p "$run_root/curriculum"
curriculum_tmp="$run_root/.curriculum-build.json.$$"
"$python" "$curriculum_builder" build --output "$run_root/curriculum" >"$curriculum_tmp"
mv "$curriculum_tmp" "$run_root/curriculum-build.json"
mkdir -p "$run_root/code-supervision"
flow_tmp="$run_root/.flow-supervision-build.json.$$"
"$python" "$flow_builder" build --output "$run_root/code-supervision" >"$flow_tmp"
mv "$flow_tmp" "$run_root/flow-supervision-build.json"
'''
            result = subprocess.run(
                ["bash", "-c", builder, "builder", str(root), sys.executable, str(script), str(flow_script)],
                cwd=repo,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root / "curriculum/train-proposal.json").is_file())
            self.assertTrue((root / "curriculum/heldout-proposal.json").is_file())
            self.assertEqual(json.loads((root / "curriculum-build.json").read_text())["status"], "prepared_unrendered")
            self.assertTrue((root / "code-supervision/manifest.json").is_file())
            self.assertTrue((root / "flow-supervision-build.json").is_file())
            self.assertEqual(list(root.glob(".curriculum-build.json.*")), [])
            self.assertEqual(list(root.glob(".flow-supervision-build.json.*")), [])

            protected = root / "protected-curriculum"
            protected.mkdir()
            sentinel = protected / "operator-note.txt"
            sentinel.write_text("preserve this file\n")
            refused = subprocess.run(
                [sys.executable, str(script), "build", "--output", str(protected)],
                cwd=repo,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("refusing to overwrite non-empty output", refused.stderr)
            self.assertEqual(sentinel.read_text(), "preserve this file\n")

    def test_warmup_upload_has_a_run_scoped_public_destination(self):
        runtime = (RUN / "runtime_env.sh").read_text()
        self.assertIn('HF_CHECKPOINT_REPO="${HF_CHECKPOINT_REPO:-CK0607/qwen3.8-27b-brush-painting}"', runtime)
        self.assertIn('HF_CHECKPOINT_RUN="${HF_CHECKPOINT_RUN:-structured-flow-rl-20261009-sft-warmup}"', runtime)

    def test_runtime_receipt_binds_active_warmup_bytes_and_preserves_parent(self):
        helper = load("structured_flow_runtime_receipt", RUN / "runtime_receipt.py")
        parent = "a" * 64
        active = "b" * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = root / "rl-config-receipt.json"
            receipt.write_text(json.dumps({
                "schema": "structured-flow-rl-config-receipt-v1",
                "policy_initializer": {"revision": "rev", "prefix": "prefix", "sha256": parent},
                "frozen_sft": {
                    "alias": "sft-step512",
                    "revision": "adf687e5b321514eda7be6eab050434fdc8e44ff",
                    "prefix": "photo-curriculum-sft-20260922-balanced/step_512",
                    "sha256": "bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740",
                },
            }))
            active_receipt = root / "active-policy.json"
            active_receipt.write_text(json.dumps({"status": "verified_warmup_adapter", "adapter_sha256": active}))
            helper.normalize(
                receipt,
                active_sha256=active,
                parent_sha256=parent,
                parent_revision="rev",
                parent_prefix="prefix",
                mode="recovered_policy",
                active_policy_json=active_receipt,
            )
            value = json.loads(receipt.read_text())
            self.assertEqual(value["adapter_sha256"], active)
            self.assertEqual(value["policy_initializer"]["sha256"], parent)
            self.assertEqual(value["active_policy"]["parent_sha256"], parent)
            self.assertEqual(value["frozen_anchor"], {
                "alias": "sft-step512",
                "revision": "adf687e5b321514eda7be6eab050434fdc8e44ff",
                "prefix": "photo-curriculum-sft-20260922-balanced/step_512",
                "checkpoint_sha256": "bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740",
                "immutable": True,
            })
            drifted = json.loads(receipt.read_text())
            drifted["frozen_sft"]["sha256"] = "c" * 64
            receipt.write_text(json.dumps(drifted))
            with self.assertRaises(ValueError):
                helper.normalize(
                    receipt,
                    active_sha256=active,
                    parent_sha256=parent,
                    parent_revision="rev",
                    parent_prefix="prefix",
                    mode="recovered_policy",
                )

    def test_run_binds_runtime_receipt_and_matched_baseline_flow(self):
        run = (RUN / "run.sh").read_text()
        baseline = (RUN / "matched_baseline.sh").read_text()
        evaluate = (RUN / "evaluate.sh").read_text()
        runtime = (RUN / "runtime_env.sh").read_text()
        self.assertIn('runtime_receipt.py', run)
        self.assertIn('PAINTER_RL_ADAPTER_SHA256="$ACTIVE_POLICY_SHA256"', run)
        self.assertIn('PAINTER_STRUCTURED_FLOW_POLICY=1', baseline)
        self.assertIn('PAINTER_PRIME_ROLE=baseline', baseline)
        self.assertIn('export PAINTER_PRIME_ROLE=baseline', baseline)
        self.assertIn('PAINTER_PRIME_ROLE=evaluation', baseline)
        self.assertIn('PAINTER_PRIME_ROLE=evaluation PAINTER_STRUCTURED_FLOW_POLICY=1', baseline)
        self.assertIn('PAINTER_PRIME_ROLE=baseline PAINTER_STRUCTURED_FLOW_POLICY=0', baseline)
        self.assertIn('evaluation-role-smoke.json', baseline)
        self.assertIn('NativePaintingTaskset', baseline)
        self.assertIn('evaluation_tasks_loaded', baseline)
        self.assertLess(baseline.index('evaluation-role-smoke.json'), baseline.index('setsid env CUDA_VISIBLE_DEVICES'))
        self.assertNotIn('PAINTER_PRIME_ROLE=orchestrator', baseline)
        self.assertNotIn('PAINTER_STRUCTURED_FLOW_POLICY="$PAINTER_STRUCTURED_FLOW_POLICY"', baseline)
        self.assertIn('CUDA_HOME=', baseline)
        self.assertIn('LD_LIBRARY_PATH="$CUDA_HOME/lib:', baseline)
        self.assertIn('"frozen_anchor"', (RUN / "runtime_receipt.py").read_text())
        frozen_opponent = (RUN.parents[2] / "painter/frozen_opponent.py").read_text()
        self.assertIn('environment["PAINTER_PRIME_ROLE"] = "evaluation"', frozen_opponent)
        self.assertIn('PAINTER_BROWSER_ROOT="$RUN/browsers"', evaluate)
        self.assertIn('PLAYWRIGHT_BROWSERS_PATH="$RUN/browsers"', evaluate)
        self.assertIn('export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$PAINTER_BROWSER_ROOT}"', runtime)

    def test_evaluate_forwards_step_scope_and_preflight_once(self):
        """Exercise the wrapper-to-generic-evaluator argv boundary offline."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stub_bin = root / "bin"
            stub_bin.mkdir()
            capture = root / "argv.txt"
            fake_python = stub_bin / "python3"
            fake_python.write_text(
                "#!/bin/sh\n"
                "target=\"$3\"\n"
                "cat >\"$target\" <<'EOF'\n"
                "#!/bin/sh\n"
                "printf '%s\\n' \"$@\" >\"$STUB_EVALUATOR_ARGV\"\n"
                "EOF\n"
                "chmod 700 \"$target\"\n"
            )
            fake_python.chmod(0o700)
            environment = os.environ.copy()
            environment.update({
                "PATH": str(stub_bin) + os.pathsep + environment.get("PATH", ""),
                "STUB_EVALUATOR_ARGV": str(capture),
            })
            for arguments, expected in (
                (("4", "all"), ["4", "all"]),
                (("4", "all", "--preflight"), ["4", "all", "--preflight"]),
            ):
                result = subprocess.run(
                    ["bash", str(RUN / "evaluate.sh"), *arguments],
                    cwd=RUN.parents[2],
                    env=environment,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(capture.read_text().splitlines(), expected)

    def test_anchor_batch_passes_verified_browser_root_to_native_eval(self):
        bank = load("structured_flow_anchor_browser_contract", RUN / "anchor_bank.py")
        frozen = importlib.import_module("painter.frozen_opponent")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            browser_root = root / "browsers"
            browser_root.mkdir()
            wrong_browser_root = root / "wrong-browsers"
            wrong_browser_root.mkdir()
            proposal_root = root / "proposal"
            proposal_root.mkdir()
            reference = proposal_root / "brief.txt"
            reference.write_text("Paint one orb.\n")
            renderer = root / "renderer.py"
            renderer.write_text("# pinned renderer fixture\n")
            renderer_python = root / "renderer-python"
            renderer_python.write_text("#!/bin/sh\n")
            eval_bin = root / "eval"
            eval_bin.write_text("#!/bin/sh\n")
            prime_root = root / "prime"
            prime_root.mkdir()
            source_png = root / "source.png"
            source_png.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
            captured = {}

            def fake_run(command, *, cwd, env, **kwargs):
                captured.update({"command": command, "cwd": cwd, "env": env})
                return subprocess.CompletedProcess(command, 0, "", "")

            row = {
                "id": "train-simple-001",
                "source_group": "train-simple-001",
                "reference": "brief.txt",
                "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
                "family": "simple",
            }
            environment = {
                "PAINTER_BROWSER_ROOT": str(browser_root),
                "PLAYWRIGHT_BROWSERS_PATH": str(wrong_browser_root),
                "PAINTER_FROZEN_OPPONENT_RENDERER": str(renderer),
                "PAINTER_FROZEN_OPPONENT_RENDERER_PYTHON": str(renderer_python),
                "PAINTER_FROZEN_OPPONENT_EVAL": str(eval_bin),
                "PAINTER_FROZEN_OPPONENT_PRIME_ROOT": str(prime_root),
            }
            with mock.patch.dict(os.environ, environment, clear=False), \
                 mock.patch.object(bank.subprocess, "run", side_effect=fake_run), \
                 mock.patch.object(frozen, "_native_eval_canvas", return_value=source_png):
                result = bank._batch_native_generation(
                    [row],
                    anchor_root=root / "anchors",
                    receipt_root=root / "anchors/native-generation",
                    proposal_root=proposal_root,
                    run_root=root / "run",
                    workers=1,
                )
            self.assertEqual(result["train-simple-001"]["valid"], True)
            self.assertEqual(captured["env"]["PAINTER_BROWSER_ROOT"], str(browser_root.resolve()))
            self.assertEqual(captured["env"]["PLAYWRIGHT_BROWSERS_PATH"], str(browser_root.resolve()))
            self.assertNotIn("wrong-browsers", captured["env"]["PLAYWRIGHT_BROWSERS_PATH"])
            self.assertEqual(captured["env"]["PAINTER_ANCHOR_REQUIRE_NO_LOOP"], "1")
            self.assertIn(str(RUN), captured["env"]["PYTHONPATH"].split(os.pathsep))
            self.assertEqual(captured["command"][0], str(eval_bin.resolve()))

    def test_anchor_no_loop_guard_rejects_unbounded_program_before_renderer(self):
        guard = load("structured_flow_anchor_no_loop_guard_contract", RUN / "anchor_no_loop_guard.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid = root / "invalid.js"
            invalid.write_text("function draw(){background('white');}\n")
            valid = root / "valid.js"
            valid.write_text("function draw(){background('white');noLoop();}\n")
            self.assertIn("noLoop", guard.no_loop_error(invalid) or "")
            self.assertIsNone(guard.no_loop_error(valid))
        sitecustomize = (RUN / "sitecustomize.py").read_text()
        self.assertIn("PAINTER_ANCHOR_REQUIRE_NO_LOOP", sitecustomize)

    def test_anchor_batch_resamples_only_model_invalid_cases_once(self):
        bank = load("structured_flow_anchor_resample_contract", RUN / "anchor_bank.py")
        rows = [
            {"id": "bad", "source_group": "bad", "reference": "bad.txt"},
            {"id": "good", "source_group": "good", "reference": "good.txt"},
        ]
        calls = []

        def fake_batch(batch_rows, **kwargs):
            retry_index = kwargs["retry_index"]
            failures = kwargs["failed_cases"]
            calls.append((retry_index, [row["id"] for row in batch_rows], kwargs["workers"]))
            if retry_index == 0:
                failures["bad"] = {"status": "invalid_model_output", "case_id": "bad", "case_seed": 11, "retry_index": 0}
                return {"good": {"valid": True}}
            failures["bad"] = {"status": "invalid_model_output", "case_id": "bad", "case_seed": 12, "retry_index": 1}
            return {}

        with mock.patch.object(bank, "_batch_native_generation", side_effect=fake_batch):
            provenance, failures = bank._batch_with_bounded_resample(
                rows,
                anchor_root=Path("/tmp/anchors"),
                receipt_root=Path("/tmp/anchors/native-generation"),
                proposal_root=Path("/tmp/proposal"),
                run_root=Path("/tmp/run"),
                workers=8,
            )
        self.assertEqual(calls, [(0, ["bad", "good"], 2), (1, ["bad"], 1)])
        self.assertEqual(sorted(provenance), ["good"])
        self.assertEqual(failures["bad"]["initial"]["case_seed"], 11)
        self.assertEqual(failures["bad"]["retry"]["case_seed"], 12)

    def test_anchor_resume_reuses_valid_paid_case_and_retries_only_invalid_case(self):
        bank = load("structured_flow_anchor_resume_contract", RUN / "anchor_bank.py")
        rows = [
            {"id": "bad", "source_group": "bad", "reference": "bad.txt"},
            {"id": "good", "source_group": "good", "reference": "good.txt"},
        ]
        calls = []
        resumed = {
            "good": {"valid": True, "generation_mode": "resumed_paid_batch"},
        }
        resumed_failures = {
            "bad": {"status": "invalid_model_output", "case_id": "bad", "retry_index": 0, "case_seed": 7},
        }

        def fake_batch(batch_rows, **kwargs):
            calls.append((kwargs["retry_index"], [row["id"] for row in batch_rows]))
            self.assertEqual(kwargs["retry_index"], 1)
            kwargs["failed_cases"].clear()
            return {"bad": {"valid": True, "generation_mode": "resampled_paid_batch"}}

        with mock.patch.object(bank, "_resume_existing_batches", return_value=(resumed, resumed_failures)), \
             mock.patch.object(bank, "_batch_native_generation", side_effect=fake_batch):
            provenance, failures = bank._batch_with_bounded_resample(
                rows,
                anchor_root=Path("/tmp/anchors"),
                receipt_root=Path("/tmp/anchors/native-generation"),
                proposal_root=Path("/tmp/proposal"),
                run_root=Path("/tmp/run"),
                workers=8,
            )
        self.assertEqual(calls, [(1, ["bad"])])
        self.assertEqual(sorted(provenance), ["bad", "good"])
        self.assertEqual(failures, {})

    def test_anchor_retry_zero_keeps_canonical_case_seed(self):
        bank = load("structured_flow_anchor_seed_contract", RUN / "anchor_bank.py")
        reference_sha = "a" * 64
        canonical = int.from_bytes(
            hashlib.sha256(f"20261009:group:{reference_sha}".encode()).digest()[:8],
            "big",
        ) & ((1 << 63) - 1)
        self.assertEqual(bank.deterministic_anchor_seed("group", reference_sha), canonical)
        self.assertNotEqual(bank.deterministic_anchor_seed("group", reference_sha), bank.deterministic_anchor_seed("group", reference_sha, 1))

    def test_anchor_receipt_records_native_current_canvas_terminal_semantics(self):
        bank = load("structured_flow_anchor_terminal_contract", RUN / "anchor_bank.py")
        trace = {
            "task": {"data": {"task_id": "case-02"}},
            "info": {
                "task_id": "case-02",
                "final_canvas": "/evidence/turn-2/painting.png",
                "final_canvas_valid": True,
                "operational_cutoff": True,
                "turns": [
                    {"turn": 1, "action": "paint", "valid": True, "canvas": "/evidence/turn-1/painting.png"},
                    {"turn": 2, "action": "paint", "valid": False, "failure_kind": "program"},
                    {"turn": 3, "action": "paint", "valid": False, "failure_kind": "program"},
                ],
            },
        }
        metadata = bank._terminal_metadata(([trace], [], []), "case-02")
        self.assertEqual(metadata["terminal_canvas_policy"], "native_current_canvas_at_episode_end")
        self.assertTrue(metadata["final_canvas_valid"])
        self.assertEqual(metadata["last_action_valid"], False)
        self.assertEqual(metadata["last_valid_canvas_turn"], 1)
        self.assertEqual(metadata["last_valid_canvas_reference"], "/evidence/turn-1/painting.png")
        self.assertEqual([row["turn"] for row in metadata["invalid_turns"]], [2, 3])
        self.assertEqual(metadata["terminal_canvas_reference"], "/evidence/turn-2/painting.png")

    def test_browser_probe_precedes_bootstrap_and_preserves_sandbox_diagnostics(self):
        prepare = (RUN / "prepare_node.sh").read_text()
        helper = (RUN / "prepare_renderer_runtime.py").read_text()
        self.assertIn('--capability-only', prepare)
        self.assertLess(prepare.index('--capability-only'), prepare.index('bash "$BOOTSTRAP_RUN/setup_node.sh"'))
        self.assertIn('PAINTER_FAST_STORAGE_HELPER', prepare)
        self.assertIn('chromium_sandbox=True', helper)
        self.assertIn('--use-angle=swiftshader', helper)
        self.assertIn('--enable-unsafe-swiftshader', helper)
        self.assertIn('"stderr": stderr', helper)
        self.assertIn('stderr_first', helper)
        self.assertIn('stderr_last', helper)
        self.assertIn('NoNewPrivs=1', helper)
        self.assertIn('Seccomp:', helper)
        self.assertIn('cap_eff', helper)
        self.assertIn('--render-probe', prepare)
        self.assertLess(prepare.index('--render-probe'), prepare.index('prepare_curriculum.py'))
        self.assertIn('PLAYWRIGHT_BROWSERS_PATH', helper)
        self.assertIn('structured-flow-native-render-probe-v1', helper)
        self.assertIn('from painter.native_painting_renderer import render_action', helper)
        self.assertIn('env=environment', helper)
        self.assertIn('"PAINTER_BROWSER_ROOT": str(browsers)', helper)
        self.assertIn('"AMBIENT_CANARY_SECRET"', (RUN / "test_run_contract.py").read_text())

    def test_native_render_probe_uses_sanitized_env_and_nested_boundary(self):
        """The paid-node probe must exercise native render_action safely.

        A mocked subprocess stands in for Prime and the renderer.  It writes
        the same native program/PNG/receipt files that render_action produces,
        which lets this test verify the real probe contract without launching
        Chromium or touching a model endpoint.
        """
        helper = load("structured_flow_renderer_probe_contract", RUN / "prepare_renderer_runtime.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = root / "run"
            renderer_root = root / "renderer"
            assets = renderer_root / "assets"
            assets.mkdir(parents=True)
            (assets / "brush.js").write_text("brush fixture\n")
            renderer = renderer_root / "renderer.py"
            renderer.write_text("# pinned renderer fixture\n")
            renderer_python = root / "prime-python"
            renderer_python.write_text("#!/bin/sh\n")
            browsers = root / "browsers"
            browsers.mkdir()
            prime_venv = root / "prime-venv"
            node_bin = prime_venv / "bin"
            node_bin.mkdir(parents=True)
            node = node_bin / "node"
            actual_node = shutil.which("node")
            if actual_node is None:
                self.fail("PATH regression requires a real Node.js binary; run with the bundled runtime on PATH")
            node.symlink_to(actual_node)
            renderer_sha = helper._sha256(renderer)
            asset_map = {
                path.relative_to(assets).as_posix(): helper._sha256(path)
                for path in sorted(assets.rglob("*"))
                if path.is_file() and not path.is_symlink()
            }
            asset_sha = hashlib.sha256(
                (json.dumps(asset_map, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
            ).hexdigest()
            captured = {}

            real_run = subprocess.run
            real_which = helper.shutil.which

            def fake_run(command, *, cwd, env, **kwargs):
                captured.update({"command": command, "cwd": cwd, "env": dict(env), "kwargs": kwargs})
                nested_root = Path(command[3])
                native_folder = Path(command[4])
                probe_source = Path(command[5])
                selected_renderer = Path(command[6])
                row_path = Path(command[8])
                node_check = real_run(
                    ["node", "--check", str(probe_source)],
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(node_check.returncode, 0, node_check.stderr)
                native_source = native_folder / "program.js"
                native_source.write_text(probe_source.read_text())
                output = native_folder / "painting.png"
                output.write_bytes(b"\x89PNG\r\n\x1a\nprobe")
                receipt = {
                    "valid": True,
                    "source_sha256": helper._sha256(native_source),
                    "png_sha256": helper._sha256(output),
                    "renderer_sha256": helper._sha256(selected_renderer),
                }
                output.with_suffix(".json").write_text(json.dumps(receipt))
                row_path.write_text(json.dumps({"valid": True, "canvas": str(output)}))
                self.assertTrue(nested_root.is_dir())
                return subprocess.CompletedProcess(command, 0, "", "")

            def which(name):
                return None if name == "runuser" else real_which(name)

            with mock.patch.dict(os.environ, {
                     "AMBIENT_CANARY_SECRET": "must-not-cross",
                     "PATH": "/controller/path-without-node",
                 }, clear=False), \
                 mock.patch.object(helper.sys, "platform", "linux"), \
                 mock.patch.object(helper.shutil, "which", side_effect=which), \
                 mock.patch.object(helper.subprocess, "run", side_effect=fake_run):
                result = helper.render_probe(
                    run_root=run_root,
                    renderer=renderer,
                    renderer_python=renderer_python,
                    browsers=browsers,
                    expected_renderer_sha256=renderer_sha,
                    prime_venv=prime_venv,
                    expected_assets_sha256=asset_sha,
                )
            self.assertTrue(result["valid"])
            self.assertEqual(captured["env"]["PAINTER_BROWSER_ROOT"], str(browsers.resolve()))
            self.assertEqual(captured["env"]["PLAYWRIGHT_BROWSERS_PATH"], str(browsers.resolve()))
            self.assertNotIn("AMBIENT_CANARY_SECRET", captured["env"])
            self.assertEqual(captured["env"]["PYTHONPATH"], str(Path(__file__).resolve().parents[3]))
            self.assertEqual(captured["env"]["PATH"].split(os.pathsep)[0], str(node_bin.resolve()))
            self.assertIn("timeout", captured["kwargs"])
            self.assertEqual(result["nested_root"], str((run_root / "runtime/native-render-probe/nested-attempt").resolve()))

    def test_prepare_node_prime_interpreter_probe_isolated_from_bootstrap(self):
        """Run the actual preparation probe with deliberately split interpreters."""
        try:
            import pydantic  # noqa: F401
        except Exception as exc:  # pragma: no cover - host-dependent test dependency
            self.skipTest(f"test host lacks pydantic: {exc}")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            (bin_dir / "uname").write_text("#!/bin/sh\nprintf '%s\\n' Linux\n")
            (bin_dir / "uname").chmod(0o755)
            right_marker = root / "right-called"
            wrong_marker = root / "wrong-called"
            right = root / "prime-python"
            right.write_text(
                "#!/bin/sh\n"
                f"printf '%s\\n' right > {shlex.quote(str(right_marker))}\n"
                f"exec {shlex.quote(sys.executable)} \"$@\"\n"
            )
            right.chmod(0o755)
            wrong = root / "bootstrap-python"
            wrong.write_text(
                "#!/bin/sh\n"
                f"printf '%s\\n' wrong > {shlex.quote(str(wrong_marker))}\n"
                "exit 97\n"
            )
            wrong.chmod(0o755)
            environment = os.environ.copy()
            environment.update({
                "PATH": f"{bin_dir}{os.pathsep}{environment.get('PATH', '')}",
                "PAINTER_PRIME_PYTHON": str(right),
                "PAINTER_BOOTSTRAP_PYTHON": str(wrong),
                "PAINTER_RUN_ROOT": str(RUN),
                "PRIME_ROOT": str(root / "prime"),
            })
            result = subprocess.run(
                ["bash", str(RUN / "prepare_node.sh"), "--interpreter-probe"],
                cwd=RUN.parents[2],
                env=environment,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('"schema": "structured-flow-prime-interpreter-probe-v1"', result.stdout)
            self.assertTrue(right_marker.is_file())
            self.assertFalse(wrong_marker.exists())
            prepare = (RUN / "prepare_node.sh").read_text()
            self.assertIn('"$PY_PRIME" "$RUN/prepare_structured_config.py"', prepare)
            self.assertIn("Runtime.run_uv_script", prepare)
            self.assertIn('"schema": "structured-flow-prime-runtime-harness-probe-v1"', prepare)
            self.assertLess(prepare.index("PRIME_RUNTIME_PROBE_RECEIPT"), prepare.index("download_adapter()"))
            self.assertNotIn('HOME="$PRIME_RUNTIME_PROBE_HOME"', prepare)
            self.assertIn("uuid.uuid4().hex", prepare)
            self.assertIn('PAINTER_STRUCTURED_FLOW_POLICY=0', prepare)

    def test_render_probe_preserves_byte_timeout_diagnostics(self):
        helper = load("structured_flow_renderer_timeout_contract", RUN / "prepare_renderer_runtime.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            renderer_root = root / "renderer"
            (renderer_root / "assets").mkdir(parents=True)
            renderer = renderer_root / "renderer.py"
            renderer.write_text("# pinned renderer fixture\n")
            renderer_python = root / "prime-python"
            renderer_python.write_text("#!/bin/sh\n")
            browsers = root / "browsers"
            browsers.mkdir()
            node = root / "node"
            node.write_text("#!/bin/sh\nexit 0\n")
            node.chmod(0o755)
            with mock.patch.object(helper.sys, "platform", "linux"), \
                 mock.patch.object(helper, "_resolve_node_executable", return_value=node), \
                 mock.patch.object(
                     helper.subprocess,
                     "run",
                     side_effect=subprocess.TimeoutExpired(["prime"], 240, output=b"partial out", stderr=b"partial err"),
                 ):
                with self.assertRaisesRegex(RuntimeError, "evidence preserved"):
                    helper.render_probe(
                        run_root=root / "run",
                        renderer=renderer,
                        renderer_python=renderer_python,
                        browsers=browsers,
                        expected_renderer_sha256=helper._sha256(renderer),
                    )
            probe_root = root / "run/runtime/native-render-probe"
            self.assertEqual((probe_root / "renderer.stdout.log").read_text(), "partial out")
            self.assertEqual((probe_root / "renderer.stderr.log").read_text(), "partial err")
            receipt = json.loads((probe_root / "renderer-probe-receipt.json").read_text())
            self.assertEqual(receipt["status"], "censored_invalid")

    def test_shell_scripts_compile(self):
        scripts = ["runtime_env.sh", "prepare_node.sh", "run.sh", "execute_single.sh", "execute_pipeline.sh", "evaluate.sh", "finalize.sh", "matched_baseline.sh"]
        for name in scripts:
            result = subprocess.run(["bash", "-n", str(RUN / name)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, f"{name}: {result.stderr}")


if __name__ == "__main__":
    unittest.main()
