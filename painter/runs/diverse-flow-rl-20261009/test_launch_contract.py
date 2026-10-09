"""Offline launch checks for the Diverse-flow Linux runtime boundary."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


RUN = Path(__file__).resolve().parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DiverseFlowLaunchContractTests(unittest.TestCase):
    def test_runtime_defaults_bind_four_gpu_roles_and_full_counts(self):
        command = f"source {RUN / 'runtime_env.sh'}; printf '%s %s %s %s %s\n' \"$PAINTER_GPUS_PER_NODE\" \"$PAINTER_NUM_TRAIN_GPUS\" \"$PAINTER_NUM_INFER_GPUS\" \"$PAINTER_EXPECTED_COUNT\" \"$PAINTER_EVAL_EXPECTED_COUNT\""
        result = subprocess.run(["bash", "-c", command], check=True, capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), "4 2 2 96 32")

    def test_training_controller_forwards_required_warmup(self):
        operation = load("diverse_flow_operation_contract", RUN / "operate_single.py")
        command = operation._detached_job_command(
            source_commit="a" * 40, rental_start_epoch=1.0,
            price_usd_per_hour=1.52, cap_usd=20.0,
        )
        self.assertIn("PAINTER_RUN_WARMUP=1", command)
        self.assertIn("PAINTER_WARMUP_REQUIRED=1", command)

    def test_warmup_uses_three_rank_nested_torchrun_and_batch_three(self):
        warmup = load("diverse_flow_warmup_contract", RUN / "warmup.py")
        self.assertEqual(warmup.resolve_warmup_gpu({}), "0,1,2")
        with self.assertRaises(ValueError):
            warmup.resolve_warmup_gpu({"PAINTER_WARMUP_GPUS": "0,1"})
        command = warmup.build_train_command(Path("/tmp/runtime"), Path("/tmp/warmup.toml"), "prime-python")
        self.assertIn("--nproc-per-node=3", command)
        self.assertEqual(command[-3:], [str(Path("/tmp/runtime") / "integrations/speedpainting/mixed_sft/train.py"), "@", "/tmp/warmup.toml"])
        config = warmup.build_config(Path("/tmp/model"), Path("/tmp/data"))
        self.assertEqual(config["deployment"], {"gpus_per_node": 3, "num_train_gpus": 3})
        self.assertEqual(config["data"]["batch_size"], 3)
        trainer = (RUN / "train_from_sft.py").read_text()
        self.assertIn('WORLD_SIZE', trainer)
        self.assertIn('expected_world != 2', trainer)
        self.assertIn('prime_runtime', trainer)
        self.assertIn('helpers_for_hooks', trainer)
        self.assertIn('structured-flow-sft-source-manifest-v1', (RUN / "warmup.py").read_text())

    def test_initialization_report_matches_runtime_for_both_modes(self):
        warmup = load("diverse_flow_warmup_initialization_contract", RUN / "warmup.py")
        runtime = (RUN.parent / "structured-flow-rl-20261009/sft_source/mixed_sft/train.py").read_text()
        self.assertIn("setup_report.get('step_zero_export_required')", runtime)
        for mode, adapter, expected in (
            ("trained_adapter", "/node/parent-adapter", False),
            ("fresh_base", None, True),
        ):
            report = warmup.initialization_contract(mode, init_adapter=adapter)
            self.assertIs(report["step_zero_export_required"], expected)
            self.assertEqual(report["step_zero_export_required"], mode == "fresh_base")
            self.assertEqual(report["optimizer"], "fresh")

    def test_prepare_writes_explicit_trained_adapter_step_zero_contract(self):
        warmup = load("diverse_flow_warmup_prepare_contract", RUN / "warmup.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            dev = root / "dev.jsonl"
            train.write_text("")
            dev.write_text("")
            parent = {
                "path": str(root / "parent-adapter"), "sha256": "a" * 64,
                "revision": "revision", "prefix": "prefix",
            }
            args = SimpleNamespace(
                output=root / "prepared", parent_adapter=root / "parent-adapter",
                train=train, dev=dev, approval_manifest=None,
                model_dir=root / "model", run_name="Diverse-flow-sft-warmup",
            )
            with patch.object(warmup, "verify_parent_adapter", return_value=parent), \
                 patch.object(warmup, "validate_inputs", return_value={"status": "approved"}), \
                 patch.object(warmup, "build_schedule", return_value=[]), \
                 patch.object(warmup, "build_schedule_receipt", return_value={"mode": "legacy_ordered_painting"}), \
                 patch.object(warmup, "validate_warmup_schedule"), \
                 patch.object(warmup, "build_config", return_value={}):
                warmup.prepare(args)
            report = json.loads((root / "prepared" / "resolved.setup.json").read_text())
            self.assertEqual(report["initialization_mode"], "trained_adapter")
            self.assertIs(report["step_zero_export_required"], False)
            self.assertEqual(report["step_zero_export_required"], report["initialization_mode"] == "fresh_base")
            self.assertEqual(report["optimizer"], "fresh")

    def test_role_isolation_and_renderer_receipts_are_explicit(self):
        baseline = (RUN / "matched_baseline.sh").read_text()
        evaluate = (RUN / "evaluate.sh").read_text()
        prepare = (RUN / "prepare_node.sh").read_text()
        self.assertIn("PAINTER_PRIME_ROLE=baseline", baseline)
        self.assertIn("PAINTER_PRIME_ROLE=evaluation", evaluate)
        self.assertIn("--tensor-parallel-size 2", baseline)
        self.assertIn("PAINTER_EVAL_GPUS:-2,3", evaluate)
        self.assertIn("dc6b9d837ed7be382784a6c84de1b64911506b17a19b499ce054c0c572e75794", prepare)
        self.assertIn('"$PY_PRIME" "$RUN/supervision_phase.py"', prepare)
        self.assertIn("anchor-evidence restore is intentionally disabled", prepare)
        self.assertIn("import PIL", (RUN / "warmup.sh").read_text())

    def test_pinned_source_probe_schema_and_token_refresh_are_accepted(self):
        warmup = load("diverse_flow_warmup_probe_contract", RUN / "warmup.py")
        config = warmup.build_config(Path("/tmp/model"), Path("/tmp/data"))
        probe = {
            "schema": "structured-flow-sft-entrypoint-probe-v1",
            "status": "config_parsed", "model_profile": "qwen38_27b",
            "renderer": config["renderer"]["name"], "max_steps": config["max_steps"],
            "batch_size": config["data"]["batch_size"],
            "micro_batch_size": config["data"]["micro_batch_size"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.jsonl"
            path.write_text(json.dumps(probe) + "\n")
            self.assertEqual(warmup._verify_entrypoint_probe(path, config)["schema"], probe["schema"])

        helper = load("diverse_flow_upload_contract", RUN.parents[1] / "rl_helpers/scripts/upload_checkpoints.py")
        launch = (RUN / "launch_rl.sh").read_text()
        self.assertIn("unset HF_TOKEN", launch)
        self.assertIn("rereads HF_TOKEN_PATH", launch)
        with tempfile.TemporaryDirectory() as directory:
            token_path = Path(directory) / "hf-token"
            token_path.write_text("first-token\n")
            token_path.chmod(0o600)
            with patch.dict(os.environ, {"HF_TOKEN_PATH": str(token_path), "HF_TOKEN": "stale-token"}, clear=False):
                helper._refresh_hf_token()
                self.assertEqual(os.environ["HF_TOKEN"], "first-token")
                token_path.write_text("rotated-token\n")
                token_path.chmod(0o600)
                helper._refresh_hf_token()
                self.assertEqual(os.environ["HF_TOKEN"], "rotated-token")

    def test_actual_legacy_warmup_loader_shards_twelve_rows_across_three_ranks(self):
        # The source is the immutable structured-flow package restored by the
        # new run; it is intentionally not re-authored under the Diverse ID.
        source = RUN.parent / "structured-flow-rl-20261009/sft_source/mixed_sft/scheduled_data.py"
        scheduled = load("diverse_flow_scheduled_data_contract", source)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            rows = [{"id": f"warmup-{index:02d}", "task_family": "painting", "split": "train"} for index in range(12)]
            (folder / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            config = SimpleNamespace(
                data=SimpleNamespace(name=str(folder), batch_size=3, shuffle=False,
                                     micro_batch_size=1, num_workers=1, splits=["train"],
                                     subsets=None, probabilities=None),
                max_steps=4,
                model=SimpleNamespace(cp=1),
                resume=None,
                deployment=SimpleNamespace(num_train_gpus=3, gpus_per_node=3),
            )
            self.assertIsNone(scheduled.guard_runtime(config, {"WORLD_SIZE": "3", "LOCAL_WORLD_SIZE": "3"}))
            dataset = scheduled.load_exact_split(folder, "train")
            shards = [[dataset[index]["id"] for index in range(rank, len(dataset), 3)] for rank in range(3)]
            self.assertEqual(sorted(sum((shard for shard in shards), [])), [row["id"] for row in rows])
            self.assertEqual(len(set().union(*map(set, shards))), 12)
            self.assertTrue(all(set(shards[index]).isdisjoint(shards[other]) for index in range(3) for other in range(index + 1, 3)))

    def test_warmup_materializes_runtime_local_uploader_override(self):
        warmup = load("diverse_flow_warmup_uploader_contract", RUN / "warmup.py")
        source = RUN.parent / "structured-flow-rl-20261009/sft_source"
        with tempfile.TemporaryDirectory() as directory:
            runtime = warmup.materialize_source_layout(source, Path(directory) / "prepared")
            wrapper = runtime / "scripts/upload_checkpoints.py"
            backend = runtime / "scripts/upload_checkpoints_backend.py"
            self.assertTrue(wrapper.is_file() and not wrapper.is_symlink())
            self.assertTrue(backend.is_file() and not backend.is_symlink())
            receipt = warmup.source_identity(runtime)
            self.assertEqual(receipt["upload_override"]["schema"], "Diverse-flow-upload-token-override-v1")
            self.assertEqual(receipt["upload_override"]["backend_sha256"], warmup.sha256_file(backend))
            self.assertIn("HF_TOKEN_PATH", wrapper.read_text())
            original_init = (source / "watercolour_sft/init_adapter.py").read_bytes()
            runtime_init = runtime / "integrations/prime/watercolour_sft/init_adapter.py"
            archived_init = runtime / "integrations/prime/watercolour_sft/init_adapter.py.archived.py"
            self.assertIn("Diverse-flow-multirank-lora-init-v1", runtime_init.read_text())
            self.assertEqual(archived_init.read_bytes(), original_init)
            self.assertEqual(
                receipt["prime_hook_override"]["original_files"]["integrations/prime/watercolour_sft/init_adapter.py"],
                warmup.sha256_file(archived_init),
            )
            update_audit = runtime / "integrations/speedpainting/mixed_sft/update_audit.py"
            group_validation = runtime / "integrations/speedpainting/mixed_sft/group_validation.py"
            self.assertIn("all_gather_object", update_audit.read_text())
            self.assertIn("rank_reports", update_audit.read_text())
            self.assertIn("group-validation.rank", group_validation.read_text())
        warmup_shell = (RUN / "warmup.sh").read_text()
        self.assertIn("structured-flow-rl-20261009/sft_source/mixed_sft", warmup_shell)

    def test_group_validation_transform_binds_run_globals_into_prime_namespace(self):
        group = load(
            "diverse_flow_group_validation_contract",
            RUN / "prime_runtime/integrations/speedpainting/mixed_sft/group_validation.py",
        )
        source = (
            "def run_validation(step: int) -> None:\n"
            "    gc_handler = GarbageCollection(config.gc.interval) if config.gc else None\n"
            "def train():\n"
            "    run_validation(0)\n"
            "    # Write final checkpoint\n"
            "    pass\n"
        )
        group.SOURCE_SHA = hashlib.sha256(source.encode()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "prime_train.py"
            source_path.write_text(source)
            fake_module = ModuleType("fake_prime_train")
            fake_module.__file__ = str(source_path)
            with patch.object(group.importlib, "import_module", return_value=fake_module):
                group.install(Path(directory) / "effective")
            self.assertEqual(fake_module.__dict__["OVERRIDE_SCHEMA"], group.OVERRIDE_SCHEMA)
            self.assertIn("os", fake_module.__dict__)
            self.assertTrue(callable(fake_module.__dict__["train"]))

    def test_run_owned_source_guard_accepts_own_blob_and_rejects_tamper(self):
        helper = RUN / "verify_run_owned_inputs.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            run = repo / "painter/runs/diverse-flow-rl-20261009"
            fake_bin = root / "bin"
            repo.mkdir(parents=True)
            run.mkdir(parents=True)
            fake_bin.mkdir()
            reviewed = root / "reviewed-launch.sh"
            reviewed.write_text("#!/bin/sh\necho reviewed\n")
            (run / "launch_rl.sh").write_bytes(reviewed.read_bytes())
            fake_git = fake_bin / "git"
            fake_git.write_text("#!/bin/sh\ncat \"$REVIEWED_BLOB\"\n")
            fake_git.chmod(0o700)
            env = dict(os.environ, PATH=f"{fake_bin}:{os.environ.get('PATH', '')}", REVIEWED_BLOB=str(reviewed))
            command = [
                sys.executable, str(helper), "--repo", str(repo), "--run-root", str(run),
                "--relative-root", "painter/runs/diverse-flow-rl-20261009", "--file", "launch_rl.sh",
            ]
            accepted = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            (run / "launch_rl.sh").write_text("tampered\n")
            rejected = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("differs", rejected.stderr)

    def test_owner_hf_refresh_copies_authorized_token_and_uses_atomic_remote_rename(self):
        operation = load("diverse_flow_operation_token_refresh_contract", RUN / "operate_single.py")

        class FakeSFTP:
            def __init__(self):
                self.files = {}
                self.renames = []

            def put(self, local, remote):
                self.files[remote] = Path(local).read_bytes()

            def chmod(self, remote, mode):
                self.files.setdefault(remote, b"")

            def posix_rename(self, source, destination):
                self.files[destination] = self.files.pop(source)
                self.renames.append((source, destination))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            authorized = root / "authorized-token"
            cache = root / "normal-cache-token"
            authorized.write_text("old-token\n")
            cache.write_text("fresh-token\n")
            authorized.chmod(0o600)
            cache.chmod(0o600)
            with patch.object(operation.subprocess, "run") as run:
                result = operation._refresh_local_hf_auth(authorized=authorized, cache=cache)
            self.assertEqual(result["status"], "refreshed")
            self.assertEqual(authorized.read_text(), "fresh-token")
            run.assert_called_once()
            self.assertNotIn("HF_TOKEN", run.call_args.kwargs["env"])

            sftp = FakeSFTP()
            digest = operation._rotate_remote_hf_token(sftp=sftp, local=authorized, timeout=2.0)
            self.assertEqual(sftp.files[str(operation.base.REMOTE_TOKEN)], b"fresh-token")
            self.assertEqual(sftp.renames, [(str(operation.HF_REMOTE_ROTATION_TMP), str(operation.base.REMOTE_TOKEN))])
            self.assertEqual(digest, operation.hashlib.sha256(b"fresh-token").hexdigest())

            context = {"events": root / "events.jsonl", "started": 1.0}
            with patch.object(operation.base, "_safe_event") as record:
                operation._record_hf_refresh(
                    {"status": "refreshed", "changed": True,
                     "authorized_sha256": "credential-derived", "remote_sha256": "credential-derived"},
                    now=lambda: 2.0, context=context,
                )
            fields = record.call_args.kwargs
            self.assertNotIn("authorized_sha256", fields)
            self.assertNotIn("remote_sha256", fields)

    def test_native_eval_keeps_traces_under_run_root(self):
        evaluate = (RUN / "evaluate.sh").read_text()
        recovery = (RUN / "recover_evaluation.sh").read_text()
        self.assertIn('PAINTER_RUN_ROOT="$RUN"', evaluate)
        self.assertIn('Diverse-flow-final-evaluation-v1', evaluate)
        self.assertIn('renderer-selection-receipt.json', (RUN / "matched_baseline.sh").read_text())
        self.assertIn("evaluation-rollouts", recovery)
        self.assertIn("matched32-manifest.json", evaluate)

    def test_renderer_file_resolution_and_eval_toml_audit_are_executable(self):
        baseline = (RUN / "matched_baseline.sh").read_text()
        evaluate = (RUN / "evaluate.sh").read_text()
        self.assertIn('Path(sys.argv[1]).resolve()', baseline)
        self.assertNotIn('cd "$RENDERER"', baseline)
        self.assertIn('rf"(?m)^\\s*{key}\\s*=\\s*(\\d+)\\s*$"', baseline)
        self.assertIn('rf"(?m)^\\s*{key}\\s*=\\s*(\\d+)\\s*$"', evaluate)
        # Exercise the same regex against a native-looking TOML fragment;
        # this catches the accidental double-backslash raw-regex regression.
        import re
        text = "num_tasks = 32\nmax_turns = 3\nmax_tokens = 8192\nmax_concurrent = 8\nseed = 918210\n"
        for key, expected in (("num_tasks", 32), ("max_turns", 3), ("max_tokens", 8192), ("max_concurrent", 8), ("seed", 918210)):
            match = re.search(rf"(?m)^\s*{key}\s*=\s*(\d+)\s*$", text)
            self.assertIsNotNone(match)
            self.assertEqual(int(match.group(1)), expected)

    def test_train_censoring_rebinds_actual_admitted_count(self):
        materialize = (RUN / "materialize.py").read_text()
        run_from_sft = (RUN / "run_from_sft.py").read_text()
        node_validate = (RUN / "node_validate.py").read_text()
        self.assertIn("MIN_ADMITTED_TRAIN_ROWS = 88", materialize)
        self.assertIn("len(admitted_train_ids)", materialize)
        self.assertIn("88 <= admitted_count <= 96", run_from_sft)
        self.assertIn("88 <= expected_train <= 96", node_validate)


if __name__ == "__main__":
    unittest.main()
