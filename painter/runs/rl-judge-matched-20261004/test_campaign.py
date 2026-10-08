"""Offline contract tests for the matched three-judge campaign."""

from __future__ import annotations

import io
import importlib.util
import json
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest


RUN = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("matched_prepare", RUN / "prepare_campaign.py")
assert SPEC and SPEC.loader
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)
VERIFY_SPEC = importlib.util.spec_from_file_location(
    "matched_public_download", RUN / "verify_public_download.py"
)
assert VERIFY_SPEC and VERIFY_SPEC.loader
verify_public_download = importlib.util.module_from_spec(VERIFY_SPEC)
VERIFY_SPEC.loader.exec_module(verify_public_download)
OPERATION_SPEC = importlib.util.spec_from_file_location("matched_operation_reporting", RUN / "operate_campaign.py")
assert OPERATION_SPEC and OPERATION_SPEC.loader
operate_campaign = importlib.util.module_from_spec(OPERATION_SPEC)
sys.modules[OPERATION_SPEC.name] = operate_campaign
OPERATION_SPEC.loader.exec_module(operate_campaign)


CONFIG = """
max_steps = 12
output_dir = "/tmp/arm/async-output"
[run]
name = "arm"
[deployment]
type = "single_node"
gpus_per_node = 2
num_train_gpus = 1
num_infer_gpus = 1
[ckpt]
interval = 4
keep_last = 1
[trainer]
max_steps = 12
[trainer.model.ac_offloading]
pin_memory = true
max_inflight_activations = 1
[trainer.optim]
lr = 1e-6
[orchestrator]
batch_size = 16
group_size = 4
max_steps = 12
max_off_policy_steps = 1
[orchestrator.concurrency]
min_inflight = 16
initial_inflight = 32
max_inflight = 32
[orchestrator.train.sampling]
temperature = 1.0
top_p = 1.0
max_completion_tokens = 8192
[[orchestrator.train.source]]
name = "text-paint-fixed-sft-pairwise"
[orchestrator.train.source.env.taskset]
root = "/tmp/arm"
manifest = "anchors-frozen/train-manifest.json"
renderer = "/tmp/arm/renderer.py"
renderer_python = "/tmp/arm/python"
expected_count = 40
max_turns = 3
preserve_manifest_order = true
lazy_opponent = false
opponent_model_alias = "sft-step512"
opponent_checkpoint_sha256 = "bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740"
opponent_protocol_version = "native-painting-rl-v1"
opponent_render_protocol = "native-painting-v1"
opponent_seed = 20260922
opponent_max_completion_tokens = 8192
opponent_context_length = 32768
"""


def _write_config(path: Path, *, lr: str = "1e-6") -> None:
    path.write_text(CONFIG.replace("lr = 1e-6", f"lr = {lr}"), encoding="utf-8")


class MatchedCampaignTests(unittest.TestCase):
    class _SFTP:
        def __init__(self, files):
            self.files = files

        def open(self, path, _mode):
            return io.BytesIO(self.files.get(path, b""))

    def _summary_sftp(self, *, state, metrics=b"", trainer_log=b"", arms=()):
        active_root = operate_campaign.REMOTE_RUN.parent / f"{operate_campaign.RUN_NAME}-flash"
        files = {
            str(operate_campaign.REMOTE_RUN / "campaign/status.json"): json.dumps({
                "status": "running", "reason": "arm_flash_running",
            }).encode(),
            str(operate_campaign.REMOTE_RUN / "materialization-receipt.json"): b"{}",
            str(operate_campaign.REMOTE_RUN / "campaign/terminal-receipt.json"): b"{}",
            str(operate_campaign.REMOTE_RUN / "campaign/arms.jsonl"): b"".join(
                json.dumps(row).encode() + b"\n" for row in arms
            ),
            str(operate_campaign.REMOTE_JOB_START): b"{}",
            str(operate_campaign.REMOTE_JOB_EXIT): b"{}",
            str(active_root / "supervisor-state.json"): json.dumps(state).encode(),
            str(active_root / f"async-output/{operate_campaign.RUN_NAME}-flash/monitors/file/metrics.jsonl"): metrics,
            str(active_root / f"async-output/{operate_campaign.RUN_NAME}-flash/logs/attempt_1/trainer.log"): trainer_log,
        }
        return self._SFTP(files)

    def test_completed_step_parser_excludes_admitted_orchestrator_and_inference_rows(self):
        metrics = b"\n".join([
            json.dumps({"step": 8, "train/agg/effective/agent/reward/mean": 0.5}).encode(),
            json.dumps({"step": 8, "producer": "orchestrator", "time/step": 99.0}).encode(),
            json.dumps({"step": 8, "producer": "inference", "optim/grad_norm": 0.01}).encode(),
            json.dumps({"step": 7, "producer": "trainer", "optim/grad_norm": 0.01}).encode(),
        ]) + b"\n"
        self.assertEqual(operate_campaign._completed_optimizer_steps_from_metrics(metrics), {7})
        self.assertEqual(
            operate_campaign._completed_optimizer_steps_from_trainer_log(b"SUCCESS ok Step 8 | done\n"),
            {8},
        )

    def test_summary_fallback_uses_trainer_log_and_does_not_double_count_active_arm(self):
        rows = [
            {"arm": "flash", "completed_optimizer_steps": 8},
            {"arm": "pro", "completed_optimizer_steps": 12},
            # A reconnect can repeat a finished arm receipt; totals must use
            # one latest row per arm rather than double-counting it.
            {"arm": "pro", "completed_optimizer_steps": 12},
            {"arm": "luna", "completed_optimizer_steps": 12},
        ]
        sftp = self._summary_sftp(
            state={"status": "running", "completed_optimizer_steps": None, "admitted_batch_counter": 8},
            metrics=(json.dumps({"step": 8, "train/agg/effective/agent/reward/mean": 0.5}) + "\n").encode(),
            trainer_log=b"INFO SUCCESS optimizer Step 8 | grad_norm=0.0107\n",
            arms=rows,
        )
        summary = operate_campaign._campaign_summary(sftp, timeout=5.0)
        self.assertEqual(summary["state"]["active_completed_optimizer_steps"], 8)
        self.assertEqual(summary["state"]["completed_optimizer_steps"], 32)
        self.assertEqual(summary["state"]["last_admitted_batch"], 8)
        self.assertNotIn("SUCCESS", json.dumps(summary))

    def test_summary_fallback_uses_trainer_metric_but_not_admitted_batch(self):
        metrics = b"\n".join([
            json.dumps({"step": 8, "train/agg/effective/agent/reward/mean": 0.5}).encode(),
            json.dumps({"step": 8, "producer": "trainer", "optim/grad_norm": 0.0107}).encode(),
        ]) + b"\n"
        summary = operate_campaign._campaign_summary(
            self._summary_sftp(
                state={"status": "running", "completed_optimizer_steps": None},
                metrics=metrics,
            ),
            timeout=5.0,
        )
        self.assertEqual(summary["state"]["active_completed_optimizer_steps"], 8)

    def test_shared_prepare_binds_fast_storage_and_renderer_hooks_before_bootstrap(self):
        script = (RUN / "prepare.sh").read_text(encoding="utf-8")
        source_at = script.index('source "$RUN/runtime_env.sh"')
        limits_at = script.index("apply_matched_resource_limits")
        cache_at = script.index('--phase cache')
        bootstrap_at = script.index('bash "$BASE/prepare_node.sh"')
        self.assertLess(source_at, bootstrap_at)
        self.assertLess(limits_at, bootstrap_at)
        self.assertLess(cache_at, bootstrap_at)
        self.assertLess(script.index('PAINTER_RENDERER_RUNTIME_HELPER'), bootstrap_at)
        self.assertIn('timeout --signal=TERM --kill-after=30 "$SHARED_PREPARE_TIMEOUT"', script)
        self.assertIn('PAINTER_FAST_STORAGE_HELPER', script)

    def test_shared_prepare_rehomes_verified_frozen_sft_path_after_bootstrap(self):
        script = (RUN / "prepare.sh").read_text(encoding="utf-8")
        bootstrap_path = script.index('FROZEN_SFT_BOOTSTRAP_PATH_FILE=')
        inherited_export = script.index('export PAINTER_FROZEN_SFT_ADAPTER_PATH_FILE="$FROZEN_SFT_BOOTSTRAP_PATH_FILE"')
        base_prepare = script.index('bash "$BASE/prepare_node.sh"')
        shared_copy = script.index('mv -f "$temporary" "$SHARED/frozen-sft-adapter-path.txt"')
        self.assertLess(bootstrap_path, inherited_export)
        self.assertLess(inherited_export, base_prepare)
        self.assertGreater(shared_copy, base_prepare)
        self.assertIn('adapter_model.safetensors', script[shared_copy - 600:shared_copy])

    def test_baseline_wrapper_uses_flat_shared_manifest_and_public_archive(self):
        """Exercise the generated baseline shell lifecycle without a node."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "painter"
            run = repo / "runs/rl-judge-matched-20261004"
            run.mkdir(parents=True)
            source_baseline = RUN / "baseline.sh"
            baseline = run / "baseline.sh"
            weight = b"downloaded-public-initializer"
            import hashlib
            weight_sha = hashlib.sha256(weight).hexdigest()
            baseline.write_text(
                source_baseline.read_text().replace(
                    "e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c",
                    weight_sha,
                ),
                encoding="utf-8",
            )
            baseline.chmod(0o755)
            (run / "verify_public_download.py").write_text(
                (RUN / "verify_public_download.py").read_text(), encoding="utf-8"
            )
            (run / "verify_public_download.py").chmod(0o755)
            (run / "runtime_env.sh").write_text((RUN / "runtime_env.sh").read_text(), encoding="utf-8")
            (run / "initializer-public-proof.json").write_text(json.dumps({
                "schema": "baseline-anonymous-public-metadata-v1",
                "repo_id": "CK0607/qwen3.8-27b-brush-painting",
                "revision": "ac4d3eda8e5925e2d23992cfdc1a63620ebde573",
                "prefix": "text-pairwise-scaled-20260929/step_16",
                "private": False, "adapter_sha256": weight_sha,
                "hf_upload_receipt_present": False,
                "anonymous_metadata_verified": True,
            }) + "\n")
            shared = repo / "runs/rl-judge-matched-20261004-shared"
            (shared / "anchors-frozen").mkdir(parents=True)
            for name in ("anchors-all", "vendor", "renderer-env", "browsers", "runtime"):
                (shared / name).mkdir()
            (shared / "preparation-receipt.json").write_text("{}\n")
            cases = [{"id": f"case-{index}"} for index in range(16)]
            (shared / "anchors-frozen/validation-matched16.json").write_text(
                json.dumps({"cases": cases})
            )
            initializer = root / "initializer"
            initializer.mkdir()
            (initializer / "adapter_model.safetensors").write_bytes(weight)
            (initializer / "adapter_config.json").write_text("{}\n")
            (shared / "policy-adapter-path.txt").write_text(str(initializer) + "\n")
            (shared / "policy-adapter-receipt.json").write_text(json.dumps({
                "repo": "CK0607/qwen3.8-27b-brush-painting",
                "revision": "ac4d3eda8e5925e2d23992cfdc1a63620ebde573",
                "prefix": "text-pairwise-scaled-20260929/step_16",
                "adapter_path": str(initializer), "adapter_sha256": weight_sha,
                "verified": True,
            }) + "\n")
            (shared / "frozen-sft-adapter-path.txt").write_text(str(initializer) + "\n")
            (repo / "runs/rl-judge-matched-20261004/prepare_arm.sh").write_text("#!/bin/sh\n")
            base = repo / "runs/text-sft512-consistent-20260930"
            base.mkdir(parents=True)
            evaluate_script = (
                "#!/bin/sh\n"
                "printf '%s %s %s\\n' \"$PAINTER_P5_BRUSH_API_REFERENCE_V2\" \"$PAINTER_INSPECT_CHOICE_FEEDBACK\" \"$PAINTER_REQUIRE_VALID_FINAL_ACTION\" > \"$PAINTER_RUN_ROOT/common-flags.txt\"\n"
                "mkdir -p \"$PAINTER_RUN_ROOT/validation/step-1-balanced16\"\n"
                "printf '%s\\n' '{\"adapter_sha256\":\"e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c\",\"scope\":\"balanced16\"}' > \"$PAINTER_RUN_ROOT/validation/step-1-balanced16/generation-receipt.json\"\n"
                "cat > \"$PAINTER_RUN_ROOT/validation/step-1-balanced16/cases.json\" <<'JSON'\n"
                + json.dumps({"cases": cases}) + "\nJSON\n"
            )
            (base / "evaluate.sh").write_text(evaluate_script)
            (base / "evaluate.sh").chmod(0o755)
            safe_archive = base / "safe_archive.py"
            safe_archive.write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, sys\n"
                "args = sys.argv[1:]\n"
                "output = pathlib.Path(args[args.index('--output') + 1])\n"
                "output.parent.mkdir(parents=True, exist_ok=True); output.write_bytes(b'archive')\n"
                "output.with_suffix(output.suffix + '.upload.json').write_text(json.dumps({'private': False, 'public_hash_verified': True}) + '\\n')\n"
            )
            safe_archive.chmod(0o755)
            prime_python = root / "prime/.venv/bin/python"
            prime_python.parent.mkdir(parents=True)
            prime_python.symlink_to(Path("/usr/bin/python3"))
            hf = root / "huggingface_hub"
            hf.mkdir()
            (hf / "__init__.py").write_text(textwrap.dedent(f"""
                class _Lfs:
                    sha256 = {weight_sha!r}
                class _Sibling:
                    rfilename = 'text-pairwise-scaled-20260929/step_16/adapter_model.safetensors'
                    lfs = _Lfs()
                class _Info:
                    sha = 'ac4d3eda8e5925e2d23992cfdc1a63620ebde573'
                    private = False
                    siblings = [_Sibling()]
                class HfApi:
                    def __init__(self, token=None):
                        assert token is False
                    def model_info(self, repo, revision, files_metadata=True):
                        return _Info()
            """), encoding="utf-8")
            token = root / "hf-token"
            token.write_text("test-token\n")
            token.chmod(0o600)
            command_bin = root / "bin"
            command_bin.mkdir()
            (command_bin / "uname").write_text("#!/bin/sh\nprintf 'Linux\\n'\n")
            (command_bin / "stat").write_text("#!/bin/sh\nif [ \"$1\" = -c ]; then printf '600\\n'; else exit 1; fi\n")
            (command_bin / "timeout").write_text(
                "#!/bin/sh\n"
                "shift; shift; shift; exec \"$@\"\n"
            )
            for path in command_bin.iterdir():
                path.chmod(0o755)
            env = os.environ.copy()
            env.update({
                "PRIME_ROOT": str(root / "prime"),
                "HF_TOKEN_PATH": str(token),
                "PAINTER_MATCHED_BASELINE_ROOT": str(root / "baseline"),
                "PATH": str(command_bin) + os.pathsep + env["PATH"],
                "PYTHONPATH": str(root),
            })
            result = subprocess.run([str(baseline)], cwd=run, env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertTrue((root / "baseline/local-evidence/baseline-evidence.tar.gz.upload.json").is_file())
            self.assertEqual((root / "baseline/common-flags.txt").read_text().strip(), "1 1 1")
            public = json.loads((root / "baseline/public-checkpoint-receipt.json").read_text())
            self.assertEqual(public["provenance_type"], "public_download")
            self.assertFalse((initializer / "hf-upload.json").exists())

    def test_public_download_normalizer_rejects_bad_receipts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            adapter = root / "adapter"
            adapter.mkdir()
            weight = adapter / "adapter_model.safetensors"
            weight.write_bytes(b"initializer")
            (adapter / "adapter_config.json").write_text("{}\n")
            import hashlib
            digest = hashlib.sha256(weight.read_bytes()).hexdigest()
            expected = {
                "repo": "repo", "revision": "r" * 40, "prefix": "p/step_16",
                "adapter_path": str(adapter), "adapter_sha256": digest, "verified": True,
            }

            class Lfs:
                sha256 = digest
            class Sibling:
                rfilename = "p/step_16/adapter_model.safetensors"
                lfs = Lfs()
            class Info:
                sha = "r" * 40
                private = False
                siblings = [Sibling()]

            def metadata(repo, revision):
                return Info()

            def normalize(value):
                receipt = root / "download.json"
                receipt.write_text(json.dumps(value))
                return verify_public_download.normalize_download(
                    download_receipt=receipt, adapter=adapter, output=root / "public.json",
                    repo="repo", revision="r" * 40, prefix="p/step_16",
                    expected_sha256=digest, metadata_lookup=metadata,
                )

            self.assertTrue(normalize(expected)["public_metadata_verified"])
            malformed = dict(expected, verified=False)
            with self.assertRaises(ValueError):
                normalize(malformed)
            private = dict(expected, private=True)
            with self.assertRaises(ValueError):
                normalize(private)
            wrong_hash = dict(expected, adapter_sha256="0" * 64)
            with self.assertRaises(ValueError):
                normalize(wrong_hash)

    def test_evaluator_preflight_accepts_download_public_receipt_without_upload(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "painter/runs/eval-run"
            base = root / "painter/runs/text-sft512-consistent-20260930"
            run.mkdir(parents=True)
            base.mkdir(parents=True)
            (root / "painter/native_painting_eval.py").write_text(textwrap.dedent("""
                import argparse
                from pathlib import Path
                parser = argparse.ArgumentParser()
                parser.add_argument('--config', type=Path, required=True)
                args, _ = parser.parse_known_args()
                args.config.write_text(
                    'base_url = "http://127.0.0.1:8000/v1"\\n'
                    'seed = 918200\\n'
                    'max_concurrent = 16\\n'
                )
            """), encoding="utf-8")
            evaluate = base / "evaluate.sh"
            evaluate.write_text((RUN.parent / "text-sft512-consistent-20260930/evaluate.sh").read_text())
            evaluate.chmod(0o755)
            prime = root / "prime/.venv/bin"
            prime.mkdir(parents=True)
            prime.joinpath("python").symlink_to(Path("/usr/bin/python3"))
            for name in ("vllm", "eval"):
                path = prime / name
                path.write_text("#!/bin/sh\nexit 0\n")
                path.chmod(0o755)
            renderer_python = run / "renderer-env/bin/python"
            renderer_python.parent.mkdir(parents=True)
            renderer_python.symlink_to(Path("/usr/bin/python3"))
            (run / "renderer.py").write_text("// fixture\n")
            (run / "anchors-frozen").mkdir()
            cases = (
                [{"id": str(index), "family": "simple"} for index in range(4)] +
                [{"id": str(index + 4), "family": "intermediate"} for index in range(4)] +
                [{"id": str(index + 8), "family": "complex"} for index in range(8)]
            )
            (run / "anchors-frozen/validation-matched16.json").write_text(json.dumps({
                "cases": cases
            }))
            adapter = run / "async-output/eval-run/artifacts/adapters/step_1"
            adapter.mkdir(parents=True)
            weight = b"downloaded-public-initializer"
            adapter.joinpath("adapter_model.safetensors").write_bytes(weight)
            import hashlib
            digest = hashlib.sha256(weight).hexdigest()
            receipt = root / "public-checkpoint-receipt.json"
            receipt.write_text(json.dumps({
                "provenance_type": "public_download", "verified": True, "private": False,
                "public_metadata_verified": True,
                "files": {"adapter_model.safetensors": {"sha256": digest}},
                "commit": "r" * 40,
            }))
            playwright = root / "playwright"
            playwright.mkdir()
            (playwright / "__init__.py").write_text("")
            env = os.environ.copy()
            env.update({
                "PRIME_ROOT": str(root / "prime"), "PAINTER_RUN_ROOT": str(run),
                "PAINTER_RUN_NAME": "eval-run", "PAINTER_EVAL_MANIFEST": str(run / "anchors-frozen/validation-matched16.json"),
                "PAINTER_EVAL_PUBLIC_RECEIPT": str(receipt), "PAINTER_EVAL_EXPECTED_COUNT": "16",
                "PAINTER_RENDERER_PATH": str(run / "renderer.py"),
                "PAINTER_RENDERER_PYTHON_PATH": str(renderer_python),
                "PYTHONPATH": str(root),
            })
            result = subprocess.run([str(evaluate), "1", "balanced16", "--preflight"],
                                    env=env, cwd=run, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_plan_and_full_native_config_equality_rejects_recipe_drift(self):
        protocol = prepare.load_protocol()
        self.assertEqual(prepare.arm_identity(protocol, "flash")["judge_model"], "xiaomi/mimo-v2.6-flash")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first, second = root / "flash.toml", root / "pro.toml"
            _write_config(first)
            _write_config(second)
            self.assertEqual(prepare.normalized_native_config(first), prepare.normalized_native_config(second))
            _write_config(second, lr="2e-6")
            with self.assertRaises(ValueError):
                prepare.assert_common_native_configs([first, second])

    def test_actual_config_writer_materializes_twelve_update_profile(self):
        config_spec = importlib.util.spec_from_file_location(
            "matched_config_writer", RUN.parent / "text-pairwise-rl-20260928" / "prepare_config.py"
        )
        assert config_spec and config_spec.loader
        writer = importlib.util.module_from_spec(config_spec)
        config_spec.loader.exec_module(writer)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "anchors-frozen/train-manifest.json"
            manifest.parent.mkdir()
            manifest.write_text(json.dumps({"rows": [
                {"id": str(index), "reference_kind": "text", "opponent_sha256": "a" * 64}
                for index in range(40)
            ]}))
            writer.prepare(
                output=root / "native.toml", model=Path("/tmp/fake-model"), run_root=root,
                manifest=manifest, max_steps=12, checkpoint_interval=4,
                run_name="rl-judge-matched-20261004-flash", expected_count=40,
                judge_protocol="text-paint-sft-pairwise-consistent-v1",
                adapter_revision=prepare.INITIALIZER["revision"], adapter_prefix=prepare.INITIALIZER["prefix"],
                adapter_sha256=prepare.INITIALIZER["sha256"], mode="recovered_policy",
                learning_rate=1e-6, gpus_per_node=2, num_train_gpus=1, num_infer_gpus=1,
                batch_size=16, group_size=4, initial_inflight=32, max_inflight=32,
                activation_cpu_offload=True,
            )
            normalized = prepare.normalized_native_config(root / "native.toml")
            self.assertEqual(normalized["max_steps"], 12)
            self.assertEqual(normalized["ckpt"]["interval"], 4)
            self.assertEqual(normalized["orchestrator"]["concurrency"]["max_inflight"], 32)

    def test_materialize_runs_actual_config_writer_for_three_flat_arms(self):
        """Exercise materialize's subprocess handoff with a fake Prime validator."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_root = root / "repo/painter/runs/rl-judge-matched-20261004"
            source_root.mkdir(parents=True)
            shared = root / "repo/painter/runs/rl-judge-matched-20261004-shared"
            (shared / "anchors-frozen").mkdir(parents=True)
            train_rows = [
                {"id": f"train-{index}", "reference_kind": "text", "opponent_sha256": "a" * 64}
                for index in range(40)
            ]
            validation_cases = (
                [{"id": f"simple-{index}", "family": "simple"} for index in range(4)] +
                [{"id": f"intermediate-{index}", "family": "intermediate"} for index in range(4)] +
                [{"id": f"complex-{index}", "family": "complex"} for index in range(39)]
            )
            (shared / "anchors-frozen/train-manifest.json").write_text(json.dumps({"rows": train_rows}))
            (shared / "anchors-frozen/validation-manifest.json").write_text(json.dumps({"cases": validation_cases}))
            (shared / "preparation-receipt.json").write_text("{}\n")
            for name in ("anchors-all", "vendor", "renderer-env", "browsers", "runtime"):
                (shared / name).mkdir()
            (shared / "runtime/watercolour-cancellable-centroid-v1").mkdir(parents=True)
            adapter = root / "adapters/initializer"
            adapter.mkdir(parents=True)
            (shared / "policy-adapter-path.txt").write_text(str(adapter) + "\n")
            (shared / "frozen-sft-adapter-path.txt").write_text(str(adapter) + "\n")
            for name in ("prepare_arm.sh", "prepare_node.sh", "run.sh", "evaluate.sh", "finalize.sh",
                         "runtime_env.sh", "finite_supervisor.py"):
                (source_root / name).write_text("#!/bin/sh\n")
            prime = root / "prime"
            module_root = prime / "src/prime_rl/configs"
            module_root.mkdir(parents=True)
            (module_root / "__init__.py").write_text("")
            (module_root.parent / "__init__.py").write_text("")
            (module_root / "rl.py").write_text(
                "class RLConfig:\n"
                "    @classmethod\n"
                "    def model_validate(cls, value):\n"
                "        return value\n"
            )
            model = root / "model"
            model.mkdir()
            prepare_config = RUN.parent / "text-pairwise-rl-20260928/prepare_config.py"
            old_pythonpath = os.environ.get("PYTHONPATH")
            os.environ["PYTHONPATH"] = os.pathsep.join(
                item for item in (str(root / "repo"), str(prime / "src"), old_pythonpath) if item
            )
            try:
                receipt = prepare.materialize(
                    shared_root=shared, prime_root=prime, model_dir=model,
                    prepare_config=prepare_config, source_root=source_root,
                )
            finally:
                if old_pythonpath is None:
                    os.environ.pop("PYTHONPATH", None)
                else:
                    os.environ["PYTHONPATH"] = old_pythonpath
            self.assertEqual(len(receipt["arms"]), 3)
            for arm in ("flash", "pro", "luna"):
                arm_root = root / f"repo/painter/runs/rl-judge-matched-20261004-{arm}"
                self.assertTrue((arm_root / "native.toml").is_file())
            self.assertTrue((arm_root / "rl-config-receipt.json").is_file())
            self.assertEqual(prepare.normalized_native_config(arm_root / "native.toml")["max_steps"], 12)
            self.assertEqual(json.loads((arm_root / "rl-config-receipt.json").read_text())["batch_size"], 16)
            # A prepared arm may be rematerialized while it has no training
            # output, but any existing async output must stop the handoff.
            (root / "repo/painter/runs/rl-judge-matched-20261004-flash/async-output").mkdir()
            with self.assertRaises(ValueError):
                prepare.materialize(
                    shared_root=shared, prime_root=prime, model_dir=model,
                    prepare_config=prepare_config, source_root=source_root,
                )

    def test_actual_three_arm_run_preflight_propagates_recovered_mode(self):
        """Run each real arm wrapper through the real Prime launcher preflight."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "painter"
            source_root = repo / "runs/rl-judge-matched-20261004"
            source_root.mkdir(parents=True)
            shutil.copy2(RUN.parents[1] / "policy_runner.sh", repo / "policy_runner.sh")
            for name in ("arm_env.sh", "run.sh", "run_arm.sh", "runtime_env.sh",
                         "prepare_arm.sh", "prepare_node.sh", "evaluate.sh", "finalize.sh",
                         "finite_supervisor.py"):
                shutil.copy2(RUN / name, source_root / name)
            shared = repo / "runs/rl-judge-matched-20261004-shared"
            (shared / "anchors-frozen").mkdir(parents=True)
            train_rows = [
                {"id": f"train-{index}", "reference_kind": "text", "opponent_sha256": "a" * 64}
                for index in range(40)
            ]
            validation_cases = (
                [{"id": f"simple-{index}", "family": "simple"} for index in range(4)] +
                [{"id": f"intermediate-{index}", "family": "intermediate"} for index in range(4)] +
                [{"id": f"complex-{index}", "family": "complex"} for index in range(39)]
            )
            (shared / "anchors-frozen/train-manifest.json").write_text(json.dumps({"rows": train_rows}))
            (shared / "anchors-frozen/validation-manifest.json").write_text(
                json.dumps({"cases": validation_cases})
            )
            (shared / "preparation-receipt.json").write_text("{}\n")
            for name in ("anchors-all", "vendor", "renderer-env", "browsers", "runtime"):
                (shared / name).mkdir()
            (shared / "runtime/watercolour-cancellable-centroid-v1").mkdir(parents=True)
            (shared / "runtime/watercolour-cancellable-centroid-v1/cancellable_watercolour_renderer.py").write_text("# fixture\n")
            (shared / "vendor/integrations/watercolour").mkdir(parents=True)
            (shared / "vendor/integrations/watercolour/renderer.py").write_text("# fixture\n")
            renderer_python = shared / "renderer-env/bin/python"
            renderer_python.parent.mkdir(parents=True)
            renderer_python.write_text("#!/bin/sh\nexec /usr/bin/python3 \"$@\"\n")
            renderer_python.chmod(0o755)
            (shared / "sitecustomize.py").write_text("# fixture\n")
            for name in ("run_from_sft.py", "train_from_sft.py", "completion_relay.py", "safe_archive.py"):
                (shared / name).write_text("# fixture\n")
            (shared / "launch_rl.sh").write_text(
                (RUN.parent / "photo-curriculum-rl-20260922/launch_rl.sh").read_text()
            )

            initial = root / "adapters/initializer"
            frozen = root / "adapters/frozen"
            initial.mkdir(parents=True)
            frozen.mkdir(parents=True)
            adapter_config = {
                "r": 16, "lora_alpha": 32,
                "target_modules": [
                    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
                ],
                "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            }
            initial_weight = b"fixture-initializer"
            frozen_weight = b"fixture-frozen"
            import hashlib
            initial_sha = hashlib.sha256(initial_weight).hexdigest()
            frozen_sha = hashlib.sha256(frozen_weight).hexdigest()
            arm_env_path = source_root / "arm_env.sh"
            arm_env_path.write_text(arm_env_path.read_text().replace(
                "e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c", initial_sha
            ))
            for path, weight in ((initial, initial_weight), (frozen, frozen_weight)):
                (path / "adapter_model.safetensors").write_bytes(weight)
                (path / "adapter_config.json").write_text(json.dumps(adapter_config))
            (shared / "policy-adapter-path.txt").write_text(str(initial) + "\n")
            (shared / "frozen-sft-adapter-path.txt").write_text(str(frozen) + "\n")

            prime = root / "prime"
            (prime / "src/prime_rl/configs").mkdir(parents=True)
            (prime / "src/prime_rl/configs/__init__.py").write_text("")
            (prime / "src/prime_rl/__init__.py").write_text("")
            (prime / "src/prime_rl/configs/rl.py").write_text(
                "class RLConfig:\n"
                "    @classmethod\n"
                "    def model_validate(cls, value):\n"
                "        return value\n"
            )
            cuda = root / "fake-cuda"
            (cuda / "bin").mkdir(parents=True)
            python = prime / ".venv/bin/python"
            python.parent.mkdir(parents=True)
            python.write_text(
                "#!/bin/sh\n"
                f"if [ \"$1\" = -c ] && echo \"$2\" | grep -q sysconfig; then echo {cuda!s}; exit 0; fi\n"
                f"exec {sys.executable} \"$@\"\n"
            )
            python.chmod(0o755)
            model = root / "node/model-cache/qwen/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
            model.mkdir(parents=True)
            (model / "config.json").write_text("{}\n")
            (root / "painter/final_action_credit.py").parent.mkdir(parents=True, exist_ok=True)
            (root / "painter/final_action_credit.py").write_text("# fixture\n")
            brush = root / "painter/rl_helpers"
            for name in (
                "integrations/prime/watercolour_sft/init_adapter.py",
                "integrations/prime/watercolour_sft/export_adapter.py",
                "integrations/prime/watercolour_sft/model_profiles.py",
                "scripts/upload_checkpoints.py",
            ):
                path = brush / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# fixture\n")
            private = root / "node/private"
            private.mkdir(parents=True)
            (private / "gateway.env").write_text("GATEWAY_TEST=1\n")
            (private / "gateway.env").chmod(0o600)
            (private / "hf-token").write_text("token-test\n")

            # The test adapters are small fixtures, so substitute their hashes
            # in copied launch/config receipts while retaining every launcher
            # identity and mode check.
            launch = shared / "launch_rl.sh"
            launch.write_text(launch.read_text().replace(
                "e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c", initial_sha
            ).replace(
                "bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740", frozen_sha
            ))
            old_prepare_config = prepare.INITIALIZER["sha256"]
            try:
                receipt = prepare.materialize(
                    shared_root=shared, prime_root=prime, model_dir=model,
                    prepare_config=RUN.parent / "text-pairwise-rl-20260928/prepare_config.py",
                    source_root=source_root,
                )
            finally:
                self.assertEqual(prepare.INITIALIZER["sha256"], old_prepare_config)
            self.assertEqual(len(receipt["arms"]), 3)
            for arm in ("flash", "pro", "luna"):
                arm_root = repo / f"runs/rl-judge-matched-20261004-{arm}"
                for path in (arm_root / "native.toml", arm_root / "rl-config-receipt.json"):
                    path.write_text(path.read_text().replace(
                        "e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c", initial_sha
                    ).replace(
                        "bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740", frozen_sha
                    ))

            # Keep the launcher dependency-free in this preflight: the actual
            # serializer round-trip runs against a tiny test stub, and no
            # Prime package or renderer is loaded.
            (root / "painter/native_painting_rl.py").write_text(textwrap.dedent("""
                class _DataType:
                    @classmethod
                    def model_validate(cls, value):
                        return cls()
                    def model_dump(self, mode='json', exclude_none=True):
                        return {'task_id': 'fixture-task'}
                class _Data:
                    task_id = 'fixture-task'
                    def model_dump(self, mode='json', exclude_none=True):
                        return {'task_id': self.task_id}
                class _Task:
                    def __init__(self, data=None, config=None):
                        self.data = data or _Data()
                        self.config = config
                        self.key = 'fixture-key'
                    def data_type(self):
                        return _DataType
                class NativePaintingRLConfig:
                    @classmethod
                    def model_validate(cls, value):
                        return value
                class NativePaintingRLTaskset:
                    def __init__(self, config):
                        self.config = config
                    def load(self):
                        return iter((_Task(config=self.config),))
            """), encoding="utf-8")
            command_bin = root / "bin"
            command_bin.mkdir()
            (command_bin / "uname").write_text("#!/bin/sh\nprintf 'Linux\\n'\n")
            (command_bin / "stat").write_text("#!/bin/sh\nif [ \"$1\" = -c ]; then printf '600\\n'; else exit 1; fi\n")
            (command_bin / "git").write_text(
                "#!/bin/sh\n"
                "if [ \"$1\" = -C ]; then shift 2; fi\n"
                "if [ \"$1\" = rev-parse ]; then printf '26b3131d2716a4f8210b165df584b83b4bc54f61\\n'; exit 0; fi\n"
                "exit 1\n"
            )
            for path in command_bin.iterdir():
                path.chmod(0o755)
            env = {
                "PATH": str(command_bin) + os.pathsep + os.environ["PATH"],
                "PAINTER_NODE_ROOT": str(root / "node"),
                "PRIME_ROOT": str(prime),
                "PAINTER_MATCHED_ARM_ROOT_BASE": str(repo / "runs"),
                "PYTHONPATH": os.pathsep.join((str(root), str(prime / "src"))),
            }
            for arm in ("flash", "pro", "luna"):
                arm_root = repo / f"runs/rl-judge-matched-20261004-{arm}"
                command = (
                    f"source {source_root / 'arm_env.sh'} {arm}; "
                    f"PAINTER_RL_SUPERVISED=1 bash {arm_root / 'run.sh'} --preflight"
                )
                result = subprocess.run(["bash", "-c", command], env=env, cwd=arm_root,
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                self.assertIn('"status":"preflight_passed"', result.stdout)
                self.assertEqual(
                    json.loads((arm_root / "initial-adapter-receipt.json").read_text())["mode"],
                    "recovered_policy",
                )
            # Reuse the same prepared inputs through the historical pilot
            # wrapper. Its profile validator remains in the thin wrapper while
            # the canonical runner owns the launch body.
            pilot_root = repo / "runs/throughput-recovery-20261003"
            shutil.copytree(repo / "runs/rl-judge-matched-20261004-flash", pilot_root,
                            symlinks=True)
            shutil.copy2(RUN.parents[1] / "runs/throughput-recovery-20261003/run_single.sh",
                         pilot_root / "run_single.sh")
            pilot_runtime = (RUN.parents[1] / "runs/throughput-recovery-20261003/runtime_env.sh").read_text()
            if (pilot_root / "runtime_env.sh").is_symlink():
                (pilot_root / "runtime_env.sh").unlink()
            (pilot_root / "runtime_env.sh").write_text(pilot_runtime.replace(
                "e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c", initial_sha
            ))
            (repo / "throughput_recovery_profile.py").write_text(
                (RUN.parents[1] / "throughput_recovery_profile.py").read_text()
            )
            pilot_env = dict(env)
            pilot_env.update({
                "PAINTER_RUN_ROOT": str(pilot_root),
                "PAINTER_FROZEN_SFT_ADAPTER_PATH_FILE": str(shared / "frozen-sft-adapter-path.txt"),
                "PAINTER_POLICY_SHA256": initial_sha,
            })
            for name in ("native.toml", "rl-config-receipt.json"):
                (pilot_root / name).unlink()
            subprocess.run([
                sys.executable, str(RUN.parent / "text-pairwise-rl-20260928/prepare_config.py"),
                "--output", str(pilot_root / "native.toml"), "--model", str(model),
                "--run-root", str(pilot_root), "--manifest", str(pilot_root / "anchors-frozen/train-manifest.json"),
                "--prime-root", str(prime), "--max-steps", "4", "--checkpoint-interval", "1",
                "--run-name", "throughput-recovery-20261003", "--output-root", "rl-rollouts",
                "--judge-protocol", "text-paint-sft-pairwise-consistent-v1", "--expected-count", "40",
                "--adapter-revision", "ac4d3eda8e5925e2d23992cfdc1a63620ebde573",
                "--adapter-prefix", "text-pairwise-scaled-20260929/step_16",
                "--adapter-sha256", initial_sha, "--mode", "recovered_policy", "--learning-rate", "1e-6",
                "--gpus-per-node", "2", "--num-train-gpus", "1", "--num-infer-gpus", "1",
                "--batch-size", "16", "--group-size", "4", "--initial-inflight", "32",
                "--max-inflight", "32", "--activation-cpu-offload",
                "--renderer", str(pilot_root / "runtime/watercolour-cancellable-centroid-v1/cancellable_watercolour_renderer.py"),
                "--renderer-python", str(pilot_root / "renderer-env/bin/python"),
            ], env=pilot_env, cwd=pilot_root, check=True, capture_output=True, text=True)
            for path in (pilot_root / "native.toml", pilot_root / "rl-config-receipt.json"):
                path.write_text(path.read_text().replace(
                    "bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740", frozen_sha
                ))
            pilot = subprocess.run([str(pilot_root / "run_single.sh"), "--preflight"],
                                   env=pilot_env, cwd=pilot_root,
                                   text=True, capture_output=True)
            self.assertEqual(pilot.returncode, 0, pilot.stderr + pilot.stdout)
            self.assertIn('"status":"preflight_passed"', pilot.stdout)
            self.assertEqual(
                json.loads((pilot_root / "initial-adapter-receipt.json").read_text())["mode"],
                "recovered_policy",
            )
            native = pilot_root / "native.toml"
            native_original = native.read_text()
            for old, new, message in (
                ("max_steps = 4", "max_steps = 5", "max_steps does not match"),
                ("interval = 1", "interval = 2", "checkpoint interval does not match"),
            ):
                mutated = native_original.replace(old, new, 1)
                self.assertNotEqual(mutated, native_original)
                native.write_text(mutated)
                invalid = subprocess.run([str(pilot_root / "run_single.sh"), "--preflight"],
                                         env=pilot_env, cwd=pilot_root,
                                         text=True, capture_output=True)
                self.assertNotEqual(invalid.returncode, 0)
                self.assertIn(message, invalid.stderr)
                native.write_text(native_original)
            mismatch = (
                f"source {source_root / 'arm_env.sh'} flash; "
                f"BRUSH_ROOT={root / 'painter/rl_helpers'} "
                f"WATERCOLOUR_INIT_ADAPTER={initial} PAINTER_FROZEN_SFT_ADAPTER={frozen} "
                f"PAINTER_RL_CONFIG={repo / 'runs/rl-judge-matched-20261004-flash/native.toml'} "
                f"PAINTER_RL_MANIFEST={repo / 'runs/rl-judge-matched-20261004-flash/anchors-frozen/train-manifest.json'} "
                f"PAINTER_RL_ADAPTER_REVISION=ac4d3eda8e5925e2d23992cfdc1a63620ebde573 "
                f"PAINTER_RL_ADAPTER_PREFIX=text-pairwise-scaled-20260929/step_16 "
                f"PAINTER_RL_ADAPTER_SHA256={initial_sha} PAINTER_RL_FIXED_OPPONENT=1 "
                f"PAINTER_RL_MODE=fresh PAINTER_RL_PREFLIGHT_ONLY=1 "
                f"PAINTER_RL_SKIP_EDITABLE_INSTALL=1 bash {repo / 'runs/rl-judge-matched-20261004-flash/launch_rl.sh'} "
                f"{repo / 'runs/rl-judge-matched-20261004-flash'}"
            )
            result = subprocess.run(["bash", "-c", mismatch], env=env, cwd=source_root,
                                    text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("config receipt mode does not match launcher", result.stderr)

    def _fake_scripts(self, root: Path, *, bad_step: bool = False) -> tuple[Path, Path, Path]:
        prepare_script = root / "prepare-fake.sh"
        prepare_script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        baseline = root / "baseline-fake.sh"
        baseline.write_text("""#!/bin/sh
mkdir -p "$PAINTER_MATCHED_BASELINE_ROOT/validation/step-1-balanced16" "$PAINTER_MATCHED_BASELINE_ROOT/anchors-frozen"
python3 - <<'PY'
import json, os
root = os.environ['PAINTER_MATCHED_BASELINE_ROOT']
identity = {'schema':'matched-fresh-rl16-baseline-v1','mode':'recovered_policy','display_checkpoint':'fresh_rl16_initializer','trained':False,'optimizer_resume':False,'initializer_sha256':'e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c'}
generation = {'adapter_sha256': identity['initializer_sha256'], 'scope': 'balanced16'}
rows = [{'id': f'case-{i}'} for i in range(16)]
with open(root + '/baseline-identity.json', 'w') as stream: json.dump(identity, stream)
with open(root + '/validation/step-1-balanced16/generation-receipt.json', 'w') as stream: json.dump(generation, stream)
for rel in ('validation/step-1-balanced16/cases.json', 'anchors-frozen/validation-matched16.json'):
    with open(root + '/' + rel, 'w') as stream: json.dump({'cases': rows}, stream)
PY
""", encoding="utf-8")
        execute = root / "execute-fake.sh"
        step = 11 if bad_step else 12
        execute.write_text(textwrap.dedent(f"""\
            #!/bin/sh
            mkdir -p "$PAINTER_RUN_ROOT/local-evidence"
            cat > "$PAINTER_RUN_ROOT/terminal-receipt.json" <<'JSON'
            {{"status":"complete","pipeline_exit_code":0,"train_exit_code":0,"evaluation_exit_code":0,"finalizer_exit_code":0,"checkpoint_step":{step},"public_archive_hash_verified":true}}
            JSON
            cat > "$PAINTER_RUN_ROOT/supervisor-state.json" <<'JSON'
            {{"completed_optimizer_steps":{step}}}
            JSON
            adapter="$PAINTER_RUN_ROOT/async-output/$PAINTER_RUN_NAME/artifacts/adapters/step_{step}"
            mkdir -p "$adapter"
            printf 'dummy-step-{step}\\n' > "$adapter/adapter_model.safetensors"
            python3 - "$adapter" "$PAINTER_RUN_ROOT/local-evidence/latest-public-checkpoint.json" <<'PY'
            import hashlib, json, pathlib, sys
            adapter, checkpoint_path = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
            digest = hashlib.sha256((adapter / 'adapter_model.safetensors').read_bytes()).hexdigest()
            upload = {{
                'verified': True, 'private': False, 'commit': 'public-test-commit',
                'path': 'matched-test/step_{step}',
                'files': {{'adapter_model.safetensors': {{'sha256': digest}}}},
            }}
            upload_path = adapter / 'hf-upload.json'
            upload_path.write_text(json.dumps(upload) + '\\n')
            checkpoint_path.write_text(json.dumps({{
                'step': {step}, 'adapter': str(adapter), 'receipt': str(upload_path),
                'adapter_sha256': digest,
                'selection': {{'primary_quality_status': 'primary'}},
            }}) + '\\n')
            PY
        """), encoding="utf-8")
        for path in (prepare_script, baseline, execute):
            path.chmod(0o755)
        return prepare_script, baseline, execute

    def test_generated_campaign_lifecycle_has_one_baseline_and_three_primary_arms(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prep, baseline, execute = self._fake_scripts(root)
            state = root / "state"
            env = os.environ.copy()
            env.update({
                "PAINTER_MATCHED_PREPARE_SCRIPT": str(prep),
                "PAINTER_MATCHED_BASELINE_SCRIPT": str(baseline),
                "PAINTER_MATCHED_EXECUTE_SCRIPT": str(execute),
                "PAINTER_MATCHED_BASELINE_ROOT": str(root / "baseline"),
                "PAINTER_MATCHED_ARM_ROOT_BASE": str(root),
                "PAINTER_CAMPAIGN_STATE_ROOT": str(state),
                "PAINTER_CAMPAIGN_ALLOW_NON_LINUX": "1",
                "PAINTER_CAMPAIGN_SKIP_ARCHIVE": "1",
                "PAINTER_CAMPAIGN_SKIP_PREFLIGHT": "1",
                "PAINTER_CAMPAIGN_PYTHON": "python3",
            })
            result = subprocess.run([str(RUN / "campaign.sh"), "1000", "3.8", "57"], cwd=RUN,
                                    env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            terminal = json.loads((state / "campaign/terminal-receipt.json").read_text())
            self.assertEqual(terminal["status"], "complete")
            rows = [json.loads(line) for line in (state / "campaign/arms.jsonl").read_text().splitlines()]
            self.assertEqual([row["arm"] for row in rows], ["flash", "pro", "luna"])
            self.assertTrue(all(row["primary_success"] for row in rows))

    def test_campaign_rejects_cost_over_cap_and_step12_mismatch_without_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prep, baseline, execute = self._fake_scripts(root, bad_step=True)
            env = os.environ.copy()
            env.update({
                "PAINTER_MATCHED_PREPARE_SCRIPT": str(prep),
                "PAINTER_MATCHED_BASELINE_SCRIPT": str(baseline),
                "PAINTER_MATCHED_EXECUTE_SCRIPT": str(execute),
                "PAINTER_MATCHED_BASELINE_ROOT": str(root / "baseline"),
                "PAINTER_MATCHED_ARM_ROOT_BASE": str(root),
                "PAINTER_CAMPAIGN_STATE_ROOT": str(root / "state"),
                "PAINTER_CAMPAIGN_ALLOW_NON_LINUX": "1",
                "PAINTER_CAMPAIGN_SKIP_ARCHIVE": "1",
                "PAINTER_CAMPAIGN_SKIP_PREFLIGHT": "1",
            })
            too_expensive = subprocess.run([str(RUN / "campaign.sh"), "1000", "3.8", "60"], cwd=RUN,
                                           env=env, text=True, capture_output=True)
            self.assertNotEqual(too_expensive.returncode, 0)
            failed = subprocess.run([str(RUN / "campaign.sh"), "1000", "3.8", "57"], cwd=RUN,
                                    env=env, text=True, capture_output=True)
            self.assertNotEqual(failed.returncode, 0)
            log = (root / "state/campaign/arms.jsonl").read_text().splitlines()
            self.assertEqual(len(log), 1)
            self.assertFalse(json.loads(log[0])["primary_success"])
            status = json.loads((root / "state/campaign/status.json").read_text())
            self.assertEqual(status["status"], "failed")

    def test_campaign_preflight_failure_stops_before_baseline_or_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            arm_base = root / "arms"
            marker = root / "calls.log"
            prepare = root / "prepare.sh"
            prepare.write_text(textwrap.dedent(f"""\
                #!/bin/sh
                set -eu
                mkdir -p "{arm_base}/rl-judge-matched-20261004-flash" \\
                         "{arm_base}/rl-judge-matched-20261004-pro" \\
                         "{arm_base}/rl-judge-matched-20261004-luna"
                for arm in flash pro luna; do
                  cat > "{arm_base}/rl-judge-matched-20261004-$arm/run.sh" <<'EOF'
                #!/bin/sh
                printf 'preflight:%s\\n' "$PAINTER_ARM_ID" >> "{marker}"
                [ "$PAINTER_ARM_ID" != flash ]
                EOF
                  chmod +x "{arm_base}/rl-judge-matched-20261004-$arm/run.sh"
                done
                printf 'prepare\\n' >> "{marker}"
            """), encoding="utf-8")
            arm_env = root / "arm_env.sh"
            arm_env.write_text('export PAINTER_ARM_ID="$1"\n', encoding="utf-8")
            baseline = root / "baseline.sh"
            baseline.write_text(f"printf 'baseline\\n' >> '{marker}'\\nexit 0\\n", encoding="utf-8")
            execute = root / "execute.sh"
            execute.write_text(f"printf 'execute\\n' >> '{marker}'\\nexit 0\\n", encoding="utf-8")
            for path in (prepare, arm_env, baseline, execute):
                path.chmod(0o755)
            state = root / "state"
            env = os.environ.copy()
            env.update({
                "PAINTER_MATCHED_PREPARE_SCRIPT": str(prepare),
                "PAINTER_MATCHED_BASELINE_SCRIPT": str(baseline),
                "PAINTER_MATCHED_EXECUTE_SCRIPT": str(execute),
                "PAINTER_MATCHED_ARM_ENV_SCRIPT": str(arm_env),
                "PAINTER_MATCHED_ARM_ROOT_BASE": str(arm_base),
                "PAINTER_CAMPAIGN_STATE_ROOT": str(state),
                "PAINTER_CAMPAIGN_ALLOW_NON_LINUX": "1",
                "PAINTER_CAMPAIGN_SKIP_ARCHIVE": "1",
                "PAINTER_MATCHED_PREFLIGHT_TIMEOUT_SECONDS": "5",
            })
            result = subprocess.run([str(RUN / "campaign.sh"), "1000", "3.8", "57"], cwd=RUN,
                                    env=env, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(
                marker.read_text().splitlines(), ["prepare", "preflight:flash"],
                (state / "campaign/preflight-flash.log").read_text(),
            )
            status = json.loads((state / "campaign/status.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["reason"], "campaign_failed")
            terminal = json.loads((state / "campaign/terminal-receipt.json").read_text())
            self.assertEqual(terminal["status"], "failed")


if __name__ == "__main__":
    unittest.main()
