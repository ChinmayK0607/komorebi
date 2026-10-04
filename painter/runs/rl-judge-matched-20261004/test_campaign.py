"""Offline contract tests for the matched three-judge campaign."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import os
import subprocess
import tempfile
import textwrap
import unittest


RUN = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("matched_prepare", RUN / "prepare_campaign.py")
assert SPEC and SPEC.loader
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


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

    def test_baseline_wrapper_uses_flat_shared_manifest_and_public_archive(self):
        """Exercise the generated baseline shell lifecycle without a node."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "painter"
            run = repo / "runs/rl-judge-matched-20261004"
            run.mkdir(parents=True)
            source_baseline = RUN / "baseline.sh"
            baseline = run / "baseline.sh"
            baseline.write_text(source_baseline.read_text(), encoding="utf-8")
            baseline.chmod(0o755)
            shared = repo / "runs/rl-judge-matched-20261004-shared"
            (shared / "anchors-frozen").mkdir(parents=True)
            for name in ("anchors-all", "vendor", "renderer-env", "browsers", "runtime"):
                (shared / name).mkdir()
            (shared / "preparation-receipt.json").write_text("{}\n")
            cases = [{"id": f"case-{index}"} for index in range(16)]
            (shared / "anchors-frozen/validation-matched16.json").write_text(
                json.dumps({"cases": cases})
            )
            (shared / "policy-adapter-path.txt").write_text(str(root / "initializer") + "\n")
            initializer = root / "initializer"
            initializer.mkdir()
            for name in ("adapter_model.safetensors", "adapter_config.json", "hf-upload.json"):
                (initializer / name).write_text("{}\n")
            (shared / "frozen-sft-adapter-path.txt").write_text(str(initializer) + "\n")
            (repo / "runs/rl-judge-matched-20261004/prepare_arm.sh").write_text("#!/bin/sh\n")
            base = repo / "runs/text-sft512-consistent-20260930"
            base.mkdir(parents=True)
            evaluate_script = (
                "#!/bin/sh\n"
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
            })
            result = subprocess.run([str(baseline)], cwd=run, env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertTrue((root / "baseline/local-evidence/baseline-evidence.tar.gz.upload.json").is_file())

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


if __name__ == "__main__":
    unittest.main()
