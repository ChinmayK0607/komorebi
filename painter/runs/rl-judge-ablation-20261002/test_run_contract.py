"""Offline contracts for the three-arm judge-only RL preparation."""

from __future__ import annotations

import importlib.util
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch

from PIL import Image


RUN = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prep = _load("judge_ablation_prepare", RUN / "prepare_matrix.py")
config_tool = _load(
    "judge_ablation_config",
    RUN.parent / "text-pairwise-rl-20260928" / "prepare_config.py",
)
from painter import text_pairwise_consistent_side_judge_v2_generic as generic
from painter import text_pairwise_consistent_side_judge_v2_model as model


class _Response:
    def __init__(self, model_id: str):
        assessment = {
            "brief_fidelity": "clear",
            "visible_evidence": "The connected form is readable.",
            "weaknesses": ["Edges are soft."],
        }
        content = json.dumps({
            "left_assessment": assessment,
            "right_assessment": assessment,
            "reason": "The panels differ in visible form.",
            "winner": "LEFT",
            "preference": "slight",
        })
        self.body = json.dumps({
            "model": model_id,
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                       "completion_tokens_details": {"reasoning_tokens": 2}},
        }).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body


class AblationContractTests(unittest.TestCase):
    def test_matrix_has_three_exact_models_and_common_shape(self):
        matrix = prep.load_matrix()
        self.assertEqual([arm["id"] for arm in matrix["arms"]], ["flash", "pro", "luna"])
        self.assertEqual(
            [arm["model"] for arm in matrix["arms"]], list(generic.MODELS)
        )
        self.assertEqual(matrix["protocol"]["reasoning"], {"field": "reasoning", "effort": "medium"})
        self.assertEqual(matrix["protocol"]["batch_size"], 16)
        self.assertEqual(matrix["protocol"]["group_size"], 4)
        self.assertEqual(matrix["protocol"]["initial_inflight"], 96)
        self.assertEqual(matrix["protocol"]["max_inflight"], 96)
        self.assertTrue(matrix["protocol"]["activation_cpu_offload"])
        self.assertEqual(
            (matrix["protocol"]["gpus_per_node"], matrix["protocol"]["num_train_gpus"],
             matrix["protocol"]["num_infer_gpus"]),
             (8, 1, 7),
        )
        self.assertEqual(matrix["source"]["training_data"]["train_briefs"], 40)
        self.assertEqual(matrix["source"]["training_data"]["heldout_briefs"], 47)
        self.assertEqual(
            matrix["protocol"]["training_seeds"], {"orchestrator": 42, "inference": 0}
        )
        self.assertEqual(matrix["evaluation"]["initial_scope"], "balanced16")
        self.assertEqual(matrix["evaluation"]["matched_case_count"], 16)
        self.assertEqual(matrix["evaluation"]["seed"], 918210)
        self.assertEqual(matrix["evaluation"]["preferred_primary_checkpoint"], 12)
        self.assertEqual(
            matrix["evaluation"]["preferred_primary_fallback"],
            "latest_verified_public_checkpoint_is_exploratory_censored",
        )

    def test_source_identity_has_flat_base_pipeline_revision_contract(self):
        identity = prep.source_identity(prep.load_matrix())
        self.assertEqual(
            identity["initializer_revision"],
            "ac4d3eda8e5925e2d23992cfdc1a63620ebde573",
        )
        self.assertEqual(
            identity["frozen_sft_model_revision"],
            "adf687e5b321514eda7be6eab050434fdc8e44ff",
        )
        self.assertEqual(
            identity["model_revision"],
            "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        )
        self.assertTrue(identity["protocol"]["activation_cpu_offload"])
        for key in (
            "base_prepare_source_sha256", "text_config_source_sha256",
            "photo_config_source_sha256", "run_from_sft_source_sha256",
            "native_config_writer_source_sha256", "runtime_sitecustomize_source_sha256",
        ):
            with self.subTest(key=key):
                self.assertRegex(identity["files"][key], r"^[0-9a-f]{64}$")

    def test_matrix_rejects_unsupported_two_trainer_initializer_shape(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "matrix.json"
            value = prep.load_matrix()
            value["protocol"]["num_train_gpus"] = 2
            value["protocol"]["num_infer_gpus"] = 6
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "optimizer shape"):
                prep.load_matrix(path)

    def test_native_config_builder_emits_registered_one_plus_seven_shape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "train-manifest.json"
            manifest.write_text(json.dumps({"rows": [
                {"id": f"brief-{i}", "reference_kind": "text", "opponent_sha256": "a" * 64}
                for i in range(40)
            ]}) + "\n")
            output = root / "native.toml"
            config_tool.prepare(
                output=output, model=root / "model", run_root=root, manifest=manifest,
                run_name="rl-judge-ablation-flash-20261002", max_steps=50,
                checkpoint_interval=4, judge_protocol="text-paint-sft-pairwise-consistent-v1",
                mode="recovered_policy", batch_size=16, group_size=4,
                initial_inflight=96, max_inflight=96, gpus_per_node=8,
                num_train_gpus=1, num_infer_gpus=7,
                adapter_revision="ac4d3eda8e5925e2d23992cfdc1a63620ebde573",
                adapter_prefix="text-pairwise-scaled-20260929/step_16",
                adapter_sha256="e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c",
            )
            value = tomllib.loads(output.read_text())
            self.assertEqual(
                {key: value["deployment"][key] for key in (
                    "gpus_per_node", "num_train_gpus", "num_infer_gpus"
                )},
                {"gpus_per_node": 8, "num_train_gpus": 1, "num_infer_gpus": 7},
            )
            self.assertEqual(value["inference"]["vllm"]["tensor_parallel_size"], 1)
            self.assertEqual(value["inference"]["vllm"]["data_parallel_size"], 7)
            self.assertEqual(value["orchestrator"]["concurrency"]["initial_inflight"], 96)
            self.assertEqual(value["orchestrator"]["concurrency"]["max_inflight"], 96)
            self.assertEqual(value["trainer"]["model"]["ac_offloading"], "None")

            opted_output = root / "native-offload.toml"
            opted_receipt = config_tool.prepare(
                output=opted_output, model=root / "model", run_root=root, manifest=manifest,
                run_name="rl-judge-ablation-flash-20261002-offload", max_steps=50,
                checkpoint_interval=4, judge_protocol="text-paint-sft-pairwise-consistent-v1",
                mode="recovered_policy", batch_size=16, group_size=4,
                initial_inflight=96, max_inflight=96, gpus_per_node=8,
                num_train_gpus=1, num_infer_gpus=7,
                adapter_revision="ac4d3eda8e5925e2d23992cfdc1a63620ebde573",
                adapter_prefix="text-pairwise-scaled-20260929/step_16",
                adapter_sha256="e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c",
                activation_cpu_offload=True,
            )
            opted = tomllib.loads(opted_output.read_text())
            self.assertEqual(
                opted["trainer"]["model"]["ac_offloading"],
                {"pin_memory": True, "max_inflight_activations": 1},
            )
            self.assertEqual(opted["deployment"], {"type": "single_node", "gpus_per_node": 8, "num_train_gpus": 1, "num_infer_gpus": 7})
            self.assertEqual(opted["inference"]["vllm"]["tensor_parallel_size"], 1)
            self.assertEqual(opted["inference"]["vllm"]["data_parallel_size"], 7)
            self.assertTrue(opted_receipt["activation_cpu_offload"])
            self.assertEqual(opted_receipt["activation_offloading"], {"pin_memory": True, "max_inflight_activations": 1})

    def test_entrypoints_bind_model_effort_format_and_unique_outputs(self):
        env = (RUN / "arm_env.sh").read_text()
        self.assertIn("PAINTER_JUDGE_EFFORT=medium", env)
        self.assertIn("PAINTER_ACTIVATION_CPU_OFFLOAD=1", env)
        self.assertIn("PAINTER_EVAL_PREFERRED_STEP=12", env)
        self.assertIn("identity-joint-side-schema-v2-generic-v1", env)
        self.assertIn("download_public_adapter.py", env)
        self.assertIn("rl-judge-ablation-shared-20261002/runtime/watercolour", env)
        prep_script = (RUN / "prepare.sh").read_text()
        self.assertIn("PAINTER_ACTIVATION_CPU_OFFLOAD=1", prep_script)
        self.assertIn("--activation-cpu-offload", prep_script)
        self.assertIn('RETRY="${PAINTER_RETRY_RUN:-$REPO/painter/runs/text-rl16-retry-20261001}"', prep_script)
        self.assertIn('bash "$RETRY/prepare_node.sh"', prep_script)
        self.assertIn('shared preparation did not preserve verified adapter path', prep_script)
        self.assertIn('PAINTER_BOOTSTRAP_RUN="$BOOTSTRAP" bash "$BOOTSTRAP/setup_node.sh"', prep_script)
        self.assertLess(
            prep_script.index('PAINTER_BOOTSTRAP_RUN="$BOOTSTRAP" bash "$BOOTSTRAP/setup_node.sh"'),
            prep_script.index('echo "verified frozen SFT path file is missing'),
        )
        base_execute = (RUN.parent / "text-sft512-consistent-20260930" / "execute_pipeline.sh").read_text()
        self.assertIn('${PAINTER_EVAL_PREFERRED_STEP:-}', base_execute)
        self.assertIn('select_preferred_public_checkpoint', base_execute)
        self.assertIn('Preserve the historical latest-verified selection', base_execute)
        base_finalize = (RUN.parent / "text-sft512-consistent-20260930" / "finalize.sh").read_text()
        self.assertIn('primary_quality_status', base_finalize)
        matrix = prep.load_matrix()
        # The existing archive/supervisor contract expects the raw rollouts
        # directory to be named exactly ``rl-rollouts`` inside each isolated
        # run root.  Isolation comes from the flat sibling run roots and
        # distinct run names, not from renaming that directory.
        self.assertEqual({arm["output_root"] for arm in matrix["arms"]}, {"rl-rollouts"})
        self.assertEqual(len({arm["run_name"] for arm in matrix["arms"]}), 3)
        self.assertEqual(len({arm["public_checkpoint_prefix"] for arm in matrix["arms"]}), 3)
        for script in ("preflight.sh", "run.sh", "execute_pipeline.sh", "evaluate.sh", "finalize.sh"):
            text = (RUN / script).read_text()
            with self.subTest(script=script):
                self.assertIn("arm_env.sh", text)

    def test_flat_arm_roots_resolve_base_repo_and_wrappers_delegate(self):
        matrix = prep.load_matrix()
        repo = RUN.parents[2]
        for arm in matrix["arms"]:
            arm_root = (RUN / arm["directory"]).resolve()
            self.assertEqual(arm_root.parent, RUN.parent)
            self.assertEqual((arm_root / "../../..").resolve(), repo)
            wrappers = prep._arm_wrappers(arm["id"])
            self.assertIn(f'"{arm["id"]}"', wrappers["prepare_node.sh"])
            self.assertIn('exec bash "$ROOT/../text-sft512-consistent-20260930/prepare_node.sh"', wrappers["prepare_node.sh"])
            for name in ("run.sh", "evaluate.sh", "finalize.sh"):
                self.assertIn(f'exec bash "$ROOT/{name}" "{arm["id"]}"', wrappers[name])

    def test_fresh_node_bootstrap_is_mocked_before_frozen_path_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "repo"
            run = repo / "painter" / "runs" / "rl-judge-ablation-20261002"
            base = root / "base"
            retry = root / "retry"
            bootstrap = root / "bootstrap"
            fake_bin = root / "bin"
            run.mkdir(parents=True)
            base.mkdir()
            retry.mkdir()
            bootstrap.mkdir()
            fake_bin.mkdir()
            prepare = run / "prepare.sh"
            prepare.write_text((RUN / "prepare.sh").read_text())
            prepare.chmod(0o755)
            (run / "prepare_matrix.py").write_text("#!/usr/bin/env python3\nraise SystemExit(0)\n")
            (base / "prepare_node.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
            (retry / "prepare_node.sh").write_text(
                "#!/usr/bin/env bash\n"
                "set -eu\n"
                "touch \"$PAINTER_TEST_RETRY_MARKER\"\n"
                "exit 99\n"
            )
            for path in (base / "prepare_node.sh", retry / "prepare_node.sh"):
                path.chmod(0o755)
            frozen = repo / "painter" / "runs" / "photo-curriculum-rl-20260922" / "initial-adapter-path.txt"
            bootstrap_marker = root / "bootstrap.called"
            retry_marker = root / "retry.called"
            (bootstrap / "setup_node.sh").write_text(
                "#!/usr/bin/env bash\n"
                "set -eu\n"
                f"mkdir -p {str(frozen.parent)!r}\n"
                f"printf '%s\\n' /tmp/mock-frozen-adapter > {str(frozen)!r}\n"
                f"touch {str(bootstrap_marker)!r}\n"
            )
            (bootstrap / "setup_node.sh").chmod(0o755)
            (fake_bin / "uname").write_text("#!/usr/bin/env bash\nprintf 'Linux\\n'\n")
            (fake_bin / "uname").chmod(0o755)
            env = {
                **os.environ,
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
                "PAINTER_BASE_RUN": str(base),
                "PAINTER_RETRY_RUN": str(retry),
                "PAINTER_BOOTSTRAP_RUN": str(bootstrap),
                "PAINTER_PREPARE_PYTHON": sys.executable,
                "PAINTER_TEST_RETRY_MARKER": str(retry_marker),
            }
            result = subprocess.run([str(prepare), "all"], env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 99)
            self.assertTrue(bootstrap_marker.exists(), result.stderr)
            self.assertTrue(retry_marker.exists(), result.stderr)
            self.assertEqual(frozen.read_text().strip(), "/tmp/mock-frozen-adapter")

    def test_fresh_node_refreshes_pinned_interpreter_for_config_generation(self):
        """A bootstrap-created Prime venv must drive all three configs."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "repo"
            run = repo / "painter" / "runs" / "rl-judge-ablation-20261002"
            base = root / "base"
            retry = root / "retry"
            bootstrap = root / "bootstrap"
            prime = root / "prime"
            fake_bin = root / "bin"
            config_tool = repo / "painter" / "runs" / "text-pairwise-rl-20260928" / "prepare_config.py"
            run.mkdir(parents=True)
            base.mkdir()
            retry.mkdir()
            bootstrap.mkdir()
            fake_bin.mkdir()
            config_tool.parent.mkdir(parents=True)
            prepare = run / "prepare.sh"
            prepare.write_text((RUN / "prepare.sh").read_text())
            prepare.chmod(0o755)
            # The plan and materialization calls are intentionally dependency
            # free. Materialization only needs to create the three flat roots;
            # the config stub below supplies the receipts the shell checks.
            (run / "prepare_matrix.py").write_text(
                "#!/usr/bin/env python3\n"
                "import pathlib, sys\n"
                "run = pathlib.Path(__file__).parent\n"
                "if '--materialize' in sys.argv:\n"
                "    for name in ('rl-judge-ablation-flash-20261002', 'rl-judge-ablation-pro-20261002', 'rl-judge-ablation-luna-20261002'):\n"
                "        (run.parent / name).mkdir(parents=True, exist_ok=True)\n"
                "raise SystemExit(0)\n"
            )
            (run / "matrix.json").write_text(json.dumps({"arms": [
                {"id": "flash", "model": "xiaomi/mimo-v2.6-flash", "run_name": "rl-judge-ablation-flash-20261002", "output_root": "rl-rollouts"},
                {"id": "pro", "model": "xiaomi/mimo-v2.6-pro", "run_name": "rl-judge-ablation-pro-20261002", "output_root": "rl-rollouts"},
                {"id": "luna", "model": "openai/gpt-6-luna", "run_name": "rl-judge-ablation-luna-20261002", "output_root": "rl-rollouts"},
            ]}))
            (base / "prepare_node.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
            (base / "prepare_node.sh").chmod(0o644)
            (retry / "prepare_node.sh").write_text(
                "#!/usr/bin/env bash\n"
                "set -eu\n"
                "mkdir -p \"$(dirname \"$PAINTER_SFT_ADAPTER_PATH_FILE\")\"\n"
                "printf '%s\\n' /tmp/mock-policy-adapter > \"$PAINTER_SFT_ADAPTER_PATH_FILE\"\n"
                "printf '%s\\n' /tmp/mock-frozen-adapter > \"$PAINTER_FROZEN_SFT_ADAPTER_PATH_FILE\"\n"
                "touch \"$PAINTER_RUN_ROOT/preparation-receipt.json\"\n"
            )
            (retry / "prepare_node.sh").chmod(0o644)
            frozen = repo / "painter" / "runs" / "photo-curriculum-rl-20260922" / "initial-adapter-path.txt"
            prime_log = root / "prime-invocations.log"
            config_log = root / "config-invocations.log"
            # Bootstrap creates the interpreter only after the initial plan
            # check. The wrapper records every later invocation and delegates
            # to the test process's real Python for the stubs.
            bootstrap_python = prime / ".venv" / "bin" / "python"
            bootstrap_python.parent.mkdir(parents=True)
            bootstrap_python.write_text(
                "#!/usr/bin/env bash\n"
                f"printf '%s\\n' \"$*\" >> {str(prime_log)!r}\n"
                f"exec {str(Path(sys.executable))!r} \"$@\"\n"
            )
            # Hide the interpreter until bootstrap is called.
            bootstrap_python.unlink()
            (bootstrap / "setup_node.sh").write_text(
                "#!/usr/bin/env bash\n"
                "set -eu\n"
                f"mkdir -p {str(frozen.parent)!r} {str(bootstrap_python.parent)!r}\n"
                f"printf '%s\\n' /tmp/mock-frozen-adapter > {str(frozen)!r}\n"
                f"cat > {str(bootstrap_python)!r} <<'PYTHON_WRAPPER'\n"
                "#!/usr/bin/env bash\n"
                f"printf '%s\\n' \"$*\" >> {str(prime_log)!r}\n"
                f"exec {str(Path(sys.executable))!r} \"$@\"\n"
                "PYTHON_WRAPPER\n"
                f"chmod 755 {str(bootstrap_python)!r}\n"
            )
            (bootstrap / "setup_node.sh").chmod(0o755)
            # This stub is run through the bootstrap-created interpreter. It
            # writes exactly the receipt fields prepare.sh validates.
            config_tool.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "from pathlib import Path\n"
                f"Path({str(config_log)!r}).open('a').write('CONFIG\\n')\n"
                "args = dict(zip(sys.argv[1::2], sys.argv[2::2]))\n"
                "out = Path(args['--output'])\n"
                "out.parent.mkdir(parents=True, exist_ok=True)\n"
                "out.write_text('[deployment]\\ngpus_per_node = 8\\nnum_train_gpus = 1\\nnum_infer_gpus = 7\\n')\n"
                "receipt = {'run_name': args['--run-name'], 'model_revision': '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0', 'adapter_revision': 'ac4d3eda8e5925e2d23992cfdc1a63620ebde573', 'adapter_prefix': 'text-pairwise-scaled-20260929/step_16', 'deployment': {'gpus_per_node': 8, 'num_train_gpus': 1, 'num_infer_gpus': 7}}\n"
                "out.with_name('rl-config-receipt.json').write_text(json.dumps(receipt))\n"
            )
            (config_tool.parent / "__init__.py").write_text("")
            (fake_bin / "uname").write_text("#!/usr/bin/env bash\nprintf 'Linux\\n'\n")
            (fake_bin / "uname").chmod(0o755)
            env = {
                **os.environ,
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
                "PRIME_ROOT": str(prime),
                "PAINTER_BASE_RUN": str(base),
                "PAINTER_RETRY_RUN": str(retry),
                "PAINTER_BOOTSTRAP_RUN": str(bootstrap),
                "MODEL_DIR": "/tmp/model",
                "PAINTER_TEST_CONFIG_LOG": str(config_log),
            }
            env.pop("PAINTER_PREPARE_PYTHON", None)
            result = subprocess.run([str(prepare), "all"], env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(config_log.read_text().splitlines()), 3)
            invocations = prime_log.read_text().splitlines()
            self.assertEqual(sum("prepare_config.py" in line for line in invocations), 3)
            self.assertTrue(all("prepare_matrix.py" not in line or "--plan" not in line for line in invocations), invocations)

    def test_clean_checkout_modes_match_bash_invocation_contract(self):
        repo = RUN.parents[2]
        paths = [
            "painter/runs/text-sft512-consistent-20260930/prepare_node.sh",
            "painter/runs/text-sft512-consistent-20260930/run.sh",
            "painter/runs/text-sft512-consistent-20260930/evaluate.sh",
            "painter/runs/text-sft512-consistent-20260930/finalize.sh",
            "painter/runs/text-pairwise-rl-20260928/setup_node.sh",
        ]
        output = subprocess.check_output(["git", "ls-files", "-s", *paths], cwd=repo, text=True)
        modes = {line.split("\t", 1)[1]: line.split()[0] for line in output.splitlines()}
        for path in paths:
            self.assertEqual(modes[path], "100644", path)
        prepare_text = (RUN / "prepare.sh").read_text()
        self.assertIn('[[ -f "$BASE/prepare_node.sh" ]]', prepare_text)
        self.assertIn('[[ -f "$RETRY/prepare_node.sh" ]]', prepare_text)
        self.assertIn('[[ -f "$BOOTSTRAP/setup_node.sh" ]]', prepare_text)

    def test_materialized_wrapper_executes_fixed_arm_route(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            matrix_root = root / "rl-judge-ablation-20261002"
            arm_root = root / "rl-judge-ablation-flash-20261002"
            base_root = root / "text-sft512-consistent-20260930"
            matrix_root.mkdir()
            arm_root.mkdir()
            base_root.mkdir()
            (matrix_root / "arm_env.sh").write_text(
                "#!/usr/bin/env bash\nexport FIXED_ARM=\"$1\"\n",
            )
            for name in ("run.sh", "evaluate.sh", "finalize.sh"):
                (matrix_root / name).write_text(
                    "#!/usr/bin/env bash\n"
                    "[[ \"$1\" == flash ]] || exit 11\n"
                    f"touch \"{root / (name + '.called')}\"\n",
                )
                (matrix_root / name).chmod(0o755)
            (base_root / "prepare_node.sh").write_text(
                "#!/usr/bin/env bash\n[[ \"$FIXED_ARM\" == flash ]] || exit 12\n"
                f"touch \"{root / 'prepare_node.called'}\"\n",
            )
            (base_root / "prepare_node.sh").chmod(0o755)
            for name, text in prep._arm_wrappers("flash").items():
                path = arm_root / name
                path.write_text(text)
                path.chmod(0o755)
            env = {**os.environ, "BASH_ENV": "/dev/null"}
            subprocess.run([str(arm_root / "prepare_node.sh")], check=True, env=env)
            subprocess.run([str(arm_root / "run.sh")], check=True, env=env)
            self.assertTrue((root / "prepare_node.called").exists())
            self.assertTrue((root / "run.sh.called").exists())

    def test_pipeline_scope_and_finalize_share_the_matched_panel(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "base"
            base.mkdir()
            for name in ("execute_pipeline.sh", "evaluate.sh", "finalize.sh"):
                script = base / name
                script.write_text(
                    "#!/usr/bin/env bash\n"
                    "printf '%s\\n' \"$*\" \"${PAINTER_EVAL_SCOPE:-}\" "
                    "\"${PAINTER_EVAL_MANIFEST:-}\" \"${PAINTER_EVAL_EXPECTED_COUNT:-}\" > \"$CAPTURE\"\n"
                )
                script.chmod(0o755)
            env = {**os.environ, "PAINTER_BASE_RUN": str(base)}
            env.pop("PAINTER_EVAL_SCOPE", None)
            capture = root / "execute.txt"
            env["CAPTURE"] = str(capture)
            subprocess.run(
                [str(RUN / "execute_pipeline.sh"), "flash", "100", "9.2", "20"],
                check=True, env=env,
            )
            self.assertEqual(capture.read_text().splitlines()[:2], ["100 9.2 20", "balanced16"])

            capture = root / "evaluate.txt"
            env["CAPTURE"] = str(capture)
            env["PAINTER_EVAL_SCOPE"] = "balanced16"
            subprocess.run(
                [str(RUN / "evaluate.sh"), "flash", "12", "all", "--preflight"],
                check=True, env=env,
            )
            evaluate_rows = capture.read_text().splitlines()
            self.assertEqual(evaluate_rows[0], "12 balanced16 --preflight")
            self.assertEqual(evaluate_rows[3], "16")
            self.assertTrue(evaluate_rows[2].endswith("rl-judge-ablation-flash-20261002/evaluation/matched16-manifest.json"))

            capture = root / "finalize.txt"
            env["CAPTURE"] = str(capture)
            subprocess.run(
                [str(RUN / "finalize.sh"), "flash", "12"],
                check=True, env=env,
            )
            self.assertEqual(capture.read_text().splitlines()[:2], ["12 ", "balanced16"])

    def test_campaign_runs_arms_in_order_and_releases_only_after_preservation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            state = root / "state"
            prepare = root / "prepare.sh"
            execute = root / "execute.sh"
            prepare.write_text(
                "#!/usr/bin/env bash\n"
                "[[ \"$1\" == all ]] || exit 20\n"
                "touch \"$PAINTER_CAMPAIGN_STATE_ROOT/prepared\"\n",
            )
            execute.write_text(
                "#!/usr/bin/env bash\n"
                "arm=\"$1\"\n"
                "printf '%s\\n' \"$arm\" >> \"$PAINTER_CAMPAIGN_STATE_ROOT/order\"\n"
                "name=\"rl-judge-ablation-${arm}-20261002\"\n"
                "run=\"$PAINTER_CAMPAIGN_STATE_ROOT/../$name\"\n"
                "mkdir -p \"$run/local-evidence\"\n"
                "printf '%s\\n' '{\"status\": \"complete\", \"public_archive_hash_verified\": true}' > \"$run/terminal-receipt.json\"\n"
                "printf '%s\\n' evidence > \"$run/local-evidence/${name}-evidence.tar.gz\"\n"
                "printf '%s\\n' '{\"public_hash_verified\": true, \"private\": false}' > \"$run/local-evidence/${name}-evidence.tar.gz.upload.json\"\n",
            )
            prepare.chmod(0o755)
            execute.chmod(0o755)
            env = {
                **os.environ,
                "PAINTER_CAMPAIGN_ALLOW_NON_LINUX": "1",
                "PAINTER_CAMPAIGN_SKIP_LOCK": "1",
                "PAINTER_CAMPAIGN_STATE_ROOT": str(state),
                "PAINTER_ABLATION_PREPARE_SCRIPT": str(prepare),
                "PAINTER_ABLATION_EXECUTE_SCRIPT": str(execute),
            }
            subprocess.run(
                [str(RUN / "campaign.sh"), "100000", "9.2", "59.4"],
                check=True, env=env,
            )
            self.assertEqual((state / "order").read_text().splitlines(), ["flash", "pro", "luna"])
            self.assertTrue((state / "campaign-release.ready").is_file())
            status = json.loads((state / "campaign/status.json").read_text())
            self.assertEqual(status["status"], "complete")
            self.assertAlmostEqual(status["per_arm_gpu_cap_usd"], 19.8, places=6)

    def test_campaign_stops_after_zero_update_training_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            state = root / "state"
            prepare = root / "prepare.sh"
            execute = root / "execute.sh"
            prepare.write_text("#!/usr/bin/env bash\nexit 0\n")
            execute.write_text(
                "#!/usr/bin/env bash\n"
                "arm=\"$1\"\n"
                "printf '%s\\n' \"$arm\" >> \"$PAINTER_CAMPAIGN_STATE_ROOT/order\"\n"
                "name=\"rl-judge-ablation-${arm}-20261002\"\n"
                "run=\"$PAINTER_CAMPAIGN_STATE_ROOT/../$name\"\n"
                "mkdir -p \"$run/local-evidence\"\n"
                "printf '%s\\n' '{\"status\": \"failed\", \"pipeline_exit_code\": 1, \"public_archive_hash_verified\": true}' > \"$run/terminal-receipt.json\"\n"
                "printf '%s\\n' '{\"schema\":\"finite-text-paint-supervisor-v1\",\"status\":\"failed\",\"completed_optimizer_steps\":null}' > \"$run/supervisor-state.json\"\n"
                "printf '%s\\n' evidence > \"$run/local-evidence/${name}-evidence.tar.gz\"\n"
                "printf '%s\\n' '{\"public_hash_verified\": true, \"private\": false}' > \"$run/local-evidence/${name}-evidence.tar.gz.upload.json\"\n"
                "exit 1\n",
            )
            prepare.chmod(0o755)
            execute.chmod(0o755)
            env = {
                **os.environ,
                "PAINTER_CAMPAIGN_ALLOW_NON_LINUX": "1",
                "PAINTER_CAMPAIGN_SKIP_LOCK": "1",
                "PAINTER_CAMPAIGN_STATE_ROOT": str(state),
                "PAINTER_ABLATION_PREPARE_SCRIPT": str(prepare),
                "PAINTER_ABLATION_EXECUTE_SCRIPT": str(execute),
            }
            result = subprocess.run(
                [str(RUN / "campaign.sh"), "100000", "9.2", "59.4"],
                check=False, env=env,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((state / "order").read_text().splitlines(), ["flash"])
            status = json.loads((state / "campaign/status.json").read_text())
            self.assertEqual(status["reason"], "zero_optimizer_updates_training_or_preflight_fatal_after_flash")
            arm_row = json.loads((state / "campaign/arms.jsonl").read_text().strip())
            self.assertTrue(arm_row["zero_optimizer_updates_training_fatal"])
            self.assertEqual(arm_row["verified_optimizer_updates"], 0)
            self.assertFalse((state / "campaign-release.ready").exists())

    def test_safe_archive_keeps_rl_rollouts_evidence(self):
        safe_archive = _load(
            "judge_ablation_safe_archive",
            RUN.parent / "text-sft512-consistent-20260930" / "safe_archive.py",
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            evidence = root / "rl-rollouts" / "group-0001.json"
            evidence.parent.mkdir(parents=True)
            evidence.write_text('{"ok": true}\n')
            selected = safe_archive.select_files(root)
            self.assertIn("rl-rollouts/group-0001.json", selected)

    def test_public_checkpoint_selector_requires_verified_exact_step(self):
        safe_archive = _load(
            "judge_ablation_safe_archive_selection",
            RUN.parent / "text-sft512-consistent-20260930" / "safe_archive.py",
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            adapter_root = root / "async-output" / "case" / "artifacts" / "adapters"
            for step in (12, 16):
                adapter = adapter_root / f"step_{step}"
                adapter.mkdir(parents=True)
                weight = adapter / "adapter_model.safetensors"
                weight.write_bytes(f"adapter-{step}".encode())
                digest = hashlib.sha256(weight.read_bytes()).hexdigest()
                (adapter / "hf-upload.json").write_text(json.dumps({
                    "verified": True,
                    "private": False,
                    "commit": f"commit-{step}",
                    "path": f"case/step_{step}",
                    "files": {"adapter_model.safetensors": {"sha256": digest}},
                }))
            selected = safe_archive.select_public_checkpoint(root, 12, run_name="case")
            self.assertEqual(selected["step"], 12)
            latest = safe_archive.select_latest_public_checkpoint(root, run_name="case")
            self.assertEqual(latest["step"], 16)
            fallback, selection = safe_archive.select_preferred_public_checkpoint(
                root, 12, run_name="case"
            )
            self.assertEqual(fallback["step"], 12)
            self.assertEqual(selection["primary_quality_status"], "primary")
            (adapter_root / "step_12" / "hf-upload.json").write_text(
                json.dumps({"verified": False, "private": False})
            )
            self.assertIsNone(safe_archive.select_public_checkpoint(root, 12, run_name="case"))
            fallback, selection = safe_archive.select_preferred_public_checkpoint(
                root, 12, run_name="case"
            )
            self.assertEqual(fallback["step"], 16)
            self.assertEqual(selection["primary_quality_status"], "exploratory_censored")

    def test_generic_transport_sends_reasoning_and_omits_temperature(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            left, right = root / "left.png", root / "right.png"
            Image.new("RGB", (12, 8), "#c54").save(left)
            Image.new("RGB", (8, 12), "#45c").save(right)
            calls = []

            def open_url(request, timeout):
                calls.append(json.loads(request.data))
                return _Response(generic.MODELS[0])

            with patch.object(model.urllib.request, "urlopen", side_effect=open_url):
                value = model._view(
                    key="secret", model=generic.MODELS[0], brief="Paint a form.",
                    a=left, b=right, a_id=model.image_id(left), b_id=model.image_id(right),
                    reasoning=generic.REASONING,
                )
            self.assertIn("assessment", value)
            self.assertEqual(calls[0]["reasoning"], {"effort": "medium"})
            self.assertNotIn("temperature", calls[0])

    def test_historical_default_transport_still_sends_temperature(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            left, right = root / "left.png", root / "right.png"
            Image.new("RGB", (12, 8), "#c54").save(left)
            Image.new("RGB", (8, 12), "#45c").save(right)
            calls = []

            def open_url(request, timeout):
                calls.append(json.loads(request.data))
                return _Response("xiaomi/mimo-v2.6-pro")

            with patch.object(model.urllib.request, "urlopen", side_effect=open_url):
                model._view(
                    key="secret", model="xiaomi/mimo-v2.6-pro", brief="Paint a form.",
                    a=left, b=right, a_id=model.image_id(left), b_id=model.image_id(right),
                )
            self.assertEqual(calls[0]["temperature"], 0)
            self.assertNotIn("reasoning", calls[0])

    def test_matched16_is_predeclared_and_reproducible(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "validation.json"
            cases = [{"id": f"t600-{case_id}", "family": "simple"} for case_id in (
                512, 516, 542, 543, 425, 433, 439, 445, 309, 366, 375, 377, 413, 466, 478, 560,
            )]
            cases.extend({"id": f"t600-extra-{i}"} for i in range(31))
            path.write_text(json.dumps({"cases": cases}))
            value = prep.matched_manifest(path)
            self.assertEqual(value["selection"]["count"], 16)
            self.assertEqual([row["id"] for row in value["cases"]], list(prep.MATCHED_CASE_IDS))


if __name__ == "__main__":
    unittest.main()
