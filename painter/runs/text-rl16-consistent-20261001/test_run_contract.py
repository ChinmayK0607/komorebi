"""Static and dependency-light contracts for the isolated RL16 continuation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import tomllib
import unittest


RUN = Path(__file__).parent
BASE = RUN.parent / "text-sft512-consistent-20260930"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class RL16RunContractTests(unittest.TestCase):
    def test_policy_and_frozen_initializers_are_distinct_and_explicit(self):
        identity = json.loads((RUN / "source-identity.json").read_text())
        policy = identity["policy_initializer"]
        frozen = identity["frozen_sft_initializer"]
        self.assertEqual(policy["revision"], "ac4d3eda8e5925e2d23992cfdc1a63620ebde573")
        self.assertEqual(policy["adapter_sha256"], "e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c")
        self.assertEqual(frozen["revision"], "adf687e5b321514eda7be6eab050434fdc8e44ff")
        self.assertEqual(frozen["adapter_sha256"], "bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740")
        self.assertNotEqual(policy["adapter_sha256"], frozen["adapter_sha256"])
        launch = (RUN / "run.sh").read_text()
        self.assertIn("PAINTER_POLICY_MODE=recovered_policy", launch)
        self.assertIn("PAINTER_FROZEN_SFT_ADAPTER_PATH_FILE", launch)

    def test_terminal_credit_and_syntax_flags_are_enabled_in_train_and_eval(self):
        run = (BASE / "run.sh").read_text()
        evaluate = (BASE / "evaluate.sh").read_text()
        for text in (run, evaluate):
            self.assertIn("PAINTER_REQUIRE_VALID_FINAL_ACTION=1", text)
            self.assertIn("PAINTER_JS_SYNTAX_PREFLIGHT=1", text)
        wrapper = load("rl16_run_from_sft", Path(__file__).parents[1] / "photo-curriculum-rl-20260922" / "run_from_sft.py")
        forwarded = wrapper._orchestrator_env({})
        # The values need to be forwarded when Prime creates its separate
        # orchestrator process; a parent-only export would be ineffective.
        forwarded_names = set(wrapper._orchestrator_env.__code__.co_consts[1])
        self.assertIn("PAINTER_REQUIRE_VALID_FINAL_ACTION", forwarded_names)
        self.assertIn("PAINTER_JS_SYNTAX_PREFLIGHT", forwarded_names)
        self.assertNotIn("HF_TOKEN", forwarded)

    def test_native_config_binds_explicit_renderer_paths(self):
        module = load("rl16_prepare_config", BASE.parent / "text-pairwise-rl-20260928" / "prepare_config.py")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / "train-manifest.json"
            manifest.write_text(json.dumps({"rows": [
                {"id": f"brief-{i}", "reference_kind": "text", "opponent_sha256": "a" * 64}
                for i in range(40)
            ]}) + "\n")
            output = root / "native.toml"
            module.prepare(
                output=output, model=root / "model", run_root=root, manifest=manifest,
                run_name="text-rl16-consistent-20261001", max_steps=50,
                checkpoint_interval=5, mode="recovered_policy",
                adapter_revision="ac4d3eda8e5925e2d23992cfdc1a63620ebde573",
                adapter_prefix="text-pairwise-scaled-20260929/step_16",
                adapter_sha256="e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c",
                judge_protocol="text-paint-sft-pairwise-consistent-v1",
                gpus_per_node=2, num_train_gpus=1, num_infer_gpus=1,
                batch_size=16, group_size=8, initial_inflight=16, max_inflight=16,
                renderer="/run/patched/renderer.py", renderer_python="/run/renderer-env/bin/python",
            )
            raw = tomllib.loads(output.read_text())
            taskset = raw["orchestrator"]["train"]["source"][0]["env"]["taskset"]
            self.assertEqual(taskset["renderer"], "/run/patched/renderer.py")
            self.assertEqual(taskset["renderer_python"], "/run/renderer-env/bin/python")

    def test_prepare_builds_one_versioned_patch_and_uses_real_downloader(self):
        text = (RUN / "prepare_node.sh").read_text()
        self.assertIn("patch_brush_centroid.py", text)
        self.assertIn("download_public_adapter.py", text)
        self.assertIn("PAINTER_BOOTSTRAP_RUN:-$REPO/painter/runs/text-pairwise-rl-20260928", text)
        self.assertIn("PAINTER_RENDERER_PATH", text)
        self.assertIn("PATCH-MANIFEST.json", text)

    def test_finite_supervisor_keeps_requested_checkpoint_stop_rule(self):
        text = (BASE / "finite_supervisor.py").read_text()
        self.assertIn("CHECKPOINT_STEPS = (15, 20, 25, 30, 35, 40, 45, 50)", text)
        self.assertIn('"decision": "continue" if delta > 0.0 else "stop"', text)
        self.assertIn("max_updates: int = 50", text)

    def test_completion_relay_accepts_new_terminal_schema_and_run_name(self):
        text = (BASE / "completion_relay.py").read_text()
        self.assertIn("text-paint-consistent-terminal-v2", text)
        self.assertIn("PAINTER_RUN_NAME", text)
        self.assertIn("PAINTER_RELAY_EVIDENCE_DIR", text)
        self.assertIn("text-rl16-consistent-20261001", (RUN / "completion_relay.py").read_text())


if __name__ == "__main__":
    unittest.main()
