"""Offline contract checks for the structured-flow launcher and handoff."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import types
import unittest
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

    def test_supervisor_source_has_arm_bound_and_no_old_checkpoint_cadence(self):
        text = (RUN / "finite_supervisor.py").read_text()
        self.assertIn("base.CHECKPOINT_STEPS = ()", text)
        self.assertIn('for checkpoint in (4, 8)', text)
        self.assertIn("max_updates=8", text)
        self.assertNotIn("max-updates 50", text)

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

    def test_node_builds_missing_proposals_and_evaluator_preserves_flow_hook(self):
        prepare = (RUN / "prepare_node.sh").read_text()
        evaluator = (RUN / "evaluate.sh").read_text()
        self.assertIn('prepare_curriculum.py" build --output "$RUN/curriculum"', prepare)
        self.assertIn('CURRICULUM_BUILD_TMP="$RUN/.curriculum-build.json.$$"', prepare)
        self.assertNotIn('>"$RUN/curriculum/build.json"', prepare)
        self.assertIn('FLOW_BUILD_TMP="$RUN/.flow-supervision-build.json.$$"', prepare)
        self.assertNotIn('>"$SUPERVISION_ROOT/build.json"', prepare)
        self.assertNotIn('flow_demonstrations.py" render', prepare)
        self.assertIn('supervision_phase.py" --data', prepare)
        self.assertIn('PAINTER_PRIME_ROLE=orchestrator', evaluator)
        self.assertIn('PAINTER_STRUCTURED_FLOW_POLICY=1', evaluator)
        self.assertIn('PYTHONPATH="$REPO:$REPO/painter:$RUN"', evaluator)

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
        self.assertNotIn('PAINTER_STRUCTURED_FLOW_POLICY="$PAINTER_STRUCTURED_FLOW_POLICY"', baseline)
        self.assertIn('CUDA_HOME=', baseline)
        self.assertIn('LD_LIBRARY_PATH="$CUDA_HOME/lib:', baseline)
        self.assertIn('"frozen_anchor"', (RUN / "runtime_receipt.py").read_text())
        self.assertIn('PAINTER_BROWSER_ROOT="$RUN/browsers"', evaluate)
        self.assertIn('PLAYWRIGHT_BROWSERS_PATH="$RUN/browsers"', evaluate)
        self.assertIn('export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$PAINTER_BROWSER_ROOT}"', runtime)

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
            self.assertEqual(captured["command"][0], str(eval_bin.resolve()))

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

            def fake_run(command, *, cwd, env, **kwargs):
                captured.update({"command": command, "cwd": cwd, "env": dict(env), "kwargs": kwargs})
                nested_root = Path(command[3])
                native_folder = Path(command[4])
                probe_source = Path(command[5])
                selected_renderer = Path(command[6])
                row_path = Path(command[8])
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

            with mock.patch.dict(os.environ, {"AMBIENT_CANARY_SECRET": "must-not-cross"}, clear=False), \
                 mock.patch.object(helper.sys, "platform", "linux"), \
                 mock.patch.object(helper.shutil, "which", return_value=None), \
                 mock.patch.object(helper.subprocess, "run", side_effect=fake_run):
                result = helper.render_probe(
                    run_root=run_root,
                    renderer=renderer,
                    renderer_python=renderer_python,
                    browsers=browsers,
                    expected_renderer_sha256=renderer_sha,
                    expected_assets_sha256=asset_sha,
                )
            self.assertTrue(result["valid"])
            self.assertEqual(captured["env"]["PAINTER_BROWSER_ROOT"], str(browsers.resolve()))
            self.assertEqual(captured["env"]["PLAYWRIGHT_BROWSERS_PATH"], str(browsers.resolve()))
            self.assertNotIn("AMBIENT_CANARY_SECRET", captured["env"])
            self.assertEqual(captured["env"]["PYTHONPATH"], str(Path(__file__).resolve().parents[3]))
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

    def test_shell_scripts_compile(self):
        scripts = ["runtime_env.sh", "prepare_node.sh", "run.sh", "execute_single.sh", "execute_pipeline.sh", "evaluate.sh", "finalize.sh", "matched_baseline.sh"]
        for name in scripts:
            result = subprocess.run(["bash", "-n", str(RUN / name)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, f"{name}: {result.stderr}")


if __name__ == "__main__":
    unittest.main()
