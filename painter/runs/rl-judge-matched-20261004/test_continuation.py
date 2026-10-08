"""Bounded offline tests for the explicit matched-campaign continuation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import os
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import unittest
import unittest.mock


RUN = Path(__file__).resolve().parent
OPERATION_SPEC = importlib.util.spec_from_file_location("matched_operation_continuation", RUN / "operate_campaign.py")
assert OPERATION_SPEC and OPERATION_SPEC.loader
operation = importlib.util.module_from_spec(OPERATION_SPEC)
OPERATION_SPEC.loader.exec_module(operation)
RESTORE_SPEC = importlib.util.spec_from_file_location(
    "matched_restore_continuation", RUN / "restore_continuation.py"
)
assert RESTORE_SPEC and RESTORE_SPEC.loader
restore = importlib.util.module_from_spec(RESTORE_SPEC)
RESTORE_SPEC.loader.exec_module(restore)


class ContinuationTests(unittest.TestCase):
    def test_recovery_archive_rejects_traversal_and_links(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "unsafe.tar.gz"
            with tarfile.open(archive, "w:gz") as stream:
                traversal = tarfile.TarInfo("../escape.txt")
                traversal.size = 1
                stream.addfile(traversal, __import__("io").BytesIO(b"x"))
            with self.assertRaises(restore.RecoveryError):
                restore._extract_safe(archive, root / "out")
            archive.unlink()
            with tarfile.open(archive, "w:gz") as stream:
                link = tarfile.TarInfo("safe-link")
                link.type = tarfile.SYMTYPE
                link.linkname = "outside"
                stream.addfile(link)
            with self.assertRaises(restore.RecoveryError):
                restore._extract_safe(archive, root / "out")

    def test_public_hf_cache_symlinks_are_copied_after_metadata_and_hash_checks(self):
        class Lfs:
            sha256 = "fixture-sha"

        class Sibling:
            rfilename = "fixture/step_12/adapter_model.safetensors"
            lfs = Lfs()

        class Info:
            sha = "fixture-revision"
            private = False
            siblings = [Sibling()]

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "snapshot/fixture/step_12"
            blobs = root / "blobs"
            source.mkdir(parents=True)
            blobs.mkdir()
            weight = b"cached-weight"
            import hashlib
            digest = hashlib.sha256(weight).hexdigest()
            (blobs / "weight").write_bytes(weight)
            (blobs / "config").write_text("{}\n")
            (source / "adapter_model.safetensors").symlink_to(blobs / "weight")
            (source / "adapter_config.json").symlink_to(blobs / "config")

            fake_hub = type("FakeHub", (), {
                "HfApi": lambda token=False: type("Api", (), {
                    "model_info": lambda self, *args, **kwargs: Info(),
                })(),
                "snapshot_download": lambda *args, **kwargs: str(root / "snapshot"),
            })
            old = (restore.MODEL_REPO, restore.MODEL_REVISION, restore.MODEL_PREFIX, dict(restore.MODEL_WEIGHT))
            restore.MODEL_REPO = "fixture/repo"
            restore.MODEL_REVISION = "fixture-revision"
            restore.MODEL_PREFIX = "fixture/step_12"
            restore.MODEL_WEIGHT = {"bytes": len(weight), "sha256": digest}
            Sibling.rfilename = "fixture/step_12/adapter_model.safetensors"
            Lfs.sha256 = digest
            try:
                with unittest.mock.patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
                    target = root / "destination"
                    restore._download_adapter(target)
            finally:
                restore.MODEL_REPO, restore.MODEL_REVISION, restore.MODEL_PREFIX, restore.MODEL_WEIGHT = old
            self.assertFalse((target / "adapter_model.safetensors").is_symlink())
            self.assertEqual((target / "adapter_model.safetensors").read_bytes(), weight)

    def test_remote_command_places_explicit_mode_before_positional_contract(self):
        operation.set_continuation_mode(True)
        command = operation._campaign_job_command(
            source_commit="a" * 40, rental_start_epoch=1000.0,
            price_usd_per_hour=3.8, cap_usd=43.5,
        )
        self.assertIn("campaign.sh --continuation 1000.0 3.8 43.5", command)
        self.assertNotIn("campaign.sh 1000.0 3.8 43.5 --continuation", command)
        operation.set_continuation_mode(False)

    def test_campaign_continuation_restores_flash_and_launches_only_pro_luna(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            state = root / "state"
            arms = root / "arms"
            shared = root / "shared"
            marker = root / "calls.log"
            token = root / "hf-token"
            token.write_text("fixture-token\n")
            token.chmod(0o600)
            (shared / "anchors-frozen").mkdir(parents=True)
            cases = [{"id": f"case-{index}"} for index in range(16)]
            (shared / "anchors-frozen/validation-matched16.json").write_text(json.dumps({"cases": cases}))

            prepare = root / "prepare.sh"
            prepare.write_text(textwrap.dedent(f"""\
                #!/bin/sh
                set -eu
                mkdir -p "{shared}/anchors-frozen" \\
                         "{arms}/rl-judge-matched-20261004-flash" \\
                         "{arms}/rl-judge-matched-20261004-pro" \\
                         "{arms}/rl-judge-matched-20261004-luna"
                cat > "{arms}/rl-judge-matched-20261004-flash/safe_archive.py" <<'PY'
                def select_preferred_public_checkpoint(run, preferred_step, *, run_name):
                    adapter = run / 'async-output' / run_name / 'artifacts/adapters/step_12'
                    receipt = adapter / 'hf-upload.json'
                    selected = {{'step': 12, 'adapter': str(adapter), 'receipt': str(receipt), 'adapter_sha256': 'fixture'}}
                    return selected, {{'primary_quality_status': 'primary', 'selected_step': 12}}
                PY
                printf 'prepare\\n' >> "{marker}"
            """))
            baseline = root / "baseline.sh"
            baseline.write_text(f"#!/bin/sh\nprintf 'baseline-called\\n' >> '{marker}'\nexit 91\n")
            restore = root / "restore.sh"
            restore.write_text(textwrap.dedent(f"""\
                #!/bin/sh
                set -eu
                baseline=flash=shared=state=
                while [ "$#" -gt 0 ]; do
                  case "$1" in
                    --baseline-root) baseline="$2"; shift 2;;
                    --flash-root) flash="$2"; shift 2;;
                    --shared-root) shared="$2"; shift 2;;
                    --state-root) state="$2"; shift 2;;
                    *) exit 2;;
                  esac
                done
                mkdir -p "$baseline/validation/step-1-balanced16" "$flash/async-output/rl-judge-matched-20261004-flash/artifacts/adapters/step_12" "$state/continuation-recovery"
                cat > "$baseline/baseline-identity.json" <<'JSON'
                {{"mode":"recovered_policy","display_checkpoint":"fresh_rl16_initializer","trained":false,"optimizer_resume":false,"initializer_sha256":"e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c","scope":"balanced16","case_count":16,"seed":918210}}
                JSON
                python3 - "$baseline" "$shared" <<'PY'
                import json, sys
                from pathlib import Path
                baseline, shared = map(Path, sys.argv[1:])
                cases = json.loads((shared / 'anchors-frozen/validation-matched16.json').read_text())['cases']
                out = baseline / 'validation/step-1-balanced16'
                (out / 'generation-receipt.json').write_text(json.dumps({{'scope':'balanced16','adapter_sha256':'e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c','checkpoint_provenance':'public_download','matched_control':'sft-step512','max_turns':3,'max_tokens':8192}}))
                (out / 'cases.json').write_text(json.dumps({{'cases': cases}}))
                (baseline / 'baseline-recovery-receipt.json').write_text(json.dumps({{'regenerated':False,'archive':{{'sha256':'6967e1304b8b51aeda5b73bb00b05365edf72e44ae92baffaabaf93327158d41'}}}}))
                PY
                cat > "$flash/supervisor-state.json" <<'JSON'
                {{"status":"stopped","completed_optimizer_steps":12,"last_optimizer_step":12,"max_updates":12}}
                JSON
                cat > "$flash/flash-recovery-receipt.json" <<'JSON'
                {{"training_invoked":false}}
                JSON
                cat > "$flash/async-output/rl-judge-matched-20261004-flash/artifacts/adapters/step_12/hf-upload.json" <<'JSON'
                {{"verified":true,"private":false,"files":{{"adapter_model.safetensors":{{"sha256":"fixture"}}}}}}
                JSON
                printf 'evaluate-called\\n' > "$flash/evaluate.sh"
                printf 'finalize-called\\n' > "$flash/finalize.sh"
                chmod +x "$flash/evaluate.sh" "$flash/finalize.sh"
                printf 'restore\\n' >> "{marker}"
            """))
            # Replace the marker-only Flash wrappers with strict contracts.
            restore.write_text(restore.read_text() + textwrap.dedent(f"""\
                cat > "$flash/evaluate.sh" <<'EOF'
                #!/bin/sh
                [ "$1" = 12 ] && [ "$2" = balanced16 ] || exit 17
                printf 'flash-evaluate:%s:%s\\n' "$1" "$2" >> "{marker}"
                mkdir -p "$PAINTER_RUN_ROOT/validation/step-12-balanced16"
                printf '{{"scope":"balanced16"}}\\n' > "$PAINTER_RUN_ROOT/validation/step-12-balanced16/generation-receipt.json"
                EOF
                cat > "$flash/finalize.sh" <<'EOF'
                [ "$1" = 12 ] || exit 18
                printf 'flash-finalize:%s\\n' "$1" >> "{marker}"
                EOF
                chmod +x "$flash/evaluate.sh" "$flash/finalize.sh"
            """))
            arm_env = root / "arm_env.sh"
            arm_env.write_text(textwrap.dedent(f"""\
                #!/bin/sh
                export PAINTER_ARM_ID="$1"
                export PAINTER_RUN_ROOT="{arms}/rl-judge-matched-20261004-$1"
                export PAINTER_RUN_NAME="rl-judge-matched-20261004-$1"
            """))
            execute = root / "execute.sh"
            execute.write_text(textwrap.dedent(f"""\
                #!/bin/sh
                set -eu
                printf 'execute:%s\\n' "$PAINTER_ARM_ID" >> "{marker}"
                [ "$PAINTER_ARM_ID" != flash ] || exit 19
                mkdir -p "$PAINTER_RUN_ROOT/local-evidence" "$PAINTER_RUN_ROOT/async-output/$PAINTER_RUN_NAME/artifacts/adapters/step_12"
                cat > "$PAINTER_RUN_ROOT/supervisor-state.json" <<'JSON'
                {{"completed_optimizer_steps":12,"last_optimizer_step":12}}
                JSON
                cat > "$PAINTER_RUN_ROOT/terminal-receipt.json" <<'JSON'
                {{"status":"complete","pipeline_exit_code":0,"train_exit_code":0,"evaluation_exit_code":0,"finalizer_exit_code":0,"checkpoint_step":12,"public_archive_hash_verified":true}}
                JSON
                printf 'fixture\\n' > "$PAINTER_RUN_ROOT/async-output/$PAINTER_RUN_NAME/artifacts/adapters/step_12/adapter_model.safetensors"
                python3 - "$PAINTER_RUN_ROOT" "$PAINTER_RUN_ROOT/local-evidence/latest-public-checkpoint.json" <<'PY'
                import hashlib, json, sys
                from pathlib import Path
                run, target = map(Path, sys.argv[1:])
                adapter = run / 'async-output' / run.name / 'artifacts/adapters/step_12'
                weight = adapter / 'adapter_model.safetensors'
                digest = hashlib.sha256(weight.read_bytes()).hexdigest()
                (adapter / 'hf-upload.json').write_text(json.dumps({{'verified':True,'private':False,'files':{{'adapter_model.safetensors':{{'sha256':digest}}}}}}))
                target.write_text(json.dumps({{'step':12,'adapter':str(adapter),'receipt':str(adapter / 'hf-upload.json'),'adapter_sha256':digest,'selection':{{'primary_quality_status':'primary'}}}}))
                PY
            """))
            archive_python = root / "prime/.venv/bin/python"
            archive_python.parent.mkdir(parents=True)
            archive_python.write_text(textwrap.dedent("""\
                #!/usr/bin/env python3
                import json, pathlib, sys
                args = sys.argv[1:]
                if args and args[0] == '-':
                    import os
                    os.execv('/usr/bin/python3', ['/usr/bin/python3'] + args)
                output = pathlib.Path(args[args.index('--output') + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(b'fixture-archive')
                output.with_suffix(output.suffix + '.upload.json').write_text(json.dumps({'private': False, 'public_hash_verified': True}))
            """))
            for path in (prepare, baseline, restore, arm_env, execute, archive_python):
                path.chmod(0o755)
            env = os.environ.copy()
            env.update({
                "PAINTER_MATCHED_PREPARE_SCRIPT": str(prepare),
                "PAINTER_MATCHED_BASELINE_SCRIPT": str(baseline),
                "PAINTER_MATCHED_RESTORE_SCRIPT": str(restore),
                "PAINTER_MATCHED_EXECUTE_SCRIPT": str(execute),
                "PAINTER_MATCHED_ARM_ENV_SCRIPT": str(arm_env),
                "PAINTER_MATCHED_ARM_ROOT_BASE": str(arms),
                "PAINTER_MATCHED_SHARED_ROOT": str(shared),
                "PAINTER_MATCHED_BASELINE_ROOT": str(root / "baseline"),
                "PAINTER_CAMPAIGN_STATE_ROOT": str(state),
                "PAINTER_CAMPAIGN_ALLOW_NON_LINUX": "1",
                "PAINTER_CAMPAIGN_SKIP_ARCHIVE": "1",
                "PAINTER_CAMPAIGN_SKIP_PREFLIGHT": "1",
                "PAINTER_CAMPAIGN_PYTHON": "python3",
                "PRIME_ROOT": str(root / "prime"),
                "HF_TOKEN_PATH": str(token),
            })
            result = subprocess.run(
                [str(RUN / "campaign.sh"), "--continuation", "1000", "3.8", "43.5"],
                cwd=RUN, env=env, text=True, capture_output=True, timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            calls = marker.read_text().splitlines()
            self.assertIn("restore", calls)
            self.assertIn("flash-evaluate:12:balanced16", calls)
            self.assertIn("flash-finalize:12", calls)
            self.assertEqual([row for row in calls if row.startswith("execute:")], ["execute:pro", "execute:luna"])
            self.assertNotIn("baseline-called", calls)
            terminal = json.loads((state / "campaign/terminal-receipt.json").read_text())
            self.assertEqual([row["arm"] for row in terminal["arms"]], ["flash", "pro", "luna"])
            self.assertEqual(terminal["arms"][0]["provenance_type"], "recovered_evaluation")
            self.assertFalse(terminal["arms"][0]["training_invoked"])
            self.assertEqual(sum(row["completed_optimizer_steps"] for row in terminal["arms"]), 36)
            status = json.loads((state / "campaign/status.json").read_text())
            self.assertEqual(status["campaign_cap_usd"], 43.5)
            self.assertEqual(status["per_arm_cap_usd"], 19.5)


if __name__ == "__main__":
    unittest.main()
