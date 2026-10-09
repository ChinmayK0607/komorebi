"""Offline tests for exact-node takeover and release behavior."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import hashlib
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock


RUN = Path(__file__).resolve().parent


def load():
    spec = importlib.util.spec_from_file_location("structured_flow_recover_contract", RUN / "recover_exact_node.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("recovery wrapper is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeAPI:
    def __init__(self):
        self.removed = []

    def ps(self):
        return [SimpleNamespace(id="pod-1")]

    def rm(self, pod):
        self.removed.append(pod.id)


class RecoveryTests(unittest.TestCase):
    def test_prime_command_reuses_existing_checkout_and_verifies_pin(self):
        module = load()
        command = module.prime_seed_command(Path("/tmp/structured-prime-seed/prime"))
        self.assertIn("https://github.com/PrimeIntellect-ai/prime-rl.git", command)
        self.assertIn("cat-file -e", command)
        self.assertIn(module.PRIME_COMMIT, command)
        self.assertIn("fsck --full", command)
        self.assertIn("test ! -L", command)

    def test_public_source_url_is_immutable_credential_free_hf_revision(self):
        module = load()
        valid = f"https://huggingface.co/datasets/example/resolve/{'a' * 40}/source.bundle"
        self.assertEqual(module._validate_public_source(valid, "b" * 64), (valid, "b" * 64))
        for invalid in (
            valid + "?token=secret",
            valid.replace("huggingface.co", "example.com"),
            valid.replace("/resolve/" + "a" * 40, "/resolve/main"),
        ):
            with self.assertRaises(ValueError):
                module._validate_public_source(invalid, "b" * 64)

    def test_source_bypass_skips_only_verified_source_and_keeps_credentials_on_sftp(self):
        module = load()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bundle"
            source.write_bytes(b"verified-source")
            expected = hashlib.sha256(source.read_bytes()).hexdigest()
            calls = []

            def original(_sftp, _local, remote, *, timeout):
                calls.append((remote, timeout))

            with mock.patch.object(module.base, "_sftp_put", side_effect=original):
                with module._verified_source_sftp_bypass(
                    source_bundle=source,
                    remote_source=Path("/tmp/structured-flow-large/source.bundle"),
                    expected_sha256=expected,
                ):
                    module.base._sftp_put(object(), source, "/tmp/structured-flow-large/source.bundle", timeout=5.0)
                    module.base._sftp_put(object(), source, "/root/painter/private/hf-token", timeout=5.0)
            self.assertEqual(calls, [("/root/painter/private/hf-token", 5.0)])

            with self.assertRaisesRegex(ValueError, "hash"):
                with module._verified_source_sftp_bypass(
                    source_bundle=source,
                    remote_source=Path("/tmp/structured-flow-large/source.bundle"),
                    expected_sha256="c" * 64,
                ):
                    pass

    def test_stage_failure_releases_exact_supplied_pod(self):
        module = load()
        api = FakeAPI()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bundle"
            token = root / "token"
            gateway = root / "gateway.env"
            source.write_bytes(b"source")
            token.write_text("token")
            gateway.write_text("gateway")
            token.chmod(0o600)
            gateway.chmod(0o600)
            with mock.patch.object(module, "_remote_prepare", side_effect=RuntimeError("stage failed")), \
                 mock.patch.object(module.base, "operate") as operate:
                with self.assertRaisesRegex(RuntimeError, "stage failed"):
                    module.recover(
                        api=api,
                        pod_id="pod-1",
                        rental_start_epoch=1_800_000_000.0,
                        price_usd_hour=3.70,
                        cap_usd=14.75,
                        source_bundle=source,
                        source_commit="a" * 40,
                        hf_token_file=token,
                        gateway_env_file=gateway,
                        output_root=root / "output",
                    )
                operate.assert_not_called()
        self.assertEqual(api.removed, ["pod-1"])

    def test_paused_predecessor_receives_term_then_continue(self):
        module = load()
        with mock.patch.object(module.os, "kill") as kill:
            module._terminate_previous_controller(41528)
        self.assertEqual(kill.call_args_list, [
            mock.call(41528, module.signal.SIGTERM),
            mock.call(41528, module.signal.SIGCONT),
        ])

    def test_release_failure_is_reported_after_successful_handoff(self):
        module = load()

        class FailingReleaseAPI(FakeAPI):
            def rm(self, _pod):
                raise RuntimeError("provider release failed")

        api = FailingReleaseAPI()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bundle"
            token = root / "token"
            gateway = root / "gateway.env"
            source.write_bytes(b"source")
            token.write_text("token")
            gateway.write_text("gateway")
            token.chmod(0o600)
            gateway.chmod(0o600)
            with mock.patch.object(module, "_remote_prepare"), \
                 mock.patch.object(module.base, "operate", return_value=0), \
                 mock.patch("sys.stderr") as stderr:
                with self.assertRaisesRegex(RuntimeError, "exact pod release failed"):
                    module.recover(
                        api=api,
                        pod_id="pod-1",
                        rental_start_epoch=1_800_000_000.0,
                        price_usd_hour=3.70,
                        cap_usd=14.75,
                        source_bundle=source,
                        source_commit="a" * 40,
                        hf_token_file=token,
                        gateway_env_file=gateway,
                        output_root=root / "output",
                    )
            stderr.write.assert_called()

    def test_handoff_uses_original_budget_and_remote_scratch_source_path(self):
        module = load()
        api = FakeAPI()
        observed = {}

        def fake_operate(**kwargs):
            observed["bundle"] = module.base.REMOTE_BUNDLE
            observed["scratch"] = kwargs["source_bundle"]
            observed["budget"] = (
                kwargs["window"].rental_start_epoch,
                kwargs["window"].price_usd_per_hour,
                kwargs["window"].cap_usd,
            )
            observed["env"] = {
                key: __import__("os").environ.get(key)
                for key in (
                    "PAINTER_LOCAL_SCRATCH_ROOT",
                    "PAINTER_PRIME_SEED_ENABLED",
                    "PAINTER_PRIME_SEED_ROOT",
                )
            }
            kwargs["api"].rm(SimpleNamespace(id="pod-1"))
            return 0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bundle"
            token = root / "token"
            gateway = root / "gateway.env"
            source.write_bytes(b"source")
            token.write_text("token")
            gateway.write_text("gateway")
            token.chmod(0o600)
            gateway.chmod(0o600)
            with mock.patch.object(module, "_remote_prepare"), \
                 mock.patch.object(module.base, "operate", side_effect=fake_operate):
                result = module.recover(
                    api=api,
                    pod_id="pod-1",
                    rental_start_epoch=1_800_000_000.0,
                    price_usd_hour=3.70,
                    cap_usd=14.75,
                    source_bundle=source,
                    source_commit="a" * 40,
                    hf_token_file=token,
                    gateway_env_file=gateway,
                    output_root=root / "output",
                )
        self.assertEqual(result, 0)
        self.assertEqual(observed["bundle"], Path("/tmp/structured-flow-large/source.bundle"))
        self.assertEqual(observed["scratch"], source.resolve())
        self.assertEqual(observed["budget"], (1_800_000_000.0, 3.70, 14.75))
        self.assertEqual(observed["env"], {
            "PAINTER_LOCAL_SCRATCH_ROOT": "/tmp/structured-flow-large",
            "PAINTER_PRIME_SEED_ENABLED": "1",
            "PAINTER_PRIME_SEED_ROOT": "/tmp/structured-prime-seed/prime",
        })
        self.assertEqual(api.removed, ["pod-1"])


if __name__ == "__main__":
    unittest.main()
