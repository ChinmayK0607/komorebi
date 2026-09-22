"""Pure controller tests for the Lium-to-node RL front end."""

from __future__ import annotations

import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from painter import rl_remote


class FakeRemote:
    def __init__(self):
        self.calls: list[tuple[list[str], bytes | None, bool]] = []

    def run(self, argv, *, input_bytes=None, check=True):
        argv = list(argv)
        self.calls.append((argv, input_bytes, check))
        if argv[:2] == ["test", "-f"]:
            return subprocess.CompletedProcess(argv, 1, "", "")
        if argv[:1] == ["cat"]:
            value = "/node/model" if argv[-1].endswith("model-path.txt") else "/node/adapter"
            return subprocess.CompletedProcess(argv, 0, value + "\n", "")
        if argv[:2] == ["python3", "-c"] and "prepared HF token path" in argv[2]:
            return subprocess.CompletedProcess(
                argv, 0, json.dumps({"HF_TOKEN_PATH": "/node/private/hf-token", "UV_BIN": "/node/bootstrap/bin/uv"}) + "\n", ""
            )
        if "prepare" in argv:
            return subprocess.CompletedProcess(argv, 0, json.dumps({"status": "prepared", "run_id": "case"}) + "\n", "")
        return subprocess.CompletedProcess(argv, 0, json.dumps({"status": "ok"}) + "\n", "")


class RemoteFrontEndTests(unittest.TestCase):
    def test_selects_ephemeral_ps_record_by_huid(self):
        connection = rl_remote.connection_from_ps(
            [{"id": "pod-1", "huid": "blue-node", "status": "RUNNING", "ip": "192.0.2.8", "ports": {"22": 2202}}],
            "blue-node",
        )
        self.assertEqual((connection.pod_id, connection.host, connection.port), ("pod-1", "192.0.2.8", 2202))

    def test_prepare_reads_node_pointers_and_keeps_manifest_remote(self):
        remote = FakeRemote()
        with patch.object(rl_remote, "stage_reviewed_code", return_value={"status": "staged"}), patch.object(
            rl_remote, "reuse_profile_assets", return_value={"status": "assets_ready"}
        ):
            result = rl_remote.prepare_remote(
                remote,
                repo_root=Path("/controller/repo"),
                profile_root=Path("/root/painter/runs/prepared"),
                remote_root=Path("/root/painter/runs/new-case"),
                remote_code_root=Path("/root/painter"),
                data="/root/painter/source-manifest.json",
                run_id="case",
                prime_root=Path("/root/painter/prime-rl"),
                max_steps=8,
                base_revision="base",
                prime_commit="prime",
                validate_prime=False,
            )
        self.assertEqual(result["prepare"]["status"], "prepared")
        prepare = next(call[0] for call in remote.calls if "prepare" in call[0])
        self.assertIn("/root/painter/source-manifest.json", prepare)
        self.assertIn("/node/model", prepare)
        self.assertIn("/node/adapter", prepare)
        self.assertNotIn("HF_TOKEN=", json.dumps(prepare))
        self.assertIn("HF_TOKEN_PATH=/node/private/hf-token", prepare)
        self.assertIn("UV_BIN=/node/bootstrap/bin/uv", prepare)

    def test_ssh_quotes_remote_argv_as_one_shell_command(self):
        connection = rl_remote.NodeConnection("node", "pod", "192.0.2.8", 22)
        completed = subprocess.CompletedProcess([], 0, b"ok\n", b"")
        with patch.object(rl_remote.subprocess, "run", return_value=completed) as run:
            rl_remote.SSHRemote(connection, identity=None, known_hosts=None).run(
                ["python3", "-c", "print('x y')", "/tmp/a b"]
            )
        command = run.call_args.args[0]
        self.assertEqual(command[-1], shlex.join(["python3", "-c", "print('x y')", "/tmp/a b"]))
        self.assertIsInstance(rl_remote._stdout(completed), str)

    def test_remote_stop_contains_pid_start_token_guard(self):
        remote = FakeRemote()
        result = rl_remote.stop_remote(remote, remote_root=Path("/node/run"))
        self.assertEqual(result["status"], "ok")
        command = next(call[0] for call in remote.calls if call[0][:2] == ["python3", "-c"])
        self.assertIn("supervisor_start_token", command[2])
        self.assertIn("os.killpg", command[2])

    def test_preflight_sets_no_install_and_runtime_paths(self):
        remote = FakeRemote()
        result = rl_remote.preflight_remote(
            remote,
            remote_root=Path("/node/run"),
            remote_code_root=Path("/node/code"),
            runtime={"HF_TOKEN_PATH": "/node/private/hf-token", "UV_BIN": "/node/bootstrap/bin/uv"},
        )
        self.assertEqual(result["status"], "ok")
        command = next(call[0] for call in remote.calls if call[0][:1] == ["env"])
        self.assertIn("PAINTER_RL_PREFLIGHT_ONLY=1", command)
        self.assertIn("PAINTER_RL_SKIP_EDITABLE_INSTALL=1", command)
        self.assertIn("HF_TOKEN_PATH=/node/private/hf-token", command)
        self.assertIn("UV_BIN=/node/bootstrap/bin/uv", command)

    def test_main_reports_failed_remote_run(self):
        with patch.object(rl_remote, "run", return_value={"status": "failed", "run_id": "case"}):
            self.assertEqual(
                rl_remote.main(["run", "--node", "node", "--data", "/node/data", "--run-id", "case"]),
                1,
            )

    def test_json_last_accepts_pretty_object_after_logs(self):
        value = rl_remote._json_last(
            '{"prepare_log": true}\nnoise\n{\n  "status": "prepared",\n  "run_id": "case"\n}\n',
            name="prepare",
        )
        self.assertEqual(value, {"status": "prepared", "run_id": "case"})

    def test_duplicate_start_does_not_spawn_local_judge(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = Path(directory) / "id"
            known_hosts = Path(directory) / "known_hosts"
            identity.write_text("local-only")
            known_hosts.write_text("local-only")
            args = rl_remote._parser().parse_args([
                "run", "--node", "blue-node", "--data", "/node/data.json", "--run-id", "case",
                "--identity", str(identity), "--known-hosts", str(known_hosts),
            ])
            fake_connection = rl_remote.NodeConnection("blue-node", "pod-1", "192.0.2.8", 2202)
            with patch.object(rl_remote, "discover_node", return_value=fake_connection), patch.object(
                rl_remote, "SSHRemote", return_value=FakeRemote()
            ), patch.object(rl_remote, "prepare_remote", return_value={"prepare": {"status": "prepared"}}), patch.object(
                rl_remote, "start_remote", return_value={"status": "already_running"}
            ), patch.object(rl_remote.subprocess, "Popen") as popen:
                result = rl_remote.run(args, repo_root=Path(directory))
            self.assertEqual(result["status"], "already_running")
            self.assertTrue(result["duplicate_start"])
            popen.assert_not_called()

    def test_existing_spec_is_checked_before_prepare(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = Path(directory) / "id"
            known_hosts = Path(directory) / "known_hosts"
            identity.write_text("local-only")
            known_hosts.write_text("local-only")
            args = rl_remote._parser().parse_args([
                "run", "--node", "blue-node", "--data", "/node/data.json", "--run-id", "case",
                "--identity", str(identity), "--known-hosts", str(known_hosts),
            ])
            connection = rl_remote.NodeConnection("blue-node", "pod-1", "192.0.2.8", 2202)
            with patch.object(rl_remote, "discover_node", return_value=connection), patch.object(
                rl_remote, "SSHRemote", return_value=FakeRemote()
            ), patch.object(rl_remote, "remote_run_exists", return_value=True), patch.object(
                rl_remote, "status_remote", return_value={"status": "running", "pid": 7}
            ), patch.object(rl_remote, "prepare_remote") as prepare, patch.object(rl_remote, "start_remote") as start:
                result = rl_remote.run(args, repo_root=Path(directory))
            self.assertEqual(result["status"], "running")
            self.assertTrue(result["duplicate_start"])
            prepare.assert_not_called()
            start.assert_not_called()

    def test_prepared_spec_can_transition_to_start(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = Path(directory) / "id"
            known_hosts = Path(directory) / "known_hosts"
            identity.write_text("local-only")
            known_hosts.write_text("local-only")
            args = rl_remote._parser().parse_args([
                "run", "--node", "blue-node", "--data", "/node/data.json", "--run-id", "case",
                "--identity", str(identity), "--known-hosts", str(known_hosts),
            ])
            connection = rl_remote.NodeConnection("blue-node", "pod-1", "192.0.2.8", 2202)
            with patch.object(rl_remote, "discover_node", return_value=connection), patch.object(
                rl_remote, "SSHRemote", return_value=FakeRemote()
            ), patch.object(rl_remote, "remote_run_exists", return_value=True), patch.object(
                rl_remote, "status_remote", return_value={"status": "not_started"}
            ), patch.object(rl_remote, "profile_runtime", return_value={"HF_TOKEN_PATH": "/node/token", "UV_BIN": "uv"}), patch.object(
                rl_remote, "start_remote", return_value={"status": "already_running"}
            ) as start, patch.object(rl_remote, "prepare_remote") as prepare:
                result = rl_remote.run(args, repo_root=Path(directory))
            self.assertEqual(result["status"], "already_running")
            prepare.assert_not_called()
            start.assert_called_once()
            self.assertEqual(start.call_args.kwargs["runtime"]["HF_TOKEN_PATH"], "/node/token")

    def test_dry_run_does_not_resolve_provider_or_ssh(self):
        args = rl_remote._parser().parse_args([
            "run", "--node", "blue-node", "--data", "/node/data.json", "--run-id", "case", "--dry-run",
        ])
        with patch.object(rl_remote, "discover_node") as discover:
            result = rl_remote.run(args)
        self.assertEqual(result["status"], "dry_run")
        self.assertFalse(result["provider_details_persisted"])
        discover.assert_not_called()


if __name__ == "__main__":
    unittest.main()
