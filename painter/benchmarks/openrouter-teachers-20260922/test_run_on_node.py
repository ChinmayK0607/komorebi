from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import sys
import subprocess
import tarfile
import types

import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("openrouter_teacher_transport", HERE / "run_on_node.py")
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class FakeTransport:
    instances: list["FakeTransport"] = []

    def __init__(self, connection, *, identity, known_hosts):
        self.calls: list[tuple[list[str], bytes | None]] = []
        self.foreground_calls: list[list[str]] = []
        self.__class__.instances.append(self)

    def run(self, argv, *, input_bytes=None, timeout=300):
        argv = list(argv)
        self.calls.append((argv, input_bytes))
        if argv and argv[0] == "python3" and "tqdm>=4.66" in " ".join(argv):
            return __import__("subprocess").CompletedProcess(
                argv, 0, stdout=b"/node/run/.benchmark-venv/bin/python\n", stderr=b""
            )
        if argv and argv[0] == "python3" and "tarfile" in " ".join(argv):
            data = io.BytesIO()
            with tarfile.open(fileobj=data, mode="w:gz") as archive:
                payload = b'{"status":"complete"}\n'
                info = tarfile.TarInfo("results/episode.json")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            return __import__("subprocess").CompletedProcess(argv, 0, stdout=data.getvalue(), stderr=b"")
        return __import__("subprocess").CompletedProcess(argv, 0, stdout=b"", stderr=b"")

    def foreground(self, argv):
        self.foreground_calls.append(list(argv))
        return 0


def plan(**kwargs):
    return module.build_package_plan(benchmark_root=HERE, **kwargs)


class TestRunOnNode(unittest.TestCase):

    def test_actual_prepared_directory_has_exact_reference_allowlist(self):
        prepared = plan()
        assert prepared.count == 40
        names = {item.archive_name for item in (*prepared.files, *prepared.references)}
        assert {"run.py", "config.json", "prompt.txt", "contract.json", "model-catalog.json", "refs.json"} <= names
        assert sum(name.startswith("references/") for name in names) == 40
        assert not any(name.startswith(("results/", "private/", "episodes/")) for name in names)


    def test_models_and_track_selection_is_versioned_in_config(self):
        prepared = plan(models=["zai/glm-5.3-flash"], tracks=["speed"])
        config = next(item for item in prepared.files if item.archive_name == "config.json")
        assert config.data is None
        assert module._effective_node_root(prepared, "/root/run") == "/root/run"

    def test_screen_track_packages_without_expanding_all(self):
        prepared = plan(models=["zai/glm-5.3-flash"], tracks=["screen"])
        assert prepared.count == 40
        assert module._parser().parse_args(["--dry-run", "--track", "all"]).track_alias == ["all"]

    def test_unknown_model_is_allowed_only_for_real_run_package(self):
        prepared = module.build_package_plan(
            benchmark_root=HERE, models=["provider/new-image-model"], allow_unknown_models=True
        )
        assert prepared.count == 40


    def test_unknown_model_is_preserved_without_catalog_network(self):
        prepared = plan(models=["provider/new-image-model"])
        assert prepared.count == 40


    def test_dry_run_does_not_read_key_or_discover(self):
        original_key = module._api_key
        original_getpass = module.getpass
        try:
            module._api_key = lambda: (_ for _ in ()).throw(AssertionError("key read"))
            module.getpass = types.SimpleNamespace(getpass=lambda _: (_ for _ in ()).throw(AssertionError("prompt")))
            assert module.main(["--dry-run", "--track", "all"]) == 0
            assert module.main(["--dry-run", "--track", "screen"]) == 0
        finally:
            module._api_key = original_key
            module.getpass = original_getpass


    def test_run_stages_key_without_putting_it_in_runner_argv(self):
        tmp_path = Path(__import__("tempfile").mkdtemp(prefix="teacher-wrapper-test-"))
        FakeTransport.instances.clear()
        result = module.run_benchmark(
            plan=plan(), pod="painter-photo-rl", node_root="/node/run", output_dir=tmp_path,
            identity=None, known_hosts=None, discover=lambda *a, **k: "connection",
            transport_factory=FakeTransport, api_key=b"secret-value", models=["zai/glm-5.3-flash", "xiaomi/mimo-v2.6-pro"],
            tracks=("quality",), limit_episodes=1,
        )
        assert result == 0
        transport = FakeTransport.instances[-1]
        assert transport.foreground_calls
        command = transport.foreground_calls[0]
        assert "secret-value" not in " ".join(command)
        assert "--root" in command and "/node/run" in command
        assert "--run" in command
        assert "--config" not in command and "--output-dir" not in command
        assert "--api-key-file" in command
        assert "--track" in command and "quality" in command
        model_index = command.index("--models")
        assert command[model_index + 1] == "zai/glm-5.3-flash,xiaomi/mimo-v2.6-pro"
        assert "--limit-episodes" in command
        key_inputs = [payload for argv, payload in transport.calls if payload == b"secret-value"]
        assert key_inputs == [b"secret-value"]
        assert (tmp_path / "results" / "episode.json").read_text() == '{"status":"complete"}\n'
        assert any(argv[:2] == ["rm", "-f"] for argv, _ in transport.calls)

    def test_runner_cli_accepts_wrapper_offline_mode(self):
        result = subprocess.run(
            [sys.executable, str(HERE / "run.py"), "--root", str(HERE), "--dry-run",
             "--track", "all", "--models", "zai/glm-5.3-flash",
             "--models", "xiaomi/mimo-v2.6-pro", "--limit-episodes", "1"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"paid_calls_made": 0', result.stdout)
        self.assertIn("zai/glm-5.3-flash", result.stdout)
        self.assertIn("xiaomi/mimo-v2.6-pro", result.stdout)


    def test_collected_archive_rejects_path_traversal(self):
        tmp_path = Path(__import__("tempfile").mkdtemp(prefix="teacher-wrapper-test-"))
        archive_path = tmp_path / "bad.tar.gz"
        with tarfile.open(archive_path, "w:gz") as archive:
            info = tarfile.TarInfo("../escape.txt")
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))
        with self.assertRaisesRegex(module.BenchmarkLaunchError, "unsafe member"):
            module._safe_extract(archive_path, tmp_path / "out")

    def test_collect_script_keeps_results_and_excludes_dependency_caches_and_dotenv(self):
        tmp_path = Path(__import__("tempfile").mkdtemp(prefix="teacher-collect-test-"))
        keep = tmp_path / "results" / "episode.json"
        keep.parent.mkdir(parents=True)
        keep.write_text('{"status":"complete"}\n')
        for relative in (
            "node_modules/ai/index.js",
            ".pnpm-store/v3/files/cache.bin",
            ".cache/tool/cache.bin",
            ".env.local",
            "private/ai-gateway.key",
            "logs/token.secret",
        ):
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"excluded")
        completed = subprocess.run(
            [sys.executable, "-c", module._collect_script(), str(tmp_path)],
            capture_output=True,
            check=True,
        )
        with tarfile.open(fileobj=io.BytesIO(completed.stdout), mode="r:gz") as archive:
            names = set(archive.getnames())
        assert "results/episode.json" in names
        assert not any(name.startswith(("node_modules/", ".pnpm-store/", ".cache/", "private/", "logs/")) for name in names)
        assert ".env.local" not in names

if __name__ == "__main__":
    unittest.main()
