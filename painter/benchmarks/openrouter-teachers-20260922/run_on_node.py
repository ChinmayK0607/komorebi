"""Run the prepared Vercel AI Gateway teacher benchmark on an existing Linux node.

The Mac-side wrapper stages only the benchmark code, JSON/TXT configuration,
the reviewed JPEG references, and a hash manifest. It never creates or stops a
pod. The node reuses its installed renderer environment and runs the benchmark
foreground so an interrupted invocation can be resumed by rerunning the same command. The
AI Gateway key is written only to a node-private ephemeral file over SSH and
is removed in ``finally``; it is never an argv value, archive member, or log.

The wrapper and ``run.py`` share one explicit CLI: ``--root``, ``--run``,
``--track``, ``--models``, ``--api-key-file``, ``--renderer``,
``--renderer-python``, ``--browser-path``, ``--run-as-user``, and
``--limit-episodes``.  The runner must
remove the key from the environment before spawning the renderer.  Keeping
that boundary explicit avoids silently collapsing per-model key bundles onto a
single default key.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import getpass
import hashlib
import io
import json
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tarfile
from typing import Any, Callable, Iterable, Sequence


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_POD = "painter-photo-rl"
DEFAULT_NODE_ROOT = "/root/painter/benchmarks/ai-gateway-teachers-20260923"
DEFAULT_NODE_RENDERER = "/root/painter/vendor/integrations/watercolour/renderer.py"
DEFAULT_NODE_RENDERER_PYTHON = "/root/painter/renderer-env/bin/python"
DEFAULT_NODE_BROWSERS = "/root/painter/browsers"
DEFAULT_KEY_PATH = "/root/painter/benchmarks/ai-gateway-teachers-20260923/private/ai-gateway.key"
RUNNER_NAME = "run.py"
STAGE_MANIFEST_NAME = "stage-manifest.json"
MAX_REFERENCE_COUNT = 40
SECRET_ENV = "AI_GATEWAY_API_KEY"


class BenchmarkLaunchError(RuntimeError):
    """The local package or node launch contract is invalid."""


@dataclass(frozen=True)
class PackageFile:
    source: Path
    archive_name: str
    sha256: str
    bytes: int
    data: bytes | None = None


@dataclass(frozen=True)
class PackagePlan:
    root: Path
    runner: Path
    files: tuple[PackageFile, ...]
    references: tuple[PackageFile, ...]
    config: Path
    prompt: Path

    @property
    def count(self) -> int:
        return len(self.references)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path, *, label: str) -> Path:
    path = path.expanduser()
    if path.is_symlink():
        raise BenchmarkLaunchError(f"{label} must be a regular non-symlink file: {path}")
    path = path.resolve(strict=True)
    if not path.is_file():
        raise BenchmarkLaunchError(f"{label} must be a regular file: {path}")
    return path


def _package_file(path: Path, archive_name: str, *, data: bytes | None = None) -> PackageFile:
    path = _regular_file(path, label=archive_name)
    if not archive_name or archive_name.startswith("/") or ".." in Path(archive_name).parts:
        raise BenchmarkLaunchError(f"unsafe archive path: {archive_name}")
    if data is None:
        digest = sha256(path)
        size = path.stat().st_size
    else:
        digest = hashlib.sha256(data).hexdigest()
        size = len(data)
    return PackageFile(path, archive_name, digest, size, data)


def _load_json(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkLaunchError(f"invalid {label}: {path}") from exc


def _validate_prepared_config(config: Path, refs_manifest: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    document = _load_json(config, label="benchmark config")
    if not isinstance(document, dict) or document.get("benchmark") != "ai-gateway-teachers-20260923":
        raise BenchmarkLaunchError("config benchmark identity is not ai-gateway-teachers-20260923")
    models = document.get("models")
    if not isinstance(models, list) or not models or not all(isinstance(item, str) and item for item in models):
        raise BenchmarkLaunchError("config must contain a non-empty model id list")
    tracks = document.get("tracks")
    if not isinstance(tracks, dict) or set(tracks) != {"quality", "speed", "screen"}:
        raise BenchmarkLaunchError("config must define quality, speed and screen tracks")
    quality = tracks["quality"]
    speed = tracks["speed"]
    screen = tracks["screen"]
    if quality.get("max_turns") != 12 or speed.get("max_turns") != 3 or speed.get("max_tokens") != 8192:
        raise BenchmarkLaunchError("track limits do not match the prepared benchmark")
    if speed.get("episode_timeout_seconds") != 900:
        raise BenchmarkLaunchError("speed track timeout must be 900 seconds")
    if screen.get("max_turns") != 6 or screen.get("max_tokens") != "native" or screen.get("episode_timeout_seconds") is not None or screen.get("reasoning_effort") != "highest_supported":
        raise BenchmarkLaunchError("screen track limits do not match the prepared benchmark")
    references = _load_json(refs_manifest, label="reference manifest")
    if not isinstance(references, dict) or not isinstance(references.get("references"), list):
        raise BenchmarkLaunchError("reference manifest must contain a references list")
    if len(references["references"]) != MAX_REFERENCE_COUNT:
        raise BenchmarkLaunchError("reference manifest must contain exactly 40 images")
    screen_ids = document.get("screen_reference_ids")
    by_id = {row.get("id"): row for row in references["references"] if isinstance(row, dict)}
    if not isinstance(screen_ids, list) or len(screen_ids) != 8 or len(set(screen_ids)) != 8:
        raise BenchmarkLaunchError("screen_reference_ids must contain eight unique references")
    if any(item not in by_id for item in screen_ids):
        raise BenchmarkLaunchError("screen_reference_ids contains an unknown reference")
    categories = [by_id[item].get("category") for item in screen_ids]
    if any(not isinstance(category, str) or not category for category in categories) or len(set(categories)) != 8:
        raise BenchmarkLaunchError("screen_reference_ids must select one reference per category")
    return document, references["references"]


def build_package_plan(
    *,
    benchmark_root: Path = ROOT,
    runner: Path | None = None,
    references: Path | None = None,
    config: Path | None = None,
    prompt: Path | None = None,
    models: Sequence[str] | None = None,
    tracks: Sequence[str] | None = None,
    allow_unknown_models: bool = False,
) -> PackagePlan:
    """Validate and describe the allowlisted package without network access."""
    root = benchmark_root.expanduser().resolve(strict=True)
    runner_path = _regular_file(runner or root / RUNNER_NAME, label="benchmark runner")
    refs_root = (references or root / "references").expanduser().resolve(strict=True)
    if not refs_root.is_dir() or refs_root.is_symlink():
        raise BenchmarkLaunchError(f"references must be a regular directory: {refs_root}")
    config_path = _regular_file(config or root / "config.json", label="benchmark config")
    prompt_path = _regular_file(prompt or root / "prompt.txt", label="benchmark prompt")
    refs_manifest = _regular_file(root / "refs.json", label="reference manifest")
    document, manifest_rows = _validate_prepared_config(config_path, refs_manifest)
    catalog = _load_json(root / "model-catalog.json", label="model catalog")
    catalog_models = {
        str(item.get("id")) for item in (catalog.get("models", []) if isinstance(catalog, dict) else [])
        if isinstance(item, dict) and item.get("id")
    }
    requested_models = list(models) if models is not None else list(document["models"])
    if not requested_models or not all(isinstance(item, str) and item for item in requested_models):
        raise BenchmarkLaunchError("models must contain at least one non-empty model id")
    # Gateway model IDs are intentionally open ended.  The Gateway request is
    # the capability probe; packaging must preserve the exact user-supplied
    # IDs instead of inventing an offline allowlist.
    by_name = {str(row.get("image")): row for row in manifest_rows if isinstance(row, dict)}
    reference_files: list[PackageFile] = []
    for image_name, row in sorted(by_name.items()):
        relative = Path(image_name)
        if relative.is_absolute() or ".." in relative.parts or relative.name != image_name.removeprefix("references/") or relative.suffix.lower() not in {".jpg", ".jpeg"}:
            raise BenchmarkLaunchError(f"reference manifest image is not a safe JPEG name: {image_name}")
        image = _regular_file(refs_root / relative.name, label="reference image")
        expected = row.get("sha256")
        if not isinstance(expected, str) or sha256(image) != expected:
            raise BenchmarkLaunchError(f"reference SHA-256 mismatch: {image.name}")
        reference_files.append(_package_file(image, f"references/{image.name}"))
    if len(reference_files) != MAX_REFERENCE_COUNT:
        raise BenchmarkLaunchError("prepared references must contain exactly 40 JPEGs")

    selected_tracks = tuple(tracks) if tracks is not None else ("quality", "speed")
    if not selected_tracks or any(track not in {"quality", "speed", "screen"} for track in selected_tracks):
        raise BenchmarkLaunchError("tracks must contain quality, speed and/or screen")
    if len(set(selected_tracks)) != len(selected_tracks):
        raise BenchmarkLaunchError("tracks contains duplicates")
    files = [
        _package_file(runner_path, RUNNER_NAME),
        _package_file(root / "gateway_transport.ts", "gateway_transport.ts"),
        _package_file(root / "gateway_prompt.ts", "gateway_prompt.ts"),
        _package_file(root / "package.json", "package.json"),
        _package_file(root / "pnpm-lock.yaml", "pnpm-lock.yaml"),
        _package_file(root / "pnpm-workspace.yaml", "pnpm-workspace.yaml"),
        _package_file(root / "tsconfig.json", "tsconfig.json"),
        _package_file(config_path, "config.json"),
        _package_file(prompt_path, "prompt.txt"),
        _package_file(root / "contract.json", "contract.json"),
        _package_file(root / "model-catalog.json", "model-catalog.json"),
        _package_file(refs_manifest, "refs.json"),
    ]
    # Include additional reviewed benchmark Python modules if the runner is
    # split into helpers. Generated results, private files, archives, and
    # credentials are deliberately excluded from the package.
    for path in sorted(root.glob("*.py")):
        if path.resolve() == runner_path or path.name.startswith("test_"):
            continue
        files.append(_package_file(path, path.name))
    return PackagePlan(root, runner_path, tuple(files), tuple(reference_files), config_path, prompt_path)


def _stage_manifest(plan: PackagePlan) -> bytes:
    entries = [
        {"path": item.archive_name, "bytes": item.bytes, "sha256": item.sha256}
        for item in (*plan.files, *plan.references)
    ]
    return json.dumps({"schema": "ai-gateway-teacher-stage-v1", "files": entries}, indent=2, sort_keys=True).encode() + b"\n"


def _archive(plan: PackagePlan) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for item in (*plan.files, *plan.references):
            info = tarfile.TarInfo(item.archive_name)
            info.size = item.bytes
            info.mode = 0o644
            stream = io.BytesIO(item.data) if item.data is not None else item.source.open("rb")
            try:
                archive.addfile(info, stream)
            finally:
                stream.close()
        manifest = _stage_manifest(plan)
        info = tarfile.TarInfo(STAGE_MANIFEST_NAME)
        info.size = len(manifest)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(manifest))
    return buffer.getvalue()


def _safe_remote_stage_script() -> str:
    return r'''import hashlib, io, json, pathlib, sys, tarfile
root = pathlib.Path(sys.argv[1]).expanduser()
root.mkdir(parents=True, exist_ok=True)
manifest = None
with tarfile.open(fileobj=sys.stdin.buffer, mode="r:gz") as archive:
    members = archive.getmembers()
    for member in members:
        target = (root / member.name).resolve()
        if target != root and root not in target.parents:
            raise SystemExit("unsafe staged path")
        if not member.isfile():
            raise SystemExit("stage archive contains a non-file")
        data = archive.extractfile(member).read()
        if member.name == "stage-manifest.json":
            manifest = json.loads(data)
            continue
        relative = pathlib.PurePosixPath(member.name)
        if relative.parts and relative.parts[0] in {"private", "episodes", "results", "review"}:
            raise SystemExit("stage archive may not replace run state")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.is_symlink():
            raise SystemExit("existing staged path is a symlink")
        if target.exists() and not target.is_file():
            raise SystemExit("existing staged path is not a file")
        if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != hashlib.sha256(data).hexdigest():
            temporary = target.with_name("." + target.name + ".stage.tmp")
            temporary.write_bytes(data)
            temporary.chmod(0o644)
            temporary.replace(target)
if not isinstance(manifest, dict) or manifest.get("schema") != "ai-gateway-teacher-stage-v1":
    raise SystemExit("missing stage manifest")
print(json.dumps({"status": "staged", "files": len(manifest.get("files", []))}, sort_keys=True))
'''


def _key_write_script() -> str:
    return r'''import os, pathlib, stat, sys
path = pathlib.Path(sys.argv[1]).expanduser()
value = sys.stdin.buffer.read().strip()
if not value or b"\x00" in value:
    raise SystemExit("empty or invalid key")
path.parent.mkdir(parents=True, exist_ok=True)
path.parent.chmod(0o700)
path.write_bytes(value + b"\n")
path.chmod(0o600)
st = path.stat()
if st.st_uid != os.getuid() or stat.S_IMODE(st.st_mode) != 0o600:
    raise SystemExit("key file permissions are not private")
'''


def _ensure_runner_env_script() -> str:
    return r'''import pathlib, subprocess, sys
root = pathlib.Path(sys.argv[1]).expanduser()
venv = root / ".benchmark-venv"
python = venv / "bin" / "python"
if not python.is_file():
    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(venv)], check=True)
probe = [str(python), "-c", "import tqdm"]
if subprocess.run(probe, check=False).returncode:
    subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-input", "tqdm>=4.66,<5"], check=True, stdout=subprocess.DEVNULL)
subprocess.run(probe, check=True)
print(python)
'''


def _ensure_gateway_env_script() -> str:
    return r'''import pathlib, subprocess, sys
root = pathlib.Path(sys.argv[1]).expanduser()
probe = subprocess.run(["pnpm", "exec", "tsx", "--version"], cwd=root, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
if probe.returncode:
    subprocess.run(["pnpm", "install", "--frozen-lockfile", "--ignore-scripts"], cwd=root, check=True)
subprocess.run(["pnpm", "exec", "tsx", "--version"], cwd=root, check=True, stdout=subprocess.DEVNULL)
print("ready")
'''


def _collect_script() -> str:
    return r'''import pathlib, sys, tarfile
root = pathlib.Path(sys.argv[1]).expanduser()
excluded_parts = {
    ".benchmark-venv", ".pnpm-store", "node_modules", ".cache", "cache",
    "__pycache__", ".git", "private", "keys", "credentials",
}
excluded_suffixes = (".key", ".pem", ".token", ".secret", ".credentials")
with tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz") as archive:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        name = path.name.lower()
        if excluded_parts.intersection(relative.parts) or name.startswith(".env") or name.endswith(excluded_suffixes):
            continue
        archive.add(path, arcname=relative.as_posix(), recursive=False)
'''


def _safe_extract(archive_path: Path, destination: Path) -> None:
    """Extract only regular, in-tree result files from the node archive."""
    destination = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            name = Path(member.name)
            if name.is_absolute() or ".." in name.parts or not member.isfile():
                raise BenchmarkLaunchError("collected archive contains an unsafe member")
            target = (destination / name).resolve()
            if destination != target and destination not in target.parents:
                raise BenchmarkLaunchError("collected archive escapes output directory")
            source = archive.extractfile(member)
            if source is None:
                raise BenchmarkLaunchError("collected archive member has no data")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as output:
                while block := source.read(1024 * 1024):
                    output.write(block)


class SSHTransport:
    """Long-running SSH transport; benchmark output is streamed locally."""

    def __init__(self, connection: Any, *, identity: Path | None, known_hosts: Path | None):
        from painter.rl_remote import _ssh_argv

        self.argv = _ssh_argv(connection, identity=identity, known_hosts=known_hosts)

    def run(self, argv: Sequence[str], *, input_bytes: bytes | None = None, timeout: float = 300) -> subprocess.CompletedProcess[bytes]:
        remote = shlex.join(list(argv))
        result = subprocess.run(self.argv + [remote], input=input_bytes, capture_output=True, timeout=timeout)
        if result.returncode:
            raise BenchmarkLaunchError(f"remote command failed with exit {result.returncode}")
        return result

    def foreground(self, argv: Sequence[str]) -> int:
        remote = shlex.join(list(argv))
        process = subprocess.Popen(self.argv + [remote])
        try:
            return process.wait()
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
            try:
                return process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                return process.wait()


def _api_key() -> bytes:
    value = __import__("os").environ.get(SECRET_ENV)
    if value is None:
        # Read only the one expected assignment from the ignored local dotenv
        # file.  The value stays in memory and is never printed or archived.
        dotenv = ROOT / ".env.local"
        try:
            for line in dotenv.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith(f"{SECRET_ENV}="):
                    value = stripped.split("=", 1)[1].strip().strip("\"'")
                    break
        except FileNotFoundError:
            pass
    if value is None:
        value = getpass.getpass("Vercel AI Gateway API key (used only for this run): ")
    value = value.strip()
    if not value:
        raise BenchmarkLaunchError("AI Gateway API key is empty")
    return value.encode()


def _api_key_bundle(path: Path | None) -> bytes:
    """Read a protected plaintext key or JSON key bundle without logging it."""
    if path is None:
        return _api_key()
    path = _regular_file(path, label="api-key-file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode())
    except (UnicodeDecodeError, json.JSONDecodeError):
        value = raw.decode(errors="strict").strip()
    if isinstance(value, str):
        if not value:
            raise BenchmarkLaunchError("api-key-file is empty")
        return value.encode()
    if not isinstance(value, dict):
        raise BenchmarkLaunchError("api-key-file must be plaintext or a JSON key bundle")
    if not isinstance(value.get("default"), str):
        raise BenchmarkLaunchError("JSON api-key-file requires a default string")
    models = value.get("models", {})
    if models is not None and (not isinstance(models, dict) or not all(isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in models.items())):
        raise BenchmarkLaunchError("JSON api-key-file models must map ids to non-empty strings")
    if not value["default"].strip() and not models:
        raise BenchmarkLaunchError("JSON api-key-file needs a default key or per-model keys")
    # Keep the bundle structure so the node runner can select per-model keys.
    return json.dumps(value, separators=(",", ":")).encode()


def _plan_json(plan: PackagePlan, *, pod: str, node_root: str) -> dict[str, Any]:
    return {
        "status": "dry_run",
        "pod": pod,
        "node_root": node_root,
        "package_file_count": len(plan.files) + len(plan.references) + 1,
        "package_bytes": sum(item.bytes for item in (*plan.files, *plan.references)),
        "reference_count": plan.count,
        "runner": plan.runner.name,
        "provider": "vercel-ai-gateway",
        "transport": "ai-sdk-generateText-jsonl",
        "renderer": {
            "python": DEFAULT_NODE_RENDERER_PYTHON,
            "module": DEFAULT_NODE_RENDERER,
            "browsers": DEFAULT_NODE_BROWSERS,
        },
        "provider_calls": False,
        "pod_lifecycle": "unchanged",
    }


def _effective_node_root(plan: PackagePlan, node_root: str) -> str:
    """Keep one resumable root while selection is passed to the runner."""
    return node_root.rstrip("/")


def run_benchmark(
    *,
    plan: PackagePlan,
    pod: str,
    node_root: str,
    output_dir: Path,
    identity: Path | None,
    known_hosts: Path | None,
    discover: Callable[..., Any],
    transport_factory: Callable[..., Any] = SSHTransport,
    api_key: bytes | None = None,
    models: Sequence[str] | None = None,
    tracks: Sequence[str] = ("quality", "speed"),
    limit_episodes: int | None = None,
) -> int:
    """Stage, execute, collect, and clean up one existing node run."""
    connection = discover(pod, user="root")
    transport = transport_factory(connection, identity=identity, known_hosts=known_hosts)
    root = node_root.rstrip("/")
    key_path = f"{root}/private/ai-gateway.key"
    archive = _archive(plan)
    key_bundle = api_key if api_key is not None else _api_key_bundle(None)
    output_dir.mkdir(parents=True, exist_ok=True)
    status = 1
    runner_env: str | None = None
    run_error: BaseException | None = None
    collect_error: BaseException | None = None
    try:
        transport.run(["python3", "-c", _safe_remote_stage_script(), root], input_bytes=archive, timeout=600)
        # Install tqdm only in a benchmark-local venv. The renderer remains in
        # its existing pinned environment and the shared training venv is never
        # mutated.
        runner_env = transport.run(
            ["python3", "-c", _ensure_runner_env_script(), root], timeout=600
        ).stdout.decode(errors="strict").strip()
        expected_runner = f"{root}/.benchmark-venv/bin/python"
        if runner_env != expected_runner:
            raise BenchmarkLaunchError("node benchmark environment returned an unexpected interpreter")
        transport.run(["python3", "-c", _ensure_gateway_env_script(), root], timeout=1200)
        transport.run(["python3", "-c", _key_write_script(), key_path], input_bytes=key_bundle, timeout=30)
        command = [
            runner_env, f"{root}/{RUNNER_NAME}",
            "--root", root,
            "--run",
            "--api-key-file", key_path,
            "--renderer", DEFAULT_NODE_RENDERER,
            "--renderer-python", DEFAULT_NODE_RENDERER_PYTHON,
            "--browser-path", DEFAULT_NODE_BROWSERS,
            "--run-as-user", "painter",
            "--track", "all" if set(tracks) == {"quality", "speed"} else tracks[0],
        ]
        if models:
            command.extend(["--models", ",".join(models)])
        if limit_episodes is not None:
            command.extend(["--limit-episodes", str(limit_episodes)])
        status = int(transport.foreground(command))
    except BaseException as exc:
        run_error = exc
    finally:
        if runner_env is not None:
            try:
                transport.run([runner_env, f"{root}/review.py", "--root", root], timeout=600)
            except Exception:
                # Raw episode evidence remains authoritative if offline gallery
                # generation is unavailable on a partial/interrupted run.
                print("warning: offline review packet generation failed", file=sys.stderr)
        try:
            collected = transport.run(["python3", "-c", _collect_script(), root], timeout=600)
            archive_path = output_dir / "node-results.tar.gz"
            archive_path.write_bytes(collected.stdout)
            _safe_extract(archive_path, output_dir)
            archive_path.unlink()
        except BaseException as exc:
            collect_error = exc
        finally:
            try:
                transport.run(["rm", "-f", key_path], timeout=30)
            except Exception:
                print("warning: node key cleanup failed", file=sys.stderr)
    if run_error is not None:
        raise run_error
    if collect_error is not None:
        raise collect_error
    return status


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod", default=DEFAULT_POD)
    parser.add_argument("--benchmark-root", type=Path, default=ROOT)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--prompt", type=Path)
    parser.add_argument("--node-root", default=DEFAULT_NODE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results-ai-gateway-20260923")
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--known-hosts", type=Path)
    parser.add_argument("--api-key-file", type=Path, help="local protected plaintext key or JSON {default,models} bundle")
    parser.add_argument("--models", nargs="+", help="catalogued model ids; defaults to all prepared models")
    parser.add_argument("--tracks", nargs="+", choices=("quality", "speed", "screen"), default=("quality", "speed"))
    parser.add_argument("--track", nargs="+", choices=("quality", "speed", "screen", "all"), dest="track_alias",
                        help="alias for --tracks; use 'all' for quality and speed")
    parser.add_argument("--limit-episodes", type=int, help="run only the first N deterministic episodes")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.track_alias is not None:
            if args.tracks != ("quality", "speed"):
                raise BenchmarkLaunchError("use only one of --track and --tracks")
            tracks = ("quality", "speed") if "all" in args.track_alias else tuple(args.track_alias)
        else:
            tracks = tuple(args.tracks)
        if set(tracks) not in ({"quality"}, {"speed"}, {"screen"}, {"quality", "speed"}):
            raise BenchmarkLaunchError("run screen separately; multi-track launch supports quality+speed only")
        if args.limit_episodes is not None and args.limit_episodes <= 0:
            raise BenchmarkLaunchError("--limit-episodes must be positive")
        plan = build_package_plan(
            benchmark_root=args.benchmark_root,
            runner=args.runner,
            references=args.references,
            config=args.config,
            prompt=args.prompt,
            models=args.models,
            tracks=tracks,
            allow_unknown_models=not args.dry_run,
        )
        effective_node_root = _effective_node_root(plan, args.node_root)
        effective_output_dir = args.output_dir
        if effective_node_root != args.node_root.rstrip("/"):
            effective_output_dir = args.output_dir / Path(effective_node_root).name
        if args.dry_run:
            print(json.dumps(_plan_json(plan, pod=args.pod, node_root=effective_node_root), indent=2, sort_keys=True))
            return 0
        if args.identity is not None and not args.identity.is_file():
            raise BenchmarkLaunchError("--identity must be a local key file")
        if args.known_hosts is not None and not args.known_hosts.is_file():
            raise BenchmarkLaunchError("--known-hosts must be a local file")
        from painter.rl_remote import discover_node

        return run_benchmark(
            plan=plan,
            pod=args.pod,
            node_root=effective_node_root,
            output_dir=effective_output_dir,
            identity=args.identity,
            known_hosts=args.known_hosts,
            discover=discover_node,
            api_key=_api_key_bundle(args.api_key_file),
            models=args.models,
            tracks=tracks,
            limit_episodes=args.limit_episodes,
        )
    except (BenchmarkLaunchError, OSError, tarfile.TarError, ValueError) as exc:
        print(f"benchmark launch error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
