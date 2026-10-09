#!/usr/bin/env python3
"""Prepare the staged run on a Linux node without starting a run.

This is intentionally a single foreground handoff.  It owns every path below
``--node-root`` and only admits public inputs after their pinned revision or
SHA-256 has been checked.  The final receipt says ``ready_no_launch``; there
is no launcher, scheduler, model generation, or judge call in this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from typing import Any, Sequence


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[2]
PRIME_REPO = "https://github.com/PrimeIntellect-ai/prime-rl.git"
PRIME_COMMIT = "26b3131d2716a4f8210b165df584b83b4bc54f61"
MODEL_REPO = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
POLICY_REPO = "CK0607/qwen3.8-27b-brush-painting"
POLICY_REVISION = "1227a2427a8ae16014d2fdca4d1adb6855d3fa1c"
POLICY_PREFIX = "rl-judge-matched-20261004-flash/step_12"
POLICY_SHA256 = "c70c47ec0d6d7172b3bc25c4f54178919cf7c079cd9652fff3bc76db9e1eca79"
FROZEN_REVISION = "adf687e5b321514eda7be6eab050434fdc8e44ff"
FROZEN_PREFIX = "photo-curriculum-sft-20260922-balanced/step_512"
FROZEN_SHA256 = "bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740"
CURRICULUM_SHA256 = "38ccfb8390730d311134b7fdddf4e4c89cd9605563d29d069e55f6b59818ea7b"
RENDERER_SHA256 = "dc6b9d837ed7be382784a6c84de1b64911506b17a19b499ce054c0c572e75794"
P5_BRUSH_SHA256 = "db5b0094cb223529a8c5fc3bbba7bf8f2d5b5f5a5c0a847062194feaf917d16b"
ARCHIVE_REVISION = "2de12b59875fbdd5cc0f206f79eed12c8ee9f43b"
ARCHIVE_URL = (
    "https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/resolve/"
    f"{ARCHIVE_REVISION}/runs/diverse-flow-rl-20261009/evidence.tar.gz"
)
ARCHIVE_SHA256 = "41f527d104e1e5fe52ef83369e3fd089ba153ce2e3112ae25d4da25f682881ef"
RUN_ID = "staged-family-balanced-20261010"
BOOTSTRAP_UV = "0.11.1"
HUB_VERSION = "1.30.0"
HF_XET_VERSION = "1.6.0"
PLAYWRIGHT_VERSION = "1.58.0"
PILLOW_VERSION = "12.3.0"
RENDERER_TIMEOUT = 90


class BootstrapError(RuntimeError):
    """A fail-closed preparation error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise BootstrapError(f"receipt is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BootstrapError(f"invalid JSON receipt: {path}") from exc
    if not isinstance(value, dict):
        raise BootstrapError(f"receipt is not an object: {path}")
    return value


def ensure_dir(path: Path, description: str, *, create: bool = True) -> Path:
    path = path.expanduser()
    if path.exists() and path.is_symlink():
        raise BootstrapError(f"{description} cannot be a symlink: {path}")
    if path.exists() and not path.is_dir():
        raise BootstrapError(f"{description} is not a directory: {path}")
    if create:
        path.mkdir(parents=True, exist_ok=True)
        os.chmod(path, 0o700)
    if not path.is_dir():
        raise BootstrapError(f"{description} is missing: {path}")
    return path.resolve(strict=True)


def ensure_regular(path: Path, description: str, *, executable: bool = False) -> Path:
    if path.is_symlink() or not path.is_file():
        raise BootstrapError(f"{description} is not a regular file: {path}")
    if executable and not os.access(path, os.X_OK):
        raise BootstrapError(f"{description} is not executable: {path}")
    return path.resolve(strict=True)


def ensure_runtime_executable(path: Path, description: str, runtime_root: Path) -> Path:
    """Validate a venv interpreter while preserving its symlink path.

    CPython venvs normally expose ``bin/python`` as a symlink to the base
    interpreter.  The link is accepted only when it lives below the expected
    venv, resolves to a regular executable, and its target is either inside
    that venv or below the interpreter home recorded in ``pyvenv.cfg``.  The
    returned path remains the venv link so Python keeps the venv prefix.
    """

    path = path.expanduser().absolute()
    runtime_root = runtime_root.expanduser().absolute()
    try:
        path.relative_to(runtime_root)
    except ValueError as exc:
        raise BootstrapError(f"{description} is outside its runtime root: {path}") from exc
    if not path.exists() and not path.is_symlink():
        raise BootstrapError(f"{description} is missing: {path}")
    target = path.resolve(strict=True)
    if not target.is_file() or not os.access(target, os.X_OK):
        raise BootstrapError(f"{description} does not resolve to an executable file: {path}")
    if path.is_symlink():
        allowed_roots = [runtime_root.resolve(strict=True)]
        allowed_targets: list[Path] = []
        pyvenv = runtime_root / "pyvenv.cfg"
        if pyvenv.is_file() and not pyvenv.is_symlink():
            for line in pyvenv.read_text(encoding="utf-8").splitlines():
                if line.strip().lower().startswith("home ="):
                    home = Path(line.split("=", 1)[1].strip()).expanduser()
                    if home.is_absolute():
                        allowed_roots.append(home.resolve(strict=True))
                elif line.strip().lower().startswith("executable ="):
                    executable = Path(line.split("=", 1)[1].strip()).expanduser()
                    if executable.is_absolute():
                        allowed_targets.append(executable.resolve(strict=True))
        if target not in allowed_targets and not any(target == root or root in target.parents for root in allowed_roots):
            raise BootstrapError(f"{description} resolves outside its trusted interpreter roots: {target}")
        current = target
        while True:
            try:
                mode = current.stat().st_mode
            except OSError as exc:
                raise BootstrapError(f"cannot inspect {description} target: {current}") from exc
            if mode & 0o022:
                raise BootstrapError(f"{description} target is writable by group/other: {current}")
            if current.parent == current:
                break
            current = current.parent
    return path


def validate_topology(*, gpus_per_node: int, num_train_gpus: int, num_infer_gpus: int, visible_gpus: int) -> None:
    if (num_train_gpus, num_infer_gpus) not in {(1, 1), (2, 2)}:
        raise ValueError("staged topology must be exactly 1+1 or 2+2 GPUs")
    if gpus_per_node != num_train_gpus + num_infer_gpus:
        raise ValueError("gpus-per-node must equal train plus inference GPUs")
    if visible_gpus < gpus_per_node or visible_gpus < 2:
        raise ValueError(f"nvidia-smi exposes {visible_gpus} GPUs; {gpus_per_node} are required")


def visible_gpu_count(*, runner: Any | None = None) -> int:
    if platform.system() != "Linux":
        raise BootstrapError(f"staged bootstrap requires Linux; found {platform.system()}")
    if shutil.which("nvidia-smi") is None:
        raise BootstrapError("nvidia-smi is required for a staged node bootstrap")
    if runner is None:
        runner = subprocess.run
    try:
        result = runner(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise BootstrapError("nvidia-smi could not enumerate GPUs") from exc
    return len([line for line in result.stdout.splitlines() if line.strip()])


def run_checked(command: Sequence[str], *, env: dict[str, str] | None = None, cwd: Path | None = None, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    """Run one foreground command without a shell or background process."""

    try:
        return subprocess.run(
            list(map(str, command)), cwd=str(cwd) if cwd else None, env=env,
            check=True, capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        command_name = Path(str(command[0])).name if command else "command"
        raise BootstrapError(f"{command_name} failed during staged preparation") from exc


def ensure_bootstrap_env(node_root: Path) -> tuple[Path, Path]:
    root = ensure_dir(node_root / "bootstrap", "bootstrap environment")
    python = root / "bin/python"
    uv = root / "bin/uv"
    if not python.exists():
        run_checked([sys.executable, "-m", "venv", str(root)])
    python = ensure_runtime_executable(python, "bootstrap Python", root)
    if not uv.exists():
        run_checked([str(python), "-m", "pip", "install", f"uv=={BOOTSTRAP_UV}"])
    uv = ensure_regular(uv, "pinned uv", executable=True)
    deps_receipt = root / "bootstrap-deps-receipt.json"
    if deps_receipt.is_file():
        deps = read_json(deps_receipt)
        if deps.get("uv") != BOOTSTRAP_UV or deps.get("huggingface_hub") != HUB_VERSION or deps.get("hf_xet") != HF_XET_VERSION:
            raise BootstrapError("bootstrap dependency receipt is not pinned")
    elif deps_receipt.exists():
        raise BootstrapError("partial bootstrap dependency receipt exists")
    else:
        run_checked([str(uv), "pip", "install", "--python", str(python), f"huggingface_hub=={HUB_VERSION}", f"hf_xet=={HF_XET_VERSION}"], timeout=1800)
        atomic_json(deps_receipt, {"schema": "staged-family-balanced-bootstrap-deps-v1", "uv": BOOTSTRAP_UV, "huggingface_hub": HUB_VERSION, "hf_xet": HF_XET_VERSION})
    return python, uv


def ensure_prime_checkout(prime_root: Path) -> Path:
    prime_root = prime_root.expanduser()
    if prime_root.exists() and prime_root.is_symlink():
        raise BootstrapError("Prime root cannot be a symlink")
    if prime_root.exists() and not (prime_root / ".git").exists():
        raise BootstrapError(f"existing Prime path is not a Git checkout: {prime_root}")
    if not prime_root.exists():
        prime_root.parent.mkdir(parents=True, exist_ok=True)
        run_checked(["git", "clone", PRIME_REPO, str(prime_root)])
    try:
        current = run_checked(["git", "-C", str(prime_root), "rev-parse", "HEAD"], timeout=20).stdout.strip()
    except BootstrapError:
        raise
    if current != PRIME_COMMIT:
        dirty = run_checked(["git", "-C", str(prime_root), "status", "--porcelain"], timeout=20).stdout.strip()
        if dirty:
            raise BootstrapError("Prime checkout is dirty; refusing to change it")
        run_checked(["git", "-C", str(prime_root), "fetch", "origin", PRIME_COMMIT], timeout=600)
        run_checked(["git", "-C", str(prime_root), "checkout", "--detach", PRIME_COMMIT], timeout=60)
    actual = run_checked(["git", "-C", str(prime_root), "rev-parse", "HEAD"], timeout=20).stdout.strip()
    if actual != PRIME_COMMIT:
        raise BootstrapError("Prime checkout did not reach the pinned commit")
    return ensure_dir(prime_root, "Prime checkout")


def ensure_prime_runtime(*, run_root: Path, prime_root: Path, uv: Path) -> Path:
    """Use the reviewed dependency-only helper; it never starts Prime."""

    helper = REPO_ROOT / "painter/runs/photo-curriculum-rl-20260922/setup_rl_runtime.sh"
    ensure_regular(helper, "reviewed Prime runtime helper")
    runtime_root = ensure_dir(run_root / "prime-runtime", "Prime runtime receipt root")
    receipt = runtime_root / "rl-runtime-receipt.json"
    prime_python = prime_root / ".venv/bin/python"
    if receipt.is_file() and prime_python.is_file():
        value = read_json(receipt)
        if value.get("setup_complete") is True and value.get("prime_revision") == PRIME_COMMIT:
            return ensure_runtime_executable(prime_python, "Prime Python", prime_root / ".venv")
        raise BootstrapError("existing Prime runtime receipt is not the pinned runtime")
    if receipt.exists() or prime_python.exists():
        raise BootstrapError("partial Prime runtime is present; refusing to overwrite it")
    env = dict(os.environ)
    env.pop("PAINTER_RENDER_ROOT", None)
    env["PRIME_ROOT"] = str(prime_root)
    env["UV_BIN"] = str(uv)
    run_checked(["bash", str(helper), str(runtime_root)], env=env, timeout=3600)
    return ensure_runtime_executable(prime_python, "Prime Python", prime_root / ".venv")


def ensure_model(*, node_root: Path, bootstrap_python: Path) -> Path:
    model_parent = ensure_dir(node_root / "models", "model directory")
    model_dir = model_parent / MODEL_REVISION
    receipt_path = model_parent / "model-receipt.json"
    if receipt_path.is_file():
        receipt = read_json(receipt_path)
        if receipt.get("repo") != MODEL_REPO or receipt.get("revision") != MODEL_REVISION:
            raise BootstrapError("base model receipt is not the pinned public revision")
        ensure_regular(model_dir / "config.json", "base model config")
        return ensure_dir(model_dir, "base model")
    if model_dir.exists():
        raise BootstrapError("partial base model directory is present; refusing to overwrite it")
    temporary = Path(tempfile.mkdtemp(prefix=f".{MODEL_REVISION}.", dir=model_parent))
    shutil.rmtree(temporary)
    code = (
        "from huggingface_hub import HfApi, snapshot_download\n"
        "import pathlib, sys\n"
        f"repo={MODEL_REPO!r}; revision={MODEL_REVISION!r}; target=pathlib.Path(sys.argv[1])\n"
        "info=HfApi().model_info(repo, revision=revision)\n"
        "if info.sha != revision: raise RuntimeError('HF model revision mismatch')\n"
        "snapshot_download(repo, revision=revision, local_dir=str(target))\n"
    )
    env = dict(os.environ)
    env["HF_HOME"] = str(ensure_dir(node_root / "hf-cache", "Hugging Face cache"))
    try:
        run_checked([str(bootstrap_python), "-c", code, str(temporary)], env=env, timeout=7200)
        ensure_regular(temporary / "config.json", "downloaded base model config")
        temporary.rename(model_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    atomic_json(receipt_path, {
        "schema": "staged-family-balanced-base-model-receipt-v1",
        "repo": MODEL_REPO,
        "revision": MODEL_REVISION,
        "path": str(model_dir),
        "config_sha256": sha256_file(model_dir / "config.json"),
        "verified_public_revision": True,
    })
    return ensure_dir(model_dir, "base model")


def _verify_adapter_receipt(receipt_path: Path, expected_revision: str, expected_prefix: str, expected_sha: str) -> Path:
    receipt = read_json(receipt_path)
    if receipt.get("repo") != POLICY_REPO or receipt.get("revision") != expected_revision or receipt.get("prefix") != expected_prefix or receipt.get("adapter_sha256") != expected_sha or receipt.get("verified") is not True:
        raise BootstrapError(f"adapter receipt is not the pinned public artifact: {receipt_path}")
    adapter = ensure_dir(Path(str(receipt.get("adapter_path", ""))), "downloaded adapter")
    weights = ensure_regular(adapter / "adapter_model.safetensors", "adapter weights")
    ensure_regular(adapter / "adapter_config.json", "adapter config")
    if sha256_file(weights) != expected_sha:
        raise BootstrapError(f"adapter weights changed after download: {adapter}")
    return adapter


def ensure_adapter(*, node_root: Path, bootstrap_python: Path, revision: str, prefix: str, expected_sha: str, name: str) -> Path:
    receipt_path = node_root / f"{name}-adapter-receipt.json"
    if receipt_path.is_file():
        return _verify_adapter_receipt(receipt_path, revision, prefix, expected_sha)
    if receipt_path.exists():
        raise BootstrapError(f"partial adapter receipt exists: {receipt_path}")
    helper = REPO_ROOT / "painter/runs/text-pairwise-rl-20260928/download_public_adapter.py"
    ensure_regular(helper, "reviewed public adapter downloader")
    temporary_receipt = receipt_path.with_suffix(".tmp.json")
    if temporary_receipt.exists():
        raise BootstrapError(f"partial adapter receipt exists: {temporary_receipt}")
    env = dict(os.environ)
    env["HF_HOME"] = str(ensure_dir(node_root / "hf-cache", "Hugging Face cache"))
    run_checked([
        str(bootstrap_python), str(helper), "--revision", revision, "--prefix", prefix,
        "--expected-sha256", expected_sha, "--receipt", str(temporary_receipt),
    ], env=env, timeout=7200)
    temporary_receipt.replace(receipt_path)
    return _verify_adapter_receipt(receipt_path, revision, prefix, expected_sha)


def download_file(url: str, destination: Path, expected_sha: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise BootstrapError(f"download destination cannot be a symlink: {destination}")
    if destination.is_file():
        if sha256_file(destination) != expected_sha:
            raise BootstrapError(f"existing download has the wrong SHA-256: {destination}")
        return destination
    if destination.exists():
        raise BootstrapError(f"download destination is not a regular file: {destination}")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.part")
    if temporary.exists():
        raise BootstrapError(f"partial download exists: {temporary}")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                digest.update(block)
                stream.write(block)
        if digest.hexdigest() != expected_sha:
            raise BootstrapError("public archive SHA-256 did not match the pinned digest")
        os.chmod(temporary, 0o600)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _safe_member_name(name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or "" in path.parts:
        raise BootstrapError(f"unsafe archive member path: {name!r}")
    return path


def safe_extract_archive(archive: Path, destination: Path, *, max_bytes: int = 2 * 1024 * 1024 * 1024) -> dict[str, Any]:
    """Extract only regular allowlisted evidence files into a new directory."""

    if destination.exists():
        raise BootstrapError(f"archive extraction destination already exists: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    allowed_exact = {
        "curriculum/pilot-train-manifest.json", "curriculum/matched32-manifest.json",
        "curriculum/train-proposal.json", "curriculum/heldout-proposal.json",
    }
    allowed_prefixes = (
        "curriculum/prompts/", "curriculum/anchors/",
        "curriculum/.materialized-pilot-swf3eehs/casebriefs/",
    )
    seen: set[str] = set()
    total = 0
    try:
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle.getmembers():
                path = _safe_member_name(member.name)
                name = path.as_posix()
                if name not in allowed_exact and not name.startswith(allowed_prefixes):
                    continue
                if name in seen:
                    raise BootstrapError(f"duplicate archive member: {name}")
                seen.add(name)
                if not member.isreg() or member.issym() or member.islnk():
                    raise BootstrapError(f"archive member is not a regular file: {name}")
                total += member.size
                if total > max_bytes:
                    raise BootstrapError("archive extraction exceeds the size limit")
                target = temporary / path
                target.parent.mkdir(parents=True, exist_ok=True)
                os.chmod(target.parent, 0o700)
                source = bundle.extractfile(member)
                if source is None:
                    raise BootstrapError(f"cannot read archive member: {name}")
                with target.open("xb") as stream:
                    shutil.copyfileobj(source, stream, length=1024 * 1024)
                os.chmod(target, 0o600)
        for required in ("curriculum/pilot-train-manifest.json", "curriculum/matched32-manifest.json"):
            ensure_regular(temporary / required, "extracted prior native manifest")
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {"path": str(destination), "files": len(seen), "bytes": total}


def ensure_source_assets(*, node_root: Path, bootstrap_python: Path, source_root: Path) -> tuple[Path, dict[str, Any]]:
    source_root = ensure_dir(source_root, "staged source root", create=False)
    curriculum = ensure_regular(source_root / "curriculum-manifest.json", "staged curriculum manifest")
    if sha256_file(curriculum) != CURRICULUM_SHA256:
        raise BootstrapError("staged curriculum manifest does not match its pinned hash")
    source = ensure_dir(node_root / "source-assets", "source asset staging")
    archive = source / "diverse-flow-evidence.tar.gz"
    extraction = source / "extracted"
    extraction_receipt = source / "extraction-receipt.json"
    if extraction_receipt.is_file():
        receipt = read_json(extraction_receipt)
        if receipt.get("archive_sha256") != ARCHIVE_SHA256:
            raise BootstrapError("source extraction receipt has the wrong archive hash")
        ensure_dir(extraction, "extracted source assets", create=False)
        ensure_regular(extraction / "curriculum/pilot-train-manifest.json", "extracted train manifest")
        ensure_regular(extraction / "curriculum/matched32-manifest.json", "extracted heldout manifest")
        return extraction / "curriculum", receipt
    if extraction.exists() or extraction_receipt.exists():
        raise BootstrapError("partial source asset extraction is present")
    download_file(ARCHIVE_URL, archive, ARCHIVE_SHA256)
    extracted = safe_extract_archive(archive, extraction)
    receipt = {"schema": "staged-family-balanced-source-assets-v1", "archive_url": ARCHIVE_URL, "archive_revision": ARCHIVE_REVISION, "archive_sha256": ARCHIVE_SHA256, **extracted}
    atomic_json(extraction_receipt, receipt)
    return extraction / "curriculum", receipt


def ensure_renderer(*, node_root: Path, bootstrap_python: Path, uv: Path, prime_root: Path) -> tuple[Path, Path, Path, dict[str, Any]]:
    """Build the reviewed centroid overlay and run its sandboxed runtime helper."""

    render_root = ensure_dir(node_root / "renderer", "renderer runtime")
    env_root = render_root / "renderer-env"
    renderer_python = env_root / "bin/python"
    browser_root = ensure_dir(render_root / "browsers", "renderer browsers")
    bundle = render_root / "cancellable-runtime"
    pin_receipt_path = render_root / "renderer-pin-receipt.json"
    helper_receipt_path = render_root / "renderer-runtime-receipt.json"
    patcher = REPO_ROOT / "painter/patch_brush_centroid.py"
    helper = REPO_ROOT / "painter/runs/diverse-flow-rl-20261009/prepare_renderer_runtime.py"
    ensure_regular(patcher, "reviewed renderer patcher")
    ensure_regular(helper, "reviewed renderer runtime helper")
    if pin_receipt_path.is_file() and helper_receipt_path.is_file():
        pin = read_json(pin_receipt_path)
        runtime = read_json(helper_receipt_path)
        renderer = ensure_regular(Path(str(pin.get("renderer", ""))), "prepared renderer")
        asset = ensure_regular(Path(str(pin.get("p5_brush", ""))), "prepared p5.brush asset")
        renderer_python = ensure_runtime_executable(Path(str(pin.get("renderer_python", ""))), "renderer interpreter", env_root)
        if pin.get("renderer_sha256") != RENDERER_SHA256 or sha256_file(renderer) != RENDERER_SHA256 or pin.get("p5_brush_sha256") != P5_BRUSH_SHA256 or sha256_file(asset) != P5_BRUSH_SHA256:
            raise BootstrapError("prepared renderer or p5.brush asset changed after verification")
        if sha256_file(renderer_python) != pin.get("renderer_python_sha256"):
            raise BootstrapError("renderer interpreter changed after preparation")
        if runtime.get("sandbox_smoke") != "passed" or runtime.get("node_check") != "passed":
            raise BootstrapError("reviewed renderer helper did not record sandbox and Node checks")
        ensure_browser_user()
        expose_renderer_runtime(node_root, render_root, browser_root, renderer_python, bundle)
        return renderer, renderer_python, browser_root, {"pin": pin, "runtime_helper": runtime}
    if pin_receipt_path.exists() or helper_receipt_path.exists() or bundle.exists() or env_root.exists():
        raise BootstrapError("partial renderer runtime is present; refusing to overwrite it")
    run_checked([str(bootstrap_python), str(patcher), "--renderer-overlay", "--output-root", str(bundle)], timeout=600)
    renderer = ensure_regular(bundle / "cancellable_watercolour_renderer.py", "generated cancellable renderer")
    asset = ensure_regular(bundle / "assets/p5.brush.js", "generated p5.brush asset")
    if sha256_file(renderer) != RENDERER_SHA256 or sha256_file(asset) != P5_BRUSH_SHA256:
        raise BootstrapError("generated renderer overlay does not match the pinned hashes")
    # The reviewed helper's path contract is deliberately explicit.  Copy the
    # generated assets into this run-owned root instead of linking to a clone
    # or to an old run's renderer tree.
    vendor_assets = render_root / "vendor/integrations/watercolour/assets"
    vendor_assets.mkdir(parents=True, exist_ok=False)
    for source in sorted((bundle / "assets").iterdir()):
        if source.is_symlink() or not source.is_file():
            raise BootstrapError(f"unexpected generated renderer asset: {source}")
        shutil.copy2(source, vendor_assets / source.name)
    run_checked([str(uv), "venv", "--allow-existing", "--python", "python3", str(env_root)], timeout=600)
    run_checked([str(uv), "pip", "install", "--python", str(renderer_python), f"playwright=={PLAYWRIGHT_VERSION}", f"pillow=={PILLOW_VERSION}"], timeout=1800)
    env = dict(os.environ)
    env["PLAYWRIGHT_BROWSERS_PATH"] = str(browser_root)
    run_checked([str(renderer_python), "-m", "playwright", "install", "--with-deps", "chromium"], env=env, timeout=1800)
    ensure_browser_user()
    expose_renderer_runtime(node_root, render_root, browser_root, renderer_python, bundle)
    run_checked([str(bootstrap_python), str(helper), "--run-root", str(render_root), "--renderer-root", str(render_root), "--renderer-python", str(renderer_python), "--browsers", str(browser_root), "--prime-venv", str(prime_root / ".venv")], timeout=600)
    runtime = read_json(helper_receipt_path)
    if runtime.get("sandbox_smoke") != "passed" or runtime.get("node_check") != "passed":
        raise BootstrapError("reviewed renderer helper did not pass sandbox and Node checks")
    renderer_python = ensure_runtime_executable(renderer_python, "renderer interpreter", env_root)
    pin = {
        "schema": "staged-family-balanced-renderer-pin-v1",
        "renderer": str(renderer), "renderer_sha256": RENDERER_SHA256,
        "p5_brush": str(asset), "p5_brush_sha256": P5_BRUSH_SHA256,
        "renderer_python": str(renderer_python), "renderer_python_sha256": sha256_file(renderer_python),
        "playwright": PLAYWRIGHT_VERSION, "pillow": PILLOW_VERSION,
        "browsers": str(browser_root), "runtime_helper_receipt": str(helper_receipt_path),
        "prepared_no_render_process": True,
    }
    atomic_json(pin_receipt_path, pin)
    return renderer, renderer_python, browser_root, {"pin": pin, "runtime_helper": runtime}


def ensure_painter_install(*, prime_python: Path, uv: Path) -> None:
    # The marker belongs in the Prime checkout, so it is never tracked here.
    marker = prime_python.parent.parent / ".staged-painter-install-v1"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == str(REPO_ROOT):
        return
    run_checked([str(uv), "pip", "install", "--python", str(prime_python), "--no-deps", "--no-build-isolation", "-e", str(REPO_ROOT / "painter")], timeout=1800)
    marker.write_text(str(REPO_ROOT) + "\n", encoding="utf-8")
    os.chmod(marker, 0o600)


def ensure_browser_user() -> None:
    """Provide the reviewed unprivileged Chromium account on a root node."""

    if os.geteuid() != 0 or shutil.which("runuser") is None:
        return
    if subprocess.run(["id", "-u", "painter"], capture_output=True, text=True, check=False).returncode != 0:
        run_checked(["useradd", "--create-home", "--shell", "/bin/bash", "painter"], timeout=60)


def expose_renderer_runtime(node_root: Path, render_root: Path, browser_root: Path, renderer_python: Path, bundle: Path) -> None:
    """Expose only non-secret renderer inputs to the unprivileged browser user."""

    # Private model, adapter, source, and Prime directories remain mode 700.
    # Chromium needs traversal/read access to this generated renderer bundle;
    # no credential or provider state is stored below it.
    os.chmod(node_root, 0o711)
    os.chmod(render_root, 0o755)
    os.chmod(browser_root, 0o755)
    for directory in (bundle, bundle / "assets", render_root / "vendor", render_root / "vendor/integrations", render_root / "vendor/integrations/watercolour", render_root / "vendor/integrations/watercolour/assets", render_root / "renderer-env"):
        if directory.is_dir():
            os.chmod(directory, 0o755)
    for path in (bundle / "cancellable_watercolour_renderer.py", bundle / "frozen_watercolour_renderer.py", *(bundle / "assets").glob("*"), *(render_root / "vendor/integrations/watercolour/assets").glob("*")):
        if path.is_file() and not path.is_symlink():
            os.chmod(path, 0o644)
    if not renderer_python.is_symlink():
        os.chmod(renderer_python, 0o755)


def ensure_materialized(*, node_root: Path, source_assets: Path, source_root: Path, bootstrap_python: Path) -> Path:
    output = node_root / "materialized"
    receipt = output / "materialization-receipt.json"
    if receipt.is_file():
        ensure_dir(output, "materialized assets")
        value = read_json(receipt)
        if value.get("counts", {}).get("train_rows") != 96 or value.get("counts", {}).get("heldout_rows") != 32:
            raise BootstrapError("existing materialized receipt does not contain 96/32 rows")
        return output
    if output.exists():
        raise BootstrapError("partial materialized output is present")
    run_checked([
        str(bootstrap_python), str(RUN_ROOT / "materialize.py"),
        "--train-manifest", str(source_assets / "pilot-train-manifest.json"),
        "--heldout-manifest", str(source_assets / "matched32-manifest.json"),
        "--asset-root", str(source_assets), "--output-root", str(output),
        "--curriculum-manifest", str(source_root / "curriculum-manifest.json"),
    ], timeout=1800)
    ensure_regular(receipt, "materialization receipt")
    return output


def ensure_prepared(*, node_root: Path, source_root: Path, materialized: Path, prime_root: Path, model: Path, policy: Path, frozen: Path, renderer: Path, renderer_python: Path, gpus_per_node: int, num_train_gpus: int, num_infer_gpus: int, bootstrap_python: Path) -> Path:
    preflight = node_root / "config-preflight"
    config_path = preflight / "native.toml"
    config_receipt = preflight / "config-receipt.json"
    if not config_path.exists():
        run_checked([
            str(bootstrap_python), str(RUN_ROOT / "prepare_config.py"), "--output", str(config_path), "--receipt-output", str(config_receipt), "--node-root", str(preflight), "--materialized-root", str(materialized), "--repo-root", str(REPO_ROOT), "--model", str(model), "--renderer", str(renderer), "--renderer-python", str(renderer_python), "--gpus-per-node", str(gpus_per_node), "--num-train-gpus", str(num_train_gpus), "--num-infer-gpus", str(num_infer_gpus),
        ], timeout=600)
    elif not config_receipt.is_file():
        raise BootstrapError("partial config preflight exists")
    prepared = node_root / "prepared"
    prepare_receipt = prepared / "prepare-receipt.json"
    if not prepare_receipt.is_file():
        if prepared.exists():
            raise BootstrapError("partial prepared node directory exists")
        run_checked([
            str(bootstrap_python), str(RUN_ROOT / "prepare_node.py"), "--source-root", str(source_root), "--node-root", str(prepared), "--materialized-root", str(materialized), "--prime-root", str(prime_root), "--model-dir", str(model), "--policy-adapter", str(policy), "--frozen-adapter", str(frozen), "--renderer", str(renderer), "--renderer-python", str(renderer_python), "--repo-root", str(REPO_ROOT), "--gpus-per-node", str(gpus_per_node), "--num-train-gpus", str(num_train_gpus), "--num-infer-gpus", str(num_infer_gpus),
        ], timeout=600)
    else:
        ensure_dir(prepared, "prepared node")
    os.chmod(prepared, 0o700)
    return prepared


def ensure_canary(*, prepared: Path, source_root: Path, materialized: Path, bootstrap_python: Path) -> dict[str, Any]:
    receipt_path = prepared / "canary-receipt.json"
    if receipt_path.is_file():
        value = read_json(receipt_path)
        if value.get("status") != "passed_import_hook_and_renderer" or value.get("orchestrator_setup_called") is not False:
            raise BootstrapError("existing canary receipt is not a passing no-launch canary")
        return value
    env = dict(os.environ)
    renderer = read_json(prepared / "prepare-receipt.json").get("renderer", {})
    renderer_python = Path(str(renderer.get("python", ""))).resolve()
    browser_root = renderer_python.parents[2] / "browsers"
    env["PAINTER_BROWSER_ROOT"] = str(browser_root)
    run_checked([str(bootstrap_python), str(RUN_ROOT / "canary.py"), "--source-root", str(source_root), "--node-root", str(prepared), "--materialized-root", str(materialized), "--repo-root", str(REPO_ROOT)], env=env, timeout=600)
    value = read_json(receipt_path)
    if value.get("status") != "passed_import_hook_and_renderer":
        raise BootstrapError("canary did not produce the required import and renderer status")
    return value


def validate_ready(path: Path, *, topology: dict[str, int]) -> dict[str, Any]:
    value = read_json(path)
    if value.get("schema") != "staged-family-balanced-bootstrap-ready-v1" or value.get("status") != "ready_no_launch" or value.get("run_id") != RUN_ID:
        raise BootstrapError("existing bootstrap receipt is not a staged no-launch receipt")
    if value.get("topology") != topology:
        raise BootstrapError("existing bootstrap receipt has a different GPU topology")
    return value


def bootstrap(args: argparse.Namespace) -> dict[str, Any]:
    visible = visible_gpu_count()
    validate_topology(gpus_per_node=args.gpus_per_node, num_train_gpus=args.num_train_gpus, num_infer_gpus=args.num_infer_gpus, visible_gpus=visible)
    node_root = ensure_dir(args.node_root, "node root")
    for name in ("private", "logs", "staging"):
        ensure_dir(node_root / name, f"protected node {name}")
    ready_path = node_root / "bootstrap-ready.json"
    topology = {"gpus_per_node": args.gpus_per_node, "num_train_gpus": args.num_train_gpus, "num_infer_gpus": args.num_infer_gpus}
    if ready_path.is_file():
        return validate_ready(ready_path, topology=topology)
    if ready_path.exists():
        raise BootstrapError("partial bootstrap receipt exists")
    bootstrap_python, uv = ensure_bootstrap_env(node_root)
    prime_root = ensure_prime_checkout(args.prime_root or node_root / "prime")
    prime_python = ensure_prime_runtime(run_root=node_root, prime_root=prime_root, uv=uv)
    ensure_painter_install(prime_python=prime_python, uv=uv)
    model = ensure_model(node_root=node_root, bootstrap_python=bootstrap_python)
    policy = ensure_adapter(node_root=node_root, bootstrap_python=bootstrap_python, revision=POLICY_REVISION, prefix=POLICY_PREFIX, expected_sha=POLICY_SHA256, name="policy")
    frozen = ensure_adapter(node_root=node_root, bootstrap_python=bootstrap_python, revision=FROZEN_REVISION, prefix=FROZEN_PREFIX, expected_sha=FROZEN_SHA256, name="frozen")
    renderer, renderer_python, browser_root, renderer_receipt = ensure_renderer(node_root=node_root, bootstrap_python=bootstrap_python, uv=uv, prime_root=prime_root)
    source_assets, source_receipt = ensure_source_assets(node_root=node_root, bootstrap_python=bootstrap_python, source_root=args.source_root)
    materialized = ensure_materialized(node_root=node_root, source_assets=source_assets, source_root=args.source_root, bootstrap_python=bootstrap_python)
    prepared = ensure_prepared(node_root=node_root, source_root=args.source_root, materialized=materialized, prime_root=prime_root, model=model, policy=policy, frozen=frozen, renderer=renderer, renderer_python=renderer_python, gpus_per_node=args.gpus_per_node, num_train_gpus=args.num_train_gpus, num_infer_gpus=args.num_infer_gpus, bootstrap_python=bootstrap_python)
    canary = ensure_canary(prepared=prepared, source_root=args.source_root, materialized=materialized, bootstrap_python=bootstrap_python)
    ready = {
        "schema": "staged-family-balanced-bootstrap-ready-v1", "status": "ready_no_launch", "run_id": RUN_ID,
        "node_root": str(node_root), "repo_root": str(REPO_ROOT), "source_root": str(args.source_root),
        "brush_root": str(REPO_ROOT / "painter"),
        "topology": topology, "visible_gpu_count": visible, "training_started": False, "model_generation": False, "judge_calls": False, "background_processes_left_running": False, "credentials_embedded": False,
        "prime": {"root": str(prime_root), "revision": PRIME_COMMIT, "python": str(prime_python)},
        "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION, "path": str(model), "config_sha256": sha256_file(model / "config.json")},
        "adapters": {"policy": {"repo": POLICY_REPO, "revision": POLICY_REVISION, "prefix": POLICY_PREFIX, "sha256": POLICY_SHA256, "path": str(policy)}, "frozen_sft": {"repo": POLICY_REPO, "revision": FROZEN_REVISION, "prefix": FROZEN_PREFIX, "sha256": FROZEN_SHA256, "path": str(frozen)}},
        "source_assets": source_receipt, "materialized_root": str(materialized), "prepared_root": str(prepared),
        "renderer": renderer_receipt, "canary": {"status": canary.get("status"), "orchestrator_setup_called": canary.get("orchestrator_setup_called")},
        "next_action": "run the separately reviewed finite launch command after human node review",
    }
    atomic_json(ready_path, ready)
    return ready


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-root", type=Path, required=True)
    parser.add_argument("--prime-root", type=Path)
    parser.add_argument("--source-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--gpus-per-node", type=int, required=True)
    parser.add_argument("--num-train-gpus", type=int, required=True)
    parser.add_argument("--num-infer-gpus", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        print(json.dumps(bootstrap(parse_args(argv)), indent=2, sort_keys=True))
    except (BootstrapError, ValueError) as exc:
        print(f"bootstrap failed closed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
