"""Build the aesthetic RL overlay or a fresh full launch bundle.

The first node bootstrap is deliberately produced by the reviewed historical
``package_async.py``.  This script packages the new taskset/judge/config/data
as a small overlay, so applying the overlay does not reinstall dependencies or
redownload the model.  ``--full`` is available for a self-contained replayable
bundle and flattens the old portable inputs in the same way as the historical
packager.

No credentials, logs, caches, model weights, optimizer state, live queues, or
archives are admitted to either bundle.  The current run's EXPERIMENT.json is
read and copied verbatim; this script never edits it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile
from typing import Iterable


ROOT = Path(__file__).resolve().parent
OLD_RUN = ROOT.parent / "revision-rl-20260918"
OLD_INPUTS = OLD_RUN / "inputs"

DENY_DIRS = {
    ".git",
    "__pycache__",
    "bootstrap",
    "browsers",
    "cache",
    "logs",
    "private",
    "runtime",
    "model",
    "models",
    "weights",
    "optimizer",
    "async-output",
    "rl-rollouts",
    "judge-queue",
    "training-handoffs",
}
DENY_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".tar", ".gz", ".zip"}
DENY_NAMES = {
    "test_quality_signal.py",
    "test_quality_signal_v2.py",
    "quality_signal.py",
    "quality_signal_v2.py",
    "taskset_impl.py",
    "reward_queue.py",
    "supervise_async.py",
    "run-input-manifest.json",
    "input-manifest.json",
    "pipeline.py",
    "sample_phase.sh",
}

# These are reviewed launch/bootstrap components from the historical run.
# Scoring/taskset modules are explicitly excluded above and are replaced by
# current-run files when present.
OLD_WRAPPERS = (
    "setup_async_node.sh",
    "activate_async.sh",
    "launch_async.sh",
    "prepare_async.py",
    "run_from_sft.py",
    "train_from_sft.py",
    "training_handoff.py",
    "deploy_async.py",
    "package_async.py",
    "setup_node.sh",
    "stage_model.py",
    "download_model.py",
    "hardware.py",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member(path: Path) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    if path.name in DENY_NAMES or path.suffix.lower() in DENY_SUFFIXES:
        return False
    if set(path.parts) & DENY_DIRS:
        return False
    return True


def _add_tree(files: dict[str, Path], tree: Path, *, prefix: str = "") -> None:
    if not tree.is_dir():
        raise ValueError(f"required input directory missing: {tree}")
    for path in tree.rglob("*"):
        if not _safe_member(path):
            continue
        relative = path.relative_to(tree).as_posix()
        name = f"{prefix.rstrip('/')}/{relative}" if prefix else relative
        if name in files:
            raise ValueError(f"duplicate bundle member: {name}")
        files[name] = path


def _add_current_overlay(files: dict[str, Path]) -> None:
    """Add current-run source/config/data, omitting generated live artifacts."""

    for path in ROOT.rglob("*"):
        # Current taskset/reward sources are allowed to use names that are
        # deliberately denied when traversing the historical run.  This is
        # what prevents an old quality_signal/test_quality_signal pair from
        # being silently reintroduced while still allowing the reviewed new
        # taskset to be named taskset_impl.py.
        if path.is_symlink() or not path.is_file():
            continue
        if path.suffix.lower() in DENY_SUFFIXES or set(path.parts) & DENY_DIRS:
            continue
        relative = path.relative_to(ROOT)
        # Full package output and historical bundle sidecars must never nest
        # inside a later package.  Anchor assets are intentional data inputs.
        if path.name.endswith(".manifest.json") or path.name.endswith(".tar.gz"):
            continue
        name = relative.as_posix()
        files[name] = path


def _add_old_inputs(files: dict[str, Path]) -> None:
    """Flatten the historical portable input archive into a fresh bundle."""

    for path in OLD_INPUTS.rglob("*"):
        if not _safe_member(path):
            continue
        relative = path.relative_to(OLD_INPUTS).as_posix()
        # The historical packager intentionally omitted these potentially
        # stale launch-only entries; package the reviewed portable inputs and
        # current wrappers instead.
        files[relative] = path


def _add_old_wrappers(files: dict[str, Path]) -> None:
    for name in OLD_WRAPPERS:
        path = OLD_RUN / name
        if path.is_file() and _safe_member(path):
            files[name] = path


def _load_experiment() -> dict:
    path = ROOT / "EXPERIMENT.json"
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid current EXPERIMENT.json: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("EXPERIMENT.json must be an object")
    run_id = value.get("restart_run_name") or value.get("run_name")
    if not isinstance(run_id, str) or not run_id or "/" in run_id or "\\" in run_id:
        raise ValueError("EXPERIMENT.json must provide a simple run name")
    return value


def _prepare_data() -> None:
    # Import only after the run directory exists; this remains a pure local
    # copy/manifest operation and never initializes rendering or ML runtime.
    from prepare_aesthetic_data import build

    build(ROOT)


def _manifest(mode: str, files: dict[str, Path], experiment: dict) -> dict:
    return {
        "schema": "painter.aesthetic-bundle.v1",
        "mode": mode,
        "run_name": experiment.get("restart_run_name") or experiment.get("run_name"),
        "source_baseline": experiment.get("source_baseline"),
        "files": {
            name: {"sha256": _sha(path), "bytes": path.stat().st_size}
            for name, path in sorted(files.items())
        },
    }


def _write_archive(target: Path, files: dict[str, Path], mode: str, experiment: dict) -> dict:
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    # The manifest is included in the archive for on-node verification but its
    # own bytes are intentionally absent from its file map to avoid a hash
    # cycle.  An identical receipt is written beside the archive.
    manifest = _manifest(mode, files, experiment)
    with tempfile.TemporaryDirectory(prefix="aesthetic-package-") as temp_name:
        manifest_path = Path(temp_name) / "bundle-file-manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        archive_files = dict(files)
        archive_files["bundle-file-manifest.json"] = manifest_path
        with tarfile.open(target, "w:gz") as archive:
            for name, path in sorted(archive_files.items()):
                if Path(name).is_absolute() or ".." in Path(name).parts:
                    raise ValueError(f"unsafe archive member: {name}")
                archive.add(path, arcname=name, recursive=False)
    receipt = {
        "schema": "painter.aesthetic-bundle-receipt.v1",
        "path": target.name,
        "mode": mode,
        "sha256": _sha(target),
        "bytes": target.stat().st_size,
        "files": manifest["files"],
    }
    target.with_suffix(target.suffix + ".manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def build(mode: str, output: Path | None = None, *, prepare_data: bool = True) -> dict:
    if mode not in {"overlay", "full"}:
        raise ValueError("mode must be overlay or full")
    experiment = _load_experiment()
    if prepare_data:
        _prepare_data()

    files: dict[str, Path] = {}
    if mode == "full":
        _add_old_inputs(files)
        _add_old_wrappers(files)
    _add_current_overlay(files)

    required = {
        "EXPERIMENT.json",
        "JUDGE_PROMPT.md",
        "aesthetic_judge.py",
        "judge_queue.py",
        "taskset_impl.py",
        "render_actions.py",
        "activate_async.sh",
        "launch_async.sh",
        "prepare_async.py",
        "aesthetic-states.json",
        "rl-states.json",
        "aesthetic-data-receipt.json",
    }
    missing = sorted(name for name in required if name not in files)
    if missing:
        raise ValueError("new aesthetic overlay is incomplete: " + ", ".join(missing))
    if mode == "overlay" and not any(name.endswith("taskset.py") or name == "taskset_impl.py" for name in files):
        # Sibling taskset implementations may use a package-local name; don't
        # guess it, but fail closed if no taskset source is present at all.
        candidates = [name for name in files if "taskset" in Path(name).stem]
        if not candidates:
            raise ValueError("overlay has no new taskset source")

    run_name = experiment.get("restart_run_name") or experiment.get("run_name")
    if output is None:
        output = ROOT / f"aesthetic-{mode}-{run_name}.tar.gz"
    receipt = _write_archive(Path(output), files, mode, experiment)
    return {key: value for key, value in receipt.items() if key != "files"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--overlay", action="store_true", help="package only current-run files (default)")
    group.add_argument("--full", action="store_true", help="package old portable inputs plus current overlays")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-prepare-data", action="store_true")
    args = parser.parse_args()
    mode = "full" if args.full else "overlay"
    print(json.dumps(build(mode, args.output, prepare_data=not args.no_prepare_data), sort_keys=True))


if __name__ == "__main__":
    main()
