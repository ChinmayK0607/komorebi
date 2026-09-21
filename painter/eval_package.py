"""Build the small, explicit input bundle for a controlled eval node.

The archive contains active source, the pinned renderer, the immutable eval
manifest, and exactly the twelve reference PNGs selected by that manifest. It
never contains model weights, HF caches, credentials, logs, provider state,
or a run-local copied harness. Model bytes are downloaded on the node by
``eval_node.sh`` and verified by ``eval_model_receipt.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "runs/controlled-comparison-20260921"
SOURCE_ALLOWLIST = (
    "CONTRACT.md",
    "README.md",
    "contract.py",
    "client.py",
    "demonstrations.py",
    "rollout.py",
    "render_demos.py",
    "recovery_data.py",
    "compile_teacher.cjs",
    "pairwise.py",
    "reward.py",
    "foundation_lock.py",
    "foundation-lock.json",
    "hardware.py",
    "pyproject.toml",
    "setup_node.sh",
    "eval_node.sh",
    "eval_serve.sh",
    "eval_model_receipt.py",
    "native_painting_eval.py",
    "native_painting.py",
    "native_painting_renderer.py",
    # Keep one flat-discoverable, dependency-light contract test in the node
    # bundle.  setup_node.sh treats unittest's no-tests exit code as failure;
    # this also catches an incomplete source transfer before model serving.
    "test_contract.py",
    "runs/controlled-comparison-20260921/EVAL_SCOPE.md",
    "runs/controlled-comparison-20260921/EXPERIMENT.json",
    "runs/controlled-comparison-20260921/PROTOCOL.md",
    "runs/controlled-comparison-20260921/eval-manifest.json",
)
VENDOR_SUFFIXES = {".py", ".js", ".html", ".txt", ".md"}
DENY_PARTS = {
    ".git",
    "__pycache__",
    "cache",
    "private",
    "bootstrap",
    "renderer-env",
    "inference-env",
    "runtime",
    "logs",
    "outputs",
    "evaluation-rollouts",
}
DENY_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".tar", ".gz", ".zip"}
DENY_NAMES = {"env.sh", "model-path.txt", "sft-adapter-path.txt", "rl-adapter-path.txt"}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(path: Path) -> bool:
    return (
        path.is_file()
        and not path.is_symlink()
        and not set(path.parts) & DENY_PARTS
        and path.name not in DENY_NAMES
        and path.suffix.lower() not in DENY_SUFFIXES
        and not any(word in path.name.lower() for word in ("token", "secret", "credential", "private-key"))
    )


def _load_manifest(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = root / "runs/controlled-comparison-20260921/eval-manifest.json"
    raw = json.loads(path.read_text())
    cases = raw.get("cases")
    if raw.get("schema_version") != "controlled-comparison-eval-v1" or not isinstance(cases, list):
        raise ValueError("controlled comparison manifest has an unexpected schema")
    if len(cases) != 12:
        raise ValueError(f"expected 12 eval cases, found {len(cases)}")
    ids = [case.get("id") for case in cases]
    groups = [case.get("source_group") for case in cases]
    families = [case.get("family") for case in cases]
    if len(set(ids)) != 12 or len(set(groups)) != 12 or len(set(families)) != 12:
        raise ValueError("eval manifest must have unique id, source_group, and family values")
    if any(not str(case["id"]).endswith("-007") for case in cases):
        raise ValueError("eval manifest contains a non-suffix-007 case")
    for case in cases:
        rel = case.get("reference_png")
        expected = case.get("sha256", {}).get("png")
        if not isinstance(rel, str) or not isinstance(expected, str):
            raise ValueError(f"case {case.get('id')} lacks reference path/hash")
        source = (path.parent / rel).resolve()
        if not source.is_file() or _sha(source) != expected:
            raise ValueError(f"reference integrity check failed for {case['id']}: {source}")
    return raw, cases


def _derived_task_manifest(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "painter.native-eval-tasks.v1",
        "source_manifest_schema": "controlled-comparison-eval-v1",
        "protocol": {
            "max_turns": 6,
            "max_tokens_per_turn": 8192,
            "context_length": 32768,
            "temperature": 1.0,
            "max_images_per_prompt": 14,
        },
        "cases": [
            {
                "id": case["id"],
                "source_group": case["source_group"],
                "family": case["family"],
                "reference": f"eval-assets/{case['id']}.png",
                "max_turns": 6,
            }
            for case in cases
        ],
    }


def _add_file(files: dict[str, Path], name: str, path: Path) -> None:
    if not _safe(path):
        raise ValueError(f"unsafe bundle source: {path}")
    if Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError(f"unsafe archive name: {name}")
    if name in files:
        raise ValueError(f"duplicate bundle member: {name}")
    files[name] = path


def build(root: Path = ROOT, output: Path | None = None, includes: tuple[str, ...] = ()) -> dict[str, Any]:
    root = root.resolve()
    raw, cases = _load_manifest(root)
    files: dict[str, Path] = {}
    required = list(SOURCE_ALLOWLIST) + list(includes)
    for relative in required:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"required allowlisted source is missing: {relative}")
        _add_file(files, relative, path)
    vendor = root / "vendor/integrations/watercolour"
    for path in sorted(vendor.rglob("*")):
        if path.is_file() and path.suffix.lower() in VENDOR_SUFFIXES:
            _add_file(files, path.relative_to(root).as_posix(), path)

    with tempfile.TemporaryDirectory(prefix="painter-eval-bundle-") as temporary:
        temp = Path(temporary)
        task_manifest = temp / "native-eval-tasks.json"
        task_manifest.write_text(json.dumps(_derived_task_manifest(cases), indent=2, sort_keys=True) + "\n")
        _add_file(files, "native-eval-tasks.json", task_manifest)
        provenance = temp / "eval-provenance.json"
        provenance.write_text(
            json.dumps(
                {
                    "schema": "painter.controlled-comparison-eval-provenance.v1",
                    "source_manifest_sha256": _sha(root / "runs/controlled-comparison-20260921/eval-manifest.json"),
                    "source_manifest": "runs/controlled-comparison-20260921/eval-manifest.json",
                    "cases": [case["id"] for case in cases],
                    "assets_copied": 12,
                    "model_weights_included": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        _add_file(files, "eval-provenance.json", provenance)
        assets: dict[str, Path] = {}
        manifest_path = root / "runs/controlled-comparison-20260921/eval-manifest.json"
        for case in cases:
            source = (manifest_path.parent / case["reference_png"]).resolve()
            target = temp / f"{case['id']}.png"
            shutil.copyfile(source, target)
            assets[f"eval-assets/{case['id']}.png"] = target
        files.update(assets)

        if output is None:
            output = root / "eval-node-inputs.tar.gz"
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        member_manifest = {
            name: {"bytes": path.stat().st_size, "sha256": _sha(path)}
            for name, path in sorted(files.items())
        }
        manifest_file = temp / "bundle-file-manifest.json"
        manifest_file.write_text(json.dumps(member_manifest, indent=2, sort_keys=True) + "\n")
        with tarfile.open(output, "w:gz") as archive:
            for name, path in sorted(files.items()):
                archive.add(path, arcname=name, recursive=False)
            archive.add(manifest_file, arcname="bundle-file-manifest.json", recursive=False)
        receipt = {
            "schema": "painter.controlled-comparison-eval-bundle.v1",
            "path": output.name,
            "bytes": output.stat().st_size,
            "sha256": _sha(output),
            "files": member_manifest,
            "includes_weights": False,
            "includes_credentials": False,
        }
        output.with_suffix(output.suffix + ".manifest.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include", action="append", default=[], help="additional relative source path; repeatable")
    args = parser.parse_args()
    receipt = build(args.root, args.output, tuple(args.include))
    print(json.dumps({key: value for key, value in receipt.items() if key != "files"}, sort_keys=True))


if __name__ == "__main__":
    main()
