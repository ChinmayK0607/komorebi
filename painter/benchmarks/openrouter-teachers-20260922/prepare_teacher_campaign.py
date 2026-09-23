#!/usr/bin/env python3
"""Prepare a selected-teacher demonstration campaign without making API calls.

The benchmark's 40 references are an evaluation holdout.  This command therefore
requires a separate training-reference manifest, verifies every referenced image,
excludes and records SHA-256 overlap with ``refs.json``, and writes three independent quality
campaign roots.  The roots are ready to copy to a Linux node; this command never
reads credentials, invokes the provider, launches a node, or runs the benchmark.

The accepted input manifest is either a JSON object with ``references`` or
``rows``, a JSON list of entries, or JSONL.  An entry needs an id and an image
path.  Common source-pool field names (``reference``, ``source_path``,
``reference_sha256``, ``source_image_sha256``, ``family`` and ``source_split``)
are accepted so the checked-in RL source manifest can be used explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Any, Iterable, Mapping


HERE = Path(__file__).resolve().parent
DEFAULT_BENCHMARK_REFS = HERE / "refs.json"
DEFAULT_PROMPT = HERE / "prompt.txt"
DEFAULT_CONTRACT = HERE / "contract.json"
DEFAULT_CATALOG = HERE / "model-catalog.json"
STAGES = (("short-02", 2), ("medium-05", 5), ("long-12", 12))
SCHEMA = "painter.teacher-campaign-setup.v1"
HEX = set("0123456789abcdef")


class CampaignError(ValueError):
    """Raised when a campaign cannot be prepared safely."""


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return value[:120] or "teacher"


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def _sha(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) != 64 or any(c not in HEX for c in value.lower()):
        raise CampaignError(f"{label} must be a SHA-256 hex string when provided")
    return value.lower()


def _load_json_or_jsonl(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        rows: list[Any] = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise CampaignError(f"invalid training manifest JSONL at line {line_no}: {exc}") from exc
        return rows


def _manifest_entries(value: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        value = value.get("references", value.get("rows"))
    if not isinstance(value, list) or not value:
        raise CampaignError(
            f"training manifest {path} must contain a non-empty 'references' or 'rows' list; "
            "the benchmark refs.json is an evaluation holdout and cannot be used here"
        )
    result: list[dict[str, Any]] = []
    for index, entry in enumerate(value, 1):
        if not isinstance(entry, dict):
            raise CampaignError(f"training manifest entry {index} is not an object")
        result.append(dict(entry))
    return result


def _resolve_image(raw: str, manifest_path: Path, reference_root: Path | None) -> Path:
    original = Path(raw)
    candidates: list[Path] = []
    if original.is_absolute():
        candidates.append(original)
    else:
        candidates.append(manifest_path.parent / original)
        if reference_root is not None:
            candidates.append(reference_root / original)
            # Source manifests sometimes record a workspace-relative painter/ path.
            if original.parts and original.parts[0] == "painter":
                candidates.append(reference_root.joinpath(*original.parts[1:]))
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and not resolved.is_symlink():
            return resolved
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise CampaignError(f"training image is missing or symlinked: {raw!r}; checked {checked}")


def _entry_id(entry: Mapping[str, Any], index: int) -> str:
    value = entry.get("id") or entry.get("source_row_id") or entry.get("source_id")
    if not isinstance(value, str) or not value.strip():
        raise CampaignError(f"training manifest entry {index} has no non-empty id/source_row_id/source_id")
    return value.strip()


def _entry_path(entry: Mapping[str, Any], index: int) -> str:
    for key in ("image", "reference", "source_path"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise CampaignError(f"training manifest entry {index} has no image/reference/source_path")


def _complexity_detail(entry: Mapping[str, Any]) -> tuple[float, bool, str]:
    for key in ("difficulty", "complexity", "annotated_object_count", "object_count"):
        value = entry.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value), True, key
    classes = entry.get("annotation_classes") or entry.get("member_families") or []
    if isinstance(classes, list):
        return float(len(classes)), True, "annotation_class_count_proxy"
    return 0.0, False, "unknown"


def _complexity(entry: Mapping[str, Any]) -> float:
    return _complexity_detail(entry)[0]


def _family(entry: Mapping[str, Any]) -> str:
    for key in ("family", "category", "primary_family"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def _stratum(entry: Mapping[str, Any]) -> str:
    source_kind = entry.get("source_kind") or entry.get("reference_kind") or "unknown"
    return f"{source_kind}::{_family(entry)}"


def _split(entry: Mapping[str, Any]) -> str | None:
    for key in ("source_split", "split"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return None


def load_training_entries(path: Path, *, reference_root: Path | None, holdout_hashes: set[str], split: str) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]]]:
    path = path.resolve()
    if path == DEFAULT_BENCHMARK_REFS.resolve():
        raise CampaignError("the benchmark refs.json is an evaluation holdout; pass a separate training manifest")
    raw = _load_json_or_jsonl(path)
    source_sha = sha_file(path)
    entries = _manifest_entries(raw, path)
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    result: list[dict[str, Any]] = []
    excluded_holdout: list[dict[str, Any]] = []
    for index, source in enumerate(entries, 1):
        ident = _entry_id(source, index)
        if ident in seen_ids:
            raise CampaignError(f"duplicate training reference id: {ident}")
        seen_ids.add(ident)
        source_split = _split(source)
        if source_split in {"dev", "test", "val", "validation", "holdout", "eval", "evaluation"}:
            raise CampaignError(f"training manifest includes non-training split {source_split!r}: {ident}")
        if split:
            if source_split is None:
                raise CampaignError(f"training manifest entry {ident} has no split; refusing to assume it is train data")
            if source_split != split:
                continue
        image = _resolve_image(_entry_path(source, index), path, reference_root)
        actual_sha = sha_file(image)
        declared = _sha(
            source.get("sha256", source.get("reference_sha256", source.get("source_image_sha256"))),
            f"training reference {ident}.sha256",
        )
        if declared is not None and declared != actual_sha:
            raise CampaignError(f"training reference hash mismatch for {ident}: declared {declared}, found {actual_sha}")
        if actual_sha in holdout_hashes:
            excluded_holdout.append({"id": ident, "sha256": actual_sha, "source_split": source_split, "reason": "benchmark_holdout_sha256_overlap"})
            continue
        if actual_sha in seen_hashes:
            raise CampaignError(f"duplicate training image SHA-256 after filtering: {ident} ({actual_sha})")
        seen_hashes.add(actual_sha)
        row = dict(source)
        complexity, ranked, complexity_source = _complexity_detail(source)
        row.update({"id": ident, "image_path": image, "sha256": actual_sha, "family": _family(source), "stratum": _stratum(source), "complexity": complexity, "complexity_ranked": ranked, "complexity_source": complexity_source, "source_split": source_split})
        result.append(row)
    if not result:
        raise CampaignError(f"no training references remain after split={split!r} filtering")
    return result, source_sha, excluded_holdout


def _diverse_order(rows: Iterable[dict[str, Any]], *, reverse: bool = False) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("stratum") or row["family"]), []).append(row)
    for group in groups.values():
        group.sort(key=lambda row: (float(row["complexity"]), str(row["id"])), reverse=reverse)
    ordered: list[dict[str, Any]] = []
    while groups:
        for family in sorted(list(groups)):
            group = groups.get(family)
            if not group:
                groups.pop(family, None)
                continue
            ordered.append(group.pop(0))
    return ordered


def select_stage_rows(rows: list[dict[str, Any]], counts: Mapping[str, int]) -> dict[str, list[dict[str, Any]]]:
    total = sum(counts.values())
    if len(rows) < total:
        raise CampaignError(f"training manifest has {len(rows)} usable unique images, but stages request {total}")
    short_n, medium_n, long_n = counts["short"], counts["medium"], counts["long"]
    targets = (short_n, medium_n, long_n)
    known = sorted((row for row in rows if row.get("complexity_ranked") is True), key=lambda row: (float(row["complexity"]), str(row["id"])))
    unknown = sorted((row for row in rows if row.get("complexity_ranked") is not True), key=lambda row: (str(row.get("stratum")), str(row["id"])))
    bands: list[list[dict[str, Any]]] = [[], [], []]
    total = sum(targets)
    cumulative = (targets[0] / total, (targets[0] + targets[1]) / total)
    # Allocate known examples by quantiles across the *whole* source pool.
    # This makes later roots harder while remaining disjoint, even when the
    # requested counts are very different (for example 24/48/96).
    for index, row in enumerate(known):
        fraction = (index + 0.5) / max(1, len(known))
        band = 0 if fraction < cumulative[0] else 1 if fraction < cumulative[1] else 2
        bands[band].append(row)
    # Unknown-complexity examples are deliberately spread through all stages;
    # they are never presented as ranked, and their source-kind strata still
    # receive round-robin coverage.
    for index, row in enumerate(unknown):
        band = min(range(3), key=lambda candidate: (len(bands[candidate]) / max(1, targets[candidate]), candidate))
        bands[band].append(row)
    selected: dict[str, list[dict[str, Any]]] = {}
    used: set[str] = set()
    names = ("short", "medium", "long")
    for index, (name, target) in enumerate(zip(names, targets)):
        candidates = _diverse_order(bands[index])
        chosen = candidates[:target]
        if len(chosen) < target:
            # Rounding or an all-unknown manifest can leave a band short. Fill
            # from the nearest unused complexity values, preserving the hard
            # no-reuse invariant and recording the heuristic in the manifest.
            remainder = [row for row in rows if row["id"] not in used and row not in chosen]
            remainder.sort(key=lambda row: (float(row["complexity"]), str(row["id"])), reverse=index == 2)
            chosen.extend(_diverse_order(remainder)[: target - len(chosen)])
        if len(chosen) != target:
            raise CampaignError(f"could not select {target} references for {name} without reuse")
        selected[name] = chosen
        used.update(str(row["id"]) for row in chosen)
    return selected


def _copy_reference(row: Mapping[str, Any], stage_root: Path) -> dict[str, Any]:
    source = Path(row["image_path"])
    suffix = source.suffix.lower() or ".img"
    target = stage_root / "references" / f"{_safe_slug(str(row['id']))}{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    digest = sha_file(target)
    if digest != row["sha256"]:
        raise CampaignError(f"copied reference hash mismatch: {row['id']}")
    entry = {k: v for k, v in row.items() if k not in {"image_path", "complexity", "stratum", "complexity_ranked"}}
    entry.update({"image": f"references/{target.name}", "sha256": digest, "campaign_complexity": row["complexity"], "campaign_complexity_source": row["complexity_source"]})
    return entry


def _stage_config(model: str, stage: str, max_turns: int) -> dict[str, Any]:
    return {
        "benchmark": f"teacher-demonstrations-{_safe_slug(model)}-{stage}",
        "models": [model],
        "tracks": {"quality": {"max_turns": max_turns, "max_tokens": "native", "episode_timeout_seconds": None, "reasoning_effort": "highest_supported"}},
        "temperature": 1.0,
        "concurrency": 3,
        "render_concurrency": 1,
        "renderer_timeout": 180,
        "samples_per_image": 1,
        "catalog": "model-catalog.json",
        "transport_script": "gateway_transport.ts",
        "references": "refs.json",
        "prompt": "prompt.txt",
        "request_timeout_seconds": 3600,
        "max_retries": 2,
    }


def _write_stage(stage_root: Path, stage: str, max_turns: int, model: str, rows: list[dict[str, Any]], source_sha: str, holdout_sha: str, split: str) -> None:
    stage_root.mkdir(parents=True, exist_ok=True)
    references = [_copy_reference(row, stage_root) for row in rows]
    refs = {
        "version": 1,
        "references": references,
        "count": len(references),
        "selection": {"stage": stage, "max_turns": max_turns, "split": split, "source_manifest_sha256": source_sha, "benchmark_holdout_refs_sha256": holdout_sha},
        "training_admission": "external training manifest only; every selected image is hash checked and absent from benchmark refs.json",
        "ordering_rule": "disjoint ascending complexity bands with round-robin source-kind/family strata; complexity is a heuristic and requires visual review",
    }
    write_json(stage_root / "refs.json", refs)
    config = _stage_config(model, stage, max_turns)
    write_json(stage_root / "config.json", config)
    shutil.copyfile(DEFAULT_PROMPT, stage_root / "prompt.txt")
    shutil.copyfile(DEFAULT_CONTRACT, stage_root / "contract.json")
    shutil.copyfile(HERE / "gateway_transport.ts", stage_root / "gateway_transport.ts")
    catalog = json.loads(DEFAULT_CATALOG.read_text(encoding="utf-8"))
    models = [row for row in catalog.get("models", []) if isinstance(row, dict) and row.get("id") == model]
    if not models:
        raise CampaignError(f"selected teacher {model!r} is absent from model-catalog.json")
    catalog = dict(catalog)
    catalog["models"] = models
    catalog["campaign_source_catalog_sha256"] = sha_file(DEFAULT_CATALOG)
    write_json(stage_root / "model-catalog.json", catalog)
    write_json(stage_root / "stage-manifest.json", {"schema": SCHEMA, "stage": stage, "max_turns": max_turns, "max_tokens": "native", "model": model, "reference_count": len(references), "reference_ids": [row["id"] for row in references], "ordering_rule": "ascending complexity band plus round-robin source-kind/family strata; heuristic pending visual review"})


def _file_hashes(root: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name not in {"file-hashes.json", "file-hashes.sha256", "campaign-manifest.sha256"}):
        relative = path.relative_to(root).as_posix()
        files[relative] = {"sha256": sha_file(path), "bytes": path.stat().st_size}
    return {"schema": f"{SCHEMA}.integrity", "files": files}


def _launch_script(campaign_slug: str, model: str) -> str:
    quoted_model = _shell_quote(model)
    return f"""#!/usr/bin/env bash
set -Eeuo pipefail
# Dry-run preparation wrote this file; invoking it only prints commands and makes no paid calls.
BENCHMARK_ROOT="${{BENCHMARK_ROOT:-/root/painter/repo/painter/benchmarks/openrouter-teachers-20260922}}"
# Resolve the physical campaign directory so arbitrary output basenames and
# copies work.  CAMPAIGN_ROOT remains an explicit override for node layouts.
CAMPAIGN_ROOT="${{CAMPAIGN_ROOT:-$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd -P)}}"
echo 'Prepared commands (run one at a time; this script itself does not launch them):'
echo "bash $BENCHMARK_ROOT/run_benchmark_node.sh --root $CAMPAIGN_ROOT/short-02 --models {quoted_model} --track quality"
echo "bash $BENCHMARK_ROOT/run_benchmark_node.sh --root $CAMPAIGN_ROOT/medium-05 --models {quoted_model} --track quality"
echo "bash $BENCHMARK_ROOT/run_benchmark_node.sh --root $CAMPAIGN_ROOT/long-12 --models {quoted_model} --track quality"
"""


def _campaign_readme(campaign_slug: str, model: str) -> str:
    return f"""# Selected-teacher campaign setup\n\nThis is a dry-run setup for `{model}` (`{campaign_slug}`). It contains three independent quality roots:\n\n- `short-02/`: maximum 2 turns\n- `medium-05/`: maximum 5 turns\n- `long-12/`: maximum 12 turns\n\nAll three stages use the model's native output allowance. The images came from a separate training manifest, were SHA-256 checked, and any overlap with the benchmark evaluation holdout was excluded and recorded in `campaign-manifest.json`. Complexity ordering is a deterministic heuristic; visually review generated episodes before admission.\n\n`launch_commands.sh` only prints the exact node launch commands. It does not start a provider request, rent a node, or prepare a node. Copy this entire campaign directory to the node under `teacher-campaigns/{campaign_slug}/`, then run the printed commands explicitly, one stage at a time.\n\nIntegrity files:\n\n- `campaign-manifest.sha256` checks `campaign-manifest.json`.\n- `file-hashes.sha256` checks `file-hashes.json`; the latter records every setup file except the integrity files themselves.\n"""


def prepare_campaign(*, output: Path, training_manifest: Path, reference_root: Path | None, model: str, benchmark_refs: Path, counts: Mapping[str, int], split: str = "train") -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise CampaignError(f"output already exists; refusing to overwrite: {output}")
    if not model or "/" not in model:
        raise CampaignError("--teacher-model must be an exact provider/model ID")
    holdout_value = _load_json_or_jsonl(benchmark_refs.resolve())
    holdout_entries = _manifest_entries(holdout_value, benchmark_refs)
    holdout_hashes = set()
    for entry in holdout_entries:
        value = entry.get("sha256")
        if isinstance(value, str) and len(value) == 64:
            holdout_hashes.add(value.lower())
    if not holdout_hashes:
        raise CampaignError("benchmark holdout manifest has no SHA-256 hashes")
    holdout_sha = sha_file(benchmark_refs.resolve())
    rows, source_sha, excluded_holdout = load_training_entries(training_manifest, reference_root=reference_root, holdout_hashes=holdout_hashes, split=split)
    selected = select_stage_rows(rows, counts)
    slug = f"{_safe_slug(model)}-{source_sha[:12]}"
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=parent))
    try:
        for stage, max_turns in STAGES:
            key = stage.split("-")[0]
            _write_stage(temporary / stage, stage, max_turns, model, selected[key], source_sha, holdout_sha, split)
        launch = _launch_script(slug, model)
        launch_path = temporary / "launch_commands.sh"
        launch_path.write_text(launch, encoding="utf-8")
        launch_path.chmod(launch_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        (temporary / "README.md").write_text(_campaign_readme(slug, model), encoding="utf-8")
        summary = {"schema": SCHEMA, "campaign_id": slug, "selected_teacher_model": model, "training_manifest": str(training_manifest.resolve()), "training_manifest_sha256": source_sha, "benchmark_holdout_manifest": str(benchmark_refs.resolve()), "benchmark_holdout_manifest_sha256": holdout_sha, "split": split, "counts": dict(counts), "excluded_benchmark_holdout_overlap": excluded_holdout, "excluded_benchmark_holdout_overlap_count": len(excluded_holdout), "stages": {stage: {"max_turns": turns, "max_tokens": "native", "references": len(selected[stage.split("-")[0]])} for stage, turns in STAGES}, "launch_script": "launch_commands.sh", "no_provider_calls": True, "no_benchmark_results_reused": True}
        write_json(temporary / "campaign-manifest.json", summary)
        hashes = _file_hashes(temporary)
        write_json(temporary / "file-hashes.json", hashes)
        hashes_sha = sha_file(temporary / "file-hashes.json")
        (temporary / "file-hashes.sha256").write_text(f"{hashes_sha}  file-hashes.json\n", encoding="utf-8")
        manifest_sha = sha_file(temporary / "campaign-manifest.json")
        (temporary / "campaign-manifest.sha256").write_text(f"{manifest_sha}  campaign-manifest.json\n", encoding="utf-8")
        for path in (temporary / "campaign-manifest.json", temporary / "campaign-manifest.sha256", temporary / "file-hashes.json", temporary / "file-hashes.sha256"):
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    summary["output"] = str(output)
    summary["campaign_manifest_sha256"] = manifest_sha
    summary["file_hashes_sha256"] = hashes_sha
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", required=True, help="required safety marker; this command never launches generation")
    parser.add_argument("--training-manifest", type=Path, required=True, help="separate training-only JSON/JSONL manifest; benchmark refs.json is rejected")
    parser.add_argument("--reference-root", type=Path, help="base directory for relative image paths")
    parser.add_argument("--output", type=Path, required=True, help="new campaign setup directory; must not already exist")
    parser.add_argument("--teacher-model", required=True, help="exact selected provider/model ID")
    parser.add_argument("--benchmark-refs", type=Path, default=DEFAULT_BENCHMARK_REFS, help="evaluation holdout refs.json")
    parser.add_argument("--short-count", type=int, default=24)
    parser.add_argument("--medium-count", type=int, default=48)
    parser.add_argument("--long-count", type=int, default=96)
    parser.add_argument("--split", default="train")
    args = parser.parse_args(argv)
    counts = {"short": args.short_count, "medium": args.medium_count, "long": args.long_count}
    if any(not isinstance(value, int) or value <= 0 for value in counts.values()):
        parser.error("stage counts must be positive")
    try:
        summary = prepare_campaign(output=args.output, training_manifest=args.training_manifest, reference_root=args.reference_root, model=args.teacher_model, benchmark_refs=args.benchmark_refs, counts=counts, split=args.split)
    except CampaignError as exc:
        parser.exit(2, f"campaign setup error: {exc}\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
