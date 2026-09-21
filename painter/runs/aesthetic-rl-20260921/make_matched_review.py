"""Build a review-only HTML comparison from matched evaluation summaries.

This helper copies existing PNG evidence and records missing or invalid
canvases.  It never renders, calls a model, scores an image, or infers an
improvement.  Node paths such as ``/root/painter/...`` are resolved only when
the caller supplies the explicit ``--node-root`` to local-root mapping.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import re
import shutil
from typing import Any, Iterable


SCHEMA = "painter.matched-review.v1"
SUMMARY_SCHEMA = "painter.matched-evaluation.v1"
SAMPLE_LIMIT = 4
ROLES = ("sft_baseline", "rl_candidate")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON summary: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"summary must be a JSON object: {path}")
    if value.get("schema") != SUMMARY_SCHEMA:
        raise ValueError(f"unsupported summary schema in {path}: {value.get('schema')!r}")
    if not isinstance(value.get("results"), list):
        raise ValueError(f"summary results must be a list: {path}")
    return value


def _rows_by_key(summary: dict[str, Any], path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(summary["results"]):
        if not isinstance(row, dict):
            raise ValueError(f"summary result {index} is not an object: {path}")
        task = row.get("task")
        if not isinstance(task, dict):
            raise ValueError(f"summary result {index} has no task object: {path}")
        key = task.get("source_group") or task.get("id")
        if not isinstance(key, str) or not key:
            raise ValueError(f"summary result {index} has no matching task key: {path}")
        if key in rows:
            raise ValueError(f"summary has duplicate task key {key!r}: {path}")
        rows[key] = row
    return rows


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return slug or "sample"


def _under(path: Path, root: Path) -> Path | None:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    if any(part == ".." for part in relative.parts):
        return None
    return relative


def _candidate_paths(
    raw: Any,
    *,
    summary_path: Path,
    node_root: Path,
    local_root: Path,
    reference_roots: Iterable[Path],
) -> tuple[list[Path], str]:
    if not isinstance(raw, str) or not raw:
        return [], "missing_path"
    value = Path(raw)
    roots = [local_root, *reference_roots]
    if value.is_absolute():
        relative = _under(value, node_root)
        if relative is not None:
            if not relative.parts:
                return [], "invalid_node_path"
            return [root / relative for root in roots], "node_mapped"
        # Locally produced summaries may already point at an existing evidence
        # file.  Remote absolute paths outside --node-root remain unmapped.
        if value.is_file():
            return [value], "direct_local"
        return [], "unmapped_absolute"
    if any(part == ".." for part in value.parts):
        return [], "invalid_relative_path"
    return [summary_path.parent / value, *[root / value for root in roots]], "relative"


def _resolve_existing(
    raw: Any,
    *,
    summary_path: Path,
    node_root: Path,
    local_root: Path,
    reference_roots: Iterable[Path],
) -> tuple[Path | None, str]:
    candidates, resolution = _candidate_paths(
        raw,
        summary_path=summary_path,
        node_root=node_root,
        local_root=local_root,
        reference_roots=reference_roots,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate, resolution
    return None, "missing_file" if candidates else resolution


def _turn_count(row: dict[str, Any]) -> int | None:
    turns = row.get("turns")
    if isinstance(turns, list):
        return len(turns)
    if isinstance(turns, int) and not isinstance(turns, bool):
        return turns
    return None


def _token_count(row: dict[str, Any]) -> int | None:
    value = row.get("generated_tokens")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    turns = row.get("turns")
    if isinstance(turns, list):
        tokens = [turn.get("tokens") for turn in turns if isinstance(turn, dict)]
        if tokens and all(isinstance(token, int) and not isinstance(token, bool) for token in tokens):
            return sum(tokens)
    return None


def _asset_name(key: str, role: str, kind: str) -> str:
    return f"{_safe_slug(key)}-{role}-{kind}.png"


def _copy_png(
    raw: Any,
    *,
    summary_path: Path,
    node_root: Path,
    local_root: Path,
    reference_roots: Iterable[Path],
    output_assets: Path,
    asset_name: str,
) -> dict[str, Any]:
    source, resolution = _resolve_existing(
        raw,
        summary_path=summary_path,
        node_root=node_root,
        local_root=local_root,
        reference_roots=reference_roots,
    )
    record: dict[str, Any] = {
        "source": raw if isinstance(raw, str) else None,
        "resolution": resolution,
        "asset": None,
        "status": "missing" if source is None else "invalid",
    }
    if source is None:
        return record
    if source.suffix.lower() != ".png":
        record["status"] = "invalid"
        return record
    output_assets.mkdir(parents=True, exist_ok=True)
    destination = output_assets / asset_name
    shutil.copy2(source, destination)
    record["asset"] = f"assets/{asset_name}"
    record["status"] = "present"
    return record


def _variant_record(
    row: dict[str, Any] | None,
    *,
    role: str,
    summary_path: Path,
    node_root: Path,
    local_root: Path,
    reference_roots: Iterable[Path],
    output_assets: Path,
    key: str,
) -> dict[str, Any]:
    if row is None:
        return {
            "status": "missing_result",
            "valid_canvas": False,
            "finished": None,
            "operational_cutoff": None,
            "turns": None,
            "generated_tokens": None,
            "canvas": {"source": None, "resolution": "missing_result", "asset": None, "status": "missing"},
        }
    valid_canvas = row.get("valid_canvas") is True
    canvas = _copy_png(
        row.get("canvas"),
        summary_path=summary_path,
        node_root=node_root,
        local_root=local_root,
        reference_roots=reference_roots,
        output_assets=output_assets,
        asset_name=_asset_name(key, role, "canvas"),
    )
    if canvas["status"] == "present":
        status = "valid" if valid_canvas else "invalid_flag"
    else:
        status = "missing" if canvas["status"] == "missing" else "invalid"
    return {
        "status": status,
        "valid_canvas": valid_canvas,
        "finished": row.get("finished") if isinstance(row.get("finished"), bool) else None,
        "operational_cutoff": row.get("operational_cutoff") if isinstance(row.get("operational_cutoff"), bool) else None,
        "turns": _turn_count(row),
        "generated_tokens": _token_count(row),
        "canvas": canvas,
    }


def _reference_record(
    row: dict[str, Any] | None,
    *,
    summary_path: Path,
    node_root: Path,
    local_root: Path,
    reference_roots: Iterable[Path],
    output_assets: Path,
    key: str,
) -> dict[str, Any]:
    raw = row.get("reference") if isinstance(row, dict) else None
    return _copy_png(
        raw,
        summary_path=summary_path,
        node_root=node_root,
        local_root=local_root,
        reference_roots=reference_roots,
        output_assets=output_assets,
        asset_name=_asset_name(key, "reference", "image"),
    )


def _img_or_status(record: dict[str, Any], label: str) -> str:
    asset = record.get("asset")
    if isinstance(asset, str):
        return f'<img src="{html.escape(asset, quote=True)}" alt="{html.escape(label, quote=True)}">'
    status = html.escape(str(record.get("status", "missing")))
    return f'<div class="missing">{html.escape(label)}<br><strong>{status}</strong><br>No image copied</div>'


def _metadata_line(record: dict[str, Any]) -> str:
    parts = []
    if record.get("turns") is not None:
        parts.append(f"turns={record['turns']}")
    if record.get("generated_tokens") is not None:
        parts.append(f"tokens={record['generated_tokens']}")
    if record.get("finished") is not None:
        parts.append(f"finished={record['finished']}")
    if record.get("status"):
        parts.append(f"canvas={record['status']}")
    return html.escape(" · ".join(parts) or "metadata unavailable")


def _html(manifest: dict[str, Any]) -> str:
    policy = manifest["metadata"]["policy_version"]
    heading_policy = (
        f"SFT: {policy.get('sft_baseline') or 'missing'}; "
        f"RL: {policy.get('rl_candidate') or 'missing'}"
    )
    cards: list[str] = []
    for sample in manifest["samples"]:
        title = html.escape(str(sample.get("source_group") or sample["key"]))
        cards.append(
            "<section class='sample'>"
            f"<h2>{title}</h2>"
            "<div class='columns'>"
            f"<figure><figcaption>Reference</figcaption>{_img_or_status(sample['reference'], 'Reference')}"
            "</figure>"
            f"<figure><figcaption>SFT baseline</figcaption>{_img_or_status(sample['sft_baseline']['canvas'], 'SFT baseline')}"
            f"<small>{_metadata_line(sample['sft_baseline'])}</small></figure>"
            f"<figure><figcaption>RL candidate</figcaption>{_img_or_status(sample['rl_candidate']['canvas'], 'RL candidate')}"
            f"<small>{_metadata_line(sample['rl_candidate'])}</small></figure>"
            "</div></section>"
        )
    return """<!doctype html>
<html><head><meta charset="utf-8"><title>Matched painting review</title>
<style>
body{font-family:system-ui,sans-serif;margin:2rem;background:#f7f7f4;color:#222}
.notice{padding:1rem;background:#fff4d6;border:1px solid #d8bf72}
.sample{margin:2rem 0;padding:1rem;background:#fff;border:1px solid #ddd}
.columns{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1rem}
figure{margin:0} figcaption{font-weight:700;margin-bottom:.5rem}
img{display:block;width:100%;height:auto;background:#eee;border:1px solid #ccc}
small{display:block;margin-top:.5rem;color:#555}.missing{min-height:8rem;padding:1rem;background:#f4eeee;color:#822}
</style></head><body>
<h1>Matched painting review</h1>
<div class="notice">Review evidence only. No automatic scoring or improvement claim is made.</div>
<p>Policy versions: POLICY_HEADING</p>
<p>Samples included: INCLUDED / AVAILABLE (limit LIMIT). Missing and invalid canvases are labelled.</p>
CARDS
</body></html>
""".replace("POLICY_HEADING", html.escape(heading_policy)).replace(
        "INCLUDED", str(manifest["metadata"]["samples_included"])
    ).replace("AVAILABLE", str(manifest["metadata"]["samples_available"])).replace(
        "LIMIT", str(manifest["metadata"]["sample_limit"])
    ).replace("CARDS", "\n".join(cards))


def build_review(
    baseline_summary: Path,
    candidate_summary: Path,
    output: Path,
    *,
    node_root: Path,
    local_root: Path,
    reference_roots: Iterable[Path] = (),
) -> dict[str, Any]:
    """Build ``comparison-manifest.json`` and ``index.html`` in ``output``."""
    baseline_summary = Path(baseline_summary).resolve()
    candidate_summary = Path(candidate_summary).resolve()
    output = Path(output).resolve()
    node_root = Path(node_root).resolve()
    local_root = Path(local_root).resolve()
    reference_roots = tuple(Path(path).resolve() for path in reference_roots)
    baseline = _load_json(baseline_summary)
    candidate = _load_json(candidate_summary)
    baseline_rows = _rows_by_key(baseline, baseline_summary)
    candidate_rows = _rows_by_key(candidate, candidate_summary)
    keys = list(baseline_rows)
    keys.extend(key for key in candidate_rows if key not in baseline_rows)
    included_keys = keys[:SAMPLE_LIMIT]

    output_assets = output / "assets"
    samples: list[dict[str, Any]] = []
    for key in included_keys:
        baseline_row = baseline_rows.get(key)
        candidate_row = candidate_rows.get(key)
        reference_row = baseline_row or candidate_row
        reference = _reference_record(
            reference_row,
            summary_path=baseline_summary if baseline_row is not None else candidate_summary,
            node_root=node_root,
            local_root=local_root,
            reference_roots=reference_roots,
            output_assets=output_assets,
            key=key,
        )
        samples.append(
            {
                "key": key,
                "task_id": (reference_row or {}).get("task", {}).get("id"),
                "source_group": (reference_row or {}).get("task", {}).get("source_group", key),
                "family": (reference_row or {}).get("task", {}).get("family"),
                "reference": reference,
                "sft_baseline": _variant_record(
                    baseline_row,
                    role="sft_baseline",
                    summary_path=baseline_summary,
                    node_root=node_root,
                    local_root=local_root,
                    reference_roots=reference_roots,
                    output_assets=output_assets,
                    key=key,
                ),
                "rl_candidate": _variant_record(
                    candidate_row,
                    role="rl_candidate",
                    summary_path=candidate_summary,
                    node_root=node_root,
                    local_root=local_root,
                    reference_roots=reference_roots,
                    output_assets=output_assets,
                    key=key,
                ),
            }
        )

    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "metadata": {
            "policy_version": {
                "sft_baseline": baseline.get("policy_version"),
                "rl_candidate": candidate.get("policy_version"),
            },
            "sample_limit": SAMPLE_LIMIT,
            "samples_available": len(keys),
            "samples_included": len(samples),
            "truncated": len(keys) > SAMPLE_LIMIT,
            "no_auto_scoring": True,
            "improvement_claim": False,
            "baseline_summary": str(baseline_summary),
            "candidate_summary": str(candidate_summary),
            "node_root_mapping": str(node_root),
        },
        "samples": samples,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "index.html").write_text(_html(manifest))
    return manifest


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--node-root", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument(
        "--reference-root",
        type=Path,
        action="append",
        default=[],
        help="Additional local roots searched for mapped reference paths; repeatable.",
    )
    args = parser.parse_args(argv)
    manifest = build_review(
        args.baseline_summary,
        args.candidate_summary,
        args.output,
        node_root=args.node_root,
        local_root=args.local_root,
        reference_roots=args.reference_root,
    )
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "samples_included": manifest["metadata"]["samples_included"],
        "samples_available": manifest["metadata"]["samples_available"],
        "sample_limit": manifest["metadata"]["sample_limit"],
    }, sort_keys=True))
    return manifest


if __name__ == "__main__":
    main()
