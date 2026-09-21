"""Materialize a provenance-preserving multimodal SFT candidate corpus.

This is data preparation only.  It accepts the new ASTRA/Sol teacher outputs,
the verified scaled procedural archive, and optional already-compatible SFT
JSONL.  It never invents teacher quality and it never puts a development row
in the training split.  Token-balanced admission happens after the Linux
node's real processor audit (``balance_by_tokens.py``).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))
from contract import NEXT_VERSION, SYSTEM, identity, paint_target, validate_teacher_program


RUN = "photo-curriculum-sft-20260922"
IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
PROGRAM_RE = re.compile(r"(?:createCanvas|function\s+setup|function\s+draw|noLoop)\s*\(", re.I)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data_uri(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = IMAGE_TYPES.get(suffix) or mimetypes.guess_type(path.name)[0]
    if mime not in {"image/png", "image/jpeg", "image/webp"}:
        raise ValueError(f"Unsupported image type: {path}")
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = "".join(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n" for row in rows).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def resolve_image(value: Any, *roots: Path) -> Path:
    if not value:
        raise ValueError("missing reference image path")
    candidate = Path(str(value))
    # Teacher metadata may preserve the controller's absolute path.  If that
    # path is absent on the Linux node, its basename is still resolved against
    # the explicitly staged reference roots.
    attempts = [candidate] if candidate.is_absolute() else []
    if candidate.is_absolute() and not candidate.is_file():
        attempts.append(Path(candidate.name))
    attempts.extend([root / candidate for root in roots])
    attempts.extend([root / candidate.name for root in roots])
    for attempt in attempts:
        if attempt.is_file():
            return attempt.resolve()
    raise FileNotFoundError(f"reference image not found: {value}")


def _parts(text: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": text}]


def full_row(*, ident: str, source_group: str, split: str, family: str, reference: Path,
             target: str, category: str, metadata: dict[str, Any], instruction: str,
             plan: str | None = None) -> dict[str, Any]:
    if not target.strip():
        raise ValueError(f"{ident}: empty assistant target")
    validate_teacher_program(target)
    if not PROGRAM_RE.search(target):
        raise ValueError(f"{ident}: target does not look like a complete p5 program")
    answer = paint_target(plan or "Paint the complete reference faithfully.", target)
    return {
        "id": ident,
        "source_group": source_group,
        "split": split,
        "family": family,
        "source_kind": category,
        "example_kind": "full_paint",
        "complete_target": True,
        "reference_sha256": digest(reference),
        "target_sha256": hashlib.sha256(answer.encode()).hexdigest(),
        "contract": identity(),
        "teacher_metadata": metadata,
        "messages": [
            {"role": "system", "content": _parts(SYSTEM)},
            {"role": "user", "content": [
                {"type": "text", "text": "REFERENCE"},
                {"type": "image_url", "image_url": {"url": data_uri(reference)}},
                {"type": "text", "text": instruction or NEXT_VERSION},
            ]},
            {"role": "assistant", "content": _parts(answer)},
        ],
    }


def correction_row(*, ident: str, source_group: str, split: str, family: str, reference: Path,
                   canvas: Path, current_program: str, target: str, category: str,
                   metadata: dict[str, Any], plan: str | None = None) -> dict[str, Any]:
    row = full_row(ident=ident, source_group=source_group, split=split, family=family,
                   reference=reference, target=target, category=category, metadata=metadata,
                   instruction=NEXT_VERSION, plan=plan)
    row["example_kind"] = "draft_to_final_correction"
    row["current_canvas_sha256"] = digest(canvas)
    row["current_program_sha256"] = hashlib.sha256(current_program.encode()).hexdigest()
    row["messages"][1]["content"] = [
        {"type": "text", "text": "REFERENCE"},
        {"type": "image_url", "image_url": {"url": data_uri(reference)}},
        {"type": "text", "text": "CURRENT CANVAS"},
        {"type": "image_url", "image_url": {"url": data_uri(canvas)}},
        {"type": "text", "text": "Current program:\n" + current_program},
        {"type": "text", "text": NEXT_VERSION},
    ]
    return row


def load_teacher_rows(root: Path, include_corrections: bool, reference_roots: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, rejected = [], []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        program = folder / "program.js"
        metadata_path = next((folder / name for name in ("metadata.json", "record.json") if (folder / name).is_file()), None)
        if not program.is_file() and metadata_path is None:
            continue
        try:
            if metadata_path is None:
                raise ValueError("metadata.json is required")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be an object")
            ident = str(metadata.get("id") or folder.name)
            source_group = str(metadata.get("source_group") or ident)
            split = str(metadata.get("split") or "train")
            if split not in {"train", "dev", "validation", "test", "sealed_test"}:
                raise ValueError(f"unknown split {split}")
            if metadata.get("teacher_status") in {"draft", "invalid", "failed"} or metadata.get("complete") is False:
                raise ValueError("teacher metadata does not mark a complete target")
            if metadata.get("reference_equals_canvas") is True:
                raise ValueError("reference-identical synthetic finish is not a painting target")
            reference = resolve_image(metadata.get("reference") or metadata.get("reference_path"), folder, root, *reference_roots)
            declared_reference_sha = metadata.get("reference_sha256")
            if declared_reference_sha and declared_reference_sha != digest(reference):
                raise ValueError("teacher reference SHA-256 does not match staged image")
            declared_program_sha = metadata.get("program_sha256")
            if declared_program_sha and declared_program_sha != digest(program):
                raise ValueError("teacher program SHA-256 does not match program.js")
            png_candidates = [folder / name for name in ("program.png", "program.js.png", "rendered.png", "painting.png") if (folder / name).is_file()]
            receipt_candidates = [folder / name for name in ("program.json", "program.js.json", "rendered.json", "render-receipt.json") if (folder / name).is_file()]
            render_png = render_receipt = receipt = None
            program_hash = digest(program)
            for png_candidate in png_candidates:
                for receipt_candidate in receipt_candidates:
                    try:
                        receipt_candidate_data = json.loads(receipt_candidate.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        continue
                    if (receipt_candidate_data.get("valid")
                            and receipt_candidate_data.get("source_sha256") == program_hash
                            and receipt_candidate_data.get("png_sha256") == digest(png_candidate)):
                        render_png, render_receipt, receipt = png_candidate, receipt_candidate, receipt_candidate_data
                        break
                if receipt is not None:
                    break
            if render_png is None or render_receipt is None or receipt is None:
                raise ValueError("complete teacher requires a valid matching program render receipt")
            target = program.read_text(encoding="utf-8")
            plan_path = folder / "plan.txt"
            plan = plan_path.read_text(encoding="utf-8").strip() if plan_path.is_file() else str(metadata.get("plan") or "Paint the complete reference faithfully.")
            row_meta = dict(metadata, teacher_program=str(program.relative_to(root)), teacher_program_sha256=digest(program))
            rows.append(full_row(ident=ident + "__full", source_group=source_group, split=split,
                                 family=str(metadata.get("family") or "photo"), reference=reference,
                                 target=target, category="real_photo_teacher", metadata=row_meta,
                                 instruction=NEXT_VERSION, plan=plan))
            if include_corrections:
                draft = next((folder / name for name in ("draft.js", "current.js") if (folder / name).is_file()), None)
                draft_image = next((folder / name for name in ("draft.png", "draft.js.png", "current.png", "canvas.png") if (folder / name).is_file()), None)
                if draft and draft_image:
                    rows.append(correction_row(ident=ident + "__correction", source_group=source_group, split=split,
                                                family=str(metadata.get("family") or "photo"), reference=reference,
                                                canvas=draft_image, current_program=draft.read_text(encoding="utf-8"),
                                                target=target, category="real_photo_teacher", metadata=row_meta, plan=plan))
                elif draft or draft_image:
                    rejected.append({"id": ident, "reason": "draft program and draft canvas must both exist for correction"})
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            rejected.append({"id": folder.name, "reason": f"{type(exc).__name__}: {exc}"})
    return rows, rejected


def _find_member(root: Path, ident: str, relative: str | None = None) -> Path:
    attempts = []
    if relative:
        attempts.extend([root / relative, root / "shards" / relative])
    attempts.extend([root / ident, root / "shards" / ident])
    for shard in sorted((root / "shards").glob("shard-*")) if (root / "shards").is_dir() else []:
        attempts.append(shard / ident)
    for attempt in attempts:
        if attempt.is_file():
            return attempt.resolve()
        if attempt.is_dir():
            # The manifest commonly stores ``id/turn-3.png`` while the archive
            # stores each id below a numbered shard.
            name = Path(relative).name if relative else ""
            if name and (attempt / name).is_file():
                return (attempt / name).resolve()
    raise FileNotFoundError(f"archive member not found: {ident} {relative or ''}")


def load_procedural(root: Path, manifest: Path, include_corrections: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = root.resolve()
    manifest = manifest.resolve()
    rows, rejected = [], []
    for raw in read_rows(manifest):
        ident = str(raw.get("id") or "")
        split = str(raw.get("split") or "")
        if not ident or split not in {"train", "dev", "validation", "test", "sealed_test"}:
            rejected.append({"id": ident, "reason": "missing id or split"}); continue
        # Keep procedural development rows as validation candidates.  Test and
        # sealed-test material never enters this SFT corpus.
        if split not in {"train", "dev", "validation"}:
            continue
        source_group = str(raw.get("source_group") or ident)
        # COCO development examples are held out from this corpus entirely;
        # their teacher rows are excluded by source group at the build call.
        if source_group.startswith("coco128-"):
            rejected.append({"id": ident, "reason": "COCO development source is excluded from SFT"}); continue
        programs = raw.get("programs") or []
        if not programs:
            rejected.append({"id": ident, "reason": "no complete program list"}); continue
        try:
            reference = _find_member(root, ident, str(raw.get("reference") or ""))
            final_program_path = _find_member(root, ident, str(programs[-1]))
            target = final_program_path.read_text(encoding="utf-8")
            metadata = dict(raw, source_manifest=str(manifest), target_program=str(final_program_path.relative_to(root)))
            family = str(raw.get("family") or raw.get("primary_family") or "procedural")
            plan = (raw.get("plans") or [None])[-1]
            target_split = "train" if split == "train" else "validation"
            rows.append(full_row(ident=ident + "__full", source_group=source_group, split=target_split, family=family,
                                 reference=reference, target=target, category="procedural_foundation",
                                 metadata=metadata, instruction=NEXT_VERSION, plan=plan))
            if include_corrections and len(programs) >= 2:
                current_path = _find_member(root, ident, str(programs[-2]))
                current_image = current_path.with_suffix(".png")
                if current_image.is_file():
                    rows.append(correction_row(ident=ident + "__correction", source_group=source_group, split=target_split,
                                                family=family, reference=reference, canvas=current_image,
                                                current_program=current_path.read_text(encoding="utf-8"), target=target,
                                                category="procedural_foundation", metadata=metadata, plan=plan))
        except (OSError, ValueError) as exc:
            rejected.append({"id": ident, "reason": f"{type(exc).__name__}: {exc}"})
    return rows, rejected


def normalize_existing(path: Path, category: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, rejected = [], []
    for row in read_rows(path):
        ident = str(row.get("id") or "")
        split = str(row.get("split") or "train")
        try:
            if not ident or not isinstance(row.get("messages"), list):
                raise ValueError("id and messages are required")
            if split not in {"train", "dev", "validation", "test", "sealed_test"}:
                raise ValueError(f"unknown split {split}")
            messages = row["messages"]
            if messages[-1].get("role") != "assistant" or not any(p.get("text", "").strip() for p in messages[-1].get("content", []) if p.get("type") == "text"):
                raise ValueError("final assistant target is empty")
            copied = dict(row)
            copied.update(id=f"{ident}__{category}", source_kind=category, source_example_id=ident,
                          teacher_metadata={"imported_from": str(path)})
            rows.append(copied)
        except (ValueError, IndexError, TypeError) as exc:
            rejected.append({"id": ident, "reason": f"{type(exc).__name__}: {exc}"})
    return rows, rejected


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    if args.teachers:
        rows, bad = load_teacher_rows(args.teachers.resolve(), args.include_corrections,
                                      [path.resolve() for path in args.reference_root])
        all_rows.extend(rows); rejected.extend(bad)
    if args.procedural_root and args.procedural_manifest:
        rows, bad = load_procedural(args.procedural_root.resolve(), args.procedural_manifest.resolve(), args.include_corrections)
        all_rows.extend(rows); rejected.extend(bad)
    for path, category in [(p, "prior_compatible") for p in args.prior_jsonl] + [(p, "diverse_foundation") for p in args.foundation_jsonl]:
        rows, bad = normalize_existing(path.resolve(), category)
        all_rows.extend(rows); rejected.extend(bad)
    for path in args.retention_jsonl:
        rows, bad = normalize_existing(path.resolve(), "retention")
        all_rows.extend(rows); rejected.extend(bad)
    if not all_rows:
        raise ValueError("No training candidates were materialized")

    # A source group may have many turns/examples, but it must belong to one
    # split. Repetition is added later by the token-balanced scheduler.
    groups: dict[str, str] = {}
    train, validation = [], []
    for row in all_rows:
        group = str(row.get("source_group") or row["id"])
        split = str(row.get("split") or "train")
        if group in args.exclude_source_group:
            rejected.append({"id": row["id"], "reason": "explicit heldout source group excluded from training"}); continue
        previous = groups.get(group)
        if previous and previous != split:
            raise ValueError(f"source group leaks across splits: {group}: {previous}, {split}")
        groups[group] = split
        (train if split == "train" else validation).append(row)
    if not train:
        raise ValueError("No train rows after split/exclusion checks")
    seen_ids = set()
    for row in train + validation:
        if row["id"] in seen_ids:
            raise ValueError(f"duplicate example id: {row['id']}")
        seen_ids.add(row["id"])
    train_hash = write_jsonl(output / "candidates-train.jsonl", train)
    validation_hash = write_jsonl(output / "candidates-validation.jsonl", validation)
    receipt = {
        "schema": 2, "run_name": RUN, "status": "candidates_prepared", "training_started": False,
        "candidate_counts": Counter(str(r["source_kind"]) for r in train),
        "example_kind_counts": Counter(str(r.get("example_kind", "existing")) for r in train),
        "unique_train_source_groups": len({r["source_group"] for r in train}),
        "train_rows": len(train), "validation_rows": len(validation),
        "train_sha256": train_hash, "validation_sha256": validation_hash,
        "excluded_rows": len(rejected), "exclusions": rejected,
        "heldout_policy": "all non-train rows plus explicitly excluded source groups stay out of training",
        "token_balance": "pending actual Prime processor audit; no row/token weighting claim yet",
    }
    (output / "candidate-receipt.json").write_text(json.dumps(receipt, indent=2, default=dict) + "\n")
    return receipt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--teachers", type=Path)
    p.add_argument("--procedural-root", type=Path)
    p.add_argument("--procedural-manifest", type=Path)
    p.add_argument("--prior-jsonl", type=Path, action="append", default=[])
    p.add_argument("--foundation-jsonl", type=Path, action="append", default=[])
    p.add_argument("--retention-jsonl", type=Path, action="append", default=[])
    p.add_argument("--include-corrections", action="store_true")
    p.add_argument("--reference-root", type=Path, action="append", default=[])
    p.add_argument("--exclude-source-group", action="append", default=[])
    args = p.parse_args()
    if bool(args.procedural_root) != bool(args.procedural_manifest):
        p.error("--procedural-root and --procedural-manifest must be supplied together")
    print(json.dumps(build(args), indent=2, default=dict))


if __name__ == "__main__":
    main()
