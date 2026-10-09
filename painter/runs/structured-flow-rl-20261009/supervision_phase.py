"""Linux render-and-approve phase for structured-flow code supervision.

The authoring module creates candidates but deliberately does not decide that
an improved image is better.  This phase binds those candidates to the pinned
Linux renderer, sends every valid initial/final pair through the existing
generic Flash side judge, retains every raw decision, and admits only strong
candidate wins with verified image hashes.  It writes filtered rich SFT rows
only when the bounded warmup has enough source groups and finish states.

No model, renderer, or provider is used at import time.  The command is
Linux-only; tests inject renderer and judge callables so they remain offline.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Awaitable, Callable, Mapping, Sequence


RUN_ID = "structured-flow-rl-20261009"
SCHEMA = "painter.structured-flow-supervision-phase.v1"
APPROVAL_SCHEMA = "painter.structured-flow-approval-candidates.v1"
MAX_JUDGE_CONCURRENCY = 4
DEFAULT_WORKERS = 4
PAIR_TIMEOUT_SECONDS = 300
MIN_TRAIN_FAMILIES = 5
MIN_DEV_FAMILIES = 1
MIN_FINISH_ROWS = 5
MAX_FINISH_ROWS = 8
STRONG_CONSENSUS = {"initial_agreement", "three_of_four", "four_of_four"}
FLASH_MODEL = "xiaomi/mimo-v2.6-flash"
HEX64 = set("0123456789abcdef")


def _load_flow_module() -> Any:
    here = Path(__file__).resolve().parent
    path = here / "flow_demonstrations.py"
    spec = importlib.util.spec_from_file_location("structured_flow_demonstrations_for_supervision", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load flow demonstration source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


flow = _load_flow_module()

try:
    from painter.text_pairwise_consistent_side_judge_v2_generic import MODEL as _GENERIC_MODEL
    from painter.text_pairwise_consistent_side_judge_v2_generic import judge_pair
except ImportError:  # pragma: no cover - direct Linux node fallback
    from text_pairwise_consistent_side_judge_v2_generic import MODEL as _GENERIC_MODEL
    from text_pairwise_consistent_side_judge_v2_generic import judge_pair

if _GENERIC_MODEL != FLASH_MODEL:
    raise RuntimeError("generic Flash judge model binding changed")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))


def _require_linux() -> None:
    if sys.platform != "linux":
        raise RuntimeError("structured-flow supervision phase is Linux-node-only")


def _hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX64


def _validate_gateway_env(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Gateway environment file is missing: {path}")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != 0o600:
        raise PermissionError("Gateway environment file must be mode 600")
    return path


def _read_manifest(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("run_id") != RUN_ID or manifest.get("schema") != flow.SCHEMA:
        raise ValueError("data directory is not the structured-flow candidate manifest")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != 16:
        raise ValueError("structured-flow supervision requires exactly sixteen authored candidates")
    return manifest


def _materialize_briefs(root: Path, manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    brief_root = root / "supervision-briefs"
    result: dict[str, dict[str, Any]] = {}
    for row in manifest["rows"]:
        candidate_id = str(row["id"])
        brief = str(row.get("brief", "")).strip()
        if not brief:
            raise ValueError(f"candidate {candidate_id} has no painting brief")
        path = brief_root / f"{candidate_id}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(brief + "\n", encoding="utf-8")
        result[candidate_id] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "text": brief,
        }
    return result


def render_and_verify(
    root: Path,
    *,
    renderer: Path,
    renderer_python: Path,
    renderer_sha256: str,
    workers: int,
) -> dict[str, Any]:
    """Run the authored 32 programs through the pinned Linux renderer."""
    _require_linux()
    if not _hash(renderer_sha256):
        raise ValueError("renderer SHA-256 must be a lowercase 64-character hash")
    if not 1 <= workers <= MAX_JUDGE_CONCURRENCY:
        raise ValueError(f"workers must be in [1,{MAX_JUDGE_CONCURRENCY}]")
    value = flow.render(
        root,
        renderer,
        renderer_python,
        workers=workers,
        run_as_user="painter",
        renderer_sha256=renderer_sha256,
    )
    identity = value.get("renderer")
    if not isinstance(identity, Mapping) or identity.get("renderer_sha256") != renderer_sha256:
        raise ValueError("render receipt is not bound to the requested published renderer SHA-256")
    results = value.get("results")
    if not isinstance(results, list) or len(results) != 32:
        raise ValueError("renderer did not produce exactly thirty-two receipts")
    # A single failed pair is censored below; usable families still proceed
    # when the per-pair receipt and PNG hashes verify.
    return value


def _render_lookup(render_manifest: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    results = render_manifest.get("results")
    if not isinstance(results, list):
        raise ValueError("render manifest has no result list")
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for item in results:
        if not isinstance(item, Mapping):
            raise ValueError("render receipt is malformed")
        key = (str(item.get("candidate_id")), str(item.get("version")))
        lookup[key] = dict(item)
    return lookup


def _pair_evidence(
    root: Path,
    candidate_id: str,
    lookup: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[Path, Path, str | None, str | None, bool, str | None]:
    initial = root / "renders" / candidate_id / "block_in.png"
    final = root / "renders" / candidate_id / "improved_final.png"
    initial_record = lookup.get((candidate_id, "block_in"))
    final_record = lookup.get((candidate_id, "improved_final"))
    if not isinstance(initial_record, Mapping) or not isinstance(final_record, Mapping):
        return initial, final, None, None, False, "render_receipt_missing"
    initial_sha = sha256_file(initial) if initial.is_file() else None
    final_sha = sha256_file(final) if final.is_file() else None
    if not initial_record.get("valid") or not final_record.get("valid"):
        return initial, final, initial_sha, final_sha, False, "render_receipt_invalid"
    if initial_sha != initial_record.get("png_sha256") or final_sha != final_record.get("png_sha256"):
        return initial, final, initial_sha, final_sha, False, "render_png_hash_mismatch"
    return initial, final, initial_sha, final_sha, True, None


JudgeFn = Callable[..., Awaitable[dict[str, Any]]]


async def judge_candidates(
    root: Path,
    manifest: Mapping[str, Any],
    render_manifest: Mapping[str, Any],
    briefs: Mapping[str, Mapping[str, Any]],
    gateway_env: Path,
    *,
    workers: int = DEFAULT_WORKERS,
    judge_fn: JudgeFn | None = None,
) -> list[dict[str, Any]]:
    """Judge all sixteen pairs concurrently, retaining raw successes/failures."""
    _require_linux()
    if not 1 <= workers <= MAX_JUDGE_CONCURRENCY:
        raise ValueError(f"workers must be in [1,{MAX_JUDGE_CONCURRENCY}]")
    gateway_env = _validate_gateway_env(gateway_env)
    lookup = _render_lookup(render_manifest)
    judge = judge_fn or judge_pair
    semaphore = asyncio.Semaphore(workers)

    async def one(row: Mapping[str, Any]) -> dict[str, Any]:
        candidate_id = str(row["id"])
        brief_path = root / str(briefs[candidate_id]["path"])
        initial, final, initial_sha, final_sha, render_valid, render_error = _pair_evidence(root, candidate_id, lookup)
        base = {
            "id": candidate_id,
            "source_group": candidate_id,
            "split": row.get("split"),
            "brief": briefs[candidate_id]["path"],
            "brief_sha256": sha256_file(brief_path),
            "initial_png": initial.relative_to(root).as_posix(),
            "initial_png_sha256": initial_sha,
            "final_png": final.relative_to(root).as_posix(),
            "final_png_sha256": final_sha,
            "render_valid": render_valid,
            "judge_model": FLASH_MODEL,
            "judge_protocol": "painter.text_pairwise_consistent_side_judge_v2_generic",
        }
        if not render_valid:
            return {
                **base,
                "raw_decision": None,
                "judge_error": {"type": "render_invalid", "reason": render_error},
            }
        try:
            async with semaphore:
                raw = await asyncio.wait_for(
                    judge(
                        brief=brief_path,
                        anchor=initial,
                        candidate=final,
                        key_file=gateway_env,
                        model=FLASH_MODEL,
                    ),
                    timeout=PAIR_TIMEOUT_SECONDS,
                )
            if not isinstance(raw, Mapping):
                raise ValueError("judge returned a non-object decision")
            return {**base, "raw_decision": copy.deepcopy(dict(raw))}
        except Exception as exc:
            return {
                **base,
                "raw_decision": None,
                "judge_error": {"type": type(exc).__name__},
            }

    return await asyncio.gather(*(one(row) for row in manifest["rows"]))


def _approved(candidate: Mapping[str, Any]) -> tuple[bool, str]:
    """Apply only the generic judge's strong candidate-win result."""
    raw = candidate.get("raw_decision")
    if not isinstance(raw, Mapping):
        return False, "judge_failed_or_timed_out"
    if raw.get("status") != "scored":
        return False, "judge_status_not_scored"
    if raw.get("consensus") not in STRONG_CONSENSUS:
        return False, "judge_consensus_not_strong"
    score = raw.get("score")
    if not isinstance(score, (int, float)) or score <= 0.5:
        return False, "candidate_did_not_strongly_win"
    if raw.get("preference") == "tied":
        return False, "tie_is_censored"
    image_hashes = raw.get("image_sha256")
    if not isinstance(image_hashes, Mapping):
        return False, "judge_image_hashes_missing"
    if image_hashes.get("anchor") != candidate.get("initial_png_sha256") or image_hashes.get("candidate") != candidate.get("final_png_sha256"):
        return False, "judge_image_hashes_mismatch"
    image_ids = raw.get("image_ids")
    if not isinstance(image_ids, Mapping) or raw.get("winner_id") != image_ids.get("candidate"):
        return False, "candidate_is_not_judge_winner"
    if raw.get("brief_sha256") != candidate.get("brief_sha256"):
        return False, "judge_brief_hash_mismatch"
    return True, "strong_candidate_win"


def _approval_item(candidate: Mapping[str, Any]) -> dict[str, Any]:
    accepted, reason = _approved(candidate)
    raw = candidate.get("raw_decision")
    item = {
        "id": candidate["id"],
        "source_group": candidate["source_group"],
        "split": candidate.get("split"),
        "brief": candidate["brief"],
        "brief_sha256": candidate["brief_sha256"],
        "initial_png": candidate["initial_png"],
        "initial_png_sha256": candidate["initial_png_sha256"],
        "final_png": candidate["final_png"],
        "final_png_sha256": candidate["final_png_sha256"],
        "render_valid": bool(candidate.get("render_valid")),
        "decision": "approved" if accepted else "censored",
        "admission": "reviewed_improvement" if accepted else "censored",
        "improved_reviewed": accepted,
        "admission_reason": reason,
        "score": raw.get("score") if isinstance(raw, Mapping) else None,
        "consensus": raw.get("consensus") if isinstance(raw, Mapping) else None,
        "status": raw.get("status") if isinstance(raw, Mapping) else "judge_error",
        "raw_decision": raw,
        "judge_error": candidate.get("judge_error"),
    }
    return item


def _finish_row(row: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Make one terminal FINISHED row from an already approved final row."""
    value = copy.deepcopy(dict(row))
    candidate_id = str(row["source_group"])
    final_png = root / "renders" / candidate_id / "improved_final.png"
    source = root / str(row["source_program"])
    manifest_row = next(item for item in flow._read_manifest(root)["rows"] if item["id"] == candidate_id)
    final_program = source.read_text(encoding="utf-8")
    value.update({
        "id": f"{candidate_id}-finish",
        "flow_stage": "finish",
        "source_program": source.relative_to(root).as_posix(),
        "source_sha256": sha256_file(source),
        "prior_canvas": {"path": final_png.relative_to(root).as_posix(), "sha256": sha256_file(final_png)},
        "previous_program_in_user_text": True,
        "render": {
            "png": final_png.relative_to(root).as_posix(),
            "png_sha256": sha256_file(final_png),
            "source_sha256": sha256_file(source),
        },
        "target_receipt": dict(row.get("target_receipt") or {}, stage="finish", png_sha256=sha256_file(final_png), verified=True, approved=True),
        "quality_status": "reviewed_visual_final",
        "admission": "reviewed_visual_final",
        "improved_reviewed": True,
        "quality_claim": False,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": flow.flow_system()}]},
            {"role": "user", "content": [
                {"type": "text", "text": flow._user_text(manifest_row, "finish", final_program)},
                flow._image_part(final_png),
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "FINISHED"}]},
        ],
    })
    return value


def _approved_rich_rows(
    root: Path,
    manifest: Mapping[str, Any],
    approval: Mapping[str, Any],
    render_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build rich rows directly, so one censored render cannot be fabricated."""
    flow._verify_sources(root, manifest)
    flow._verify_prompt(root, manifest)
    lookup = _render_lookup(render_manifest)
    approved_ids = {
        str(item["id"])
        for item in approval["candidates"]
        if item.get("decision") == "approved" and item.get("improved_reviewed") is True
    }
    approvals_by_id = {str(item["id"]): item for item in approval["candidates"]}
    rows: list[dict[str, Any]] = []
    for candidate in manifest["rows"]:
        candidate_id = str(candidate["id"])
        if candidate_id not in approved_ids:
            continue
        approval_item = approvals_by_id[candidate_id]
        initial_source = root / candidate["programs"][0]
        final_source = root / candidate["programs"][1]
        initial_program = initial_source.read_text(encoding="utf-8")
        final_program = final_source.read_text(encoding="utf-8")
        initial_png, final_png = _pair_evidence(root, candidate_id, lookup)[:2]
        initial_record = lookup[(candidate_id, "block_in")]
        final_record = lookup[(candidate_id, "improved_final")]
        initial_sha = sha256_file(initial_source)
        final_sha = sha256_file(final_source)
        if initial_record.get("source_sha256") != initial_sha or final_record.get("source_sha256") != final_sha:
            raise ValueError(f"candidate {candidate_id} source hash does not match render receipt")
        if not initial_png.is_file() or not final_png.is_file() or sha256_file(initial_png) != initial_record.get("png_sha256") or sha256_file(final_png) != final_record.get("png_sha256"):
            raise ValueError(f"candidate {candidate_id} approved PNG hash is not current")
        for version, program, source, png, prior_program, prior_canvas, render_record in (
            ("block_in", initial_program, initial_source, initial_png, None, None, initial_record),
            ("improved_final", final_program, final_source, final_png, initial_program, initial_png, final_record),
        ):
            target = flow.target_block(program)
            target_receipt = {
                "verified": True,
                "valid": True,
                "approved": True,
                "source_group": candidate_id,
                "review_status": approval.get("status"),
                "decision": approval_item.get("decision"),
                "improved_reviewed": True,
                "stage": version,
                "png_sha256": sha256_file(png),
                "source_sha256": sha256_file(source),
                "initial_png_sha256": approval_item.get("initial_png_sha256"),
                "final_png_sha256": approval_item.get("final_png_sha256"),
            }
            user_content: list[dict[str, Any]] = [{"type": "text", "text": flow._user_text(candidate, version, prior_program)}]
            if prior_canvas is not None:
                user_content.append(flow._image_part(prior_canvas))
            rows.append({
                "schema": flow.SFT_SCHEMA,
                "id": f"{candidate_id}-{version}",
                "source_group": candidate_id,
                "category": candidate["category"],
                "level": candidate["level"],
                "split": candidate["split"],
                "flow_stage": version,
                "source_program": source.relative_to(root).as_posix(),
                "source_sha256": sha256_file(source),
                "prior_canvas": None if prior_canvas is None else {"path": prior_canvas.relative_to(root).as_posix(), "sha256": sha256_file(prior_canvas)},
                "previous_program_in_user_text": prior_program is not None,
                "render": {"png": png.relative_to(root).as_posix(), "png_sha256": render_record.get("png_sha256"), "source_sha256": render_record.get("source_sha256")},
                "target_receipt": target_receipt,
                "quality_status": "reviewed_improvement",
                "admission": "reviewed_improvement",
                "improved_reviewed": True,
                "quality_claim": False,
                "messages": [
                    {"role": "system", "content": [{"type": "text", "text": flow.flow_system()}]},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": [{"type": "text", "text": target}]},
                ],
            })
    return rows


def _write_filtered_exports(root: Path, approval: Mapping[str, Any], render_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Export hash-verified rich rows, then retain approved source groups only."""
    manifest = flow._read_manifest(root)
    rows = _approved_rich_rows(root, manifest, approval, render_manifest)
    exported = {"rows": len(rows), "teacher_rows": 0, "finish_rows": 0, "quality_status": "reviewed_visual_subset", "quality_claim": False}
    train = [row for row in rows if row.get("split") == "train"]
    dev = [row for row in rows if row.get("split") == "dev"]
    approved_train = [row for row in train if row.get("flow_stage") == "improved_final"]
    finish_candidates = []
    seen: set[str] = set()
    for row in approved_train:
        group = str(row["source_group"])
        if group in seen or len(finish_candidates) >= MAX_FINISH_ROWS:
            continue
        seen.add(group)
        finish_candidates.append(_finish_row(row, root))
    train.extend(finish_candidates)
    train_groups = {str(row["source_group"]) for row in train}
    dev_groups = {str(row["source_group"]) for row in dev}
    if len(train_groups) < MIN_TRAIN_FAMILIES or len(dev_groups) < MIN_DEV_FAMILIES or len(finish_candidates) < MIN_FINISH_ROWS:
        # flow.export_sft writes its complete candidate split as an audit
        # artifact.  Do not leave those unfiltered files at the warmup input
        # paths when this review is insufficient for training.
        for stale in (root / "train.jsonl", root / "dev.jsonl"):
            if stale.is_file() or stale.is_symlink():
                stale.unlink()
        receipt = {
            "schema": SCHEMA,
            "run_id": RUN_ID,
            "status": "reviewed_insufficient_for_warmup",
            "requirements": {"train_families": MIN_TRAIN_FAMILIES, "dev_families": MIN_DEV_FAMILIES, "finish_rows": MIN_FINISH_ROWS},
            "actual": {"train_families": len(train_groups), "dev_families": len(dev_groups), "finish_rows": len(finish_candidates)},
            "flow_export": exported,
        }
        write_json(root / "supervision-phase.json", receipt)
        return receipt
    approved_sft = rows
    write_jsonl(root / "sft-approved.jsonl", approved_sft)
    write_jsonl(root / "train.jsonl", train)
    write_jsonl(root / "dev.jsonl", dev)
    return {
        "schema": SCHEMA,
        "run_id": RUN_ID,
        "status": "reviewed_ready_for_warmup",
        "requirements": {"train_families": MIN_TRAIN_FAMILIES, "dev_families": MIN_DEV_FAMILIES, "finish_rows": MIN_FINISH_ROWS},
        "actual": {"train_families": len(train_groups), "dev_families": len(dev_groups), "finish_rows": len(finish_candidates), "train_rows": len(train), "dev_rows": len(dev)},
        "flow_export": exported,
        "paths": {"approved_sft": "sft-approved.jsonl", "train": "train.jsonl", "dev": "dev.jsonl"},
    }



def _validate_cached_review(root: Path, renderer_sha256: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate a restored render/review tree before re-exporting its rows."""
    if not _hash(renderer_sha256):
        raise ValueError("cached review requires a lowercase renderer SHA-256")
    manifest = _read_manifest(root)
    flow._verify_sources(root, manifest)
    flow._verify_prompt(root, manifest)
    render_path = root / "render-manifest.json"
    approval_path = root / "approval-candidates.json"
    if render_path.is_symlink() or not render_path.is_file() or approval_path.is_symlink() or not approval_path.is_file():
        raise ValueError("cached review is missing render or approval manifest")
    render_manifest = json.loads(render_path.read_text(encoding="utf-8"))
    if not isinstance(render_manifest, dict) or render_manifest.get("schema") != "painter.structured-flow-render.v1" or render_manifest.get("run_id") != RUN_ID:
        raise ValueError("cached render manifest protocol is not recognized")
    if render_manifest.get("source_manifest_sha256") != manifest.get("source_manifest_sha256"):
        raise ValueError("cached render source manifest hash is stale")
    renderer = render_manifest.get("renderer")
    if not isinstance(renderer, Mapping) or renderer.get("renderer_sha256") != renderer_sha256:
        raise ValueError("cached render renderer hash does not match the current renderer")
    results = render_manifest.get("results")
    if not isinstance(results, list) or len(results) != 32 or render_manifest.get("program_count") != 32 or render_manifest.get("valid_program_count") != 32 or render_manifest.get("complete_hash_verified") is not True:
        raise ValueError("cached render manifest is incomplete")
    lookup = _render_lookup(render_manifest)
    if len(lookup) != 32:
        raise ValueError("cached render manifest has duplicate or incomplete pair identities")
    for row in manifest["rows"]:
        candidate_id = str(row["id"])
        for version, source_name in (("block_in", "block_in.js"), ("improved_final", "improved_final.js")):
            record = lookup.get((candidate_id, version))
            source = root / "programs" / candidate_id / source_name
            png = root / "renders" / candidate_id / f"{version}.png"
            if not isinstance(record, Mapping) or record.get("valid") is not True or not source.is_file() or not png.is_file():
                raise ValueError(f"cached render pair is incomplete: {candidate_id}/{version}")
            if record.get("source_sha256") != flow.file_sha256(source) or record.get("png_sha256") != flow.file_sha256(png):
                raise ValueError(f"cached render hashes are stale: {candidate_id}/{version}")
            receipt = record.get("receipt")
            if not isinstance(receipt, Mapping) or receipt.get("valid") is not True or receipt.get("renderer_sha256") != renderer_sha256:
                raise ValueError(f"cached render receipt is unbound: {candidate_id}/{version}")
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if not isinstance(approval, dict) or approval.get("schema") != APPROVAL_SCHEMA or approval.get("run_id") != RUN_ID or approval.get("status") != "reviewed" or approval.get("review_protocol") != "painter.text_pairwise_consistent_side_judge_v2_generic" or approval.get("judge_model") != FLASH_MODEL:
        raise ValueError("cached approval manifest protocol is not the pinned generic Flash review")
    candidates = approval.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 16 or approval.get("candidate_count") != 16:
        raise ValueError("cached approval manifest must retain all sixteen decisions")
    candidate_ids = {str(row["id"]) for row in manifest["rows"]}
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("cached approval candidate is malformed")
        candidate_id = str(candidate.get("id"))
        if candidate_id in seen or candidate_id not in candidate_ids:
            raise ValueError("cached approval candidate IDs do not match authored candidates")
        seen.add(candidate_id)
        raw_brief = str(candidate.get("brief", ""))
        brief_rel = Path(raw_brief)
        if brief_rel.is_absolute() or any(part in {"", ".", ".."} for part in brief_rel.parts) or brief_rel.parts[:1] != ("supervision-briefs",):
            raise ValueError(f"cached approval brief path is unsafe: {candidate_id}")
        brief = root / brief_rel
        if brief.is_symlink() or not brief.is_file() or flow.file_sha256(brief) != candidate.get("brief_sha256"):
            raise ValueError(f"cached approval brief hash mismatch: {candidate_id}")
        for version, field in (("block_in", "initial_png_sha256"), ("improved_final", "final_png_sha256")):
            record = lookup[(candidate_id, version)]
            if candidate.get(field) != record.get("png_sha256"):
                raise ValueError(f"cached approval/render hash mismatch: {candidate_id}/{version}")
        if candidate.get("decision") not in {"approved", "censored"}:
            raise ValueError(f"cached approval decision is invalid: {candidate_id}")
    if seen != candidate_ids:
        raise ValueError("cached approval manifest does not cover all authored candidates")
    return manifest, render_manifest, approval


def refresh_filtered_exports(data: str | Path, renderer_sha256: str) -> dict[str, Any]:
    """Reuse paid renders/decisions and only recompute the bounded rich export."""
    _require_linux()
    root = Path(data).expanduser().resolve()
    manifest, render_manifest, approval = _validate_cached_review(root, renderer_sha256)
    receipt = _write_filtered_exports(root, approval, render_manifest)
    receipt.update({
        "reuse_mode": "hash_verified_paid_supervision_archive",
        "cached_candidate_manifest_sha256": sha256_file(root / "manifest.json"),
        "cached_render_manifest_sha256": sha256_file(root / "render-manifest.json"),
        "cached_approval_manifest_sha256": sha256_file(root / "approval-candidates.json"),
        "renderer": render_manifest.get("renderer"),
        "judge_model": approval.get("judge_model"),
        "judge_protocol": approval.get("review_protocol"),
        "approved_candidates": approval.get("approved_count"),
        "censored_candidates": len(approval.get("candidates", [])) - int(approval.get("approved_count", 0)),
    })
    write_json(root / "supervision-phase.json", receipt)
    return receipt

def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def run_phase(
    data: str | Path,
    renderer: str | Path,
    renderer_python: str | Path,
    renderer_sha256: str,
    gateway_env: str | Path,
    *,
    workers: int = DEFAULT_WORKERS,
    render_fn: Callable[..., dict[str, Any]] | None = None,
    judge_fn: JudgeFn | None = None,
) -> dict[str, Any]:
    """Run render, exact generic Flash review, and bounded rich export."""
    _require_linux()
    if not 1 <= workers <= MAX_JUDGE_CONCURRENCY:
        raise ValueError(f"workers must be in [1,{MAX_JUDGE_CONCURRENCY}]")
    root = Path(data).expanduser().resolve()
    manifest = _read_manifest(root)
    briefs = _materialize_briefs(root, manifest)
    renderer_callable = render_fn or render_and_verify
    render_manifest = renderer_callable(
        root,
        renderer=Path(renderer),
        renderer_python=Path(renderer_python),
        renderer_sha256=renderer_sha256,
        workers=workers,
    )
    decisions = asyncio.run(judge_candidates(root, manifest, render_manifest, briefs, Path(gateway_env), workers=workers, judge_fn=judge_fn))
    candidates = [_approval_item(item) for item in decisions]
    accepted = [item for item in candidates if item["decision"] == "approved"]
    approval = {
        "schema": APPROVAL_SCHEMA,
        "run_id": RUN_ID,
        "status": "reviewed",
        "review_protocol": "painter.text_pairwise_consistent_side_judge_v2_generic",
        "judge_model": FLASH_MODEL,
        "strong_consensus": sorted(STRONG_CONSENSUS),
        "quality_claim": False,
        "renderer": render_manifest.get("renderer"),
        "candidate_count": len(candidates),
        "approved_count": len(accepted),
        "candidates": candidates,
    }
    write_json(root / "approval-candidates.json", approval)
    receipt = _write_filtered_exports(root, approval, render_manifest)
    receipt.update({
        "approval_manifest": "approval-candidates.json",
        "approval_manifest_sha256": sha256_file(root / "approval-candidates.json"),
        "renderer": render_manifest.get("renderer"),
        "judge_model": FLASH_MODEL,
        "judge_protocol": "painter.text_pairwise_consistent_side_judge_v2_generic",
        "judged_candidates": len(candidates),
        "approved_candidates": len(accepted),
        "censored_candidates": len(candidates) - len(accepted),
        "briefs": {key: {"path": value["path"], "sha256": value["sha256"]} for key, value in briefs.items()},
    })
    write_json(root / "supervision-phase.json", receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--renderer", type=Path)
    parser.add_argument("--renderer-python", type=Path)
    parser.add_argument("--renderer-sha256", required=True)
    parser.add_argument("--gateway-env", type=Path)
    parser.add_argument("--reuse-approved", action="store_true")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.reuse_approved:
        receipt = refresh_filtered_exports(args.data, args.renderer_sha256)
    else:
        if args.renderer is None or args.renderer_python is None or args.gateway_env is None:
            parser.error("normal review requires --renderer, --renderer-python, and --gateway-env")
        receipt = run_phase(
            args.data, args.renderer, args.renderer_python, args.renderer_sha256, args.gateway_env,
            workers=args.workers,
        )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0 if receipt.get("status") == "reviewed_ready_for_warmup" else 2


if __name__ == "__main__":
    raise SystemExit(main())
