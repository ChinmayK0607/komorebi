#!/usr/bin/env python3
"""Build a portable, offline visual comparison for the three RL arms.

The input is a small, frozen JSON manifest.  Each arm names one or more
checkpoint evaluation roots containing the native painting evidence
(``painting.json``, ``painting.png`` and, when available, ``program.js``).
This module only reads and copies evidence.  It never renders, imports a
model, calls an API, or selects a result by visual quality.

The terminal selection follows ``painter.final_action_credit.validate_last_turn``:
the final action is the last recorded turn.  A missing or invalid terminal
action remains missing or invalid in the gallery, even if an earlier turn has
a valid PNG.  The earlier evidence is retained in the turn ledger for audit.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path
from typing import Any


SCHEMA = "text-paint-rl-judge-ablation-comparison-v1"
ARM_COUNT = 3
CASE_COUNT = 16
DEFAULT_MAX_TURNS = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _safe_id(value: Any, what: str = "id") -> str:
    if not isinstance(value, str) or not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"unsafe {what}: {value!r}")
    return value


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {path}") from exc


def _hash_if_file(path: Path | None) -> str | None:
    return sha256(path) if path is not None and path.is_file() and not path.is_symlink() else None


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _copy(source: Path, destination: Path) -> bool:
    if not source.is_file() or source.is_symlink():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return True


def _verify_hash(path: Path, expected: Any, *, name: str) -> str:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"missing {name}: {path}")
    actual = sha256(path)
    if expected is not None and actual != expected:
        raise ValueError(f"{name} hash mismatch: {path}: expected {expected}, got {actual}")
    return actual


def _normalise_checkpoint(raw: dict[str, Any], *, fallback_id: str) -> dict[str, Any]:
    """Normalize one checkpoint without dropping provenance fields."""
    value = dict(raw)
    checkpoint_id = value.get("id") or value.get("checkpoint_id") or value.get("step") or fallback_id
    value["id"] = _safe_id(str(checkpoint_id), "checkpoint id")
    root = value.get("eval_root") or value.get("evaluation_root") or value.get("root")
    if not isinstance(root, str) or not root:
        raise ValueError(f"checkpoint {value['id']} is missing eval_root")
    value["eval_root"] = root
    value.setdefault("label", value["id"])
    return value


def _normalise_arm(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("arm must be an object")
    arm = dict(raw)
    arm["id"] = _safe_id(arm.get("id"), "arm id")
    if not isinstance(arm.get("label", arm["id"]), str):
        raise ValueError(f"arm {arm['id']} label must be text")
    arm.setdefault("label", arm["id"])
    checkpoints = arm.get("checkpoints")
    if checkpoints is None:
        checkpoints = [arm]
    elif isinstance(checkpoints, dict):
        checkpoints = [dict(item, id=key) if isinstance(item, dict) else {"id": key, "eval_root": item}
                       for key, item in checkpoints.items()]
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError(f"arm {arm['id']} needs a non-empty checkpoints list")
    arm["checkpoints"] = [_normalise_checkpoint(item, fallback_id="final") for item in checkpoints]
    ids = [item["id"] for item in arm["checkpoints"]]
    if len(set(ids)) != len(ids):
        raise ValueError(f"arm {arm['id']} has duplicate checkpoint ids")
    return arm


def _normalise_reference(raw: dict[str, Any]) -> dict[str, Any]:
    reference = _normalise_arm(raw)
    reference["kind"] = raw.get("kind", "reference")
    return reference


def _materialize_prompt(case: dict[str, Any], manifest_path: Path, assets_root: Path) -> tuple[str | None, str | None, dict[str, Any]]:
    """Load only the case's declared brief and bind its hash.

    Text-panel manifests commonly carry ``reference_kind: text`` plus a
    ``reference``/``reference_path`` file rather than embedding ``brief``.
    The path is resolved relative to the manifest and is never searched for
    by case id.  A declared text file must carry a matching content hash.
    """
    direct = next((case.get(key) for key in ("prompt", "prompt_text", "brief", "brief_text")
                   if isinstance(case.get(key), str) and case.get(key).strip()), None)
    expected_direct = case.get("brief_sha256") or case.get("prompt_sha256")
    if expected_direct is None and str(case.get("reference_kind", "image")) == "text":
        expected_direct = case.get("reference_sha256")
    if direct is not None:
        encoded = direct.encode("utf-8")
        if expected_direct is not None and expected_direct != _sha_bytes(encoded):
            raise ValueError(f"prompt hash mismatch for {case['id']}")
        text = direct.strip()
        source = {"kind": "embedded", "sha256": _sha_bytes(encoded)}
    else:
        reference_kind = str(case.get("reference_kind", "image"))
        value = case.get("brief_path") or case.get("reference_path") or case.get("reference")
        if reference_kind != "text" and not case.get("brief_path") and not case.get("reference_path"):
            return None, None, {"kind": "absent"}
        if not isinstance(value, str) or not value:
            raise ValueError(f"text case {case['id']} needs a declared reference path")
        source_path = Path(value) if Path(value).is_absolute() else manifest_path.parent / value
        if source_path.is_symlink() or not source_path.is_file():
            raise FileNotFoundError(f"declared prompt source is missing for {case['id']}: {source_path}")
        raw = source_path.read_bytes()
        actual = _sha_bytes(raw)
        expected = case.get("reference_sha256") or case.get("brief_sha256") or case.get("prompt_sha256")
        if not isinstance(expected, str) or expected != actual:
            raise ValueError(f"declared prompt hash mismatch for {case['id']}: {actual}")
        try:
            text = raw.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise ValueError(f"declared prompt is not UTF-8 for {case['id']}") from exc
        source = {"kind": "declared_file", "path": str(source_path), "sha256": actual}
    destination = assets_root / "cases" / "prompts" / f"{case['id']}.txt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text + "\n", encoding="utf-8")
    return text, _relative(destination, assets_root.parent), source


def load_manifest(path: Path, *, expected_case_count: int = CASE_COUNT) -> tuple[dict[str, Any], str]:
    """Load and validate the explicit matched panel and arm identity."""
    path = path.expanduser().resolve()
    payload = _json(path)
    if not isinstance(payload, dict):
        raise ValueError("comparison manifest must be an object")
    schema = payload.get("schema")
    if schema not in {SCHEMA, "text-paint-rl-judge-ablation-matched-eval-v2"} and not str(schema).startswith("text-paint-rl-"):
        raise ValueError(f"unexpected comparison manifest schema: {schema!r}")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != expected_case_count:
        raise ValueError(f"comparison manifest must contain exactly {expected_case_count} cases")
    seen: set[str] = set()
    normalized_cases: list[dict[str, Any]] = []
    for raw in cases:
        if not isinstance(raw, dict):
            raise ValueError("manifest case must be an object")
        case = dict(raw)
        case["id"] = _safe_id(case.get("id") or case.get("case_id"), "case id")
        if case["id"] in seen:
            raise ValueError(f"duplicate case id: {case['id']}")
        seen.add(case["id"])
        case.setdefault("max_turns", DEFAULT_MAX_TURNS)
        if isinstance(case["max_turns"], bool) or not isinstance(case["max_turns"], int) or case["max_turns"] < 1:
            raise ValueError(f"invalid max_turns for {case['id']}")
        normalized_cases.append(case)
    arms_raw = payload.get("arms")
    if not isinstance(arms_raw, list) or len(arms_raw) != ARM_COUNT:
        raise ValueError(f"comparison manifest must contain exactly {ARM_COUNT} arms")
    arms = [_normalise_arm(raw) for raw in arms_raw]
    if len({arm["id"] for arm in arms}) != ARM_COUNT:
        raise ValueError("arm ids must be unique")

    # A panel hash is optional for compatibility with the node finalizer, but
    # if registered it must bind the exact case list before any copying occurs.
    panel_hash = payload.get("cases_sha256") or payload.get("panel_sha256")
    if panel_hash is not None:
        actual_panel_hash = canonical_sha(cases)
        if panel_hash != actual_panel_hash:
            raise ValueError(f"panel hash mismatch: expected {panel_hash}, got {actual_panel_hash}")

    value = dict(payload)
    value["schema"] = schema
    value["cases"] = normalized_cases
    value["arms"] = arms
    value["references"] = [_normalise_reference(item) for item in payload.get("references", [])]
    manifest_hash = sha256(path)
    value["manifest_path"] = str(path)
    value["manifest_sha256"] = manifest_hash
    return value, manifest_hash


def _candidate_roots(root: Path, case_id: str) -> list[Path]:
    direct = root / case_id
    if direct.is_dir() and not direct.is_symlink():
        return [direct]
    # Evaluation collectors sometimes put one run or episode directory above
    # the case folder.  Restrict the search to directories named exactly like
    # the panel case and reject ambiguity rather than guessing.
    matches = sorted(path for path in root.rglob(case_id) if path.is_dir() and not path.is_symlink())
    return matches


def _case_root(root: Path, case: dict[str, Any]) -> Path | None:
    explicit = case.get("eval_relpath") or case.get("relative_path")
    if explicit is not None:
        if not isinstance(explicit, str) or Path(explicit).is_absolute() or ".." in Path(explicit).parts:
            raise ValueError(f"unsafe eval path for {case['id']}")
        found = root / explicit
        return found if found.is_dir() and not found.is_symlink() else None
    matches = _candidate_roots(root, case["id"])
    if len(matches) > 1:
        raise ValueError(f"ambiguous evaluation case root for {case['id']}: {matches}")
    return matches[0] if matches else None


def _episode_root(case_root: Path | None) -> Path | None:
    if case_root is None:
        return None
    if any(path.is_dir() and path.name.startswith("turn-") for path in case_root.iterdir()):
        return case_root
    children = sorted(path for path in case_root.iterdir() if path.is_dir() and not path.is_symlink())
    turn_children = [path for path in children if any(item.is_dir() and item.name.startswith("turn-") for item in path.iterdir())]
    if len(turn_children) == 1:
        return turn_children[0]
    return case_root


def _read_receipt(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file() or path.is_symlink():
        return None, None
    try:
        value = _json(path)
    except ValueError:
        return None, "malformed receipt"
    if not isinstance(value, dict):
        return None, "receipt is not an object"
    return value, None


def _episode_final_metadata(case_root: Path | None) -> dict[str, Any]:
    """Read optional trace metadata without trusting paths inside it."""
    if case_root is None:
        return {}
    for name in ("episode.json", "trace.json", "result.json", "rollout.json"):
        path = case_root / name
        if not path.is_file() or path.is_symlink():
            continue
        try:
            value = _json(path)
        except ValueError:
            continue
        if isinstance(value, dict):
            return {"metadata_file": str(path), "metadata": value}
    return {}


def _metadata_turns(meta: dict[str, Any]) -> list[dict[str, Any]]:
    value = meta.get("metadata")
    turns = value.get("turns") if isinstance(value, dict) else None
    return [item for item in turns if isinstance(item, dict)] if isinstance(turns, list) else []


def _final_turn_metadata(meta: dict[str, Any]) -> dict[str, Any] | None:
    turns = _metadata_turns(meta)
    return turns[-1] if turns else None


def _turn_numbers(root: Path | None) -> list[int]:
    if root is None or not root.is_dir():
        return []
    result = []
    for path in root.iterdir():
        if path.is_dir() and not path.is_symlink() and path.name.startswith("turn-"):
            suffix = path.name[5:]
            if suffix.isdigit() and int(suffix) > 0:
                result.append(int(suffix))
    return sorted(set(result))


def _status(receipt: dict[str, Any] | None, error: str | None, png: Path, png_hash_ok: bool, *, is_final: bool) -> str:
    prefix = "final " if is_final else ""
    if receipt is None:
        if error:
            return prefix + "invalid receipt"
        if png.is_file():
            return prefix + "PNG without receipt (unverified)"
        return prefix + "missing"
    if receipt.get("valid") is True and png_hash_ok:
        return prefix + "valid"
    if receipt.get("valid") is True:
        return prefix + "valid action, image unverified"
    if png_hash_ok:
        return prefix + "invalid action (partial PNG)"
    if png.is_file():
        return prefix + "invalid action (PNG hash unverified)"
    return prefix + "invalid action"


def _copy_turn(
    turn_root: Path | None,
    *,
    arm_id: str,
    checkpoint_id: str,
    case_id: str,
    turn: int,
    assets_root: Path,
    is_final: bool,
) -> dict[str, Any]:
    receipt_path = turn_root / "painting.json" if turn_root else Path("/missing/painting.json")
    png_path = turn_root / "painting.png" if turn_root else Path("/missing/painting.png")
    program_path = turn_root / "program.js" if turn_root else Path("/missing/program.js")
    receipt, receipt_error = _read_receipt(receipt_path)
    actual_png_hash = _hash_if_file(png_path)
    expected_png_hash = receipt.get("png_sha256") if receipt else None
    png_hash_ok = bool(actual_png_hash and expected_png_hash and actual_png_hash == expected_png_hash)
    dest = assets_root / "evidence" / arm_id / checkpoint_id / case_id / f"turn-{turn}"
    dest_receipt = dest / "painting.json" if _copy(receipt_path, dest / "painting.json") else None
    dest_program = dest / "program.js" if _copy(program_path, dest / "program.js") else None
    dest_png = None
    if png_path.is_file() and not png_path.is_symlink():
        name = "painting.png" if receipt and receipt.get("valid") is True and png_hash_ok else (
            "painting-invalid.png" if receipt and receipt.get("valid") is not True else "painting-unverified.png"
        )
        dest_png = dest / name
        _copy(png_path, dest_png)
    return {
        "turn": turn,
        "is_final": is_final,
        "status": _status(receipt, receipt_error, png_path, png_hash_ok, is_final=is_final),
        "receipt": _relative(dest_receipt, assets_root.parent) if dest_receipt else None,
        "program": _relative(dest_program, assets_root.parent) if dest_program else None,
        "png": _relative(dest_png, assets_root.parent) if dest_png else None,
        "source_receipt": str(receipt_path) if receipt else None,
        "source_png": str(png_path) if actual_png_hash else None,
        "receipt_sha256": _hash_if_file(receipt_path),
        "png_sha256": actual_png_hash,
        "expected_png_sha256": expected_png_hash,
        "png_hash_ok": png_hash_ok,
        "receipt_valid": receipt.get("valid") if receipt else None,
        "receipt_error": receipt_error,
        "receipt_data": receipt,
    }


def _terminal_validity(final_turn: dict[str, Any] | None, metadata: dict[str, Any]) -> bool | None:
    """Prefer the trace's explicit terminal flag, then the receipt validity."""
    meta_final = _final_turn_metadata(metadata)
    if isinstance(meta_final, dict) and type(meta_final.get("valid")) is bool:
        return meta_final["valid"]
    validation = metadata.get("metadata", {}).get("final_action_validation") if isinstance(metadata.get("metadata"), dict) else None
    if isinstance(validation, dict) and type(validation.get("valid")) is bool:
        return validation["valid"]
    return final_turn.get("receipt_valid") if final_turn else None


def _build_case(
    case: dict[str, Any], checkpoint: dict[str, Any], *, arm_id: str, assets_root: Path,
    display_checkpoint_id: str | None = None,
) -> dict[str, Any]:
    root_value = checkpoint["eval_root"]
    raw_root = Path(root_value).expanduser()
    if raw_root.is_symlink():
        raise ValueError(f"evaluation root must not be a symlink: {raw_root}")
    root = raw_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"evaluation root for {arm_id}/{checkpoint['id']} is missing: {root}")
    case_root = _case_root(root, case)
    episode_root = _episode_root(case_root)
    metadata = _episode_final_metadata(episode_root)
    if not metadata:
        metadata = _episode_final_metadata(case_root)
    numbers = _turn_numbers(episode_root)
    meta_turn = _final_turn_metadata(metadata)
    meta_number = meta_turn.get("turn") if isinstance(meta_turn, dict) else None
    if isinstance(meta_number, int) and meta_number > 0:
        final_number = meta_number
    else:
        final_number = max(numbers) if numbers else None
    # This is deliberately the last *recorded* turn.  max_turns is shown as a
    # scope field, but it is never used to promote an earlier valid turn.
    turns: list[dict[str, Any]] = []
    max_turn = int(case.get("max_turns", DEFAULT_MAX_TURNS))
    for number in range(1, max(max_turn, max(numbers, default=0)) + 1):
        turn_root = episode_root / f"turn-{number}" if episode_root and (episode_root / f"turn-{number}").is_dir() else None
        turns.append(_copy_turn(
            turn_root,
            arm_id=arm_id,
            checkpoint_id=checkpoint["id"],
            case_id=case["id"],
            turn=number,
            assets_root=assets_root,
            is_final=number == final_number,
        ))
    final_turn = next((item for item in turns if item["turn"] == final_number), None) if final_number else None
    terminal_valid = _terminal_validity(final_turn, metadata)
    # A valid final result requires the native terminal action, a valid
    # renderer receipt, and the receipt-bound PNG.  Keep the action flag
    # separate so a valid action with missing/unverified pixels is censored.
    final_valid = (
        True if terminal_valid is True and final_turn is not None
        and final_turn.get("receipt_valid") is True and final_turn.get("png_hash_ok")
        else False if terminal_valid is False
        else None
    )
    if final_turn is None:
        final_status = "final missing"
    elif terminal_valid is False or final_turn.get("receipt_valid") is False:
        final_status = "final invalid action"
    elif final_valid is True:
        final_status = "final valid"
    elif terminal_valid is True:
        final_status = "final valid action, image unverified"
    else:
        final_status = "final censored / unverified"
    return {
        "case_id": case["id"],
        "arm_id": arm_id,
        # A reference may have a natural id such as ``sft512`` while it is
        # displayed beside a selected arm checkpoint.  Keep both identities.
        "checkpoint_id": display_checkpoint_id or checkpoint["id"],
        "actual_checkpoint_id": checkpoint["id"],
        "checkpoint": {key: value for key, value in checkpoint.items() if key != "eval_root"},
        "eval_root": str(root),
        "case_root": str(case_root) if case_root else None,
        "metadata_file": metadata.get("metadata_file"),
        "expected_max_turn": max_turn,
        "final_turn": final_number,
        "terminal_action_valid": terminal_valid,
        "final_valid": final_valid,
        "final_status": final_status,
        "final_png": final_turn.get("png") if final_turn else None,
        "final_receipt": final_turn.get("receipt") if final_turn else None,
        "turns": turns,
        "source_case_sha256": case.get("source_sha256") or case.get("brief_sha256") or case.get("prompt_sha256"),
        "seed": case.get("seed"),
        "provenance": case.get("provenance") or case.get("source_id") or case.get("source_group"),
    }


def _common_checkpoint(arms: list[dict[str, Any]], selected: str | None) -> str:
    common = set(item["id"] for item in arms[0]["checkpoints"])
    for arm in arms[1:]:
        common &= {item["id"] for item in arm["checkpoints"]}
    if selected is not None:
        if selected not in common:
            raise ValueError(f"checkpoint {selected!r} is not present in every arm; common={sorted(common)}")
        return selected
    if not common:
        raise ValueError("arms do not share a checkpoint id")
    return sorted(common)[0]


def _common_checkpoints(arms: list[dict[str, Any]], selected: str | None) -> list[str]:
    """Return every checkpoint id shared by the three actual arms."""
    common = set(item["id"] for item in arms[0]["checkpoints"])
    for arm in arms[1:]:
        common &= {item["id"] for item in arm["checkpoints"]}
    if selected is not None:
        if selected not in common:
            raise ValueError(f"checkpoint {selected!r} is not present in every arm; common={sorted(common)}")
        return [selected]
    if not common:
        raise ValueError("arms do not share a checkpoint id")
    return sorted(common)


def _pick_checkpoint(arm: dict[str, Any], checkpoint_id: str) -> dict[str, Any]:
    return next(item for item in arm["checkpoints"] if item["id"] == checkpoint_id)


def _html_text(value: Any, fallback: str = "—") -> str:
    return html.escape(fallback if value is None else str(value))


def _page(data: dict[str, Any]) -> str:
    # Labels and paths are in the data payload so the reveal control can show
    # them; visible default cards expose only fixed LEFT/RIGHT and opaque ids.
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Three-arm RL judge ablation · fixed visual comparison</title>
<style>
:root{{--ink:#17212b;--muted:#667789;--line:#d8e0e8;--paper:#f4f6f8;--card:#fff;--good:#176b4c;--bad:#a52b2b;--warn:#9a5b11;--blue:#155ca6}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}main{{max-width:1450px;margin:auto;padding:28px 22px 60px}}h1{{font-size:clamp(2rem,4vw,3.1rem);line-height:1.05;letter-spacing:-.04em;margin:.1rem 0 .8rem}}h2,h3,p{{margin-top:0}}h2{{font-size:1.2rem;margin-bottom:.2rem}}h3{{font-size:1rem;margin:1.1rem 0 .55rem}}.lede{{max-width:1000px;color:#334454}}.scope{{background:#fff8e7;border-left:5px solid #d39716;border-radius:8px;padding:11px 14px;max-width:1100px}}.controls{{display:flex;flex-wrap:wrap;gap:10px;align-items:end;background:var(--card);border:1px solid var(--line);padding:14px;border-radius:9px;margin:16px 0}}label{{display:grid;gap:4px;color:var(--muted);font-size:.8rem;font-weight:650}}select,input[type=checkbox]{{font:inherit}}select{{min-width:175px;padding:7px 8px;border:1px solid #bdc9d5;border-radius:6px;background:#fff;color:var(--ink)}}.stats{{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0}}.stat{{background:#fff;border:1px solid var(--line);border-radius:8px;padding:9px 12px;min-width:125px}}.stat b{{display:block;font-size:1.3rem}}.stat span{{font-size:.78rem;color:var(--muted)}}.pair{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px;margin-top:15px}}.panel{{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}}.panel h3{{margin:0;padding:11px 13px;background:#edf2f6;font-size:1rem;display:flex;justify-content:space-between;gap:8px;align-items:baseline}}.panel h3 small{{font-size:.75rem;color:var(--muted);font-weight:500}}.image-wrap{{min-height:280px;display:grid;place-items:center;background:#eef1f4}}.image-wrap img{{width:100%;aspect-ratio:1;object-fit:contain;background:#fff;display:block;cursor:zoom-in}}.missing{{min-height:280px;display:grid;place-content:center;text-align:center;padding:25px;color:#765f49;background:#f5f0ea}}.missing b{{font-size:1.05rem}}.missing span{{display:block;max-width:340px;margin-top:5px;font-size:.85rem}}.metadata,.ledger{{padding:10px 13px;border-top:1px solid var(--line);font-size:.84rem}}.metadata{{color:var(--muted)}}.status{{font-weight:750}}.status.good{{color:var(--good)}}.status.bad{{color:var(--bad)}}.status.warn{{color:var(--warn)}}.ledger summary{{cursor:pointer;color:var(--blue);font-weight:650}}table{{width:100%;border-collapse:collapse;font-size:.8rem;margin-top:7px}}th,td{{text-align:left;padding:4px 5px;border-bottom:1px solid #e8edf1;vertical-align:top}}th{{color:var(--muted);font-weight:650}}code{{font-size:.77em;overflow-wrap:anywhere}}.reveal-only{{display:none}}body.revealed .reveal-only{{display:block}}body.revealed .blind-label{{display:none}}.foot{{margin-top:20px;color:var(--muted);font-size:.83rem}}.case-prompt{{background:#f8fafb;border-left:4px solid #8da7bf;padding:10px 12px;white-space:pre-wrap;margin-top:13px}}@media(max-width:760px){{main{{padding:20px 11px 45px}}.pair{{grid-template-columns:1fr}}.image-wrap,.missing{{min-height:220px}}}}
</style></head><body><main>
<p class="blind-label" style="color:var(--blue);font-weight:750;letter-spacing:.08em;font-size:.75rem">BLINDED VISUAL EVIDENCE · FIXED POSITIONS</p>
<h1>Three-arm RL judge ablation</h1>
<p class="lede">This offline gallery places exactly two fixed-position images side by side for each selected pair. The left and right positions come from the manifest and are never swapped using a score, image hash, or visual heuristic. The terminal image is the last recorded action; invalid, missing, and censored terminal evidence stays visible.</p>
<p class="scope"><b>Blind mode:</b> arm and checkpoint labels stay hidden until reveal is enabled. Blind mode is a presentation aid, not a security boundary; the copied evidence and manifest remain local audit artifacts.</p>
<div class="controls"><label>Case<select id="case"></select></label><label>Pair<select id="pair"></select></label><label>Checkpoint<select id="checkpoint"></select></label><label style="display:flex;align-items:center;gap:6px;padding-bottom:8px"><input id="reveal" type="checkbox"> Reveal labels and provenance</label></div>
<div id="case-content"></div><p class="foot">Manifest SHA-256: <code>{html.escape(str(data["manifest_sha256"]))}</code>. Generated by a read-only local builder. A valid earlier turn is retained in the ledger but is never promoted when the selected terminal action is invalid or missing.</p>
</main><script>const DATA={blob};
const caseSelect=document.getElementById('case'),pairSelect=document.getElementById('pair'),checkpointSelect=document.getElementById('checkpoint'),reveal=document.getElementById('reveal'),content=document.getElementById('case-content');
const byId=(xs,key)=>xs.find(x=>x.id===key); const esc=(v)=>String(v??'—').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#39;');
function statusClass(s){{return s.includes('invalid')?'bad':s.includes('valid')&&!s.includes('unverified')?'good':'warn'}}
function sourceLabel(id){{const x=DATA.arm_meta.find(a=>a.id===id)||DATA.reference_meta.find(a=>a.id===id);return x?x.label:id}}
function metaLine(row){{const c=row.checkpoint||{{}};return `<span class="blind-label">fixed evidence</span><span class="reveal-only"><b>${{esc(sourceLabel(row.arm_id))}}</b> · checkpoint ${{esc(c.label||row.checkpoint_id)}} · seed ${{esc(row.seed)}} · source ${{esc(row.provenance)}}</span>`}}
function panel(row,side){{const status=row?row.final_status:'missing';const image=row&&row.final_png?`<div class="image-wrap"><img src="${{esc(row.final_png)}}" alt="${{esc(side)}} final painting" loading="lazy"></div>`:`<div class="missing"><b>${{esc(status)}}</b><span>No promoted image is available for this terminal action. Earlier valid turns remain in the ledger.</span></div>`;const rows=(row?.turns||[]).map(t=>`<tr><td>${{t.turn}}</td><td class="status ${{statusClass(t.status)}}">${{esc(t.status)}}</td><td>${{t.png?`<a href="${{esc(t.png)}}">PNG</a>`:'—'}} ${{t.receipt?`<a href="${{esc(t.receipt)}}">receipt</a>`:'—'}}</td></tr>`).join('');return `<section class="panel"><h3><span class="blind-label">${{side}}</span><span class="reveal-only">${{esc(sourceLabel(row?.arm_id||''))}}</span><small>${{esc(side)}} · fixed</small></h3>${{image}}<div class="metadata"><span class="status ${{statusClass(status)}}">${{esc(status)}}</span><br>${{row?metaLine(row):'No case evidence'}}</div><details class="ledger"><summary>Turn evidence and receipt paths</summary><table><thead><tr><th>Turn</th><th>Status</th><th>Copied evidence</th></tr></thead><tbody>${{rows||'<tr><td colspan="3">No saved turns</td></tr>'}}</tbody></table></details></section>`}}
function render(){{const c=byId(DATA.cases,caseSelect.value),pair=JSON.parse(pairSelect.value),checkpoint=checkpointSelect.value;const left=DATA.rows.find(r=>r.case_id===c.id&&r.arm_id===pair[0]&&r.checkpoint_id===checkpoint);const right=DATA.rows.find(r=>r.case_id===c.id&&r.arm_id===pair[1]&&r.checkpoint_id===checkpoint);content.innerHTML=`<h2>${{esc(c.id)}} <span class="reveal-only">· ${{esc(c.category||c.source_group)}}</span></h2><div class="case-prompt">${{esc(c.prompt||c.prompt_text||'Prompt unavailable')}}</div><div class="pair">${{panel(left,'LEFT')}}${{panel(right,'RIGHT')}}</div>`}}
function populatePairs(){{const selected=pairSelect.value;pairSelect.innerHTML='';DATA.pair_options.forEach(p=>pairSelect.add(new Option(reveal.checked?p.revealed_label:p.blind_label,JSON.stringify(p.ids))));if(selected)pairSelect.value=selected}}
DATA.cases.forEach(c=>caseSelect.add(new Option(c.id,c.id))); DATA.checkpoints.forEach(c=>checkpointSelect.add(new Option(c,c)));caseSelect.value=DATA.cases[0].id;checkpointSelect.value=DATA.checkpoints[0];
function sync(){{document.body.classList.toggle('revealed',reveal.checked);populatePairs();if(!pairSelect.value)pairSelect.value=JSON.stringify(DATA.pairs[0]);render()}}[caseSelect,pairSelect,checkpointSelect,reveal].forEach(x=>x.addEventListener('change',sync));sync();</script></body></html>'''


def build_gallery(manifest_path: Path, output_root: Path, *, checkpoint_id: str | None = None) -> dict[str, Any]:
    manifest, manifest_hash = load_manifest(manifest_path)
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    assets_root = output_root / "assets"
    if assets_root.exists():
        shutil.rmtree(assets_root)
    assets_root.mkdir(parents=True)
    checkpoint_ids = _common_checkpoints(manifest["arms"], checkpoint_id)
    cases: list[dict[str, Any]] = []
    for raw_case in manifest["cases"]:
        case = dict(raw_case)
        prompt_text, prompt_asset, prompt_source = _materialize_prompt(case, Path(manifest["manifest_path"]), assets_root)
        case["prompt_text"] = prompt_text
        case["prompt_asset"] = prompt_asset
        case["prompt_source"] = prompt_source
        cases.append(case)
    arm_meta = [{key: value for key, value in arm.items() if key != "checkpoints"} for arm in manifest["arms"]]
    reference_meta = [{key: value for key, value in reference.items() if key != "checkpoints"} for reference in manifest["references"]]
    selected = manifest["arms"] + manifest["references"]
    rows: list[dict[str, Any]] = []
    selected_meta: dict[tuple[str, str], Any] = {}
    for display_checkpoint in checkpoint_ids:
        for arm in selected:
            ids = {item["id"] for item in arm["checkpoints"]}
            if display_checkpoint in ids:
                cp = _pick_checkpoint(arm, display_checkpoint)
            elif arm in manifest["arms"]:
                raise ValueError(f"required checkpoint {display_checkpoint} missing from arm {arm['id']}")
            elif len(arm["checkpoints"]) == 1:
                # Optional references are allowed to use a natural id such as
                # sft512; their one checkpoint is paired at every selected
                # arm checkpoint while retaining actual_checkpoint_id.
                cp = arm["checkpoints"][0]
            else:
                continue
            selected_meta[(arm["id"], display_checkpoint)] = cp
            for case in cases:
                rows.append(_build_case(
                    case, cp, arm_id=arm["id"], assets_root=assets_root,
                    display_checkpoint_id=display_checkpoint,
                ))
    arm_ids = [arm["id"] for arm in manifest["arms"]]
    reference_ids = [reference["id"] for reference in manifest["references"]
                     if any((reference["id"], checkpoint) in selected_meta for checkpoint in checkpoint_ids)]
    pairs = [[arm_ids[i], arm_ids[j]] for i in range(len(arm_ids)) for j in range(i + 1, len(arm_ids))]
    for ref_id in reference_ids:
        pairs.extend([[ref_id, arm_id] for arm_id in arm_ids])
    if not pairs:
        raise ValueError("no fixed comparison pairs available")
    source_codes = {arm_id: chr(ord("A") + index) for index, arm_id in enumerate(arm_ids)}
    source_codes.update({ref_id: f"R{index + 1}" for index, ref_id in enumerate(reference_ids)})
    pair_options = [{
        "ids": pair,
        "blind_label": f"{source_codes[pair[0]]} vs {source_codes[pair[1]]}",
        "revealed_label": f"{pair[0]} vs {pair[1]}",
    } for pair in pairs]
    summary = {
        "cases": len(cases),
        "arms": len(manifest["arms"]),
        "rows": len(rows),
        "final_valid": sum(row["final_status"] == "final valid" for row in rows),
        "final_invalid": sum(row["final_valid"] is False for row in rows),
        "final_missing_or_censored": sum(row["final_valid"] is None for row in rows),
        "final_image_unverified": sum("unverified" in row["final_status"] for row in rows),
        "checkpoint": checkpoint_ids[0] if len(checkpoint_ids) == 1 else checkpoint_ids,
    }
    data = {
        "schema": SCHEMA,
        "manifest_sha256": manifest_hash,
        "manifest_path": manifest["manifest_path"],
        "selected_checkpoint": checkpoint_ids[0] if len(checkpoint_ids) == 1 else None,
        "cases": cases,
        "rows": rows,
        "pairs": pairs,
        "checkpoints": checkpoint_ids,
        "arm_meta": arm_meta,
        "reference_meta": reference_meta,
        "pair_options": pair_options,
        "summary": summary,
    }
    # Keep receipt paths and per-turn provenance in gallery.json for offline
    # review; the HTML carries the same data for a single-file portable view.
    (output_root / "gallery.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "index.html").write_text(_page(data), encoding="utf-8")
    return {"output": str(output_root / "index.html"), "gallery": str(output_root / "gallery.json"), "summary": summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", help="shared checkpoint id; default is the first common id")
    args = parser.parse_args(argv)
    report = build_gallery(args.manifest, args.output, checkpoint_id=args.checkpoint)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
