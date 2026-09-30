#!/usr/bin/env python3
"""Finite eight-call side-keyed transport probe.

The probe reuses four deterministic failed-initial pair receipts and their
hash-verified frozen PNGs.  It sends exactly two direct views per pair, with
the image order reversed, and deliberately performs no retry/recheck.  Use
``--dry-run`` to validate selection and hashes without reading a key or making
network requests.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from painter import text_pairwise_consistent_side_judge as judge  # noqa: E402


DEFAULT_MANIFEST = Path("/private/tmp/painter-sft512-cpu-evidence/extracted/anchors-frozen/train-manifest.json")
DEFAULT_RECEIPTS = Path("/private/tmp/painter-sft512-cpu-evidence/extracted/rl-rollouts")
DEFAULT_EVIDENCE_ROOT = Path("/private/tmp/painter-sft512-cpu-evidence/extracted")
DEFAULT_OUTPUT = ROOT / "painter/runs/text-sft512-consistent-20260930/local-evidence/side-transport-probe"

ROLE_ALIASES = {
    "frozen-sft": "frozen SFT",
    "rl-step-16": "RL-step-16",
    "pilot-step-24": "pilot-step-24",
    "sft": "SFT",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_id(value: str) -> str:
    if not isinstance(value, str) or not value or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in value):
        raise ValueError("unsafe pair identifier")
    return value


def _load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("input manifest must contain cases")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError("manifest case is malformed")
        result[row["id"]] = row
    if not result:
        raise ValueError("input manifest has no cases")
    return result


def _receipt_files(receipts: Path) -> list[Path]:
    if not receipts.is_dir():
        raise FileNotFoundError(receipts)
    return sorted(path for path in receipts.rglob("*.json") if path.is_file() and not path.is_symlink())


def _failed_pairs(receipts: Path) -> list[dict[str, Any]]:
    selected = []
    for path in _receipt_files(receipts):
        payload = json.loads(path.read_text(encoding="utf-8"))
        # Native rollout receipts are flat judge objects.  The older alignment
        # wrapper is still recognized for offline tests, but is not the default
        # probe source.
        judge_payload = payload.get("judge") if isinstance(payload, dict) and isinstance(payload.get("judge"), dict) else payload
        identity = payload.get("identity") if isinstance(payload, dict) else None
        if not isinstance(judge_payload, dict) or judge_payload.get("consensus") != "failed_initial_call":
            continue
        if isinstance(identity, dict) and isinstance(identity.get("pair_id"), str):
            pair_id = identity["pair_id"]
        elif path.name == "text-judge.json" and isinstance(judge_payload.get("image_sha256"), dict):
            pair_id = f"{path.parent.name}"
        else:
            continue
        selected.append({"path": path, "receipt": payload, "judge": judge_payload,
                         "identity": identity or {"pair_id": pair_id}, "pair_id": pair_id})
    selected.sort(key=lambda row: row["pair_id"])
    return selected


def _role_name(token: str) -> str:
    try:
        return ROLE_ALIASES[token]
    except KeyError as exc:
        raise ValueError(f"unsupported pair role: {token}") from exc


def _resolve_image(evidence_root: Path, manifest_path: Path, image: dict[str, Any]) -> Path:
    relative = image.get("path")
    expected = image.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected, str):
        raise ValueError("manifest image path/hash is missing")
    candidates = [manifest_path.parent / relative, evidence_root / relative, evidence_root / "images" / Path(relative).name]
    candidates.extend(evidence_root.rglob(Path(relative).name) if evidence_root.is_dir() else ())
    matches = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file() and not candidate.is_symlink() and candidate not in matches:
            matches.append(candidate)
    hashed = [candidate for candidate in matches if sha256(candidate) == expected]
    if len(hashed) != 1:
        raise ValueError(f"could not resolve one hash-verified image: {Path(relative).name}")
    return hashed[0]


def _find_hash_file(evidence_root: Path, expected: str, suffixes: tuple[str, ...]) -> Path:
    matches = []
    for candidate in evidence_root.rglob("*"):
        if not candidate.is_file() or candidate.is_symlink() or candidate.suffix.lower() not in suffixes:
            continue
        if sha256(candidate) == expected:
            matches.append(candidate.resolve())
    if len(matches) != 1:
        raise ValueError(f"expected one hash-verified archive file, found {len(matches)}")
    return matches[0]


def _prepare_native_pairs(*, manifest_path: Path, receipts: Path, evidence_root: Path, limit: int = 4) -> list[dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("native train manifest must contain rows")
    by_prompt = {row.get("reference_sha256"): row for row in rows if isinstance(row, dict)}
    prepared = []
    for item in _failed_pairs(receipts):
        judge_payload = item["judge"]
        image_hashes = judge_payload.get("image_sha256")
        brief_sha = judge_payload.get("brief_sha256")
        if not isinstance(image_hashes, dict) or not isinstance(brief_sha, str):
            continue
        anchor_sha = image_hashes.get("anchor")
        candidate_sha = image_hashes.get("candidate")
        row = by_prompt.get(brief_sha)
        if not isinstance(anchor_sha, str) or not isinstance(candidate_sha, str) or not isinstance(row, dict):
            continue
        if row.get("opponent_sha256") != anchor_sha:
            raise ValueError(f"native anchor hash mismatch for {item['pair_id']}")
        brief_path = _find_hash_file(evidence_root, brief_sha, (".txt",))
        left_path = _find_hash_file(evidence_root, anchor_sha, (".png",))
        right_path = _find_hash_file(evidence_root, candidate_sha, (".png",))
        raw_brief = brief_path.read_text(encoding="utf-8")
        if hashlib.sha256(raw_brief.encode("utf-8")).hexdigest() != brief_sha:
            raise ValueError(f"brief bytes changed for {item['pair_id']}")
        trace_id = _safe_id(item["pair_id"])
        pair_id = f"{row['id']}__frozen-sft512__current-sft512-{trace_id}"
        prepared.append({
            "pair_id": pair_id,
            "case_id": row["id"],
            "brief": raw_brief.strip(),
            "brief_sha256": brief_sha,
            "left_role": "frozen SFT512",
            "right_role": "current SFT512 rollout",
            "left_path": left_path,
            "right_path": right_path,
            "left_sha256": anchor_sha,
            "right_sha256": candidate_sha,
            "source_receipt": str(item["path"]),
        })
        if len(prepared) >= limit:
            break
    if len(prepared) != limit:
        raise ValueError(f"need {limit} native failed-initial pairs; found {len(prepared)}")
    return prepared


def _prepare_pairs(*, manifest_path: Path, receipts: Path, evidence_root: Path, limit: int = 4) -> list[dict[str, Any]]:
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(manifest_payload, dict) and isinstance(manifest_payload.get("rows"), list):
        return _prepare_native_pairs(manifest_path=manifest_path, receipts=receipts, evidence_root=evidence_root, limit=limit)
    cases = _load_manifest(manifest_path)
    failed = _failed_pairs(receipts)
    prepared = []
    for row in failed:
        pair_id = _safe_id(row["identity"]["pair_id"])
        parts = pair_id.split("__")
        if len(parts) != 3:
            continue
        case_id, left_token, right_token = parts
        case = cases.get(case_id)
        if case is None:
            continue
        left_role, right_role = _role_name(left_token), _role_name(right_token)
        images = case.get("images")
        if not isinstance(images, dict) or left_role not in images or right_role not in images:
            continue
        left_path = _resolve_image(evidence_root, manifest_path, images[left_role])
        right_path = _resolve_image(evidence_root, manifest_path, images[right_role])
        image_hashes = row["identity"].get("image_sha256")
        if isinstance(image_hashes, dict):
            if image_hashes.get(left_role) != sha256(left_path) or image_hashes.get(right_role) != sha256(right_path):
                raise ValueError(f"receipt/image hash mismatch for {pair_id}")
        brief = case.get("brief")
        if not isinstance(brief, str) or not brief.strip():
            raise ValueError(f"missing brief for {case_id}")
        brief_sha = hashlib.sha256(brief.strip().encode("utf-8")).hexdigest()
        if case.get("brief_sha256") not in {None, brief_sha}:
            raise ValueError(f"brief hash mismatch for {case_id}")
        prepared.append({
            "pair_id": pair_id,
            "case_id": case_id,
            "brief": brief.strip(),
            "brief_sha256": brief_sha,
            "left_role": left_role,
            "right_role": right_role,
            "left_path": left_path,
            "right_path": right_path,
            "left_sha256": sha256(left_path),
            "right_sha256": sha256(right_path),
            "source_receipt": str(row["path"]),
        })
        if len(prepared) >= limit:
            break
    if len(prepared) != limit:
        raise ValueError(f"need {limit} hash-verified failed-initial pairs; found {len(prepared)}")
    return prepared


def _cost(usage: Any) -> float | None:
    if not isinstance(usage, dict):
        return None
    for key in ("cost", "gateway_cost", "market_cost"):
        value = usage.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _call_record(pair: dict[str, Any], *, side: str, call: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "side-transport-probe-call-v1",
        "pair_id": pair["pair_id"],
        "case_id": pair["case_id"],
        "display_side": side,
        "request_order": {"A": pair["left_role"] if side == "left-to-right" else pair["right_role"],
                           "B": pair["right_role"] if side == "left-to-right" else pair["left_role"]},
        "brief_sha256": pair["brief_sha256"],
        "image_sha256": {"A": call.get("image_sha256", {}).get("A"), "B": call.get("image_sha256", {}).get("B")},
        "image_ids": call.get("image_ids"),
        "sheet_sha256": call.get("sheet_sha256"),
        "prompt_sha256": call.get("prompt_sha256"),
        "format_version": call.get("format_version"),
        "source_format_sha256": call.get("source_format_sha256"),
        "status": "parsed" if "assessment" in call else "censored",
        "parse_error": call.get("parse_error"),
        "provider_error": call.get("provider_error"),
        "finish_reason": call.get("finish_reason"),
        "usage": call.get("usage"),
        "cost": _cost(call.get("usage")),
        "elapsed_seconds": call.get("elapsed_seconds"),
        "raw_response": call.get("raw_response", ""),
        "assessment": call.get("assessment"),
    }


async def run_probe(*, pairs: list[dict[str, Any]], output: Path, key_file: Path | None, dry_run: bool) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema": "side-transport-probe-v1",
        "format_version": judge.FORMAT_VERSION,
        "source_format_sha256": judge.SOURCE_FORMAT_SHA256,
        "prompt_sha256": hashlib.sha256(judge.PROMPT.encode()).hexdigest(),
        "calls_planned": len(pairs) * 2,
        "pairs": [{key: value for key, value in pair.items() if key.endswith("_sha256") or key in {"pair_id", "case_id", "left_role", "right_role", "source_receipt"}} for pair in pairs],
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if dry_run:
        return {**plan, "status": "dry_run", "calls_completed": 0}
    if key_file is None:
        raise ValueError("--key-file is required unless --dry-run is used")
    key = judge.load_key(key_file)
    async def one(pair: dict[str, Any], side: str) -> dict[str, Any]:
        if side == "left-to-right":
            a, b = pair["left_path"], pair["right_path"]
        else:
            a, b = pair["right_path"], pair["left_path"]
        call = await asyncio.to_thread(judge._view, key=key, brief=pair["brief"], a=a, b=b,
                                        a_id=judge.image_id(a), b_id=judge.image_id(b))
        record = _call_record(pair, side=side, call=call)
        name = pair["pair_id"] + "--" + side + ".json"
        (output / name).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return record
    results = await asyncio.gather(*(one(pair, side) for pair in pairs for side in ("left-to-right", "right-to-left")))
    summary = {
        **plan,
        "status": "complete",
        "calls_completed": len(results),
        "parsed": sum(record["status"] == "parsed" for record in results),
        "censored": sum(record["status"] != "parsed" for record in results),
        "calls": [{key: value for key, value in record.items() if key != "raw_response" and key != "assessment"} for record in results],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--receipts", type=Path, default=DEFAULT_RECEIPTS)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    pairs = _prepare_pairs(manifest_path=args.manifest, receipts=args.receipts, evidence_root=args.evidence_root)
    result = asyncio.run(run_probe(pairs=pairs, output=args.output, key_file=args.key_file, dry_run=args.dry_run))
    print(json.dumps({key: value for key, value in result.items() if key not in {"pairs", "calls"}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
