"""Deterministic offline audit for preserved painter rollouts and rewards.

The audit deliberately treats ``result.json`` as the source of action and
``sample.json`` as the source of state.  In particular, a result whose canvas
is the shared state canvas is still joined using its recorded action; the
canvas path is never used to infer whether the transition was a paint or a
finish action.

This module is read-only with respect to evidence.  It accepts either an
evidence directory or a ``final-evidence.tar.gz`` archive and returns a JSON
serialisable report.  Raw labels are read only to derive v1/v2 normalized
signals; label observations and reasons are never copied into the report.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from quality_signal import reconcile as reconcile_v1
from quality_signal import reward as reward_v1
from quality_signal_v2 import reconcile_v2, reward_v2


AUDIT_VERSION = "offline_reward_audit_v1"
ROLES = ("reference", "anchor", "candidate")
VALID_ACTIONS = frozenset(("paint", "repair", "finish"))
JUDGE_MODEL = "gpt-5.6-luna"
_TOKEN_ID_KEYS = frozenset((
    "token_ids", "completion_token_ids", "output_token_ids", "generated_token_ids",
))
_LOGPROB_KEYS = frozenset((
    "logprobs", "completion_logprobs", "output_logprobs", "generated_logprobs",
))
_MASK_KEYS = frozenset(("mask", "attention_mask", "loss_mask"))


def request_key(image_sha256: Mapping[str, str]) -> str:
    """Return the exact visual request key used by ``taskset_impl.py``."""

    if set(image_sha256) != set(ROLES):
        raise ValueError("image_sha256 must contain reference, anchor, candidate")
    if not all(isinstance(image_sha256[key], str) for key in ROLES):
        raise ValueError("image_sha256 values must be strings")
    payload = json.dumps(dict(image_sha256), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _safe_path(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    candidate = path if path.is_absolute() else root / path
    try:
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root.resolve()):
            return None
    except (OSError, ValueError):
        return None
    return resolved


def _safe_extract(archive: Path, destination: Path) -> None:
    """Extract an evidence archive without allowing path traversal."""

    base = destination.resolve()
    with tarfile.open(archive, "r:*") as tar:
        members = tar.getmembers()
        for member in members:
            target = (base / member.name).resolve()
            if not target.is_relative_to(base):
                raise ValueError(f"unsafe archive member: {member.name}")
            if member.issym() or member.islnk():
                link_target = (target.parent / member.linkname).resolve()
                if not link_target.is_relative_to(base):
                    raise ValueError(f"unsafe archive link: {member.name}")
        tar.extractall(base, members=members)


class _Evidence:
    def __init__(self, source: Path):
        self.source = source
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self.root: Path | None = None

    def __enter__(self) -> Path:
        if self.source.is_dir():
            self.root = self.source.resolve()
            return self.root
        if not self.source.is_file():
            raise FileNotFoundError(self.source)
        self._temporary = tempfile.TemporaryDirectory(prefix="painter-offline-audit-")
        destination = Path(self._temporary.name)
        _safe_extract(self.source, destination)
        self.root = destination
        return destination

    def __exit__(self, exc_type, exc, traceback):
        if self._temporary is not None:
            self._temporary.cleanup()
        self.root = None


def _find_archive_root(root: Path) -> Path:
    """Accept archives that contain a single enclosing directory."""

    children = [path for path in root.iterdir() if path.name != ".DS_Store"]
    if len(children) == 1 and children[0].is_dir():
        child = children[0]
        if (child / "rl-rollouts").exists() or (child / "reward-batches").exists():
            return child
    return root


def _native_value(value: Any) -> bool:
    return isinstance(value, list) and bool(value)


def _has_native_field(value: Any, keys: frozenset[str]) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in keys and _native_value(child):
                return True
            if _has_native_field(child, keys):
                return True
    elif isinstance(value, list):
        return any(_has_native_field(child, keys) for child in value)
    return False


def _group_id(value: Any) -> str | None:
    if isinstance(value, dict):
        group = value.get("group")
        if isinstance(group, dict) and isinstance(group.get("id"), str):
            return group["id"]
        for key in ("group_id", "native_group_id"):
            if isinstance(value.get(key), str):
                return value[key]
    return None


def _trace_provenance(root: Path) -> dict[str, dict[str, Any]]:
    """Read native trace metadata without retaining prompts or raw responses."""

    by_rollout: dict[str, dict[str, Any]] = {}
    duplicate_traces: list[str] = []
    paths = sorted(path for path in root.rglob("*.jsonl")
                   if any("trace" in part for part in path.parts))
    for path in paths:
        try:
            lines = path.open()
        except OSError:
            continue
        with lines:
            for line_number, line in enumerate(lines, 1):
                try:
                    envelope = json.loads(line)
                except (TypeError, ValueError):
                    continue
                traces = envelope.get("traces") if isinstance(envelope, dict) else None
                if not isinstance(traces, list):
                    continue
                native_group = _group_id(envelope)
                for trace in traces:
                    if not isinstance(trace, dict) or not isinstance(trace.get("id"), str):
                        continue
                    rollout_id = trace["id"]
                    row = {
                        "group_id": native_group,
                        "token_ids_present": _has_native_field(trace, _TOKEN_ID_KEYS),
                        "logprobs_present": _has_native_field(trace, _LOGPROB_KEYS),
                        "mask_present": _has_native_field(trace, _MASK_KEYS),
                        "source": _rel(path, root),
                        "line": line_number,
                    }
                    if rollout_id in by_rollout:
                        duplicate_traces.append(rollout_id)
                        # Stable first-wins behavior keeps the report deterministic.
                        continue
                    by_rollout[rollout_id] = row
    return {
        "by_rollout": by_rollout,
        "duplicate_rollout_ids": sorted(set(duplicate_traces)),
    }


def _auto_group_size(root: Path) -> int | None:
    candidates: list[tuple[str, int]] = []
    for path in sorted(root.rglob("orchestrator.json")):
        try:
            doc = _json(path)
        except (OSError, ValueError):
            continue
        values = []
        if isinstance(doc, dict):
            values.append(doc.get("group_size"))
            train = doc.get("train")
            if isinstance(train, dict):
                sources = train.get("source")
                if isinstance(sources, list):
                    values.extend(
                        source.get("group_size") for source in sources
                        if isinstance(source, dict)
                    )
        for value in values:
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                candidates.append((_rel(path, root), value))
    if not candidates:
        return None
    return sorted(candidates)[0][1]


def _exclusions(source: Path | Mapping[str, Any] | None) -> dict[str, str]:
    if source is None:
        return {}
    doc = _json(source) if isinstance(source, Path) else source
    if isinstance(doc, dict) and isinstance(doc.get("exclusions"), list):
        doc = doc["exclusions"]
    elif isinstance(doc, dict) and isinstance(doc.get("excluded_requests"), (dict, list)):
        doc = doc["excluded_requests"]
    elif isinstance(doc, dict) and isinstance(doc.get("requests"), dict):
        doc = doc["requests"]
    values: dict[str, str] = {}
    if isinstance(doc, list):
        for entry in doc:
            if not isinstance(entry, dict) or not isinstance(entry.get("request_id"), str):
                raise ValueError("each exclusion needs a string request_id")
            request_id = entry["request_id"]
            reason = entry.get("reason", "")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"exclusion {request_id} needs a non-empty reason")
            values[request_id] = reason.strip()
        return values
    if isinstance(doc, dict):
        for request_id, entry in doc.items():
            if not isinstance(request_id, str):
                raise ValueError("exclusion request IDs must be strings")
            if isinstance(entry, str) and entry.strip():
                values[request_id] = entry.strip()
            elif isinstance(entry, dict) and isinstance(entry.get("reason"), str) and entry["reason"].strip():
                values[request_id] = entry["reason"].strip()
            elif entry is True:
                values[request_id] = "explicit grounding exclusion"
            else:
                raise ValueError(f"exclusion {request_id} needs a non-empty reason")
        return values
    raise ValueError("exclusions must be a mapping or list")


def _receipt_paths(root: Path) -> list[Path]:
    paths = sorted(root.glob("judge-queue/*/receipt.json"))
    if (root / "receipts").is_dir():
        paths.extend(root.glob("receipts/*/receipt.json"))
    elif (root / "reward-batches").is_dir():
        paths.extend(root.glob("reward-batches/*/receipts/*/receipt.json"))
    else:
        paths.extend(root.glob("*/receipts/*/receipt.json"))
    return sorted(set(paths))


def _load_receipts(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    receipts: dict[str, dict[str, Any]] = {}
    duplicates: dict[str, list[str]] = collections.defaultdict(list)
    for path in _receipt_paths(root):
        try:
            doc = _json(path)
        except (OSError, ValueError):
            continue
        request_id = doc.get("id") if isinstance(doc, dict) else None
        if not isinstance(request_id, str):
            continue
        duplicates[request_id].append(_rel(path, root))
        doc["_audit_source"] = _rel(path, root)
        receipts.setdefault(request_id, doc)
    return receipts, {key: value for key, value in duplicates.items() if len(value) > 1}


def _load_label_fallback(root: Path) -> dict[str, dict[str, Any]]:
    """Reconstruct receipt-like views from identity keys and labels when needed."""

    grouped: dict[str, dict[str, Any]] = collections.defaultdict(dict)
    if (root / "identity-key.json").is_file():
        batches = [root]
    elif (root / "reward-batches").is_dir():
        batches = sorted((root / "reward-batches").glob("*"))
    elif root.name == "reward-batches":
        batches = sorted(root.glob("*"))
    else:
        batches = sorted(root.glob("*/"))
    for batch in batches:
        key_path = batch / "identity-key.json"
        if not key_path.is_file():
            continue
        try:
            key_doc = _json(key_path)
        except (OSError, ValueError):
            continue
        if not isinstance(key_doc, dict):
            continue
        view_info = {
            ident: row for ident, row in key_doc.items()
            if isinstance(ident, str) and isinstance(row, dict)
        }
        for label_path in sorted(batch.glob("labels-*.json")):
            try:
                doc = _json(label_path)
            except (OSError, ValueError):
                continue
            for label in doc.get("judgments", []) if isinstance(doc, dict) else []:
                if not isinstance(label, dict) or not isinstance(label.get("id"), str):
                    continue
                info = view_info.get(label["id"])
                if not info or not isinstance(info.get("request_id"), str):
                    continue
                side = info.get("candidate_side")
                if side not in ("A", "B"):
                    continue
                grouped[info["request_id"]][side] = label
    # Build canonical rows from the identity key itself, retaining labels only
    # as in-memory inputs to the normalization functions.
    result: dict[str, dict[str, Any]] = {}
    for batch in batches:
        key_path = batch / "identity-key.json"
        if not key_path.is_file():
            continue
        try:
            key_doc = _json(key_path)
        except (OSError, ValueError):
            continue
        requests: dict[str, dict[str, Any]] = {}
        for ident, info in key_doc.items() if isinstance(key_doc, dict) else []:
            if not isinstance(info, dict) or not isinstance(info.get("request_id"), str):
                continue
            rid = info["request_id"]
            hashes = info.get("image_sha256")
            side = info.get("candidate_side")
            if not isinstance(hashes, dict) or side not in ("A", "B"):
                continue
            entry = requests.setdefault(rid, {"id": rid, "views": {}})
            entry["image_sha256"] = {
                "reference": hashes.get("reference"),
                "anchor": hashes.get("B" if side == "A" else "A"),
                "candidate": hashes.get(side),
            }
            entry["views"][side] = grouped.get(rid, {}).get(side)
        for rid, entry in requests.items():
            if rid in result:
                continue
            result[rid] = {
                "id": rid,
                "image_sha256": entry.get("image_sha256"),
                "candidate_A": entry["views"].get("A"),
                "candidate_B": entry["views"].get("B"),
                "_source": "identity-key-and-labels",
            }
    return result


def _judgments(receipt: Mapping[str, Any] | None) -> tuple[Any, Any]:
    if not isinstance(receipt, Mapping):
        return None, None
    first, reverse = receipt.get("candidate_A"), receipt.get("candidate_B")
    v1 = v2 = None
    if isinstance(first, dict) and isinstance(reverse, dict):
        try:
            v1 = reconcile_v1(first, reverse)
        except (KeyError, TypeError, ValueError):
            v1 = None
        try:
            v2 = reconcile_v2(first, reverse)
        except (KeyError, TypeError, ValueError):
            v2 = None
    return v1, v2


def _receipt_usable(receipt: Mapping[str, Any]) -> bool:
    """Require persisted judge and identity provenance before scoring."""

    if receipt.get("judge_model") != JUDGE_MODEL or receipt.get("reasoning_effort") != "low":
        return False
    for name in ("candidate_A", "candidate_B"):
        label = receipt.get(name)
        if not isinstance(label, Mapping):
            return False
        rolemap = label.get("rolemap")
        displayed = label.get("displayed_asset_ids")
        if (not isinstance(rolemap, Mapping)
                or set(rolemap) != {"reference", "A", "B"}
                or not all(isinstance(value, str) and value for value in rolemap.values())
                or not isinstance(displayed, list)
                or displayed != list(dict.fromkeys(rolemap.values()))):
            return False
    provenance = receipt.get("visual_provenance")
    if provenance is not None and (
            not isinstance(provenance, Mapping)
            or provenance.get("mode") != "content_addressed_unique_assets"):
        return False
    return True


def _score(function, result: Mapping[str, Any], judgment: Any) -> Any:
    try:
        return function(result, judgment)
    except (KeyError, TypeError, ValueError):
        return None


def _base_mask(result: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(result, Mapping):
        return ["missing_result"]
    failure = result.get("failure_kind")
    if failure in ("infrastructure", "cutoff"):
        return [str(failure)]
    if result.get("valid") is not True:
        return ["invalid_result"]
    if result.get("action") not in VALID_ACTIONS:
        return ["invalid_action"]
    return []


def _label_mask(result: Mapping[str, Any], version: str, judgment: Any) -> list[str]:
    if judgment is None:
        return [f"{version}_judgment_unavailable"]
    action = result.get("action")
    if version == "v1":
        return ["v1_signal_unavailable"]
    if action in ("paint", "repair") and judgment.get("delta") is None:
        return ["v2_relative_signal_unavailable"]
    if action == "finish" and judgment.get("candidate_acceptable") is None:
        return ["v2_candidate_acceptability_unavailable"]
    return ["v2_signal_unavailable"]


def _native_row(sample: Mapping[str, Any], trace: Mapping[str, Any] | None) -> dict[str, Any]:
    token_ids = _has_native_field(sample, _TOKEN_ID_KEYS)
    logprobs = _has_native_field(sample, _LOGPROB_KEYS)
    mask = _has_native_field(sample, _MASK_KEYS)
    group_id = sample.get("group_id") if isinstance(sample.get("group_id"), str) else None
    source = "sample.json"
    if trace:
        token_ids = token_ids or bool(trace.get("token_ids_present"))
        logprobs = logprobs or bool(trace.get("logprobs_present"))
        mask = mask or bool(trace.get("mask_present"))
        if group_id is None and isinstance(trace.get("group_id"), str):
            group_id = trace["group_id"]
            source = trace.get("source", "trace")
    return {
        "token_ids": "present" if token_ids else "missing",
        "logprobs": "present" if logprobs else "missing",
        "mask": "present" if mask else "missing",
        "group_id": "present" if group_id else "missing",
        "group_id_value": group_id,
        "source": source,
    }


def _rollout_rows(root: Path, receipts: Mapping[str, dict[str, Any]],
                  fallback: Mapping[str, dict[str, Any]], traces: Mapping[str, dict[str, Any]],
                  exclusions: Mapping[str, str]) -> list[dict[str, Any]]:
    folders = {path.parent for path in root.glob("rl-rollouts/*/sample.json")}
    folders.update(path.parent for path in root.glob("rl-rollouts/*/result.json"))
    rows: list[dict[str, Any]] = []
    for folder in sorted(folders, key=lambda path: path.as_posix()):
        sample_path, result_path = folder / "sample.json", folder / "result.json"
        try:
            sample = _json(sample_path) if sample_path.is_file() else None
        except (OSError, ValueError):
            sample = None
        try:
            result = _json(result_path) if result_path.is_file() else None
        except (OSError, ValueError):
            result = None
        sample_id = sample.get("id") if isinstance(sample, dict) else None
        result_id = result.get("id") if isinstance(result, dict) else None
        rollout_id = sample_id if isinstance(sample_id, str) else result_id
        if not isinstance(rollout_id, str):
            rollout_id = folder.name
        sample_state = sample.get("state") if isinstance(sample, dict) else None
        state_id = sample_state.get("id") if isinstance(sample_state, dict) else None
        result_state_id = result.get("state_id") if isinstance(result, dict) else None
        row: dict[str, Any] = {
            "rollout_id": rollout_id,
            "state_id": state_id if isinstance(state_id, str) else result_state_id,
            "source_group": sample_state.get("source_group") if isinstance(sample_state, dict) else None,
            "action": result.get("action") if isinstance(result, dict) else None,
            "valid": result.get("valid") if isinstance(result, dict) else None,
            "no_change": result.get("no_change") if isinstance(result, dict) else None,
            "failure_kind": result.get("failure_kind") if isinstance(result, dict) else None,
            "sample_path": _rel(sample_path, root) if sample_path.is_file() else None,
            "result_path": _rel(result_path, root) if result_path.is_file() else None,
            "request_id": None,
            "receipt_id": None,
            "receipt_source": None,
            "computed_image_sha256": None,
            "hash_match": None,
            "duplicate_request_count": 0,
            "excluded": False,
            "exclusion_reason": None,
            "v1_reward": None,
            "v2_reward": None,
            "v1_masked_reasons": [],
            "v2_masked_reasons": [],
            "stored_reward": None,
            "stored_reward_match": None,
            "native_provenance": _native_row(
                sample if isinstance(sample, dict) else {}, traces.get(rollout_id)
            ),
        }
        masks = _base_mask(result)
        if sample is None:
            masks.append("missing_sample")
        if isinstance(sample, dict) and isinstance(result, dict):
            if isinstance(sample_id, str) and isinstance(result_id, str) and sample_id != result_id:
                masks.append("rollout_id_mismatch")
            if isinstance(state_id, str) and isinstance(result_state_id, str) and state_id != result_state_id:
                masks.append("state_id_mismatch")
        visual_ready = (
            isinstance(sample_state, dict)
            and isinstance(result, dict)
            and result.get("valid") is True
            and result.get("action") in VALID_ACTIONS
        )
        if visual_ready:
            paths = {
                "reference": _safe_path(root, sample_state.get("reference")),
                "anchor": _safe_path(root, sample_state.get("canvas")),
                "candidate": _safe_path(root, result.get("canvas")),
            }
            if any(path is None or not path.is_file() for path in paths.values()):
                masks.append("missing_visual_artifact")
            else:
                hashes = {key: _sha(path) for key, path in paths.items() if path is not None}
                row["computed_image_sha256"] = hashes
                try:
                    row["request_id"] = request_key(hashes)
                except ValueError:
                    masks.append("invalid_image_hashes")
                receipt = receipts.get(row["request_id"]) or fallback.get(row["request_id"])
                if receipt is not None:
                    row["receipt_id"] = receipt.get("id")
                    row["receipt_source"] = receipt.get("_audit_source")
                    expected = receipt.get("image_sha256")
                    row["hash_match"] = expected == hashes
                    if expected != hashes:
                        masks.append("hash_mismatch")
                    if receipt.get("id") != row["request_id"]:
                        masks.append("request_key_mismatch")
                    if not _receipt_usable(receipt):
                        masks.append("receipt_provenance_mismatch")
                    v1_judgment, v2_judgment = _judgments(receipt)
                    if (row["hash_match"] is True
                            and not exclusions.get(row["request_id"])
                            and "receipt_provenance_mismatch" not in masks):
                        row["v1_reward"] = _score(reward_v1, result, v1_judgment)
                        row["v2_reward"] = _score(reward_v2, result, v2_judgment)
                        if row["v1_reward"] is None:
                            row["v1_masked_reasons"] = _label_mask(result, "v1", v1_judgment)
                        if row["v2_reward"] is None:
                            row["v2_masked_reasons"] = _label_mask(result, "v2", v2_judgment)
                    else:
                        if row["hash_match"] is not True:
                            row["v1_masked_reasons"] = ["hash_mismatch"]
                            row["v2_masked_reasons"] = ["hash_mismatch"]
                            if "receipt_provenance_mismatch" in masks:
                                row["v1_masked_reasons"].append("receipt_provenance_mismatch")
                                row["v2_masked_reasons"].append("receipt_provenance_mismatch")
                        elif row["request_id"] in exclusions:
                            row["excluded"] = True
                            row["exclusion_reason"] = exclusions[row["request_id"]]
                            row["v1_masked_reasons"] = ["grounding_exclusion"]
                            row["v2_masked_reasons"] = ["grounding_exclusion"]
                            if "receipt_provenance_mismatch" in masks:
                                row["v1_masked_reasons"].append("receipt_provenance_mismatch")
                                row["v2_masked_reasons"].append("receipt_provenance_mismatch")
                        else:
                            row["v1_masked_reasons"] = ["receipt_provenance_mismatch"]
                            row["v2_masked_reasons"] = ["receipt_provenance_mismatch"]
                            if v1_judgment is None:
                                row["v1_masked_reasons"].append("v1_judgment_unavailable")
                            if v2_judgment is None:
                                row["v2_masked_reasons"].append("v2_judgment_unavailable")
                else:
                    row["v1_masked_reasons"] = ["v1_judgment_unavailable"]
                    row["v2_masked_reasons"] = ["v2_judgment_unavailable"]
                if row["request_id"] in exclusions and not row["excluded"]:
                    row["excluded"] = True
                    row["exclusion_reason"] = exclusions[row["request_id"]]
                    row["v1_reward"] = row["v2_reward"] = None
                    row["v1_masked_reasons"] = list(dict.fromkeys(
                        row["v1_masked_reasons"] + ["grounding_exclusion"]
                    ))
                    row["v2_masked_reasons"] = list(dict.fromkeys(
                        row["v2_masked_reasons"] + ["grounding_exclusion"]
                    ))
        else:
            # Invalid-program penalties are replayable without visual
            # artifacts or a judgment. Operational failures remain masked.
            row["v1_reward"] = _score(reward_v1, result, None) if isinstance(result, dict) else None
            row["v2_reward"] = _score(reward_v2, result, None) if isinstance(result, dict) else None
            row["v1_masked_reasons"] = ([] if row["v1_reward"] is not None
                                         else list(dict.fromkeys(masks)))
            row["v2_masked_reasons"] = ([] if row["v2_reward"] is not None
                                         else list(dict.fromkeys(masks)))
        # Structural joins always mask visual rewards. Keep the explicit
        # invalid-program penalty, which does not depend on a visual join.
        structural_masks = [mask for mask in dict.fromkeys(masks)
                            if not (isinstance(result, dict)
                                    and result.get("failure_kind") == "program"
                                    and result.get("valid") is not True)]
        if structural_masks:
            row["v1_reward"] = row["v2_reward"] = None
            row["v1_masked_reasons"] = list(dict.fromkeys(
                row["v1_masked_reasons"] + structural_masks
            ))
            row["v2_masked_reasons"] = list(dict.fromkeys(
                row["v2_masked_reasons"] + structural_masks
            ))
        stored_path = folder / "reward.json"
        if stored_path.is_file():
            try:
                stored = _json(stored_path)
            except (OSError, ValueError):
                stored = None
            if isinstance(stored, dict):
                row["stored_reward"] = stored.get("reward")
                if row["v1_reward"] is not None and isinstance(row["stored_reward"], (int, float)):
                    row["stored_reward_match"] = row["v1_reward"] == row["stored_reward"]
        rows.append(row)
    return rows


def audit(evidence: str | Path, *, exclusions: str | Path | Mapping[str, Any] | None = None,
          expected_group_size: int | None = None,
          reward_batches: str | Path | Iterable[str | Path] | None = None) -> dict[str, Any]:
    """Audit preserved evidence and return a deterministic report."""

    with _Evidence(Path(evidence)) as extracted:
        root = _find_archive_root(extracted)
        exclusion_map = _exclusions(Path(exclusions) if isinstance(exclusions, (str, Path)) else exclusions)
        receipts, duplicate_receipts = _load_receipts(root)
        fallback = _load_label_fallback(root)
        if reward_batches is not None:
            batch_roots = ([reward_batches] if isinstance(reward_batches, (str, Path))
                           else list(reward_batches))
            for batch_source in batch_roots:
                batch_root = Path(batch_source).resolve()
                extra_receipts, extra_duplicates = _load_receipts(batch_root)
                receipts.update({key: value for key, value in extra_receipts.items()
                                 if key not in receipts})
                fallback.update({key: value for key, value in _load_label_fallback(batch_root).items()
                                 if key not in fallback})
                duplicate_receipts.update(extra_duplicates)
        trace_doc = _trace_provenance(root)
        rows = _rollout_rows(root, receipts, fallback, trace_doc["by_rollout"], exclusion_map)
        request_counts = collections.Counter(
            row["request_id"] for row in rows if isinstance(row.get("request_id"), str)
        )
        for row in rows:
            if row.get("request_id") in request_counts:
                row["duplicate_request_count"] = request_counts[row["request_id"]]
        group_size = expected_group_size if expected_group_size is not None else _auto_group_size(root)
        groups: dict[str, list[str]] = collections.defaultdict(list)
        for row in rows:
            group_id = row["native_provenance"].get("group_id_value")
            if isinstance(group_id, str):
                groups[group_id].append(row["rollout_id"])
        group_report = []
        for group_id in sorted(groups):
            members = sorted(groups[group_id])
            complete = None if group_size is None else len(members) == group_size
            group_report.append({
                "group_id": group_id,
                "member_count": len(members),
                "member_rollout_ids": members,
                "expected_member_count": group_size,
                "status": "observed_only" if complete is None else ("complete" if complete else "incomplete"),
            })
        signal_groups: list[dict[str, Any]] = []
        for group_id in sorted(groups):
            members = [row for row in rows
                       if row["native_provenance"].get("group_id_value") == group_id]
            signals: dict[str, dict[str, Any]] = {}
            for version, field in (("v1", "v1_reward"), ("v2", "v2_reward")):
                values = [row[field] for row in members
                          if isinstance(row.get(field), (int, float))]
                mean = sum(values) / len(values) if values else None
                variance = (sum((value - mean) ** 2 for value in values) / len(values)
                            if values else None)
                signals[version] = {
                    "scored_count": len(values),
                    "nonzero_variance": bool(variance is not None and variance > 1e-12),
                    "variance": variance,
                    "eligible_for_group_signal": bool(
                        len(values) >= 4 and variance is not None and variance > 1e-12
                    ),
                }
            signal_groups.append({"group_id": group_id, "member_count": len(members),
                                  "signals": signals})
        native_counts = {
            field: sum(row["native_provenance"][field] == "present" for row in rows)
            for field in ("token_ids", "logprobs", "mask", "group_id")
        }
        readiness_reasons = ["native_batch_not_validated"]
        if not rows:
            readiness_reasons.append("no_rollouts")
        readiness_reasons.extend([
            f"missing_native_{field}" for field, count in native_counts.items()
            if count != len(rows)
        ])
        v1_mask_counts = collections.Counter()
        v2_mask_counts = collections.Counter()
        for row in rows:
            v1_mask_counts.update(row["v1_masked_reasons"])
            v2_mask_counts.update(row["v2_masked_reasons"])
        mask_counts = v1_mask_counts + v2_mask_counts
        receipt_sources = collections.Counter(
            row["receipt_source"] for row in rows
            if isinstance(row.get("receipt_source"), str)
        )
        receipt_request_ids = sorted({
            row["receipt_id"] for row in rows if isinstance(row.get("receipt_id"), str)
        })
        source_requests: dict[str, set[str]] = collections.defaultdict(set)
        for row in rows:
            source = row.get("receipt_source")
            receipt_id = row.get("receipt_id")
            if isinstance(source, str) and isinstance(receipt_id, str):
                source_requests[source.split("/", 1)[0]].add(receipt_id)
        report = {
            "audit_version": AUDIT_VERSION,
            "evidence": {"rollout_root": _rel(root, root)},
            "rollouts": rows,
            "summary": {
                "rollout_count": len(rows),
                "valid_transition_count": sum(row["request_id"] is not None for row in rows),
                "request_count": len(request_counts),
                "receipt_join_count": sum(row["receipt_id"] is not None for row in rows),
                "receipt_request_count": len(receipt_request_ids),
                "receipt_source_counts": dict(sorted(receipt_sources.items())),
                "receipt_source_request_counts": {
                    source: len(requests) for source, requests in sorted(source_requests.items())
                },
                "duplicate_request_ids": {
                    key: count for key, count in sorted(request_counts.items()) if count > 1
                },
                "duplicate_receipt_ids": duplicate_receipts,
                "mask_reason_counts": dict(sorted(mask_counts.items())),
                "v1_mask_reason_counts": dict(sorted(v1_mask_counts.items())),
                "v2_mask_reason_counts": dict(sorted(v2_mask_counts.items())),
                "v1_scored_count": sum(row["v1_reward"] is not None for row in rows),
                "v2_scored_count": sum(row["v2_reward"] is not None for row in rows),
                "grounding_exclusion_count": sum(row["excluded"] for row in rows),
            },
            "native_provenance": {
                "field_counts": native_counts,
                "replay_fields_present": {
                    field: count > 0
                    for field, count in native_counts.items()
                },
                "trace_duplicate_rollout_ids": trace_doc["duplicate_rollout_ids"],
                "training_ready": False,
                "not_training_ready_reasons": readiness_reasons,
            },
            "group_completeness": {
                "expected_group_size": group_size,
                "groups": group_report,
                "signal_groups": signal_groups,
                "status": "unavailable" if not group_report else (
                    "complete" if all(group["status"] == "complete" for group in group_report)
                    else "observed_only" if all(group["status"] == "observed_only" for group in group_report)
                    else "incomplete"
                ),
            },
        }
        return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path, help="evidence directory or final-evidence.tar.gz")
    parser.add_argument("--exclusions", type=Path, help="reviewed request exclusion manifest")
    parser.add_argument("--reward-batches", type=Path, action="append",
                        help="preserved reward-batches root or batch; repeat for supplements")
    parser.add_argument("--expected-group-size", type=int)
    parser.add_argument("--output", type=Path, required=True, help="report JSON destination")
    args = parser.parse_args(argv)
    report = audit(args.evidence, exclusions=args.exclusions,
                   expected_group_size=args.expected_group_size,
                   reward_batches=args.reward_batches)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
