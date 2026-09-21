"""Build one native Prime batch from a completed painter trace group.

This module deliberately keeps the replay boundary narrow.  It reads the
serialized verifier wire records, restores omitted multimodal tensors with the
same renderer, checks the rendered prompt against the persisted native prefix,
and then delegates sample conversion and packing to Prime.  It never calls an
inference provider or a judge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import sys
import tarfile
from collections.abc import Iterable, Mapping
from typing import Any


ARCHIVE_SHA256 = "b00964e4e486b4ec7220738e7ae52fa72f63ba2380810ab2b3d93d9f8a38739a"
TRACE_MEMBER = "async-output/revision-rl-neutral-20260921/monitors/file/traces/stream/00000.jsonl"
PRIME_REVISION = "26b3131d2716a4f8210b165df584b83b4bc54f61"
VERIFIERS_REVISION = "828488fffe31aa3332b9d1bd4bd9ee320e375cf1"
MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
INITIALIZER_REVISION = "b7103dd9923f19449ccf814f9b67bb2dcfed0909"
INVALID_PROGRAM_REWARD = -2.0


class ReplayError(RuntimeError):
    """A fail-closed replay validation error."""


def _first(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _dump_model(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    return value


def _trace(record: Mapping[str, Any]) -> Mapping[str, Any]:
    traces = record.get("traces")
    if not isinstance(traces, list) or len(traces) != 1 or not isinstance(traces[0], Mapping):
        raise ReplayError(f"record {record.get('id')!r} must contain exactly one trace")
    return traces[0]


def _aliases(record: Mapping[str, Any]) -> set[str]:
    trace = _trace(record)
    info = trace.get("info") if isinstance(trace.get("info"), Mapping) else {}
    painting = info.get("painting_result") if isinstance(info.get("painting_result"), Mapping) else {}
    aliases: set[str] = set()
    for value in (
        record.get("id"),
        trace.get("id"),
        painting.get("id"),
        info.get("judge_request_id"),
        info.get("judge_receipt_id"),
    ):
        if value is not None:
            aliases.add(str(value))
    # Judge metadata has appeared in both flattened and nested forms.
    for key in ("judge_request", "judge_receipt"):
        nested = info.get(key)
        if isinstance(nested, Mapping):
            for subkey in ("id", "request_id", "receipt_id"):
                if nested.get(subkey) is not None:
                    aliases.add(str(nested[subkey]))
    return aliases


def load_score_document(path: str | os.PathLike[str]) -> dict[str, float]:
    """Load a small explicit score whitelist keyed by any trace alias."""
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    if isinstance(document, Mapping):
        for key in ("scores", "verified_scores", "trace_scores"):
            if isinstance(document.get(key), (Mapping, list)):
                document = document[key]
                break
    rows: list[tuple[str, Any]] = []
    if isinstance(document, Mapping):
        rows = [(str(key), value) for key, value in document.items()]
    elif isinstance(document, list):
        for row in document:
            if not isinstance(row, Mapping):
                raise ReplayError("score rows must be objects")
            key = _first(row, "trace_id", "request_id", "judge_request_id", "id", "sample_id")
            if key is None:
                raise ReplayError(f"score row has no trace/request id: {row!r}")
            rows.append((str(key), row))
    else:
        raise ReplayError("score document must be a mapping or list")

    scores: dict[str, float] = {}
    for key, raw in rows:
        value = raw
        if isinstance(raw, Mapping):
            value = _first(raw, "score", "reward", "value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ReplayError(f"score for {key!r} must be numeric")
        value = float(value)
        if not math.isfinite(value):
            raise ReplayError(f"score for {key!r} is not finite")
        if key in scores and scores[key] != value:
            raise ReplayError(f"duplicate score key {key!r} has conflicting values")
        scores[key] = value
    return scores


def _score_for(
    record: Mapping[str, Any], scores: Mapping[str, float], *, allow_invalid_program_penalty: bool
) -> tuple[float, str]:
    trace = _trace(record)
    if trace.get("ok") is False:
        raise ReplayError(f"trace {trace.get('id')!r} has ok=false and cannot be replayed")
    info = trace.get("info") if isinstance(trace.get("info"), Mapping) else {}
    painting = info.get("painting_result") if isinstance(info.get("painting_result"), Mapping) else {}
    valid = painting.get("valid")
    aliases = _aliases(record)
    matched = [key for key in aliases if key in scores]
    if valid is False and painting.get("failure_kind") == "program":
        if not allow_invalid_program_penalty:
            raise ReplayError(
                f"invalid program trace {trace.get('id')!r} requires --allow-invalid-program-penalty"
            )
        for key in matched:
            if scores[key] != INVALID_PROGRAM_REWARD:
                raise ReplayError(f"explicit invalid-program score for {key!r} is not -2.0")
        return INVALID_PROGRAM_REWARD, "deterministic_invalid_program_penalty"
    if valid is False:
        raise ReplayError(f"trace {trace.get('id')!r} has an unsupported invalid result")
    if len(matched) != 1:
        raise ReplayError(
            f"trace {trace.get('id')!r} needs exactly one verified score alias; matched {matched!r}"
        )
    return scores[matched[0]], matched[0]


def _records_from_jsonl(handle: Iterable[bytes | str]) -> Iterable[dict[str, Any]]:
    for line in handle:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        line = line.strip()
        if line:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ReplayError("trace stream rows must be JSON objects")
            yield record


def iter_archive_records(archive: str | os.PathLike[str], member: str = TRACE_MEMBER) -> Iterable[dict[str, Any]]:
    """Stream native records from the evidence archive without extracting raw data."""
    path = pathlib.Path(archive)
    if path.name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(path, "r:gz" if path.name.endswith((".tar.gz", ".tgz")) else "r") as bundle:
            try:
                target = bundle.getmember(member)
            except KeyError as exc:
                raise ReplayError(f"archive member not found: {member}") from exc
            stream = bundle.extractfile(target)
            if stream is None:
                raise ReplayError(f"archive member is not a regular file: {member}")
            yield from _records_from_jsonl(stream)
    else:
        with open(path, "rb") as stream:
            yield from _records_from_jsonl(stream)


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _policy_start(record: Mapping[str, Any]) -> int | None:
    run = record.get("run")
    if not isinstance(run, Mapping):
        return None
    work = run.get("work")
    if not isinstance(work, Mapping):
        return None
    policy = work.get("policy")
    return policy.get("start") if isinstance(policy, Mapping) else None


def _sampling_config(record: Mapping[str, Any]) -> Mapping[str, Any]:
    trace = _trace(record)
    agent = trace.get("agent") if isinstance(trace.get("agent"), Mapping) else {}
    config = agent.get("config") if isinstance(agent.get("config"), Mapping) else {}
    sampling = config.get("sampling")
    if not isinstance(sampling, Mapping):
        raise ReplayError(f"trace {trace.get('id')!r} has no persisted sampling config")
    return sampling


def validate_native_record(record: Mapping[str, Any]) -> None:
    trace = _trace(record)
    if not record.get("group", {}).get("id"):
        raise ReplayError(f"record {record.get('id')!r} has no group id")
    nodes = trace.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ReplayError(f"trace {trace.get('id')!r} has no native nodes")
    sampled_nodes = 0
    for index, node in enumerate(nodes):
        if not isinstance(node, Mapping):
            raise ReplayError(f"trace {trace.get('id')!r} node {index} is not an object")
        token_ids = node.get("token_ids")
        mask = node.get("mask")
        logprobs = node.get("logprobs") or []
        if not isinstance(token_ids, list) or not isinstance(mask, list) or len(token_ids) != len(mask):
            raise ReplayError(f"trace {trace.get('id')!r} node {index} token/mask lengths differ")
        if not isinstance(logprobs, list) or len(logprobs) != sum(bool(value) for value in mask):
            raise ReplayError(f"trace {trace.get('id')!r} node {index} logprobs are not sampled-token aligned")
        if any(mask):
            sampled_nodes += 1
    if not sampled_nodes:
        raise ReplayError(f"trace {trace.get('id')!r} has no sampled tokens")


def select_complete_group(
    records: Iterable[Mapping[str, Any]], group_id: str, *, policy_start: int = 0, group_size: int = 8
) -> list[Mapping[str, Any]]:
    selected = [record for record in records if str(record.get("group", {}).get("id")) == group_id]
    if len(selected) != group_size:
        raise ReplayError(f"group {group_id!r} has {len(selected)} records; expected exactly {group_size}")
    seen: set[str] = set()
    for record in selected:
        validate_native_record(record)
        trace_id = str(_trace(record).get("id"))
        if trace_id in seen:
            raise ReplayError(f"duplicate trace id in group: {trace_id}")
        seen.add(trace_id)
        if _policy_start(record) != policy_start:
            raise ReplayError(f"trace {trace_id} has policy.start={_policy_start(record)!r}, expected {policy_start}")
    return selected


def exclude_trace_ids(records: Iterable[Mapping[str, Any]], excluded: Iterable[str]) -> list[Mapping[str, Any]]:
    """Select an explicit reviewed subset after checking the source group is complete."""
    excluded_set = {str(value) for value in excluded}
    available = {str(_trace(record).get("id")) for record in records}
    missing = excluded_set - available
    if missing:
        raise ReplayError(f"requested exclusions are not members of the complete group: {sorted(missing)}")
    kept = [record for record in records if str(_trace(record).get("id")) not in excluded_set]
    if not kept:
        raise ReplayError("explicit exclusions removed every native group member")
    return kept


def _runtime_roots(prime_root: str | None, verifiers_root: str | None) -> None:
    roots = []
    if prime_root:
        roots.extend([str(pathlib.Path(prime_root) / "src"), str(pathlib.Path(prime_root) / "deps" / "renderers")])
    if verifiers_root:
        roots.append(str(pathlib.Path(verifiers_root)))
    for root in reversed(roots):
        if root not in sys.path:
            sys.path.insert(0, root)


def restore_multimodal(
    trace: Any, renderer: Any, *, trace_id: str
) -> dict[str, Any]:
    """Restore processor tensors and require exact persisted prompt alignment."""
    nodes = trace.nodes
    if not nodes:
        raise ReplayError(f"trace {trace_id!r} has no nodes")
    final_index = len(nodes) - 1
    while final_index >= 0 and not getattr(nodes[final_index], "sampled", False):
        final_index -= 1
    if final_index < 0:
        raise ReplayError(f"trace {trace_id!r} has no sampled node")
    prompt_nodes = nodes[:final_index]
    sampled_node = nodes[final_index]
    prefix_count = 0
    for value in sampled_node.mask:
        if value:
            break
        prefix_count += 1
    if any(not value for value in sampled_node.mask[prefix_count:]):
        raise ReplayError(f"trace {trace_id!r} sampled mask is not a false-prefix/true-suffix")
    messages = [_dump_model(node.message) for node in prompt_nodes]
    rendered = renderer.render(messages, add_generation_prompt=True)
    persisted = [token for node in prompt_nodes for token in node.token_ids]
    persisted.extend(sampled_node.token_ids[:prefix_count])
    if list(rendered.token_ids) != persisted:
        mismatch = next(
            (index for index, (left, right) in enumerate(zip(rendered.token_ids, persisted)) if left != right),
            min(len(rendered.token_ids), len(persisted)),
        )
        raise ReplayError(
            f"trace {trace_id!r} renderer prompt mismatch at token {mismatch}: "
            f"rendered_len={len(rendered.token_ids)} persisted_len={len(persisted)}"
        )
    persisted_map = getattr(trace, "mm_token_type_id_map", None) or {}
    rendered_map = getattr(renderer, "mm_token_type_id_map", None) or {}
    if persisted_map and rendered_map and {str(k): int(v) for k, v in persisted_map.items()} != {
        str(k): int(v) for k, v in rendered_map.items()
    }:
        raise ReplayError(f"trace {trace_id!r} multimodal token-type map differs from renderer")
    trace.mm_token_type_id_map = dict(rendered_map)
    has_media = any(
        isinstance(_dump_model(node.message).get("content"), list)
        and any(part.get("type") in ("image_url", "video_url", "image", "video") for part in _dump_model(node.message)["content"] if isinstance(part, Mapping))
        for node in prompt_nodes
    )
    mmd = getattr(rendered, "multi_modal_data", None)
    if has_media and (mmd is None or mmd.is_empty()):
        raise ReplayError(f"trace {trace_id!r} prompt contains media but renderer returned no tensors")
    if not has_media and mmd is not None and not mmd.is_empty():
        raise ReplayError(f"trace {trace_id!r} renderer returned unexpected media tensors")
    if mmd is not None and not mmd.is_empty():
        from verifiers.v1.graph import _attribute_mm

        path = [(index, node.message) for index, node in enumerate(prompt_nodes)]
        _attribute_mm(trace, path, 0, mmd)
    return {
        "prompt_tokens": len(persisted),
        "media": bool(has_media),
        "media_keys": sorted((mmd.mm_items if mmd is not None else {}).keys()) if mmd is not None else [],
    }


def _make_renderer(model_path: str, renderer_name: str) -> Any:
    from renderers import Qwen35RendererConfig, Qwen38RendererConfig, create_renderer
    from renderers.base import load_tokenizer

    tokenizer = load_tokenizer(model_path)
    if renderer_name == "qwen3.5":
        return create_renderer(tokenizer, Qwen35RendererConfig(enable_thinking=False))
    if renderer_name == "qwen3.8":
        return create_renderer(tokenizer, Qwen38RendererConfig(enable_thinking=False))
    raise ReplayError(f"unsupported persisted native renderer {renderer_name!r}")


def _persisted_renderer_name(record: Mapping[str, Any]) -> str:
    trace = _trace(record)
    agent = trace.get("agent") if isinstance(trace.get("agent"), Mapping) else {}
    config = agent.get("config") if isinstance(agent.get("config"), Mapping) else {}
    client = config.get("client") if isinstance(config.get("client"), Mapping) else {}
    renderer = client.get("renderer") if isinstance(client.get("renderer"), Mapping) else {}
    name = renderer.get("name")
    if not isinstance(name, str):
        raise ReplayError(f"trace {trace.get('id')!r} has no persisted renderer name")
    return name


def _float32_mean(values: list[float]) -> float:
    import torch

    return float(torch.tensor(values, dtype=torch.float32).mean().item())


def build_rank_batch(
    records: list[Mapping[str, Any]],
    scores: Mapping[str, float],
    *,
    model_path: str,
    output_dir: str | os.PathLike[str],
    env_name: str,
    seq_len: int,
    allow_invalid_program_penalty: bool,
    prime_root: str | None = None,
    verifiers_root: str | None = None,
    excluded_trace_ids: Iterable[str] = (),
    trainer_step: int = 1,
) -> dict[str, Any]:
    """Build and serialize rank 0's native packed batch."""
    _runtime_roots(prime_root, verifiers_root)
    import msgspec
    import verifiers.v1 as vf
    from prime_rl.orchestrator.algo.routing import assign_advantages
    from prime_rl.orchestrator.trajectories import trace_to_samples
    from prime_rl.trainer.batch import build_bin_cost, prepare_batch

    renderer_name = _persisted_renderer_name(records[0])
    if any(_persisted_renderer_name(record) != renderer_name for record in records):
        raise ReplayError("native group mixes renderer names")
    sampling = dict(_sampling_config(records[0]))
    if any(dict(_sampling_config(record)) != sampling for record in records):
        raise ReplayError("native group mixes sampling configurations")
    renderer = _make_renderer(model_path, renderer_name)
    rewards: list[float] = []
    score_keys: list[str] = []
    wire_traces: list[Any] = []
    for record in records:
        reward, score_key = _score_for(record, scores, allow_invalid_program_penalty=allow_invalid_program_penalty)
        episode = vf.WireEpisode.model_validate(record)
        trace = episode.traces[0]
        restore_info = restore_multimodal(trace, renderer, trace_id=trace.id)
        rewards.append(reward)
        score_keys.append(score_key)
        wire_traces.append((trace, restore_info))
    mean = _float32_mean(rewards)
    advantages = [float(value - mean) for value in rewards]
    samples = []
    restore_infos = []
    for (trace, restore_info), advantage in zip(wire_traces, advantages, strict=True):
        assign_advantages(trace, advantage)
        converted = trace_to_samples(trace, env_name=env_name)
        if len(converted) != 1:
            raise ReplayError(f"trace {trace.id!r} yielded {len(converted)} samples; expected one branch")
        temperature = float(trace.agent.config.sampling.temperature)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ReplayError(f"trace {trace.id!r} has invalid native sampling temperature {temperature!r}")
        converted[0].temperatures = [temperature] * len(converted[0].token_ids)
        samples.extend(converted)
        restore_infos.append(restore_info)
    bins = prepare_batch(samples, seq_len, num_train_workers=1, bin_cost=build_bin_cost(None), pad_to_multiple_of=1)
    if len(bins) != 1 or len(bins[0]) == 0:
        raise ReplayError(f"native packer returned an unexpected rank layout: {[len(row) for row in bins]}")
    if any(not batch.trace_ids for batch in bins[0]):
        raise ReplayError("packed batch lost native trace identities")
    payload = msgspec.msgpack.encode(bins[0])
    output = pathlib.Path(output_dir)
    if trainer_step < 1:
        raise ReplayError("trainer step must be >= 1; Prime's fresh Progress starts at step 1")
    batch_path = output / "batches" / f"step_{trainer_step}" / "rank_0.bin"
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    temp = batch_path.with_suffix(".tmp")
    temp.write_bytes(payload)
    temp.replace(batch_path)
    manifest = {
        "schema": "painter.native-replay-smoke.v1",
        "training_ready": True,
        "quality_proof": False,
        "gpu_update_required": True,
        "trainer_step": trainer_step,
        "prime_revision": PRIME_REVISION,
        "verifiers_revision": VERIFIERS_REVISION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "initializer_revision": INITIALIZER_REVISION,
        "renderer_name": renderer_name,
        "group_id": str(records[0].get("group", {}).get("id")),
        "source_group_size": len(records) + len(list(excluded_trace_ids)),
        "selected_group_size": len(records),
        "trace_ids": [str(_trace(record).get("id")) for record in records],
        "excluded_trace_ids": [str(value) for value in excluded_trace_ids],
        "policy_starts": [_policy_start(record) for record in records],
        "source_sampling": sampling,
        "score_keys": score_keys,
        "rewards": rewards,
        "float32_group_mean": mean,
        "advantages": advantages,
        "sample_count": len(samples),
        "microbatch_count": len(bins[0]),
        "microbatch_token_counts": [len(batch.input_ids) for batch in bins[0]],
        "multimodal": restore_infos,
        "rank_bin": str(batch_path),
        "rank_bin_sha256": hashlib.sha256(payload).hexdigest(),
        "padding": {"num_train_workers": 1, "pad_to_multiple_of": 1, "dummy_batches": 0},
        "serialized_logprobs": "preserved from disk wire records; rounded by Trace.to_record",
    }
    manifest_path = output / "replay-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--scores", required=True, help="JSON score whitelist keyed by native trace or judge request id")
    parser.add_argument("--group-id", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seq-len", type=int, default=16384)
    parser.add_argument("--env-name", default="painter")
    parser.add_argument("--member", default=TRACE_MEMBER)
    parser.add_argument("--archive-sha256", default=ARCHIVE_SHA256)
    parser.add_argument("--prime-root")
    parser.add_argument("--verifiers-root")
    parser.add_argument("--policy-start", type=int, default=0)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--trainer-step", type=int, default=1)
    parser.add_argument("--allow-invalid-program-penalty", action="store_true")
    parser.add_argument(
        "--exclude-trace-id",
        action="append",
        default=[],
        help="explicitly exclude a member after complete-group validation (repeatable)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    actual_sha = sha256_file(args.archive)
    if args.archive_sha256 and actual_sha != args.archive_sha256:
        raise ReplayError(f"archive sha256 mismatch: expected {args.archive_sha256}, got {actual_sha}")
    records = list(iter_archive_records(args.archive, args.member))
    selected = select_complete_group(records, args.group_id, policy_start=args.policy_start, group_size=args.group_size)
    source_group_size = len(selected)
    selected = exclude_trace_ids(selected, args.exclude_trace_id)
    scores = load_score_document(args.scores)
    manifest = build_rank_batch(
        selected,
        scores,
        model_path=args.model_path,
        output_dir=args.output_dir,
        env_name=args.env_name,
        seq_len=args.seq_len,
        allow_invalid_program_penalty=args.allow_invalid_program_penalty,
        prime_root=args.prime_root,
        verifiers_root=args.verifiers_root,
        excluded_trace_ids=args.exclude_trace_id,
        trainer_step=args.trainer_step,
    )
    manifest["source_group_size"] = source_group_size
    manifest["selected_group_size"] = len(selected)
    manifest["excluded_trace_ids"] = list(args.exclude_trace_id)
    pathlib.Path(args.output_dir, "replay-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReplayError as exc:
        print(f"native replay refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
