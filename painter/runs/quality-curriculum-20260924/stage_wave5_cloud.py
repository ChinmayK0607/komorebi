#!/usr/bin/env python3
"""Restore one hash-pinned wave-5 teacher shard on a Codex Cloud CPU VM."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parent
BENCH = ROOT / "painter/benchmarks/openrouter-teachers-20260922"
OUT = ROOT / "painter/collected/quality-curriculum-20260924/mimo-wave5"
SHARDS = ("easy-a", "easy-b", "hard-a", "hard-b")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def stage(shard: str) -> Path:
    if shard not in SHARDS:
        raise ValueError(f"unexpected shard: {shard}")
    source = json.loads((RUN / "wave5-public-source.json").read_text())
    manifest_bytes = (RUN / "reference-manifest-wave5.json").read_bytes()
    if sha(manifest_bytes) != source["manifest_sha256"]:
        raise ValueError("tracked wave-5 manifest hash changed")
    manifest = json.loads(manifest_bytes)
    url = (f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/"
           f"{source['path']}?download=true")
    with urlopen(url, timeout=180) as response:
        raw = response.read()
    if sha(raw) != source["archive_sha256"]:
        raise ValueError("public source archive hash mismatch")
    archive = tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz")
    embedded = archive.extractfile("manifest.json")
    if embedded is None or embedded.read() != manifest_bytes:
        raise ValueError("embedded source manifest does not match tracked manifest")
    selected = [r for r in manifest["references"] if r["shard"] == shard]
    if len(selected) != 12 or len({r["sha256"] for r in selected}) != 12:
        raise ValueError("shard must contain 12 distinct reference hashes")
    model = selected[0]["model"]
    turns = selected[0]["max_turns"]
    if any(r["model"] != model or r["max_turns"] != turns or r["split"] != "train" for r in selected):
        raise ValueError("shard has mixed model, budget or split")
    root = OUT / shard
    (root / "references").mkdir(parents=True, exist_ok=True)
    references = []
    for row in selected:
        member = archive.extractfile(row["path"])
        if member is None:
            raise ValueError(f"source member missing: {row['id']}")
        image = member.read()
        if sha(image) != row["sha256"] or len(image) != row["bytes"]:
            raise ValueError(f"source member mismatch: {row['id']}")
        target = root / "references" / Path(row["path"]).name
        target.write_bytes(image)
        references.append({"id": row["id"], "image": f"references/{target.name}",
                           "sha256": row["sha256"], "split": "train",
                           "category": f"COCO128 {row['tier']} ({row['object_count']} annotated objects)",
                           "tier": row["tier"], "source_kind": "coco128_train_photo"})
    for name in ("prompt.txt", "contract.json", "model-catalog.json", "gateway_transport.ts",
                 "gateway_prompt.ts", "package.json", "pnpm-lock.yaml", "pnpm-workspace.yaml", "tsconfig.json"):
        shutil.copyfile(BENCH / name, root / name)
    # This is teacher-only guidance for common observed renderer mistakes; the
    # student contract remains frozen for the matched baseline until evaluated.
    with (root / "prompt.txt").open("a") as stream:
        stream.write("\nUse the global brush object directly. Never construct Brush or p5.brush and never call brush.init. "
                     "brush.fill colors brush.polygon, while p5 native fill colors native rect/ellipse/shape. "
                     "Check renderer feedback and replace the complete sketch after an invalid render.\n")
    config = {"benchmark": f"mimo-wave5-{shard}-20260925", "models": [model],
              "tracks": {"quality": {"max_turns": turns, "max_tokens": "native",
                                     "episode_timeout_seconds": None,
                                     "reasoning_effort": "highest_supported"}},
              "temperature": 0.5, "concurrency": 2, "render_concurrency": 1,
              "renderer_timeout": 240, "samples_per_image": 1,
              "catalog": "model-catalog.json", "transport_script": "gateway_transport.ts",
              "references": "refs.json", "prompt": "prompt.txt",
              "request_timeout_seconds": 900, "max_retries": 0}
    (root / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (root / "refs.json").write_text(json.dumps({"version": 1, "count": 12, "references": references}, indent=2) + "\n")
    (root / "restored-source.json").write_text(json.dumps({
        "schema": "painter.mimo-wave5-restored-source.v1", "shard": shard,
        "model": model, "max_turns": turns, "source_revision": source["revision"],
        "source_archive_sha256": source["archive_sha256"],
        "manifest_sha256": source["manifest_sha256"],
        "prompt_sha256": sha((root / "prompt.txt").read_bytes()),
        "reference_ids": [r["id"] for r in selected],
        "status": "candidate_generation_not_training_admission",
    }, indent=2) + "\n")
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", choices=SHARDS, required=True)
    args = parser.parse_args()
    print(stage(args.shard))


if __name__ == "__main__":
    main()
