#!/usr/bin/env python3
"""Stage one hash-pinned, finite text or photo teacher shard on Linux CPU."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import time
from urllib.error import HTTPError
from urllib.request import urlopen

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BENCH = ROOT / "painter/benchmarks/openrouter-teachers-20260922"
OUT = ROOT / "painter/collected/diverse-teacher-wave-20260929"
DATASET = "CK0607/komorebi-painter-teachers"
COPY = ("contract.json", "model-catalog.json", "gateway_transport.ts", "gateway_prompt.ts",
        "package.json", "pnpm-lock.yaml", "pnpm-workspace.yaml", "tsconfig.json")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_archive(local: bool) -> bytes:
    receipt = json.loads((HERE / "source-receipt.json").read_text())
    if local:
        raw = (HERE / "source.tar.gz").read_bytes()
    else:
        public = json.loads((HERE / "source-public.json").read_text())
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{public['revision']}/{public['path']}?download=true"
        for attempt in range(3):
            try:
                with urlopen(url, timeout=180) as response:
                    raw = response.read(receipt["archive_bytes"] + 1)
                break
            except HTTPError as error:
                if error.code != 429 or attempt == 2:
                    if error.code != 429:
                        raise
                    cached = hf_hub_download(DATASET, public["path"], repo_type="dataset",
                        revision=public["revision"], token=False,
                        cache_dir=OUT / ".hf-public-cache")
                    raw = Path(cached).read_bytes()
                    break
                time.sleep(2 ** attempt)
        if public["archive_sha256"] != receipt["archive_sha256"]:
            raise ValueError("public source pointer and local receipt disagree")
    if len(raw) != receipt["archive_bytes"] or sha(raw) != receipt["archive_sha256"]:
        raise ValueError("teacher source archive hash/size mismatch")
    return raw


def stage(shard: str, *, local: bool = False) -> Path:
    receipt = json.loads((HERE / "source-receipt.json").read_text())
    if shard not in receipt["shards"]:
        raise ValueError("unexpected shard")
    raw = load_archive(local)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        manifest_raw = archive.extractfile("manifest.json").read()
        if sha(manifest_raw) != receipt["manifest_sha256"] or manifest_raw != (HERE / "manifest.json").read_bytes():
            raise ValueError("source manifest hash mismatch")
        manifest = json.loads(manifest_raw)
        rows = [row for row in manifest["rows"] if row["shard"] == shard]
        if len(rows) != receipt["shards"][shard] or not rows:
            raise ValueError("unexpected shard source count")
        mode = rows[0]["mode"]
        tier = rows[0]["tier"]
        if any(row["mode"] != mode or row["tier"] != tier for row in rows):
            raise ValueError("mixed mode/tier in shard")
        target = OUT / shard
        target.mkdir(parents=True, exist_ok=False)
        (target / "references").mkdir()
        references = []
        for row in rows:
            if mode == "text_to_paint":
                if sha(row["task_text"].encode()) != row["sha256"]:
                    raise ValueError("text brief hash mismatch")
                references.append({"id": row["id"], "image": None,
                    "sha256": row["sha256"], "source_kind": "text_prompt",
                    "task_text": row["task_text"], "category": row["category"], "split": "train"})
            else:
                name = row["path"]
                if not name.startswith("references/") or ".." in Path(name).parts:
                    raise ValueError("unsafe photo member path")
                member = archive.extractfile(name)
                if member is None:
                    raise ValueError("photo member missing")
                photo = member.read()
                if len(photo) != row["bytes"] or sha(photo) != row["sha256"]:
                    raise ValueError("photo hash/size mismatch")
                (target / name).write_bytes(photo)
                references.append({"id": row["id"], "image": name,
                    "sha256": row["sha256"], "source_kind": "licensed_photo",
                    "category": row["category"], "split": "train",
                    "captions": row["captions"], "source_url": row["source_url"],
                    "license_name": row["license_name"], "license_url": row["license_url"]})
    model = f"xiaomi/mimo-v2.6-{tier}"
    (target / "refs.json").write_text(json.dumps({"version": 1, "count": len(references),
        "references": references}, indent=2, sort_keys=True) + "\n")
    prompt = (ROOT / "painter/runs/quality-curriculum-20260924/quality-prompt.txt").read_text()
    if mode == "text_to_paint":
        prompt = prompt.replace("Paint the supplied reference", "Paint the supplied scene brief")
        prompt = prompt.replace("Inspect the reference and current canvas", "Inspect the scene brief and current canvas")
        prompt = prompt.replace("matches the reference well", "fulfills the brief beautifully")
    prompt += ("\nPAINTING LOOP: Make a complete 600x600 WEBGL sketch; inspect the actual rendered canvas "
               "before the next turn. Fix composition and silhouette before detail. Favor expressive but precise "
               "warmth, clean color, visible brush character, and coherent boundaries over mechanical hatching "
               "or dark muddiness. Do not invent extra subjects. Keep the primary opaque underpaint reliable. "
               "Use a brief plan and roughly 150-300 lines of purposeful code when possible; there is no hard "
               "line count. Stop early with FINISHED only after inspecting a satisfying canvas.\n")
    (target / "prompt.txt").write_text(prompt)
    config = {"benchmark": "diverse-teacher-wave-20260929-" + shard, "models": [model],
              "tracks": {"quality": {"max_turns": 4, "max_tokens": 32768,
                                     "episode_timeout_seconds": 2400, "reasoning_effort": "highest_supported"}},
              "reasoning_overrides": {model: "low"}, "temperature": 0.65,
              "concurrency": 2, "render_concurrency": 2,
              "canonicalize_webgl_setup": True, "renderer_timeout": 300,
              "samples_per_image": 1, "catalog": "model-catalog.json",
              "transport_script": "gateway_transport.ts", "references": "refs.json",
              "prompt": "prompt.txt", "request_timeout_seconds": 900, "max_retries": 0}
    (target / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    for name in COPY:
        shutil.copyfile(BENCH / name, target / name)
    provenance = {"schema": "painter.diverse-teacher-stage.v1", "shard": shard,
        "mode": mode, "tier": tier, "model": model,
        "source_archive_sha256": receipt["archive_sha256"],
        "manifest_sha256": receipt["manifest_sha256"],
        "config_sha256": sha((target / "config.json").read_bytes()),
        "prompt_sha256": sha((target / "prompt.txt").read_bytes()),
        "refs_sha256": sha((target / "refs.json").read_bytes()),
        "training_admission": "none_until_visual_review"}
    (target / "restored-source.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard")
    parser.add_argument("--local-source", action="store_true")
    args = parser.parse_args()
    print(stage(args.shard, local=args.local_source))
