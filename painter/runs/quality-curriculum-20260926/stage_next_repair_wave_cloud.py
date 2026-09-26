#!/usr/bin/env python3
"""Stage the pinned public eight-scene repair wave for the visual teacher runner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from urllib.error import HTTPError
from urllib.request import urlopen

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BENCH = ROOT / "painter/benchmarks/openrouter-teachers-20260922"
BASE = ROOT / "painter/collected/quality-curriculum-20260926/rendered-cloud"
PUBLIC = json.loads((HERE / "NEXT_REPAIR_8_PUBLIC.json").read_text())
DATASET = PUBLIC["dataset"]
COPY = ("contract.json", "model-catalog.json", "gateway_transport.ts", "gateway_prompt.ts",
        "package.json", "pnpm-lock.yaml", "pnpm-workspace.yaml", "tsconfig.json")
MODELS = {"flash": "xiaomi/mimo-v2.6-flash", "pro": "xiaomi/mimo-v2.6-pro"}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def source_archive(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha(destination.read_bytes()) == PUBLIC["archive_sha256"]:
        return destination
    url = (f"https://huggingface.co/datasets/{DATASET}/resolve/{PUBLIC['dataset_commit']}/"
           f"{PUBLIC['path']}?download=true")
    try:
        with urlopen(url, timeout=180) as response:
            raw = response.read(20_000_001)
    except HTTPError as exc:
        if exc.code != 429:
            raise
        cached = hf_hub_download(DATASET, PUBLIC["path"], repo_type="dataset",
                                 revision=PUBLIC["dataset_commit"], token=False,
                                 cache_dir=destination.parent / ".hf-public-cache")
        raw = Path(cached).read_bytes()
    if len(raw) != PUBLIC["archive_bytes"] or sha(raw) != PUBLIC["archive_sha256"]:
        raise ValueError("public repair source archive hash/size mismatch")
    destination.write_bytes(raw)
    return destination


def unpack(archive: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        if len(members) != 33 or sum(m.size for m in members) > 20_000_000:
            raise ValueError("unexpected repair source member count/size")
        for member in members:
            path = Path(member.name)
            if (not member.isfile() or path.is_absolute() or ".." in path.parts
                    or (member.name != "manifest.json" and
                        (len(path.parts) != 2 or path.parts[0] not in {"sources", "priors", "programs", "prompts"}))):
                raise ValueError(f"unsafe source member: {member.name}")
            target = output / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.extractfile(member).read())
    manifest = json.loads((output / "manifest.json").read_text())
    if (manifest.get("schema") != "painter.teacher600-repair-source.v1"
            or manifest.get("count") != 8
            or manifest.get("plan_sha256") != PUBLIC["plan_sha256"]):
        raise ValueError("public repair manifest identity mismatch")
    for row in manifest["rows"]:
        for key in ("source", "prior_canvas", "prior_program", "prompt"):
            if sha((output / row[key]).read_bytes()) != row[f"{key}_sha256"]:
                raise ValueError(f"repair source member hash mismatch: {row['audit_id']} {key}")
        if row["mode"] == "image_to_image":
            rights = row.get("rights") or {}
            if not all(rights.get(k) for k in ("name", "url", "source_url")):
                raise ValueError(f"missing photo source metadata: {row['audit_id']}")
    return manifest


def stage(run_id: str, tier: str, archive: Path | None = None) -> Path:
    if tier not in MODELS:
        raise ValueError("tier must be flash or pro")
    model = MODELS[tier]
    output = BASE / run_id / tier
    output.mkdir(parents=True, exist_ok=False)
    source = source_archive(archive or output / "source.tar.gz")
    manifest = unpack(source, output / "source-staging")
    selected = [r for r in manifest["rows"] if (r["difficulty"] == "hard") == (tier == "pro")]
    if len(selected) != (3 if tier == "pro" else 5):
        raise ValueError("unexpected tier selection")
    (output / "references").mkdir()
    refs = []
    for row in selected:
        audit_id = row["audit_id"]
        inp_ext = ".jpg" if row["mode"] == "image_to_image" else ".png"
        ref_name = f"references/{audit_id}{inp_ext}"
        ref_src = row["source"] if row["mode"] == "image_to_image" else row["prior_canvas"]
        shutil.copyfile(output / "source-staging" / ref_src, output / ref_name)
        ref_sha = row["source_sha256"] if row["mode"] == "image_to_image" else row["prior_canvas_sha256"]
        photo_prior = f"references/{audit_id}-first-paint.png"
        if row["mode"] == "image_to_image":
            shutil.copyfile(output / "source-staging" / row["prior_canvas"], output / photo_prior)
        prompt_text = (output / "source-staging" / row["prompt"]).read_text()
        task_text = ("Source scene brief:\n" + prompt_text.strip() + "\n\n"
                     "Visually inspected weakness to fix:\n" + row["parent_visual_weakness"])
        item = {"id": audit_id, "image": ref_name, "sha256": ref_sha,
                "category": row["category"], "split": "train", "task_text": task_text}
        if row["mode"] == "image_to_image":
            item.update(prior_canvas=photo_prior, prior_canvas_sha256=row["prior_canvas_sha256"])
        refs.append(item)
    (output / "refs.json").write_text(json.dumps({"version": 1, "count": len(refs),
                                                 "references": refs}, indent=2, sort_keys=True) + "\n")
    base_prompt = (ROOT / "painter/runs/quality-curriculum-20260924/quality-prompt.txt").read_text()
    prompt = (base_prompt + "\n\nCORRECTION TASK: improve the hash-pinned first painting. "
              "For photo scenes, image one is the true source photo and image two is the existing first paint. "
              "For text scenes, the attached image is the existing first paint and the exact scene brief is in "
              "the user message. Make a complete new sketch. On subsequent turns inspect your actual new canvas "
              "and revise its largest mismatch. Preserve recognizable structure; do not merely trace the prior.\n")
    (output / "prompt.txt").write_text(prompt)
    config = {"benchmark": f"teacher600-repair-eight-{tier}-20260927", "models": [model],
              "tracks": {"quality": {"max_turns": 4, "max_tokens": 16384,
                                     "episode_timeout_seconds": 2400,
                                     "reasoning_effort": "highest_supported"}},
              "temperature": 0.7, "concurrency": 2, "render_concurrency": 2,
              "renderer_timeout": 300, "samples_per_image": 1,
              "catalog": "model-catalog.json", "transport_script": "gateway_transport.ts",
              "references": "refs.json", "prompt": "prompt.txt", "request_timeout_seconds": 900,
              "max_retries": 0}
    (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    for name in COPY:
        shutil.copyfile(BENCH / name, output / name)
    provenance = {"schema": "painter.teacher600-repair-stage.v1", "model": model,
                  "tier": tier, "run_id": run_id, "source_dataset_commit": PUBLIC["dataset_commit"],
                  "source_archive_sha256": PUBLIC["archive_sha256"],
                  "source_manifest_sha256": sha((output / "source-staging/manifest.json").read_bytes()),
                  "config_sha256": sha((output / "config.json").read_bytes()),
                  "prompt_sha256": sha((output / "prompt.txt").read_bytes()),
                  "refs_sha256": sha((output / "refs.json").read_bytes()),
                  "reference_ids": [r["audit_id"] for r in selected],
                  "training_admission": "none_until_pairwise_review"}
    (output / "restored-source.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"stage": str(output), "tier": tier, "model": model,
                      "references": len(selected)}), flush=True)
    return output


def finalize(run_id: str, tier: str) -> None:
    output = BASE / run_id / tier
    manifest = json.loads((output / "source-staging/manifest.json").read_text())
    for row in manifest["rows"]:
        if (row["difficulty"] == "hard") != (tier == "pro"):
            continue
        matches = list((output / "episodes").glob(f"*--{row['audit_id']}--s01/episode.json"))
        if len(matches) != 1:
            raise ValueError(f"missing/duplicate episode: {row['audit_id']}")
        episode = matches[0].parent
        state = json.loads(matches[0].read_text())
        if state.get("reference_id") != row["audit_id"]:
            raise ValueError("episode identity mismatch")
        for field, name in (("source", "source-input" + (".jpg" if row["mode"] == "image_to_image" else ".txt")),
                            ("prior_canvas", "first-paint.png"),
                            ("prior_program", "first-paint.js"),
                            ("prompt", "source-prompt.txt")):
            shutil.copyfile(output / "source-staging" / row[field], episode / name)
    print(json.dumps({"finalized": tier, "episodes": len(list((output / "episodes").glob("*/episode.json")))}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("tier", choices=sorted(MODELS))
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if not args.run_id.startswith("teacher600-repair-") or not args.run_id.replace("-", "").isalnum():
        parser.error("invalid run ID")
    if args.finalize:
        finalize(args.run_id, args.tier)
    else:
        stage(args.run_id, args.tier, args.source_archive)
