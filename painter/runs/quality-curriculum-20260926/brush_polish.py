#!/usr/bin/env python3
"""Prepare, publish, and stage four source-matched brush-feel revisions."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BENCH = ROOT / "painter/benchmarks/openrouter-teachers-20260922"
POOL = ROOT / "painter/collected/quality-curriculum-20260926"
OUT = POOL / "brush-polish-four-20260927"
RENDERS = POOL / "rendered-cloud"
DATASET = "CK0607/komorebi-painter-teachers"
REMOTE = "curricula/teacher600/brush-polish-four-20260927/source.tar.gz"
PUBLIC = HERE / "BRUSH_POLISH_PUBLIC.json"
IDS = ("t600-080", "t600-332", "t600-033", "t600-595")
COPY = ("contract.json", "model-catalog.json", "gateway_transport.ts", "gateway_prompt.ts",
        "package.json", "pnpm-lock.yaml", "pnpm-workspace.yaml", "tsconfig.json")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def archive_bytes(members: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as tar:
            for name, raw in sorted(members.items()):
                info = tarfile.TarInfo(name)
                info.size = len(raw)
                info.mtime = info.uid = info.gid = 0
                info.mode = 0o644
                tar.addfile(info, io.BytesIO(raw))
    return output.getvalue()


def prepare() -> dict:
    review = json.loads((HERE / "MANUAL_MULTI_TEACHER_REVIEW.json").read_text())
    decisions = {row["audit_id"]: row for row in review["rows"]}
    batches = {}
    for teacher in ("sol", "astra"):
        path = POOL / f"manual-repair-{teacher}-five-20260927/manifest.json"
        batches.update({row["id"]: row for row in json.loads(path.read_text())["items"]})
    render_names = {"t600-080": "manual-repair-sol-render-20260927",
                    **{ident: "manual-repair-astra-render-20260927" for ident in IDS[1:]}}
    rows = []
    members = {}
    for ident in IDS:
        row = batches[ident]
        decision = decisions[ident]
        episode = RENDERS / render_names[ident] / "episodes" / ident
        source = (ROOT / row["input_path"]).read_bytes()
        source_ext = ".jpg" if row["mode"] == "image_to_image" else ".txt"
        if sha(source) != row["input_sha256"] or sha((episode / "canvas.png").read_bytes()) != decision["candidate_canvas_sha256"]:
            raise ValueError(f"source or current canvas mismatch: {ident}")
        program = (episode / "program.js").read_bytes()
        if sha(program) != decision["candidate_program_sha256"]:
            raise ValueError(f"current program mismatch: {ident}")
        rights = row.get("source_metadata")
        if row["mode"] == "image_to_image" and not rights or (rights and not all(rights.get(k) for k in ("name", "url", "source_url"))):
            raise ValueError(f"photo rights missing: {ident}")
        source_name = f"source/{ident}{source_ext}"
        canvas_name = f"current/{ident}.png"
        program_name = f"programs/{ident}.js"
        members[source_name] = source
        members[canvas_name] = (episode / "canvas.png").read_bytes()
        members[program_name] = program
        rows.append({"id": ident, "mode": row["mode"], "category": row["category"],
                     "source": source_name, "source_sha256": sha(source),
                     "current_canvas": canvas_name, "current_canvas_sha256": sha(members[canvas_name]),
                     "current_program": program_name, "current_program_sha256": sha(program),
                     "parent_source_label": decision["decision"], "source_rights": rights,
                     "manual_render_run_id": render_names[ident]})
    manifest = {"schema": "painter.brush-polish-source.v1", "count": len(rows),
                "hypothesis": "Targeted p5.brush surface marks improve perceived brush feel while preserving the appealing native-shape composition.",
                "matched_baseline": "The exact four manually repaired, user-liked canvases and their original prompts/photos.",
                "source_review_sha256": sha((HERE / "MANUAL_MULTI_TEACHER_REVIEW.json").read_bytes()),
                "rows": rows, "sft_admitted": False}
    members["manifest.json"] = json_bytes(manifest)
    OUT.mkdir(parents=True, exist_ok=True)
    raw = archive_bytes(members)
    path = OUT / "source.tar.gz"
    if path.exists() and sha(path.read_bytes()) != sha(raw):
        raise ValueError("existing source archive differs")
    path.write_bytes(raw)
    return {"count": len(rows), "archive_sha256": sha(raw), "archive_bytes": len(raw),
            "manifest_sha256": sha(members["manifest.json"]), "path": str(path)}


def publish() -> dict:
    from huggingface_hub import HfApi
    prepared = prepare()
    api = HfApi()
    if api.repo_info(DATASET, repo_type="dataset").private:
        raise ValueError("teacher dataset must be public")
    commit = api.upload_file(path_or_fileobj=prepared["path"], path_in_repo=REMOTE,
                             repo_id=DATASET, repo_type="dataset",
                             commit_message="Add four source-matched brush polish inputs")
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{commit.oid}/{REMOTE}?download=true"
    with urlopen(url, timeout=180) as response:
        remote = response.read(prepared["archive_bytes"] + 1)
    if sha(remote) != prepared["archive_sha256"]:
        raise ValueError("anonymous public download hash mismatch")
    public = {"schema": "painter.brush-polish-public.v1", "dataset": DATASET,
              "dataset_commit": commit.oid, "path": REMOTE, "count": 4,
              "archive_sha256": prepared["archive_sha256"],
              "archive_bytes": prepared["archive_bytes"],
              "manifest_sha256": prepared["manifest_sha256"],
              "anonymous_hash_verified": True}
    PUBLIC.write_bytes(json_bytes(public))
    return public


def stage() -> Path:
    public = json.loads(PUBLIC.read_text())
    if public.get("schema") != "painter.brush-polish-public.v1" or public.get("count") != 4:
        raise ValueError("unexpected public source metadata")
    url = (f"https://huggingface.co/datasets/{public['dataset']}/resolve/"
           f"{public['dataset_commit']}/{public['path']}?download=true")
    with urlopen(url, timeout=180) as response:
        raw = response.read(public["archive_bytes"] + 1)
    if len(raw) != public["archive_bytes"] or sha(raw) != public["archive_sha256"]:
        raise ValueError("public source archive hash/size mismatch")
    target = RENDERS / "brush-polish-four-20260927"
    if target.exists():
        raise ValueError("stage already exists; refuse to overwrite results")
    target.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        entries = tar.getmembers()
        if len(entries) != 13 or sum(x.size for x in entries) > 40_000_000:
            raise ValueError("source member count/size mismatch")
        for entry in entries:
            name = Path(entry.name)
            if not entry.isfile() or name.is_absolute() or ".." in name.parts or len(name.parts) > 2:
                raise ValueError("unsafe source member")
            path = target / "source-staging" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(tar.extractfile(entry).read())
    source = target / "source-staging"
    if sha((source / "manifest.json").read_bytes()) != public["manifest_sha256"]:
        raise ValueError("source manifest hash mismatch")
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("schema") != "painter.brush-polish-source.v1" or [r["id"] for r in manifest["rows"]] != list(IDS):
        raise ValueError("source manifest mismatch")
    refs = []
    (target / "references").mkdir()
    for row in manifest["rows"]:
        ident = row["id"]
        for key in ("source", "current_canvas", "current_program"):
            if sha((source / row[key]).read_bytes()) != row[key + "_sha256"]:
                raise ValueError(f"source member hash mismatch: {ident} {key}")
        current = f"references/{ident}-current.png"
        shutil.copyfile(source / row["current_canvas"], target / current)
        if row["mode"] == "image_to_image":
            original = f"references/{ident}-source.jpg"
            shutil.copyfile(source / row["source"], target / original)
            image = original
            extra = {"prior_canvas": current, "prior_canvas_sha256": row["current_canvas_sha256"]}
            task = "Image one is the original reference photo; image two is the current painting."
        else:
            image = current
            extra = {}
            task = "This image is the current painting. Original scene brief:\n" + (source / row["source"]).read_text().strip()
        code = (source / row["current_program"]).read_text()
        task += ("\n\nKeep the current painting's appealing composition, shapes, identity, and palette. "
                 "Make a complete revised sketch that adds visible but restrained painterly brushwork in subject-specific areas. "
                 "Avoid a global filter, blanket translucency, scribbles, or geometry changes. "
                 "This is a brush-feel polish, not a new composition. Current complete program follows:\n```javascript\n"
                 + code + "\n```")
        refs.append({"id": ident, "image": image,
                     "sha256": row["source_sha256"] if row["mode"] == "image_to_image" else row["current_canvas_sha256"],
                     "category": row["category"], "split": "train", "task_text": task, **extra})
    (target / "refs.json").write_bytes(json_bytes({"version": 1, "count": 4, "references": refs}))
    prompt = (ROOT / "painter/runs/quality-curriculum-20260924/quality-prompt.txt").read_text()
    prompt += ("\n\nBRUSH POLISH: improve brush-like feeling without changing the established subject, "
               "composition, silhouettes, or palette. Start from the given complete program. Use opaque native p5 "
               "for main forms; add local p5.brush strokes, granulation, broken edges, and pigment variation only "
               "where they support the object's material. Do not use broad translucent fills over important forms. "
               "On turn one return a complete runnable javascript sketch; after the actual render, inspect it and "
               "correct any loss of structure or excess marks. Canvas contract: createCanvas(600,600,WEBGL).\n")
    (target / "prompt.txt").write_text(prompt)
    config = {"benchmark": "brush-polish-four-20260927", "models": ["xiaomi/mimo-v2.6-pro"],
              "tracks": {"quality": {"max_turns": 2, "max_tokens": 32768,
                                     "episode_timeout_seconds": 2400,
                                     "reasoning_effort": "highest_supported"}},
              "reasoning_overrides": {"xiaomi/mimo-v2.6-pro": "low"},
              "temperature": 0.5, "concurrency": 2, "render_concurrency": 2,
              "task_text_max_chars": 12000,
              "canonicalize_webgl_setup": True, "renderer_timeout": 300,
              "samples_per_image": 1, "catalog": "model-catalog.json",
              "transport_script": "gateway_transport.ts", "references": "refs.json",
              "prompt": "prompt.txt", "request_timeout_seconds": 900, "max_retries": 0}
    (target / "config.json").write_bytes(json_bytes(config))
    for name in COPY:
        shutil.copyfile(BENCH / name, target / name)
    (target / "provenance.json").write_bytes(json_bytes({
        "public_source": public, "source_review_sha256": manifest["source_review_sha256"],
        "config_sha256": sha((target / "config.json").read_bytes()),
        "prompt_sha256": sha((target / "prompt.txt").read_bytes()),
        "refs_sha256": sha((target / "refs.json").read_bytes()),
        "sft_admitted": False}))
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "publish", "stage"))
    args = parser.parse_args()
    if args.action == "prepare":
        print(json.dumps(prepare(), sort_keys=True))
    elif args.action == "publish":
        print(json.dumps(publish(), sort_keys=True))
    else:
        print(stage())


if __name__ == "__main__":
    main()
