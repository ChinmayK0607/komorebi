#!/usr/bin/env python3
"""Create a minimal, credential-free Linux rendering handoff for manual studies."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parent
COLLECTED = ROOT / "painter/collected/quality-curriculum-20260924"
VENDOR = ROOT / "painter/vendor/integrations/watercolour"
OUTPUT = COLLECTED / "manual-render-handoff.tar.gz"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    refs = json.loads((RUN / "reference-manifest.json").read_text(encoding="utf-8"))
    programs = json.loads((RUN / "manual-program-manifest.json").read_text(encoding="utf-8"))
    manual_refs = {row["id"]: row for row in refs["references"] if row["tier"] == "manual hard"}
    if len(manual_refs) != 10 or len(programs["programs"]) != 10:
        raise ValueError("expected exactly ten reviewed hard references and programs")
    files: list[Path] = [RUN / "reference-manifest.json", RUN / "manual-program-manifest.json",
                         RUN / "setup_manual_node.sh", VENDOR / "renderer.py"]
    files.extend(sorted((VENDOR / "assets").iterdir()))
    for row in programs["programs"]:
        ident = row["reference_id"]
        if ident not in manual_refs:
            raise ValueError(f"program has no reviewed source: {ident}")
        reference = ROOT / manual_refs[ident]["reference_path"]
        program = ROOT / row["program_path"]
        if sha(reference) != row["reference_sha256"] or sha(program) != row["program_sha256"]:
            raise ValueError(f"hash mismatch: {ident}")
        files.extend((reference, program))
    if any(not path.is_file() or path.is_symlink() for path in files):
        raise ValueError("handoff contains missing, non-file, or symlinked path")
    for path in files:
        data = path.read_bytes()
        if b"AI_GATEWAY_API_KEY" in data or b"sk-or-v1-" in data:
            raise ValueError(f"possible credential in handoff file: {path}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(OUTPUT, "w:gz") as archive:
        for path in files:
            info = archive.gettarinfo(str(path), arcname=str(path.relative_to(ROOT)))
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with path.open("rb") as stream:
                archive.addfile(info, stream)
    receipt = {"schema": "painter.manual-render-handoff.v1",
               "file_count": len(files), "bytes": OUTPUT.stat().st_size,
               "sha256": sha(OUTPUT), "archive": str(OUTPUT.relative_to(ROOT)),
               "contents": [{"path": str(path.relative_to(ROOT)), "sha256": sha(path)} for path in files],
               "credential_files_included": False,
               "note": "Prepared locally only; upload requires separate authorization."}
    (RUN / "receipt-handoff.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"prepared {len(files)} files, {receipt['bytes']} bytes, SHA-256 {receipt['sha256']}")


if __name__ == "__main__":
    main()
