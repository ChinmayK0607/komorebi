#!/usr/bin/env python3
"""Freeze 60 novel text briefs and 60 disjoint licensed photos for CPU teachers."""

from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
POOL = ROOT / "painter/collected/quality-curriculum-20260926/coco-val2017"
PRIOR = ROOT / "painter/collected/quality-curriculum-20260926"

# Hand-authored, spatially explicit briefs. Six different subjects per family.
BRIEFS = {
    "still_life": [
        "Paint a chipped blue enamel kettle on a pale kitchen shelf, with its curved spout to the left, wooden handle above, and one folded tea towel below. Soft morning light; keep the silhouette intact.",
        "Paint three figs in a shallow cream bowl, one cut open in front of two whole fruits. The bowl sits on a linen table, with quiet mauve shadows and generous empty space.",
        "Paint a brass desk bell beside a stack of two postcards on dark walnut. Make the bell's dome, small button, and base clear, with warm reflected light and no extra objects.",
        "Paint a green glass bottle behind a peeled orange and two loose leaves. Keep the bottle translucent, the fruit round and grounded, and the tabletop edge lower than the objects.",
        "Paint a pair of weathered leather gloves draped over a small wooden box. One glove overlaps the other; preserve all fingers without turning them into rigid parallel bars.",
        "Paint an old film camera on a cream cloth beside one red apple. Show the lens as a dark ellipse, a readable camera body, gentle folds, and warm side lighting.",
    ],
    "flowers_plants": [
        "Paint a single yellow iris in a narrow terracotta vase. Its three open petals and upright leaves should remain distinct against a soft blue wall.",
        "Paint a cluster of white daisies leaning out of a chipped watering can. Some stems cross, but the can's handle and spout remain readable; quiet garden light.",
        "Paint a small bonsai pine on a low stone plinth. Separate its twisting trunk, three foliage masses, and a few roots; leave calm negative space around the crown.",
        "Paint two sunflowers in a blue ceramic jug, one facing forward and one turned partly away. Keep the dark seed heads, petals, jug rim, and tabletop distinct.",
        "Paint a drooping bunch of lavender tied with string above a rustic sink. Preserve the stems' direction and violet clusters without turning them into a purple blur.",
        "Paint a broad monstera leaf in afternoon window light, with its holes and split edges clear. A small pale pot anchors it against a warm cream background.",
    ],
    "animals": [
        "Paint a red fox curled asleep on a mossy log. Its tail wraps in front of the body, ears point backward, and a dim woodland recedes softly behind it.",
        "Paint a gray donkey standing three-quarter view in a grassy paddock. Keep the long ears, muzzle, four legs, and hoof placement coherent; avoid scratchy fur detail.",
        "Paint a kingfisher perched on a reed above still water. Preserve the orange chest, blue back, slim beak, two feet on the reed, and its small reflection.",
        "Paint a sea turtle gliding diagonally through turquoise water. Show a single domed shell, four separate flippers, and gentle shadow beneath, without crowded coral.",
        "Paint two rabbits under a garden bench, one white in front and one brown partly behind. Their ears, eyes, paws, and overlap must stay legible.",
        "Paint a black-and-white goat standing on a low hillside at dusk. Keep its horn direction, long beard, slender legs, and the slope beneath its hooves coherent.",
    ],
    "vehicles": [
        "Paint a red city tram turning through a quiet street. The front is nearer than the long side; align its windows, two tracks, roof cable, and street perspective.",
        "Paint a small yellow excavator at rest beside a mound of sand. Show its tracked base, cab, one bent arm and bucket as a single plausible machine.",
        "Paint a vintage bicycle leaning against a white garden wall. Keep the two wheels circular, frame triangles joined, handlebars above the front wheel, and a soft ground shadow.",
        "Paint a blue fishing boat seen from above-left in a harbor. Keep bow, cabin, mast, and wake in one consistent perspective; use restrained sea greens.",
        "Paint a cream delivery van parked by a bakery at sunrise. Show the front-left corner, long side, two aligned wheels, door opening, and warm shop window.",
        "Paint a propeller airplane on a grassy airfield under soft cloud. Make wing, tail, cockpit and landing wheels connect correctly, with atmospheric rather than technical detail.",
    ],
    "architecture": [
        "Paint a stone lighthouse on a low island at sunset. Keep the tower vertical, lantern room on top, island beneath, and horizon behind the structure.",
        "Paint a narrow bookshop facade in a rainy lane. Its awning covers a door between two windows; warm interior light contrasts with cool wet paving.",
        "Paint a small red barn behind a white fence, with one open doorway and a sloped roof. Show the foreground fence clearly in front of the barn.",
        "Paint a glass greenhouse in a walled garden. The pitched roof and panes should share one perspective; plants fill only the lower half.",
        "Paint a blue-domed observatory on a hill at twilight. Keep the dome seated on its cylindrical base, one shutter opening, and a winding path leading up.",
        "Paint a simple wooden footbridge crossing a stream. Near railing is higher and longer than the far railing; the planks and water flow agree in perspective.",
    ],
    "landscapes": [
        "Paint a winding footpath through a meadow toward distant blue mountains. A single birch stands on the right; preserve near-to-far scale and an airy sky.",
        "Paint a quiet salt marsh at low tide with three shallow water channels. Keep the horizon low, grasses in the foreground, and warm clouds reflected in the water.",
        "Paint a snow-covered mountain pass with a small cabin in the middle distance. Make the cabin tiny relative to the cliffs, with a readable roof and smoke plume.",
        "Paint a rocky coast with one white sea stack and a broad wave folding around it. Leave room for the sky; do not darken all water into a single mass.",
        "Paint terraced rice fields curving around a green hillside in early morning mist. Keep the terrace bands distinct and the distant village small.",
        "Paint a desert canyon at golden hour with a narrow stream below. Layer near orange rock, middle shadow, and distant pale cliffs without flattening depth.",
    ],
    "interiors": [
        "Paint a sunlit reading nook: one soft armchair beside a tall window, a small round side table, and a book on the chair. Keep each object's footprint separate.",
        "Paint a pottery studio corner with a wheel in front, three shelves behind, and one unfinished clay bowl. Preserve depth and avoid turning tools into clutter.",
        "Paint a modest kitchen table set for breakfast: two plates, a glass jug, and a chair partly tucked under it. The table ellipse and chair legs must align.",
        "Paint a warmly lit tailor's workbench with folded blue fabric, a wooden mannequin torso behind it, and one pair of scissors in front.",
        "Paint a greenhouse interior seen down its central aisle. Rows of pots recede on both sides beneath a pitched glass roof; leave the aisle open.",
        "Paint an old library stair landing with a curved banister, two bookcases, and a small lamp. Use coherent verticals and warm pools of light.",
    ],
    "people_actions": [
        "Paint a baker sliding one loaf into a brick oven with a long wooden peel. Show both hands holding the peel, the loaf in front of the oven, and warm firelight.",
        "Paint a gardener kneeling beside a rose bed, watering one young plant. The watering can, arm, spout, and stream should connect plausibly.",
        "Paint a child in a yellow raincoat holding a red umbrella on wet pavement. Keep the umbrella above the child, boots on the ground, and a simple reflection.",
        "Paint a violinist seated by a window, bow crossing the violin at a believable angle. Show the seated posture, instrument contact, and quiet blue evening light.",
        "Paint a fisherman repairing a green net on a dock. The person sits on a crate, both hands near the net, with one small boat behind.",
        "Paint a cyclist walking a bicycle up a cobbled hill, not riding it. One hand holds the handlebar; the wheels touch the slope and houses recede uphill.",
    ],
    "spatial_stories": [
        "Paint a ceramic teapot in front of a taller copper kettle, with one white cup to the right. Keep all three handles distinct and avoid merging their silhouettes.",
        "Paint a cat beneath a wooden dining chair looking toward a fishbowl on the seat. The bowl must sit on the chair while the cat stays underneath.",
        "Paint a toy rocket standing beside a tipped-over wooden block tower. The rocket is upright and nearer; the fallen blocks spread to the left.",
        "Paint two sailboats crossing on a lake, the nearer red-sailed boat in front and to the left of a smaller white-sailed boat. Keep masts and hulls separate.",
        "Paint a brass lantern hanging from a hook above a stack of three books. The lantern must not touch the books; warm light falls across their spines.",
        "Paint a fox peeking from behind a large blue watering can in a garden. Show enough of its face and tail to read the overlap, with flowers kept secondary.",
    ],
    "unusual_subjects": [
        "Paint a delicate glass snow globe containing one tiny lighthouse. Keep the globe's sphere, wooden base, miniature island, and a few suspended flakes separate.",
        "Paint a folded paper crane beside its long cast shadow on a desk. Preserve the angular beak and wings while using warm, imperfect painterly edges.",
        "Paint an old brass astrolabe on dark velvet. Show a large round outer ring, a few thin inner arcs and one pointer; avoid dense illegible markings.",
        "Paint a woven basket holding three seashells and a spool of blue thread. Keep the shells' curved openings clear and the basket's weave understated.",
        "Paint a hand-cranked coffee grinder beside loose beans. Distinguish the top handle from the rectangular wooden body and show one drawer at the base.",
        "Paint a weathered carousel horse under a striped canopy. Keep the horse's anatomy, vertical pole, saddle, and canopy relation coherent, with festive but gentle color.",
    ],
}

# Reviewed as a 96-photo contact sheet. These 60 avoid empty/sign-only views,
# very small subjects and photos whose catalog category alone was misleading.
PHOTO_IDS = {
    "animals": "139872 558073 046378 181859 278353 555705",
    "vehicles": "257896 502347 577932 338986 572620 246963",
    "food": "541634 157601 188592 402346 100582 352900",
    "people": "379332 162415 423123 147740 549930 051938",
    "interiors": "109441 090108 186632 175364 125778 031735",
    "nature": "000785 357742 384666 546829 229997 296969",
    "streets": "361142 322968 181816 119516 221754 058636",
    "sports": "522889 479953 457884 019432 229849 535306",
    "objects": "160666 248631 262938 213224 465718 434479",
    "other": "399296 238866 261161 458054 081061 398652",
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def prior_photo_ids() -> set[str]:
    used: set[str] = set()
    for path in list(PRIOR.glob("*/manifest.json")) + list((ROOT / "painter/runs/quality-curriculum-20260926").glob("*.json")):
        used.update(re.findall(r"coco-val2017-\d{12}", path.read_text()))
    return used


def build(output: Path) -> dict:
    assert len(BRIEFS) == 10 and all(len(v) == 6 for v in BRIEFS.values())
    briefs = [(category, text) for category, group in BRIEFS.items() for text in group]
    assert len(briefs) == len({text for _, text in briefs}) == 60
    prior_text = "\n".join(path.read_text() for path in PRIOR.glob("*/manifest.json") if path.stat().st_size < 2_000_000)
    if any(text in prior_text for _, text in briefs):
        raise ValueError("text brief already appears in prior source manifests")
    pool_path = POOL / "reference-pool.json"
    pool = json.loads(pool_path.read_text())
    used = prior_photo_ids()
    photo_categories = ("animals", "vehicles", "food", "people", "interiors", "nature", "streets", "sports", "objects", "other")
    by_id = {row["id"]: row for row in pool["entries"]}
    selected = []
    for category in photo_categories:
        for suffix in PHOTO_IDS[category].split():
            ident = f"coco-val2017-{suffix:0>12}"
            row = by_id[ident]
            if row["split"] != "train" or ident in used:
                raise ValueError(f"selected photo overlaps old authored data: {ident}")
            if not (POOL / "references" / f"{row['source_id']}.jpg").is_file():
                raise ValueError(f"selected photo unavailable locally: {ident}")
            selected.append((category, row))
    rows = []
    for category, text in briefs:
        index = len(rows) + 1
        rows.append({"id": f"fresh-text-{index:03d}", "mode": "text_to_paint", "category": category,
                     "task_text": text, "sha256": sha(text.encode()), "split": "train",
                     "tier": "pro" if category in {"people_actions", "spatial_stories", "unusual_subjects"} else "flash"})
    for visual_family, row in selected:
        photo = POOL / "references" / f"{row['source_id']}.jpg"
        raw = photo.read_bytes()
        if not raw.startswith(b"\xff\xd8\xff"):
            raise ValueError(f"invalid JPEG {photo.name}")
        rows.append({"id": row["id"], "mode": "image_to_paint", "category": visual_family,
                     "path": f"references/{row['source_id']}.jpg", "sha256": sha(raw), "bytes": len(raw),
                     "split": "train", "tier": "pro" if row["selection_category"] in {"people", "interiors", "streets", "sports"} else "flash",
                     "catalog_category": row["selection_category"],
                     "captions": row["captions"], "source_url": row["source_url"],
                     "license_name": row["license_name"], "license_url": row["license_url"]})
    assert len(rows) == 120 and len({r["id"] for r in rows}) == 120
    # Twelve-source shards keep provider calls bounded and recoverable.
    by_tier: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        by_tier.setdefault((row["mode"], row["tier"]), []).append(row)
    shards = {}
    for (mode, tier), group in by_tier.items():
        for offset in range(0, len(group), 12):
            name = f"{'text' if mode == 'text_to_paint' else 'photo'}-{tier}-{offset // 12:02d}"
            for row in group[offset:offset + 12]:
                row["shard"] = name
            shards[name] = len(group[offset:offset + 12])
    manifest = {"schema": "painter.diverse-teacher-wave.v1", "status": "candidate_generation_not_training_admission",
                "source_pool_sha256": sha(pool_path.read_bytes()), "excluded_prior_ids": len(used),
                "count": len(rows), "modes": dict(Counter(r["mode"] for r in rows)), "shards": shards,
                "rows": rows}
    raw_manifest = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                info = tarfile.TarInfo("manifest.json")
                info.size, info.mtime, info.mode = len(raw_manifest), 0, 0o644
                archive.addfile(info, io.BytesIO(raw_manifest))
                for row in rows:
                    if row["mode"] != "image_to_paint":
                        continue
                    raw = (POOL / "references" / Path(row["path"]).name).read_bytes()
                    info = tarfile.TarInfo(row["path"])
                    info.size, info.mtime, info.mode = len(raw), 0, 0o644
                    archive.addfile(info, io.BytesIO(raw))
    (HERE / "manifest.json").write_bytes(raw_manifest)
    receipt = {"schema": "painter.diverse-teacher-source-receipt.v1", "archive_sha256": sha(output.read_bytes()),
               "archive_bytes": output.stat().st_size, "manifest_sha256": sha(raw_manifest), "rows": 120, "shards": shards}
    (HERE / "source-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


if __name__ == "__main__":
    print(json.dumps(build(HERE / "source.tar.gz")))
