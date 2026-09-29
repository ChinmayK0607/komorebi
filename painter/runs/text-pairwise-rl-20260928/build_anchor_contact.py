"""Make review sheets for frozen SFT training opponents on a Linux node."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def build(manifest: Path, output: Path, per_sheet: int = 30) -> None:
    rows = json.loads(manifest.read_text())["rows"]
    output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    width, height, columns = 220, 230, 5
    for start in range(0, len(rows), per_sheet):
        part = rows[start:start + per_sheet]
        sheet = Image.new("RGB", (width * columns,
                                  height * ((len(part) + columns - 1) // columns)), "#ffffff")
        draw = ImageDraw.Draw(sheet)
        for offset, row in enumerate(part):
            x = (offset % columns) * width
            y = (offset // columns) * height
            image = Image.open(row["opponent"]).convert("RGB")
            image.thumbnail((200, 185))
            sheet.paste(image, (x + (width - image.width) // 2, y))
            brief = Path(row["reference"]).read_text().strip().replace("\n", " ")
            draw.text((x + 4, y + 190), f'{row["id"]} {row["family"]}', fill="black", font=font)
            draw.text((x + 4, y + 205), brief[:32], fill="black", font=font)
        sheet.save(output / f"anchors-{start // per_sheet + 1:02d}.jpg", quality=88)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.manifest, args.output)
