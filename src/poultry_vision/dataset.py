"""The image set: 68 Open Images photos with chicken boxes, listed in data/manifest.csv.

Images are not committed (24 MB); `fetch_images` downloads them from the public
Open Images bucket. Every photo is CC BY 2.0 (author and source URL are in the manifest).
"""

from __future__ import annotations

import csv
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from poultry_vision.boxes import Box

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "manifest.csv"
IMAGE_DIR = ROOT / "data" / "images"
BUCKET = "https://open-images-dataset.s3.amazonaws.com"


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    split: str  # validation | test (Open Images subset)
    kind: str  # multi: 2+ boxed chickens; single: one; flock: crowd covered by group boxes
    gt_chickens: int  # per-bird boxes (group boxes excluded)
    gt_group_boxes: int
    gt_other_birds: int  # Bird / Duck / Turkey boxes in the same image
    boxes: tuple[Box, ...]
    license: str
    author: str
    source_url: str

    @property
    def url(self) -> str:
        return f"{BUCKET}/{self.split}/{self.image_id}.jpg"

    def path(self, image_dir: Path = IMAGE_DIR) -> Path:
        return image_dir / f"{self.image_id}.jpg"


def _parse_boxes(text: str) -> tuple[Box, ...]:
    out = []
    for item in text.split(";"):
        if item:
            x1, y1, x2, y2, grp = item.split(",")
            out.append(Box(float(x1), float(y1), float(x2), float(y2), group=grp == "1"))
    return tuple(out)


def load_manifest(path: Path = MANIFEST) -> list[ImageRecord]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [
            ImageRecord(
                image_id=r["image_id"],
                split=r["split"],
                kind=r["kind"],
                gt_chickens=int(r["gt_chickens"]),
                gt_group_boxes=int(r["gt_group_boxes"]),
                gt_other_birds=int(r["gt_other_birds"]),
                boxes=_parse_boxes(r["boxes"]),
                license=r["license"],
                author=r["author"],
                source_url=r["source_url"],
            )
            for r in csv.DictReader(fh)
        ]


def fetch_images(
    records: list[ImageRecord], image_dir: Path = IMAGE_DIR, timeout: float = 60
) -> list[Path]:
    """Download any missing image; returns the local paths. Idempotent."""
    image_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for rec in records:
        p = rec.path(image_dir)
        if not p.exists() or p.stat().st_size == 0:
            with urllib.request.urlopen(rec.url, timeout=timeout) as resp:  # noqa: S310
                p.write_bytes(resp.read())
        paths.append(p)
    return paths
