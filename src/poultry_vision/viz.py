"""Draw detections, ground truth and heatmaps onto images (PIL + matplotlib)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from poultry_vision.boxes import Box

GT_COLOUR = (40, 200, 40)
DET_COLOUR = (230, 40, 40)
GROUP_COLOUR = (240, 180, 0)


def draw_boxes(image: Image.Image, dets: list[Box], gts: list[Box] = ()) -> Image.Image:
    """Ground truth in green (group boxes dashed yellow), detections in red with confidence."""
    img = image.convert("RGB").copy()
    w, h = img.size
    d = ImageDraw.Draw(img)
    for g in gts:
        colour = GROUP_COLOUR if g.group else GT_COLOUR
        d.rectangle([g.xmin * w, g.ymin * h, g.xmax * w, g.ymax * h], outline=colour, width=3)
    for b in dets:
        x1, y1, x2, y2 = b.xmin * w, b.ymin * h, b.xmax * w, b.ymax * h
        d.rectangle([x1, y1, x2, y2], outline=DET_COLOUR, width=3)
        d.rectangle([x1, max(0, y1 - 14), x1 + 46, y1], fill=DET_COLOUR)
        d.text((x1 + 3, max(0, y1 - 13)), f"{b.conf:.2f}", fill="white")
    return img


def heatmap_overlay(image: Image.Image, grid: np.ndarray, out: Path, title: str = "") -> None:
    """Save the image with the (smoothed) density grid blended on top."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6 * image.height / image.width))
    ax.imshow(image)
    ax.imshow(
        grid,
        cmap="inferno",
        alpha=0.45,
        extent=(0, image.width, image.height, 0),
        interpolation="bilinear",
    )
    ax.set_axis_off()
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
