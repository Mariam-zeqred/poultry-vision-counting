"""Axis-aligned boxes in normalised image coordinates (0..1, origin top-left)."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    conf: float = 1.0  # detector confidence; 1.0 for ground truth
    group: bool = False  # Open Images "IsGroupOf": one box around a crowd

    @property
    def centre(self) -> tuple[float, float]:
        return (self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2

    @property
    def area(self) -> float:
        return max(0.0, self.xmax - self.xmin) * max(0.0, self.ymax - self.ymin)


def iou(a: Box, b: Box) -> float:
    """Intersection over union; 0 when the boxes do not overlap."""
    w = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    h = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / (a.area + b.area - inter)


def centre_distance(a: Box, b: Box) -> float:
    """Distance between box centres, normalised so the image diagonal is 1."""
    (ax, ay), (bx, by) = a.centre, b.centre
    return math.hypot(ax - bx, ay - by) / math.sqrt(2)
