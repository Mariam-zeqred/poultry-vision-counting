"""Compare detections with Open Images ground-truth boxes: matching, count error, P/R."""

from __future__ import annotations

from dataclasses import dataclass, field

from poultry_vision.boxes import Box, iou


def match(dets: list[Box], gts: list[Box], iou_thr: float = 0.5) -> tuple[int, int, int]:
    """Greedy one-to-one matching, highest-confidence detection first.

    Returns (true positives, false positives, false negatives). Group-of boxes are
    excluded from the count but any detection inside one is forgiven (not a false
    positive), because the crowd it covers has no per-bird boxes to match.
    """
    singles = [g for g in gts if not g.group]
    groups = [g for g in gts if g.group]
    used = [False] * len(singles)
    tp = fp = 0
    for d in sorted(dets, key=lambda b: -b.conf):
        best, best_iou = -1, iou_thr
        for i, g in enumerate(singles):
            if not used[i]:
                v = iou(d, g)
                if v >= best_iou:
                    best, best_iou = i, v
        if best >= 0:
            used[best] = True
            tp += 1
        elif not any(_inside(d, g) for g in groups):
            fp += 1
    return tp, fp, len(singles) - tp


def _inside(d: Box, g: Box) -> bool:
    cx, cy = d.centre
    return g.xmin <= cx <= g.xmax and g.ymin <= cy <= g.ymax


@dataclass
class ImageResult:
    image_id: str
    kind: str  # multi | single | flock
    gt: int  # per-bird ground-truth boxes (group boxes excluded)
    pred: int
    tp: int
    fp: int
    fn: int
    extra: dict = field(default_factory=dict)

    @property
    def error(self) -> int:
        return self.pred - self.gt


def summarise(results: list[ImageResult]) -> dict:
    """Aggregate metrics overall and per image kind."""

    def agg(rs: list[ImageResult]) -> dict:
        n = len(rs)
        if n == 0:
            return {"images": 0}
        tp = sum(r.tp for r in rs)
        fp = sum(r.fp for r in rs)
        fn = sum(r.fn for r in rs)
        return {
            "images": n,
            "gt_birds": sum(r.gt for r in rs),
            "pred_birds": sum(r.pred for r in rs),
            "mae": sum(abs(r.error) for r in rs) / n,
            "bias": sum(r.error for r in rs) / n,
            "exact_rate": sum(r.error == 0 for r in rs) / n,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
        }

    out = {"all": agg(results)}
    for kind in sorted({r.kind for r in results}):
        out[kind] = agg([r for r in results if r.kind == kind])
    return out
