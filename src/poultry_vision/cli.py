"""Command line: fetch | run | evaluate | alerts.  `python -m poultry_vision --help`."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image

from poultry_vision import dataset
from poultry_vision.evaluate import ImageResult, match, summarise
from poultry_vision.heatmap import density_grid, hotspot_ratio, occupancy_fraction, smooth
from poultry_vision.rules import Alert, FrameStats, VisionAlertEngine, VisionThresholds


def cmd_fetch(a: argparse.Namespace) -> None:
    recs = dataset.load_manifest(a.manifest)[: a.limit]
    paths = dataset.fetch_images(recs, a.images)
    print(f"{len(paths)} images in {a.images}")


def _detector(a: argparse.Namespace):
    from poultry_vision.detect import BirdDetector

    return BirdDetector(a.weights, conf=a.conf, imgsz=a.imgsz)


def cmd_run(a: argparse.Namespace) -> None:
    """Detect -> count -> heatmap for every JPEG/PNG in --images; CSV + annotated images."""
    det = _detector(a)
    out = Path(a.out)
    (out / "annotated").mkdir(parents=True, exist_ok=True)
    gts = {r.image_id: r for r in dataset.load_manifest(a.manifest)} if a.manifest.exists() else {}
    paths = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in {".jpg", ".png"})
    rows = []
    for p in paths[: a.limit]:
        boxes = det.detect(p)
        grid = density_grid(boxes, a.rows, a.cols)
        rows.append(
            {
                "image": p.name,
                "count": len(boxes),
                "max_conf": round(max((b.conf for b in boxes), default=0.0), 3),
                "hotspot_ratio": round(hotspot_ratio(grid), 2),
                "occupancy": round(occupancy_fraction(grid), 3),
                "gt_chickens": gts[p.stem].gt_chickens if p.stem in gts else "",
            }
        )
        if not a.no_images:
            from poultry_vision.viz import draw_boxes, heatmap_overlay

            img = Image.open(p)
            gt_boxes = list(gts[p.stem].boxes) if p.stem in gts else []
            draw_boxes(img, boxes, gt_boxes).save(out / "annotated" / p.name)
            heatmap_overlay(
                img,
                smooth(grid, 0.8),
                out / "annotated" / f"{p.stem}_heat.png",
                f"{len(boxes)} birds",
            )
        print(f"{p.name}: {rows[-1]['count']} birds, hotspot {rows[-1]['hotspot_ratio']}")
    _write_csv(out / "counts.csv", rows)
    print(f"wrote {out / 'counts.csv'} ({len(rows)} rows)")


def evaluate_records(det, recs, image_dir: Path, conf: float | None = None) -> list[ImageResult]:
    results = []
    for r in recs:
        boxes = det.detect(r.path(image_dir), conf=conf)
        tp, fp, fn = match(boxes, list(r.boxes))
        grid = density_grid(boxes)
        results.append(
            ImageResult(
                r.image_id,
                r.kind,
                r.gt_chickens,
                len(boxes),
                tp,
                fp,
                fn,
                {"hotspot_ratio": round(hotspot_ratio(grid), 2), "group_boxes": r.gt_group_boxes},
            )
        )
    return results


def cmd_evaluate(a: argparse.Namespace) -> None:
    """Detections vs the Open Images boxes for every manifest image that is on disk."""
    det = _detector(a)
    recs = [r for r in dataset.load_manifest(a.manifest) if r.path(a.images).exists()][: a.limit]
    results = evaluate_records(det, recs, a.images)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "image_id": r.image_id,
            "kind": r.kind,
            "gt": r.gt,
            "pred": r.pred,
            "tp": r.tp,
            "fp": r.fp,
            "fn": r.fn,
            **r.extra,
        }
        for r in results
    ]
    _write_csv(out / f"per_image_{det.weights.removesuffix('.pt')}.csv", rows)
    summary = {"weights": det.weights, "conf": det.conf, "imgsz": det.imgsz, **summarise(results)}
    (out / f"summary_{det.weights.removesuffix('.pt')}.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))


def cmd_alerts(a: argparse.Namespace) -> None:
    """Replay a frames CSV (ts_ms,count,hotspot_ratio,activity) through the alert engine."""
    th = VisionThresholds(**json.loads(a.thresholds)) if a.thresholds else VisionThresholds()
    engine = VisionAlertEngine(th, baseline_count=a.baseline)
    alerts: list[Alert] = []
    with Path(a.frames).open(newline="") as fh:
        for row in csv.DictReader(fh):
            act = row.get("activity")
            f = FrameStats(
                int(row["ts_ms"]),
                int(row["count"]),
                float(row.get("hotspot_ratio") or 0),
                float(act) if act not in (None, "") else None,
            )
            alerts.extend(engine.update(f))
    if a.now is not None and (s := engine.check_stale(a.now)):
        alerts.append(s)
    print(json.dumps([x.to_firebase() for x in alerts], indent=2))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        if rows:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="poultry_vision", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--manifest", type=Path, default=dataset.MANIFEST)
        sp.add_argument("--images", type=Path, default=dataset.IMAGE_DIR)
        sp.add_argument("--limit", type=int, default=None)

    def model(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--weights", default="yolov8n.pt")
        sp.add_argument("--conf", type=float, default=0.25)
        sp.add_argument("--imgsz", type=int, default=640)

    s = sub.add_parser("fetch", help="download the manifest images")
    common(s)
    s.set_defaults(fn=cmd_fetch)
    s = sub.add_parser("run", help="detect, count, heatmap -> CSV + annotated images")
    common(s)
    model(s)
    s.add_argument("--out", type=Path, default=Path("out/run"))
    s.add_argument("--rows", type=int, default=6)
    s.add_argument("--cols", type=int, default=8)
    s.add_argument("--no-images", action="store_true", help="CSV only")
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser("evaluate", help="compare with Open Images ground truth")
    common(s)
    model(s)
    s.add_argument("--out", type=Path, default=Path("out/eval"))
    s.set_defaults(fn=cmd_evaluate)
    s = sub.add_parser("alerts", help="replay a frames CSV through the alert rules")
    s.add_argument("frames", type=Path)
    s.add_argument("--thresholds", help="JSON, e.g. '{\"capacity\": 12}'")
    s.add_argument("--baseline", type=float, default=None)
    s.add_argument("--now", type=int, default=None, help="ms timestamp for the stale check")
    s.set_defaults(fn=cmd_alerts)
    return p


def main(argv: list[str] | None = None) -> None:
    a = build_parser().parse_args(argv)
    a.fn(a)
