"""Reproduce figures/*.png and figures/results.json: three YOLOv8 sizes vs Open Images boxes.

Run `make data` first. CPU time is a few minutes. Every number in the README comes from
the results.json this script writes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from poultry_vision import dataset  # noqa: E402
from poultry_vision.cli import evaluate_records  # noqa: E402
from poultry_vision.detect import BirdDetector  # noqa: E402
from poultry_vision.evaluate import summarise  # noqa: E402
from poultry_vision.heatmap import density_grid, hotspot_ratio, smooth  # noqa: E402
from poultry_vision.rules import HOTSPOT_GRID  # noqa: E402
from poultry_vision.viz import draw_boxes, heatmap_overlay  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures"
WEIGHTS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
CONFS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]


def main() -> None:
    FIG.mkdir(exist_ok=True)
    recs = [r for r in dataset.load_manifest() if r.path().exists()]
    if len(recs) < len(dataset.load_manifest()):
        sys.exit("run `make data` first")
    by_id = {r.image_id: r for r in recs}
    results = {"images": len(recs), "models": {}, "conf_sweep": {}}
    per_model = {}
    for w in WEIGHTS:
        det = BirdDetector(w, conf=0.25)
        rs = evaluate_records(det, recs, dataset.IMAGE_DIR)
        per_model[w] = rs
        results["models"][w] = summarise(rs)
        sweep = {}
        for c in CONFS:
            sweep[str(c)] = summarise(evaluate_records(det, recs, dataset.IMAGE_DIR, conf=c))["all"]
        results["conf_sweep"][w] = sweep
        print(w, json.dumps(results["models"][w]["all"]))

    # 01: predicted vs true count, one panel per model (flock images excluded: no truth).
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    for ax, w in zip(axes, WEIGHTS, strict=True):
        rs = [r for r in per_model[w] if r.kind != "flock"]
        ax.plot([0, 7], [0, 7], "--", color="grey", lw=1)
        ax.scatter([r.gt for r in rs], [r.pred for r in rs], alpha=0.5, s=36)
        m = results["models"][w]["all"]
        ax.set_title(f"{w}: MAE {m['mae']:.2f}, recall {m['recall']:.2f}", fontsize=10)
        ax.set_xlabel("chickens in Open Images boxes")
        ax.set_xticks(range(8))
    axes[0].set_ylabel("birds detected (COCO 'bird', conf 0.25)")
    fig.tight_layout()
    fig.savefig(FIG / "01_count_vs_truth.png", dpi=120)
    plt.close(fig)

    # 02: confidence sweep for every model.
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for w in WEIGHTS:
        sw = results["conf_sweep"][w]
        xs = [float(c) for c in sw]
        axes[0].plot(xs, [sw[c]["recall"] for c in sw], marker="o", label=w)
        axes[1].plot(xs, [sw[c]["precision"] for c in sw], marker="o", label=w)
        axes[2].plot(xs, [sw[c]["mae"] for c in sw], marker="o", label=w)
    for ax, name in zip(axes, ["recall", "precision", "count MAE"], strict=True):
        ax.set_xlabel("confidence threshold")
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "02_conf_sweep.png", dpi=120)
    plt.close(fig)

    # 03/04: best and worst yolov8n cases, drawn with ground truth (green) and detections (red).
    det_n = BirdDetector("yolov8n.pt", conf=0.25)
    rs_n = per_model["yolov8n.pt"]
    multi = [r for r in rs_n if r.kind == "multi"]
    best = sorted(multi, key=lambda r: (abs(r.error), -r.gt))[:3]
    worst = sorted(multi, key=lambda r: (r.error, -r.gt))[:3]
    flock = sorted((r for r in rs_n if r.kind == "flock"), key=lambda r: -r.pred)[:3]
    for name, chosen in [("03_good_cases", best), ("04_failure_cases", worst + flock)]:
        fig, axes = plt.subplots(1, len(chosen), figsize=(4 * len(chosen), 4))
        for ax, r in zip(axes, chosen, strict=True):
            rec = by_id[r.image_id]
            img = draw_boxes(Image.open(rec.path()), det_n.detect(rec.path()), list(rec.boxes))
            ax.imshow(img)
            truth = f"truth {r.gt}" if r.kind != "flock" else "truth: crowd (group box)"
            ax.set_title(f"{truth}, detected {r.pred}", fontsize=9)
            ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(FIG / f"{name}.png", dpi=110)
        plt.close(fig)
    results["examples"] = {
        "good": [r.image_id for r in best],
        "worst_undercount": [(r.image_id, r.gt, r.pred) for r in worst],
        "flock": [(r.image_id, r.pred) for r in flock],
    }

    # 05: density heatmap on the image with the most detections.
    top = max(rs_n, key=lambda r: r.pred)
    rec = by_id[top.image_id]
    boxes = det_n.detect(rec.path())
    grid = density_grid(boxes)
    ratio = hotspot_ratio(density_grid(boxes, *HOTSPOT_GRID))
    heatmap_overlay(
        Image.open(rec.path()),
        smooth(grid, 0.8),
        FIG / "05_heatmap.png",
        f"{len(boxes)} detections; hotspot ratio {ratio:.1f} on the rule grid {HOTSPOT_GRID}",
    )
    results["heatmap_example"] = {
        "image_id": rec.image_id,
        "detections": len(boxes),
        "hotspot_ratio": round(ratio, 2),
        "grid_6x8": grid.tolist(),
    }
    (FIG / "results.json").write_text(json.dumps(results, indent=2))
    print("wrote", FIG / "results.json")


if __name__ == "__main__":
    main()
