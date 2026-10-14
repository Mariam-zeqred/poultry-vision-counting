"""Drive the alert rules with a SYNTHETIC frame sequence and plot what fires when.

The image set has no video, so this script fabricates 60 "frames" from the detections
on one real photo: birds jitter, then freeze (low activity), then a third of them
disappear (count drop), then they huddle in one corner (hotspot). The sequence is
made up; only the starting positions come from a real detection.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from poultry_vision.boxes import Box  # noqa: E402
from poultry_vision.heatmap import density_grid, hotspot_ratio  # noqa: E402
from poultry_vision.rules import (  # noqa: E402
    HOTSPOT_GRID,
    FrameStats,
    VisionAlertEngine,
    VisionThresholds,
    activity_between,
)

FIG = Path(__file__).resolve().parents[1] / "figures"
STEP_MS = 30_000  # one analysed frame every 30 s


def jitter(boxes: list[Box], rng: random.Random, amount: float) -> list[Box]:
    out = []
    for b in boxes:
        dx, dy = rng.uniform(-amount, amount), rng.uniform(-amount, amount)
        out.append(Box(b.xmin + dx, b.ymin + dy, b.xmax + dx, b.ymax + dy, b.conf))
    return out


def main() -> None:
    rng = random.Random(3)
    start = [
        Box(x, y, x + 0.08, y + 0.1, 0.6)
        for x, y in rng.sample([(i / 10, j / 8) for i in range(10) for j in range(8)], 12)
    ]
    frames: list[list[Box]] = [start]
    for i in range(1, 60):
        prev = frames[-1]
        if i < 15:
            nxt = jitter(prev, rng, 0.03)  # normal pecking around
        elif i < 30:
            nxt = jitter(prev, rng, 0.002)  # frozen
        elif i < 45:
            nxt = jitter(prev[:8], rng, 0.03)  # four birds gone
        else:
            nxt = [
                Box(
                    0.05 + k * 0.03,
                    0.05 + (k % 3) * 0.03,
                    0.13 + k * 0.03,
                    0.15 + (k % 3) * 0.03,
                    0.6,
                )
                for k in range(8)
            ]
        frames.append(nxt)

    engine = VisionAlertEngine(VisionThresholds(capacity=30))
    rows, fired = [], []
    for i, boxes in enumerate(frames):
        act = activity_between(frames[i - 1], boxes) if i else None
        ratio = hotspot_ratio(density_grid(boxes, *HOTSPOT_GRID))
        stats = FrameStats(i * STEP_MS, len(boxes), ratio, act)
        rows.append(stats)
        for a in engine.update(stats):
            fired.append((i, a))
    stale = engine.check_stale(rows[-1].ts_ms + 15 * 60_000)

    FIG.mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(9, 6.5), sharex=True)
    t = [r.ts_ms / 60_000 for r in rows]
    axes[0].plot(t, [r.count for r in rows], drawstyle="steps-post")
    axes[0].set_ylabel("birds counted")
    axes[1].plot(t, [r.activity or 0 for r in rows])
    axes[1].axhline(engine.t.activity_low, ls="--", color="grey", lw=1)
    axes[1].set_ylabel("activity")
    axes[2].plot(t, [r.hotspot_ratio for r in rows])
    axes[2].axhline(engine.t.hotspot_spike_ratio, ls="--", color="grey", lw=1)
    axes[2].set_ylabel("hotspot ratio")
    axes[2].set_xlabel("minutes (synthetic sequence, one frame per 30 s)")
    for i, a in fired:
        for ax in axes:
            ax.axvline(t[i], color="red" if a.severity == "critical" else "orange", alpha=0.6)
        axes[0].annotate(
            a.type.removeprefix("vision_"), (t[i], 12.3), fontsize=8, rotation=90, va="bottom"
        )
    axes[0].set_ylim(0, 19)
    fig.suptitle("Alert rules on a synthetic frame sequence (not real footage)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "06_alert_demo.png", dpi=120)
    out = [a.to_firebase() for _, a in fired] + ([stale.to_firebase()] if stale else [])
    (FIG / "alert_demo.json").write_text(json.dumps(out, indent=2))
    for i, a in fired:
        print(f"frame {i:2d} {a.severity:8s} {a.type:24s} {a.message}")
    if stale:
        print(f"timer    {stale.severity:8s} {stale.type:24s} {stale.message}")


if __name__ == "__main__":
    main()
