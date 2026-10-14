"""Vision alert rules in the shape the smart-poultry-farm app stores under farm/alerts.

The app's schema (docs/FIREBASE_SCHEMA.md in that repo) gives each alert a `type`,
`severity` ("info" | "warning" | "critical"), `message`, `value`, `threshold`,
`createdAt` (ms), `deviceId` ("vision-ai" for the vision engine) and `params` for
localised rendering. The vision types it lists include vision_overcrowding,
vision_activity_low, vision_bird_count_drop and vision_stale, with threshold names
hotspotSpikeRatio, activityLowSustainedCount, birdCountDropPct,
birdCountSustainedCount and staleMinutes. The rule logic itself is not public, so
the rules below are this repo's interpretation of those names.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from poultry_vision.boxes import Box, centre_distance

DEVICE_ID = "vision-ai"


@dataclass
class VisionThresholds:
    capacity: int = 30  # birds the camera's floor area may hold (stocking density * area)
    hotspot_spike_ratio: float = 3.0  # peak cell / mean cell before a huddle counts as a hotspot
    activity_low: float = 0.01  # mean centre movement per frame, image diagonal = 1
    activity_low_sustained_count: int = 3  # consecutive quiet frames before alerting
    bird_count_drop_pct: float = 20.0  # % below the baseline count
    bird_count_sustained_count: int = 3  # consecutive low frames before alerting
    stale_minutes: float = 10.0  # minutes without a frame


@dataclass
class FrameStats:
    ts_ms: int
    count: int
    hotspot_ratio: float = 0.0
    activity: float | None = None  # None when it could not be measured (first frame, no birds)


@dataclass
class Alert:
    type: str
    severity: str
    message: str
    value: float
    threshold: float
    created_at: int
    params: dict = field(default_factory=dict)

    def to_firebase(self) -> dict:
        """The dict the app expects to find under farm/alerts/{pushId}."""
        return {
            "type": self.type,
            "severity": self.severity,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "createdAt": self.created_at,
            "deviceId": DEVICE_ID,
            "params": self.params,
        }


def activity_between(prev: list[Box], cur: list[Box]) -> float | None:
    """Mean movement of matched bird centres between two frames (diagonal = 1).

    Greedy nearest-centre matching; unmatched birds are ignored. None if nothing matched.
    """
    if not prev or not cur:
        return None
    free = list(cur)
    moves: list[float] = []
    for p in prev:
        if not free:
            break
        nearest = min(free, key=lambda c: centre_distance(p, c))
        moves.append(centre_distance(p, nearest))
        free.remove(nearest)
    return sum(moves) / len(moves) if moves else None


class VisionAlertEngine:
    """Feed one FrameStats per analysed frame; get alerts on rising edges only.

    A rule that is already raised stays silent until it clears (recorded in `resolved`),
    except that a warning is re-emitted once if it escalates to critical.

    The baseline count for the drop rule is the mean of the first `baseline_frames`
    frames unless `baseline_count` is given (e.g. the number of birds placed).
    """

    def __init__(
        self,
        thresholds: VisionThresholds | None = None,
        baseline_count: float | None = None,
        baseline_frames: int = 5,
    ) -> None:
        self.t = thresholds or VisionThresholds()
        self.baseline = baseline_count
        self._baseline_frames = baseline_frames
        self._early: list[int] = []
        self._quiet = 0
        self._low = 0
        self._active: dict[str, str] = {}  # rule key -> severity currently raised
        self.last_ts: int | None = None
        self.resolved: list[tuple[str, int]] = []

    def _edge(self, alert: Alert, key: str, condition: bool, ts: int) -> Alert | None:
        """Return the alert on a rising edge or when its severity escalates; else None."""
        if condition:
            if self._active.get(key) in (None, "warning") and alert.severity != self._active.get(
                key
            ):
                self._active[key] = alert.severity
                return alert
        elif key in self._active:
            del self._active[key]
            self.resolved.append((key, ts))
        return None

    def update(self, f: FrameStats) -> list[Alert]:
        t, ts = self.t, f.ts_ms
        self.last_ts = ts
        out: list[Alert] = []
        if "vision_stale" in self._active:
            del self._active["vision_stale"]
            self.resolved.append(("vision_stale", ts))

        # Overcrowding: more birds than the area allows, or a tight huddle (hotspot).
        over = f.count > t.capacity
        severity = "critical" if f.count > 1.25 * t.capacity else "warning"
        a = Alert(
            "vision_overcrowding",
            severity,
            f"{f.count} birds in view, capacity {t.capacity}",
            f.count,
            t.capacity,
            ts,
            {"count": f.count, "capacity": t.capacity},
        )
        if (r := self._edge(a, "vision_overcrowding", over, ts)) is not None:
            out.append(r)
        hot = f.count >= 2 and f.hotspot_ratio > t.hotspot_spike_ratio
        a = Alert(
            "vision_overcrowding",
            "warning",
            f"birds huddled: hotspot ratio {f.hotspot_ratio:.1f}",
            round(f.hotspot_ratio, 2),
            t.hotspot_spike_ratio,
            ts,
            {"hotspotRatio": round(f.hotspot_ratio, 2), "count": f.count},
        )
        if (r := self._edge(a, "vision_hotspot", hot, ts)) is not None:
            out.append(r)

        # Low activity: birds barely move for N consecutive frames.
        if f.activity is not None:
            self._quiet = self._quiet + 1 if f.activity < t.activity_low else 0
        quiet = self._quiet >= t.activity_low_sustained_count
        severity = "critical" if self._quiet >= 2 * t.activity_low_sustained_count else "warning"
        a = Alert(
            "vision_activity_low",
            severity,
            f"low activity for {self._quiet} frames",
            round(f.activity or 0.0, 4),
            t.activity_low,
            ts,
            {"frames": self._quiet, "activity": round(f.activity or 0.0, 4)},
        )
        if (r := self._edge(a, "vision_activity_low", quiet, ts)) is not None:
            out.append(r)

        # Bird-count drop: sustained fall below the baseline (mortality, escape, camera moved).
        if self.baseline is None:
            self._early.append(f.count)
            if len(self._early) >= self._baseline_frames:
                self.baseline = sum(self._early) / len(self._early)
        if self.baseline:
            limit = self.baseline * (1 - t.bird_count_drop_pct / 100)
            self._low = self._low + 1 if f.count < limit else 0
            drop = self._low >= t.bird_count_sustained_count
            drop_pct = 100 * (1 - f.count / self.baseline)
            a = Alert(
                "vision_bird_count_drop",
                "critical",
                f"bird count {f.count} is {drop_pct:.0f}% below baseline {self.baseline:.0f}",
                f.count,
                round(limit, 1),
                ts,
                {"dropPct": round(drop_pct, 1), "baseline": self.baseline, "frames": self._low},
            )
            if (r := self._edge(a, "vision_bird_count_drop", drop, ts)) is not None:
                out.append(r)
        return out

    def check_stale(self, now_ms: int) -> Alert | None:
        """Call from a timer: warns once when no frame has arrived for staleMinutes."""
        if self.last_ts is None:
            return None
        minutes = (now_ms - self.last_ts) / 60_000
        a = Alert(
            "vision_stale",
            "warning",
            f"no vision frame for {minutes:.0f} min",
            round(minutes, 1),
            self.t.stale_minutes,
            now_ms,
            {"minutes": round(minutes, 1)},
        )
        return self._edge(a, "vision_stale", minutes > self.t.stale_minutes, now_ms)
