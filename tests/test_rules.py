from poultry_vision.boxes import Box
from poultry_vision.rules import (
    DEVICE_ID,
    FrameStats,
    VisionAlertEngine,
    VisionThresholds,
    activity_between,
)

T = VisionThresholds(
    capacity=10,
    hotspot_spike_ratio=3.0,
    activity_low=0.01,
    activity_low_sustained_count=3,
    bird_count_drop_pct=20,
    bird_count_sustained_count=2,
    stale_minutes=10,
)


def frames(counts, **kw):
    return [FrameStats(ts_ms=1000 * i, count=c, **kw) for i, c in enumerate(counts)]


def run(engine, fs):
    return [a for f in fs for a in engine.update(f)]


def test_overcrowding_rising_edge_and_severity():
    e = VisionAlertEngine(T, baseline_count=10)
    alerts = run(e, frames([8, 11, 11, 12, 9]))
    assert [a.type for a in alerts] == ["vision_overcrowding"]
    assert alerts[0].severity == "warning" and alerts[0].value == 11 and alerts[0].threshold == 10
    assert ("vision_overcrowding", 4000) in e.resolved
    e2 = VisionAlertEngine(T, baseline_count=10)
    assert [a.severity for a in run(e2, frames([11, 13, 13]))] == ["warning", "critical"]


def test_hotspot_huddle_is_overcrowding_warning():
    e = VisionAlertEngine(T, baseline_count=10)
    alerts = run(e, frames([10, 10], hotspot_ratio=5.0))
    assert len(alerts) == 1 and alerts[0].type == "vision_overcrowding"
    assert alerts[0].params["hotspotRatio"] == 5.0
    assert run(VisionAlertEngine(T, baseline_count=10), frames([1], hotspot_ratio=48.0)) == []


def test_low_activity_needs_sustained_frames():
    e = VisionAlertEngine(T, baseline_count=10)
    assert run(e, frames([10, 10], activity=0.001)) == []
    alerts = run(e, frames([10], activity=0.001))
    assert [a.type for a in alerts] == ["vision_activity_low"] and alerts[0].params["frames"] == 3
    assert run(e, frames([10, 10], activity=0.001)) == []  # still active, no repeat
    assert run(e, frames([10], activity=0.05)) == [] and e.resolved[-1][0] == "vision_activity_low"
    e3 = VisionAlertEngine(T, baseline_count=10)
    escalated = run(e3, frames([10] * 7, activity=0.0))
    assert [a.severity for a in escalated] == ["warning", "critical"]
    assert [a.params["frames"] for a in escalated] == [3, 6]


def test_bird_count_drop_uses_learned_baseline():
    e = VisionAlertEngine(T, baseline_frames=3)
    assert run(e, frames([10, 10, 10])) == [] and e.baseline == 10
    assert run(e, frames([7])) == []
    alerts = run(e, frames([7]))
    assert [a.type for a in alerts] == ["vision_bird_count_drop"]
    assert alerts[0].severity == "critical" and alerts[0].params["dropPct"] == 30.0


def test_stale_fires_once_and_clears_on_next_frame():
    e = VisionAlertEngine(T)
    assert e.check_stale(0) is None
    e.update(FrameStats(0, 5))
    assert e.check_stale(5 * 60_000) is None
    a = e.check_stale(11 * 60_000)
    assert a is not None and a.type == "vision_stale" and a.params["minutes"] == 11.0
    assert e.check_stale(12 * 60_000) is None
    e.update(FrameStats(13 * 60_000, 5))
    assert e.resolved[-1][0] == "vision_stale"


def test_firebase_shape():
    e = VisionAlertEngine(T, baseline_count=10)
    d = run(e, frames([12]))[0].to_firebase()
    assert set(d) == {
        "type",
        "severity",
        "message",
        "value",
        "threshold",
        "createdAt",
        "deviceId",
        "params",
    }
    assert d["deviceId"] == DEVICE_ID and d["createdAt"] == 0


def test_activity_between():
    a = [Box(0.1, 0.1, 0.2, 0.2), Box(0.6, 0.6, 0.7, 0.7)]
    b = [Box(0.6, 0.6, 0.7, 0.7), Box(0.1, 0.1, 0.2, 0.2)]
    assert activity_between(a, b) == 0.0
    moved = [Box(0.6, 0.6, 0.7, 0.7), Box(0.2, 0.1, 0.3, 0.2)]
    assert abs(activity_between(a, moved) - 0.1 / 2**0.5 / 2) < 1e-9
    assert activity_between([], b) is None
