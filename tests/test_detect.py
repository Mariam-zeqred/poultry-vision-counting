"""Smoke test for the real detector; skipped when torch/ultralytics or the image is absent."""

import pytest

from poultry_vision import dataset
from poultry_vision.evaluate import match

pytest.importorskip("ultralytics")

EASY = "9b021e285d010caa"  # six boxed chickens, test split


@pytest.fixture(scope="module")
def detector():
    from poultry_vision.detect import BirdDetector

    return BirdDetector("yolov8n.pt", conf=0.25)


def test_detect_returns_normalised_boxes(detector):
    rec = next(r for r in dataset.load_manifest() if r.image_id == EASY)
    if not rec.path().exists():
        pytest.skip("run `make data` first")
    boxes = detector.detect(rec.path())
    assert boxes, "expected at least one bird on the easy image"
    for b in boxes:
        assert 0 <= b.xmin < b.xmax <= 1 and 0 <= b.ymin < b.ymax <= 1 and 0.25 <= b.conf <= 1
    tp, _fp, _fn = match(boxes, list(rec.boxes))
    assert tp >= 1
