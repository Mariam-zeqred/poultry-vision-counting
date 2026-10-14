from poultry_vision.boxes import Box
from poultry_vision.evaluate import ImageResult, match, summarise

GT = [Box(0.1, 0.1, 0.3, 0.3), Box(0.6, 0.6, 0.8, 0.8)]


def test_match_perfect():
    assert match([Box(0.1, 0.1, 0.3, 0.3, 0.9), Box(0.6, 0.6, 0.8, 0.8, 0.8)], GT) == (2, 0, 0)


def test_match_miss_and_false_positive():
    dets = [Box(0.1, 0.1, 0.3, 0.3, 0.9), Box(0.4, 0.4, 0.5, 0.5, 0.7)]
    assert match(dets, GT) == (1, 1, 1)


def test_match_is_one_to_one():
    dets = [Box(0.1, 0.1, 0.3, 0.3, 0.9), Box(0.11, 0.1, 0.31, 0.3, 0.5)]
    assert match(dets, GT) == (1, 1, 1)


def test_detection_inside_group_box_is_forgiven():
    gts = [Box(0.0, 0.0, 0.5, 0.5, group=True)]
    assert match([Box(0.1, 0.1, 0.2, 0.2, 0.9)], gts) == (0, 0, 0)
    assert match([Box(0.7, 0.7, 0.9, 0.9, 0.9)], gts) == (0, 1, 0)


def test_summarise():
    rs = [
        ImageResult("a", "multi", 3, 2, 2, 0, 1),
        ImageResult("b", "multi", 2, 3, 2, 1, 0),
        ImageResult("c", "single", 1, 1, 1, 0, 0),
    ]
    s = summarise(rs)
    assert s["all"]["images"] == 3 and s["all"]["exact_rate"] == 1 / 3
    assert abs(s["all"]["mae"] - 2 / 3) < 1e-9 and abs(s["all"]["bias"]) < 1e-9
    assert s["multi"]["precision"] == 4 / 5 and s["multi"]["recall"] == 4 / 5
    assert s["single"]["precision"] == 1.0
