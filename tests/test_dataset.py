from collections import Counter

from poultry_vision import dataset


def test_manifest_is_consistent():
    recs = dataset.load_manifest()
    assert len(recs) == 68
    assert Counter(r.kind for r in recs) == {"multi": 45, "single": 12, "flock": 11}
    for r in recs:
        assert r.license == "https://creativecommons.org/licenses/by/2.0/"
        assert r.gt_chickens == sum(not b.group for b in r.boxes)
        assert r.gt_group_boxes == sum(b.group for b in r.boxes)
        assert all(0 <= b.xmin < b.xmax <= 1 and 0 <= b.ymin < b.ymax <= 1 for b in r.boxes)
        assert r.url.endswith(f"/{r.split}/{r.image_id}.jpg")
    assert sum(r.gt_chickens for r in recs) == 140
