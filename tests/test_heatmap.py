import numpy as np

from poultry_vision.boxes import Box, centre_distance, iou
from poultry_vision.heatmap import density_grid, hotspot_ratio, occupancy_fraction, smooth


def box_at(cx: float, cy: float, s: float = 0.05) -> Box:
    return Box(cx - s, cy - s, cx + s, cy + s)


def test_iou_and_distance():
    a, b = Box(0, 0, 1, 1), Box(0.5, 0.5, 1.5, 1.5)
    assert abs(iou(a, b) - 0.25 / 1.75) < 1e-9
    assert iou(a, Box(2, 2, 3, 3)) == 0.0
    assert abs(centre_distance(Box(0, 0, 0, 0), Box(1, 1, 1, 1)) - 1.0) < 1e-9


def test_density_grid_counts_centres():
    grid = density_grid([box_at(0.05, 0.05), box_at(0.95, 0.95), box_at(0.99, 0.99)], 4, 4)
    assert grid.sum() == 3 and grid[0, 0] == 1 and grid[3, 3] == 2
    assert density_grid([], 4, 4).sum() == 0


def test_hotspot_ratio_bounds():
    rows, cols = 4, 4
    even = [box_at((c + 0.5) / cols, (r + 0.5) / rows) for r in range(rows) for c in range(cols)]
    assert hotspot_ratio(density_grid(even, rows, cols)) == 1.0
    huddle = [box_at(0.1, 0.1)] * 6
    assert hotspot_ratio(density_grid(huddle, rows, cols)) == rows * cols
    assert hotspot_ratio(np.zeros((2, 2), dtype=int)) == 0.0
    assert occupancy_fraction(density_grid(huddle, rows, cols)) == 1 / 16


def test_smooth_keeps_total_and_spreads():
    grid = density_grid([box_at(0.5, 0.5)] * 4, 6, 8)
    sm = smooth(grid, sigma=1.0)
    assert abs(sm.sum() - 4) < 1e-9
    assert sm.max() < grid.max() and (sm > 0).sum() > 1
    assert np.array_equal(smooth(grid, 0), grid.astype(float))
