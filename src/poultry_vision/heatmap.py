"""Density heatmap of bird positions. Pure NumPy so it is cheap to test and to port."""

from __future__ import annotations

import numpy as np

from poultry_vision.boxes import Box


def density_grid(boxes: list[Box], rows: int = 6, cols: int = 8) -> np.ndarray:
    """Count box centres per grid cell. Returns an int array of shape (rows, cols)."""
    grid = np.zeros((rows, cols), dtype=int)
    for b in boxes:
        cx, cy = b.centre
        c = min(cols - 1, max(0, int(cx * cols)))
        r = min(rows - 1, max(0, int(cy * rows)))
        grid[r, c] += 1
    return grid


def smooth(grid: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gaussian-blur the grid (separable, zero-padded, renormalised so the total is kept)."""
    if sigma <= 0:
        return grid.astype(float)
    radius = int(3 * sigma)
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    out = grid.astype(float)
    for axis in (0, 1):
        out = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), axis, out)
    total = grid.sum()
    return out * (total / out.sum()) if out.sum() > 0 else out


def hotspot_ratio(grid: np.ndarray) -> float:
    """Peak cell count divided by the mean cell count.

    1.0 means birds are spread evenly; rows*cols means every bird is in one cell.
    0.0 when there are no birds. This is the "hotspot" number the alert rules use.
    """
    total = grid.sum()
    if total == 0:
        return 0.0
    return float(grid.max() / (total / grid.size))


def occupancy_fraction(grid: np.ndarray) -> float:
    """Fraction of grid cells that contain at least one bird."""
    return float((grid > 0).mean())
