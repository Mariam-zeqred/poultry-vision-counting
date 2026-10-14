"""Count birds in poultry images, build density heatmaps and raise vision alerts."""

from poultry_vision.evaluate import ImageResult, match, summarise
from poultry_vision.heatmap import density_grid, hotspot_ratio, occupancy_fraction, smooth
from poultry_vision.rules import Alert, FrameStats, VisionAlertEngine, VisionThresholds

__all__ = [
    "Alert",
    "FrameStats",
    "ImageResult",
    "VisionAlertEngine",
    "VisionThresholds",
    "density_grid",
    "hotspot_ratio",
    "match",
    "occupancy_fraction",
    "smooth",
    "summarise",
]
