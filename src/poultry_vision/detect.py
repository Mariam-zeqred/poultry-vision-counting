"""Thin wrapper around an Ultralytics YOLO model that returns normalised Box objects."""

from __future__ import annotations

from pathlib import Path

from poultry_vision.boxes import Box

COCO_BIRD = "bird"  # COCO has no chicken class; class 14 is the generic "bird"


class BirdDetector:
    """Detect COCO "bird" objects with a pre-trained YOLOv8 model (no fine-tuning).

    `weights` is any Ultralytics checkpoint name or path ("yolov8n.pt" downloads once).
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf: float = 0.25,
        imgsz: int = 640,
        classes: tuple[str, ...] = (COCO_BIRD,),
    ) -> None:
        from ultralytics import YOLO  # imported here so the rules/heatmap tests need no torch

        self.model = YOLO(weights)
        self.weights = Path(weights).name
        self.conf = conf
        self.imgsz = imgsz
        names = self.model.names
        self.class_ids = [i for i, n in names.items() if n in classes]
        if not self.class_ids:
            raise ValueError(f"none of {classes} in model classes {sorted(names.values())}")

    def detect(self, image: str | Path, conf: float | None = None) -> list[Box]:
        """Run the model on one image (path or array) and return boxes in 0..1 coordinates."""
        result = self.model.predict(
            str(image),
            conf=self.conf if conf is None else conf,
            imgsz=self.imgsz,
            classes=self.class_ids,
            verbose=False,
        )[0]
        xyxyn = result.boxes.xyxyn.tolist()
        confs = result.boxes.conf.tolist()
        return [
            Box(x1, y1, x2, y2, conf=c) for (x1, y1, x2, y2), c in zip(xyxyn, confs, strict=True)
        ]


def count(boxes: list[Box]) -> int:
    return len(boxes)
