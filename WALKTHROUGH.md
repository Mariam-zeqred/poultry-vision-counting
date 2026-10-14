# Walkthrough (for Mariam)

## Why it is built this way

The job was to count chickens with a detector never trained on chickens, and be honest
about the result. The repo is built around one measured question: how far is a COCO "bird"
detector from being useful in a poultry house? The detection call is twenty lines
(`detect.py`); the value is in the manifest with real boxes, the matching code, the figures,
and the alert rules that connect the numbers to the app you shipped.

The images come from Open Images because Roboflow and Kaggle were unreachable from the build
sandbox. A good trade: the chicken boxes are human-drawn and verified, and the "group of"
flag marks the crowds, so the repo never pretends to know the true count of a packed barn.

## File map

- `data/manifest.csv` - 68 image ids, Open Images boxes, author, licence, source URL.
  `dataset.py` reads it and downloads the photos (idempotent).
- `boxes.py` - the `Box` dataclass (normalised 0..1), IoU, centre distance.
- `detect.py` - `BirdDetector`: wraps Ultralytics YOLO, keeps only COCO class 14 (`bird`);
  torch is imported lazily.
- `heatmap.py` - NumPy only: grid counts, Gaussian smoothing, hotspot ratio, occupancy.
- `evaluate.py` - greedy IoU matching, per-image TP/FP/FN, count error, `summarise()` by
  image kind. Detections inside a group box are not counted as false positives.
- `rules.py` - `VisionAlertEngine` and the `farm/alerts` shaped `Alert`: rising edge,
  escalation warning → critical, resolution tracking, stale check.
- `cli.py` - `fetch | run | evaluate | alerts`. `scripts/make_figures.py` reproduces every
  figure and `results.json`; `scripts/demo_alerts.py` runs a synthetic sequence through the
  rules.
- `tests/` - 18 tests: heatmap maths, matching, every rule, manifest consistency, and a
  real-detector smoke test (skipped without torch or images).

## Three questions a professor is likely to ask

**1. "Recall 0.58 for the nano model: why not just lower the confidence threshold?"**
Figure 02 answers it. At 0.1 the nano model's recall only reaches 0.66 while precision falls
to 0.75, and its count error never gets below 0.9 birds. The missed birds are not
low-confidence detections, they are not detected at all, because they do not look like COCO's
idea of a bird (big white bodies, overlapping, top-down views). A threshold cannot fix a
training-set mismatch; only training on poultry images can.

**2. "How do you know the ground truth is right, and what about the crowds?"**
The boxes are Open Images' human-drawn, verified annotations. For crowds the annotators drew
one group box, so those eleven images are reported only qualitatively (how many the detector
found, and that none fell outside the group box). Counting a packed house
needs density-map regression (a density per pixel, the count is its integral); boxes stop
meaning anything when birds overlap.

**3. "Where does the alert engine's logic come from, and how was it validated?"**
The alert names, fields and threshold names come from the smart-poultry-farm repo's
Firebase schema. The app's engine (`vision_alert_engine.dart`) is public too, but it works on
the analyser's per-snapshot summary and uses relative rules (a spike against the median of
the last five snapshots), so this repo's rules are a simpler re-interpretation for a
box-per-frame detector, and the README lists the differences. Validation is unit tests for
every rule plus a synthetic sequence; nothing is validated on real footage because the data
set has no video.

## One thing to extend

Fine-tune `yolov8n` on 200–300 labelled poultry-house images (one command:
`yolo train model=yolov8n.pt data=chickens.yaml epochs=50`) and rerun `make figures`; the
evaluation code, figures and README table need no change. Going further: a density-map
model for the flock images, checked against a few counts done by hand.
