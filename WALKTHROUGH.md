# Walkthrough (for Mariam)

## Why it is built this way

The job was to count chickens with a detector never trained on chickens, and be honest
about the result. So the repo is built around one measured question: how far is a COCO
"bird" detector from being useful in a poultry house? The detection call is twenty lines
(`detect.py`); the value is in the manifest with real boxes, the matching code, the figures,
and the alert rules that connect the numbers to the app you shipped.

The images come from Open Images because Roboflow and Kaggle were unreachable from the build
sandbox. It was a good trade: the chicken boxes are human-drawn and verified, and the "group
of" flag marks the crowds, so the repo never pretends to know the true count of a packed barn.

## File map

- `data/manifest.csv` - 68 image ids, Open Images boxes, author, licence, source URL.
  `dataset.py` reads it and downloads the photos (idempotent).
- `boxes.py` - the `Box` dataclass (normalised 0..1), IoU, centre distance.
- `detect.py` - `BirdDetector`: wraps Ultralytics YOLO, keeps only COCO class 14 (`bird`).
  Torch is imported lazily so every other test runs without it.
- `heatmap.py` - NumPy only: grid counts, Gaussian smoothing, hotspot ratio, occupancy.
- `evaluate.py` - greedy IoU matching, per-image TP/FP/FN, count error, `summarise()` by
  image kind. Detections inside a group box are forgiven, not counted as false positives.
- `rules.py` - `VisionAlertEngine` and the `farm/alerts` shaped `Alert`: rising edge,
  escalation warning → critical, resolution tracking, stale check.
- `cli.py` - `fetch | run | evaluate | alerts`. `scripts/make_figures.py` reproduces every
  figure and `figures/results.json`; `scripts/demo_alerts.py` runs a labelled synthetic
  sequence through the rules.
- `tests/` - 18 tests: heatmap maths, matching, every rule, manifest consistency, and a
  real-detector smoke test that skips without torch or images.

## Three questions a professor is likely to ask

**1. "Recall 0.58 for the nano model: why not just lower the confidence threshold?"**
Figure 02 answers it. At 0.1 the nano model's recall only reaches 0.66 while precision falls
to 0.75, and its count error never gets below 0.95 birds. The missed birds are not
low-confidence detections, they are not detected at all, because they do not look like COCO's
idea of a bird (big white bodies, overlapping, top-down views). A threshold cannot fix a
training-set mismatch; only training on poultry images can.

**2. "How do you know the ground truth is right, and what about the crowds?"**
The boxes are Open Images' human-drawn, verified annotations. For crowds the annotators drew
one group box, so those eleven images are reported only qualitatively (how many the detector
found, and that none of its detections fell outside the group box). Counting a packed house
needs density-map regression, where the network predicts a density per pixel and the count is
its integral; boxes stop meaning anything when birds overlap.

**3. "Where does the alert engine's logic come from, and how was it validated?"**
The alert names, fields and threshold names come from the smart-poultry-farm repo's public
Firebase schema; the rule logic is this repo's interpretation, and the README says so.
Validation is unit tests for every rule plus a synthetic sequence exercising them in
order. It is not validated on real footage because the data set has no video; that is the
stated next step.

## One thing to extend

Fine-tune `yolov8n` on 200–300 labelled poultry-house images (one command:
`yolo train model=yolov8n.pt data=chickens.yaml epochs=50`) and rerun `make figures`. The
evaluation code, figures and README table need no change, so the before/after comparison
writes itself. Going further: a density-map model for the flock images, checked against a
few counts done by hand.
