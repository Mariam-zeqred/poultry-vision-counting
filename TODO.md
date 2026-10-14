# TODO

Spec (portfolio-plan PLAN.md #6) is complete: detect → count → heatmap → CSV, `rules.py`
with the smart-poultry-farm alert types, README with the failure cases, WALKTHROUGH on
fine-tuning. What follows is what the next session or Mariam should do.

## Open (QC 2026-10-14)

1. **Align `rules.py` with the app's real engine, or keep the difference deliberate.** The
   review found that `smart-poultry-farm/app/lib/core/services/vision_alert_engine.dart`
   (and its JS prototype) is public; the first version of this README said it was not.
   The app's overcrowding and bird-count-drop rules are relative to the median of the
   previous five snapshots and its low-activity rule takes the analyser's `low` label.
   Either port those rules (a `FrameStats` → snapshot adapter plus the same median logic,
   with the Dart engine's nine scenarios as tests) or keep this repo's simpler fixed-capacity
   rules and leave the README's list of differences as the explanation. Porting is the
   better answer in an interview.
2. Code is about 1 050 lines including the two scripts, above the 600-line guideline for a
   small project; the evaluation and figure code is where it went. Nothing to cut without
   losing the measured comparison, but say so if asked.
3. `make check` needs CPU torch; the pinned `--extra-index-url download.pytorch.org` is
   blocked from the build sandbox, so the review session installed from PyPI instead.

## Needs Mariam (a GPU or a real camera)

- Fine-tune `yolov8n` on a labelled poultry-house set and rerun `make figures`
  (WALKTHROUGH "One thing to extend").
- Run on a short real clip from the poultry house to set `activityLow` and
  `hotspotSpikeRatio` from data; the demo sequence is synthetic.
- Measure inference time of `yolov8n` (and an ONNX/TFLite export) on the target board.
