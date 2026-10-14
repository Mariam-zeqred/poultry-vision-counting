PY ?= python3
export PYTHONPATH := src

.PHONY: check lint test data run evaluate figures smoke clean

check: lint test

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

test:
	$(PY) -m pytest -q

# Download the 68 Open Images chicken photos listed in data/manifest.csv (about 24 MB).
data:
	$(PY) -m poultry_vision fetch

# Detect -> count -> heatmap -> CSV + annotated images for every image in data/images.
run:
	$(PY) -m poultry_vision run --images data/images --out out/run

# Compare detections with the Open Images ground truth, write out/eval/*.csv and summary.json.
evaluate:
	$(PY) -m poultry_vision evaluate --out out/eval

# Reproduce every PNG in figures/ and figures/results.json (runs n, s and m models; ~5 min on CPU).
figures:
	$(PY) scripts/make_figures.py
	$(PY) scripts/demo_alerts.py

# Three images end to end; what CI runs after the unit tests.
smoke:
	$(PY) -m poultry_vision fetch --limit 3
	$(PY) -m poultry_vision run --images data/images --out out/smoke --limit 3
	test -s out/smoke/counts.csv

clean:
	rm -rf out .pytest_cache .ruff_cache
