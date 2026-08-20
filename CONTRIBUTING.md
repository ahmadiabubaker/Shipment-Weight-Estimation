# Contributing

Practical setup and workflow for this repo. The deliverable is the
`shipment_weight` Python library; FastAPI (`api/main.py`) is optional and
secondary. Real warehouse data now exists (see [MODEL_CARD.md](MODEL_CARD.md))
alongside the Phase 1 synthetic pipeline, which is still used by tests and
the notebook.

## Clone

```bash
git clone <repo-url>
cd Shipment-Weight-Estimation
```

## Set up a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (cmd)
.venv\Scripts\activate.bat
```

## Install dependencies

```bash
pip install -e ".[dev]"
```

This installs `shipment_weight` itself (editable) plus everything needed to
train, test, and run the notebook. Add `.[api]` too if you need the FastAPI
wrapper (`pip install -e ".[dev,api]"`). The library's own runtime
dependencies (pandas, numpy, scikit-learn, scipy, joblib) are installed
either way — `pip install -e .` alone is enough for a consumer who only
wants to call `shipment_weight.predict`.

Benchmarking alternative model libraries (LightGBM/XGBoost/CatBoost, e.g.
`scripts/evaluate_model_sweep.py`) needs `.[experiments]` on top of that —
these are never imported by `shipment_weight` itself and stay out of `dev`
so they don't get pulled in just to run tests or train the production model.

## Train the model

On real data (requires the Medusa Excel exports):

```bash
python scripts/train_real_data.py \
    --shipments order_shipments_anonymized.xlsx \
    --lines order_lines_in_shipment_anonymized.xlsx
```

Saves to `models/model.joblib` by default (`--out` to change it). This is
what `ShipmentWeightPredictor.load()` reads by default.

On synthetic data (Phase 1 pipeline, no real files needed):

```bash
python -m shipment_weight.train --out models/model.joblib
```

Useful flags: `--csv <path>` to train on a CSV instead of generating
synthetic data; `--model <name>` to choose
`linear_regression`/`ridge`/`random_forest`/`gradient_boosted_trees`;
`--n-shipments`/`--seed` to control synthetic data size/reproducibility.

## Use the library

```python
from shipment_weight.predict import ShipmentWeightPredictor, Shipment, Box, ItemLine

predictor = ShipmentWeightPredictor.load()  # models/model.joblib, or set MODEL_PATH
result = predictor.predict(Shipment(items=[...], box=Box(...), ship_method="..."))
```

See `src/shipment_weight/predict.py`'s module docstring for the full
contract and why it takes raw box dimensions/item lines rather than
pre-computed features.

## Packaging (building a distributable wheel)

`pip install -e .` (above) is enough for local development — it makes
`shipment_weight` importable in place and `ShipmentWeightPredictor.load()`
falls back to the repo-relative `models/model.joblib` when there's no
packaged copy yet. Building an actual wheel for distribution (`python -m
build`), or installing it somewhere with no repo checkout to fall back to,
needs one extra manual step first:

```bash
python scripts/package_model.py     # copies models/model.joblib -> src/shipment_weight/models/model.joblib
python -m build --wheel             # now the wheel genuinely contains the model
```

This is a **manual step, not an automatic build hook**, by design: a hook
that always fires on `python -m build` would silently package whatever
happens to be sitting in `models/model.joblib` at that moment, including a
stale artifact left over from an earlier training run — the project doesn't
have a release pipeline yet where "always re-copy" is obviously safe, so an
explicit step keeps you in control of exactly which trained model ships.
Re-run `scripts/package_model.py` any time you retrain and want the new
model in the next build.

`src/shipment_weight/models/` is gitignored (`.gitignore`) — the copy made
there is a build input, not something to commit; `models/model.joblib` at
the repo root stays the single source of truth.

`tests/test_package_model.py` covers this: it verifies the copy step works
and — via `python -m build --no-isolation` inside the test — that a real
built wheel actually contains `shipment_weight/models/model.joblib`, not
just that pyproject.toml claims it should. That second check needs
`build`/`wheel`/`setuptools>=68`, all part of the `dev` extra; it skips
cleanly if they're missing rather than failing an unrelated install.

## Run the API (optional, secondary)

```bash
uvicorn api.main:app --reload
```

Requires a model artifact at `models/model.joblib` (or set `MODEL_PATH`).
Docs at `http://127.0.0.1:8000/docs` once running. Read `api/main.py`'s
docstring first — it documents a known limitation (no box-dims/item-line
input, so it can't compute `fill_ratio`/`void_volume_in3` from a real
carton) that the library does not have.

## Run tests

```bash
pytest tests/
```

Tests train a small model in-process (no need to train one beforehand, and
no dependency on the real Excel files).

## Before opening a PR

- Run `pytest tests/` and make sure it passes.
- If you touch `src/shipment_weight/features.py` or `ingest.py`, remember
  they're shared by training (`scripts/train_real_data.py`) AND the library
  (`shipment_weight/predict.py`) — that's the point, don't fork a second
  feature-building path for one of them.
- If you touch `src/shipment_weight/data_gen.py`, regenerate
  `data/synthetic_shipments.csv` if anyone depends on it, and review
  [docs/data_assumptions.md](docs/data_assumptions.md) for assumptions that
  may need updating.
