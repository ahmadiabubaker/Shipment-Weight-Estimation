# Contributing

Practical setup and workflow for this repo. This is Phase 1: synthetic data,
no real warehouse data yet (see [MODEL_CARD.md](MODEL_CARD.md)).

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
pip install -r requirements.txt
```

## Train the model

Generates synthetic shipment data and trains the default candidate
(gradient boosted trees), saving the bundle to `models/model.joblib`:

```bash
PYTHONPATH=src python -m shipment_weight.train --out models/model.joblib
```

Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m shipment_weight.train --out models/model.joblib
```

Useful flags:

- `--csv <path>` — train on a real CSV instead of generated synthetic data
- `--model <name>` — choose `linear_regression`, `ridge`, `random_forest`, or `gradient_boosted_trees`
- `--n-shipments` / `--seed` — control synthetic data size/reproducibility

## Run the API

```bash
PYTHONPATH=src uvicorn api.main:app --reload
```

Requires a model artifact at `models/model.joblib` (or set `MODEL_PATH`).
Docs at `http://127.0.0.1:8000/docs` once running.

## Run tests

```bash
pytest tests/
```

Tests train a small model in-process (no need to train one beforehand).

## Before opening a PR

- Run `pytest tests/` and make sure it passes.
- If you touch `src/shipment_weight/features.py`, check `api/main.py` and the
  notebook still use the same feature list — they're meant to stay in sync.
- If you touch `src/shipment_weight/data_gen.py`, regenerate
  `data/synthetic_shipments.csv` if anyone depends on it, and review
  [docs/data_assumptions.md](docs/data_assumptions.md) for assumptions that
  may need updating.
