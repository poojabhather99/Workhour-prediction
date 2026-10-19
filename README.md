# Software Effort Estimator

[![CI](https://github.com/poojabhather99/Workhour-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/poojabhather99/Workhour-prediction/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

Predicts the effort (person-hours) needed to deliver a software project from its size and team characteristics, and serves the estimate through a Streamlit app.

The dataset is the classic Desharnais software-effort dataset: 81 projects described by team and manager experience, project length, programming language, and function-point measures.

## Results

Evaluated with repeated 5-fold cross-validation (5 repeats) on all 81 projects. Effort is right-skewed, so every model is fit on log(effort) and predictions are transformed back to hours.

| Model | MAE (hours) | RMSE (hours) | R² |
|---|---|---|---|
| **Gradient boosting** | **1,946** | **2,987** | **0.42** |
| Random forest | 2,025 | 3,091 | 0.38 |
| Ridge (linear baseline) | 2,265 | 3,852 | 0.06 |

Gradient boosting is selected automatically. On a single 20% hold-out split it scores R² 0.65, but with only 16 test projects that number is noisy, so the cross-validated figure is the one to trust. With a dataset this small, expect estimates to be indicative rather than precise.

## What changed from v0

The first version trained a single linear regression on every Streamlit rerun, never measured accuracy, and appended the user's input to the test set, which could overwrite real rows. This version:

- evaluates three models with cross-validation and reports MAE, RMSE and R²
- trains once, saves the model, and caches it in the app
- predicts from a validated one-row DataFrame instead of mutating data
- is a tested package (pytest) with linting and CI on Python 3.10 and 3.12

## Project structure

```
data/effort.csv          dataset
src/workhour/data.py     loading and validation
src/workhour/model.py    candidate models, cross-validation, persistence, prediction
src/workhour/train.py    training entry point, writes models/metrics.json
src/workhour/app.py      Streamlit app
tests/                   unit tests
.github/workflows/       CI (ruff + pytest)
```

## Run it

```bash
git clone https://github.com/poojabhather99/Workhour-prediction.git
cd Workhour-prediction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
export PYTHONPATH=src          # Windows: set PYTHONPATH=src

python -m workhour.train       # cross-validate, pick best model, save it
streamlit run src/workhour/app.py
python -m pytest               # run the tests
```

## Limitations and next steps

- 81 projects from one dataset: the model will not generalise to modern teams without retraining on your own data.
- Add prediction intervals so users see a range, not a single number.
- Add SHAP or permutation importance to explain which features drive each estimate.
