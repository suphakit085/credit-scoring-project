# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Credit default risk scoring on the Home Credit Default Risk dataset. A leakage-safe
training pipeline produces a single self-contained model bundle, which a Streamlit app
consumes for single-applicant predictions. Documentation and UI text are in Thai;
code, identifiers and column names are English.

`PROJECT_WORKFLOW.md` is the detailed pipeline reference. Note that its PowerShell
snippets have mangled paths (`..venvScriptspython.exe`); use `python` or
`.venv/Scripts/python.exe` instead.

## Commands

The virtualenv lives at `.venv/` (Python 3.12).

```powershell
python src/data_preprocessing.py --reuse-aggregates   # build data/processed/*_cleaned.csv
python src/check_data_quality.py                      # exits 1 on any quality failure
python src/model_training.py                          # full retrain (~20 min, needs ~8GB RAM)
python src/inspect_model.py                           # print bundle metadata + top features
python -m streamlit run app/streamlit_app.py
```

Tests **must** be run as `python -m pytest`, never bare `pytest`. There is no
`pytest.ini`/`pyproject.toml` and no `tests/__init__.py`, so only the `-m` form puts the
repo root on `sys.path` for the `from src.… import …` statements. Bare `pytest` fails
collection on all three test files.

```powershell
python -m pytest -q                                        # all tests (currently 9)
python -m pytest -q tests/test_inference.py                # one file
python -m pytest -q tests/test_inference.py::test_risk_level_uses_bundle_thresholds
```

Smoke test on a data subset. **Pass `--output` pointing outside `models/`**: sibling
artifacts are written with fixed names, so `--output models/dev_bundle.joblib` silently
overwrites the production `models/model_metrics.json` and `models/feature_importance.csv`
(see `src/model_training.py:251`).

```powershell
python src/model_training.py --sample-rows 20000 --output /tmp/dev/dev_bundle.joblib
```

## Architecture

### The bundle is the contract

`src/model_training.py` writes one `models/credit_scoring_bundle.joblib` holding the
input schema, training-set defaults, fitted preprocessor, variance filter, selected
feature indices, LightGBM model, isotonic calibrator and decision threshold. There are
no standalone scaler/imputer artifacts. Anything that changes preprocessing, feature
formulas or column order requires a **retrain**, not an edit on the serving side.
Legacy `models/best_model_lgbm.pkl`, `models/imputer.joblib` and `models/scaler.joblib`
are dead files from the retired pipeline and are read by nothing.

`src/inference.py:23` validates a required-key set on load, but that set omits
`selected_feature_names`, `metrics` and `risk_thresholds`, which `inspect_model.py` and
the app read directly. Add keys there when extending the bundle.

### Leakage discipline (the core design constraint)

Split happens before anything that learns from data. `train_test_split` at 70/30, then
the 30% is halved into calibration and validation, all stratified with `random_state=42`.

| Split | Share | Only role |
|---|---|---|
| train | 70% | fit defaults, preprocessor, variance filter, selector, LightGBM |
| calibration | 15% | fit isotonic calibrator, pick F1-optimal threshold |
| validation | 15% | reported metrics, touched exactly once |

The notebooks in `notebooks/` applied SMOTE before splitting; their metrics are inflated
and must not be quoted. The current pipeline uses no resampling and keeps the natural
8.07% positive rate. Do not reintroduce SMOTE, and do not fit any transformer on the
full frame.

### Shared feature formulas prevent training/serving skew

`src/feature_generation.py` is stateless and imported by both `model_training.py` and
`inference.py`. It is the only place domain features are defined. Every function guards
against a zero denominator by producing NaN, never infinity, and the module returns
`replace([inf, -inf], nan)` as its last step. `check_data_quality.py` enforces the
no-infinity invariant on the processed data.

Each `src/` module uses a `try: from .x import y / except ImportError: from x import y`
dual-import so it works both as `python src/foo.py` and as `from src.foo import …`.
Keep that pattern when adding modules.

### Feature selection is form-aware

`REQUIRED_APP_FEATURES` in `src/model_training.py:39` pins the 63 post-transform feature
names the Streamlit form controls. `choose_feature_indices` keeps all of them, then fills
to `--max-features` (200) by ANOVA F-score. Names are post-`ColumnTransformer` names, so
one-hot entries appear as `CODE_GENDER_F`, `NAME_EDUCATION_TYPE_Higher education`, etc.
A name that no longer matches is skipped **silently**, dropping the guarantee that the
user's input reaches the model. After changing form fields, categories or the
preprocessor, verify coverage rather than assuming it:

```powershell
python -c "import sys; sys.path.insert(0,'.'); from src.inference import load_bundle; from src.model_training import REQUIRED_APP_FEATURES as R; s=set(load_bundle()['selected_feature_names']); print(sorted(R-s))"
```

### Serving path

`app/streamlit_app.py` inserts the repo root on `sys.path`, then calls only the four
public functions in `src/inference.py`. The form collects 21 Home Credit fields;
`build_input_frame` overlays them onto the 493-column training-default row. Keys not in
`input_columns` are dropped without warning (`src/inference.py:44`), so a renamed column
degrades to a default instead of raising.

Consequence to keep in mind when reasoning about predictions: only 63 of the model's 200
features respond to the form. The other 137, holding roughly 52% of LightGBM split
importance, are pinned to training medians for every applicant. That includes all 57
missing-indicator features, which are therefore always 0 at serve time even though the
underlying bureau and credit-card columns were absent for ~62% of training rows.

`probability_to_score` is a presentation-only odds mapping (600 points at 50:1 odds, 20
points per doubling), not a bureau score. Risk bands come from the bundle's
`risk_thresholds`, not from hardcoded cutoffs.

## Data and git

`.gitignore` excludes `*.csv`, `models/*.joblib` and `models/*.json` repo-wide, so no
data or model artifact is version-controlled. A fresh clone therefore cannot run the app
until someone supplies the raw Kaggle CSVs in `data/raw/` and retrains. `data/features/`
holds retired SMOTE-era CSVs that nothing reads.

Untracked scratch that is **not** covered by `.gitignore` and would be committed by a
`git add -A`: `.codex_tmp/`, `output/`, and a stray zero-byte file named
`x.includes('Main')`. `src/inference.py`, `PROJECT_WORKFLOW.md` and all three current
test files are untracked, while the committed `tests/test_scaling.py` tests a retired
`+1`-denominator API and fails against current code. The committed HEAD does not run.
