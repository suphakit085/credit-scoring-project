# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Credit default risk scoring on the Home Credit Default Risk dataset. A leakage-safe
training pipeline produces a single self-contained model bundle, which a Streamlit app
consumes for single-applicant predictions. Documentation and UI text are in Thai;
code, identifiers and column names are English.

`PROJECT_WORKFLOW.md` is the detailed pipeline reference. `DATA_RECOVERY.md` records
what survives of the dataset and how to restore the rest.

## Commands

The virtualenv lives at `.venv/` (Python 3.12).

```powershell
python src/check_data_quality.py                      # exits 1 on any quality failure
python src/model_training.py                          # retrain (form scope, ~4 min)
python src/inspect_model.py                           # print bundle metadata + top features
python -m streamlit run app/streamlit_app.py
pytest                                                # 32 tests
pytest tests/test_serving_contract.py                 # one file
pytest tests/test_inference.py::test_risk_level_uses_bundle_thresholds
```

`pyproject.toml` sets `pythonpath = ["."]`, so bare `pytest` works. Do not reintroduce a
layout that requires `python -m pytest`.

`src/data_preprocessing.py` cannot run: `data/raw/` was lost. Everything downstream
still works because it reads `data/processed/train_cleaned.csv`, which survives. See
`DATA_RECOVERY.md` before suggesting anything that touches `data/`.

Smoke test on a subset. Report files are named from the bundle stem, so this no longer
overwrites the production reports:

```powershell
python src/model_training.py --sample-rows 20000 --output models/dev_bundle.joblib
```

## Architecture

### Scope the model to what can actually be served

`--feature-scope` (default `form`) decides which raw columns the model may learn from.
This is the project's central lesson, and it is measured, not assumed:

| Configuration | Validation ROC-AUC |
|---|---:|
| 495-column model, full data available | 0.7756 |
| same model, as the Streamlit app serves it | 0.7612 |
| model trained on the form fields only | 0.7672 |

The 495-column model relied on bureau and payment-history aggregates the UI never
supplies, so 137 of its 200 features sat frozen at training medians and it lost to a
much smaller honest model. Narrowing scope raised real-world accuracy. Keep the scope
aligned with what the UI collects; widening it is only correct once real data
integration exists.

### The bundle is the contract

`src/model_training.py` writes one `models/credit_scoring_bundle.joblib` (version 3)
holding the input schema, `served_columns`, training-set defaults, fitted preprocessor,
variance filter, selected feature indices, LightGBM model, isotonic calibrator and
thresholds. Anything that changes preprocessing, feature formulas or column order
requires a **retrain**, not an edit on the serving side.

`served_columns` is the list of raw columns the UI must supply. `src/inference.py`
treats it as the boundary between two fallback behaviours, and this distinction is the
fix for the project's worst defect:

- A **served** column the caller omits becomes `NaN`, so the imputer's missing indicator
  fires exactly as it did in training.
- A **non-served** column falls back to its training default.

Filling a served column with a median instead told the model "value present, equal to
average" for fields that were genuinely absent in most training rows (`EXT_SOURCE_1` is
missing 56% of the time). Unknown keys now raise instead of being dropped silently.

Legacy `models/best_model_lgbm.pkl`, `models/imputer.joblib` and `models/scaler.joblib`
are dead files from the retired pipeline. `models/*.v2_full.*` are the superseded
495-column bundle and its reports, kept only for comparison.

### Leakage discipline

Split happens before anything that learns from data, stratified with `random_state=42`.

| Split | Share | Only role |
|---|---|---|
| train | 70% | fit defaults, preprocessor, variance filter, selector, LightGBM |
| calibration | 10% | fit the isotonic calibrator |
| threshold | 10% | choose the F1-optimal decision threshold |
| validation | 10% | reported metrics, touched exactly once |

Calibration and threshold selection use separate splits on purpose. Choosing the
threshold on the calibrator's own split made the operating point optimistic, because
isotonic regression is in-sample there.

The notebooks in `notebooks/` applied SMOTE before splitting; their metrics are inflated
and must not be quoted. Do not reintroduce SMOTE, and do not fit any transformer on the
full frame. Report `average_precision` alongside ROC-AUC: the positive rate is 8%.

### Shared feature formulas prevent training/serving skew

`src/feature_generation.py` is stateless and imported by both training and inference. It
is the only place domain features are defined. `DERIVED_INPUTS` and
`DERIVED_INPUT_PREFIXES` there map each derived feature to the raw columns it reads;
keep them in step when adding a formula, because training uses them to prove every form
field reaches the model.

Every formula turns a zero denominator into NaN, never infinity, and the module ends
with `replace([inf, -inf], nan)`. `check_data_quality.py` enforces that invariant.

Each `src/` module uses a `try: from .x import y / except ImportError: from x import y`
dual-import so it works both as `python src/foo.py` and as `from src.foo import …`.

### Fail loudly at the boundaries

Three silent failure modes were removed; do not reintroduce them.

- `verify_served_coverage` raises when a form field survives into no selected feature.
  Selection used to skip unmatched names quietly, so a renamed column could remove a
  user input from the model with no signal.
- `build_input_frame` raises on keys outside the schema.
- `app/streamlit_app.py` compares its payload keys against `served_columns` before
  scoring and stops with a diff if they disagree.

### Serving path

The form collects 22 fields matching `served_columns` exactly. Two mappings encode real
properties of the training data and must stay in sync with it:

- `NAME_INCOME_TYPE` of `Pensioner` or `Unemployed` sends `DAYS_EMPLOYED = NaN` and
  `DAYS_EMPLOYED_ANOM = 1`. In `train_cleaned.csv` those two types carry the 365243
  sentinel in 100% of rows and no other type ever does.
- Each `EXT_SOURCE` field is entered by a loan officer from the external credit system,
  with a "not found" checkbox that sends NaN. They are not applicant self-estimates.
  These three fields dominate the prediction (0.006 to 0.45 across their range, versus
  under 0.01 for annual income from 30,000 to 5,000,000), so inventing a value for them
  effectively invents the answer.

`probability_to_score` is a presentation-only odds mapping (600 points at 50:1 odds, 20
points per doubling), not a bureau score. Risk bands come from the bundle's
`risk_thresholds`.

## Data and git

`.gitignore` excludes `*.csv`, `models/*.joblib` and generated model reports, so no data
or model artifact is version-controlled. `data/features/` holds retired SMOTE-era CSVs
that nothing reads.
