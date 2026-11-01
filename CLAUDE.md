# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Credit default risk scoring on the Home Credit Default Risk dataset. A leakage-safe
training pipeline produces a single self-contained model bundle, which a Streamlit app
consumes for single-applicant predictions by a loan officer. Documentation and UI text
are in Thai; code, identifiers and column names are English.

`PROJECT_WORKFLOW.md` is the detailed pipeline reference with every current number.
`DATA_RECOVERY.md` records what survives of the dataset and how to restore the rest.

## Commands

The virtualenv lives at `.venv/` (Python 3.12).

```powershell
python src/check_data_quality.py                      # exits 1 on any quality failure
python src/model_training.py                          # retrain, ~4 min, deterministic
python src/model_training.py --cost-ratio 8           # different decision threshold
python src/inspect_model.py                           # bundle metadata + top features
python -m streamlit run app/streamlit_app.py
pytest                                                # 65 tests
pytest tests/test_serving_contract.py                 # one file
pytest tests/test_inference.py::test_risk_level_uses_bundle_thresholds
```

`pyproject.toml` sets `pythonpath = ["."]`, so bare `pytest` works. Do not reintroduce a
layout that requires `python -m pytest`. `requirements.txt` is runtime only; the
historical notebooks need `requirements-notebooks.txt` as well.

`src/data_preprocessing.py` cannot run: `data/raw/` was lost. Everything downstream
still works because it reads `data/processed/train_cleaned.csv`, which survives. See
`DATA_RECOVERY.md` before suggesting anything that touches `data/`. `archive/` holds
retired SMOTE-era CSVs and pre-v4 model files; nothing reads it.

Smoke test on a subset; report files are named from the bundle stem, so this never
overwrites the production reports:

```powershell
python src/model_training.py --sample-rows 20000 --output models/dev_bundle.joblib
```

## Architecture

### Scope the model to what can actually be served

`--feature-scope` (default `form`) decides which raw columns the model may learn from.
This is the project's central lesson. The first model used 495 columns including
bureau and payment-history aggregates the UI never supplies, so 137 of its 200 features
sat frozen at training medians when served. Paired bootstrap on 23,091 rows neither
model was fitted or tuned on:

| Comparison | ROC-AUC difference | 95% interval |
|---|---:|---|
| current (form, 20 cols) − first model as served | +0.0042 | [+0.0008, +0.0075] |
| current − first model with full data | −0.0106 | [−0.0152, −0.0058] |

Narrowing scope wins where it matters and has a real cost against an unservable ideal.
Widening scope is only correct once real data integration exists.

### Protected attributes are excluded in every scope

`PROTECTED_ATTRIBUTES` (`CODE_GENDER`, `NAME_FAMILY_STATUS`) is stripped by
`scope_columns` for all three scopes, so widening scope cannot reintroduce them, and
`test_shipped_model_never_sees_protected_attributes` checks the bundle. Sex alone used
to move an identical applicant's PD by 1.45x. Dropping both costs 0.0017 AUC with a
paired interval spanning zero. Age is kept deliberately: dropping it costs a
significant 0.0043.

Removing the inputs did not remove the outcome gap. Men default at 10.3% versus 6.9%
for women (1.50x) but are flagged 15.0% versus 8.8% (1.70x), partly via correlated
features such as occupation. That is a policy question, documented in
`PROJECT_WORKFLOW.md` section 13, not something to "fix" with group-specific thresholds,
which are themselves unlawful in many places.

### The bundle is the contract

`src/model_training.py` writes one `models/credit_scoring_bundle.joblib` (version 5):
input schema, `served_columns`, training defaults, dropdown categories and missing
rates, fitted preprocessor, variance filter, selected indices, LightGBM, isotonic
calibrator, `probability_bounds`, `score_scale`, `cost_ratio`, `decision_threshold`,
`risk_thresholds` and metrics. Changing preprocessing, feature formulas or column
order requires a **retrain**, not a serving-side edit.

Training also writes `credit_scoring_bundle.joblib.sha256` (sha256sum format, LF line
ending). `load_bundle` verifies it **before** `joblib.load`, because unpickling runs
code. A missing sidecar loads with a warning; a mismatch refuses to load.

`served_columns` is the boundary between two fallbacks in `src/inference.py`, and the
distinction is the fix for the project's worst defect:

- A **served** column the caller omits becomes `NaN`, so the imputer's missing indicator
  fires exactly as in training. `EXT_SOURCE_1` is missing in 56% of training rows.
- A **non-served** column falls back to its training default.

Unknown keys raise instead of being dropped silently.

### Leakage discipline

Split happens before anything that learns from data, stratified, `random_state=42`.

| Split | Share | Rows | Only role |
|---|---|---:|---|
| train | 70% | 215,257 | defaults, preprocessor, variance filter, selector, LightGBM |
| calibration | 20% | 61,502 | isotonic calibrator and probability bounds |
| validation | 10% | 30,752 | reported metrics, touched exactly once |

The code still splits holdout in two steps and concatenates the two calibration parts.
That keeps the validation rows identical to every earlier measurement; do not simplify
it into one split without accepting that historical comparisons break.

The notebooks applied SMOTE before splitting; their metrics are inflated and must not
be quoted. Do not reintroduce SMOTE or fit any transformer on the full frame.

### Decisions come from a stated cost, not from F1

`cost_ratio_threshold` returns `1 / (1 + R)`, the expected-cost optimum for calibrated
probabilities, where R is the cost of approving a defaulter relative to rejecting a
good applicant. Default R = 5 gives 16.67%. The old F1-maximising search moved from
0.1665 to 0.1873 between two near-identical trainings and encoded no trade-off; do not
bring it back. Risk bands: low is below the portfolio base rate (8.07%), high is at or
above the threshold.

### Probability bounds come from evidence, not curve shape

Isotonic emits exactly 0 and 1 in its outermost bins, which rested on 86 and 4 rows
when calibration had 30,751 rows. `calibrated_probability_bounds` pools whole levels
inward until `MIN_CALIBRATION_SUPPORT` (200) rows are covered and takes the Jeffreys
posterior mean: currently **0.21% to 58.1%**. A version that used half the lowest
non-zero level produced a 99.84% ceiling while the 370 riskiest calibration rows
defaulted at 49%. Training applies the same clamp before computing metrics.

### The risk index is calibrated to this portfolio, not to FICO

`probability_to_score` is presentation only. `score_scale` in the bundle anchors 600
points at the training base rate (11.39:1 odds) and sets PDO. `make_score_scale` caps
PDO at 50 and lowers it until both probability bounds land inside 300-850, so nothing
clips; currently PDO 46 and a served range of 417 to 848. A fixed 50:1 anchor used to
put the median applicant at 557, which reads as "bad credit" on a FICO-like scale.

### Report numbers with their uncertainty

One validation split of 30,752 rows. Bootstrap 95% intervals: ROC-AUC 0.7647
[0.755, 0.774], average precision 0.242 [0.226, 0.259]. **Quote ROC-AUC to two
decimals, i.e. 0.76**; the interval is 0.019 wide. Compare models with a **paired** bootstrap on identical rows: the marginal
interval on one AUC is about ±0.009, the paired interval on a difference is far tighter.

Calibration passes: all 10 deciles inside the interval on the observed rate,
Hosmer-Lemeshow p = 0.43.

### Shared feature formulas prevent training/serving skew

`src/feature_generation.py` is stateless and imported by both training and inference,
and is the only place domain features are defined. `DERIVED_INPUTS` and
`DERIVED_INPUT_PREFIXES` map each derived feature to the raw columns it reads; keep them
in step when adding a formula, because training uses them to prove every form field
reaches the model. Zero denominators become NaN, never infinity.

Each `src/` module uses a `try: from .x import y / except ImportError: from x import y`
dual-import so it works both as `python src/foo.py` and as `from src.foo import …`.

### Fail loudly at the boundaries

Do not reintroduce any of these silent failure modes:

- `verify_served_coverage` raises when a form field survives into no selected feature.
- `build_input_frame` raises on keys outside the schema.
- The app compares its payload keys against `served_columns` before scoring.
- `load_bundle` refuses a bundle whose hash does not match its sidecar.

### Serving path

The form collects the 20 `served_columns`. Mappings that encode real properties of the
training data and must stay in sync with it:

- `NAME_INCOME_TYPE` of `Pensioner` or `Unemployed` sends `DAYS_EMPLOYED = NaN` and
  `DAYS_EMPLOYED_ANOM = 1`: those two types carry the 365243 sentinel in 100% of rows
  and no other type ever does.
- `CNT_FAM_MEMBERS` is adults plus children, both asked directly. It used to be
  inferred from marital status, which the model no longer sees.
- Each `EXT_SOURCE` field is entered by a loan officer from the external credit system,
  with a "not found" checkbox that sends NaN. These three dominate the prediction.

The sidebar is deliberately **not** an `st.form`: inside a form no widget reruns until
submit, so the "not found" checkbox could not disable its input. Each assessment
stores a fingerprint of its inputs, and the app warns when the form drifts from what
was scored.

Tests that prove an input reaches the model must compare the **raw** model output
(`raw_probability` in `tests/test_serving_contract.py`). Isotonic is a step function: a
pensioner and an identical employee scored 0.04625 and 0.04662 raw but 4.72% each after
calibration.

## Data and git

`.gitignore` excludes `*.csv`, `models/*.joblib`, generated reports, `*.sha256` and
`archive/`, so no data or model artifact is version-controlled.
