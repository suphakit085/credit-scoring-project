"""Stable inference API used by the Streamlit application and tests."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    from .feature_generation import generate_features
except ImportError:
    from feature_generation import generate_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_PATH = PROJECT_ROOT / "models" / "credit_scoring_bundle.joblib"

REQUIRED_BUNDLE_KEYS = {
    "bundle_version",
    "input_columns",
    "input_defaults",
    "preprocessor",
    "variance_filter",
    "selected_indices",
    "model",
    "calibrator",
    "decision_threshold",
    # Read directly by the app and by inspect_model.py, so validate them here
    # rather than failing with a KeyError far from the load site.
    "selected_feature_names",
    "risk_thresholds",
    "metrics",
}


def load_bundle(path: Path = DEFAULT_BUNDLE_PATH) -> dict:
    bundle = joblib.load(path)
    missing = REQUIRED_BUNDLE_KEYS.difference(bundle)
    if missing:
        raise ValueError(f"Invalid model bundle; missing keys: {sorted(missing)}")
    return bundle


def build_input_frame(
    values: dict[str, object], bundle: dict, strict: bool = True
) -> pd.DataFrame:
    """Build one scoring row from caller-supplied values.

    A column the serving path is expected to supply falls back to NaN, not to a
    training median. Filling it with a median tells the imputer's missing
    indicator that the value was present, a state the model rarely saw during
    training whenever the underlying field is often absent. Columns outside
    ``served_columns`` keep the training-set default, which is the documented
    behaviour for anything the UI cannot collect.
    """
    columns = bundle["input_columns"]
    served = set(bundle.get("served_columns", ()))
    defaults = bundle["input_defaults"]

    unknown = sorted(set(values).difference(columns))
    if unknown:
        if strict:
            raise KeyError(
                f"Unknown input columns for this bundle: {unknown}. "
                "Silently dropping them would score the applicant on defaults."
            )
        values = {key: value for key, value in values.items() if key in set(columns)}

    row: dict[str, object] = {}
    for column in columns:
        if column in values:
            row[column] = values[column]
        elif column in served:
            row[column] = np.nan
        else:
            row[column] = defaults.get(column)
    return pd.DataFrame([row], columns=columns)


def predict_default_probability(
    values: dict[str, object], bundle: dict, strict: bool = True
) -> float:
    frame = build_input_frame(values, bundle, strict=strict)
    features = generate_features(frame)
    matrix = bundle["preprocessor"].transform(features)
    nonconstant = bundle["variance_filter"].transform(matrix)
    selected = nonconstant[:, bundle["selected_indices"]]
    raw_probability = bundle["model"].predict_proba(selected)[:, 1]
    probability = float(bundle["calibrator"].predict(raw_probability)[0])
    return float(np.clip(probability, 0.0, 1.0))


def probability_to_score(probability: float) -> int:
    """Convert calibrated PD to a conventional odds-based 300–850 index.

    This is a presentation index, not a consumer bureau score. The mapping uses
    score 600 at 50:1 good-to-bad odds and 20 points to double the odds.
    """
    probability = float(np.clip(probability, 1e-6, 1 - 1e-6))
    odds = (1.0 - probability) / probability
    factor = 20.0 / np.log(2.0)
    score = 600.0 + factor * np.log(odds / 50.0)
    return int(np.clip(round(score), 300, 850))


def risk_level(probability: float, bundle: dict) -> str:
    thresholds = bundle.get(
        "risk_thresholds",
        {"low": bundle["decision_threshold"] * 0.5, "high": bundle["decision_threshold"]},
    )
    if probability < thresholds["low"]:
        return "low"
    if probability < thresholds["high"]:
        return "medium"
    return "high"
