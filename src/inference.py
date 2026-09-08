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
    # Isotonic calibration emits exactly 0 and exactly 1 for its outermost
    # bins. Serving those would tell an officer an applicant cannot default,
    # or certainly will, on evidence that supports neither.
    low, high = bundle.get("probability_bounds", (0.0, 1.0))
    return float(np.clip(probability, low, high))


# Used only for bundles predating the stored scale (version 3 and earlier).
LEGACY_SCORE_SCALE = {"anchor_odds": 50.0, "anchor_points": 600.0, "pdo": 20.0}


def probability_to_score(probability: float, bundle: dict | None = None) -> int:
    """Convert calibrated PD to an odds-based 300–850 index.

    A presentation index, not a consumer bureau score. The scale lives in the
    bundle so it stays fixed for a given model: `anchor_points` is the score at
    `anchor_odds` good:bad, and `pdo` is the points that double the odds.
    """
    scale = (bundle or {}).get("score_scale") or LEGACY_SCORE_SCALE
    probability = float(np.clip(probability, 1e-6, 1 - 1e-6))
    odds = (1.0 - probability) / probability
    factor = float(scale["pdo"]) / np.log(2.0)
    score = float(scale["anchor_points"]) + factor * np.log(odds / float(scale["anchor_odds"]))
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
