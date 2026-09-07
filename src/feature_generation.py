"""Deterministic domain features shared by training and inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# Which raw columns each derived feature consumes. Kept beside the formulas so
# the two stay in step. Training uses it to prove that every field the UI
# collects actually reaches the model, directly or through a derived feature.
# Prefix entries mean "any raw column starting with this".
DERIVED_INPUTS: dict[str, set[str]] = {
    "CREDIT_TO_ANNUITY_RATIO": {"AMT_CREDIT", "AMT_ANNUITY"},
    "CREDIT_TO_GOODS_RATIO": {"AMT_CREDIT", "AMT_GOODS_PRICE"},
    "ANNUITY_TO_INCOME_RATIO": {"AMT_ANNUITY", "AMT_INCOME_TOTAL"},
    "CREDIT_TO_INCOME_RATIO": {"AMT_CREDIT", "AMT_INCOME_TOTAL"},
    "INCOME_PER_PERSON": {"AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"},
    "AGE_YEARS": {"DAYS_BIRTH"},
    "EMPLOYMENT_YEARS": {"DAYS_EMPLOYED"},
    "REGISTRATION_YEARS": {"DAYS_REGISTRATION"},
    "ID_PUBLISH_YEARS": {"DAYS_ID_PUBLISH"},
    "EMPLOYMENT_TO_AGE_RATIO": {"DAYS_EMPLOYED", "DAYS_BIRTH"},
    "EXT_SOURCE_MEAN": {"EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"},
    "EXT_SOURCE_STD": {"EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"},
    "EXT_SOURCE_MIN": {"EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"},
    "EXT_SOURCE_MAX": {"EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"},
}

DERIVED_INPUT_PREFIXES: dict[str, str] = {
    "TOTAL_DOCUMENTS": "FLAG_DOCUMENT_",
    "BUREAU_AVG_CREDIT": "BUREAU_",
    "BUREAU_TOTAL_COUNT": "BUREAU_",
}


def derived_features_consuming(column: str) -> set[str]:
    """Derived feature names that read ``column``."""
    consumers = {name for name, inputs in DERIVED_INPUTS.items() if column in inputs}
    consumers |= {
        name for name, prefix in DERIVED_INPUT_PREFIXES.items() if column.startswith(prefix)
    }
    return consumers


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide while treating a zero denominator as missing, never as infinity."""
    denominator = denominator.replace(0, np.nan)
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain features without learning from the data."""
    result = df.copy()

    ratios = {
        "CREDIT_TO_ANNUITY_RATIO": ("AMT_CREDIT", "AMT_ANNUITY"),
        "CREDIT_TO_GOODS_RATIO": ("AMT_CREDIT", "AMT_GOODS_PRICE"),
        "ANNUITY_TO_INCOME_RATIO": ("AMT_ANNUITY", "AMT_INCOME_TOTAL"),
        "CREDIT_TO_INCOME_RATIO": ("AMT_CREDIT", "AMT_INCOME_TOTAL"),
        "INCOME_PER_PERSON": ("AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"),
    }
    for output, (numerator, denominator) in ratios.items():
        if numerator in result and denominator in result:
            result[output] = _safe_divide(result[numerator], result[denominator])

    day_features = {
        "AGE_YEARS": "DAYS_BIRTH",
        "EMPLOYMENT_YEARS": "DAYS_EMPLOYED",
        "REGISTRATION_YEARS": "DAYS_REGISTRATION",
        "ID_PUBLISH_YEARS": "DAYS_ID_PUBLISH",
    }
    for output, source in day_features.items():
        if source in result:
            result[output] = -result[source] / 365.25

    if {"EMPLOYMENT_YEARS", "AGE_YEARS"}.issubset(result.columns):
        result["EMPLOYMENT_TO_AGE_RATIO"] = _safe_divide(
            result["EMPLOYMENT_YEARS"], result["AGE_YEARS"]
        )

    ext_cols = [c for c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3") if c in result]
    if ext_cols:
        result["EXT_SOURCE_MEAN"] = result[ext_cols].mean(axis=1)
        result["EXT_SOURCE_STD"] = result[ext_cols].std(axis=1)
        result["EXT_SOURCE_MIN"] = result[ext_cols].min(axis=1)
        result["EXT_SOURCE_MAX"] = result[ext_cols].max(axis=1)

    doc_cols = [c for c in result if c.startswith("FLAG_DOCUMENT_")]
    if doc_cols:
        result["TOTAL_DOCUMENTS"] = result[doc_cols].sum(axis=1)

    bureau_credit_cols = [
        c for c in result if c.startswith("BUREAU_") and "AMT_CREDIT" in c and c.endswith("_mean")
    ]
    if bureau_credit_cols:
        result["BUREAU_AVG_CREDIT"] = result[bureau_credit_cols].mean(axis=1)

    bureau_count_cols = [c for c in result if c.startswith("BUREAU_") and c.endswith("_count")]
    if bureau_count_cols:
        result["BUREAU_TOTAL_COUNT"] = result[bureau_count_cols].sum(axis=1)

    return result.replace([np.inf, -np.inf], np.nan)


class DomainFeatureTransformer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible wrapper around :func:`generate_features`."""

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        return generate_features(X)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("input_features are required")
        empty = pd.DataFrame(columns=list(input_features))
        return np.asarray(generate_features(empty).columns, dtype=object)
