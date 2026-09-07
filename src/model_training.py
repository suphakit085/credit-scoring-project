"""Leakage-safe training entry point for the credit scoring model."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

try:
    from .feature_generation import generate_features
except ImportError:  # Allows `python src/model_training.py`.
    from feature_generation import generate_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_VERSION = 2

REQUIRED_APP_FEATURES = {
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "CNT_CHILDREN", "CNT_FAM_MEMBERS", "DAYS_BIRTH", "DAYS_EMPLOYED",
    "FLAG_WORK_PHONE", "CREDIT_TO_ANNUITY_RATIO", "CREDIT_TO_GOODS_RATIO",
    "ANNUITY_TO_INCOME_RATIO", "CREDIT_TO_INCOME_RATIO", "INCOME_PER_PERSON",
    "AGE_YEARS", "EMPLOYMENT_YEARS", "EMPLOYMENT_TO_AGE_RATIO",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "EXT_SOURCE_MEAN",
    "EXT_SOURCE_STD", "EXT_SOURCE_MIN", "EXT_SOURCE_MAX",
    "CODE_GENDER_F", "CODE_GENDER_M",
    "FLAG_OWN_CAR_N", "FLAG_OWN_CAR_Y", "FLAG_OWN_REALTY_N", "FLAG_OWN_REALTY_Y",
    "NAME_EDUCATION_TYPE_Secondary / secondary special",
    "NAME_EDUCATION_TYPE_Higher education", "NAME_EDUCATION_TYPE_Incomplete higher",
    "NAME_EDUCATION_TYPE_Lower secondary", "NAME_EDUCATION_TYPE_Academic degree",
    "NAME_FAMILY_STATUS_Married", "NAME_FAMILY_STATUS_Single / not married",
    "NAME_FAMILY_STATUS_Civil marriage", "NAME_FAMILY_STATUS_Separated",
    "NAME_FAMILY_STATUS_Widow", "NAME_HOUSING_TYPE_House / apartment",
    "NAME_HOUSING_TYPE_With parents", "NAME_HOUSING_TYPE_Municipal apartment",
    "NAME_HOUSING_TYPE_Rented apartment", "NAME_HOUSING_TYPE_Office apartment",
    "NAME_HOUSING_TYPE_Co-op apartment", "NAME_INCOME_TYPE_Working",
    "NAME_INCOME_TYPE_Commercial associate", "NAME_INCOME_TYPE_State servant",
    "NAME_INCOME_TYPE_Pensioner", "NAME_INCOME_TYPE_Unemployed",
    "OCCUPATION_TYPE_Laborers", "OCCUPATION_TYPE_Core staff",
    "OCCUPATION_TYPE_Sales staff", "OCCUPATION_TYPE_Managers",
    "OCCUPATION_TYPE_Drivers", "OCCUPATION_TYPE_Accountants", "OCCUPATION_TYPE_IT staff",
    "ORGANIZATION_TYPE_Business Entity Type 3", "ORGANIZATION_TYPE_Self-employed",
    "ORGANIZATION_TYPE_Government", "ORGANIZATION_TYPE_School", "ORGANIZATION_TYPE_XNA",
}


def _json_value(value):
    if pd.isna(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def make_input_defaults(df: pd.DataFrame) -> dict[str, object]:
    """Learn deployment defaults from training rows only."""
    defaults: dict[str, object] = {}
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            defaults[column] = _json_value(series.median())
        else:
            mode = series.dropna().mode()
            defaults[column] = _json_value(mode.iloc[0]) if not mode.empty else "Unknown"
    return defaults


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    numeric = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical = [c for c in df.columns if c not in numeric]
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", RobustScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)],
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )


def optimal_f1_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if thresholds.size == 0:
        return 0.5
    scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(scores))])


def evaluate(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }


def choose_feature_indices(matrix, y: pd.Series, names: np.ndarray, limit: int) -> np.ndarray:
    """Keep every form-controlled feature, then fill remaining slots by ANOVA score."""
    required = [
        index
        for index, name in enumerate(names)
        if name in REQUIRED_APP_FEATURES
    ]
    if len(required) > limit:
        raise ValueError(f"max_features={limit} is below {len(required)} required app features")
    scores, _ = f_classif(matrix, y)
    scores = np.nan_to_num(scores, nan=-np.inf)
    ranked = np.argsort(scores)[::-1].tolist()
    required_set = set(required)
    selected = required + [index for index in ranked if index not in required_set]
    return np.asarray(selected[: min(limit, matrix.shape[1])], dtype=int)


def train_model(
    data_path: Path,
    output_path: Path,
    max_features: int = 200,
    sample_rows: int | None = None,
) -> dict:
    """Train on untouched class balance and evaluate on an isolated holdout."""
    df = pd.read_csv(data_path, nrows=sample_rows, low_memory=False)
    if "TARGET" not in df:
        raise ValueError(f"TARGET column is missing from {data_path}")

    y = df.pop("TARGET").astype(int)
    raw_x = df.drop(columns=["SK_ID_CURR"], errors="ignore")
    train_x, holdout_x, train_y, holdout_y = train_test_split(
        raw_x, y, test_size=0.30, random_state=42, stratify=y
    )
    calibration_x, validation_x, calibration_y, validation_y = train_test_split(
        holdout_x, holdout_y, test_size=0.50, random_state=42, stratify=holdout_y
    )

    defaults = make_input_defaults(train_x)
    train_features = generate_features(train_x)
    calibration_features = generate_features(calibration_x)
    validation_features = generate_features(validation_x)

    preprocessor = build_preprocessor(train_features)
    train_matrix = preprocessor.fit_transform(train_features)
    calibration_matrix = preprocessor.transform(calibration_features)
    validation_matrix = preprocessor.transform(validation_features)

    variance_filter = VarianceThreshold()
    train_nonconstant = variance_filter.fit_transform(train_matrix)
    calibration_nonconstant = variance_filter.transform(calibration_matrix)
    validation_nonconstant = variance_filter.transform(validation_matrix)

    transformed_names = np.asarray(preprocessor.get_feature_names_out(), dtype=object)
    nonconstant_names = transformed_names[variance_filter.get_support()]
    selected_indices = choose_feature_indices(
        train_nonconstant, train_y, nonconstant_names, max_features
    )
    train_selected = train_nonconstant[:, selected_indices]
    calibration_selected = calibration_nonconstant[:, selected_indices]
    validation_selected = validation_nonconstant[:, selected_indices]

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=0.2,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(train_selected, train_y)

    calibration_raw = model.predict_proba(calibration_selected)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(calibration_raw, calibration_y)
    calibration_probs = calibrator.predict(calibration_raw)
    threshold = optimal_f1_threshold(calibration_y, calibration_probs)

    validation_raw = model.predict_proba(validation_selected)[:, 1]
    validation_probs = calibrator.predict(validation_raw)
    metrics = evaluate(validation_y, validation_probs, threshold)

    selected_names = nonconstant_names[selected_indices].tolist()
    importance = pd.DataFrame(
        {"feature": selected_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    bundle = {
        "bundle_version": BUNDLE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_columns": raw_x.columns.tolist(),
        "input_defaults": defaults,
        "preprocessor": preprocessor,
        "variance_filter": variance_filter,
        "selected_indices": selected_indices,
        "model": model,
        "calibrator": calibrator,
        "decision_threshold": threshold,
        "risk_thresholds": {"low": threshold * 0.5, "high": threshold},
        "selected_feature_names": selected_names,
        "metrics": metrics,
        "training": {
            "data_path": str(data_path),
            "rows": int(len(df)),
            "train_rows": int(len(train_x)),
            "calibration_rows": int(len(calibration_x)),
            "validation_rows": int(len(validation_x)),
            "positive_rate": float(y.mean()),
            "smote_used": False,
            "random_state": 42,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)
    importance.to_csv(output_path.with_name("feature_importance.csv"), index=False)
    output_path.with_name("model_metrics.json").write_text(
        json.dumps({"metrics": metrics, "training": bundle["training"]}, indent=2),
        encoding="utf-8",
    )
    return bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "train_cleaned.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "models" / "credit_scoring_bundle.joblib",
    )
    parser.add_argument("--max-features", type=int, default=200)
    parser.add_argument("--sample-rows", type=int, default=None, help="Development smoke test only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = train_model(args.data, args.output, args.max_features, args.sample_rows)
    print(json.dumps(bundle["metrics"], indent=2))
    print(f"Saved model bundle to {args.output}")


if __name__ == "__main__":
    main()
