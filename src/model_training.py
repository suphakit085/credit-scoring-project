"""Leakage-safe training entry point for the credit scoring model.

The model is deliberately scoped to the columns the serving path can actually
supply. Training on features that are frozen at defaults during inference makes
the offline metric describe a system that is never deployed: measured on this
dataset, the 495-column model scored ROC-AUC 0.7756 offline but only 0.7612 as
the Streamlit app served it, while a model trained on the form fields alone
scored 0.7672. Keep `--feature-scope` aligned with what the UI collects.
"""

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
    average_precision_score,
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
    from .feature_generation import derived_features_consuming, generate_features
except ImportError:  # Allows `python src/model_training.py`.
    from feature_generation import derived_features_consuming, generate_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_VERSION = 4
RANDOM_STATE = 42

# Presentation scale for the 300-850 risk index. Anchored on the training
# portfolio's own base rate rather than a fixed 50:1, because a fixed anchor
# made the median applicant score 557 on a scale people read as FICO-like,
# where 557 means "bad credit" rather than "perfectly ordinary". PDO is the
# points needed to double the good:bad odds; 50 spreads this portfolio's
# realistic odds range across the scale, where the conventional 20 left 98%
# of applicants squeezed into 500-650.
SCORE_ANCHOR_POINTS = 600.0
SCORE_PDO = 50.0

# Raw columns the Streamlit form supplies for every applicant. Anything outside
# this list is unavailable at inference time and must not be relied on.
SERVED_COLUMNS = [
    "CODE_GENDER",
    "DAYS_BIRTH",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "NAME_INCOME_TYPE",
    "DAYS_EMPLOYED",
    "DAYS_EMPLOYED_ANOM",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "FLAG_WORK_PHONE",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
]

# Prefixes produced by data_preprocessing.py when aggregating the history tables.
AGGREGATE_PREFIXES = ("BUREAU_", "PREV_", "POS_", "CC_", "INSTALL_")

FEATURE_SCOPES = ("form", "application", "full")

INDICATOR_PREFIX = "missingindicator_"


def _json_value(value):
    if pd.isna(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def scope_columns(columns: list[str], scope: str) -> list[str]:
    """Restrict the raw column set to what the chosen serving mode can supply."""
    if scope not in FEATURE_SCOPES:
        raise ValueError(f"scope must be one of {FEATURE_SCOPES}, got {scope!r}")
    if scope == "form":
        selected = [c for c in SERVED_COLUMNS if c in columns]
        missing = [c for c in SERVED_COLUMNS if c not in columns]
        if missing:
            raise ValueError(f"Dataset is missing form-scope columns: {missing}")
        return selected
    if scope == "application":
        return [c for c in columns if not c.startswith(AGGREGATE_PREFIXES)]
    return list(columns)


def make_input_defaults(df: pd.DataFrame) -> dict[str, object]:
    """Learn deployment fallbacks from training rows only.

    These are used only for columns the serving path never supplies. Columns
    listed in ``served_columns`` fall back to NaN instead, so the imputer and
    its missing indicator see the same "absent" signal they were trained on.
    """
    defaults: dict[str, object] = {}
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            defaults[column] = _json_value(series.median())
        else:
            mode = series.dropna().mode()
            defaults[column] = _json_value(mode.iloc[0]) if not mode.empty else "Unknown"
    return defaults


def describe_served_inputs(
    df: pd.DataFrame, served: list[str]
) -> tuple[dict[str, list[str]], dict[str, float]]:
    """Record the choices and missing rates the UI must reproduce.

    The form used to hard-code a handful of categories per dropdown, covering
    only 59% of ORGANIZATION_TYPE and 54% of OCCUPATION_TYPE values actually
    present. An officer whose applicant fell outside that subset had to pick a
    wrong value. Shipping the real category list inside the bundle keeps the
    form and the encoder describing the same world.
    """
    categories: dict[str, list[str]] = {}
    missing_rates: dict[str, float] = {}
    for column in served:
        series = df[column]
        missing_rates[column] = float(series.isna().mean())
        if not pd.api.types.is_numeric_dtype(series):
            # Frequency order puts the common choices at the top of the dropdown.
            categories[column] = [str(v) for v in series.dropna().value_counts().index]
    return categories, missing_rates


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


def served_feature_names(served: list[str]) -> set[str]:
    """Names reachable from served columns, including derived domain features."""
    empty = pd.DataFrame(columns=list(served))
    return set(generate_features(empty).columns)


def is_served_feature(name: str, reachable: set[str]) -> bool:
    """True when a post-transform feature traces back to a served raw column."""
    base = name[len(INDICATOR_PREFIX) :] if name.startswith(INDICATOR_PREFIX) else name
    if base in reachable:
        return True
    # One-hot columns are emitted as "<column>_<category>"; categories may
    # themselves contain underscores, so match on the column prefix.
    return any(base.startswith(f"{column}_") for column in reachable)


def choose_feature_indices(
    matrix, y: pd.Series, names: np.ndarray, limit: int, served: list[str]
) -> np.ndarray:
    """Keep every served-derived feature, then fill remaining slots by ANOVA score."""
    reachable = served_feature_names(served)
    protected = [i for i, name in enumerate(names) if is_served_feature(str(name), reachable)]
    if len(protected) > limit:
        raise ValueError(
            f"max_features={limit} is below the {len(protected)} features derived from "
            f"served columns; raise --max-features or narrow --feature-scope"
        )
    scores, _ = f_classif(matrix, y)
    scores = np.nan_to_num(scores, nan=-np.inf)
    ranked = np.argsort(scores)[::-1].tolist()
    protected_set = set(protected)
    selected = protected + [index for index in ranked if index not in protected_set]
    return np.asarray(selected[: min(limit, matrix.shape[1])], dtype=int)


def verify_served_coverage(selected_names: list[str], served: list[str]) -> None:
    """Fail loudly when an input the officer fills in cannot reach the model.

    Selection used to skip unmatched names silently, so a renamed column or a
    dropped category quietly removed a user input from the model with no signal.
    A column counts as covered when it survives directly, as a missing
    indicator, as a one-hot column, or through any derived feature that reads it.
    """
    names = [str(name) for name in selected_names]
    orphaned = []
    for column in served:
        consumers = derived_features_consuming(column)
        covered = False
        for name in names:
            base = name[len(INDICATOR_PREFIX) :] if name.startswith(INDICATOR_PREFIX) else name
            if base == column or base.startswith(f"{column}_") or base in consumers:
                covered = True
                break
        if not covered:
            orphaned.append(column)
    if orphaned:
        raise ValueError(
            f"These served columns produced no selected feature: {orphaned}. "
            "The form would collect them and the model would ignore them."
        )


MIN_CALIBRATION_SUPPORT = 200


def calibrated_probability_bounds(
    calibrated: np.ndarray, y_true, min_support: int = MIN_CALIBRATION_SUPPORT
) -> tuple[float, float]:
    """The most extreme probabilities the calibration data actually supports.

    Isotonic regression's outermost bins rest on very few rows: here the 0% bin
    held 86 calibration rows and the 100% bin held 4. Serving those values
    states certainty that no sample of that size can justify.

    Picking a bound from the shape of the curve is not enough either. An
    earlier version used half the lowest non-zero level, giving a 0.27% floor
    and a 99.84% ceiling; the ceiling in particular was far beyond the
    evidence, since the 370 riskiest calibration rows defaulted at 49%, not 99%.

    So pool whole levels inward from each end until `min_support` rows are
    covered, and take the Jeffreys posterior mean of the pooled region. That
    ties the handful of applicants in the unsupported tails to the nearest
    value the data can defend, and nothing is served beyond it.
    """
    frame = pd.DataFrame({"level": np.asarray(calibrated), "y": np.asarray(y_true)})
    grouped = frame.groupby("level")["y"].agg(["size", "sum"]).sort_index()
    counts = grouped["size"].to_numpy()
    defaults = grouped["sum"].to_numpy()

    def pooled(sizes: np.ndarray, hits: np.ndarray) -> float:
        total = covered = 0
        for size, hit in zip(sizes, hits):
            total += int(size)
            covered += int(hit)
            if total >= min_support:
                break
        return (covered + 0.5) / (total + 1)

    low = pooled(counts, defaults)
    high = pooled(counts[::-1], defaults[::-1])
    if low >= high:  # degenerate calibration set; fall back to a usable span
        low, high = min(low, high), max(low, high)
    return float(low), float(high)


def make_score_scale(base_rate: float) -> dict[str, float]:
    """Odds-based scale anchored on the portfolio the model was trained for."""
    base_rate = float(np.clip(base_rate, 1e-6, 1 - 1e-6))
    return {
        "anchor_odds": (1.0 - base_rate) / base_rate,
        "anchor_points": SCORE_ANCHOR_POINTS,
        "pdo": SCORE_PDO,
        "anchor_base_rate": base_rate,
    }


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
        # Positive rate is ~8%, so average precision is the more honest headline.
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "positive_rate": float(np.mean(y_true)),
    }


def train_model(
    data_path: Path,
    output_path: Path,
    max_features: int = 200,
    sample_rows: int | None = None,
    feature_scope: str = "form",
) -> dict:
    """Train on untouched class balance and evaluate on an isolated holdout.

    Four splits, none reused: the calibrator is fitted on one, the decision
    threshold is chosen on a second, and the third is scored exactly once.
    Choosing the threshold on the calibrator's own split made the operating
    point optimistic, because isotonic regression is in-sample there.
    """
    df = pd.read_csv(data_path, nrows=sample_rows, low_memory=False)
    if "TARGET" not in df:
        raise ValueError(f"TARGET column is missing from {data_path}")

    y = df.pop("TARGET").astype(int)
    available = df.drop(columns=["SK_ID_CURR"], errors="ignore")
    columns = scope_columns(available.columns.tolist(), feature_scope)
    raw_x = available[columns]
    served = [c for c in SERVED_COLUMNS if c in columns]
    del df, available

    train_x, holdout_x, train_y, holdout_y = train_test_split(
        raw_x, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    calibration_x, rest_x, calibration_y, rest_y = train_test_split(
        holdout_x, holdout_y, test_size=2 / 3, random_state=RANDOM_STATE, stratify=holdout_y
    )
    threshold_x, validation_x, threshold_y, validation_y = train_test_split(
        rest_x, rest_y, test_size=0.50, random_state=RANDOM_STATE, stratify=rest_y
    )

    defaults = make_input_defaults(train_x)
    categories, missing_rates = describe_served_inputs(train_x, served)
    train_features = generate_features(train_x)
    calibration_features = generate_features(calibration_x)
    threshold_features = generate_features(threshold_x)
    validation_features = generate_features(validation_x)

    preprocessor = build_preprocessor(train_features)
    train_matrix = preprocessor.fit_transform(train_features)
    calibration_matrix = preprocessor.transform(calibration_features)
    threshold_matrix = preprocessor.transform(threshold_features)
    validation_matrix = preprocessor.transform(validation_features)

    variance_filter = VarianceThreshold()
    train_nonconstant = variance_filter.fit_transform(train_matrix)
    calibration_nonconstant = variance_filter.transform(calibration_matrix)
    threshold_nonconstant = variance_filter.transform(threshold_matrix)
    validation_nonconstant = variance_filter.transform(validation_matrix)

    transformed_names = np.asarray(preprocessor.get_feature_names_out(), dtype=object)
    nonconstant_names = transformed_names[variance_filter.get_support()]
    selected_indices = choose_feature_indices(
        train_nonconstant, train_y, nonconstant_names, max_features, served
    )
    selected_names = nonconstant_names[selected_indices].tolist()
    verify_served_coverage(selected_names, served)

    train_selected = train_nonconstant[:, selected_indices]
    calibration_selected = calibration_nonconstant[:, selected_indices]
    threshold_selected = threshold_nonconstant[:, selected_indices]
    validation_selected = validation_nonconstant[:, selected_indices]

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.85,
        # LightGBM ignores bagging_fraction unless bagging_freq >= 1, so without
        # this the subsample setting above silently did nothing.
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=0.2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(train_selected, train_y)

    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(model.predict_proba(calibration_selected)[:, 1], calibration_y)

    # Applied here as well as at inference so the reported metrics describe the
    # probabilities the app actually serves.
    calibration_probs = calibrator.predict(model.predict_proba(calibration_selected)[:, 1])
    low, high = calibrated_probability_bounds(calibration_probs, calibration_y)

    threshold_probs = np.clip(
        calibrator.predict(model.predict_proba(threshold_selected)[:, 1]), low, high
    )
    threshold = optimal_f1_threshold(threshold_y, threshold_probs)

    validation_probs = np.clip(
        calibrator.predict(model.predict_proba(validation_selected)[:, 1]), low, high
    )
    metrics = evaluate(validation_y, validation_probs, threshold)

    importance = pd.DataFrame(
        {"feature": selected_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    bundle = {
        "bundle_version": BUNDLE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_scope": feature_scope,
        "input_columns": raw_x.columns.tolist(),
        "served_columns": served,
        "input_defaults": defaults,
        "input_categories": categories,
        "input_missing_rates": missing_rates,
        "preprocessor": preprocessor,
        "variance_filter": variance_filter,
        "selected_indices": selected_indices,
        "model": model,
        "calibrator": calibrator,
        "probability_bounds": (low, high),
        "score_scale": make_score_scale(float(train_y.mean())),
        "decision_threshold": threshold,
        "risk_thresholds": {"low": threshold * 0.5, "high": threshold},
        "selected_feature_names": selected_names,
        "metrics": metrics,
        "training": {
            "data_path": str(data_path),
            "rows": int(len(raw_x)),
            "train_rows": int(len(train_x)),
            "calibration_rows": int(len(calibration_x)),
            "threshold_rows": int(len(threshold_x)),
            "validation_rows": int(len(validation_x)),
            "positive_rate": float(y.mean()),
            "smote_used": False,
            "random_state": RANDOM_STATE,
            "feature_scope": feature_scope,
            "n_input_columns": len(raw_x.columns),
            "n_selected_features": len(selected_names),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)
    # Report names follow the bundle stem. Fixed names let a --sample-rows smoke
    # test overwrite the reports belonging to the full production bundle.
    stem = output_path.stem
    importance.to_csv(output_path.with_name(f"{stem}_feature_importance.csv"), index=False)
    output_path.with_name(f"{stem}_metrics.json").write_text(
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
    parser.add_argument(
        "--feature-scope",
        choices=FEATURE_SCOPES,
        default="form",
        help=(
            "form: only columns the UI collects (default, matches what is served). "
            "application: the application table without history aggregates. "
            "full: every column, which leaves most features frozen at inference."
        ),
    )
    parser.add_argument("--sample-rows", type=int, default=None, help="Development smoke test only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = train_model(
        args.data, args.output, args.max_features, args.sample_rows, args.feature_scope
    )
    print(json.dumps(bundle["metrics"], indent=2))
    print(f"Scope: {bundle['feature_scope']} | features: {len(bundle['selected_feature_names'])}")
    print(f"Saved model bundle to {args.output}")


if __name__ == "__main__":
    main()
