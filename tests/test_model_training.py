import numpy as np
import pandas as pd
import pytest

from sklearn.isotonic import IsotonicRegression

from src.model_training import (
    SERVED_COLUMNS,
    calibrated_probability_bounds,
    evaluate,
    is_served_feature,
    make_score_scale,
    optimal_f1_threshold,
    scope_columns,
    served_feature_names,
    verify_served_coverage,
)


def test_form_scope_keeps_only_columns_the_ui_supplies():
    columns = SERVED_COLUMNS + ["BUREAU_AMT_CREDIT_mean", "PREV_AMT_ANNUITY_max", "OWN_CAR_AGE"]
    assert scope_columns(columns, "form") == SERVED_COLUMNS


def test_form_scope_rejects_a_dataset_that_cannot_feed_the_form():
    columns = [c for c in SERVED_COLUMNS if c != "EXT_SOURCE_1"]
    with pytest.raises(ValueError, match="EXT_SOURCE_1"):
        scope_columns(columns, "form")


def test_application_scope_drops_history_aggregates_only():
    columns = ["AMT_CREDIT", "BUREAU_X_mean", "PREV_X_max", "POS_X_min", "CC_X_sum", "INSTALL_X_std"]
    assert scope_columns(columns, "application") == ["AMT_CREDIT"]


def test_full_scope_keeps_everything():
    columns = ["AMT_CREDIT", "BUREAU_X_mean"]
    assert scope_columns(columns, "full") == columns


def test_unknown_scope_is_rejected():
    with pytest.raises(ValueError, match="scope must be one of"):
        scope_columns(["AMT_CREDIT"], "everything")


def test_served_feature_names_include_derived_domain_features():
    reachable = served_feature_names(["AMT_CREDIT", "AMT_ANNUITY"])
    assert "CREDIT_TO_ANNUITY_RATIO" in reachable
    assert "AMT_CREDIT" in reachable


def test_is_served_feature_matches_indicators_and_one_hot_columns():
    reachable = served_feature_names(["AMT_CREDIT", "CODE_GENDER"])
    assert is_served_feature("AMT_CREDIT", reachable)
    assert is_served_feature("missingindicator_AMT_CREDIT", reachable)
    assert is_served_feature("CODE_GENDER_F", reachable)
    assert not is_served_feature("BUREAU_AMT_CREDIT_mean", reachable)


def test_verify_served_coverage_accepts_a_column_reached_only_via_derived_feature():
    """DAYS_BIRTH may be dropped as long as AGE_YEARS survives selection."""
    verify_served_coverage(["AGE_YEARS", "AMT_CREDIT"], ["DAYS_BIRTH", "AMT_CREDIT"])


def test_verify_served_coverage_raises_when_a_form_field_reaches_nothing():
    with pytest.raises(ValueError, match="AMT_ANNUITY"):
        verify_served_coverage(["AGE_YEARS"], ["DAYS_BIRTH", "AMT_ANNUITY"])


def test_optimal_f1_threshold_prefers_a_separating_cut():
    y = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])
    assert 0.2 < optimal_f1_threshold(y, probabilities) <= 0.8


def fitted_degenerate_calibrator():
    """Isotonic fitted on data whose end bins are all-clean and all-default."""
    raw = np.linspace(0.0, 1.0, 40)
    y = (raw > 0.5).astype(int)
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, y)


def test_probability_bounds_exclude_certainty():
    calibrator = fitted_degenerate_calibrator()
    grid = calibrator.predict(np.linspace(0, 1, 1001))
    assert grid.min() == 0.0 and grid.max() == 1.0, "expected a degenerate calibrator"

    low, high = calibrated_probability_bounds(calibrator)
    assert 0.0 < low < high < 1.0


def test_probability_bounds_stay_below_the_lowest_resolvable_level():
    """Clamping must not merge the outermost bin into its neighbour.

    Sitting strictly below the lowest non-zero level keeps applicants in the
    zero bin ranked ahead of the next bin, so ranking metrics are unchanged.
    """
    calibrator = fitted_degenerate_calibrator()
    grid = calibrator.predict(np.linspace(0, 1, 20001))
    low, high = calibrated_probability_bounds(calibrator)
    assert low < grid[grid > 0].min()
    assert high > grid[grid < 1].max()


def test_score_scale_is_anchored_on_the_training_base_rate():
    scale = make_score_scale(0.08)
    assert scale["anchor_odds"] == pytest.approx(0.92 / 0.08)
    assert scale["anchor_base_rate"] == pytest.approx(0.08)
    assert scale["pdo"] > 0


def test_evaluate_reports_average_precision_for_the_imbalanced_target():
    y = pd.Series([0, 0, 0, 1])
    probabilities = np.array([0.1, 0.2, 0.3, 0.9])
    metrics = evaluate(y, probabilities, 0.5)
    assert metrics["average_precision"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["positive_rate"] == pytest.approx(0.25)
