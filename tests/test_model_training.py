import numpy as np
import pandas as pd
import pytest

from src.model_training import (
    PROTECTED_ATTRIBUTES,
    SERVED_COLUMNS,
    calibrated_probability_bounds,
    cost_ratio_threshold,
    evaluate,
    is_served_feature,
    make_risk_thresholds,
    make_score_scale,
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


def test_full_scope_keeps_everything_except_protected_attributes():
    columns = ["AMT_CREDIT", "BUREAU_X_mean"]
    assert scope_columns(columns, "full") == columns


@pytest.mark.parametrize("scope", ["application", "full"])
def test_wider_scopes_cannot_reintroduce_protected_attributes(scope):
    columns = ["AMT_CREDIT", *PROTECTED_ATTRIBUTES, "DAYS_BIRTH"]
    selected = scope_columns(columns, scope)
    assert not set(PROTECTED_ATTRIBUTES) & set(selected)
    assert {"AMT_CREDIT", "DAYS_BIRTH"} <= set(selected)


def test_form_scope_excludes_protected_attributes():
    assert not set(PROTECTED_ATTRIBUTES) & set(SERVED_COLUMNS)


def test_unknown_scope_is_rejected():
    with pytest.raises(ValueError, match="scope must be one of"):
        scope_columns(["AMT_CREDIT"], "everything")


def test_served_feature_names_include_derived_domain_features():
    reachable = served_feature_names(["AMT_CREDIT", "AMT_ANNUITY"])
    assert "CREDIT_TO_ANNUITY_RATIO" in reachable
    assert "AMT_CREDIT" in reachable


def test_is_served_feature_matches_indicators_and_one_hot_columns():
    reachable = served_feature_names(["AMT_CREDIT", "FLAG_OWN_CAR"])
    assert is_served_feature("AMT_CREDIT", reachable)
    assert is_served_feature("missingindicator_AMT_CREDIT", reachable)
    assert is_served_feature("FLAG_OWN_CAR_Y", reachable)
    assert not is_served_feature("BUREAU_AMT_CREDIT_mean", reachable)


def test_verify_served_coverage_accepts_a_column_reached_only_via_derived_feature():
    """DAYS_BIRTH may be dropped as long as AGE_YEARS survives selection."""
    verify_served_coverage(["AGE_YEARS", "AMT_CREDIT"], ["DAYS_BIRTH", "AMT_CREDIT"])


def test_verify_served_coverage_raises_when_a_form_field_reaches_nothing():
    with pytest.raises(ValueError, match="AMT_ANNUITY"):
        verify_served_coverage(["AGE_YEARS"], ["DAYS_BIRTH", "AMT_ANNUITY"])


@pytest.mark.parametrize("ratio, expected", [(1, 0.5), (3, 0.25), (5, 1 / 6), (9, 0.1)])
def test_cost_ratio_threshold_is_the_expected_cost_break_even(ratio, expected):
    assert cost_ratio_threshold(ratio) == pytest.approx(expected)


def test_cost_ratio_threshold_rejects_a_non_positive_ratio():
    with pytest.raises(ValueError):
        cost_ratio_threshold(0)


def test_flagging_at_the_cost_threshold_minimises_expected_cost():
    """At p = threshold, approving and rejecting cost the same."""
    ratio = 5.0
    p = cost_ratio_threshold(ratio)
    cost_of_approving = p * ratio
    cost_of_rejecting = 1 - p
    assert cost_of_approving == pytest.approx(cost_of_rejecting)


def test_low_band_is_the_portfolio_base_rate():
    bands = make_risk_thresholds(base_rate=0.08, decision_threshold=1 / 6)
    assert bands == {"low": pytest.approx(0.08), "high": pytest.approx(1 / 6)}


def test_low_band_stays_below_high_when_the_threshold_is_cautious():
    bands = make_risk_thresholds(base_rate=0.08, decision_threshold=0.05)
    assert bands["low"] < bands["high"]


def degenerate_calibration(n_tail: int = 5, n_body: int = 400):
    """Calibrated output whose end levels are all-clean and all-default.

    The tails are deliberately tiny, mirroring the real calibrator where the
    0% level rested on 86 rows and the 100% level on 4.
    """
    levels = np.concatenate([
        np.zeros(n_tail),
        np.full(n_body, 0.20),
        np.full(n_body, 0.60),
        np.ones(n_tail),
    ])
    y = np.concatenate([
        np.zeros(n_tail),
        (np.arange(n_body) < n_body * 0.20).astype(int),
        (np.arange(n_body) < n_body * 0.60).astype(int),
        np.ones(n_tail),
    ])
    return levels, y


def test_probability_bounds_exclude_certainty():
    levels, y = degenerate_calibration()
    assert levels.min() == 0.0 and levels.max() == 1.0, "expected degenerate tails"

    low, high = calibrated_probability_bounds(levels, y)
    assert 0.0 < low < high < 1.0


def test_probability_bounds_ignore_a_tail_too_small_to_support_itself():
    """A 5-row tail must not set the bound; the pooled level should.

    The all-clean tail has 5 rows, so the bound has to pool into the 20% level
    beside it rather than trusting five observations.
    """
    levels, y = degenerate_calibration(n_tail=5, n_body=400)
    low, high = calibrated_probability_bounds(levels, y, min_support=200)
    assert low > 0.10, "floor should be pulled up toward the 20% level it pooled with"
    assert high < 0.90, "ceiling should be pulled down toward the 60% level"


def test_probability_bounds_respect_min_support():
    """A larger support requirement pools in further, giving tighter bounds.

    Pooling works level by level, so this needs a small level between the tail
    and the body for the two support settings to stop at different places.
    """
    levels = np.concatenate([
        np.zeros(5), np.full(50, 0.05), np.full(400, 0.30),
        np.full(400, 0.70), np.full(50, 0.95), np.ones(5),
    ])
    y = np.concatenate([
        np.zeros(5),
        (np.arange(50) < 50 * 0.05).astype(int),
        (np.arange(400) < 400 * 0.30).astype(int),
        (np.arange(400) < 400 * 0.70).astype(int),
        (np.arange(50) < 50 * 0.95).astype(int),
        np.ones(5),
    ])
    loose_low, loose_high = calibrated_probability_bounds(levels, y, min_support=10)
    tight_low, tight_high = calibrated_probability_bounds(levels, y, min_support=300)
    assert tight_low > loose_low
    assert tight_high < loose_high


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
