import numpy as np
import pytest

from src.inference import build_input_frame, probability_to_score, risk_level


def bundle_stub():
    return {
        "input_columns": ["AMT_CREDIT", "CODE_GENDER", "BUREAU_AMT_CREDIT_mean"],
        "served_columns": ["AMT_CREDIT", "CODE_GENDER"],
        "input_defaults": {
            "AMT_CREDIT": 100.0,
            "CODE_GENDER": "F",
            "BUREAU_AMT_CREDIT_mean": 42.0,
        },
    }


def test_build_input_frame_uses_supplied_values():
    frame = build_input_frame({"AMT_CREDIT": 250.0, "CODE_GENDER": "M"}, bundle_stub())
    assert frame.loc[0, "AMT_CREDIT"] == 250.0
    assert frame.loc[0, "CODE_GENDER"] == "M"


def test_unsupplied_served_column_becomes_missing_not_median():
    """A field the officer left blank must look absent, not average.

    Filling it with the training median sets the imputer's missing indicator to
    zero, telling the model the value was present.
    """
    frame = build_input_frame({"AMT_CREDIT": 250.0}, bundle_stub())
    assert frame.loc[0, "CODE_GENDER"] is np.nan or frame["CODE_GENDER"].isna().iloc[0]


def test_unserved_column_still_falls_back_to_training_default():
    frame = build_input_frame({"AMT_CREDIT": 250.0, "CODE_GENDER": "F"}, bundle_stub())
    assert frame.loc[0, "BUREAU_AMT_CREDIT_mean"] == 42.0


def test_unknown_key_raises_instead_of_being_dropped_silently():
    with pytest.raises(KeyError, match="UNKNOWN_COLUMN"):
        build_input_frame({"AMT_CREDIT": 1.0, "UNKNOWN_COLUMN": 9}, bundle_stub())


def test_unknown_key_can_be_dropped_when_strict_is_disabled():
    frame = build_input_frame(
        {"AMT_CREDIT": 1.0, "UNKNOWN_COLUMN": 9}, bundle_stub(), strict=False
    )
    assert list(frame.columns) == bundle_stub()["input_columns"]


def test_probability_to_score_is_monotonic_and_bounded():
    probabilities = [0.001, 0.01, 0.1, 0.5, 0.999]
    scores = [probability_to_score(value) for value in probabilities]
    assert scores == sorted(scores, reverse=True)
    assert all(300 <= score <= 850 for score in scores)


def test_score_scale_comes_from_the_bundle():
    """The scale is per-model, so a bundle must override the legacy default."""
    bundle = {"score_scale": {"anchor_odds": 11.39, "anchor_points": 600.0, "pdo": 50.0}}
    # At the anchor odds the score is exactly the anchor points.
    at_anchor = 1.0 / (1.0 + 11.39)
    assert probability_to_score(at_anchor, bundle) == 600
    # One doubling of the odds is worth exactly one PDO.
    doubled = 1.0 / (1.0 + 22.78)
    assert probability_to_score(doubled, bundle) == 650


def test_score_scale_falls_back_for_bundles_without_one():
    legacy_anchor = 1.0 / (1.0 + 50.0)
    assert probability_to_score(legacy_anchor) == 600
    assert probability_to_score(legacy_anchor, {}) == 600


def test_risk_level_uses_bundle_thresholds():
    bundle = {"decision_threshold": 0.2, "risk_thresholds": {"low": 0.1, "high": 0.2}}
    assert risk_level(0.05, bundle) == "low"
    assert risk_level(0.15, bundle) == "medium"
    assert risk_level(0.25, bundle) == "high"
