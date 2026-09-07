import numpy as np

from src.inference import build_input_frame, probability_to_score, risk_level


def test_build_input_frame_uses_defaults_and_known_overrides_only():
    bundle = {
        "input_columns": ["AMT_CREDIT", "CODE_GENDER"],
        "input_defaults": {"AMT_CREDIT": 100.0, "CODE_GENDER": "F"},
    }
    frame = build_input_frame({"AMT_CREDIT": 250.0, "UNKNOWN": 1}, bundle)
    assert frame.to_dict("records") == [{"AMT_CREDIT": 250.0, "CODE_GENDER": "F"}]


def test_probability_to_score_is_monotonic_and_bounded():
    probabilities = [0.001, 0.01, 0.1, 0.5, 0.999]
    scores = [probability_to_score(value) for value in probabilities]
    assert scores == sorted(scores, reverse=True)
    assert all(300 <= score <= 850 for score in scores)


def test_risk_level_uses_bundle_thresholds():
    bundle = {"decision_threshold": 0.2, "risk_thresholds": {"low": 0.1, "high": 0.2}}
    assert risk_level(0.05, bundle) == "low"
    assert risk_level(0.15, bundle) == "medium"
    assert risk_level(0.25, bundle) == "high"
