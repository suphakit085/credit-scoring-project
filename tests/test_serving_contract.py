"""End-to-end checks against the shipped bundle.

These guard the boundary that produced the worst defect in this project: the
model scoring an applicant on training defaults instead of on what the loan
officer actually entered.
"""

import numpy as np
import pytest

from src.inference import (
    DEFAULT_BUNDLE_PATH,
    build_input_frame,
    load_bundle,
    predict_default_probability,
    risk_level,
)

pytestmark = pytest.mark.skipif(
    not DEFAULT_BUNDLE_PATH.exists(),
    reason="model bundle not built; run python src/model_training.py",
)


@pytest.fixture(scope="module")
def bundle():
    return load_bundle()


def officer_payload(**overrides):
    """The payload app/streamlit_app.py sends for an employed applicant."""
    values = {
        "CODE_GENDER": "F",
        "DAYS_BIRTH": -35 * 365.25,
        "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_FAMILY_STATUS": "Married",
        "NAME_HOUSING_TYPE": "House / apartment",
        "CNT_CHILDREN": 0,
        "CNT_FAM_MEMBERS": 2,
        "AMT_INCOME_TOTAL": 150_000.0,
        "AMT_CREDIT": 300_000.0,
        "AMT_ANNUITY": 20_000.0,
        "AMT_GOODS_PRICE": 270_000.0,
        "NAME_INCOME_TYPE": "Working",
        "DAYS_EMPLOYED": -5 * 365.25,
        "DAYS_EMPLOYED_ANOM": 0,
        "OCCUPATION_TYPE": "Core staff",
        "ORGANIZATION_TYPE": "Government",
        "FLAG_OWN_CAR": "N",
        "FLAG_OWN_REALTY": "Y",
        "FLAG_WORK_PHONE": 1,
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_2": 0.5,
        "EXT_SOURCE_3": 0.5,
    }
    values.update(overrides)
    return values


def test_form_payload_matches_the_bundle_contract_exactly(bundle):
    assert set(officer_payload()) == set(bundle["served_columns"])


def test_every_served_column_is_also_a_model_input(bundle):
    assert set(bundle["served_columns"]).issubset(bundle["input_columns"])


def test_prediction_is_a_valid_probability(bundle):
    probability = predict_default_probability(officer_payload(), bundle)
    assert 0.0 <= probability <= 1.0
    assert risk_level(probability, bundle) in {"low", "medium", "high"}


def test_external_scores_move_the_prediction(bundle):
    weak = predict_default_probability(
        officer_payload(EXT_SOURCE_1=0.05, EXT_SOURCE_2=0.05, EXT_SOURCE_3=0.05), bundle
    )
    strong = predict_default_probability(
        officer_payload(EXT_SOURCE_1=0.95, EXT_SOURCE_2=0.95, EXT_SOURCE_3=0.95), bundle
    )
    assert weak > strong


def test_missing_external_score_is_accepted_and_changes_the_answer(bundle):
    """'ไม่พบในระบบ' must reach the model as missing, not as an invented 0.5."""
    known = predict_default_probability(officer_payload(), bundle)
    unknown = predict_default_probability(
        officer_payload(EXT_SOURCE_1=np.nan, EXT_SOURCE_2=np.nan, EXT_SOURCE_3=np.nan), bundle
    )
    assert 0.0 <= unknown <= 1.0
    assert unknown != known


def test_missing_external_score_sets_the_imputer_indicator(bundle):
    """The indicator the model trained on must actually fire at serve time."""
    names = bundle["selected_feature_names"]
    indicator = "missingindicator_EXT_SOURCE_1"
    if indicator not in names:
        pytest.skip("EXT_SOURCE_1 indicator was not selected for this bundle")

    def indicator_value(values):
        from src.feature_generation import generate_features

        frame = generate_features(build_input_frame(values, bundle))
        matrix = bundle["variance_filter"].transform(bundle["preprocessor"].transform(frame))
        selected = matrix[:, bundle["selected_indices"]]
        dense = selected.toarray() if hasattr(selected, "toarray") else np.asarray(selected)
        return dense.ravel()[names.index(indicator)]

    assert indicator_value(officer_payload(EXT_SOURCE_1=np.nan)) != indicator_value(
        officer_payload(EXT_SOURCE_1=0.5)
    )


def test_pensioner_is_scored_as_missing_employment(bundle):
    pensioner = officer_payload(
        NAME_INCOME_TYPE="Pensioner", DAYS_EMPLOYED=np.nan, DAYS_EMPLOYED_ANOM=1
    )
    probability = predict_default_probability(pensioner, bundle)
    assert 0.0 <= probability <= 1.0
    assert probability != predict_default_probability(officer_payload(), bundle)


def test_a_renamed_form_field_fails_loudly(bundle):
    with pytest.raises(KeyError):
        predict_default_probability(officer_payload(TYPO_COLUMN=1), bundle)


def test_bundle_ships_the_categories_the_dropdowns_must_offer(bundle):
    categories = bundle["input_categories"]
    for column in ("OCCUPATION_TYPE", "ORGANIZATION_TYPE", "NAME_EDUCATION_TYPE"):
        assert categories[column], f"{column} has no options for the form to render"


def test_every_shipped_category_scores_without_error(bundle):
    """A category the officer can pick must be one the encoder knows."""
    for column, options in bundle["input_categories"].items():
        for option in options:
            probability = predict_default_probability(
                officer_payload(**{column: option}), bundle
            )
            assert 0.0 <= probability <= 1.0


def test_occupation_can_be_left_unspecified(bundle):
    """31% of training rows have no occupation, so the form must allow that."""
    assert bundle["input_missing_rates"]["OCCUPATION_TYPE"] > 0.3
    probability = predict_default_probability(
        officer_payload(OCCUPATION_TYPE=np.nan), bundle
    )
    assert 0.0 <= probability <= 1.0
