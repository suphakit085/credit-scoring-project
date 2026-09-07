import numpy as np
import pandas as pd

from src.feature_generation import generate_features


def base_frame():
    return pd.DataFrame(
        {
            "AMT_CREDIT": [10_000.0, 20_000.0],
            "AMT_ANNUITY": [1_000.0, 0.0],
            "AMT_GOODS_PRICE": [5_000.0, 0.0],
            "AMT_INCOME_TOTAL": [50_000.0, 100_000.0],
            "CNT_FAM_MEMBERS": [2.0, 1.0],
            "DAYS_BIRTH": [-3652.5, -7305.0],
            "DAYS_EMPLOYED": [-365.25, -730.5],
            "DAYS_REGISTRATION": [-100.0, -200.0],
            "DAYS_ID_PUBLISH": [-300.0, -400.0],
            "EXT_SOURCE_1": [0.2, 0.3],
            "EXT_SOURCE_2": [0.5, 0.6],
            "EXT_SOURCE_3": [0.8, 0.9],
        }
    )


def test_generate_features_has_consistent_financial_ratios():
    features = generate_features(base_frame())
    assert features.loc[0, "CREDIT_TO_ANNUITY_RATIO"] == 10.0
    assert features.loc[0, "CREDIT_TO_GOODS_RATIO"] == 2.0
    assert features.loc[0, "CREDIT_TO_INCOME_RATIO"] == 0.2
    assert np.isnan(features.loc[1, "CREDIT_TO_ANNUITY_RATIO"])
    assert np.isnan(features.loc[1, "CREDIT_TO_GOODS_RATIO"])


def test_generate_features_creates_all_time_features():
    features = generate_features(base_frame())
    assert features.loc[0, "AGE_YEARS"] == 10.0
    assert features.loc[0, "EMPLOYMENT_YEARS"] == 1.0
    assert np.isclose(features.loc[0, "EMPLOYMENT_TO_AGE_RATIO"], 0.1)
    assert "REGISTRATION_YEARS" in features
    assert "ID_PUBLISH_YEARS" in features


def test_external_source_std_matches_pandas_training_definition():
    features = generate_features(base_frame().iloc[[0]])
    assert np.isclose(features.iloc[0]["EXT_SOURCE_STD"], np.std([0.2, 0.5, 0.8], ddof=1))


def test_generate_features_preserves_rows_and_never_returns_infinity():
    features = generate_features(base_frame())
    assert len(features) == 2
    numeric = features.select_dtypes(include=[np.number]).to_numpy()
    assert not np.isinf(numeric).any()
