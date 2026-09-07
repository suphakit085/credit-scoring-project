import numpy as np
import pandas as pd

from src.data_preprocessing import aggregate_numeric, clean_application_data


def test_clean_application_replaces_employment_sentinel_and_adds_flag():
    result = clean_application_data(pd.DataFrame({"DAYS_EMPLOYED": [365243, -100]}))
    assert np.isnan(result.loc[0, "DAYS_EMPLOYED"])
    assert result["DAYS_EMPLOYED_ANOM"].tolist() == [1, 0]


def test_aggregate_numeric_returns_one_row_per_applicant():
    source = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 2],
            "SK_ID_PREV": [10, 11, 12],
            "AMT": [10.0, 30.0, 5.0],
            "CATEGORY": ["a", "b", "a"],
        }
    )
    result = aggregate_numeric(source, "PREV", excluded_columns={"SK_ID_PREV"})
    assert result["SK_ID_CURR"].tolist() == [1, 2]
    assert result.loc[0, "PREV_AMT_mean"] == 20.0
    assert result.loc[0, "PREV_AMT_count"] == 2
