import numpy as np
import pandas as pd

from src.check_data_quality import check_data


def write(tmp_path, frame):
    path = tmp_path / "train.csv"
    frame.to_csv(path, index=False)
    return path


def clean_frame(rows=10):
    return pd.DataFrame(
        {
            "SK_ID_CURR": range(rows),
            "TARGET": [0, 1] * (rows // 2),
            "DAYS_EMPLOYED": [-100.0] * rows,
            "AMT": np.linspace(1, 2, rows),
        }
    )


def test_clean_data_passes(tmp_path):
    assert check_data(write(tmp_path, clean_frame()), chunk_rows=3) == []


def test_duplicate_id_split_across_chunks_is_caught(tmp_path):
    """Chunked reading must not hide a duplicate whose copies land in different chunks."""
    frame = clean_frame()
    frame.loc[9, "SK_ID_CURR"] = 0  # first and last row, chunks 1 and 4 at chunk_rows=3
    assert "SK_ID_CURR is not unique" in check_data(write(tmp_path, frame), chunk_rows=3)


def test_sentinel_and_infinity_are_caught(tmp_path):
    frame = clean_frame()
    frame.loc[7, "DAYS_EMPLOYED"] = 365243
    frame.loc[8, "AMT"] = np.inf
    errors = check_data(write(tmp_path, frame), chunk_rows=3)
    assert "DAYS_EMPLOYED sentinel 365243 is still present" in errors
    assert "Numeric data contains infinity" in errors


def test_invalid_target_is_caught(tmp_path):
    frame = clean_frame()
    frame.loc[5, "TARGET"] = 2
    assert "TARGET contains values other than 0 and 1" in check_data(
        write(tmp_path, frame), chunk_rows=3
    )


def test_missing_required_column_fails_fast(tmp_path):
    frame = clean_frame().drop(columns=["DAYS_EMPLOYED"])
    errors = check_data(write(tmp_path, frame), chunk_rows=3)
    assert errors and "DAYS_EMPLOYED" in errors[0]
