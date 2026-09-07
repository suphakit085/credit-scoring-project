"""Fail-fast quality checks for the processed training dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def check_data(path: Path, sample_rows: int | None = None) -> list[str]:
    df = pd.read_csv(path, nrows=sample_rows, low_memory=False)
    errors: list[str] = []
    required = {"SK_ID_CURR", "TARGET", "DAYS_EMPLOYED"}
    missing = required.difference(df.columns)
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")
        return errors
    if df["SK_ID_CURR"].duplicated().any():
        errors.append("SK_ID_CURR is not unique")
    if not set(df["TARGET"].dropna().unique()).issubset({0, 1}):
        errors.append("TARGET contains values other than 0 and 1")
    if df["DAYS_EMPLOYED"].eq(365243).any():
        errors.append("DAYS_EMPLOYED sentinel 365243 is still present")
    numeric = df.select_dtypes(include=[np.number])
    if np.isinf(numeric.to_numpy()).any():
        errors.append("Numeric data contains infinity")

    print(f"Shape: {df.shape}")
    print(f"Positive rate: {df['TARGET'].mean():.4%}")
    print(f"Columns over 50% missing: {(df.isna().mean() > 0.5).sum()}")
    print("Quality check: PASS" if not errors else "Quality check: FAIL")
    for error in errors:
        print(f"- {error}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "train_cleaned.csv",
    )
    parser.add_argument("--sample-rows", type=int, default=None)
    args = parser.parse_args()
    raise SystemExit(1 if check_data(args.data, args.sample_rows) else 0)


if __name__ == "__main__":
    main()
