"""Build leakage-safe, joined Home Credit application datasets."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGGREGATIONS = {
    "bureau.csv": ("BUREAU", {"SK_ID_BUREAU"}),
    "previous_application.csv": ("PREV", {"SK_ID_PREV"}),
    "POS_CASH_balance.csv": ("POS", {"SK_ID_PREV"}),
    "credit_card_balance.csv": ("CC", {"SK_ID_PREV"}),
    "installments_payments.csv": ("INSTALL", {"SK_ID_PREV"}),
}


def clean_application_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix known sentinels while leaving imputation for the training split."""
    result = df.copy()
    if "DAYS_EMPLOYED" in result:
        anomaly = result["DAYS_EMPLOYED"].eq(365243)
        result["DAYS_EMPLOYED_ANOM"] = anomaly.astype("int8")
        result.loc[anomaly, "DAYS_EMPLOYED"] = np.nan
    return result


def aggregate_numeric(
    df: pd.DataFrame,
    prefix: str,
    group_column: str = "SK_ID_CURR",
    excluded_columns: set[str] | None = None,
) -> pd.DataFrame:
    """Aggregate every non-ID numeric column per applicant."""
    excluded = set(excluded_columns or ()) | {group_column}
    numeric = [c for c in df.select_dtypes(include=[np.number]).columns if c not in excluded]
    if not numeric:
        return df[[group_column]].drop_duplicates().reset_index(drop=True)
    aggregated = df.groupby(group_column, sort=False)[numeric].agg(
        ["count", "mean", "max", "min", "sum", "std"]
    )
    aggregated.columns = [f"{prefix}_{column}_{stat}" for column, stat in aggregated.columns]
    return aggregated.reset_index()


def _write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temporary, index=False)
    temporary.replace(path)


def process_dataset(
    raw_dir: Path, output_dir: Path, reuse_aggregates: bool = False
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = clean_application_data(pd.read_csv(raw_dir / "application_train.csv", low_memory=False))
    test = clean_application_data(pd.read_csv(raw_dir / "application_test.csv", low_memory=False))

    for filename, (prefix, excluded) in AGGREGATIONS.items():
        path = raw_dir / filename
        aggregate_path = output_dir / f"{prefix.lower()}_agg.csv"
        if reuse_aggregates and aggregate_path.exists():
            print(f"Reusing {aggregate_path.name}...")
            aggregated = pd.read_csv(aggregate_path, low_memory=False)
        elif not path.exists():
            print(f"Skipping missing optional table: {path}")
            continue
        else:
            print(f"Aggregating {filename}...")
            source = pd.read_csv(path, low_memory=False)
            aggregated = aggregate_numeric(source, prefix, excluded_columns=excluded)
            del source
            gc.collect()
            _write_csv_atomic(aggregated, aggregate_path)
        train = train.merge(aggregated, on="SK_ID_CURR", how="left", validate="one_to_one")
        test = test.merge(aggregated, on="SK_ID_CURR", how="left", validate="one_to_one")
        del aggregated
        gc.collect()

    train_path = output_dir / "train_cleaned.csv"
    test_path = output_dir / "test_cleaned.csv"
    _write_csv_atomic(train, train_path)
    _write_csv_atomic(test, test_path)
    print(f"Saved {train.shape} to {train_path}")
    print(f"Saved {test.shape} to {test_path}")
    return train_path, test_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument(
        "--reuse-aggregates",
        action="store_true",
        help="Reuse existing *_agg.csv files instead of recomputing large history tables",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_dataset(args.raw_dir, args.output_dir, args.reuse_aggregates)


if __name__ == "__main__":
    main()
