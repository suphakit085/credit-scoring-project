"""Fail-fast quality checks for the processed training dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = {"SK_ID_CURR", "TARGET", "DAYS_EMPLOYED"}
DEFAULT_CHUNK_ROWS = 20_000


def check_data(
    path: Path, sample_rows: int | None = None, chunk_rows: int = DEFAULT_CHUNK_ROWS
) -> list[str]:
    """Run every check while streaming the file in chunks.

    Reading the whole 795 MB CSV at once and calling ``to_numpy()`` on its
    numeric columns allocated over a gigabyte for checks that only ever need
    one chunk at a time. Uniqueness is the one global property, and it needs
    just the ID column, which is kept in full.
    """
    errors: list[str] = []
    ids: list[np.ndarray] = []
    target_values: set = set()
    positives = rows = target_rows = 0
    missing_counts: pd.Series | None = None
    columns: list[str] = []
    sentinel_found = infinity_found = False

    reader = pd.read_csv(path, nrows=sample_rows, chunksize=chunk_rows, low_memory=False)
    for chunk in reader:
        if not columns:
            columns = list(chunk.columns)
            missing = REQUIRED_COLUMNS.difference(columns)
            if missing:
                errors.append(f"Missing required columns: {sorted(missing)}")
                print("Quality check: FAIL")
                print(f"- {errors[0]}")
                return errors

        rows += len(chunk)
        # Copy out just the values. A column selected from a chunk can be a
        # view onto that chunk's whole consolidated numeric block, so keeping
        # the Series kept every chunk alive and peak memory stayed at 1.2 GB.
        ids.append(chunk["SK_ID_CURR"].to_numpy(copy=True))
        target = chunk["TARGET"].dropna()
        target_values.update(target.unique().tolist())
        positives += int(target.eq(1).sum())
        target_rows += len(target)
        sentinel_found |= bool(chunk["DAYS_EMPLOYED"].eq(365243).any())
        # np.isinf on the frame yields a 1-byte boolean result per cell; the
        # old to_numpy(dtype=float) made a full 8-byte copy of every number.
        infinity_found |= bool(np.isinf(chunk.select_dtypes(include=[np.number])).to_numpy().any())
        counts = chunk.isna().sum()
        missing_counts = counts if missing_counts is None else missing_counts.add(counts, fill_value=0)
        # Release before the reader parses the next chunk; otherwise the loop
        # variable keeps this one alive and two chunks coexist at the peak.
        del chunk

    all_ids = np.concatenate(ids) if ids else np.array([])
    if len(np.unique(all_ids)) != len(all_ids):
        errors.append("SK_ID_CURR is not unique")
    if not target_values.issubset({0, 1}):
        errors.append("TARGET contains values other than 0 and 1")
    if sentinel_found:
        errors.append("DAYS_EMPLOYED sentinel 365243 is still present")
    if infinity_found:
        errors.append("Numeric data contains infinity")

    over_half_missing = int((missing_counts / max(rows, 1) > 0.5).sum()) if missing_counts is not None else 0
    print(f"Shape: ({rows}, {len(columns)})")
    print(f"Positive rate: {positives / max(target_rows, 1):.4%}")
    print(f"Columns over 50% missing: {over_half_missing}")
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
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    args = parser.parse_args()
    raise SystemExit(1 if check_data(args.data, args.sample_rows, args.chunk_rows) else 0)


if __name__ == "__main__":
    main()
