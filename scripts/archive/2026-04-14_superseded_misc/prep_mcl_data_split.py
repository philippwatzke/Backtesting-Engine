#!/usr/bin/env python
"""Create a strict chronological IS/OOS split for raw MCL 1-minute data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("data/raw/MCL_1m_2022_heute_raw.parquet")
DEFAULT_IS_OUTPUT = Path("data/processed/MCL_1m_IS.parquet")
DEFAULT_OOS_OUTPUT = Path("data/processed/MCL_1m_OOS.parquet")
IS_FRACTION = 0.70


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--is-output", type=Path, default=DEFAULT_IS_OUTPUT)
    parser.add_argument("--oos-output", type=Path, default=DEFAULT_OOS_OUTPUT)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    df = pd.read_parquet(args.input, engine="pyarrow")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex in raw MCL parquet")
    if len(df) < 2:
        raise ValueError("Need at least 2 rows to build IS/OOS split")

    df = df.sort_index(kind="stable")
    split_idx = int(len(df) * IS_FRACTION)
    if split_idx <= 0 or split_idx >= len(df):
        raise ValueError("Split index is out of bounds")

    df_is = df.iloc[:split_idx].copy()
    df_oos = df.iloc[split_idx:].copy()

    args.is_output.parent.mkdir(parents=True, exist_ok=True)
    df_is.to_parquet(args.is_output, engine="pyarrow")
    df_oos.to_parquet(args.oos_output, engine="pyarrow")

    print(f"In-Sample Start: {df_is.index[0]}")
    print(f"In-Sample End: {df_is.index[-1]}")
    print(f"Out-of-Sample Start: {df_oos.index[0]}")
    print(f"Out-of-Sample End: {df_oos.index[-1]}")
    print(f"In-Sample Rows: {len(df_is)}")
    print(f"Out-of-Sample Rows: {len(df_oos)}")


if __name__ == "__main__":
    main()
