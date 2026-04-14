#!/usr/bin/env python
"""Convert a NinjaTrader 8 text export into a parquet dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("data/raw/RB 05-26.Last.txt")
TIMESTAMP_FORMAT = "%Y%m%d %H%M%S"


def _infer_symbol(txt_path: Path) -> str:
    stem = txt_path.stem.strip()
    if not stem:
        raise ValueError(f"Could not infer symbol from path: {txt_path}")
    first_token = stem.split()[0]
    first_token = first_token.split("_")[0]
    first_token = first_token.split("-")[0]
    return first_token.upper()


def _default_output_path(txt_path: Path) -> Path:
    symbol = _infer_symbol(txt_path)
    return Path("data/processed") / f"{symbol}_1m_raw.parquet"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def load_nt8_txt(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep=";", header=None)
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], format=TIMESTAMP_FORMAT, errors="raise")
    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="raise")
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    return df


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    output_path = args.output if args.output is not None else _default_output_path(args.input)
    df = load_nt8_txt(args.input)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"Input file: {args.input}")
    print(f"Output file: {output_path}")
    print(f"Start date: {df['timestamp'].iloc[0]}")
    print(f"End date: {df['timestamp'].iloc[-1]}")
    print(f"Rows processed: {len(df)}")


if __name__ == "__main__":
    main()
