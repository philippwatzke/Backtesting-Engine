#!/usr/bin/env python
"""Build full-session MGC train, validation, or test splits with extended intraday coverage."""
import argparse
from pathlib import Path

import pandas as pd


RAW_PATH = Path("data/raw/MGC_1m_2022_heute_raw.parquet")
TRAIN_SPLIT_PATH = Path("data/processed/MGC_1m_train.parquet")
VALIDATION_SPLIT_PATH = Path("data/processed/MGC_1m_validation.parquet")
TEST_SPLIT_PATH = Path("data/processed/MGC_1m_test.parquet")
TRAIN_OUTPUT_PATH = Path("data/processed/MGC_1m_full_train.parquet")
VALIDATION_OUTPUT_PATH = Path("data/processed/MGC_1m_full_val.parquet")
TEST_OUTPUT_PATH = Path("data/processed/MGC_1m_full_test.parquet")
SESSION_START = "08:00"
SESSION_END = "15:59"
SESSION_TZ = "America/New_York"


def _load_split_bounds() -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    train = pd.read_parquet(TRAIN_SPLIT_PATH)
    train_idx = train.index.tz_convert(SESSION_TZ)
    validation = pd.read_parquet(VALIDATION_SPLIT_PATH)
    validation_idx = validation.index.tz_convert(SESSION_TZ)
    test = pd.read_parquet(TEST_SPLIT_PATH)
    test_idx = test.index.tz_convert(SESSION_TZ)
    train_start = train_idx.min().normalize()
    validation_start = validation_idx.min().normalize()
    test_start = test_idx.min().normalize()
    test_end = test_idx.max().normalize() + pd.Timedelta(days=1)
    return train_start, validation_start, test_start, test_end


def _build_full_split(
    raw_df: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    output_path: Path,
    split_name: str,
) -> None:
    split_df = raw_df.loc[(raw_df.index >= start_ts) & (raw_df.index < end_ts)]
    split_df = split_df.between_time(SESSION_START, SESSION_END)

    if split_df.empty:
        raise ValueError(f"No MGC raw rows available inside the requested {split_name} session window")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_parquet(output_path)

    print(f"Wrote {len(split_df)} rows to {output_path}")
    print("Date range:", split_df.index.min(), "to", split_df.index.max())
    print(
        "Time coverage:",
        split_df.index.strftime("%H:%M").min(),
        "to",
        split_df.index.strftime("%H:%M").max(),
    )
    print(f"Unique session dates: {split_df.index.normalize().nunique()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    args = parser.parse_args()

    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw MGC file not found: {RAW_PATH}")
    if not TRAIN_SPLIT_PATH.exists():
        raise FileNotFoundError(f"Reference train split not found: {TRAIN_SPLIT_PATH}")
    if not VALIDATION_SPLIT_PATH.exists():
        raise FileNotFoundError(f"Reference validation split not found: {VALIDATION_SPLIT_PATH}")
    if not TEST_SPLIT_PATH.exists():
        raise FileNotFoundError(f"Reference test split not found: {TEST_SPLIT_PATH}")

    train_start, validation_start, test_start, test_end = _load_split_bounds()

    raw_df = pd.read_parquet(RAW_PATH)
    raw_df = raw_df.sort_index(kind="stable").copy()
    raw_df.index = raw_df.index.tz_convert(SESSION_TZ)
    raw_df = raw_df.loc[~raw_df.index.duplicated(keep="first")]

    if args.split == "train":
        _build_full_split(
            raw_df=raw_df,
            start_ts=train_start,
            end_ts=validation_start,
            output_path=TRAIN_OUTPUT_PATH,
            split_name="full-train",
        )
        return

    _build_full_split(
        raw_df=raw_df,
        start_ts=validation_start if args.split == "val" else test_start,
        end_ts=test_start if args.split == "val" else test_end,
        output_path=VALIDATION_OUTPUT_PATH if args.split == "val" else TEST_OUTPUT_PATH,
        split_name="full-validation" if args.split == "val" else "full-test",
    )


if __name__ == "__main__":
    main()
