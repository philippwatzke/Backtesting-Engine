#!/usr/bin/env python
"""Build the M6A overnight training split from raw 1-minute data."""
from pathlib import Path

import numpy as np
import pandas as pd

from propfirm.market.data_loader import _session_labels


RAW_PATH = Path("data/raw/M6A_1m_2022_heute_raw.parquet")
TRAIN_SPLIT_PATH = Path("data/processed/M6A_1m_train.parquet")
VALIDATION_SPLIT_PATH = Path("data/processed/M6A_1m_validation.parquet")
OUTPUT_PATH = Path("data/processed/M6A_1m_overnight_train.parquet")
SESSION_START = "18:00"
SESSION_END = "02:00"
SESSION_LENGTH_MINUTES = 481
REQUIRED_PRICE_COLS = ["open", "high", "low", "close", "volume"]
STATIC_COLS = ["rtype", "publisher_id", "instrument_id", "symbol"]


def _load_split_bounds() -> tuple[pd.Timestamp, pd.Timestamp]:
    train = pd.read_parquet(TRAIN_SPLIT_PATH)
    train_idx = train.index.tz_convert("America/New_York")
    if VALIDATION_SPLIT_PATH.exists():
        validation = pd.read_parquet(VALIDATION_SPLIT_PATH)
        validation_start = validation.index.tz_convert("America/New_York").min().normalize()
    else:
        validation_start = (train_idx.max() + pd.Timedelta(days=1)).normalize()
    train_start = train_idx.min().normalize()
    return train_start, validation_start


def _build_session(full_df: pd.DataFrame, session_start: pd.Timestamp) -> pd.DataFrame | None:
    expected_index = pd.date_range(
        start=session_start,
        periods=SESSION_LENGTH_MINUTES,
        freq="min",
        tz="America/New_York",
    )
    session_end = expected_index[-1]
    session_slice = full_df.loc[(full_df.index >= session_start) & (full_df.index <= session_end)].copy()
    if session_slice.empty:
        return None

    prev_rows = full_df.loc[full_df.index < session_start]
    if prev_rows.empty:
        return None
    seed = prev_rows.iloc[[-1]].copy()
    seed.index = pd.DatetimeIndex([session_start - pd.Timedelta(minutes=1)], tz="America/New_York")

    session_with_seed = pd.concat([seed, session_slice], axis=0)
    reindexed = session_with_seed.reindex(
        pd.DatetimeIndex([seed.index[0]], tz="America/New_York").append(expected_index)
    )

    close = reindexed["close"].ffill()
    if close.iloc[1:].isna().any():
        return None

    actual_mask = reindexed["close"].notna()
    for col in ["open", "high", "low", "close"]:
        reindexed[col] = reindexed[col].where(actual_mask, close)
    reindexed["volume"] = reindexed["volume"].fillna(0).astype(np.int64)

    for col in STATIC_COLS:
        if col in reindexed.columns:
            reindexed[col] = reindexed[col].ffill().bfill()

    built = reindexed.iloc[1:].copy()
    built.index.name = full_df.index.name
    return built


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw M6A file not found: {RAW_PATH}")
    if not TRAIN_SPLIT_PATH.exists():
        raise FileNotFoundError(f"Reference train split not found: {TRAIN_SPLIT_PATH}")

    train_start, validation_start = _load_split_bounds()
    latest_session_start = validation_start - pd.Timedelta(days=1)

    raw_df = pd.read_parquet(RAW_PATH)
    raw_df = raw_df.sort_index(kind="stable").copy()
    raw_df.index = raw_df.index.tz_convert("America/New_York")
    raw_df = raw_df.loc[~raw_df.index.duplicated(keep="first")]

    labels = _session_labels(raw_df.index, SESSION_START, SESSION_END)
    candidate_sessions = sorted(
        {
            pd.Timestamp(label).tz_localize("America/New_York")
            for label in labels
            if train_start.date() <= label < latest_session_start.date()
        }
    )

    built_sessions = []
    for session_start_ts in candidate_sessions:
        built = _build_session(raw_df, session_start_ts + pd.Timedelta(hours=18))
        if built is not None:
            built_sessions.append(built)

    if not built_sessions:
        raise ValueError("No overnight train sessions could be built from raw M6A data")

    overnight_train = pd.concat(built_sessions, axis=0)
    overnight_train = overnight_train[[col for col in raw_df.columns if col in overnight_train.columns]]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    overnight_train.to_parquet(OUTPUT_PATH)

    print(f"Wrote {len(overnight_train)} rows to {OUTPUT_PATH}")
    print(
        "Date range:",
        overnight_train.index.min(),
        "to",
        overnight_train.index.max(),
    )
    print(f"Sessions built: {len(built_sessions)}")


if __name__ == "__main__":
    main()
