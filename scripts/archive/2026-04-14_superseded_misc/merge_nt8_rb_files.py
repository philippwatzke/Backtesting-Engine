#!/usr/bin/env python
"""Merge all NinjaTrader RB text exports into a continuous parquet dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")
OUTPUT_PATH = Path("data/processed/RB_1m_continuous.parquet")
TIMESTAMP_FORMAT = "%Y%m%d %H%M%S"
GAP_THRESHOLD = 0.03


def _load_rb_txt(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep=";", header=None)
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], format=TIMESTAMP_FORMAT, errors="raise")
    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="raise")
    df["source_file"] = filepath.name
    return df


def _find_rb_files(raw_dir: Path) -> list[Path]:
    files = sorted(path for path in raw_dir.glob("*.txt") if "RB" in path.name.upper())
    if not files:
        raise FileNotFoundError(f"No RB .txt files found in {raw_dir}")
    return files


def _print_rollover_gaps(df: pd.DataFrame) -> None:
    prev_close = df["close"].shift(1)
    gap_ratio = (df["open"] - prev_close) / prev_close
    gap_mask = gap_ratio.abs() > GAP_THRESHOLD

    print("Rollover gaps > 3.00%:")
    if not gap_mask.any():
        print("  None detected")
        return

    gap_rows = df.loc[gap_mask, ["timestamp", "open"]].copy()
    gap_rows["prev_timestamp"] = df["timestamp"].shift(1).loc[gap_mask].to_numpy()
    gap_rows["prev_close"] = prev_close.loc[gap_mask].to_numpy()
    gap_rows["gap_pct"] = gap_ratio.loc[gap_mask].to_numpy() * 100.0

    for row in gap_rows.itertuples(index=False):
        print(
            f"  {row.prev_timestamp} -> {row.timestamp} | "
            f"close={row.prev_close:.4f} open={row.open:.4f} gap={row.gap_pct:.2f}%"
        )


def main() -> None:
    rb_files = _find_rb_files(RAW_DIR)
    frames = [_load_rb_txt(path) for path in rb_files]
    master = pd.concat(frames, ignore_index=True)
    exact_dupes_removed = int(master.duplicated().sum())
    master = master.drop_duplicates()

    duplicate_timestamp_rows = int(master["timestamp"].duplicated(keep=False).sum())
    master = (
        master.sort_values(["timestamp", "volume", "source_file"], ascending=[True, False, True], kind="stable")
        .drop_duplicates(subset=["timestamp"], keep="first")
        .sort_values("timestamp", kind="stable")
        .reset_index(drop=True)
    )

    _print_rollover_gaps(master)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    master.drop(columns=["source_file"]).to_parquet(OUTPUT_PATH, index=False)

    print(f"Merged files: {len(rb_files)}")
    print(f"Exact duplicates removed: {exact_dupes_removed}")
    print(f"Overlapping timestamp rows resolved: {duplicate_timestamp_rows}")
    print(f"Output file: {OUTPUT_PATH}")
    print(f"Start date: {master['timestamp'].iloc[0]}")
    print(f"End date: {master['timestamp'].iloc[-1]}")
    print(f"Rows processed: {len(master)}")


if __name__ == "__main__":
    main()
