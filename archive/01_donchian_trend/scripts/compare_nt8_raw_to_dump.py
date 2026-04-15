#!/usr/bin/env python
"""Rebuild higher-timeframe bars from current NT8 raw 1m dumps and compare them to NT8 chart dumps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SESSION_TEMPLATE_TZ = "America/Chicago"
SESSION_START_HOUR = 17
DEFAULT_START = "2026-02-23"
DEFAULT_END = "2026-03-27"

SPECS = {
    "MNQ": {
        "timeframe_minutes": 30,
        "raw_csv": Path("data/ninjatrader/NT8_RawDump_MNQ.csv"),
        "dump_csv": Path("data/ninjatrader/NT8_DumpMNQ.csv"),
    },
    "MGC": {
        "timeframe_minutes": 60,
        "raw_csv": Path("data/ninjatrader/NT8_RawDump_MGC.csv"),
        "dump_csv": Path("data/ninjatrader/NT8_DumpMGC.csv"),
    },
}

COMPARE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "sma_50",
    "atr_14_wilder",
    "donchian_high_5",
    "donchian_low_5",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", choices=sorted(SPECS.keys()), required=True)
    parser.add_argument("--raw-csv", type=Path)
    parser.add_argument("--dump-csv", type=Path)
    parser.add_argument("--start-date", default=DEFAULT_START)
    parser.add_argument("--end-date", default=DEFAULT_END)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _wilder_rma(values: np.ndarray, period: int) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float64)
    if len(values) < period:
        return result
    seed = float(np.mean(values[:period]))
    result[period - 1] = seed
    for idx in range(period, len(values)):
        result[idx] = (result[idx - 1] * (period - 1) + values[idx]) / period
    return result


def _compute_session_date(timestamp_utc: pd.Series) -> pd.Series:
    local = timestamp_utc.dt.tz_convert(SESSION_TEMPLATE_TZ)
    session_date = local.dt.floor("D")
    rollover_mask = local.dt.hour >= SESSION_START_HOUR
    session_date = session_date + pd.to_timedelta(rollover_mask.astype(np.int8), unit="D")
    return session_date.dt.strftime("%Y-%m-%d")


def _load_dump(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp_utc"] = pd.to_datetime(df["Timestamp_UTC"], utc=True)
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "SMA_50": "sma_50",
        "ATR_14_Wilder": "atr_14_wilder",
        "DonchianHigh_5": "donchian_high_5",
        "DonchianLow_5": "donchian_low_5",
    }
    df = df.rename(columns=rename_map)
    for col in COMPARE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["session_date"] = _compute_session_date(df["timestamp_utc"])
    return df[["timestamp_utc", "session_date", *COMPARE_COLUMNS]].sort_values("timestamp_utc", kind="stable").reset_index(drop=True)


def _load_raw_and_rebuild(path: Path, timeframe_minutes: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp_utc"] = pd.to_datetime(df["Timestamp_UTC"], utc=True)
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[["timestamp_utc", "open", "high", "low", "close", "volume"]].sort_values("timestamp_utc", kind="stable")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.set_index(df["timestamp_utc"].dt.tz_convert(SESSION_TEMPLATE_TZ))

    rebuilt = df.resample(
        f"{timeframe_minutes}min",
        label="right",
        closed="right",
        origin="start_day",
        offset="17h",
    ).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    rebuilt = rebuilt.dropna(subset=["open", "high", "low", "close"])

    highs = rebuilt["high"].to_numpy(dtype=np.float64)
    lows = rebuilt["low"].to_numpy(dtype=np.float64)
    closes = rebuilt["close"].to_numpy(dtype=np.float64)
    tr = np.maximum(highs - lows, np.zeros(len(rebuilt), dtype=np.float64))
    if len(rebuilt) > 1:
        prev_close = np.roll(closes, 1)
        prev_close[0] = closes[0]
        tr = np.maximum(tr, np.abs(highs - prev_close))
        tr = np.maximum(tr, np.abs(lows - prev_close))

    rebuilt["sma_50"] = rebuilt["close"].rolling(window=50, min_periods=50).mean()
    rebuilt["atr_14_wilder"] = _wilder_rma(tr, 14)
    rebuilt["donchian_high_5"] = rebuilt["high"].shift(1).rolling(window=5, min_periods=5).max()
    rebuilt["donchian_low_5"] = rebuilt["low"].shift(1).rolling(window=5, min_periods=5).min()
    rebuilt["timestamp_utc"] = rebuilt.index.tz_convert("UTC")
    rebuilt["session_date"] = _compute_session_date(rebuilt["timestamp_utc"])

    return rebuilt.reset_index(drop=True)[["timestamp_utc", "session_date", *COMPARE_COLUMNS]]


def _filter_window(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    mask = (df["session_date"] >= start_date) & (df["session_date"] <= end_date)
    return df.loc[mask].reset_index(drop=True)


def _build_report(dump_df: pd.DataFrame, rebuilt_df: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = dump_df.merge(rebuilt_df, on="timestamp_utc", how="outer", suffixes=("_dump", "_rebuilt"), indicator=True)
    matched = merged.loc[merged["_merge"] == "both"].copy()
    for col in COMPARE_COLUMNS:
        matched[f"{col}_abs_delta"] = (matched[f"{col}_dump"] - matched[f"{col}_rebuilt"]).abs()

    dump_only = merged.loc[merged["_merge"] == "left_only", ["timestamp_utc"]].copy()
    rebuilt_only = merged.loc[merged["_merge"] == "right_only", ["timestamp_utc"]].copy()

    summary = {
        "row_counts": {
            "dump_rows": int(len(dump_df)),
            "rebuilt_rows": int(len(rebuilt_df)),
            "matched_rows": int(len(matched)),
            "dump_only_timestamps": int(len(dump_only)),
            "rebuilt_only_timestamps": int(len(rebuilt_only)),
        },
        "column_deltas": {},
    }

    for col in COMPARE_COLUMNS:
        delta_col = f"{col}_abs_delta"
        finite = matched[delta_col].replace([np.inf, -np.inf], np.nan).dropna()
        summary["column_deltas"][col] = {
            "mean_abs_delta": float(finite.mean()) if len(finite) else None,
            "max_abs_delta": float(finite.max()) if len(finite) else None,
            "nonzero_delta_rows": int((finite > 0.0).sum()) if len(finite) else 0,
        }

    mismatches = matched.copy()
    mismatch_cols = [f"{col}_abs_delta" for col in COMPARE_COLUMNS]
    mismatches["max_abs_delta"] = mismatches[mismatch_cols].max(axis=1, skipna=True)
    mismatches = mismatches.sort_values("max_abs_delta", ascending=False, kind="stable")
    return summary, matched, dump_only, rebuilt_only, mismatches


def main() -> None:
    args = parse_args()
    spec = SPECS[args.instrument]
    raw_csv = args.raw_csv or spec["raw_csv"]
    dump_csv = args.dump_csv or spec["dump_csv"]
    output_dir = args.output or Path("output/nt8_validation") / f"{args.instrument.lower()}_march_crucible_raw_vs_dump"

    dump_df = _load_dump(dump_csv)
    rebuilt_df = _load_raw_and_rebuild(raw_csv, spec["timeframe_minutes"])
    dump_df = _filter_window(dump_df, args.start_date, args.end_date)
    rebuilt_df = _filter_window(rebuilt_df, args.start_date, args.end_date)

    summary, matched, dump_only, rebuilt_only, mismatches = _build_report(dump_df, rebuilt_df)
    summary["instrument"] = args.instrument
    summary["raw_csv"] = str(raw_csv)
    summary["dump_csv"] = str(dump_csv)
    summary["start_date"] = args.start_date
    summary["end_date"] = args.end_date
    summary["timeframe_minutes"] = spec["timeframe_minutes"]

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    matched.to_csv(output_dir / "matched_deltas.csv", index=False)
    dump_only.to_csv(output_dir / "dump_only_timestamps.csv", index=False)
    rebuilt_only.to_csv(output_dir / "rebuilt_only_timestamps.csv", index=False)
    mismatches.head(250).to_csv(output_dir / "largest_mismatches.csv", index=False)

    print(f"Output: {output_dir}")
    print(f"Dump rows: {summary['row_counts']['dump_rows']}")
    print(f"Rebuilt rows: {summary['row_counts']['rebuilt_rows']}")
    print(f"Matched rows: {summary['row_counts']['matched_rows']}")
    print(f"Dump-only timestamps: {summary['row_counts']['dump_only_timestamps']}")
    print(f"Rebuilt-only timestamps: {summary['row_counts']['rebuilt_only_timestamps']}")


if __name__ == "__main__":
    main()
