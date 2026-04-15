#!/usr/bin/env python
"""Compare an NT8 DataDumpExporter CSV against Python-recomputed ETH bars.

This is the first hard control check for "The March Crucible":
2026-02-23 through 2026-03-27.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SESSION_TZ = "America/New_York"
SESSION_TEMPLATE_TZ = "America/Chicago"
SESSION_START_HOUR = 17
DEFAULT_START = "2026-02-23"
DEFAULT_END = "2026-03-27"

SPECS = {
    "MNQ": {
        "timeframe_minutes": 30,
        "python_data": Path("data/processed/MNQ_1m_adjusted_NT8.parquet"),
    },
    "MGC": {
        "timeframe_minutes": 60,
        "python_data": Path("data/processed/MGC_1m_full_test.parquet"),
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
    parser.add_argument("--nt8-csv", type=Path, required=True)
    parser.add_argument("--python-data", type=Path)
    parser.add_argument("--start-date", default=DEFAULT_START)
    parser.add_argument("--end-date", default=DEFAULT_END)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _wilder_rma(values: np.ndarray, period: int) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float64)
    if len(values) == 0:
        return result
    if len(values) < period:
        return result

    seed = float(np.mean(values[:period]))
    result[period - 1] = seed
    for idx in range(period, len(values)):
        result[idx] = (result[idx - 1] * (period - 1) + values[idx]) / period
    return result


def _load_nt8_dump(path: Path) -> pd.DataFrame:
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
    df = df[["timestamp_utc", *COMPARE_COLUMNS]].sort_values("timestamp_utc", kind="stable").reset_index(drop=True)
    return df


def _compute_session_date(timestamp_utc: pd.Series) -> pd.Series:
    local = timestamp_utc.dt.tz_convert(SESSION_TEMPLATE_TZ)
    session_date = local.dt.floor("D")
    rollover_mask = local.dt.hour >= SESSION_START_HOUR
    session_date = session_date + pd.to_timedelta(rollover_mask.astype(np.int8), unit="D")
    return session_date.dt.strftime("%Y-%m-%d")


def _load_python_bars(path: Path, timeframe_minutes: int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Python parquet must be indexed by DatetimeIndex")
    if df.index.tz is None:
        raise ValueError("Python parquet index must be timezone-aware")

    df = df.sort_index(kind="stable").copy()
    df.index = df.index.tz_convert(SESSION_TEMPLATE_TZ)
    df = df.loc[~df.index.duplicated(keep="first")]

    # Match NT8 chart bars produced from the CME ETH template.
    # NT8 bars are effectively close-time labeled on the exchange-session grid.
    df = df.between_time("17:00", "16:59", inclusive="both")
    resampled = df.resample(
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
    resampled = resampled.dropna(subset=["open", "high", "low", "close"])

    highs = resampled["high"].to_numpy(dtype=np.float64)
    lows = resampled["low"].to_numpy(dtype=np.float64)
    closes = resampled["close"].to_numpy(dtype=np.float64)

    tr = np.maximum(highs - lows, np.zeros(len(resampled), dtype=np.float64))
    if len(resampled) > 1:
        prev_close = np.roll(closes, 1)
        prev_close[0] = closes[0]
        tr = np.maximum(tr, np.abs(highs - prev_close))
        tr = np.maximum(tr, np.abs(lows - prev_close))

    resampled["sma_50"] = resampled["close"].rolling(window=50, min_periods=50).mean()
    resampled["atr_14_wilder"] = _wilder_rma(tr, 14)
    resampled["donchian_high_5"] = resampled["high"].shift(1).rolling(window=5, min_periods=5).max()
    resampled["donchian_low_5"] = resampled["low"].shift(1).rolling(window=5, min_periods=5).min()
    resampled["timestamp_utc"] = resampled.index.tz_convert("UTC")
    resampled["session_date"] = _compute_session_date(resampled["timestamp_utc"])

    return resampled.reset_index(drop=True)[["timestamp_utc", "session_date", *COMPARE_COLUMNS]]


def _filter_window(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if "session_date" not in df.columns:
        df = df.copy()
        df["session_date"] = _compute_session_date(df["timestamp_utc"])
    mask = (df["session_date"] >= start_date) & (df["session_date"] <= end_date)
    return df.loc[mask].reset_index(drop=True)


def _build_report(nt8_df: pd.DataFrame, py_df: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = nt8_df.merge(py_df, on="timestamp_utc", how="outer", suffixes=("_nt8", "_py"), indicator=True)

    matched = merged.loc[merged["_merge"] == "both"].copy()
    for col in COMPARE_COLUMNS:
        matched[f"{col}_abs_delta"] = (matched[f"{col}_nt8"] - matched[f"{col}_py"]).abs()

    nt8_only = merged.loc[merged["_merge"] == "left_only", ["timestamp_utc"]].copy()
    py_only = merged.loc[merged["_merge"] == "right_only", ["timestamp_utc"]].copy()

    summary = {
        "row_counts": {
            "nt8_rows": int(len(nt8_df)),
            "python_rows": int(len(py_df)),
            "matched_rows": int(len(matched)),
            "nt8_only_timestamps": int(len(nt8_only)),
            "python_only_timestamps": int(len(py_only)),
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

    return summary, matched, nt8_only, py_only, mismatches


def main() -> None:
    args = parse_args()
    spec = SPECS[args.instrument]
    python_data = args.python_data or spec["python_data"]
    output_dir = args.output or Path("output/nt8_validation") / f"{args.instrument.lower()}_march_crucible_dump_compare"

    nt8_df = _load_nt8_dump(args.nt8_csv)
    py_df = _load_python_bars(python_data, spec["timeframe_minutes"])

    nt8_df = _filter_window(nt8_df, args.start_date, args.end_date)
    py_df = _filter_window(py_df, args.start_date, args.end_date)

    summary, matched, nt8_only, py_only, mismatches = _build_report(nt8_df, py_df)
    summary["instrument"] = args.instrument
    summary["nt8_csv"] = str(args.nt8_csv)
    summary["python_data"] = str(python_data)
    summary["start_date"] = args.start_date
    summary["end_date"] = args.end_date
    summary["timeframe_minutes"] = spec["timeframe_minutes"]

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    matched.to_csv(output_dir / "matched_deltas.csv", index=False)
    nt8_only.to_csv(output_dir / "nt8_only_timestamps.csv", index=False)
    py_only.to_csv(output_dir / "python_only_timestamps.csv", index=False)
    mismatches.head(250).to_csv(output_dir / "largest_mismatches.csv", index=False)

    print(f"Output: {output_dir}")
    print(f"NT8 rows: {summary['row_counts']['nt8_rows']}")
    print(f"Python rows: {summary['row_counts']['python_rows']}")
    print(f"Matched rows: {summary['row_counts']['matched_rows']}")
    print(f"NT8-only timestamps: {summary['row_counts']['nt8_only_timestamps']}")
    print(f"Python-only timestamps: {summary['row_counts']['python_only_timestamps']}")


if __name__ == "__main__":
    main()
