#!/usr/bin/env python
"""OHLC-first forensic parity report for NT8 dump vs raw rebuild."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SESSION_CHI = "America/Chicago"
SESSION_START_HOUR_CHI = 17
DEFAULT_START = "2026-02-23"
DEFAULT_END = "2026-03-27"

SPECS = {
    "MNQ": {
        "timeframe_minutes": 30,
        "tick_size": 0.25,
    },
    "MGC": {
        "timeframe_minutes": 60,
        "tick_size": 0.10,
    },
}

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
INDICATOR_COLUMNS = ["sma_50", "atr_14_wilder", "donchian_high_5", "donchian_low_5"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", choices=sorted(SPECS.keys()), required=True)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--dump-csv", type=Path, required=True)
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
    local = timestamp_utc.dt.tz_convert(SESSION_CHI)
    session_date = local.dt.floor("D")
    rollover_mask = local.dt.hour >= SESSION_START_HOUR_CHI
    session_date = session_date + pd.to_timedelta(rollover_mask.astype(np.int8), unit="D")
    return session_date.dt.strftime("%Y-%m-%d")


def _load_dump(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp_utc"] = pd.to_datetime(df["Timestamp_UTC"], utc=True)
    df = df.rename(
        columns={
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
    )
    for col in [*OHLCV_COLUMNS, *INDICATOR_COLUMNS]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["session_date"] = _compute_session_date(df["timestamp_utc"])
    return df[
        [
            "timestamp_utc",
            "session_date",
            *OHLCV_COLUMNS,
            *INDICATOR_COLUMNS,
        ]
    ].sort_values("timestamp_utc", kind="stable").reset_index(drop=True)


def _infer_anchor_offset_minutes(dump_df: pd.DataFrame, timeframe_minutes: int) -> int:
    phase = (dump_df["timestamp_utc"].dt.hour * 60 + dump_df["timestamp_utc"].dt.minute) % timeframe_minutes
    phase_counts = phase.value_counts(sort=True)
    if phase_counts.empty:
        raise ValueError("Unable to infer anchor offset from NT8 dump timestamps.")
    return int(phase_counts.index[0])


def _rebuild_from_raw(path: Path, timeframe_minutes: int, anchor_offset_minutes: int) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw["timestamp_utc"] = pd.to_datetime(raw["Timestamp_UTC"], utc=True)
    raw = raw.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    for col in OHLCV_COLUMNS:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw[["timestamp_utc", *OHLCV_COLUMNS]].dropna(subset=["open", "high", "low", "close"])
    raw = raw.sort_values("timestamp_utc", kind="stable").set_index("timestamp_utc")

    rebuilt = raw.resample(
        f"{timeframe_minutes}min",
        label="right",
        closed="right",
        origin="start_day",
        offset=f"{anchor_offset_minutes}min",
    ).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    rebuilt = rebuilt.dropna(subset=["open", "high", "low", "close"])
    rebuilt["timestamp_utc"] = rebuilt.index
    rebuilt = rebuilt.reset_index(drop=True)
    rebuilt["session_date"] = _compute_session_date(rebuilt["timestamp_utc"])
    return rebuilt[["timestamp_utc", "session_date", *OHLCV_COLUMNS]]


def _compute_indicators_from_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    highs = result["high"].to_numpy(dtype=np.float64)
    lows = result["low"].to_numpy(dtype=np.float64)
    closes = result["close"].to_numpy(dtype=np.float64)

    tr = np.maximum(highs - lows, np.zeros(len(result), dtype=np.float64))
    if len(result) > 1:
        prev_close = np.roll(closes, 1)
        prev_close[0] = closes[0]
        tr = np.maximum(tr, np.abs(highs - prev_close))
        tr = np.maximum(tr, np.abs(lows - prev_close))

    result["sma_50_py"] = result["close"].rolling(window=50, min_periods=50).mean()
    result["atr_14_wilder_py"] = _wilder_rma(tr, 14)
    result["donchian_high_5_py"] = result["high"].shift(1).rolling(window=5, min_periods=5).max()
    result["donchian_low_5_py"] = result["low"].shift(1).rolling(window=5, min_periods=5).min()
    return result


def _filter_window(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    mask = (df["session_date"] >= start_date) & (df["session_date"] <= end_date)
    return df.loc[mask].reset_index(drop=True)


def _build_ohlc_report(
    dump_df: pd.DataFrame,
    rebuilt_df: pd.DataFrame,
    tick_size: float,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = dump_df.merge(rebuilt_df, on="timestamp_utc", how="outer", suffixes=("_dump", "_rebuilt"), indicator=True)
    matched = merged.loc[merged["_merge"] == "both"].copy()

    for col in OHLCV_COLUMNS:
        matched[f"{col}_abs_delta"] = (matched[f"{col}_dump"] - matched[f"{col}_rebuilt"]).abs()
    for col in ["open", "high", "low", "close"]:
        matched[f"{col}_tick_delta"] = matched[f"{col}_abs_delta"] / tick_size

    dump_only = merged.loc[merged["_merge"] == "left_only", ["timestamp_utc"]].copy()
    rebuilt_only = merged.loc[merged["_merge"] == "right_only", ["timestamp_utc"]].copy()

    summary = {
        "dump_rows": int(len(dump_df)),
        "rebuilt_rows": int(len(rebuilt_df)),
        "matched_rows": int(len(matched)),
        "dump_only_timestamps": int(len(dump_only)),
        "rebuilt_only_timestamps": int(len(rebuilt_only)),
        "columns": {},
    }
    for col in OHLCV_COLUMNS:
        delta_col = f"{col}_abs_delta"
        finite = matched[delta_col].replace([np.inf, -np.inf], np.nan).dropna()
        entry = {
            "mean_abs_delta": float(finite.mean()) if len(finite) else 0.0,
            "max_abs_delta": float(finite.max()) if len(finite) else 0.0,
            "nonzero_delta_rows": int((finite > 0.0).sum()) if len(finite) else 0,
        }
        if col in {"open", "high", "low", "close"}:
            tick_delta_col = f"{col}_tick_delta"
            tick_finite = matched[tick_delta_col].replace([np.inf, -np.inf], np.nan).dropna()
            entry["max_tick_delta"] = float(tick_finite.max()) if len(tick_finite) else 0.0
        summary["columns"][col] = entry

    mismatches = matched.copy()
    mismatch_cols = [f"{col}_abs_delta" for col in OHLCV_COLUMNS]
    mismatches["max_abs_delta"] = mismatches[mismatch_cols].max(axis=1, skipna=True)
    mismatches = mismatches.sort_values("max_abs_delta", ascending=False, kind="stable")
    return summary, matched, dump_only, rebuilt_only, mismatches


def _build_indicator_report(dump_df_full: pd.DataFrame, start_date: str, end_date: str) -> tuple[dict, pd.DataFrame]:
    calculated = _compute_indicators_from_ohlc(dump_df_full)
    calculated = _filter_window(calculated, start_date, end_date)
    report_df = calculated[["timestamp_utc", "session_date", *OHLCV_COLUMNS, *INDICATOR_COLUMNS, "sma_50_py", "atr_14_wilder_py", "donchian_high_5_py", "donchian_low_5_py"]].copy()

    summary = {"matched_rows": int(len(report_df)), "columns": {}}
    column_map = {
        "sma_50": "sma_50_py",
        "atr_14_wilder": "atr_14_wilder_py",
        "donchian_high_5": "donchian_high_5_py",
        "donchian_low_5": "donchian_low_5_py",
    }
    for nt8_col, py_col in column_map.items():
        report_df[f"{nt8_col}_abs_delta"] = (report_df[nt8_col] - report_df[py_col]).abs()
        finite = report_df[f"{nt8_col}_abs_delta"].replace([np.inf, -np.inf], np.nan).dropna()
        summary["columns"][nt8_col] = {
            "mean_abs_delta": float(finite.mean()) if len(finite) else 0.0,
            "max_abs_delta": float(finite.max()) if len(finite) else 0.0,
            "nonzero_delta_rows": int((finite > 0.0).sum()) if len(finite) else 0,
            "missing_mismatch_rows": int((report_df[nt8_col].isna() ^ report_df[py_col].isna()).sum()),
        }

    report_df["max_abs_delta"] = report_df[[f"{col}_abs_delta" for col in column_map]].max(axis=1, skipna=True)
    report_df = report_df.sort_values("max_abs_delta", ascending=False, kind="stable")
    return summary, report_df


def _build_verdict(ohlc_summary: dict, indicator_summary: dict) -> str:
    ohlc_fail = any(ohlc_summary["columns"][col]["nonzero_delta_rows"] > 0 for col in OHLCV_COLUMNS)
    indicator_fail = any(indicator_summary["columns"][col]["nonzero_delta_rows"] > 0 or indicator_summary["columns"][col]["missing_mismatch_rows"] > 0 for col in INDICATOR_COLUMNS)
    if ohlc_fail:
        return "Bar-rebuild problem: aggregated OHLCV does not match NT8 dump."
    if indicator_fail:
        return "Indicator-math problem: Python formulas on NT8 truth bars do not match NT8 indicator outputs."
    return "Clean parity: OHLCV and indicator math both match NT8."


def main() -> None:
    args = parse_args()
    spec = SPECS[args.instrument]
    output_dir = args.output or Path("output/nt8_validation") / f"{args.instrument.lower()}_march_crucible_ohlc_forensics_150d"

    dump_full = _load_dump(args.dump_csv)
    anchor_offset_minutes = _infer_anchor_offset_minutes(dump_full, spec["timeframe_minutes"])
    rebuilt_full = _rebuild_from_raw(args.raw_csv, spec["timeframe_minutes"], anchor_offset_minutes)

    dump_window = _filter_window(dump_full, args.start_date, args.end_date)
    rebuilt_window = _filter_window(rebuilt_full, args.start_date, args.end_date)

    ohlc_summary, ohlc_matched, dump_only, rebuilt_only, ohlc_mismatches = _build_ohlc_report(
        dump_window,
        rebuilt_window,
        spec["tick_size"],
    )
    indicator_summary, indicator_details = _build_indicator_report(dump_full, args.start_date, args.end_date)

    summary = {
        "instrument": args.instrument,
        "raw_csv": str(args.raw_csv),
        "dump_csv": str(args.dump_csv),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "timeframe_minutes": spec["timeframe_minutes"],
        "tick_size": spec["tick_size"],
        "anchor_offset_minutes": anchor_offset_minutes,
        "ohlc_rebuild": ohlc_summary,
        "indicator_math_on_dump_truth": indicator_summary,
    }
    summary["verdict"] = _build_verdict(ohlc_summary, indicator_summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ohlc_matched.to_csv(output_dir / "ohlc_matched_deltas.csv", index=False)
    dump_only.to_csv(output_dir / "ohlc_dump_only_timestamps.csv", index=False)
    rebuilt_only.to_csv(output_dir / "ohlc_rebuilt_only_timestamps.csv", index=False)
    ohlc_mismatches.head(250).to_csv(output_dir / "ohlc_largest_mismatches.csv", index=False)
    indicator_details.head(250).to_csv(output_dir / "indicator_largest_mismatches.csv", index=False)

    print(f"Output: {output_dir}")
    print(f"Anchor offset minutes: {anchor_offset_minutes}")
    print(f"OHLC matched rows: {ohlc_summary['matched_rows']}")
    print(f"OHLC dump-only timestamps: {ohlc_summary['dump_only_timestamps']}")
    print(f"OHLC rebuilt-only timestamps: {ohlc_summary['rebuilt_only_timestamps']}")
    print(summary["verdict"])


if __name__ == "__main__":
    main()
