#!/usr/bin/env python
"""Validate Python signal parity against NT8-native chart truth.

This script compares:
1. NT8 baseline decisions computed from the verified DataDumpExporter bars
2. Python engine decisions computed from rebuilt higher-timeframe bars from the NT8 raw 1m dump
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from propfirm.core.types import (
    PARAMS_ARRAY_LENGTH,
    PARAMS_BLOCKED_WEEKDAY,
    PARAMS_DAILY_TARGET,
    PARAMS_MAX_TRADES,
    PARAMS_TRIGGER_END_MINUTE,
    PARAMS_TRIGGER_START_MINUTE,
    SIGNAL_LONG,
    SIGNAL_SHORT,
)
from propfirm.strategy.mgc_h1_trend_strategy import mgc_h1_trend_signal


SESSION_ET = "America/New_York"
SESSION_CHI = "America/Chicago"
SESSION_START_HOUR_CHI = 17
RESEARCH_SESSION_START_MINUTE_ET = 8 * 60
RESEARCH_SESSION_END_MINUTE_ET = 15 * 60 + 59
DEFAULT_START = "2026-02-23"
DEFAULT_END = "2026-03-27"

SPECS = {
    "MNQ": {
        "timeframe_minutes": 30,
        "raw_csv": Path("data/ninjatrader/NT8_RawDump_MNQ.csv"),
        "dump_csv": Path("data/ninjatrader/NT8_DumpMNQ.csv"),
        "trigger_start_minute": 150,
        "trigger_end_minute": 180,
        "tick_size": 0.25,
        "stop_atr_multiplier": 1.5,
        "target_atr_multiplier": 10.0,
    },
    "MGC": {
        "timeframe_minutes": 60,
        "raw_csv": Path("data/ninjatrader/NT8_RawDump_MGC.csv"),
        "dump_csv": Path("data/ninjatrader/NT8_DumpMGC.csv"),
        "trigger_start_minute": 60,
        "trigger_end_minute": 180,
        "tick_size": 0.10,
        "stop_atr_multiplier": 1.5,
        "target_atr_multiplier": 10.0,
    },
}

STRUCTURE_COLUMNS = ["sma_50", "atr_14_wilder", "donchian_high_5", "donchian_low_5", "daily_regime_bias"]
EVENT_COLUMNS = [
    "session_date",
    "signal_timestamp_utc",
    "entry_timestamp_utc",
    "direction",
    "entry_price",
    "stop_ticks",
    "target_ticks",
    "stop_price",
    "target_price",
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
    local = timestamp_utc.dt.tz_convert(SESSION_CHI)
    session_date = local.dt.floor("D")
    rollover_mask = local.dt.hour >= SESSION_START_HOUR_CHI
    session_date = session_date + pd.to_timedelta(rollover_mask.astype(np.int8), unit="D")
    return session_date.dt.strftime("%Y-%m-%d")


def _compute_regime_bias(df: pd.DataFrame) -> pd.Series:
    ts_et = df["timestamp_utc"].dt.tz_convert(SESSION_ET)
    total_minutes = ts_et.dt.hour * 60 + ts_et.dt.minute
    in_research = (total_minutes >= RESEARCH_SESSION_START_MINUTE_ET) & (total_minutes <= RESEARCH_SESSION_END_MINUTE_ET)

    research = df.loc[in_research, ["timestamp_utc", "close"]].copy()
    research["session_date_et"] = research["timestamp_utc"].dt.tz_convert(SESSION_ET).dt.strftime("%Y-%m-%d")
    session_close = research.groupby("session_date_et", sort=True)["close"].last()
    session_sma = session_close.rolling(window=50, min_periods=50).mean()

    bias_by_session: dict[str, float] = {}
    dates = list(session_close.index)
    for idx, session_date in enumerate(dates):
        if idx == 0:
            bias_by_session[session_date] = 0.0
            continue
        prev_date = dates[idx - 1]
        prev_close = float(session_close.iloc[idx - 1])
        prev_sma = float(session_sma.iloc[idx - 1]) if np.isfinite(session_sma.iloc[idx - 1]) else np.nan
        if not np.isfinite(prev_sma):
            bias_by_session[session_date] = 0.0
        elif prev_close > prev_sma:
            bias_by_session[session_date] = 1.0
        elif prev_close < prev_sma:
            bias_by_session[session_date] = -1.0
        else:
            bias_by_session[session_date] = 0.0

    full_dates = ts_et.dt.strftime("%Y-%m-%d")
    return full_dates.map(bias_by_session).fillna(0.0).astype(np.float64)


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
    for col in ["open", "high", "low", "close", "volume", "sma_50", "atr_14_wilder", "donchian_high_5", "donchian_low_5"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["session_date"] = _compute_session_date(df["timestamp_utc"])
    df["daily_regime_bias"] = _compute_regime_bias(df)
    return df[
        [
            "timestamp_utc",
            "session_date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "sma_50",
            "atr_14_wilder",
            "donchian_high_5",
            "donchian_low_5",
            "daily_regime_bias",
        ]
    ].sort_values("timestamp_utc", kind="stable").reset_index(drop=True)


def _infer_anchor_offset_minutes(dump_df: pd.DataFrame, timeframe_minutes: int) -> int:
    if dump_df.empty:
        raise ValueError("Cannot infer anchor from empty NT8 dump.")

    phase = (dump_df["timestamp_utc"].dt.hour * 60 + dump_df["timestamp_utc"].dt.minute) % timeframe_minutes
    phase_counts = phase.value_counts(sort=True)
    if phase_counts.empty:
        raise ValueError("Unable to infer anchor offset from NT8 dump timestamps.")
    return int(phase_counts.index[0])


def _rebuild_from_raw(path: Path, timeframe_minutes: int, anchor_offset_minutes: int) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw["timestamp_utc"] = pd.to_datetime(raw["Timestamp_UTC"], utc=True)
    raw = raw.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    for col in ["open", "high", "low", "close", "volume"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw[["timestamp_utc", "open", "high", "low", "close", "volume"]].dropna(subset=["open", "high", "low", "close"])
    raw = raw.sort_values("timestamp_utc", kind="stable")
    raw = raw.set_index("timestamp_utc")

    rebuilt = raw.resample(
        f"{timeframe_minutes}min",
        label="right",
        closed="right",
        origin="start_day",
        offset=f"{anchor_offset_minutes}min",
    ).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
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
    rebuilt["timestamp_utc"] = rebuilt.index
    rebuilt = rebuilt.reset_index(drop=True)
    rebuilt["session_date"] = _compute_session_date(rebuilt["timestamp_utc"])
    rebuilt["daily_regime_bias"] = _compute_regime_bias(rebuilt).to_numpy(dtype=np.float64)

    return rebuilt[
        [
            "timestamp_utc",
            "session_date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "sma_50",
            "atr_14_wilder",
            "donchian_high_5",
            "donchian_low_5",
            "daily_regime_bias",
        ]
    ]


def _filter_window(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    mask = (df["session_date"] >= start_date) & (df["session_date"] <= end_date)
    return df.loc[mask].reset_index(drop=True)


def _minute_of_day_et(timestamp_utc: pd.Series) -> np.ndarray:
    ts_et = timestamp_utc.dt.tz_convert(SESSION_ET)
    return ((ts_et.dt.hour * 60 + ts_et.dt.minute) - RESEARCH_SESSION_START_MINUTE_ET).to_numpy(dtype=np.int16)


def _day_of_week_et(timestamp_utc: pd.Series) -> np.ndarray:
    ts_et = timestamp_utc.dt.tz_convert(SESSION_ET)
    return ts_et.dt.weekday.to_numpy(dtype=np.int8)


def _is_inside_research_session(timestamp_utc: pd.Timestamp) -> bool:
    ts_et = timestamp_utc.tz_convert(SESSION_ET)
    total_minutes = ts_et.hour * 60 + ts_et.minute
    return RESEARCH_SESSION_START_MINUTE_ET <= total_minutes <= RESEARCH_SESSION_END_MINUTE_ET


def _build_params(spec: dict) -> np.ndarray:
    params = np.zeros(PARAMS_ARRAY_LENGTH, dtype=np.float64)
    params[PARAMS_BLOCKED_WEEKDAY] = -1
    params[PARAMS_DAILY_TARGET] = 1e12
    params[PARAMS_MAX_TRADES] = 1
    params[PARAMS_TRIGGER_START_MINUTE] = spec["trigger_start_minute"]
    params[PARAMS_TRIGGER_END_MINUTE] = spec["trigger_end_minute"]
    return params


def _build_nt8_baseline_events(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    events: list[dict] = []
    daily_trade_count = 0
    current_session_date = None

    for idx, row in df.iterrows():
        ts = row["timestamp_utc"]
        ts_et = ts.tz_convert(SESSION_ET)
        session_date = ts_et.strftime("%Y-%m-%d")

        if _is_inside_research_session(ts) and session_date != current_session_date:
            current_session_date = session_date
            daily_trade_count = 0

        if not _is_inside_research_session(ts):
            continue
        if daily_trade_count >= 1:
            continue

        hhmmss = ts_et.hour * 10000 + ts_et.minute * 100 + ts_et.second
        start_hhmm = 103000 if spec["timeframe_minutes"] == 30 else 90000
        end_hhmm = 110000
        if hhmmss < start_hhmm or hhmmss > end_hhmm:
            continue

        sma = row["sma_50"]
        atr = row["atr_14_wilder"]
        d_high = row["donchian_high_5"]
        d_low = row["donchian_low_5"]
        regime = row["daily_regime_bias"]
        close = row["close"]
        if not np.isfinite(sma) or not np.isfinite(atr) or not np.isfinite(d_high) or not np.isfinite(d_low):
            continue

        direction = 0
        if close > d_high and close > sma and regime > 0.0:
            direction = 1
        elif close < d_low and close < sma and regime < 0.0:
            direction = -1
        if direction == 0 or idx + 1 >= len(df):
            continue

        next_row = df.iloc[idx + 1]
        entry_price = float(next_row["open"])
        stop_distance = float(atr * spec["stop_atr_multiplier"])
        target_distance = float(atr * spec["target_atr_multiplier"])
        stop_ticks = max(1.0, stop_distance / spec["tick_size"])
        target_ticks = max(1.0, target_distance / spec["tick_size"])

        events.append(
            {
                "session_date": session_date,
                "signal_timestamp_utc": ts,
                "entry_timestamp_utc": next_row["timestamp_utc"],
                "direction": direction,
                "entry_price": entry_price,
                "stop_ticks": stop_ticks,
                "target_ticks": target_ticks,
                "stop_price": entry_price - stop_distance if direction == 1 else entry_price + stop_distance,
                "target_price": entry_price + target_distance if direction == 1 else entry_price - target_distance,
            }
        )
        daily_trade_count = 1

    return pd.DataFrame(events, columns=EVENT_COLUMNS)


def _build_python_engine_events(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    params = _build_params(spec)
    minute_of_day = _minute_of_day_et(df["timestamp_utc"])
    day_of_week = _day_of_week_et(df["timestamp_utc"])

    opens = df["open"].to_numpy(dtype=np.float64)
    highs = df["high"].to_numpy(dtype=np.float64)
    lows = df["low"].to_numpy(dtype=np.float64)
    closes = df["close"].to_numpy(dtype=np.float64)
    volumes = df["volume"].to_numpy(dtype=np.float64)
    atr = df["atr_14_wilder"].to_numpy(dtype=np.float64)
    sma = df["sma_50"].to_numpy(dtype=np.float64)
    regime = df["daily_regime_bias"].to_numpy(dtype=np.float64)
    d_high = df["donchian_high_5"].to_numpy(dtype=np.float64)
    d_low = df["donchian_low_5"].to_numpy(dtype=np.float64)

    events: list[dict] = []
    daily_trade_count = 0
    current_session_date = None

    for idx in range(len(df)):
        ts = df.iloc[idx]["timestamp_utc"]
        ts_et = ts.tz_convert(SESSION_ET)
        session_date = ts_et.strftime("%Y-%m-%d")

        if _is_inside_research_session(ts) and session_date != current_session_date:
            current_session_date = session_date
            daily_trade_count = 0

        signal = mgc_h1_trend_signal(
            idx,
            opens,
            highs,
            lows,
            closes,
            volumes,
            atr,
            np.zeros(len(df), dtype=np.float64),
            np.zeros(len(df), dtype=np.float64),
            np.zeros(len(df), dtype=np.float64),
            sma,
            regime,
            d_high,
            d_low,
            minute_of_day,
            day_of_week,
            0.0,
            0.0,
            0,
            0.0,
            False,
            daily_trade_count,
            params,
        )

        if signal not in (SIGNAL_LONG, SIGNAL_SHORT) or idx + 1 >= len(df):
            continue

        next_row = df.iloc[idx + 1]
        entry_price = float(next_row["open"])
        stop_distance = float(atr[idx] * spec["stop_atr_multiplier"])
        target_distance = float(atr[idx] * spec["target_atr_multiplier"])
        stop_ticks = max(1.0, stop_distance / spec["tick_size"])
        target_ticks = max(1.0, target_distance / spec["tick_size"])

        events.append(
            {
                "session_date": session_date,
                "signal_timestamp_utc": ts,
                "entry_timestamp_utc": next_row["timestamp_utc"],
                "direction": 1 if signal == SIGNAL_LONG else -1,
                "entry_price": entry_price,
                "stop_ticks": stop_ticks,
                "target_ticks": target_ticks,
                "stop_price": entry_price - stop_distance if signal == SIGNAL_LONG else entry_price + stop_distance,
                "target_price": entry_price + target_distance if signal == SIGNAL_LONG else entry_price - target_distance,
            }
        )
        daily_trade_count = 1

    return pd.DataFrame(events, columns=EVENT_COLUMNS)


def _compare_structure(dump_df: pd.DataFrame, rebuilt_df: pd.DataFrame) -> dict:
    merged = dump_df.merge(rebuilt_df, on="timestamp_utc", how="inner", suffixes=("_nt8", "_py"))
    report: dict[str, dict[str, float | int]] = {}
    for col in STRUCTURE_COLUMNS:
        nt8_values = merged[f"{col}_nt8"]
        py_values = merged[f"{col}_py"]
        nt8_missing = nt8_values.isna()
        py_missing = py_values.isna()
        missing_mismatch = nt8_missing ^ py_missing
        delta = (nt8_values - py_values).abs()
        finite = delta.replace([np.inf, -np.inf], np.nan).dropna()
        report[col] = {
            "matched_rows": int(len(merged)),
            "mean_abs_delta": float(finite.mean()) if len(finite) else 0.0,
            "max_abs_delta": float(finite.max()) if len(finite) else 0.0,
            "nonzero_delta_rows": int((finite > 0.0).sum()) if len(finite) else 0,
            "missing_mismatch_rows": int(missing_mismatch.sum()),
        }
    return report


def _compare_events(nt8_events: pd.DataFrame, py_events: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    if len(nt8_events) == 0 and len(py_events) == 0:
        return {
            "nt8_event_count": 0,
            "python_event_count": 0,
            "exact_signal_timestamp_matches": 0,
            "exact_entry_timestamp_matches": 0,
            "exact_direction_matches": 0,
            "exact_structural_matches": 0,
            "nt8_only_events": 0,
            "python_only_events": 0,
        }, pd.DataFrame()

    merged = nt8_events.merge(
        py_events,
        on=["session_date", "direction"],
        how="outer",
        suffixes=("_nt8", "_py"),
        indicator=True,
    ).sort_values(["session_date", "direction"], kind="stable")

    both = merged.loc[merged["_merge"] == "both"].copy()
    if len(both):
        both["signal_timestamp_match"] = both["signal_timestamp_utc_nt8"] == both["signal_timestamp_utc_py"]
        both["entry_timestamp_match"] = both["entry_timestamp_utc_nt8"] == both["entry_timestamp_utc_py"]
        both["entry_price_abs_delta"] = (both["entry_price_nt8"] - both["entry_price_py"]).abs()
        both["stop_ticks_abs_delta"] = (both["stop_ticks_nt8"] - both["stop_ticks_py"]).abs()
        both["target_ticks_abs_delta"] = (both["target_ticks_nt8"] - both["target_ticks_py"]).abs()
        both["exact_structural_match"] = (
            both["signal_timestamp_match"]
            & both["entry_timestamp_match"]
            & (both["entry_price_abs_delta"] == 0.0)
            & (both["stop_ticks_abs_delta"] == 0.0)
            & (both["target_ticks_abs_delta"] == 0.0)
        )
    else:
        both["signal_timestamp_match"] = pd.Series(dtype=bool)
        both["entry_timestamp_match"] = pd.Series(dtype=bool)
        both["entry_price_abs_delta"] = pd.Series(dtype=float)
        both["stop_ticks_abs_delta"] = pd.Series(dtype=float)
        both["target_ticks_abs_delta"] = pd.Series(dtype=float)
        both["exact_structural_match"] = pd.Series(dtype=bool)

    summary = {
        "nt8_event_count": int(len(nt8_events)),
        "python_event_count": int(len(py_events)),
        "exact_signal_timestamp_matches": int(both["signal_timestamp_match"].sum()) if len(both) else 0,
        "exact_entry_timestamp_matches": int(both["entry_timestamp_match"].sum()) if len(both) else 0,
        "exact_direction_matches": int(len(both)),
        "exact_structural_matches": int(both["exact_structural_match"].sum()) if len(both) else 0,
        "nt8_only_events": int((merged["_merge"] == "left_only").sum()),
        "python_only_events": int((merged["_merge"] == "right_only").sum()),
    }
    return summary, both


def main() -> None:
    args = parse_args()
    spec = SPECS[args.instrument]
    raw_csv = args.raw_csv or spec["raw_csv"]
    dump_csv = args.dump_csv or spec["dump_csv"]
    output_dir = args.output or Path("output/nt8_validation") / f"{args.instrument.lower()}_march_crucible_signal_parity"

    full_dump_df = _load_dump(dump_csv)
    anchor_offset_minutes = _infer_anchor_offset_minutes(full_dump_df, spec["timeframe_minutes"])
    dump_df = _filter_window(full_dump_df, args.start_date, args.end_date)
    rebuilt_df = _filter_window(_rebuild_from_raw(raw_csv, spec["timeframe_minutes"], anchor_offset_minutes), args.start_date, args.end_date)

    nt8_events = _build_nt8_baseline_events(dump_df, spec)
    py_events = _build_python_engine_events(rebuilt_df, spec)
    structure = _compare_structure(dump_df, rebuilt_df)
    decision_summary, decision_details = _compare_events(nt8_events, py_events)

    summary = {
        "instrument": args.instrument,
        "raw_csv": str(raw_csv),
        "dump_csv": str(dump_csv),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "anchor_offset_minutes": anchor_offset_minutes,
        "structure": structure,
        "decision": decision_summary,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    nt8_events.to_csv(output_dir / "nt8_baseline_events.csv", index=False)
    py_events.to_csv(output_dir / "python_engine_events.csv", index=False)
    decision_details.to_csv(output_dir / "decision_matches.csv", index=False)

    print(f"Output: {output_dir}")
    print(f"NT8 events: {decision_summary['nt8_event_count']}")
    print(f"Python events: {decision_summary['python_event_count']}")
    print(f"Exact signal timestamp matches: {decision_summary['exact_signal_timestamp_matches']}")
    print(f"Exact entry timestamp matches: {decision_summary['exact_entry_timestamp_matches']}")
    print(f"Exact structural matches: {decision_summary['exact_structural_matches']}")
    print(f"NT8-only events: {decision_summary['nt8_only_events']}")
    print(f"Python-only events: {decision_summary['python_only_events']}")


if __name__ == "__main__":
    main()
