#!/usr/bin/env python
"""Forensic TradingView-vs-Python comparison for the local MNQ overlap window."""

from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


SESSION_TZ = "America/New_York"
DEFAULT_DATA = Path("data/processed/MNQ_1m_full_test.parquet")
OVERLAP_START = "2026-03-23"
OVERLAP_END = "2026-03-27"
ENTRY_START_MINUTE = 9 * 60
ENTRY_END_MINUTE = 11 * 60 + 30
TV_LOG = """TradeID;Type;Entry_Time_ET;Entry_Price;Exit_Time_ET;Exit_Price;Exit_Reason;PnL_USD
1;Long;2026-03-23 10:00:00;24590.50;2026-03-23 11:55:00;24407.75;StopLossLong;-365.50
2;Short;2026-03-24 10:00:00;24242.25;2026-03-24 15:59:00;24218.75;HardCloseShort;47.00
3;Long;2026-03-25 10:00:00;24342.00;2026-03-25 15:59:00;24372.25;HardCloseLong;60.50
4;Short;2026-03-26 10:00:00;24212.00;2026-03-26 15:59:00;23787.00;HardCloseShort;850.00
5;Short;2026-03-27 10:00:00;23506.25;2026-03-27 15:59:00;23308.75;HardCloseShort;395.00
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    return parser.parse_args()


def _load_tv_log() -> pd.DataFrame:
    tv = pd.read_csv(StringIO(TV_LOG), sep=";")
    tv["Entry_Time_ET"] = pd.to_datetime(tv["Entry_Time_ET"]).dt.tz_localize(SESSION_TZ)
    tv["Exit_Time_ET"] = pd.to_datetime(tv["Exit_Time_ET"]).dt.tz_localize(SESSION_TZ)
    return tv


def _load_complete_rth_1m(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).sort_index()
    df.index = df.index.tz_convert(SESSION_TZ)
    df = df.between_time("09:30", "15:59")[["open", "high", "low", "close", "volume"]]

    keep: list[pd.DataFrame] = []
    for session_date, session_df in df.groupby(df.index.date, sort=True):
        expected = pd.date_range(
            start=pd.Timestamp(session_date).tz_localize(SESSION_TZ) + pd.Timedelta(hours=9, minutes=30),
            periods=390,
            freq="min",
        )
        if len(session_df) == 390 and session_df.index.equals(expected):
            keep.append(session_df)
    if not keep:
        raise ValueError("No complete RTH sessions found in overlap data")
    return pd.concat(keep, axis=0)


def _build_signal_bars(df_1m: pd.DataFrame) -> pd.DataFrame:
    bars: list[pd.DataFrame] = []
    for session_date, session_df in df_1m.groupby(df_1m.index.date, sort=True):
        resampled = session_df.resample("30min", label="left", closed="left").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        expected = pd.date_range(
            start=pd.Timestamp(session_date).tz_localize(SESSION_TZ) + pd.Timedelta(hours=9, minutes=30),
            periods=13,
            freq="30min",
        )
        bars.append(resampled.loc[expected])
    return pd.concat(bars, axis=0)


def _compute_wilder_rma(values: pd.Series, length: int) -> pd.Series:
    arr = values.to_numpy(dtype=np.float64)
    out = np.full(len(arr), np.nan, dtype=np.float64)
    if len(arr) < length:
        return pd.Series(out, index=values.index)
    out[length - 1] = float(np.mean(arr[:length]))
    for idx in range(length, len(arr)):
        out[idx] = (out[idx - 1] * (length - 1) + arr[idx]) / length
    return pd.Series(out, index=values.index)


def _simulate(df_1m: pd.DataFrame, signal_bars: pd.DataFrame) -> pd.DataFrame:
    tr = pd.concat(
        [
            signal_bars["high"] - signal_bars["low"],
            (signal_bars["high"] - signal_bars["close"].shift(1)).abs(),
            (signal_bars["low"] - signal_bars["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    if len(tr) > 0:
        tr.iloc[0] = signal_bars["high"].iloc[0] - signal_bars["low"].iloc[0]

    atr_rma_14 = _compute_wilder_rma(tr, 14)
    sma_50 = signal_bars["close"].rolling(window=50, min_periods=50).mean()
    donchian_high_5 = signal_bars["high"].shift(1).rolling(window=5, min_periods=5).max()
    donchian_low_5 = signal_bars["low"].shift(1).rolling(window=5, min_periods=5).min()

    rows: list[dict] = []
    for session_date, bars in signal_bars.groupby(signal_bars.index.date, sort=True):
        day_1m = df_1m.loc[
            pd.Timestamp(session_date).tz_localize(SESSION_TZ) + pd.Timedelta(hours=9, minutes=30):
            pd.Timestamp(session_date).tz_localize(SESSION_TZ) + pd.Timedelta(hours=15, minutes=59)
        ]

        for ts, row in bars.iterrows():
            entry_time = ts + pd.Timedelta(minutes=30)
            entry_minutes = entry_time.hour * 60 + entry_time.minute
            if entry_minutes < ENTRY_START_MINUTE or entry_minutes > ENTRY_END_MINUTE:
                continue
            if entry_time not in day_1m.index:
                continue

            atr = float(atr_rma_14.loc[ts])
            sma = float(sma_50.loc[ts])
            channel_high = float(donchian_high_5.loc[ts])
            channel_low = float(donchian_low_5.loc[ts])
            signal_close = float(row["close"])
            if not np.isfinite(atr) or not np.isfinite(sma) or not np.isfinite(channel_high) or not np.isfinite(channel_low):
                continue

            direction = 0
            if signal_close > channel_high and signal_close > sma:
                direction = 1
            elif signal_close < channel_low and signal_close < sma:
                direction = -1
            if direction == 0:
                continue

            entry_price = float(day_1m.loc[entry_time, "close"])
            stop_level = entry_price - 1.5 * atr if direction > 0 else entry_price + 1.5 * atr
            target_level = entry_price + 10.0 * atr if direction > 0 else entry_price - 10.0 * atr

            exit_time = pd.NaT
            exit_price = np.nan
            exit_reason = ""
            after_entry = day_1m.loc[entry_time + pd.Timedelta(minutes=1):]
            for bar_ts, bar in after_entry.iterrows():
                bar_open = float(bar["open"])
                bar_high = float(bar["high"])
                bar_low = float(bar["low"])
                bar_close = float(bar["close"])

                if direction > 0:
                    if bar_open <= stop_level:
                        exit_time = bar_ts
                        exit_price = bar_open
                        exit_reason = "StopGapOpenLong"
                        break
                    if bar_low <= stop_level:
                        exit_time = bar_ts
                        exit_price = stop_level
                        exit_reason = "StopLossLong"
                        break
                    if bar_open >= target_level:
                        exit_time = bar_ts
                        exit_price = bar_open
                        exit_reason = "TargetGapOpenLong"
                        break
                    if bar_high >= target_level:
                        exit_time = bar_ts
                        exit_price = target_level
                        exit_reason = "ProfitTargetLong"
                        break
                else:
                    if bar_open >= stop_level:
                        exit_time = bar_ts
                        exit_price = bar_open
                        exit_reason = "StopGapOpenShort"
                        break
                    if bar_high >= stop_level:
                        exit_time = bar_ts
                        exit_price = stop_level
                        exit_reason = "StopLossShort"
                        break
                    if bar_open <= target_level:
                        exit_time = bar_ts
                        exit_price = bar_open
                        exit_reason = "TargetGapOpenShort"
                        break
                    if bar_low <= target_level:
                        exit_time = bar_ts
                        exit_price = target_level
                        exit_reason = "ProfitTargetShort"
                        break

                if bar_ts.hour == 15 and bar_ts.minute == 59:
                    exit_time = bar_ts
                    exit_price = bar_close
                    exit_reason = "HardCloseLong" if direction > 0 else "HardCloseShort"
                    break

            pnl_usd = (exit_price - entry_price) * 2.0 if direction > 0 else (entry_price - exit_price) * 2.0
            rows.append(
                {
                    "Entry_Time_ET": entry_time,
                    "Type": "Long" if direction > 0 else "Short",
                    "Signal_Time_ET": ts,
                    "Signal_Close": signal_close,
                    "ATR_RMA14": atr,
                    "SMA50": sma,
                    "Donchian_High_5": channel_high,
                    "Donchian_Low_5": channel_low,
                    "Entry_Price": entry_price,
                    "Stop_Level": stop_level,
                    "Target_Level": target_level,
                    "Exit_Time_ET": exit_time,
                    "Exit_Price": exit_price,
                    "Exit_Reason": exit_reason,
                    "PnL_USD": pnl_usd,
                }
            )
            break

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.loc[
        (result["Entry_Time_ET"] >= pd.Timestamp(OVERLAP_START, tz=SESSION_TZ))
        & (result["Entry_Time_ET"] < pd.Timestamp("2026-03-28", tz=SESSION_TZ))
    ].reset_index(drop=True)


def _build_report(tv: pd.DataFrame, sim: pd.DataFrame) -> pd.DataFrame:
    merged = tv.merge(sim, on="Entry_Time_ET", how="left", suffixes=("_tv", "_py"))
    merged["entry_price_delta"] = merged["Entry_Price_py"] - merged["Entry_Price_tv"]
    merged["exit_price_delta"] = merged["Exit_Price_py"] - merged["Exit_Price_tv"]
    merged["exit_time_delta_minutes"] = (
        (merged["Exit_Time_ET_py"] - merged["Exit_Time_ET_tv"]).dt.total_seconds() / 60.0
    )
    merged["pnl_delta"] = merged["PnL_USD_py"] - merged["PnL_USD_tv"]
    merged["tv_inferred_atr"] = np.where(
        merged["Exit_Reason_tv"].str.contains("StopLoss"),
        (merged["Entry_Price_tv"] - merged["Exit_Price_tv"]).abs() / 1.5,
        np.nan,
    )
    merged["atr_delta"] = merged["ATR_RMA14"] - merged["tv_inferred_atr"]
    return merged


def main() -> None:
    args = parse_args()
    tv = _load_tv_log()
    df_1m = _load_complete_rth_1m(args.data)
    signal_bars = _build_signal_bars(df_1m)
    sim = _simulate(df_1m, signal_bars)
    report = _build_report(tv, sim)

    print("=" * 88)
    print("MNQ TV FORENSIC OVERLAP REPORT")
    print("=" * 88)
    print(f"Data: {args.data}")
    print(f"Window: {OVERLAP_START} to {OVERLAP_END}")
    print(f"Simulated trades: {len(sim)}")
    print()
    print(
        report[
            [
                "TradeID",
                "Type_tv",
                "Entry_Time_ET",
                "Entry_Price_tv",
                "Entry_Price_py",
                "entry_price_delta",
                "Exit_Time_ET_tv",
                "Exit_Time_ET_py",
                "exit_time_delta_minutes",
                "Exit_Price_tv",
                "Exit_Price_py",
                "exit_price_delta",
                "Exit_Reason_tv",
                "Exit_Reason_py",
                "ATR_RMA14",
                "tv_inferred_atr",
                "atr_delta",
                "PnL_USD_tv",
                "PnL_USD_py",
                "pnl_delta",
            ]
        ].to_string(index=False)
    )

    trade1 = report.loc[report["TradeID"] == 1].iloc[0]
    print()
    print("Trade 1 focus")
    print(f"  TV stop exit:        {trade1['Exit_Price_tv']:.2f} @ {trade1['Exit_Time_ET_tv']}")
    print(f"  Python stop exit:    {trade1['Exit_Price_py']:.6f} @ {trade1['Exit_Time_ET_py']}")
    print(f"  TV implied ATR14:    {trade1['tv_inferred_atr']:.6f}")
    print(f"  Python ATR RMA14:    {trade1['ATR_RMA14']:.6f}")
    print(f"  ATR delta:           {trade1['atr_delta']:.6f}")
    print(f"  Exit price delta:    {trade1['exit_price_delta']:.6f}")


if __name__ == "__main__":
    main()
