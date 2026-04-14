#!/usr/bin/env python
"""Explore the MGC overlap train split on 5-minute bars."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from propfirm.market.data_loader import SESSION_TZ, load_session_data


TICK_SIZE = 0.10


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    tr = np.maximum(high - low, 0.0)
    if len(close) <= 1:
        return tr
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(tr, np.abs(high - prev_close))
    tr = np.maximum(tr, np.abs(low - prev_close))
    return tr


def _rolling_atr(tr: np.ndarray, period: int) -> np.ndarray:
    if len(tr) == 0:
        return np.array([], dtype=np.float64)
    atr = np.full(len(tr), np.nan, dtype=np.float64)
    if len(tr) < period:
        return atr
    kernel = np.ones(period, dtype=np.float64) / float(period)
    atr[period - 1:] = np.convolve(tr, kernel, mode="valid")
    return atr


def _validate_session_coverage(parquet_path: Path, session_start: str, session_end: str) -> None:
    df = pd.read_parquet(parquet_path, columns=["open"])
    index = df.index.tz_convert(SESSION_TZ)
    observed_start = index.strftime("%H:%M").min()
    observed_end = index.strftime("%H:%M").max()
    if observed_start > session_start or observed_end < session_end:
        raise ValueError(
            f"Requested {session_start}-{session_end} ET, but {parquet_path.name} only covers "
            f"{observed_start}-{observed_end} ET."
        )


def _session_index(session_dates: np.ndarray, minute_of_day: np.ndarray, session_start: str) -> pd.DatetimeIndex:
    start_hour, start_minute = map(int, session_start.split(":"))
    base = pd.to_datetime(session_dates).tz_localize(SESSION_TZ)
    open_offset = pd.to_timedelta((start_hour * 60) + start_minute, unit="m")
    bar_offset = pd.to_timedelta(minute_of_day.astype(np.int32), unit="m")
    return pd.DatetimeIndex(base + open_offset + bar_offset)


def _build_frame(data: dict, session_start: str) -> pd.DataFrame:
    session_dates = np.empty(len(data["open"]), dtype=object)
    for day_idx, (start, end) in enumerate(data["day_boundaries"]):
        session_dates[start:end] = data["session_dates"][day_idx]
    index = _session_index(session_dates, data["minute_of_day"], session_start)
    return pd.DataFrame(
        {
            "open": data["open"],
            "high": data["high"],
            "low": data["low"],
            "close": data["close"],
            "session_date": session_dates,
        },
        index=index,
    )


def explore(parquet_path: Path, atr_period: int, session_start: str, session_end: str) -> dict:
    _validate_session_coverage(parquet_path, session_start, session_end)
    data = load_session_data(
        parquet_path,
        atr_period=atr_period,
        trailing_atr_days=5,
        timeframe_minutes=5,
        session_start=session_start,
        session_end=session_end,
    )
    frame = _build_frame(data, session_start=session_start)

    tr = _true_range(frame["high"].to_numpy(), frame["low"].to_numpy(), frame["close"].to_numpy())
    atr = _rolling_atr(tr, atr_period)
    avg_atr_ticks = float(np.nanmean(atr) / TICK_SIZE)

    frame["range_ticks"] = (frame["high"] - frame["low"]) / TICK_SIZE
    frame["clock_time"] = frame.index.strftime("%H:%M")
    by_time = frame.groupby("clock_time", sort=True)["range_ticks"].mean().sort_values(ascending=False)
    hotspot_mean = float(by_time.mean())
    hotspot_std = float(by_time.std(ddof=0))
    hotspot_df = by_time.reset_index(name="avg_range_ticks")
    hotspot_df["zscore"] = (hotspot_df["avg_range_ticks"] - hotspot_mean) / hotspot_std if hotspot_std > 0 else 0.0

    next_open = frame["open"].shift(-1)
    next_close = frame["close"].shift(-1)
    same_session_next = frame["session_date"] == frame["session_date"].shift(-1)
    prior_session_high = frame.groupby("session_date")["high"].cummax().shift(1)
    same_session_prior = frame["session_date"] == frame["session_date"].shift(1)
    new_session_high = same_session_prior & (frame["high"] > prior_session_high)
    event_mask = new_session_high & same_session_next & next_open.notna() & next_close.notna()
    next_bar_delta = next_close[event_mask] - next_open[event_mask]
    positive_count = int((next_bar_delta > 0).sum())
    negative_count = int((next_bar_delta < 0).sum())
    flat_count = int((next_bar_delta == 0).sum())
    total_events = int(len(next_bar_delta))

    return {
        "average_5m_atr_ticks": avg_atr_ticks,
        "volatility_hotspots": {
            "top_3_times": hotspot_df.head(3).to_dict(orient="records"),
        },
        "new_session_high_follow_through": {
            "events": total_events,
            "positive_next_bar_probability": (positive_count / total_events) if total_events else 0.0,
            "negative_next_bar_probability": (negative_count / total_events) if total_events else 0.0,
            "flat_next_bar_probability": (flat_count / total_events) if total_events else 0.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/MGC_1m_train.parquet"))
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--session-start", type=str, default="08:00")
    parser.add_argument("--session-end", type=str, default="11:30")
    args = parser.parse_args()

    result = explore(
        parquet_path=args.data,
        atr_period=args.atr_period,
        session_start=args.session_start,
        session_end=args.session_end,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
