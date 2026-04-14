#!/usr/bin/env python
"""Tick-resolution Renko ORB backtest for MGC."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numba import njit

from propfirm.core.types import (
    EXIT_HARD_CLOSE,
    EXIT_STOP,
    EXIT_TARGET,
    SIGNAL_LONG,
    SIGNAL_SHORT,
    TRADE_LOG_DTYPE,
)
from propfirm.market.data_loader import SESSION_TZ


MGC_TICK_SIZE = 0.10
MGC_TICK_VALUE = 1.00
MGC_COMMISSION_PER_SIDE = 0.54
SLIPPAGE_TICKS = 1.22
SLIPPAGE_POINTS = SLIPPAGE_TICKS * MGC_TICK_SIZE
BRICK_SIZE = 0.30
STOP_DISTANCE = 0.90
TARGET_DISTANCE = 1.80
SESSION_OPEN_MINUTE = 9 * 60 + 30
SESSION_CLOSE_MINUTE = 16 * 60
DEFAULT_TICK_DATA = Path("data/raw/MGC_mbp1_6m_raw.parquet")
DEFAULT_OUTPUT = Path("output/tick_renko_orb_mgc_trades.npy")


def _schema_columns(parquet_path: Path) -> set[str]:
    return set(pq.read_schema(parquet_path).names)


def load_mgc_ticks(
    parquet_path: str | Path,
    timezone: str = SESSION_TZ,
) -> dict[str, np.ndarray | str]:
    """Load MGC tick data as UTC nanoseconds plus a tradeable mid-price stream."""
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Tick parquet not found: {parquet_path}")

    columns = _schema_columns(parquet_path)
    quote_schema = {"bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"}.issubset(columns)
    trade_schema = {"price", "size"}.issubset(columns)

    if quote_schema:
        frame = pd.read_parquet(
            parquet_path,
            columns=["ts_event", "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"],
            engine="pyarrow",
        )
        bid = frame["bid_px_00"].to_numpy(dtype=np.float64, copy=False)
        ask = frame["ask_px_00"].to_numpy(dtype=np.float64, copy=False)
        bid_sz = frame["bid_sz_00"].fillna(0).to_numpy(dtype=np.int32, copy=False)
        ask_sz = frame["ask_sz_00"].fillna(0).to_numpy(dtype=np.int32, copy=False)
        valid = np.isfinite(bid) & np.isfinite(ask) & (bid > 0.0) & (ask > 0.0)
        prices = ((bid + ask) * 0.5)[valid].astype(np.float64, copy=False)
        volumes = (bid_sz + ask_sz)[valid].astype(np.int32, copy=False)
        ts_event = frame.loc[valid, "ts_event"]
        source_type = "midquote"
    elif trade_schema:
        frame = pd.read_parquet(
            parquet_path,
            columns=["ts_event", "price", "size"],
            engine="pyarrow",
        )
        prices = frame["price"].to_numpy(dtype=np.float64, copy=False)
        volumes = frame["size"].fillna(0).to_numpy(dtype=np.int32, copy=False)
        valid = np.isfinite(prices) & (volumes > 0)
        prices = prices[valid].astype(np.float64, copy=False)
        volumes = volumes[valid].astype(np.int32, copy=False)
        ts_event = frame.loc[valid, "ts_event"]
        source_type = "trade"
    else:
        raise ValueError(
            "Unsupported MGC tick parquet schema; expected bid/ask snapshot columns "
            "or trade columns ('ts_event', 'price', 'size')."
        )

    utc_times = pd.to_datetime(ts_event, utc=True)
    if getattr(utc_times.dt, "tz", None) is None:
        utc_times = utc_times.dt.tz_localize("UTC")
    utc_ns = utc_times.astype("int64").to_numpy(dtype=np.int64, copy=False)
    local_times = utc_times.dt.tz_convert(timezone).dt.tz_localize(None)
    local_ns = local_times.astype("int64").to_numpy(dtype=np.int64, copy=False)

    order = np.argsort(utc_ns, kind="stable")
    return {
        "tick_times": utc_ns[order],
        "local_times": local_ns[order],
        "tick_prices": prices[order],
        "tick_volumes": volumes[order],
        "source_type": source_type,
    }


def _filter_session_ticks(
    tick_times: np.ndarray,
    local_times: np.ndarray,
    tick_prices: np.ndarray,
    tick_volumes: np.ndarray,
) -> dict[str, np.ndarray]:
    local_index = pd.DatetimeIndex(local_times)
    minute_total = local_index.hour * 60 + local_index.minute
    session_mask = (minute_total >= SESSION_OPEN_MINUTE) & (minute_total < SESSION_CLOSE_MINUTE)

    tick_times = tick_times[session_mask]
    local_times = local_times[session_mask]
    tick_prices = tick_prices[session_mask]
    tick_volumes = tick_volumes[session_mask]

    local_index = pd.DatetimeIndex(local_times)
    session_labels = local_index.normalize().asi8
    boundaries: list[tuple[int, int, int]] = []
    if len(session_labels):
        start = 0
        current = int(session_labels[0])
        for idx in range(1, len(session_labels)):
            if int(session_labels[idx]) != current:
                boundaries.append((start, idx, current))
                start = idx
                current = int(session_labels[idx])
        boundaries.append((start, len(session_labels), current))

    return {
        "tick_times": tick_times,
        "local_times": local_times,
        "tick_prices": tick_prices,
        "tick_volumes": tick_volumes,
        "day_boundaries": np.array(boundaries, dtype=np.int64) if boundaries else np.zeros((0, 3), dtype=np.int64),
    }


@njit(cache=True)
def _upper_bound(values: np.ndarray, target: np.int64, start: int, end: int) -> int:
    lo = start
    hi = end
    while lo < hi:
        mid = (lo + hi) // 2
        if values[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _simulate_day(
    utc_times: np.ndarray,
    local_times: np.ndarray,
    prices: np.ndarray,
    start_idx: int,
    end_idx: int,
    session_label_ns: np.int64,
) -> tuple[np.int64, np.int64, np.float64, np.float64, np.int8, np.int8]:
    if end_idx - start_idx < 2:
        return -1, -1, np.nan, np.nan, 0, EXIT_HARD_CLOSE

    or_end_ns = session_label_ns + (10 * 60 * 60 * 1_000_000_000)
    renko_start_idx = _upper_bound(local_times, or_end_ns, start_idx, end_idx)
    if renko_start_idx <= start_idx or renko_start_idx >= end_idx:
        return -1, -1, np.nan, np.nan, 0, EXIT_HARD_CLOSE

    or_high = prices[start_idx]
    or_low = prices[start_idx]
    for idx in range(start_idx + 1, renko_start_idx):
        price = prices[idx]
        if price > or_high:
            or_high = price
        if price < or_low:
            or_low = price

    last_brick_close = prices[renko_start_idx - 1]
    entry_idx = -1
    signal = 0
    entry_raw = np.nan

    for idx in range(renko_start_idx, end_idx):
        price = prices[idx]
        while price >= last_brick_close + BRICK_SIZE:
            last_brick_close += BRICK_SIZE
            if last_brick_close > or_high:
                entry_idx = idx
                signal = SIGNAL_LONG
                entry_raw = price
                break
        if entry_idx >= 0:
            break
        while price <= last_brick_close - BRICK_SIZE:
            last_brick_close -= BRICK_SIZE
            if last_brick_close < or_low:
                entry_idx = idx
                signal = SIGNAL_SHORT
                entry_raw = price
                break
        if entry_idx >= 0:
            break

    if entry_idx < 0:
        return -1, -1, np.nan, np.nan, 0, EXIT_HARD_CLOSE

    if signal == SIGNAL_LONG:
        entry_price = entry_raw + SLIPPAGE_POINTS
    else:
        entry_price = entry_raw - SLIPPAGE_POINTS

    stop_level = entry_price - STOP_DISTANCE if signal == SIGNAL_LONG else entry_price + STOP_DISTANCE
    target_level = entry_price + TARGET_DISTANCE if signal == SIGNAL_LONG else entry_price - TARGET_DISTANCE

    for idx in range(entry_idx + 1, end_idx):
        price = prices[idx]
        if signal == SIGNAL_LONG:
            if price <= stop_level:
                return entry_idx, idx, entry_price, price - SLIPPAGE_POINTS, signal, EXIT_STOP
            if price >= target_level:
                return entry_idx, idx, entry_price, price - SLIPPAGE_POINTS, signal, EXIT_TARGET
        else:
            if price >= stop_level:
                return entry_idx, idx, entry_price, price + SLIPPAGE_POINTS, signal, EXIT_STOP
            if price <= target_level:
                return entry_idx, idx, entry_price, price + SLIPPAGE_POINTS, signal, EXIT_TARGET

    exit_idx = end_idx - 1
    exit_raw = prices[exit_idx]
    exit_price = exit_raw - SLIPPAGE_POINTS if signal == SIGNAL_LONG else exit_raw + SLIPPAGE_POINTS
    return entry_idx, exit_idx, entry_price, exit_price, signal, EXIT_HARD_CLOSE


def backtest_tick_renko_orb_mgc(tick_data: dict[str, np.ndarray | str]) -> np.ndarray:
    prepared = _filter_session_ticks(
        tick_times=tick_data["tick_times"],
        local_times=tick_data["local_times"],
        tick_prices=tick_data["tick_prices"],
        tick_volumes=tick_data["tick_volumes"],
    )
    tick_times = prepared["tick_times"]
    local_times = prepared["local_times"]
    tick_prices = prepared["tick_prices"]
    day_boundaries = prepared["day_boundaries"]

    trades: list[np.ndarray] = []
    for day_id, (start_idx, end_idx, session_label_ns) in enumerate(day_boundaries):
        entry_idx, exit_idx, entry_price, exit_price, signal, exit_reason = _simulate_day(
            tick_times,
            local_times,
            tick_prices,
            int(start_idx),
            int(end_idx),
            np.int64(session_label_ns),
        )
        if entry_idx < 0:
            continue

        gross_pnl = (
            (exit_price - entry_price) / MGC_TICK_SIZE * MGC_TICK_VALUE
            if signal == SIGNAL_LONG
            else (entry_price - exit_price) / MGC_TICK_SIZE * MGC_TICK_VALUE
        )
        trade = np.zeros((), dtype=TRADE_LOG_DTYPE)
        trade["day_id"] = day_id
        trade["phase_id"] = 0
        trade["payout_cycle_id"] = -1
        trade["entry_time"] = int(tick_times[entry_idx])
        trade["exit_time"] = int(tick_times[exit_idx])
        trade["entry_price"] = float(entry_price)
        trade["exit_price"] = float(exit_price)
        trade["entry_slippage"] = SLIPPAGE_POINTS
        trade["exit_slippage"] = SLIPPAGE_POINTS
        trade["entry_commission"] = MGC_COMMISSION_PER_SIDE
        trade["exit_commission"] = MGC_COMMISSION_PER_SIDE
        trade["contracts"] = 1
        trade["gross_pnl"] = float(gross_pnl)
        trade["net_pnl"] = float(gross_pnl - 2.0 * MGC_COMMISSION_PER_SIDE)
        trade["signal_type"] = signal
        trade["exit_reason"] = int(exit_reason)
        trades.append(trade)

    if not trades:
        return np.zeros(0, dtype=TRADE_LOG_DTYPE)
    return np.array(trades, dtype=TRADE_LOG_DTYPE)


def _compute_metrics(trade_log: np.ndarray) -> dict[str, float]:
    total_trades = int(len(trade_log))
    if total_trades == 0:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "average_trade_pnl": 0.0,
            "average_duration_seconds": 0.0,
        }
    pnl = trade_log["net_pnl"].astype(np.float64)
    gross_profit = float(pnl[pnl > 0.0].sum())
    gross_loss = float(-pnl[pnl < 0.0].sum())
    duration_seconds = (
        trade_log["exit_time"].astype(np.int64) - trade_log["entry_time"].astype(np.int64)
    ) / 1_000_000_000.0
    return {
        "total_trades": total_trades,
        "win_rate": float(np.mean(pnl > 0.0)),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0.0 else float("inf"),
        "average_trade_pnl": float(np.mean(pnl)),
        "average_duration_seconds": float(np.mean(duration_seconds)),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tick-data", type=Path, default=DEFAULT_TICK_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    tick_data = load_mgc_ticks(args.tick_data)
    trade_log = backtest_tick_renko_orb_mgc(tick_data)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, trade_log)

    metrics = _compute_metrics(trade_log)
    print(f"Tick source: {tick_data['source_type']}")
    print(f"Saved trade log to {args.output}")
    print(f"Total trades: {metrics['total_trades']}")
    print(f"Win rate: {metrics['win_rate']:.2%}")
    print(f"Profit factor: {metrics['profit_factor']:.2f}")
    print(f"Average Trade PnL: ${metrics['average_trade_pnl']:.2f}")
    print(f"Average Time in Market: {metrics['average_duration_seconds']:.2f} seconds")


if __name__ == "__main__":
    main()
