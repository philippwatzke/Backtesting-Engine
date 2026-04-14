#!/usr/bin/env python
"""Backtest a causal Renko ORB strategy for MGC and MNQ."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from propfirm.core.types import (
    EXIT_HARD_CLOSE,
    EXIT_STOP,
    EXIT_TARGET,
    SIGNAL_LONG,
    SIGNAL_SHORT,
    TRADE_LOG_DTYPE,
)
from propfirm.market.data_loader import load_session_data


SLIPPAGE_TICKS = 1.22
EMA_PERIOD = 200
TIME_EXIT_MINUTE = 330  # 15:00 ET, matching user-specified 21:00 CET.
DEFAULT_OUTPUT = Path("output/renko_orb_trades.npy")
PORTFOLIO_TRADE_LOG_DTYPE = np.dtype([("asset", "U8")] + TRADE_LOG_DTYPE.descr)


@dataclass(frozen=True)
class AssetSpec:
    name: str
    data_path: Path
    brick_size: float
    opening_range_minutes: int
    tick_size: float
    tick_value: float
    commission_per_side: float
    contracts: int = 1

    @property
    def slippage_points(self) -> float:
        return SLIPPAGE_TICKS * self.tick_size


def _compute_raw_metrics(trade_log: np.ndarray) -> dict[str, float]:
    total_trades = int(len(trade_log))
    if total_trades == 0:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "average_trade_pnl": 0.0,
            "net_profit": 0.0,
        }
    pnl = trade_log["net_pnl"].astype(np.float64)
    gross_profit = float(pnl[pnl > 0.0].sum())
    gross_loss = float(-pnl[pnl < 0.0].sum())
    return {
        "total_trades": total_trades,
        "win_rate": float(np.mean(pnl > 0.0)),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0.0 else float("inf"),
        "average_trade_pnl": float(np.mean(pnl)),
        "net_profit": float(np.sum(pnl)),
    }


def _load_asset_data(spec: AssetSpec) -> dict[str, np.ndarray]:
    data = load_session_data(
        spec.data_path,
        atr_period=14,
        trailing_atr_days=5,
        timeframe_minutes=1,
        session_start="09:30",
        session_end="15:59",
    )
    ema_200 = (
        pd.Series(data["close"])
        .ewm(span=EMA_PERIOD, adjust=False)
        .mean()
        .to_numpy(dtype=np.float64)
    )
    data["ema_200"] = ema_200
    return data


def _find_first_renko_signal(
    closes: np.ndarray,
    ema_200: np.ndarray,
    start_idx: int,
    end_idx: int,
    or_high: float,
    or_low: float,
    brick_size: float,
) -> tuple[int, int] | None:
    last_brick_close = float(closes[start_idx - 1])
    for idx in range(start_idx, end_idx):
        price = float(closes[idx])
        while price >= last_brick_close + brick_size:
            last_brick_close += brick_size
            if last_brick_close > or_high and price > ema_200[idx]:
                return idx, SIGNAL_LONG
        while price <= last_brick_close - brick_size:
            last_brick_close -= brick_size
            if last_brick_close < or_low and price < ema_200[idx]:
                return idx, SIGNAL_SHORT
    return None


def _gross_pnl(signal: int, entry_price: float, exit_price: float, spec: AssetSpec) -> float:
    direction = 1.0 if signal == SIGNAL_LONG else -1.0
    ticks = direction * (exit_price - entry_price) / spec.tick_size
    return float(ticks * spec.tick_value * spec.contracts)


def _simulate_trade(
    spec: AssetSpec,
    day_id: int,
    timestamps: np.ndarray,
    minute_of_day: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    signal_idx: int,
    signal: int,
) -> np.ndarray | None:
    entry_idx = signal_idx + 1
    if entry_idx >= len(opens):
        return None
    if minute_of_day[entry_idx] >= TIME_EXIT_MINUTE:
        return None

    slippage_points = spec.slippage_points
    raw_entry = float(opens[entry_idx])
    if signal == SIGNAL_LONG:
        entry_price = raw_entry + slippage_points
        stop_level = entry_price - 3.0 * spec.brick_size
        target_level = entry_price + 6.0 * spec.brick_size
    else:
        entry_price = raw_entry - slippage_points
        stop_level = entry_price + 3.0 * spec.brick_size
        target_level = entry_price - 6.0 * spec.brick_size

    exit_idx = -1
    exit_reason = EXIT_HARD_CLOSE
    raw_exit = float(opens[-1])
    for idx in range(entry_idx, len(opens)):
        if minute_of_day[idx] >= TIME_EXIT_MINUTE:
            exit_idx = idx
            raw_exit = float(opens[idx])
            exit_reason = EXIT_HARD_CLOSE
            break

        if signal == SIGNAL_LONG:
            stop_hit = float(lows[idx]) <= stop_level
            target_hit = float(highs[idx]) >= target_level
            if stop_hit:
                exit_idx = idx
                raw_exit = stop_level
                exit_reason = EXIT_STOP
                break
            if target_hit:
                exit_idx = idx
                raw_exit = target_level
                exit_reason = EXIT_TARGET
                break
        else:
            stop_hit = float(highs[idx]) >= stop_level
            target_hit = float(lows[idx]) <= target_level
            if stop_hit:
                exit_idx = idx
                raw_exit = stop_level
                exit_reason = EXIT_STOP
                break
            if target_hit:
                exit_idx = idx
                raw_exit = target_level
                exit_reason = EXIT_TARGET
                break

    if exit_idx < 0:
        exit_idx = len(opens) - 1
        raw_exit = float(opens[exit_idx])
        exit_reason = EXIT_HARD_CLOSE

    if signal == SIGNAL_LONG:
        exit_price = raw_exit - slippage_points
    else:
        exit_price = raw_exit + slippage_points

    gross_pnl = _gross_pnl(signal, entry_price, exit_price, spec)
    total_commission = 2.0 * spec.commission_per_side * spec.contracts
    trade = np.zeros((), dtype=TRADE_LOG_DTYPE)
    trade["day_id"] = day_id
    trade["phase_id"] = 0
    trade["payout_cycle_id"] = -1
    trade["entry_time"] = int(timestamps[entry_idx])
    trade["exit_time"] = int(timestamps[exit_idx])
    trade["entry_price"] = entry_price
    trade["exit_price"] = exit_price
    trade["entry_slippage"] = slippage_points
    trade["exit_slippage"] = slippage_points
    trade["entry_commission"] = spec.commission_per_side * spec.contracts
    trade["exit_commission"] = spec.commission_per_side * spec.contracts
    trade["contracts"] = spec.contracts
    trade["gross_pnl"] = gross_pnl
    trade["net_pnl"] = gross_pnl - total_commission
    trade["signal_type"] = signal
    trade["exit_reason"] = exit_reason
    return trade


def backtest_asset(spec: AssetSpec) -> np.ndarray:
    data = _load_asset_data(spec)
    trades: list[np.ndarray] = []
    for day_id, (start, end) in enumerate(data["day_boundaries"]):
        day_slice = slice(start, end)
        minute_of_day = data["minute_of_day"][day_slice]
        timestamps = data["timestamps"][day_slice]
        opens = data["open"][day_slice]
        highs = data["high"][day_slice]
        lows = data["low"][day_slice]
        closes = data["close"][day_slice]
        ema_200 = data["ema_200"][day_slice]

        if len(opens) <= spec.opening_range_minutes:
            continue
        if not np.isfinite(ema_200[spec.opening_range_minutes:]).all():
            continue

        or_high = float(np.max(highs[: spec.opening_range_minutes]))
        or_low = float(np.min(lows[: spec.opening_range_minutes]))
        signal = _find_first_renko_signal(
            closes=closes,
            ema_200=ema_200,
            start_idx=spec.opening_range_minutes,
            end_idx=len(closes),
            or_high=or_high,
            or_low=or_low,
            brick_size=spec.brick_size,
        )
        if signal is None:
            continue

        signal_idx, signal_side = signal
        trade = _simulate_trade(
            spec=spec,
            day_id=day_id,
            timestamps=timestamps,
            minute_of_day=minute_of_day,
            opens=opens,
            highs=highs,
            lows=lows,
            signal_idx=signal_idx,
            signal=signal_side,
        )
        if trade is None:
            continue
        combined = np.zeros((), dtype=PORTFOLIO_TRADE_LOG_DTYPE)
        combined["asset"] = spec.name
        for field in TRADE_LOG_DTYPE.names:
            combined[field] = trade[field]
        trades.append(combined)

    if not trades:
        return np.zeros(0, dtype=PORTFOLIO_TRADE_LOG_DTYPE)
    return np.array(trades, dtype=PORTFOLIO_TRADE_LOG_DTYPE)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnq-data", type=Path, default=Path("data/processed/MNQ_1m_full_test.parquet"))
    parser.add_argument("--mgc-data", type=Path, default=Path("data/processed/MGC_1m_full_test.parquet"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    specs = [
        AssetSpec(
            name="MNQ",
            data_path=args.mnq_data,
            brick_size=10.0,
            opening_range_minutes=15,
            tick_size=0.25,
            tick_value=0.50,
            commission_per_side=0.54,
        ),
        AssetSpec(
            name="MGC",
            data_path=args.mgc_data,
            brick_size=0.20,
            opening_range_minutes=30,
            tick_size=0.10,
            tick_value=1.00,
            commission_per_side=0.54,
        ),
    ]

    asset_logs = [backtest_asset(spec) for spec in specs]
    trade_log = (
        np.concatenate(asset_logs).astype(PORTFOLIO_TRADE_LOG_DTYPE, copy=False)
        if any(len(log) for log in asset_logs)
        else np.zeros(0, dtype=PORTFOLIO_TRADE_LOG_DTYPE)
    )
    if len(trade_log):
        order = np.argsort(trade_log["entry_time"], kind="stable")
        trade_log = trade_log[order]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, trade_log)

    metrics = _compute_raw_metrics(trade_log)
    print(f"Saved trade log to {args.output}")
    print(f"Total trades: {metrics['total_trades']}")
    print(f"Win rate: {metrics['win_rate']:.2%}")
    print(f"Profit factor: {metrics['profit_factor']:.2f}")
    print(f"Average Trade PnL: ${metrics['average_trade_pnl']:.2f}")


if __name__ == "__main__":
    main()
