#!/usr/bin/env python
"""Backtest a long-only VWAP band mean-reversion system on 1m futures data."""

from __future__ import annotations

import argparse
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
from propfirm.market.data_loader import SESSION_TZ


SLIPPAGE_TICKS = 1.22
STOP_ATR_MULTIPLIER = 1.5
BAND_MULTIPLIER = 2.5
ATR_PERIOD = 14
MACD_FAST = 24
MACD_SLOW = 52
MACD_SIGNAL = 18
ENTRY_START_MINUTE = 12 * 60
ENTRY_END_MINUTE = 12 * 60 + 59
HARD_CLOSE_MINUTE = 14 * 60 + 29
ENTRY_WEEKDAYS = (0, 1, 3)
DEFAULT_DATA = Path("data/processed/RB_1m_continuous.parquet")
DEFAULT_PNL_OUTPUT = Path("output/vwap_rb_trades.npy")
DEFAULT_TRADE_LOG_OUTPUT = Path("output/vwap_rb_trade_log.npy")
DEFAULT_TICK_SIZE = 0.0001
DEFAULT_TICK_VALUE = 4.20
DEFAULT_COMMISSION_PER_SIDE = 0.54


def _session_ids(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series((index + pd.Timedelta(hours=6)).normalize(), index=index)


def _load_mcl_frame(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    keep_columns = [col for col in ["timestamp", "ts_event", "open", "high", "low", "close", "volume"] if col in df.columns]
    if keep_columns:
        df = df.loc[:, keep_columns]
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
        df = df.set_index("timestamp")
    elif "ts_event" in df.columns:
        df["ts_event"] = pd.to_datetime(df["ts_event"], errors="raise", utc=True)
        df = df.set_index("ts_event")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Expected DatetimeIndex or timestamp column in parquet")
    df = df.sort_index(kind="stable").copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize(SESSION_TZ, ambiguous="infer", nonexistent="shift_forward")
    else:
        df.index = df.index.tz_convert(SESSION_TZ)
    return df


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["session_id"] = _session_ids(out.index).to_numpy()
    typical_price = (out["high"] + out["low"] + out["close"]) / 3.0
    volume = out["volume"].astype(np.float64)

    out["cum_vol"] = volume.groupby(out["session_id"]).cumsum()
    out["cum_pv"] = (typical_price * volume).groupby(out["session_id"]).cumsum()
    out["cum_p2v"] = ((typical_price ** 2) * volume).groupby(out["session_id"]).cumsum()
    out["vwap"] = out["cum_pv"] / out["cum_vol"]

    variance = (out["cum_p2v"] / out["cum_vol"]) - (out["vwap"] ** 2)
    variance = variance.clip(lower=0.0)
    out["std"] = np.sqrt(variance)
    out["upper_band"] = out["vwap"] + BAND_MULTIPLIER * out["std"]
    out["lower_band"] = out["vwap"] - BAND_MULTIPLIER * out["std"]

    prev_close = out["close"].shift(1)
    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr_14"] = tr.rolling(window=ATR_PERIOD, min_periods=ATR_PERIOD).mean()
    out["vwap_prev"] = out.groupby("session_id")["vwap"].shift(1)

    ema_fast = out["close"].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = out["close"].ewm(span=MACD_SLOW, adjust=False).mean()
    out["macd_line"] = ema_fast - ema_slow
    out["macd_signal"] = out["macd_line"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    out["macd_hist"] = out["macd_line"] - out["macd_signal"]

    minute_total = out.index.hour * 60 + out.index.minute
    out["minute_total"] = minute_total.astype(np.int16)
    out["weekday"] = out.index.dayofweek.astype(np.int8)
    return out


def _is_signal_window(weekday: int, minute_total: int) -> bool:
    if weekday not in ENTRY_WEEKDAYS:
        return False
    if minute_total < ENTRY_START_MINUTE or minute_total > ENTRY_END_MINUTE:
        return False
    return True


def _compute_metrics(trade_log: np.ndarray) -> dict[str, float]:
    total_trades = int(len(trade_log))
    if total_trades == 0:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "average_trade_pnl": 0.0,
        }
    pnl = trade_log["net_pnl"].astype(np.float64)
    gross_profit = float(pnl[pnl > 0.0].sum())
    gross_loss = float(-pnl[pnl < 0.0].sum())
    return {
        "total_trades": total_trades,
        "win_rate": float(np.mean(pnl > 0.0)),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0.0 else float("inf"),
        "average_trade_pnl": float(np.mean(pnl)),
    }


def _compute_direction_metrics(trade_log: np.ndarray, signal_type: int) -> dict[str, float]:
    direction_trades = trade_log[trade_log["signal_type"] == signal_type]
    return _compute_metrics(direction_trades)


def _finalize_trade(
    day_id: int,
    entry_time: int,
    exit_time: int,
    entry_price: float,
    exit_price: float,
    signal: int,
    exit_reason: int,
    tick_size: float,
    tick_value: float,
    commission_per_side: float,
    slippage_points: float,
) -> np.ndarray:
    gross_pnl = (
        (exit_price - entry_price) / tick_size * tick_value
        if signal == SIGNAL_LONG
        else (entry_price - exit_price) / tick_size * tick_value
    )
    trade = np.zeros((), dtype=TRADE_LOG_DTYPE)
    trade["day_id"] = day_id
    trade["phase_id"] = 0
    trade["payout_cycle_id"] = -1
    trade["entry_time"] = entry_time
    trade["exit_time"] = exit_time
    trade["entry_price"] = entry_price
    trade["exit_price"] = exit_price
    trade["entry_slippage"] = slippage_points
    trade["exit_slippage"] = slippage_points
    trade["entry_commission"] = commission_per_side
    trade["exit_commission"] = commission_per_side
    trade["contracts"] = 1
    trade["gross_pnl"] = gross_pnl
    trade["net_pnl"] = gross_pnl - 2.0 * commission_per_side
    trade["signal_type"] = signal
    trade["exit_reason"] = exit_reason
    return trade


def backtest_vwap_reversion(
    df: pd.DataFrame,
    tick_size: float = DEFAULT_TICK_SIZE,
    tick_value: float = DEFAULT_TICK_VALUE,
    commission_per_side: float = DEFAULT_COMMISSION_PER_SIDE,
) -> np.ndarray:
    trades: list[np.ndarray] = []
    slippage_points = SLIPPAGE_TICKS * tick_size

    for day_id, (_, session_df) in enumerate(df.groupby("session_id", sort=True)):
        session_df = session_df.copy()
        idx = session_df.index
        opens = session_df["open"].to_numpy(dtype=np.float64)
        highs = session_df["high"].to_numpy(dtype=np.float64)
        lows = session_df["low"].to_numpy(dtype=np.float64)
        closes = session_df["close"].to_numpy(dtype=np.float64)
        upper_band = session_df["upper_band"].to_numpy(dtype=np.float64)
        lower_band = session_df["lower_band"].to_numpy(dtype=np.float64)
        vwap_prev = session_df["vwap_prev"].to_numpy(dtype=np.float64)
        atr_14 = session_df["atr_14"].to_numpy(dtype=np.float64)
        macd_hist = session_df["macd_hist"].to_numpy(dtype=np.float64)
        weekday = session_df["weekday"].to_numpy(dtype=np.int8)
        minute_total = session_df["minute_total"].to_numpy(dtype=np.int16)
        utc_ns = idx.tz_convert("UTC").asi8

        position = 0
        entry_idx = -1
        entry_price = np.nan
        stop_price = np.nan
        pending_signal = 0
        pending_atr = np.nan

        for i in range(1, len(session_df)):
            hard_close_now = minute_total[i] == HARD_CLOSE_MINUTE

            if position != 0 and hard_close_now:
                raw_exit = opens[i]
                exit_price = raw_exit - slippage_points if position == SIGNAL_LONG else raw_exit + slippage_points
                trades.append(
                    _finalize_trade(
                        day_id=day_id,
                        entry_time=int(utc_ns[entry_idx]),
                        exit_time=int(utc_ns[i]),
                        entry_price=float(entry_price),
                        exit_price=float(exit_price),
                        signal=int(position),
                        exit_reason=EXIT_HARD_CLOSE,
                        tick_size=tick_size,
                        tick_value=tick_value,
                        commission_per_side=commission_per_side,
                        slippage_points=slippage_points,
                    )
                )
                position = 0
                entry_idx = -1
                pending_signal = 0

            if position == 0 and pending_signal != 0:
                if not hard_close_now and _is_signal_window(int(weekday[i]), int(minute_total[i])):
                    raw_entry = opens[i]
                    entry_price = raw_entry + slippage_points if pending_signal == SIGNAL_LONG else raw_entry - slippage_points
                    stop_distance = STOP_ATR_MULTIPLIER * float(pending_atr)
                    stop_price = (
                        entry_price - stop_distance if pending_signal == SIGNAL_LONG else entry_price + stop_distance
                    )
                    position = pending_signal
                    entry_idx = i
                pending_signal = 0

            if position != 0:
                # Conservative intrabar ordering: stop has priority when both levels touch.
                if position == SIGNAL_LONG:
                    stop_hit = lows[i] <= stop_price
                    target_level = vwap_prev[i]
                    target_hit = np.isfinite(target_level) and highs[i] >= target_level
                    if stop_hit:
                        exit_price = stop_price - slippage_points
                        trades.append(
                            _finalize_trade(
                                day_id=day_id,
                                entry_time=int(utc_ns[entry_idx]),
                                exit_time=int(utc_ns[i]),
                                entry_price=float(entry_price),
                                exit_price=float(exit_price),
                                signal=SIGNAL_LONG,
                                exit_reason=EXIT_STOP,
                                tick_size=tick_size,
                                tick_value=tick_value,
                                commission_per_side=commission_per_side,
                                slippage_points=slippage_points,
                            )
                        )
                        position = 0
                        entry_idx = -1
                    elif target_hit:
                        exit_price = float(target_level) - slippage_points
                        trades.append(
                            _finalize_trade(
                                day_id=day_id,
                                entry_time=int(utc_ns[entry_idx]),
                                exit_time=int(utc_ns[i]),
                                entry_price=float(entry_price),
                                exit_price=float(exit_price),
                                signal=SIGNAL_LONG,
                                exit_reason=EXIT_TARGET,
                                tick_size=tick_size,
                                tick_value=tick_value,
                                commission_per_side=commission_per_side,
                                slippage_points=slippage_points,
                            )
                        )
                        position = 0
                        entry_idx = -1
                else:
                    stop_hit = highs[i] >= stop_price
                    target_level = vwap_prev[i]
                    target_hit = np.isfinite(target_level) and lows[i] <= target_level
                    if stop_hit:
                        exit_price = stop_price + SLIPPAGE_POINTS
                        trades.append(
                            _finalize_trade(
                                day_id=day_id,
                                entry_time=int(utc_ns[entry_idx]),
                                exit_time=int(utc_ns[i]),
                                entry_price=float(entry_price),
                                exit_price=float(exit_price),
                                signal=SIGNAL_SHORT,
                                exit_reason=EXIT_STOP,
                            )
                        )
                        position = 0
                        entry_idx = -1
                    elif target_hit:
                        exit_price = float(target_level) + SLIPPAGE_POINTS
                        trades.append(
                            _finalize_trade(
                                day_id=day_id,
                                entry_time=int(utc_ns[entry_idx]),
                                exit_time=int(utc_ns[i]),
                                entry_price=float(entry_price),
                                exit_price=float(exit_price),
                                signal=SIGNAL_SHORT,
                                exit_reason=EXIT_TARGET,
                            )
                        )
                        position = 0
                        entry_idx = -1

            if position != 0:
                continue
            if i + 1 >= len(session_df):
                continue
            if not np.isfinite(upper_band[i - 1]) or not np.isfinite(lower_band[i - 1]):
                continue
            if (
                not np.isfinite(upper_band[i])
                or not np.isfinite(lower_band[i])
                or not np.isfinite(atr_14[i])
                or not np.isfinite(macd_hist[i])
                or not np.isfinite(macd_hist[i - 1])
            ):
                continue
            if not _is_signal_window(int(weekday[i]), int(minute_total[i])):
                continue
            if not _is_signal_window(int(weekday[i + 1]), int(minute_total[i + 1])):
                continue
            if minute_total[i + 1] >= HARD_CLOSE_MINUTE:
                continue

            long_setup = (
                closes[i - 1] < lower_band[i - 1]
                and closes[i] > lower_band[i]
                and macd_hist[i] > macd_hist[i - 1]
            )
            if long_setup:
                pending_signal = SIGNAL_LONG
                pending_atr = float(atr_14[i])

        if position != 0:
            last_idx = len(session_df) - 1
            raw_exit = closes[last_idx]
            exit_price = raw_exit - slippage_points if position == SIGNAL_LONG else raw_exit + slippage_points
            trades.append(
                _finalize_trade(
                    day_id=day_id,
                    entry_time=int(utc_ns[entry_idx]),
                    exit_time=int(utc_ns[last_idx]),
                    entry_price=float(entry_price),
                    exit_price=float(exit_price),
                    signal=int(position),
                    exit_reason=EXIT_HARD_CLOSE,
                    tick_size=tick_size,
                    tick_value=tick_value,
                    commission_per_side=commission_per_side,
                    slippage_points=slippage_points,
                )
            )

    if not trades:
        return np.zeros(0, dtype=TRADE_LOG_DTYPE)
    return np.array(trades, dtype=TRADE_LOG_DTYPE)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-pnls", type=Path, default=DEFAULT_PNL_OUTPUT)
    parser.add_argument("--output-trade-log", type=Path, default=DEFAULT_TRADE_LOG_OUTPUT)
    parser.add_argument("--tick-size", type=float, default=DEFAULT_TICK_SIZE)
    parser.add_argument("--tick-value", type=float, default=DEFAULT_TICK_VALUE)
    parser.add_argument("--commission-per-side", type=float, default=DEFAULT_COMMISSION_PER_SIDE)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    df = _load_mcl_frame(args.data)
    df = _compute_indicators(df)
    trade_log = backtest_vwap_reversion(
        df,
        tick_size=args.tick_size,
        tick_value=args.tick_value,
        commission_per_side=args.commission_per_side,
    )

    args.output_pnls.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_pnls, trade_log["net_pnl"].astype(np.float64))
    np.save(args.output_trade_log, trade_log)

    metrics = _compute_metrics(trade_log)
    long_metrics = _compute_direction_metrics(trade_log, SIGNAL_LONG)
    short_metrics = _compute_direction_metrics(trade_log, SIGNAL_SHORT)
    print(f"Saved trade PnLs to {args.output_pnls}")
    print(f"Saved trade log to {args.output_trade_log}")
    print(f"Total trades: {metrics['total_trades']}")
    print(f"Win rate: {metrics['win_rate']:.2%}")
    print(f"Profit factor: {metrics['profit_factor']:.2f}")
    print(f"Average Trade PnL: ${metrics['average_trade_pnl']:.2f}")
    print("Longs:")
    print(f"  Trades: {long_metrics['total_trades']}")
    print(f"  Win rate: {long_metrics['win_rate']:.2%}")
    print(f"  Profit factor: {long_metrics['profit_factor']:.2f}")
    print(f"  Average Trade PnL: ${long_metrics['average_trade_pnl']:.2f}")
    print("Shorts:")
    print(f"  Trades: {short_metrics['total_trades']}")
    print(f"  Win rate: {short_metrics['win_rate']:.2%}")
    print(f"  Profit factor: {short_metrics['profit_factor']:.2f}")
    print(f"  Average Trade PnL: ${short_metrics['average_trade_pnl']:.2f}")


if __name__ == "__main__":
    main()
