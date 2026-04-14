#!/usr/bin/env python
"""Stress-test the MCL lunch-fade strategy across nearby parameter shifts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from propfirm.core.types import EXIT_HARD_CLOSE, EXIT_STOP, EXIT_TARGET, SIGNAL_LONG, TRADE_LOG_DTYPE
from propfirm.market.data_loader import SESSION_TZ


MCL_TICK_SIZE = 0.01
MCL_TICK_VALUE = 1.00
MCL_COMMISSION_PER_SIDE = 0.54
SLIPPAGE_TICKS = 1.22
SLIPPAGE_POINTS = SLIPPAGE_TICKS * MCL_TICK_SIZE
ATR_PERIOD = 14
MACD_FAST = 24
MACD_SLOW = 52
MACD_SIGNAL = 18
ENTRY_WEEKDAYS = (0, 1, 3)
HARD_CLOSE_MINUTE = 14 * 60 + 29
IS_DATA = Path("data/processed/MCL_1m_IS.parquet")
OOS_DATA = Path("data/processed/MCL_1m_OOS.parquet")


@dataclass(frozen=True)
class StressCase:
    name: str
    start_minute: int
    end_minute: int
    band_multiplier: float
    stop_atr_multiplier: float


def _session_ids(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series((index + pd.Timedelta(hours=6)).normalize(), index=index)


def _load_mcl_frame(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(
        parquet_path,
        columns=["open", "high", "low", "close", "volume"],
        engine="pyarrow",
    )
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Expected DatetimeIndex in MCL parquet")
    if df.index.tz is None:
        raise ValueError("Expected timezone-aware timestamps in MCL parquet")
    df = df.sort_index(kind="stable").copy()
    df.index = df.index.tz_convert(SESSION_TZ)
    return df


def _load_combined_frame() -> pd.DataFrame:
    frames = [_load_mcl_frame(IS_DATA), _load_mcl_frame(OOS_DATA)]
    combined = pd.concat(frames).sort_index(kind="stable")
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined


def _prepare_base_frame(df: pd.DataFrame) -> pd.DataFrame:
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


def _is_signal_window(weekday: int, minute_total: int, start_minute: int, end_minute: int) -> bool:
    if weekday not in ENTRY_WEEKDAYS:
        return False
    if minute_total < start_minute or minute_total > end_minute:
        return False
    return True


def _finalize_trade(
    day_id: int,
    entry_time: int,
    exit_time: int,
    entry_price: float,
    exit_price: float,
    exit_reason: int,
) -> np.ndarray:
    gross_pnl = (exit_price - entry_price) / MCL_TICK_SIZE * MCL_TICK_VALUE
    trade = np.zeros((), dtype=TRADE_LOG_DTYPE)
    trade["day_id"] = day_id
    trade["phase_id"] = 0
    trade["payout_cycle_id"] = -1
    trade["entry_time"] = entry_time
    trade["exit_time"] = exit_time
    trade["entry_price"] = entry_price
    trade["exit_price"] = exit_price
    trade["entry_slippage"] = SLIPPAGE_POINTS
    trade["exit_slippage"] = SLIPPAGE_POINTS
    trade["entry_commission"] = MCL_COMMISSION_PER_SIDE
    trade["exit_commission"] = MCL_COMMISSION_PER_SIDE
    trade["contracts"] = 1
    trade["gross_pnl"] = gross_pnl
    trade["net_pnl"] = gross_pnl - 2.0 * MCL_COMMISSION_PER_SIDE
    trade["signal_type"] = SIGNAL_LONG
    trade["exit_reason"] = exit_reason
    return trade


def _compute_metrics(trade_log: np.ndarray) -> tuple[int, float, float, float]:
    total_trades = int(len(trade_log))
    if total_trades == 0:
        return 0, 0.0, 0.0, 0.0
    pnl = trade_log["net_pnl"].astype(np.float64)
    gross_profit = float(pnl[pnl > 0.0].sum())
    gross_loss = float(-pnl[pnl < 0.0].sum())
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0.0 else float("inf")
    return total_trades, float(np.mean(pnl > 0.0)), profit_factor, float(np.sum(pnl))


def run_case(base_df: pd.DataFrame, case: StressCase) -> tuple[int, float, float, float]:
    upper_band = base_df["vwap"] + case.band_multiplier * base_df["std"]
    lower_band = base_df["vwap"] - case.band_multiplier * base_df["std"]
    trades: list[np.ndarray] = []

    grouped = base_df.groupby("session_id", sort=True)
    for day_id, (_, session_df) in enumerate(grouped):
        idx = session_df.index
        opens = session_df["open"].to_numpy(dtype=np.float64)
        highs = session_df["high"].to_numpy(dtype=np.float64)
        lows = session_df["low"].to_numpy(dtype=np.float64)
        closes = session_df["close"].to_numpy(dtype=np.float64)
        session_upper = upper_band.loc[idx].to_numpy(dtype=np.float64)
        session_lower = lower_band.loc[idx].to_numpy(dtype=np.float64)
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
        pending_long = False
        pending_atr = np.nan

        for i in range(1, len(session_df)):
            hard_close_now = minute_total[i] == HARD_CLOSE_MINUTE

            if position != 0 and hard_close_now:
                exit_price = opens[i] - SLIPPAGE_POINTS
                trades.append(
                    _finalize_trade(
                        day_id=day_id,
                        entry_time=int(utc_ns[entry_idx]),
                        exit_time=int(utc_ns[i]),
                        entry_price=float(entry_price),
                        exit_price=float(exit_price),
                        exit_reason=EXIT_HARD_CLOSE,
                    )
                )
                position = 0
                entry_idx = -1
                pending_long = False

            if position == 0 and pending_long:
                if not hard_close_now and _is_signal_window(
                    int(weekday[i]), int(minute_total[i]), case.start_minute, case.end_minute
                ):
                    entry_price = opens[i] + SLIPPAGE_POINTS
                    stop_price = entry_price - case.stop_atr_multiplier * float(pending_atr)
                    position = SIGNAL_LONG
                    entry_idx = i
                pending_long = False

            if position != 0:
                stop_hit = lows[i] <= stop_price
                target_level = vwap_prev[i]
                target_hit = np.isfinite(target_level) and highs[i] >= target_level
                if stop_hit:
                    exit_price = stop_price - SLIPPAGE_POINTS
                    trades.append(
                        _finalize_trade(
                            day_id=day_id,
                            entry_time=int(utc_ns[entry_idx]),
                            exit_time=int(utc_ns[i]),
                            entry_price=float(entry_price),
                            exit_price=float(exit_price),
                            exit_reason=EXIT_STOP,
                        )
                    )
                    position = 0
                    entry_idx = -1
                elif target_hit:
                    exit_price = float(target_level) - SLIPPAGE_POINTS
                    trades.append(
                        _finalize_trade(
                            day_id=day_id,
                            entry_time=int(utc_ns[entry_idx]),
                            exit_time=int(utc_ns[i]),
                            entry_price=float(entry_price),
                            exit_price=float(exit_price),
                            exit_reason=EXIT_TARGET,
                        )
                    )
                    position = 0
                    entry_idx = -1

            if position != 0:
                continue
            if i + 1 >= len(session_df):
                continue
            if not np.isfinite(session_lower[i - 1]):
                continue
            if (
                not np.isfinite(session_lower[i])
                or not np.isfinite(atr_14[i])
                or not np.isfinite(macd_hist[i])
                or not np.isfinite(macd_hist[i - 1])
            ):
                continue
            if not _is_signal_window(int(weekday[i]), int(minute_total[i]), case.start_minute, case.end_minute):
                continue
            if not _is_signal_window(
                int(weekday[i + 1]), int(minute_total[i + 1]), case.start_minute, case.end_minute
            ):
                continue
            if minute_total[i + 1] >= HARD_CLOSE_MINUTE:
                continue

            long_setup = (
                closes[i - 1] < session_lower[i - 1]
                and closes[i] > session_lower[i]
                and macd_hist[i] > macd_hist[i - 1]
            )
            if long_setup:
                pending_long = True
                pending_atr = float(atr_14[i])

        if position != 0:
            last_idx = len(session_df) - 1
            exit_price = closes[last_idx] - SLIPPAGE_POINTS
            trades.append(
                _finalize_trade(
                    day_id=day_id,
                    entry_time=int(utc_ns[entry_idx]),
                    exit_time=int(utc_ns[last_idx]),
                    entry_price=float(entry_price),
                    exit_price=float(exit_price),
                    exit_reason=EXIT_HARD_CLOSE,
                )
            )

    if not trades:
        return 0, 0.0, 0.0, 0.0
    trade_log = np.array(trades, dtype=TRADE_LOG_DTYPE)
    return _compute_metrics(trade_log)


def _format_minutes(minute_total: int) -> str:
    hours, minutes = divmod(minute_total, 60)
    return f"{hours:02d}:{minutes:02d}"


def main() -> None:
    base_df = _prepare_base_frame(_load_combined_frame())
    cases = [
        StressCase("A", 11 * 60 + 45, 12 * 60 + 45, 2.5, 1.5),
        StressCase("B", 12 * 60 + 15, 13 * 60 + 15, 2.5, 1.5),
        StressCase("C", 12 * 60, 12 * 60 + 59, 2.3, 1.5),
        StressCase("D", 12 * 60, 12 * 60 + 59, 2.7, 1.5),
        StressCase("E", 12 * 60, 12 * 60 + 59, 2.5, 1.2),
        StressCase("F", 12 * 60, 12 * 60 + 59, 2.5, 1.8),
    ]

    print("MCL VWAP Lunch-Fade Stress Test (IS + OOS combined)")
    print(
        f"{'Test':<4} {'Window':<13} {'Band':>6} {'Stop':>6} {'Trades':>8} {'Winrate':>9} {'PF':>8} {'Net PnL':>12}"
    )
    print("-" * 74)
    for case in cases:
        trades, win_rate, profit_factor, net_pnl = run_case(base_df, case)
        window = f"{_format_minutes(case.start_minute)}-{_format_minutes(case.end_minute)}"
        print(
            f"{case.name:<4} {window:<13} {case.band_multiplier:>6.1f} {case.stop_atr_multiplier:>6.1f} "
            f"{trades:>8d} {win_rate:>8.2%} {profit_factor:>8.2f} {net_pnl:>12.2f}"
        )


if __name__ == "__main__":
    main()
