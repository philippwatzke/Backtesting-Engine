import numpy as np
import pytest
from numba import njit
from propfirm.core.engine import run_day_kernel
from propfirm.core.types import (
    TRADE_LOG_DTYPE, EXIT_TARGET, EXIT_STOP, EXIT_HARD_CLOSE,
    EXIT_CIRCUIT_BREAKER, SIGNAL_LONG, SIGNAL_SHORT, SIGNAL_NONE,
)


def make_flat_bars(n_bars, base_price=20000.0, spread=5.0):
    """Create synthetic flat market data."""
    opens = np.full(n_bars, base_price, dtype=np.float64)
    highs = np.full(n_bars, base_price + spread, dtype=np.float64)
    lows = np.full(n_bars, base_price - spread, dtype=np.float64)
    closes = np.full(n_bars, base_price, dtype=np.float64)
    volumes = np.full(n_bars, 1000, dtype=np.uint64)
    timestamps = np.arange(n_bars, dtype=np.int64) + 1_640_000_000_000_000_000
    minute_of_day = np.arange(n_bars, dtype=np.int16)
    bar_atr = np.full(n_bars, spread * 2, dtype=np.float64)
    trailing_atr = np.full(n_bars, spread * 2, dtype=np.float64)
    slippage_lookup = np.ones(390, dtype=np.float64)
    return (opens, highs, lows, closes, volumes, timestamps, minute_of_day,
            bar_atr, trailing_atr, slippage_lookup)


@njit(cache=True)
def null_strategy(bar_idx, opens, highs, lows, closes, volumes,
                  minute_of_day, equity, intraday_pnl, position,
                  entry_price, halted, daily_trade_count, params):
    """Strategy that generates no signals."""
    return 0


@njit(cache=True)
def always_long_strategy(bar_idx, opens, highs, lows, closes, volumes,
                         minute_of_day, equity, intraday_pnl, position,
                         entry_price, halted, daily_trade_count, params):
    """Strategy that goes long on bar 15 (after 'range' period)."""
    if minute_of_day[bar_idx] == 15 and position == 0 and not halted:
        return 1
    return 0


@njit(cache=True)
def long_on_bar_15_and_16_strategy(bar_idx, opens, highs, lows, closes, volumes,
                                   minute_of_day, equity, intraday_pnl, position,
                                   entry_price, halted, daily_trade_count, params):
    """Used to verify that an exit bar cannot immediately re-enter on the same OHLC bar."""
    if minute_of_day[bar_idx] == 15 or minute_of_day[bar_idx] == 16:
        if position == 0 and not halted:
            return 1
    return 0


class TestRunDayKernel:
    def test_no_trades_with_null_strategy(self):
        bars = make_flat_bars(390)
        trade_log = np.zeros(100, dtype=TRADE_LOG_DTYPE)
        params = np.array([15.0, 40.0, 60.0, 10.0, -750.0, 600.0, 2.0, 2.0, 0.0, 1.5, 0.54],
                          dtype=np.float64)
        n_trades, final_equity, final_pnl = run_day_kernel(
            *bars, 7, 0, -1, null_strategy, trade_log, 0, 0.0, 0.0, params
        )
        assert n_trades == 0
        assert final_pnl == 0.0

    def test_single_long_trade_target(self):
        n = 390
        base = 20000.0
        bars = list(make_flat_bars(n, base, 5.0))
        for i in range(16, n):
            bars[0][i] = base + 20.0
            bars[1][i] = base + 30.0
            bars[3][i] = base + 25.0
        bars = tuple(bars)

        trade_log = np.zeros(100, dtype=TRADE_LOG_DTYPE)
        params = np.array([15.0, 40.0, 20.0, 1.0, -750.0, 600.0, 2.0, 0.0, 0.0, 1.5, 0.54],
                          dtype=np.float64)
        n_trades, final_equity, final_pnl = run_day_kernel(
            *bars, 7, 0, -1, always_long_strategy, trade_log, 0, 0.0, 0.0, params
        )
        assert n_trades >= 1
        assert trade_log[0]["day_id"] == 7
        assert trade_log[0]["phase_id"] == 0
        assert trade_log[0]["payout_cycle_id"] == -1
        assert trade_log[0]["entry_time"] > 0
        assert trade_log[0]["exit_time"] >= trade_log[0]["entry_time"]
        assert trade_log[0]["signal_type"] == SIGNAL_LONG

    def test_target_exit_does_not_use_stop_penalty(self):
        n = 390
        base = 20000.0
        bars = list(make_flat_bars(n, base, 5.0))
        for i in range(16, n):
            bars[1][i] = base + 30.0
            bars[3][i] = base + 25.0
        bars = tuple(bars)

        params_lo = np.array([15.0, 40.0, 20.0, 1.0, -750.0, 600.0, 2.0, 0.0, 0.0, 1.0, 0.54],
                             dtype=np.float64)
        params_hi = np.array([15.0, 40.0, 20.0, 1.0, -750.0, 600.0, 2.0, 0.0, 0.0, 10.0, 0.54],
                             dtype=np.float64)

        log_lo = np.zeros(10, dtype=TRADE_LOG_DTYPE)
        log_hi = np.zeros(10, dtype=TRADE_LOG_DTYPE)
        _, _, pnl_lo = run_day_kernel(*bars, 7, 0, -1, always_long_strategy, log_lo, 0, 0.0, 0.0, params_lo)
        _, _, pnl_hi = run_day_kernel(*bars, 7, 0, -1, always_long_strategy, log_hi, 0, 0.0, 0.0, params_hi)
        assert log_lo[0]["exit_reason"] == EXIT_TARGET
        assert log_hi[0]["exit_reason"] == EXIT_TARGET
        assert np.isclose(pnl_lo, pnl_hi)

    def test_hard_close_at_1559(self):
        n = 390
        base = 20000.0
        bars = make_flat_bars(n, base, 2.0)
        trade_log = np.zeros(100, dtype=TRADE_LOG_DTYPE)
        params = np.array([15.0, 200.0, 200.0, 1.0, -750.0, 600.0, 2.0, 0.0, 0.0, 1.5, 0.54],
                          dtype=np.float64)
        n_trades, final_equity, final_pnl = run_day_kernel(
            *bars, 7, 0, -1, always_long_strategy, trade_log, 0, 0.0, 0.0, params
        )
        assert n_trades >= 1
        assert trade_log[0]["exit_reason"] == EXIT_HARD_CLOSE

    def test_exit_bar_cannot_reenter_same_bar(self):
        n = 390
        base = 20000.0
        bars = list(make_flat_bars(n, base, 5.0))
        for i in range(16, n):
            bars[1][i] = base + 30.0
            bars[3][i] = base + 25.0
        bars = tuple(bars)

        trade_log = np.zeros(100, dtype=TRADE_LOG_DTYPE)
        params = np.array([15.0, 40.0, 20.0, 1.0, -750.0, 600.0, 3.0, 0.0, 0.0, 1.5, 0.54],
                          dtype=np.float64)
        n_trades, _, _ = run_day_kernel(
            *bars, 7, 0, -1, long_on_bar_15_and_16_strategy, trade_log, 0, 0.0, 0.0, params
        )
        assert n_trades == 1

    def test_circuit_breaker_halts_new_entries(self):
        n = 390
        base = 20000.0
        bars = list(make_flat_bars(n, base, 5.0))
        for i in range(16, n):
            bars[0][i] = base - 500.0
            bars[1][i] = base - 490.0
            bars[2][i] = base - 510.0
            bars[3][i] = base - 500.0
        bars = tuple(bars)
        trade_log = np.zeros(100, dtype=TRADE_LOG_DTYPE)
        params = np.array([15.0, 40.0, 60.0, 5.0, -200.0, 600.0, 3.0, 0.0, 0.0, 1.5, 0.54],
                          dtype=np.float64)
        n_trades, final_equity, final_pnl = run_day_kernel(
            *bars, 7, 0, -1, always_long_strategy, trade_log, 0, 0.0, 0.0, params
        )
        assert n_trades <= 1

    def test_trade_log_net_pnl_matches_equity_delta(self):
        bars = list(make_flat_bars(390, 20000.0, 5.0))
        for i in range(16, 390):
            bars[1][i] = 20035.0
            bars[3][i] = 20025.0
        bars = tuple(bars)
        trade_log = np.zeros(100, dtype=TRADE_LOG_DTYPE)
        params = np.array([15.0, 40.0, 20.0, 1.0, -750.0, 600.0, 2.0, 0.0, 0.0, 1.5, 0.54],
                          dtype=np.float64)
        n_trades, final_equity, final_pnl = run_day_kernel(
            *bars, 3, 0, -1, always_long_strategy, trade_log, 0, 0.0, 0.0, params
        )
        assert np.isclose(trade_log[:n_trades]["net_pnl"].sum(), final_equity)
