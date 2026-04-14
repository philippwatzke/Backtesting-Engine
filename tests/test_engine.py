import numpy as np
import pytest
from numba import njit
from propfirm.core.engine import run_day_kernel, run_day_kernel_portfolio
from propfirm.core.types import (
    TRADE_LOG_DTYPE, EXIT_TARGET, EXIT_STOP, EXIT_HARD_CLOSE,
    EXIT_CIRCUIT_BREAKER, SIGNAL_LONG, SIGNAL_SHORT, SIGNAL_NONE,
    PARAMS_ARRAY_LENGTH, PARAMS_RANGE_MINUTES, PARAMS_STOP_TICKS,
    PARAMS_TARGET_TICKS, PARAMS_CONTRACTS, PARAMS_DAILY_STOP,
    PARAMS_DAILY_TARGET, PARAMS_MAX_TRADES, PARAMS_BUFFER_TICKS,
    PARAMS_VOL_THRESHOLD, PARAMS_STOP_PENALTY, PARAMS_COMMISSION,
    PARAMS_BREAKEVEN_TRIGGER_TICKS,
    PROFILE_RISK_PER_TRADE_USD, PROFILE_STOP_ATR_MULTIPLIER,
    PROFILE_TARGET_ATR_MULTIPLIER, PROFILE_BREAKEVEN_TRIGGER_TICKS,
    PROFILE_RISK_BUFFER_FRACTION,
    PROFILE_ARRAY_LENGTH,
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
                  bar_atr, trailing_atr,
                  minute_of_day, equity, intraday_pnl, position,
                  entry_price, halted, daily_trade_count, params):
    """Strategy that generates no signals."""
    return 0


@njit(cache=True)
def always_long_strategy(bar_idx, opens, highs, lows, closes, volumes,
                         bar_atr, trailing_atr,
                         minute_of_day, equity, intraday_pnl, position,
                         entry_price, halted, daily_trade_count, params):
    """Strategy that goes long on bar 15 (after 'range' period)."""
    if minute_of_day[bar_idx] == 15 and position == 0 and not halted:
        return 1
    return 0


@njit(cache=True)
def long_on_bar_15_and_16_strategy(bar_idx, opens, highs, lows, closes, volumes,
                                   bar_atr, trailing_atr,
                                   minute_of_day, equity, intraday_pnl, position,
                                   entry_price, halted, daily_trade_count, params):
    """Used to verify that an exit bar cannot immediately re-enter on the same OHLC bar."""
    if minute_of_day[bar_idx] == 15 or minute_of_day[bar_idx] == 16:
        if position == 0 and not halted:
            return 1
    return 0


@njit(cache=True)
def always_portfolio_breakout_strategy(bar_idx, opens, highs, lows, closes, volumes,
                                       bar_atr, trailing_atr, daily_atr_ratio, rvol, close_sma_50,
                                       daily_regime_bias,
                                       donchian_high_5, donchian_low_5,
                                       minute_of_day, day_of_week, equity, intraday_pnl, position,
                                       entry_price, halted, daily_trade_count, params):
    if minute_of_day[bar_idx] == 15 and position == 0 and not halted:
        return 2
    return 0


class TestRunDayKernel:
    def _make_params(self, stop_ticks=40.0, target_ticks=60.0, contracts=10.0,
                     daily_stop=-750.0, daily_target=600.0, max_trades=2.0,
                     buffer_ticks=2.0, volume_threshold=0.0, stop_penalty=1.5,
                     commission=0.54, breakeven_trigger_ticks=0.0):
        params = np.zeros(PARAMS_ARRAY_LENGTH, dtype=np.float64)
        params[PARAMS_RANGE_MINUTES] = 15.0
        params[PARAMS_STOP_TICKS] = stop_ticks
        params[PARAMS_TARGET_TICKS] = target_ticks
        params[PARAMS_CONTRACTS] = contracts
        params[PARAMS_DAILY_STOP] = daily_stop
        params[PARAMS_DAILY_TARGET] = daily_target
        params[PARAMS_MAX_TRADES] = max_trades
        params[PARAMS_BUFFER_TICKS] = buffer_ticks
        params[PARAMS_VOL_THRESHOLD] = volume_threshold
        params[PARAMS_STOP_PENALTY] = stop_penalty
        params[PARAMS_COMMISSION] = commission
        params[PARAMS_BREAKEVEN_TRIGGER_TICKS] = breakeven_trigger_ticks
        return params

    def test_no_trades_with_null_strategy(self):
        bars = make_flat_bars(390)
        trade_log = np.zeros(100, dtype=TRADE_LOG_DTYPE)
        params = self._make_params()
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
        params = self._make_params(target_ticks=20.0, contracts=1.0, buffer_ticks=0.0)
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

    def test_signal_fills_on_next_bar_open(self):
        n = 390
        base = 20000.0
        bars = list(make_flat_bars(n, base, 5.0))
        bars[0][16] = base + 40.0
        bars[1][16] = base + 60.0
        bars[2][16] = base + 35.0
        bars[3][16] = base + 50.0
        bars = tuple(bars)

        trade_log = np.zeros(10, dtype=TRADE_LOG_DTYPE)
        params = self._make_params(target_ticks=20.0, contracts=1.0, buffer_ticks=0.0)
        run_day_kernel(*bars, 7, 0, -1, always_long_strategy, trade_log, 0, 0.0, 0.0, params)

        assert trade_log[0]["entry_time"] == bars[5][16]
        assert trade_log[0]["entry_price"] > bars[0][16]

    def test_long_stop_gaps_through_to_next_open(self):
        n = 390
        base = 20000.0
        bars = list(make_flat_bars(n, base, 5.0))
        bars[0][16] = base + 40.0
        bars[1][16] = base + 45.0
        bars[2][16] = base + 36.0
        bars[3][16] = base + 42.0
        bars[0][17] = base - 80.0
        bars[1][17] = base - 70.0
        bars[2][17] = base - 90.0
        bars[3][17] = base - 75.0
        bars = tuple(bars)

        trade_log = np.zeros(10, dtype=TRADE_LOG_DTYPE)
        params = self._make_params(stop_ticks=20.0, target_ticks=400.0, contracts=1.0, buffer_ticks=0.0)
        run_day_kernel(*bars, 7, 0, -1, always_long_strategy, trade_log, 0, 0.0, 0.0, params)

        assert trade_log[0]["exit_reason"] == EXIT_STOP
        assert trade_log[0]["exit_time"] == bars[5][17]
        assert trade_log[0]["exit_price"] < trade_log[0]["entry_price"] - 20.0 * 0.25

    def test_target_exit_does_not_use_stop_penalty(self):
        n = 390
        base = 20000.0
        bars = list(make_flat_bars(n, base, 5.0))
        for i in range(16, n):
            bars[1][i] = base + 30.0
            bars[3][i] = base + 25.0
        bars = tuple(bars)

        params_lo = self._make_params(target_ticks=20.0, contracts=1.0, buffer_ticks=0.0, stop_penalty=1.0)
        params_hi = self._make_params(target_ticks=20.0, contracts=1.0, buffer_ticks=0.0, stop_penalty=10.0)

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
        params = self._make_params(stop_ticks=200.0, target_ticks=200.0, contracts=1.0, buffer_ticks=0.0)
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
        params = self._make_params(target_ticks=20.0, contracts=1.0, max_trades=3.0, buffer_ticks=0.0)
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
        params = self._make_params(contracts=5.0, daily_stop=-200.0, max_trades=3.0, buffer_ticks=0.0)
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
        params = self._make_params(target_ticks=20.0, contracts=1.0, buffer_ticks=0.0)
        n_trades, final_equity, final_pnl = run_day_kernel(
            *bars, 3, 0, -1, always_long_strategy, trade_log, 0, 0.0, 0.0, params
        )
        assert np.isclose(trade_log[:n_trades]["net_pnl"].sum(), final_equity)

    def test_breakeven_trigger_reduces_loss_after_trigger(self):
        n = 390
        base = 20000.0
        bars = list(make_flat_bars(n, base, 2.0))
        bars[1][16] = base + 20.0
        bars[2][16] = base + 14.0
        bars[3][16] = base + 18.0
        bars[1][17] = base + 16.0
        bars[2][17] = base - 1.0
        bars[3][17] = base
        bars = tuple(bars)

        no_be_log = np.zeros(10, dtype=TRADE_LOG_DTYPE)
        be_log = np.zeros(10, dtype=TRADE_LOG_DTYPE)
        no_be_params = self._make_params(stop_ticks=40.0, target_ticks=140.0, contracts=1.0, buffer_ticks=0.0)
        be_params = self._make_params(
            stop_ticks=40.0,
            target_ticks=140.0,
            contracts=1.0,
            buffer_ticks=0.0,
            breakeven_trigger_ticks=60.0,
        )

        run_day_kernel(*bars, 7, 0, -1, always_long_strategy, no_be_log, 0, 0.0, 0.0, no_be_params)
        run_day_kernel(*bars, 7, 0, -1, always_long_strategy, be_log, 0, 0.0, 0.0, be_params)

        assert be_log[0]["exit_reason"] == EXIT_STOP
        assert be_log[0]["net_pnl"] > no_be_log[0]["net_pnl"]
        assert be_log[0]["net_pnl"] > 0.0

    def test_portfolio_kernel_computes_dynamic_contracts_from_atr(self):
        n = 390
        base = 20000.0
        bars = list(make_flat_bars(n, base, 2.0))
        for i in range(16, n):
            bars[1][i] = base + 80.0
            bars[3][i] = base + 60.0
        bars[7][:] = 5.0
        bars[8][:] = 5.0
        bars = tuple(bars)
        daily_ratio = np.ones(n, dtype=np.float64)
        rvol = np.ones(n, dtype=np.float64)
        close_sma_50 = np.full(n, base, dtype=np.float64)
        daily_regime_bias = np.full(n, np.nan, dtype=np.float64)
        donchian_high_5 = np.full(n, base + 10.0, dtype=np.float64)
        donchian_low_5 = np.full(n, base - 10.0, dtype=np.float64)
        day_of_week = np.full(n, 1, dtype=np.int8)

        trade_log = np.zeros(20, dtype=TRADE_LOG_DTYPE)
        params = self._make_params(contracts=50.0, max_trades=1.0, buffer_ticks=0.0)
        strategy_profiles = np.zeros((2, PROFILE_ARRAY_LENGTH), dtype=np.float64)
        strategy_profiles[1, PROFILE_RISK_PER_TRADE_USD] = 400.0
        strategy_profiles[1, PROFILE_STOP_ATR_MULTIPLIER] = 2.0
        strategy_profiles[1, PROFILE_TARGET_ATR_MULTIPLIER] = 3.5
        strategy_profiles[1, PROFILE_BREAKEVEN_TRIGGER_TICKS] = 0.0
        strategy_profiles[1, PROFILE_RISK_BUFFER_FRACTION] = 0.15

        n_trades, _, _ = run_day_kernel_portfolio(
            bars[0], bars[1], bars[2], bars[3], bars[4],
            bars[5], bars[6], bars[7], bars[8], daily_ratio, rvol, close_sma_50,
            daily_regime_bias,
            donchian_high_5, donchian_low_5, day_of_week, bars[9],
            7, 0, -1, -2000.0, always_portfolio_breakout_strategy,
            trade_log, 0, 0.0, 0.0, params, strategy_profiles
        )

        assert n_trades >= 1
        assert trade_log[0]["contracts"] == 15

    def test_portfolio_kernel_fills_on_next_bar_open(self):
        n = 390
        base = 20000.0
        bars = list(make_flat_bars(n, base, 2.0))
        bars[0][16] = base + 40.0
        bars[1][16] = base + 80.0
        bars[2][16] = base + 35.0
        bars[3][16] = base + 60.0
        bars[7][:] = 5.0
        bars[8][:] = 5.0
        bars = tuple(bars)
        daily_ratio = np.ones(n, dtype=np.float64)
        rvol = np.ones(n, dtype=np.float64)
        close_sma_50 = np.full(n, base, dtype=np.float64)
        daily_regime_bias = np.full(n, np.nan, dtype=np.float64)
        donchian_high_5 = np.full(n, base + 10.0, dtype=np.float64)
        donchian_low_5 = np.full(n, base - 10.0, dtype=np.float64)
        day_of_week = np.full(n, 1, dtype=np.int8)

        trade_log = np.zeros(20, dtype=TRADE_LOG_DTYPE)
        params = self._make_params(contracts=50.0, max_trades=1.0, buffer_ticks=0.0)
        strategy_profiles = np.zeros((2, PROFILE_ARRAY_LENGTH), dtype=np.float64)
        strategy_profiles[1, PROFILE_RISK_PER_TRADE_USD] = 400.0
        strategy_profiles[1, PROFILE_STOP_ATR_MULTIPLIER] = 2.0
        strategy_profiles[1, PROFILE_TARGET_ATR_MULTIPLIER] = 3.5
        strategy_profiles[1, PROFILE_BREAKEVEN_TRIGGER_TICKS] = 0.0
        strategy_profiles[1, PROFILE_RISK_BUFFER_FRACTION] = 0.15

        n_trades, _, _ = run_day_kernel_portfolio(
            bars[0], bars[1], bars[2], bars[3], bars[4],
            bars[5], bars[6], bars[7], bars[8], daily_ratio, rvol, close_sma_50,
            daily_regime_bias,
            donchian_high_5, donchian_low_5, day_of_week, bars[9],
            7, 0, -1, -2000.0, always_portfolio_breakout_strategy,
            trade_log, 0, 0.0, 0.0, params, strategy_profiles
        )

        assert n_trades >= 1
        assert trade_log[0]["entry_time"] == bars[5][16]
