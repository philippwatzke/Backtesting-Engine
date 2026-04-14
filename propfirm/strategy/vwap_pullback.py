import numpy as np
from numba import njit
from propfirm.core.types import (
    PARAMS_DAILY_TARGET,
    PARAMS_MAX_TRADES,
    PARAMS_DISTANCE_TICKS,
    PARAMS_SMA_PERIOD,
    PARAMS_TICK_SIZE,
)


@njit(cache=True)
def _vwap_pullback_signal_impl(
    bar_idx,
    opens, highs, lows, closes, volumes,
    minute_of_day,
    intraday_pnl, position,
    halted, daily_trade_count,
    daily_target, max_trades, zone_points, sma_period,
):
    mod = minute_of_day[bar_idx]
    if mod < 15 or mod > 360:
        return 0
    if halted:
        return 0
    if position != 0:
        return 0
    if intraday_pnl >= daily_target:
        return 0
    if daily_trade_count >= max_trades:
        return 0
    if sma_period <= 0 or bar_idx + 1 < sma_period:
        return 0

    day_start = 0
    cum_pv = 0.0
    cum_vol = 0.0
    for i in range(day_start, bar_idx + 1):
        vol = float(volumes[i])
        typical_price = (highs[i] + lows[i] + closes[i]) / 3.0
        cum_pv += typical_price * vol
        cum_vol += vol
    if cum_vol <= 0.0:
        return 0
    vwap = cum_pv / cum_vol

    sma_sum = 0.0
    sma_start = bar_idx - sma_period + 1
    for i in range(sma_start, bar_idx + 1):
        sma_sum += closes[i]
    sma = sma_sum / sma_period

    bar_open = opens[bar_idx]
    bar_close = closes[bar_idx]
    bar_high = highs[bar_idx]
    bar_low = lows[bar_idx]

    long_zone = bar_low >= vwap - zone_points and bar_low <= vwap + zone_points
    if sma > vwap and bar_close > vwap and bar_close > bar_open and long_zone:
        return 1

    short_zone = bar_high >= vwap - zone_points and bar_high <= vwap + zone_points
    if sma < vwap and bar_close < vwap and bar_close < bar_open and short_zone:
        return -1

    return 0


@njit(cache=True)
def vwap_pullback_signal(
    bar_idx,
    opens, highs, lows, closes, volumes,
    bar_atr, trailing_atr,
    minute_of_day,
    equity, intraday_pnl, position, entry_price,
    halted, daily_trade_count,
    params,
):
    """VWAP pullback / hold signal generator.

    Returns: 1 (long), -1 (short), 0 (no signal)
    """
    daily_target = params[PARAMS_DAILY_TARGET]
    max_trades = int(params[PARAMS_MAX_TRADES])
    distance_ticks = params[PARAMS_DISTANCE_TICKS]
    tick_size = params[PARAMS_TICK_SIZE]
    sma_period = int(params[PARAMS_SMA_PERIOD])
    if tick_size <= 0.0:
        tick_size = 0.25
    zone_points = distance_ticks * tick_size
    return _vwap_pullback_signal_impl(
        bar_idx,
        opens, highs, lows, closes, volumes,
        minute_of_day,
        intraday_pnl, position,
        halted, daily_trade_count,
        daily_target, max_trades, zone_points, sma_period,
    )
