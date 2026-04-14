import numpy as np
from numba import njit

from propfirm.core.types import (
    PARAMS_BAND_MULTIPLIER,
    PARAMS_DAILY_TARGET,
    PARAMS_MAX_TRADES,
    PARAMS_SMA_PERIOD,
    SIGNAL_LONG,
    SIGNAL_NONE,
    SIGNAL_SHORT,
)


@njit(cache=True)
def m6a_fade_signal(
    bar_idx,
    opens,
    highs,
    lows,
    closes,
    volumes,
    bar_atr,
    trailing_atr,
    daily_atr_ratio,
    minute_of_day,
    equity,
    intraday_pnl,
    position,
    entry_price,
    halted,
    daily_trade_count,
    params,
):
    """Bollinger-band fade for the M6A overnight session on 5-minute bars."""
    if halted or position != 0:
        return SIGNAL_NONE

    max_trades = int(params[PARAMS_MAX_TRADES])
    if daily_trade_count >= max_trades:
        return SIGNAL_NONE

    daily_target = params[PARAMS_DAILY_TARGET]
    if intraday_pnl >= daily_target:
        return SIGNAL_NONE

    bb_period = int(params[PARAMS_SMA_PERIOD])
    if bb_period <= 1 or bar_idx + 1 < bb_period:
        return SIGNAL_NONE

    band_multiplier = params[PARAMS_BAND_MULTIPLIER]
    if band_multiplier <= 0.0:
        return SIGNAL_NONE

    start = bar_idx - bb_period + 1
    window = closes[start:bar_idx + 1]
    sma = np.mean(window)
    std = np.std(window)
    upper = sma + band_multiplier * std
    lower = sma - band_multiplier * std
    close = closes[bar_idx]

    if close < lower:
        return SIGNAL_LONG
    if close > upper:
        return SIGNAL_SHORT
    return SIGNAL_NONE
