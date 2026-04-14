import numpy as np
from numba import njit
from propfirm.core.types import (
    MNQ_TICK_SIZE,
    PARAMS_RANGE_MINUTES, PARAMS_DAILY_TARGET, PARAMS_MAX_TRADES,
    PARAMS_BUFFER_TICKS, PARAMS_VOL_THRESHOLD,
)


@njit(cache=True)
def orb_signal(
    bar_idx,
    opens, highs, lows, closes, volumes,
    minute_of_day,
    equity, intraday_pnl, position, entry_price,
    halted, daily_trade_count,
    params,
):
    """Opening Range Breakout signal generator.

    Returns: 1 (long), -1 (short), 0 (no signal)
    """
    range_minutes = int(params[PARAMS_RANGE_MINUTES])
    daily_target = params[PARAMS_DAILY_TARGET]
    max_trades = int(params[PARAMS_MAX_TRADES])
    buffer_ticks = params[PARAMS_BUFFER_TICKS]
    volume_threshold = params[PARAMS_VOL_THRESHOLD]

    mod = minute_of_day[bar_idx]

    if mod < range_minutes:
        return 0
    if mod > 120:
        return 0

    if halted:
        return 0
    if position != 0:
        return 0
    if intraday_pnl >= daily_target:
        return 0
    if daily_trade_count >= max_trades:
        return 0

    day_start = bar_idx - mod
    daily_open = opens[day_start]
    range_high = -1e18
    range_low = 1e18
    for i in range(day_start, day_start + range_minutes):
        if i >= 0 and i < len(highs):
            if highs[i] > range_high:
                range_high = highs[i]
            if lows[i] < range_low:
                range_low = lows[i]

    buffer_points = buffer_ticks * MNQ_TICK_SIZE

    if volume_threshold > 0.0:
        avg_vol = 0.0
        count = 0
        for i in range(day_start, day_start + range_minutes):
            if i >= 0 and i < len(volumes):
                avg_vol += volumes[i]
                count += 1
        if count <= 0:
            return 0
        avg_vol /= count
        if avg_vol <= 0.0:
            return 0
        if volumes[bar_idx] / avg_vol < volume_threshold:
            return 0

    bar_close = closes[bar_idx]
    if bar_close > range_high + buffer_points and bar_close > daily_open:
        return 1
    if bar_close < range_low - buffer_points and bar_close < daily_open:
        return -1

    return 0
