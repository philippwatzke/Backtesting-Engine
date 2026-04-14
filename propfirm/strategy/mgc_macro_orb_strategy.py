from numba import njit

from propfirm.core.types import (
    PARAMS_DAILY_TARGET,
    PARAMS_MAX_RVOL,
    PARAMS_MAX_TRADES,
    PARAMS_MIN_RVOL,
    PARAMS_RANGE_MINUTES,
    PARAMS_TRIGGER_END_MINUTE,
    PARAMS_TRIGGER_START_MINUTE,
    SIGNAL_LONG,
    SIGNAL_NONE,
    SIGNAL_SHORT,
)


@njit(cache=True)
def mgc_macro_orb_signal(
    bar_idx,
    opens,
    highs,
    lows,
    closes,
    volumes,
    bar_atr,
    trailing_atr,
    daily_atr_ratio,
    rvol,
    minute_of_day,
    equity,
    intraday_pnl,
    position,
    entry_price,
    halted,
    daily_trade_count,
    params,
):
    """Relative-volume filtered macro breakout for MGC on the London/New York overlap."""
    if halted or position != 0:
        return SIGNAL_NONE

    max_trades = int(params[PARAMS_MAX_TRADES])
    if daily_trade_count >= max_trades:
        return SIGNAL_NONE

    daily_target = params[PARAMS_DAILY_TARGET]
    if intraday_pnl >= daily_target:
        return SIGNAL_NONE

    mod = minute_of_day[bar_idx]
    range_minutes = int(params[PARAMS_RANGE_MINUTES])
    trigger_start = int(params[PARAMS_TRIGGER_START_MINUTE])
    trigger_end = int(params[PARAMS_TRIGGER_END_MINUTE])
    min_rvol = params[PARAMS_MIN_RVOL]
    max_rvol = params[PARAMS_MAX_RVOL]

    if mod < range_minutes or mod < trigger_start or mod > trigger_end:
        return SIGNAL_NONE
    if rvol[bar_idx] < min_rvol or rvol[bar_idx] > max_rvol:
        return SIGNAL_NONE

    day_start = bar_idx - mod
    range_high = -1e18
    range_low = 1e18
    for i in range(day_start, day_start + range_minutes):
        if highs[i] > range_high:
            range_high = highs[i]
        if lows[i] < range_low:
            range_low = lows[i]

    broke_high = highs[bar_idx] > range_high
    broke_low = lows[bar_idx] < range_low
    if broke_high and broke_low:
        return SIGNAL_NONE
    if broke_high:
        return SIGNAL_LONG
    if broke_low:
        return SIGNAL_SHORT
    return SIGNAL_NONE
