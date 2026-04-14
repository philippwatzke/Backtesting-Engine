from numba import njit

from propfirm.core.types import (
    PARAMS_DAILY_TARGET,
    PARAMS_MAX_TRADES,
    PARAMS_RANGE_MINUTES,
    PARAMS_TRIGGER_END_MINUTE,
    PARAMS_TRIGGER_START_MINUTE,
    SIGNAL_LONG,
    SIGNAL_NONE,
    SIGNAL_SHORT,
)


@njit(cache=True)
def mcl_orb_fade_signal(
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
    """Opening range fade on the first 30 available MCL session minutes."""
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

    if mod < range_minutes or mod < trigger_start or mod > trigger_end:
        return SIGNAL_NONE

    day_start = bar_idx - mod
    range_high = -1e18
    range_low = 1e18
    for i in range(day_start, day_start + range_minutes):
        if highs[i] > range_high:
            range_high = highs[i]
        if lows[i] < range_low:
            range_low = lows[i]

    fakeout_long = highs[bar_idx] > range_high and closes[bar_idx] < range_high
    fakeout_short = lows[bar_idx] < range_low and closes[bar_idx] > range_low
    if fakeout_long and fakeout_short:
        return SIGNAL_NONE
    if fakeout_long:
        return SIGNAL_SHORT
    if fakeout_short:
        return SIGNAL_LONG
    return SIGNAL_NONE
