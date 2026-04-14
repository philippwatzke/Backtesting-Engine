import numpy as np
from numba import njit

from propfirm.core.types import (
    PARAMS_BLOCKED_WEEKDAY,
    PARAMS_DAILY_TARGET,
    PARAMS_MAX_TRADES,
    PARAMS_POC_LOOKBACK,
    PARAMS_TRIGGER_END_MINUTE,
    PARAMS_TRIGGER_START_MINUTE,
    SIGNAL_LONG,
    SIGNAL_NONE,
    SIGNAL_SHORT,
)


@njit(cache=True)
def mgc_h1_trend_signal(
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
    close_sma_50,
    daily_regime_bias,
    donchian_high_5,
    donchian_low_5,
    minute_of_day,
    day_of_week,
    equity,
    intraday_pnl,
    position,
    entry_price,
    halted,
    daily_trade_count,
    params,
):
    """H1 Donchian breakout for MGC gated by a slow trend filter."""
    if halted or position != 0:
        return SIGNAL_NONE

    blocked_weekday = int(params[PARAMS_BLOCKED_WEEKDAY])
    if blocked_weekday >= 0 and int(day_of_week[bar_idx]) == blocked_weekday:
        return SIGNAL_NONE

    entry_minute = minute_of_day[bar_idx]
    trigger_start_minute = int(params[PARAMS_TRIGGER_START_MINUTE])
    trigger_end_minute = int(params[PARAMS_TRIGGER_END_MINUTE])
    if entry_minute < trigger_start_minute or entry_minute > trigger_end_minute:
        return SIGNAL_NONE

    max_trades = int(params[PARAMS_MAX_TRADES])
    if daily_trade_count >= max_trades:
        return SIGNAL_NONE

    daily_target = params[PARAMS_DAILY_TARGET]
    if intraday_pnl >= daily_target:
        return SIGNAL_NONE

    sma_value = close_sma_50[bar_idx]
    channel_high = donchian_high_5[bar_idx]
    channel_low = donchian_low_5[bar_idx]
    if not np.isfinite(sma_value) or not np.isfinite(channel_high) or not np.isfinite(channel_low):
        return SIGNAL_NONE

    regime_bias = daily_regime_bias[bar_idx]
    close = closes[bar_idx]
    if close > channel_high and close > sma_value:
        if np.isfinite(regime_bias) and regime_bias <= 0.0:
            return SIGNAL_NONE
        return SIGNAL_LONG
    if close < channel_low and close < sma_value:
        if np.isfinite(regime_bias) and regime_bias >= 0.0:
            return SIGNAL_NONE
        return SIGNAL_SHORT
    return SIGNAL_NONE
