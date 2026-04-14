from numba import njit

from propfirm.core.types import (
    MOC_ENTRY_MINUTE,
    PARAMS_DAILY_TARGET,
    PARAMS_ENTRY_MINUTE,
    PARAMS_MAX_TRADES,
    PARAMS_TREND_THRESHOLD_PCT,
    SIGNAL_MOC_LONG,
    SIGNAL_MOC_SHORT,
    SIGNAL_NONE,
)


@njit(cache=True)
def _moc_flow_signal_impl(
    bar_idx,
    opens,
    minute_of_day,
    intraday_pnl,
    position,
    halted,
    daily_trade_count,
    daily_target,
    max_trades,
    entry_minute,
    trend_threshold_pct,
):
    """Trade only the 15:00 cash-session flow on a 5-minute chart."""
    if halted or position != 0 or daily_trade_count >= max_trades:
        return SIGNAL_NONE
    if intraday_pnl >= daily_target:
        return SIGNAL_NONE
    if minute_of_day[bar_idx] != entry_minute:
        return SIGNAL_NONE

    session_open = opens[0]
    current_price = opens[bar_idx]
    if session_open <= 0.0:
        return SIGNAL_NONE

    change_pct = (current_price - session_open) / session_open
    if change_pct >= trend_threshold_pct:
        return SIGNAL_MOC_LONG
    if change_pct <= -trend_threshold_pct:
        return SIGNAL_MOC_SHORT
    return SIGNAL_NONE


@njit(cache=True)
def moc_flow_signal(
    bar_idx,
    opens,
    highs,
    lows,
    closes,
    volumes,
    bar_atr,
    trailing_atr,
    minute_of_day,
    equity,
    intraday_pnl,
    position,
    entry_price,
    halted,
    daily_trade_count,
    params,
):
    daily_target = params[PARAMS_DAILY_TARGET]
    max_trades = int(params[PARAMS_MAX_TRADES])
    entry_minute = int(params[PARAMS_ENTRY_MINUTE])
    if entry_minute < 0:
        entry_minute = MOC_ENTRY_MINUTE
    trend_threshold_pct = params[PARAMS_TREND_THRESHOLD_PCT]
    return _moc_flow_signal_impl(
        bar_idx,
        opens,
        minute_of_day,
        intraday_pnl,
        position,
        halted,
        daily_trade_count,
        daily_target,
        max_trades,
        entry_minute,
        trend_threshold_pct,
    )
