import numpy as np
from numba import njit

from propfirm.core.types import (
    PARAMS_DAILY_TARGET,
    PARAMS_MAX_TRADES,
    PARAMS_SMA_PERIOD,
    PARAMS_BAND_MULTIPLIER,
    PARAMS_POC_LOOKBACK,
)


@njit(cache=True)
def _vwap_poc_breakout_signal_impl(
    bar_idx,
    opens, highs, lows, closes, volumes,
    minute_of_day,
    intraday_pnl, position,
    halted, daily_trade_count,
    daily_target, max_trades, sma_period, band_multiplier, poc_lookback,
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
    if poc_lookback <= 0:
        return 0

    day_start = 0

    cum_pv = 0.0
    cum_vol = 0.0
    mean_tp = 0.0
    m2_tp = 0.0
    n_tp = 0
    for i in range(day_start, bar_idx + 1):
        typical_price = (highs[i] + lows[i] + closes[i]) / 3.0
        vol = float(volumes[i])
        cum_pv += typical_price * vol
        cum_vol += vol
        n_tp += 1
        delta = typical_price - mean_tp
        mean_tp += delta / n_tp
        delta2 = typical_price - mean_tp
        m2_tp += delta * delta2
    if cum_vol <= 0.0 or n_tp < 2:
        return 0

    vwap = cum_pv / cum_vol
    stddev = np.sqrt(m2_tp / n_tp)
    upper_band = vwap + band_multiplier * stddev
    lower_band = vwap - band_multiplier * stddev

    sma_sum = 0.0
    sma_start = bar_idx - sma_period + 1
    for i in range(sma_start, bar_idx + 1):
        sma_sum += closes[i]
    sma = sma_sum / sma_period

    poc_start = day_start
    if bar_idx - poc_lookback + 1 > poc_start:
        poc_start = bar_idx - poc_lookback + 1
    max_vol = -1.0
    poc_price = closes[bar_idx]
    for i in range(poc_start, bar_idx + 1):
        vol = float(volumes[i])
        if vol > max_vol:
            max_vol = vol
            poc_price = (highs[i] + lows[i] + closes[i]) / 3.0

    bar_open = opens[bar_idx]
    bar_close = closes[bar_idx]
    poc_in_band = lower_band < poc_price and poc_price < upper_band

    if (
        bar_close >= upper_band
        and poc_in_band
        and bar_close > sma
        and bar_open < upper_band
    ):
        return 1

    if (
        bar_close <= lower_band
        and poc_in_band
        and bar_close < sma
        and bar_open > lower_band
    ):
        return -1

    return 0


@njit(cache=True)
def vwap_poc_breakout_signal(
    bar_idx,
    opens, highs, lows, closes, volumes,
    bar_atr, trailing_atr,
    minute_of_day,
    equity, intraday_pnl, position, entry_price,
    halted, daily_trade_count,
    params,
):
    """VWAP + POC breakout signal generator."""
    daily_target = params[PARAMS_DAILY_TARGET]
    max_trades = int(params[PARAMS_MAX_TRADES])
    sma_period = int(params[PARAMS_SMA_PERIOD])
    band_multiplier = params[PARAMS_BAND_MULTIPLIER]
    poc_lookback = int(params[PARAMS_POC_LOOKBACK])
    return _vwap_poc_breakout_signal_impl(
        bar_idx,
        opens, highs, lows, closes, volumes,
        minute_of_day,
        intraday_pnl, position,
        halted, daily_trade_count,
        daily_target, max_trades, sma_period, band_multiplier, poc_lookback,
    )
