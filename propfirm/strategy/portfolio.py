from numba import njit

from propfirm.strategy.mgc_h1_trend_strategy import mgc_h1_trend_signal


@njit(cache=True)
def combined_portfolio_signal(
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
    """Single-strategy portfolio wrapper for the parameterized MGC H1 trend strategy."""
    return mgc_h1_trend_signal(
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
    )
