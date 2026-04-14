import numpy as np
from numba import njit
from propfirm.core.types import (
    TRADE_LOG_DTYPE, SIGNAL_LONG, SIGNAL_SHORT, SIGNAL_NONE,
    SIGNAL_PULLBACK_LONG, SIGNAL_PULLBACK_SHORT,
    SIGNAL_POC_BREAKOUT_LONG, SIGNAL_POC_BREAKOUT_SHORT,
    EXIT_TARGET, EXIT_STOP, EXIT_HARD_CLOSE, EXIT_CIRCUIT_BREAKER,
    HARD_CLOSE_MINUTE, MNQ_TICK_SIZE, MNQ_TICK_VALUE,
    PARAMS_RANGE_MINUTES, PARAMS_STOP_TICKS, PARAMS_TARGET_TICKS,
    PARAMS_CONTRACTS, PARAMS_DAILY_STOP, PARAMS_DAILY_TARGET,
    PARAMS_MAX_TRADES, PARAMS_STOP_PENALTY, PARAMS_COMMISSION,
    PARAMS_BREAKEVEN_TRIGGER_TICKS, PARAMS_EXTRA_SLIPPAGE_TICKS,
    PARAMS_ENTRY_MINUTE, PARAMS_TIME_STOP_MINUTE, PARAMS_TICK_SIZE,
    PARAMS_TICK_VALUE, PARAMS_TRAIL_BAR_EXTREME, PARAMS_ENTRY_ON_CLOSE,
    PROFILE_RISK_PER_TRADE_USD, PROFILE_STOP_ATR_MULTIPLIER,
    PROFILE_TARGET_ATR_MULTIPLIER, PROFILE_BREAKEVEN_TRIGGER_TICKS,
    PROFILE_RISK_BUFFER_FRACTION,
)
from propfirm.market.slippage import compute_slippage as _compute_slippage
from propfirm.risk.risk import check_circuit_breaker


@njit(cache=True)
def run_day_kernel(
    opens, highs, lows, closes, volumes,
    timestamps, minute_of_day, bar_atr, trailing_atr, slippage_lookup,
    current_day_id,
    current_phase_id,
    current_payout_cycle_id,
    strategy_fn, trade_log, trade_log_offset,
    starting_equity, starting_pnl,
    params,
):
    """Run the bar-loop for a single trading day.

    params is a flat float64 array.
    Returns: (n_trades, final_equity, final_intraday_pnl)
    """
    stop_ticks = params[PARAMS_STOP_TICKS]
    target_ticks = params[PARAMS_TARGET_TICKS]
    contracts = int(params[PARAMS_CONTRACTS])
    daily_stop = params[PARAMS_DAILY_STOP]
    daily_target = params[PARAMS_DAILY_TARGET]
    max_trades = int(params[PARAMS_MAX_TRADES])
    stop_penalty = params[PARAMS_STOP_PENALTY]
    commission_per_side = params[PARAMS_COMMISSION]
    tick_size = params[PARAMS_TICK_SIZE]
    if tick_size <= 0.0:
        tick_size = MNQ_TICK_SIZE
    tick_value = params[PARAMS_TICK_VALUE]
    if tick_value <= 0.0:
        tick_value = MNQ_TICK_VALUE
    time_stop_minute = int(params[PARAMS_TIME_STOP_MINUTE])
    if time_stop_minute <= 0:
        time_stop_minute = HARD_CLOSE_MINUTE
    breakeven_trigger_ticks = params[PARAMS_BREAKEVEN_TRIGGER_TICKS]
    extra_slippage_points = params[PARAMS_EXTRA_SLIPPAGE_TICKS] * tick_size
    entry_on_close = params[PARAMS_ENTRY_ON_CLOSE] > 0.5
    breakeven_trigger_points = breakeven_trigger_ticks * tick_size
    breakeven_offset_points = 2.0 * tick_size

    n_bars = len(opens)
    equity = starting_equity
    intraday_pnl = starting_pnl
    entry_price = 0.0
    position = 0
    halted = False
    daily_trade_count = 0
    trade_idx = trade_log_offset
    open_trade_idx = -1
    stop_level = 0.0
    target_level = 0.0
    breakeven_active = False
    pending_signal = SIGNAL_NONE

    for bar_idx in range(n_bars):
        mod = minute_of_day[bar_idx]
        is_last_bar = bar_idx == n_bars - 1
        bar_open = opens[bar_idx]
        bar_high = highs[bar_idx]
        bar_low = lows[bar_idx]
        bar_close = closes[bar_idx]
        current_atr = bar_atr[bar_idx]
        current_trailing = trailing_atr[bar_idx]
        exited_this_bar = False

        if position == 0 and pending_signal != SIGNAL_NONE and not entry_on_close:
            if halted or mod >= time_stop_minute:
                pending_signal = SIGNAL_NONE
            else:
                if trade_idx >= len(trade_log):
                    raise RuntimeError("trade_log capacity exceeded")
                slip = _compute_slippage(
                    mod, current_atr, current_trailing, slippage_lookup,
                    False, stop_penalty, tick_size, extra_slippage_points,
                )
                entry_commission = commission_per_side * contracts
                if pending_signal == SIGNAL_LONG:
                    fill_price = bar_open + slip
                    stop_level = fill_price - stop_ticks * tick_size
                    target_level = fill_price + target_ticks * tick_size
                    position = contracts
                else:
                    fill_price = bar_open - slip
                    stop_level = fill_price + stop_ticks * tick_size
                    target_level = fill_price - target_ticks * tick_size
                    position = -contracts

                entry_price = fill_price
                intraday_pnl -= entry_commission
                equity -= entry_commission
                daily_trade_count += 1
                breakeven_active = False

                open_trade_idx = trade_idx
                trade_log[open_trade_idx]["day_id"] = current_day_id
                trade_log[open_trade_idx]["phase_id"] = current_phase_id
                trade_log[open_trade_idx]["payout_cycle_id"] = current_payout_cycle_id
                trade_log[open_trade_idx]["entry_time"] = timestamps[bar_idx]
                trade_log[open_trade_idx]["entry_price"] = fill_price
                trade_log[open_trade_idx]["entry_slippage"] = slip
                trade_log[open_trade_idx]["entry_commission"] = entry_commission
                trade_log[open_trade_idx]["contracts"] = contracts
                trade_log[open_trade_idx]["signal_type"] = pending_signal
                trade_idx += 1
                pending_signal = SIGNAL_NONE

        # --- Check exits for open position ---
        if position != 0:
            exit_price = 0.0
            exit_reason = -1

            if mod >= time_stop_minute:
                if entry_on_close:
                    slip = 0.0
                else:
                    slip = _compute_slippage(
                        mod, current_atr, current_trailing, slippage_lookup,
                        False, stop_penalty, tick_size, extra_slippage_points,
                    )
                if position > 0:
                    exit_price = bar_close if entry_on_close else bar_open - slip
                else:
                    exit_price = bar_close if entry_on_close else bar_open + slip
                exit_reason = EXIT_HARD_CLOSE
            elif position > 0:  # Long
                if bar_open <= stop_level:
                    if entry_on_close:
                        slip = 0.0
                    else:
                        slip = _compute_slippage(
                            mod, current_atr, current_trailing, slippage_lookup,
                            True, stop_penalty, tick_size, extra_slippage_points,
                        )
                    exit_price = bar_open if entry_on_close else bar_open - slip
                    exit_reason = EXIT_STOP
                elif bar_low <= stop_level:
                    if entry_on_close:
                        slip = 0.0
                    else:
                        slip = _compute_slippage(
                            mod, current_atr, current_trailing, slippage_lookup,
                            True, stop_penalty, tick_size, extra_slippage_points,
                        )
                    exit_price = stop_level if entry_on_close else stop_level - slip
                    exit_reason = EXIT_STOP
                elif bar_high >= target_level:
                    if entry_on_close:
                        slip = 0.0
                    else:
                        slip = _compute_slippage(
                            mod, current_atr, current_trailing, slippage_lookup,
                            False, stop_penalty, tick_size, extra_slippage_points,
                        )
                    exit_price = target_level if entry_on_close else target_level - slip
                    exit_reason = EXIT_TARGET
            else:  # Short
                if bar_open >= stop_level:
                    if entry_on_close:
                        slip = 0.0
                    else:
                        slip = _compute_slippage(
                            mod, current_atr, current_trailing, slippage_lookup,
                            True, stop_penalty, tick_size, extra_slippage_points,
                        )
                    exit_price = bar_open if entry_on_close else bar_open + slip
                    exit_reason = EXIT_STOP
                elif bar_high >= stop_level:
                    if entry_on_close:
                        slip = 0.0
                    else:
                        slip = _compute_slippage(
                            mod, current_atr, current_trailing, slippage_lookup,
                            True, stop_penalty, tick_size, extra_slippage_points,
                        )
                    exit_price = stop_level if entry_on_close else stop_level + slip
                    exit_reason = EXIT_STOP
                elif bar_low <= target_level:
                    if entry_on_close:
                        slip = 0.0
                    else:
                        slip = _compute_slippage(
                            mod, current_atr, current_trailing, slippage_lookup,
                            False, stop_penalty, tick_size, extra_slippage_points,
                        )
                    exit_price = target_level if entry_on_close else target_level + slip
                    exit_reason = EXIT_TARGET

            # Hard close on the final bar of the retained session.
            if exit_reason == -1 and is_last_bar:
                if entry_on_close:
                    slip = 0.0
                else:
                    slip = _compute_slippage(
                        mod, current_atr, current_trailing, slippage_lookup,
                        False, stop_penalty, tick_size, extra_slippage_points,
                    )
                if position > 0:
                    exit_price = bar_close - slip
                else:
                    exit_price = bar_close + slip
                exit_reason = EXIT_HARD_CLOSE

            if exit_reason >= 0:
                abs_contracts = abs(position)
                exit_commission = commission_per_side * abs_contracts
                entry_comm_logged = commission_per_side * abs_contracts
                if exit_reason == EXIT_STOP and breakeven_active:
                    breakeven_points = (entry_comm_logged + exit_commission) * tick_size / (
                        abs_contracts * tick_value
                    )
                    if position > 0:
                        min_exit_price = entry_price + breakeven_points
                        if exit_price < min_exit_price:
                            exit_price = min_exit_price
                    else:
                        max_exit_price = entry_price - breakeven_points
                        if exit_price > max_exit_price:
                            exit_price = max_exit_price

                if position > 0:
                    gross_pnl = (exit_price - entry_price) * abs_contracts / tick_size * tick_value
                else:
                    gross_pnl = (entry_price - exit_price) * abs_contracts / tick_size * tick_value
                net_pnl = gross_pnl - entry_comm_logged - exit_commission

                if open_trade_idx < 0:
                    raise RuntimeError("open_trade_idx missing on exit")
                trade_log[open_trade_idx]["day_id"] = current_day_id
                trade_log[open_trade_idx]["phase_id"] = current_phase_id
                trade_log[open_trade_idx]["payout_cycle_id"] = current_payout_cycle_id
                trade_log[open_trade_idx]["exit_time"] = timestamps[bar_idx]
                trade_log[open_trade_idx]["exit_price"] = exit_price
                trade_log[open_trade_idx]["exit_slippage"] = slip
                trade_log[open_trade_idx]["exit_commission"] = exit_commission
                trade_log[open_trade_idx]["gross_pnl"] = gross_pnl
                trade_log[open_trade_idx]["net_pnl"] = net_pnl
                trade_log[open_trade_idx]["exit_reason"] = exit_reason

                intraday_pnl += gross_pnl - exit_commission
                equity += gross_pnl - exit_commission
                position = 0
                entry_price = 0.0
                open_trade_idx = -1
                stop_level = 0.0
                target_level = 0.0
                breakeven_active = False
                exited_this_bar = True

        # --- Circuit breaker check ---
        if not halted and check_circuit_breaker(intraday_pnl, daily_stop):
            halted = True

        if exited_this_bar:
            continue

        if position != 0 and not breakeven_active and breakeven_trigger_ticks > 0.0:
            if position > 0 and bar_high >= entry_price + breakeven_trigger_points:
                be_stop = entry_price + breakeven_offset_points
                if be_stop > stop_level:
                    stop_level = be_stop
                breakeven_active = True
            elif position < 0 and bar_low <= entry_price - breakeven_trigger_points:
                be_stop = entry_price - breakeven_offset_points
                if be_stop < stop_level:
                    stop_level = be_stop
                breakeven_active = True

        # --- No new entries on or after the configured time stop ---
        if is_last_bar or mod >= time_stop_minute:
            continue

        # --- Signal generation ---
        if position == 0 and pending_signal == SIGNAL_NONE and not halted and daily_trade_count < max_trades:
            signal = strategy_fn(
                bar_idx, opens, highs, lows, closes, volumes,
                bar_atr, trailing_atr,
                minute_of_day, equity, intraday_pnl, position,
                entry_price, halted, daily_trade_count, params
            )

            if signal == SIGNAL_LONG or signal == SIGNAL_SHORT:
                if entry_on_close:
                    if trade_idx >= len(trade_log):
                        raise RuntimeError("trade_log capacity exceeded")
                    entry_commission = commission_per_side * contracts
                    if signal == SIGNAL_LONG:
                        fill_price = bar_close
                        stop_level = fill_price - stop_ticks * tick_size
                        target_level = fill_price + target_ticks * tick_size
                        position = contracts
                    else:
                        fill_price = bar_close
                        stop_level = fill_price + stop_ticks * tick_size
                        target_level = fill_price - target_ticks * tick_size
                        position = -contracts

                    entry_price = fill_price
                    intraday_pnl -= entry_commission
                    equity -= entry_commission
                    daily_trade_count += 1
                    breakeven_active = False

                    open_trade_idx = trade_idx
                    trade_log[open_trade_idx]["day_id"] = current_day_id
                    trade_log[open_trade_idx]["phase_id"] = current_phase_id
                    trade_log[open_trade_idx]["payout_cycle_id"] = current_payout_cycle_id
                    trade_log[open_trade_idx]["entry_time"] = timestamps[bar_idx]
                    trade_log[open_trade_idx]["entry_price"] = fill_price
                    trade_log[open_trade_idx]["entry_slippage"] = 0.0
                    trade_log[open_trade_idx]["entry_commission"] = entry_commission
                    trade_log[open_trade_idx]["contracts"] = contracts
                    trade_log[open_trade_idx]["signal_type"] = signal
                    trade_idx += 1
                else:
                    pending_signal = signal

    n_trades = trade_idx - trade_log_offset
    return n_trades, equity, intraday_pnl


@njit(cache=True)
def run_day_kernel_portfolio(
    opens, highs, lows, closes, volumes,
    timestamps, minute_of_day, bar_atr, trailing_atr, daily_atr_ratio, rvol, close_sma_50,
    daily_regime_bias,
    donchian_high_5, donchian_low_5, day_of_week, slippage_lookup,
    current_day_id,
    current_phase_id,
    current_payout_cycle_id,
    liquidation_floor_equity,
    strategy_fn, trade_log, trade_log_offset,
    starting_equity, starting_pnl,
    params,
    strategy_profiles,
):
    """Run the bar-loop for a single trading day with multiple trade profiles."""
    daily_stop = params[PARAMS_DAILY_STOP]
    daily_target = params[PARAMS_DAILY_TARGET]
    max_trades = int(params[PARAMS_MAX_TRADES])
    stop_penalty = params[PARAMS_STOP_PENALTY]
    commission_per_side = params[PARAMS_COMMISSION]
    tick_size = params[PARAMS_TICK_SIZE]
    if tick_size <= 0.0:
        tick_size = MNQ_TICK_SIZE
    tick_value = params[PARAMS_TICK_VALUE]
    if tick_value <= 0.0:
        tick_value = MNQ_TICK_VALUE
    time_stop_minute = int(params[PARAMS_TIME_STOP_MINUTE])
    if time_stop_minute <= 0:
        time_stop_minute = HARD_CLOSE_MINUTE
    extra_slippage_points = params[PARAMS_EXTRA_SLIPPAGE_TICKS] * tick_size
    entry_on_close = params[PARAMS_ENTRY_ON_CLOSE] > 0.5
    breakeven_offset_points = 2.0 * tick_size
    trail_bar_extreme = params[PARAMS_TRAIL_BAR_EXTREME] > 0.5

    n_bars = len(opens)
    equity = starting_equity
    intraday_pnl = starting_pnl
    entry_price = 0.0
    position = 0
    halted = False
    daily_trade_count = 0
    trade_idx = trade_log_offset
    open_trade_idx = -1
    stop_level = 0.0
    target_level = 0.0
    current_breakeven_trigger_points = 0.0
    breakeven_active = False
    pending_signal = SIGNAL_NONE
    pending_contracts = 0
    pending_stop_ticks = 0.0
    pending_target_ticks = 0.0
    pending_breakeven_trigger_points = 0.0

    for bar_idx in range(n_bars):
        mod = minute_of_day[bar_idx]
        is_last_bar = bar_idx == n_bars - 1
        bar_open = opens[bar_idx]
        bar_high = highs[bar_idx]
        bar_low = lows[bar_idx]
        bar_close = closes[bar_idx]
        current_atr = bar_atr[bar_idx]
        current_trailing = trailing_atr[bar_idx]
        exited_this_bar = False

        if position == 0 and pending_signal != SIGNAL_NONE and not entry_on_close:
            if halted or mod >= time_stop_minute:
                pending_signal = SIGNAL_NONE
                pending_contracts = 0
                pending_stop_ticks = 0.0
                pending_target_ticks = 0.0
                pending_breakeven_trigger_points = 0.0
            else:
                if pending_contracts <= 0:
                    raise RuntimeError("pending entry contracts must be positive")
                if trade_idx >= len(trade_log):
                    raise RuntimeError("trade_log capacity exceeded")
                slip = _compute_slippage(
                    mod, current_atr, current_trailing, slippage_lookup,
                    False, stop_penalty, tick_size, extra_slippage_points,
                )
                entry_commission = commission_per_side * pending_contracts
                current_breakeven_trigger_points = pending_breakeven_trigger_points

                if pending_signal > 0:
                    fill_price = bar_open + slip
                    stop_level = fill_price - pending_stop_ticks * tick_size
                    target_level = fill_price + pending_target_ticks * tick_size
                    position = pending_contracts
                else:
                    fill_price = bar_open - slip
                    stop_level = fill_price + pending_stop_ticks * tick_size
                    target_level = fill_price - pending_target_ticks * tick_size
                    position = -pending_contracts

                entry_price = fill_price
                intraday_pnl -= entry_commission
                equity -= entry_commission
                daily_trade_count += 1
                breakeven_active = False

                open_trade_idx = trade_idx
                trade_log[open_trade_idx]["day_id"] = current_day_id
                trade_log[open_trade_idx]["phase_id"] = current_phase_id
                trade_log[open_trade_idx]["payout_cycle_id"] = current_payout_cycle_id
                trade_log[open_trade_idx]["entry_time"] = timestamps[bar_idx]
                trade_log[open_trade_idx]["entry_price"] = fill_price
                trade_log[open_trade_idx]["entry_slippage"] = slip
                trade_log[open_trade_idx]["entry_commission"] = entry_commission
                trade_log[open_trade_idx]["contracts"] = pending_contracts
                trade_log[open_trade_idx]["signal_type"] = pending_signal
                trade_idx += 1

                pending_signal = SIGNAL_NONE
                pending_contracts = 0
                pending_stop_ticks = 0.0
                pending_target_ticks = 0.0
                pending_breakeven_trigger_points = 0.0

        if position != 0:
            exit_price = 0.0
            exit_reason = -1

            if mod >= time_stop_minute:
                if entry_on_close:
                    slip = 0.0
                else:
                    slip = _compute_slippage(
                        mod, current_atr, current_trailing, slippage_lookup,
                        False, stop_penalty, tick_size, extra_slippage_points,
                    )
                if position > 0:
                    exit_price = bar_close if entry_on_close else bar_open - slip
                else:
                    exit_price = bar_close if entry_on_close else bar_open + slip
                exit_reason = EXIT_HARD_CLOSE
            elif position > 0:
                if bar_open <= stop_level:
                    if entry_on_close:
                        slip = 0.0
                    else:
                        slip = _compute_slippage(
                            mod, current_atr, current_trailing, slippage_lookup,
                            True, stop_penalty, tick_size, extra_slippage_points,
                        )
                    exit_price = bar_open if entry_on_close else bar_open - slip
                    exit_reason = EXIT_STOP
                elif bar_low <= stop_level:
                    if entry_on_close:
                        slip = 0.0
                    else:
                        slip = _compute_slippage(
                            mod, current_atr, current_trailing, slippage_lookup,
                            True, stop_penalty, tick_size, extra_slippage_points,
                        )
                    exit_price = stop_level if entry_on_close else stop_level - slip
                    exit_reason = EXIT_STOP
                elif bar_high >= target_level:
                    if entry_on_close:
                        slip = 0.0
                    else:
                        slip = _compute_slippage(
                            mod, current_atr, current_trailing, slippage_lookup,
                            False, stop_penalty, tick_size, extra_slippage_points,
                        )
                    exit_price = target_level if entry_on_close else target_level - slip
                    exit_reason = EXIT_TARGET
            else:
                if bar_open >= stop_level:
                    if entry_on_close:
                        slip = 0.0
                    else:
                        slip = _compute_slippage(
                            mod, current_atr, current_trailing, slippage_lookup,
                            True, stop_penalty, tick_size, extra_slippage_points,
                        )
                    exit_price = bar_open if entry_on_close else bar_open + slip
                    exit_reason = EXIT_STOP
                elif bar_high >= stop_level:
                    if entry_on_close:
                        slip = 0.0
                    else:
                        slip = _compute_slippage(
                            mod, current_atr, current_trailing, slippage_lookup,
                            True, stop_penalty, tick_size, extra_slippage_points,
                        )
                    exit_price = stop_level if entry_on_close else stop_level + slip
                    exit_reason = EXIT_STOP
                elif bar_low <= target_level:
                    if entry_on_close:
                        slip = 0.0
                    else:
                        slip = _compute_slippage(
                            mod, current_atr, current_trailing, slippage_lookup,
                            False, stop_penalty, tick_size, extra_slippage_points,
                        )
                    exit_price = target_level if entry_on_close else target_level + slip
                    exit_reason = EXIT_TARGET

            if exit_reason == -1 and is_last_bar:
                if entry_on_close:
                    slip = 0.0
                else:
                    slip = _compute_slippage(
                        mod, current_atr, current_trailing, slippage_lookup,
                        False, stop_penalty, tick_size, extra_slippage_points,
                    )
                if position > 0:
                    exit_price = bar_close - slip
                else:
                    exit_price = bar_close + slip
                exit_reason = EXIT_HARD_CLOSE

            if exit_reason >= 0:
                abs_contracts = abs(position)
                exit_commission = commission_per_side * abs_contracts
                entry_comm_logged = commission_per_side * abs_contracts
                if exit_reason == EXIT_STOP and breakeven_active:
                    breakeven_points = (entry_comm_logged + exit_commission) * tick_size / (
                        abs_contracts * tick_value
                    )
                    if position > 0:
                        min_exit_price = entry_price + breakeven_points
                        if exit_price < min_exit_price:
                            exit_price = min_exit_price
                    else:
                        max_exit_price = entry_price - breakeven_points
                        if exit_price > max_exit_price:
                            exit_price = max_exit_price

                if position > 0:
                    gross_pnl = (exit_price - entry_price) * abs_contracts / tick_size * tick_value
                else:
                    gross_pnl = (entry_price - exit_price) * abs_contracts / tick_size * tick_value
                net_pnl = gross_pnl - entry_comm_logged - exit_commission

                if open_trade_idx < 0:
                    raise RuntimeError("open_trade_idx missing on exit")
                trade_log[open_trade_idx]["day_id"] = current_day_id
                trade_log[open_trade_idx]["phase_id"] = current_phase_id
                trade_log[open_trade_idx]["payout_cycle_id"] = current_payout_cycle_id
                trade_log[open_trade_idx]["exit_time"] = timestamps[bar_idx]
                trade_log[open_trade_idx]["exit_price"] = exit_price
                trade_log[open_trade_idx]["exit_slippage"] = slip
                trade_log[open_trade_idx]["exit_commission"] = exit_commission
                trade_log[open_trade_idx]["gross_pnl"] = gross_pnl
                trade_log[open_trade_idx]["net_pnl"] = net_pnl
                trade_log[open_trade_idx]["exit_reason"] = exit_reason

                intraday_pnl += gross_pnl - exit_commission
                equity += gross_pnl - exit_commission
                position = 0
                entry_price = 0.0
                open_trade_idx = -1
                stop_level = 0.0
                target_level = 0.0
                current_breakeven_trigger_points = 0.0
                breakeven_active = False
                exited_this_bar = True

        if not halted and check_circuit_breaker(intraday_pnl, daily_stop):
            halted = True

        if exited_this_bar:
            continue

        if position != 0 and not breakeven_active and current_breakeven_trigger_points > 0.0:
            if position > 0 and bar_high >= entry_price + current_breakeven_trigger_points:
                be_stop = entry_price + breakeven_offset_points
                if be_stop > stop_level:
                    stop_level = be_stop
                breakeven_active = True
            elif position < 0 and bar_low <= entry_price - current_breakeven_trigger_points:
                be_stop = entry_price - breakeven_offset_points
                if be_stop < stop_level:
                    stop_level = be_stop
                breakeven_active = True

        if position != 0 and trail_bar_extreme:
            if position > 0 and bar_low > stop_level:
                stop_level = bar_low
            elif position < 0 and bar_high < stop_level:
                stop_level = bar_high

        if is_last_bar or mod >= time_stop_minute:
            continue

        if position == 0 and pending_signal == SIGNAL_NONE and not halted and daily_trade_count < max_trades:
            signal = strategy_fn(
                bar_idx, opens, highs, lows, closes, volumes,
                bar_atr, trailing_atr, daily_atr_ratio, rvol, close_sma_50,
                daily_regime_bias,
                donchian_high_5, donchian_low_5,
                minute_of_day, day_of_week, equity, intraday_pnl, position,
                entry_price, halted, daily_trade_count, params
            )

            abs_signal = abs(signal)
            if abs_signal == 1 or abs_signal == 2:
                profile_idx = abs_signal - 1
                risk_per_trade_usd = strategy_profiles[profile_idx, PROFILE_RISK_PER_TRADE_USD]
                stop_atr_multiplier = strategy_profiles[profile_idx, PROFILE_STOP_ATR_MULTIPLIER]
                target_atr_multiplier = strategy_profiles[profile_idx, PROFILE_TARGET_ATR_MULTIPLIER]
                risk_buffer_fraction = strategy_profiles[profile_idx, PROFILE_RISK_BUFFER_FRACTION]
                drawdown_buffer = equity - liquidation_floor_equity
                if drawdown_buffer <= 0.0:
                    continue
                if risk_buffer_fraction <= 0.0:
                    risk_buffer_fraction = 0.15
                risk_cap = drawdown_buffer * risk_buffer_fraction
                if risk_cap < risk_per_trade_usd:
                    risk_per_trade_usd = risk_cap
                if risk_per_trade_usd <= 0.0:
                    continue
                dynamic_stop_ticks = (current_atr * stop_atr_multiplier) / tick_size
                dynamic_target_ticks = (current_atr * target_atr_multiplier) / tick_size
                risk_per_contract = dynamic_stop_ticks * tick_value
                if dynamic_stop_ticks <= 0.0 or dynamic_target_ticks <= 0.0 or risk_per_contract <= 0.0:
                    continue
                contracts = int(np.floor(risk_per_trade_usd / risk_per_contract))
                if contracts < 1:
                    continue
                max_contracts = int(params[PARAMS_CONTRACTS])
                if max_contracts > 0 and contracts > max_contracts:
                    contracts = max_contracts
                current_breakeven_trigger_points = (
                    strategy_profiles[profile_idx, PROFILE_BREAKEVEN_TRIGGER_TICKS] * tick_size
                )
                if entry_on_close:
                    if trade_idx >= len(trade_log):
                        raise RuntimeError("trade_log capacity exceeded")
                    entry_commission = commission_per_side * contracts
                    if signal > 0:
                        fill_price = bar_close
                        stop_level = fill_price - dynamic_stop_ticks * tick_size
                        target_level = fill_price + dynamic_target_ticks * tick_size
                        position = contracts
                    else:
                        fill_price = bar_close
                        stop_level = fill_price + dynamic_stop_ticks * tick_size
                        target_level = fill_price - dynamic_target_ticks * tick_size
                        position = -contracts

                    entry_price = fill_price
                    intraday_pnl -= entry_commission
                    equity -= entry_commission
                    daily_trade_count += 1
                    breakeven_active = False

                    open_trade_idx = trade_idx
                    trade_log[open_trade_idx]["day_id"] = current_day_id
                    trade_log[open_trade_idx]["phase_id"] = current_phase_id
                    trade_log[open_trade_idx]["payout_cycle_id"] = current_payout_cycle_id
                    trade_log[open_trade_idx]["entry_time"] = timestamps[bar_idx]
                    trade_log[open_trade_idx]["entry_price"] = fill_price
                    trade_log[open_trade_idx]["entry_slippage"] = 0.0
                    trade_log[open_trade_idx]["entry_commission"] = entry_commission
                    trade_log[open_trade_idx]["contracts"] = contracts
                    trade_log[open_trade_idx]["signal_type"] = signal
                    trade_idx += 1
                else:
                    pending_signal = signal
                    pending_contracts = contracts
                    pending_stop_ticks = dynamic_stop_ticks
                    pending_target_ticks = dynamic_target_ticks
                    pending_breakeven_trigger_points = current_breakeven_trigger_points

    n_trades = trade_idx - trade_log_offset
    return n_trades, equity, intraday_pnl
