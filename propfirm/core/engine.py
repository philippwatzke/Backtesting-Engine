import numpy as np
from numba import njit
from propfirm.core.types import (
    TRADE_LOG_DTYPE, SIGNAL_LONG, SIGNAL_SHORT, SIGNAL_NONE,
    EXIT_TARGET, EXIT_STOP, EXIT_HARD_CLOSE, EXIT_CIRCUIT_BREAKER,
    HARD_CLOSE_MINUTE, MNQ_TICK_SIZE, MNQ_TICK_VALUE,
    PARAMS_RANGE_MINUTES, PARAMS_STOP_TICKS, PARAMS_TARGET_TICKS,
    PARAMS_CONTRACTS, PARAMS_DAILY_STOP, PARAMS_DAILY_TARGET,
    PARAMS_MAX_TRADES, PARAMS_STOP_PENALTY, PARAMS_COMMISSION,
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

    params is a flat float64 array with 11 elements.
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

    for bar_idx in range(n_bars):
        mod = minute_of_day[bar_idx]
        bar_open = opens[bar_idx]
        bar_high = highs[bar_idx]
        bar_low = lows[bar_idx]
        bar_close = closes[bar_idx]
        current_atr = bar_atr[bar_idx]
        current_trailing = trailing_atr[bar_idx]
        exited_this_bar = False

        # --- Check exits for open position ---
        if position != 0:
            exit_price = 0.0
            exit_reason = -1

            if position > 0:  # Long
                if bar_low <= stop_level:
                    slip = _compute_slippage(mod, current_atr, current_trailing,
                                             slippage_lookup, True, stop_penalty)
                    exit_price = stop_level - slip
                    exit_reason = EXIT_STOP
                elif bar_high >= target_level:
                    slip = _compute_slippage(mod, current_atr, current_trailing,
                                             slippage_lookup, False, stop_penalty)
                    exit_price = target_level - slip
                    exit_reason = EXIT_TARGET
            else:  # Short
                if bar_high >= stop_level:
                    slip = _compute_slippage(mod, current_atr, current_trailing,
                                             slippage_lookup, True, stop_penalty)
                    exit_price = stop_level + slip
                    exit_reason = EXIT_STOP
                elif bar_low <= target_level:
                    slip = _compute_slippage(mod, current_atr, current_trailing,
                                             slippage_lookup, False, stop_penalty)
                    exit_price = target_level + slip
                    exit_reason = EXIT_TARGET

            # Hard close at 15:59
            if exit_reason == -1 and mod >= HARD_CLOSE_MINUTE:
                slip = _compute_slippage(mod, current_atr, current_trailing,
                                         slippage_lookup, False, stop_penalty)
                if position > 0:
                    exit_price = bar_close - slip
                else:
                    exit_price = bar_close + slip
                exit_reason = EXIT_HARD_CLOSE

            if exit_reason >= 0:
                abs_contracts = abs(position)
                if position > 0:
                    gross_pnl = (exit_price - entry_price) * abs_contracts / MNQ_TICK_SIZE * MNQ_TICK_VALUE
                else:
                    gross_pnl = (entry_price - exit_price) * abs_contracts / MNQ_TICK_SIZE * MNQ_TICK_VALUE
                exit_commission = commission_per_side * abs_contracts
                entry_comm_logged = commission_per_side * abs_contracts
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
                exited_this_bar = True

        # --- Circuit breaker check ---
        if not halted and check_circuit_breaker(intraday_pnl, daily_stop):
            halted = True

        if exited_this_bar:
            continue

        # --- No new entries at or after 15:59 ---
        if mod >= HARD_CLOSE_MINUTE:
            continue

        # --- Signal generation ---
        if position == 0 and not halted and daily_trade_count < max_trades:
            signal = strategy_fn(
                bar_idx, opens, highs, lows, closes, volumes,
                minute_of_day, equity, intraday_pnl, position,
                entry_price, halted, daily_trade_count, params
            )

            if signal == SIGNAL_LONG or signal == SIGNAL_SHORT:
                if trade_idx >= len(trade_log):
                    raise RuntimeError("trade_log capacity exceeded")
                slip = _compute_slippage(mod, current_atr, current_trailing,
                                         slippage_lookup, False, stop_penalty)
                entry_commission = commission_per_side * contracts

                if signal == SIGNAL_LONG:
                    fill_price = bar_close + slip
                    stop_level = fill_price - stop_ticks * MNQ_TICK_SIZE
                    target_level = fill_price + target_ticks * MNQ_TICK_SIZE
                    position = contracts
                else:
                    fill_price = bar_close - slip
                    stop_level = fill_price + stop_ticks * MNQ_TICK_SIZE
                    target_level = fill_price - target_ticks * MNQ_TICK_SIZE
                    position = -contracts

                entry_price = fill_price
                intraday_pnl -= entry_commission
                equity -= entry_commission
                daily_trade_count += 1

                open_trade_idx = trade_idx
                trade_log[open_trade_idx]["day_id"] = current_day_id
                trade_log[open_trade_idx]["phase_id"] = current_phase_id
                trade_log[open_trade_idx]["payout_cycle_id"] = current_payout_cycle_id
                trade_log[open_trade_idx]["entry_time"] = timestamps[bar_idx]
                trade_log[open_trade_idx]["entry_price"] = fill_price
                trade_log[open_trade_idx]["entry_slippage"] = slip
                trade_log[open_trade_idx]["entry_commission"] = entry_commission
                trade_log[open_trade_idx]["contracts"] = contracts
                trade_log[open_trade_idx]["signal_type"] = signal
                trade_idx += 1

    n_trades = trade_idx - trade_log_offset
    return n_trades, equity, intraday_pnl
