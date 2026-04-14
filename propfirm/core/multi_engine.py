import numpy as np

from propfirm.core.types import (
    EXIT_CIRCUIT_BREAKER,
    EXIT_HARD_CLOSE,
    EXIT_STOP,
    EXIT_TARGET,
    HARD_CLOSE_MINUTE,
    MNQ_TICK_SIZE,
    MNQ_TICK_VALUE,
    PARAMS_COMMISSION,
    PARAMS_CONTRACTS,
    PARAMS_EXTRA_SLIPPAGE_TICKS,
    PARAMS_MAX_TRADES,
    PARAMS_STOP_PENALTY,
    PARAMS_TICK_SIZE,
    PARAMS_TICK_VALUE,
    PARAMS_TIME_STOP_MINUTE,
    PARAMS_TRAIL_BAR_EXTREME,
    PROFILE_BREAKEVEN_TRIGGER_TICKS,
    PROFILE_RISK_BUFFER_FRACTION,
    PROFILE_RISK_PER_TRADE_USD,
    PROFILE_STOP_ATR_MULTIPLIER,
    PROFILE_TARGET_ATR_MULTIPLIER,
    SIGNAL_NONE,
)
from propfirm.market.slippage import compute_slippage


def _optional_array(config: dict, key: str, n: int, fill: float = 0.0, dtype=np.float64) -> np.ndarray:
    values = config.get(key)
    if values is None:
        return np.full(n, fill, dtype=dtype)
    array = np.asarray(values, dtype=dtype)
    if len(array) != n:
        raise ValueError(f"{config['name']}: {key} length {len(array)} != expected {n}")
    return array


def _clear_pending(state: dict) -> None:
    state["pending_signal"] = SIGNAL_NONE
    state["pending_contracts"] = 0
    state["pending_stop_ticks"] = 0.0
    state["pending_target_ticks"] = 0.0
    state["pending_breakeven_trigger_points"] = 0.0


def _mark_to_market(position: int, entry_price: float, mark_price: float, tick_size: float, tick_value: float) -> float:
    if position == 0:
        return 0.0
    abs_contracts = abs(position)
    if position > 0:
        return (mark_price - entry_price) * abs_contracts / tick_size * tick_value
    return (entry_price - mark_price) * abs_contracts / tick_size * tick_value


def _build_asset_state(config: dict) -> dict:
    opens = np.asarray(config["opens"], dtype=np.float64)
    n_bars = len(opens)
    params = np.asarray(config["params"], dtype=np.float64)

    tick_size = params[PARAMS_TICK_SIZE]
    if tick_size <= 0.0:
        tick_size = MNQ_TICK_SIZE
    tick_value = params[PARAMS_TICK_VALUE]
    if tick_value <= 0.0:
        tick_value = MNQ_TICK_VALUE
    time_stop_minute = int(params[PARAMS_TIME_STOP_MINUTE])
    if time_stop_minute <= 0:
        time_stop_minute = HARD_CLOSE_MINUTE

    state = {
        "name": config["name"],
        "opens": opens,
        "highs": np.asarray(config.get("highs", np.empty(n_bars)), dtype=np.float64),
        "lows": np.asarray(config.get("lows", np.empty(n_bars)), dtype=np.float64),
        "closes": np.asarray(config.get("closes", np.empty(n_bars)), dtype=np.float64),
        "volumes": np.asarray(config.get("volumes", np.empty(n_bars, dtype=np.uint64))),
        "timestamps": np.asarray(config.get("timestamps", np.empty(n_bars, dtype=np.int64)), dtype=np.int64),
        "minute_of_day": np.asarray(config.get("minute_of_day", np.empty(n_bars, dtype=np.int16)), dtype=np.int16),
        "bar_atr": np.asarray(config.get("bar_atr", np.empty(n_bars)), dtype=np.float64),
        "trailing_atr": np.asarray(config.get("trailing_atr", np.empty(n_bars)), dtype=np.float64),
        "daily_atr_ratio": _optional_array(config, "daily_atr_ratio", n_bars, fill=1.0),
        "rvol": _optional_array(config, "rvol", n_bars, fill=1.0),
        "close_sma_50": _optional_array(config, "close_sma_50", n_bars, fill=np.nan),
        "daily_regime_bias": _optional_array(config, "daily_regime_bias", n_bars, fill=0.0),
        "donchian_high_5": _optional_array(config, "donchian_high_5", n_bars, fill=np.nan),
        "donchian_low_5": _optional_array(config, "donchian_low_5", n_bars, fill=np.nan),
        "day_of_week": _optional_array(config, "day_of_week", n_bars, fill=0, dtype=np.int8),
        "slippage_lookup": np.asarray(config["slippage_lookup"], dtype=np.float64),
        "strategy_fn": config["strategy_fn"],
        "strategy_profiles": np.asarray(config["strategy_profiles"], dtype=np.float64),
        "trade_log": config["trade_log"],
        "trade_idx": int(config.get("trade_log_offset", 0)),
        "trade_log_offset": int(config.get("trade_log_offset", 0)),
        "current_day_id": int(config.get("current_day_id", 0)),
        "current_phase_id": int(config.get("current_phase_id", 0)),
        "current_payout_cycle_id": int(config.get("current_payout_cycle_id", -1)),
        "liquidation_floor_equity": float(config.get("liquidation_floor_equity", 0.0)),
        "params": params,
        "tick_size": float(tick_size),
        "tick_value": float(tick_value),
        "commission_per_side": float(params[PARAMS_COMMISSION]),
        "stop_penalty": float(params[PARAMS_STOP_PENALTY]),
        "extra_slippage_points": float(params[PARAMS_EXTRA_SLIPPAGE_TICKS] * tick_size),
        "max_trades": int(params[PARAMS_MAX_TRADES]),
        "time_stop_minute": int(time_stop_minute),
        "trail_bar_extreme": bool(params[PARAMS_TRAIL_BAR_EXTREME] > 0.5),
        "n_bars": n_bars,
        "bar_idx": 0,
        "equity": float(config.get("starting_equity", 0.0)),
        "realized_pnl": float(config.get("starting_pnl", 0.0)),
        "unrealized_pnl": 0.0,
        "entry_price": 0.0,
        "position": 0,
        "daily_trade_count": 0,
        "open_trade_idx": -1,
        "stop_level": 0.0,
        "target_level": 0.0,
        "breakeven_active": False,
        "current_breakeven_trigger_points": 0.0,
        "pending_signal": SIGNAL_NONE,
        "pending_contracts": 0,
        "pending_stop_ticks": 0.0,
        "pending_target_ticks": 0.0,
        "pending_breakeven_trigger_points": 0.0,
        "forced_exit_pending": False,
        "forced_exit_after_timestamp": -1,
        "forced_exit_count": 0,
        "completed": n_bars == 0,
    }
    return state


def _record_entry(state: dict, bar_idx: int) -> None:
    if state["trade_idx"] >= len(state["trade_log"]):
        raise RuntimeError(f"{state['name']}: trade_log capacity exceeded")

    mod = int(state["minute_of_day"][bar_idx])
    slip = compute_slippage(
        mod,
        float(state["bar_atr"][bar_idx]),
        float(state["trailing_atr"][bar_idx]),
        state["slippage_lookup"],
        False,
        state["stop_penalty"],
        state["tick_size"],
        state["extra_slippage_points"],
    )
    contracts = int(state["pending_contracts"])
    entry_commission = state["commission_per_side"] * contracts
    if state["pending_signal"] > 0:
        fill_price = float(state["opens"][bar_idx]) + slip
        state["position"] = contracts
        state["stop_level"] = fill_price - state["pending_stop_ticks"] * state["tick_size"]
        state["target_level"] = fill_price + state["pending_target_ticks"] * state["tick_size"]
    else:
        fill_price = float(state["opens"][bar_idx]) - slip
        state["position"] = -contracts
        state["stop_level"] = fill_price + state["pending_stop_ticks"] * state["tick_size"]
        state["target_level"] = fill_price - state["pending_target_ticks"] * state["tick_size"]

    state["entry_price"] = fill_price
    state["realized_pnl"] -= entry_commission
    state["equity"] -= entry_commission
    state["daily_trade_count"] += 1
    state["breakeven_active"] = False
    state["current_breakeven_trigger_points"] = state["pending_breakeven_trigger_points"]

    open_trade_idx = state["trade_idx"]
    state["open_trade_idx"] = open_trade_idx
    trade_log = state["trade_log"]
    trade_log[open_trade_idx]["day_id"] = state["current_day_id"]
    trade_log[open_trade_idx]["phase_id"] = state["current_phase_id"]
    trade_log[open_trade_idx]["payout_cycle_id"] = state["current_payout_cycle_id"]
    trade_log[open_trade_idx]["entry_time"] = state["timestamps"][bar_idx]
    trade_log[open_trade_idx]["entry_price"] = fill_price
    trade_log[open_trade_idx]["entry_slippage"] = slip
    trade_log[open_trade_idx]["entry_commission"] = entry_commission
    trade_log[open_trade_idx]["contracts"] = contracts
    trade_log[open_trade_idx]["signal_type"] = state["pending_signal"]
    state["trade_idx"] += 1
    _clear_pending(state)


def _record_exit(state: dict, bar_idx: int, exit_price: float, exit_reason: int, slip: float) -> None:
    if state["open_trade_idx"] < 0:
        raise RuntimeError(f"{state['name']}: open_trade_idx missing on exit")

    abs_contracts = abs(state["position"])
    exit_commission = state["commission_per_side"] * abs_contracts
    entry_commission = float(state["trade_log"][state["open_trade_idx"]]["entry_commission"])

    if state["position"] > 0:
        gross_pnl = (exit_price - state["entry_price"]) * abs_contracts / state["tick_size"] * state["tick_value"]
    else:
        gross_pnl = (state["entry_price"] - exit_price) * abs_contracts / state["tick_size"] * state["tick_value"]
    net_pnl = gross_pnl - entry_commission - exit_commission

    trade_log = state["trade_log"]
    trade_log[state["open_trade_idx"]]["day_id"] = state["current_day_id"]
    trade_log[state["open_trade_idx"]]["phase_id"] = state["current_phase_id"]
    trade_log[state["open_trade_idx"]]["payout_cycle_id"] = state["current_payout_cycle_id"]
    trade_log[state["open_trade_idx"]]["exit_time"] = state["timestamps"][bar_idx]
    trade_log[state["open_trade_idx"]]["exit_price"] = exit_price
    trade_log[state["open_trade_idx"]]["exit_slippage"] = slip
    trade_log[state["open_trade_idx"]]["exit_commission"] = exit_commission
    trade_log[state["open_trade_idx"]]["gross_pnl"] = gross_pnl
    trade_log[state["open_trade_idx"]]["net_pnl"] = net_pnl
    trade_log[state["open_trade_idx"]]["exit_reason"] = exit_reason

    state["realized_pnl"] += gross_pnl - exit_commission
    state["equity"] += gross_pnl - exit_commission
    state["unrealized_pnl"] = 0.0
    state["position"] = 0
    state["entry_price"] = 0.0
    state["open_trade_idx"] = -1
    state["stop_level"] = 0.0
    state["target_level"] = 0.0
    state["breakeven_active"] = False
    state["current_breakeven_trigger_points"] = 0.0
    state["forced_exit_pending"] = False
    state["forced_exit_after_timestamp"] = -1
    if exit_reason == EXIT_CIRCUIT_BREAKER:
        state["forced_exit_count"] += 1


def _process_signal(state: dict, bar_idx: int, global_halt: bool) -> None:
    if state["position"] != 0 or state["pending_signal"] != SIGNAL_NONE or global_halt:
        return
    if state["daily_trade_count"] >= state["max_trades"]:
        return

    mod = int(state["minute_of_day"][bar_idx])
    if mod >= state["time_stop_minute"]:
        return

    signal = state["strategy_fn"](
        bar_idx,
        state["opens"],
        state["highs"],
        state["lows"],
        state["closes"],
        state["volumes"],
        state["bar_atr"],
        state["trailing_atr"],
        state["daily_atr_ratio"],
        state["rvol"],
        state["close_sma_50"],
        state["daily_regime_bias"],
        state["donchian_high_5"],
        state["donchian_low_5"],
        state["minute_of_day"],
        state["day_of_week"],
        state["equity"],
        state["realized_pnl"] + state["unrealized_pnl"],
        state["position"],
        state["entry_price"],
        global_halt,
        state["daily_trade_count"],
        state["params"],
    )

    abs_signal = abs(int(signal))
    if abs_signal < 1 or abs_signal > len(state["strategy_profiles"]):
        return

    profile_idx = abs_signal - 1
    current_atr = float(state["bar_atr"][bar_idx])
    stop_atr_multiplier = state["strategy_profiles"][profile_idx, PROFILE_STOP_ATR_MULTIPLIER]
    target_atr_multiplier = state["strategy_profiles"][profile_idx, PROFILE_TARGET_ATR_MULTIPLIER]
    risk_per_trade_usd = state["strategy_profiles"][profile_idx, PROFILE_RISK_PER_TRADE_USD]
    risk_buffer_fraction = state["strategy_profiles"][profile_idx, PROFILE_RISK_BUFFER_FRACTION]
    if risk_buffer_fraction <= 0.0:
        risk_buffer_fraction = 0.15

    dynamic_stop_ticks = (current_atr * stop_atr_multiplier) / state["tick_size"]
    dynamic_target_ticks = (current_atr * target_atr_multiplier) / state["tick_size"]
    risk_per_contract = dynamic_stop_ticks * state["tick_value"]
    if dynamic_stop_ticks <= 0.0 or dynamic_target_ticks <= 0.0 or risk_per_contract <= 0.0:
        return

    drawdown_buffer = state["equity"] - state["liquidation_floor_equity"]
    if drawdown_buffer <= 0.0:
        return
    risk_cap = drawdown_buffer * risk_buffer_fraction
    if risk_cap < risk_per_trade_usd:
        risk_per_trade_usd = risk_cap
    if risk_per_trade_usd <= 0.0:
        return

    contracts = int(np.floor(risk_per_trade_usd / risk_per_contract))
    max_contracts = int(state["params"][PARAMS_CONTRACTS])
    if max_contracts > 0 and contracts > max_contracts:
        contracts = max_contracts
    if contracts < 1:
        return

    state["pending_signal"] = int(signal)
    state["pending_contracts"] = contracts
    state["pending_stop_ticks"] = dynamic_stop_ticks
    state["pending_target_ticks"] = dynamic_target_ticks
    state["pending_breakeven_trigger_points"] = (
        state["strategy_profiles"][profile_idx, PROFILE_BREAKEVEN_TRIGGER_TICKS] * state["tick_size"]
    )


def _process_asset_event(state: dict, bar_idx: int, global_halt: bool) -> None:
    mod = int(state["minute_of_day"][bar_idx])
    ts = int(state["timestamps"][bar_idx])
    is_last_bar = bar_idx == state["n_bars"] - 1
    current_atr = float(state["bar_atr"][bar_idx])
    current_trailing = float(state["trailing_atr"][bar_idx])
    bar_open = float(state["opens"][bar_idx])
    bar_high = float(state["highs"][bar_idx])
    bar_low = float(state["lows"][bar_idx])
    bar_close = float(state["closes"][bar_idx])
    exited_this_bar = False

    if state["forced_exit_pending"] and state["position"] != 0 and ts > state["forced_exit_after_timestamp"]:
        slip = compute_slippage(
            mod,
            current_atr,
            current_trailing,
            state["slippage_lookup"],
            False,
            state["stop_penalty"],
            state["tick_size"],
            state["extra_slippage_points"],
        )
        exit_price = bar_open - slip if state["position"] > 0 else bar_open + slip
        _record_exit(state, bar_idx, exit_price, EXIT_CIRCUIT_BREAKER, slip)
        exited_this_bar = True
    elif state["position"] == 0 and state["pending_signal"] != SIGNAL_NONE:
        if global_halt or mod >= state["time_stop_minute"]:
            _clear_pending(state)
        else:
            _record_entry(state, bar_idx)

    if state["position"] != 0 and not state["forced_exit_pending"]:
        exit_price = 0.0
        exit_reason = -1
        slip = 0.0

        if mod >= state["time_stop_minute"]:
            slip = compute_slippage(
                mod,
                current_atr,
                current_trailing,
                state["slippage_lookup"],
                False,
                state["stop_penalty"],
                state["tick_size"],
                state["extra_slippage_points"],
            )
            exit_price = bar_open - slip if state["position"] > 0 else bar_open + slip
            exit_reason = EXIT_HARD_CLOSE
        elif state["position"] > 0:
            if bar_open <= state["stop_level"]:
                slip = compute_slippage(
                    mod,
                    current_atr,
                    current_trailing,
                    state["slippage_lookup"],
                    True,
                    state["stop_penalty"],
                    state["tick_size"],
                    state["extra_slippage_points"],
                )
                exit_price = bar_open - slip
                exit_reason = EXIT_STOP
            elif bar_low <= state["stop_level"]:
                slip = compute_slippage(
                    mod,
                    current_atr,
                    current_trailing,
                    state["slippage_lookup"],
                    True,
                    state["stop_penalty"],
                    state["tick_size"],
                    state["extra_slippage_points"],
                )
                exit_price = state["stop_level"] - slip
                exit_reason = EXIT_STOP
            elif bar_high >= state["target_level"]:
                slip = compute_slippage(
                    mod,
                    current_atr,
                    current_trailing,
                    state["slippage_lookup"],
                    False,
                    state["stop_penalty"],
                    state["tick_size"],
                    state["extra_slippage_points"],
                )
                exit_price = state["target_level"] - slip
                exit_reason = EXIT_TARGET
        else:
            if bar_open >= state["stop_level"]:
                slip = compute_slippage(
                    mod,
                    current_atr,
                    current_trailing,
                    state["slippage_lookup"],
                    True,
                    state["stop_penalty"],
                    state["tick_size"],
                    state["extra_slippage_points"],
                )
                exit_price = bar_open + slip
                exit_reason = EXIT_STOP
            elif bar_high >= state["stop_level"]:
                slip = compute_slippage(
                    mod,
                    current_atr,
                    current_trailing,
                    state["slippage_lookup"],
                    True,
                    state["stop_penalty"],
                    state["tick_size"],
                    state["extra_slippage_points"],
                )
                exit_price = state["stop_level"] + slip
                exit_reason = EXIT_STOP
            elif bar_low <= state["target_level"]:
                slip = compute_slippage(
                    mod,
                    current_atr,
                    current_trailing,
                    state["slippage_lookup"],
                    False,
                    state["stop_penalty"],
                    state["tick_size"],
                    state["extra_slippage_points"],
                )
                exit_price = state["target_level"] + slip
                exit_reason = EXIT_TARGET

        if exit_reason == -1 and is_last_bar:
            slip = compute_slippage(
                mod,
                current_atr,
                current_trailing,
                state["slippage_lookup"],
                False,
                state["stop_penalty"],
                state["tick_size"],
                state["extra_slippage_points"],
            )
            exit_price = bar_close - slip if state["position"] > 0 else bar_close + slip
            exit_reason = EXIT_HARD_CLOSE

        if exit_reason >= 0:
            _record_exit(state, bar_idx, exit_price, exit_reason, slip)
            exited_this_bar = True

    if state["position"] != 0 and not state["breakeven_active"] and state["current_breakeven_trigger_points"] > 0.0:
        if state["position"] > 0 and bar_high >= state["entry_price"] + state["current_breakeven_trigger_points"]:
            be_stop = state["entry_price"] + 2.0 * state["tick_size"]
            if be_stop > state["stop_level"]:
                state["stop_level"] = be_stop
            state["breakeven_active"] = True
        elif state["position"] < 0 and bar_low <= state["entry_price"] - state["current_breakeven_trigger_points"]:
            be_stop = state["entry_price"] - 2.0 * state["tick_size"]
            if be_stop < state["stop_level"]:
                state["stop_level"] = be_stop
            state["breakeven_active"] = True

    if state["position"] != 0 and state["trail_bar_extreme"]:
        if state["position"] > 0 and bar_low > state["stop_level"]:
            state["stop_level"] = bar_low
        elif state["position"] < 0 and bar_high < state["stop_level"]:
            state["stop_level"] = bar_high

    state["unrealized_pnl"] = _mark_to_market(
        state["position"], state["entry_price"], bar_close, state["tick_size"], state["tick_value"]
    )
    if not exited_this_bar:
        _process_signal(state, bar_idx, global_halt)
    state["bar_idx"] += 1
    if state["bar_idx"] >= state["n_bars"]:
        state["completed"] = True


def run_multi_asset_day_kernel(asset_configs: list[dict], circuit_breaker_threshold: float = -800.0) -> dict:
    """Run a merged multi-asset event loop for a single session date."""
    if len(asset_configs) < 2:
        raise ValueError("run_multi_asset_day_kernel requires at least two assets")

    states = [_build_asset_state(cfg) for cfg in asset_configs]
    global_halt = False
    trigger_timestamp = -1
    trigger_global_pnl = np.nan
    trigger_snapshot = None

    while True:
        active_states = [state for state in states if not state["completed"]]
        if not active_states:
            break

        next_timestamp = min(int(state["timestamps"][state["bar_idx"]]) for state in active_states)
        for state in states:
            if state["completed"]:
                continue
            if int(state["timestamps"][state["bar_idx"]]) != next_timestamp:
                continue

            _process_asset_event(state, state["bar_idx"], global_halt)
            global_daily_pnl = sum(s["realized_pnl"] + s["unrealized_pnl"] for s in states)

            if not global_halt and global_daily_pnl <= circuit_breaker_threshold:
                global_halt = True
                trigger_timestamp = next_timestamp
                trigger_global_pnl = float(global_daily_pnl)
                trigger_snapshot = {
                    s["name"]: {
                        "position": int(s["position"]),
                        "realized_pnl": float(s["realized_pnl"]),
                        "unrealized_pnl": float(s["unrealized_pnl"]),
                    }
                    for s in states
                }
                for s in states:
                    _clear_pending(s)
                    if s["position"] != 0:
                        s["forced_exit_pending"] = True
                        s["forced_exit_after_timestamp"] = next_timestamp

    for state in states:
        if state["position"] == 0:
            continue
        final_idx = state["n_bars"] - 1
        mod = int(state["minute_of_day"][final_idx])
        slip = compute_slippage(
            mod,
            float(state["bar_atr"][final_idx]),
            float(state["trailing_atr"][final_idx]),
            state["slippage_lookup"],
            False,
            state["stop_penalty"],
            state["tick_size"],
            state["extra_slippage_points"],
        )
        bar_close = float(state["closes"][final_idx])
        exit_price = bar_close - slip if state["position"] > 0 else bar_close + slip
        exit_reason = EXIT_CIRCUIT_BREAKER if state["forced_exit_pending"] else EXIT_HARD_CLOSE
        _record_exit(state, final_idx, exit_price, exit_reason, slip)

    asset_results = {}
    for state in states:
        asset_results[state["name"]] = {
            "n_trades": int(state["trade_idx"] - state["trade_log_offset"]),
            "equity": float(state["equity"]),
            "realized_pnl": float(state["realized_pnl"]),
            "unrealized_pnl": float(state["unrealized_pnl"]),
            "forced_exit_count": int(state["forced_exit_count"]),
            "global_halt_blocked_entries": bool(global_halt),
        }

    return {
        "global_halt": bool(global_halt),
        "trigger_timestamp": int(trigger_timestamp),
        "trigger_global_pnl": float(trigger_global_pnl) if global_halt else np.nan,
        "trigger_snapshot": trigger_snapshot,
        "assets": asset_results,
    }
