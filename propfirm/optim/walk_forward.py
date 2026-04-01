import numpy as np
from itertools import product
from propfirm.core.engine import run_day_kernel
from propfirm.core.types import (
    TRADE_LOG_DTYPE, DAILY_LOG_DTYPE, PARAMS_ARRAY_LENGTH,
    PARAMS_CONTRACTS, PARAMS_MAX_TRADES,
    PARAMS_RANGE_MINUTES, PARAMS_STOP_TICKS, PARAMS_TARGET_TICKS,
    PARAMS_DAILY_STOP, PARAMS_DAILY_TARGET, PARAMS_BUFFER_TICKS,
    PARAMS_VOL_THRESHOLD, PARAMS_STOP_PENALTY, PARAMS_COMMISSION,
)
from propfirm.risk.risk import validate_position_size
from propfirm.rules.mff import MFFState
from propfirm.strategy.orb import orb_signal
from propfirm.monte_carlo.bootstrap import (
    run_monte_carlo,
    split_daily_log_for_mc,
)


PARAM_INDEX_TO_NAME = {
    PARAMS_RANGE_MINUTES: "range_minutes",
    PARAMS_STOP_TICKS: "stop_ticks",
    PARAMS_TARGET_TICKS: "target_ticks",
    PARAMS_CONTRACTS: "contracts",
    PARAMS_DAILY_STOP: "daily_stop",
    PARAMS_DAILY_TARGET: "daily_target",
    PARAMS_MAX_TRADES: "max_trades_day",
    PARAMS_BUFFER_TICKS: "buffer_ticks",
    PARAMS_VOL_THRESHOLD: "volume_threshold",
    PARAMS_STOP_PENALTY: "stop_penalty",
    PARAMS_COMMISSION: "commission_per_side",
}


def _backtest_param_set(
    session_data: dict,
    day_range: tuple[int, int],
    params_eval: np.ndarray,
    params_funded: np.ndarray,
    slippage_lookup: np.ndarray,
    mff_config: dict,
) -> dict:
    state = MFFState(mff_config)
    funded_payout_cycle_id = 0
    start_day, end_day = day_range
    max_trades_per_day = int(max(params_eval[PARAMS_MAX_TRADES], params_funded[PARAMS_MAX_TRADES]))
    max_possible_trades = max(1, (end_day - start_day) * max_trades_per_day)
    all_trades = np.zeros(max_possible_trades, dtype=TRADE_LOG_DTYPE)
    daily_log = np.zeros(end_day - start_day, dtype=DAILY_LOG_DTYPE)
    total_trade_count = 0
    total_day_count = 0

    for day_idx in range(start_day, end_day):
        start, end = session_data["day_boundaries"][day_idx]
        active_params = params_eval if state.phase == "eval" else params_funded.copy()
        phase_id = 0 if state.phase == "eval" else 1
        payout_cycle_id = -1 if state.phase == "eval" else funded_payout_cycle_id
        if state.phase == "funded":
            active_params[PARAMS_CONTRACTS] = float(
                validate_position_size(
                    int(active_params[PARAMS_CONTRACTS]),
                    state.get_max_contracts(),
                )
            )
        n_trades, equity, pnl = run_day_kernel(
            session_data["open"][start:end],
            session_data["high"][start:end],
            session_data["low"][start:end],
            session_data["close"][start:end],
            session_data["volume"][start:end],
            session_data["timestamps"][start:end],
            session_data["minute_of_day"][start:end],
            session_data["bar_atr"][start:end],
            session_data["trailing_median_atr"][start:end],
            slippage_lookup,
            day_idx,
            phase_id,
            payout_cycle_id,
            orb_signal,
            all_trades[total_trade_count:],
            0, state.equity, 0.0, active_params,
        )
        total_trade_count += n_trades
        result = state.update_eod(pnl, state.equity + pnl)
        net_payout = 0.0
        if result == "blown":
            pass
        elif state.phase == "funded" and state.payout_eligible:
            net_payout = state.process_payout()
            if net_payout > 0:
                funded_payout_cycle_id += 1
        daily_log[total_day_count]["day_id"] = day_idx
        daily_log[total_day_count]["phase_id"] = phase_id
        daily_log[total_day_count]["payout_cycle_id"] = payout_cycle_id
        daily_log[total_day_count]["had_trade"] = 1 if n_trades > 0 else 0
        daily_log[total_day_count]["n_trades"] = n_trades
        daily_log[total_day_count]["day_pnl"] = pnl
        daily_log[total_day_count]["net_payout"] = net_payout
        total_day_count += 1
        if result == "passed" and state.phase == "eval":
            state.transition_to_funded()
            funded_payout_cycle_id = 0
        if result == "blown":
            break

    return {
        "trade_log": all_trades[:total_trade_count],
        "daily_log": daily_log[:total_day_count],
    }


def _build_params_array(base_params: np.ndarray, overrides: dict, phase: str) -> np.ndarray:
    p = base_params.copy()
    for idx, val in overrides.items():
        override_phase, override_idx = idx
        if override_phase == phase:
            p[override_idx] = float(val)
    return p


def _serialize_param_overrides(overrides: dict | None) -> dict | None:
    if overrides is None:
        return None
    result = {"eval": {}, "funded": {}}
    for (phase, param_idx), value in overrides.items():
        result[phase][PARAM_INDEX_TO_NAME.get(param_idx, f"param_{param_idx}")] = float(value)
    if not result["eval"]:
        result.pop("eval")
    if not result["funded"]:
        result.pop("funded")
    return result


def run_walk_forward(
    session_data: dict,
    slippage_lookup: np.ndarray,
    base_params_eval: np.ndarray,
    base_params_funded: np.ndarray,
    param_grid: dict[tuple[str, int], list[float]],
    mff_config: dict,
    window_train_days: int = 120,
    window_test_days: int = 60,
    step_days: int = 60,
    n_mc_sims: int = 500,
    mc_block_min: int = 5,
    mc_block_max: int = 10,
    seed: int = 42,
    n_workers: int = 1,
) -> list[dict]:
    n_total_days = len(session_data["day_boundaries"])
    results = []
    window_idx = 0

    grid_keys = sorted(param_grid.keys())
    grid_values = [param_grid[k] for k in grid_keys]
    all_combos = list(product(*grid_values))

    train_end = window_train_days
    while train_end + window_test_days <= n_total_days:
        test_end = train_end + window_test_days

        best_nve = -np.inf
        best_combo = None
        best_is_result = None

        for combo in all_combos:
            overrides = dict(zip(grid_keys, combo))
            params_eval = _build_params_array(base_params_eval, overrides, "eval")
            params_funded = _build_params_array(base_params_funded, overrides, "funded")

            artifacts = _backtest_param_set(
                session_data, (0, train_end), params_eval, params_funded, slippage_lookup, mff_config)

            if len(artifacts["daily_log"]) < 10:
                continue

            try:
                phase_pools = split_daily_log_for_mc(artifacts["daily_log"])
            except ValueError:
                continue
            mc_result = run_monte_carlo(
                phase_pools["eval_day_pnls"], mff_config,
                funded_pnls=phase_pools["funded_day_pnls"],
                n_sims=n_mc_sims, seed=seed + window_idx,
                n_workers=n_workers,
                block_mode="daily",
                block_min=mc_block_min,
                block_max=mc_block_max,
            )

            if mc_result.nve > best_nve:
                best_nve = mc_result.nve
                best_combo = overrides
                best_is_result = mc_result

        if best_combo is not None:
            best_params_eval = _build_params_array(base_params_eval, best_combo, "eval")
            best_params_funded = _build_params_array(base_params_funded, best_combo, "funded")
            oos_artifacts = _backtest_param_set(
                session_data, (train_end, test_end),
                best_params_eval, best_params_funded, slippage_lookup, mff_config)

            if len(oos_artifacts["daily_log"]) >= 5:
                try:
                    oos_phase_pools = split_daily_log_for_mc(oos_artifacts["daily_log"])
                    oos_mc = run_monte_carlo(
                        oos_phase_pools["eval_day_pnls"], mff_config,
                        funded_pnls=oos_phase_pools["funded_day_pnls"],
                        n_sims=n_mc_sims, seed=seed + window_idx + 50000,
                        n_workers=n_workers,
                        block_mode="daily",
                        block_min=mc_block_min,
                        block_max=mc_block_max,
                    )
                    oos_nve = oos_mc.nve
                    oos_payout_rate = oos_mc.payout_rate
                    oos_status = "ok"
                except ValueError:
                    oos_nve = None
                    oos_payout_rate = None
                    oos_status = "not_scored"
            else:
                oos_nve = None
                oos_payout_rate = None
                oos_status = "not_scored"
        else:
            oos_nve = None
            oos_payout_rate = None
            oos_status = "not_scored"

        results.append({
            "window": window_idx,
            "train_date_range": (
                session_data["session_dates"][0],
                session_data["session_dates"][train_end - 1],
            ),
            "test_date_range": (
                session_data["session_dates"][train_end],
                session_data["session_dates"][test_end - 1],
            ),
            "best_params": _serialize_param_overrides(best_combo),
            "in_sample_nve": best_nve if best_combo else None,
            "in_sample_payout_rate": (best_is_result.payout_rate
                                      if best_is_result else None),
            "oos_nve": oos_nve,
            "oos_payout_rate": oos_payout_rate,
            "status": "ok" if best_combo else "not_scored",
            "oos_status": oos_status if best_combo else "not_scored",
        })

        train_end += step_days
        window_idx += 1

    return results
