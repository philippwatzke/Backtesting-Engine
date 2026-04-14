import numpy as np
from itertools import product
from numba import njit
from propfirm.core.engine import run_day_kernel, run_day_kernel_portfolio
from propfirm.core.types import (
    TRADE_LOG_DTYPE, DAILY_LOG_DTYPE, PARAMS_ARRAY_LENGTH,
    PARAMS_CONTRACTS, PARAMS_MAX_TRADES,
    PARAMS_RANGE_MINUTES, PARAMS_STOP_TICKS, PARAMS_TARGET_TICKS,
    PARAMS_DAILY_STOP, PARAMS_DAILY_TARGET, PARAMS_BUFFER_TICKS,
    PARAMS_VOL_THRESHOLD, PARAMS_STOP_PENALTY, PARAMS_COMMISSION,
    PARAMS_DISTANCE_TICKS, PARAMS_SMA_PERIOD, PARAMS_BREAKEVEN_TRIGGER_TICKS,
    PARAMS_BAND_MULTIPLIER, PARAMS_POC_LOOKBACK,
    MNQ_TICK_SIZE,
    PROFILE_ARRAY_LENGTH, PROFILE_RISK_PER_TRADE_USD,
    PROFILE_STOP_ATR_MULTIPLIER, PROFILE_TARGET_ATR_MULTIPLIER,
    PROFILE_BREAKEVEN_TRIGGER_TICKS,
    SIGNAL_NONE, SIGNAL_PULLBACK_LONG, SIGNAL_PULLBACK_SHORT,
    SIGNAL_POC_BREAKOUT_LONG, SIGNAL_POC_BREAKOUT_SHORT,
)
from propfirm.risk.risk import validate_position_size
from propfirm.rules.mff import MFFState
from propfirm.strategy.vwap_poc_breakout import vwap_poc_breakout_signal
from propfirm.strategy.vwap_pullback import _vwap_pullback_signal_impl
from propfirm.strategy.vwap_poc_breakout import _vwap_poc_breakout_signal_impl
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
    PARAMS_DISTANCE_TICKS: "distance_ticks",
    PARAMS_SMA_PERIOD: "sma_period",
    PARAMS_BREAKEVEN_TRIGGER_TICKS: "breakeven_trigger_ticks",
    PARAMS_BAND_MULTIPLIER: "band_multiplier",
    PARAMS_POC_LOOKBACK: "poc_lookback",
}

MAX_MC_CANDIDATES_PER_WINDOW = 24
PULLBACK_DISTANCE_TICKS = 2.0
PULLBACK_SMA_PERIOD = 16
POC_BREAKOUT_SMA_PERIOD = 60
POC_BREAKOUT_BAND_MULTIPLIER = 1.5
POC_BREAKOUT_POC_LOOKBACK = 60
LUNCH_START_MINUTE = 120
LUNCH_END_MINUTE = 240


@njit(cache=True)
def _combined_portfolio_signal_wfo(
    bar_idx,
    opens, highs, lows, closes, volumes,
    bar_atr, trailing_atr, daily_atr_ratio, rvol, close_sma_50,
    daily_regime_bias,
    donchian_high_5, donchian_low_5,
    minute_of_day, day_of_week,
    equity, intraday_pnl, position, entry_price,
    halted, daily_trade_count,
    params,
):
    daily_target = params[PARAMS_DAILY_TARGET]
    max_trades = int(params[PARAMS_MAX_TRADES])
    ratio = daily_atr_ratio[bar_idx]
    if ratio < 0.7 or ratio > 1.5:
        return SIGNAL_NONE
    if LUNCH_START_MINUTE <= minute_of_day[bar_idx] < LUNCH_END_MINUTE:
        return SIGNAL_NONE

    pullback_signal = _vwap_pullback_signal_impl(
        bar_idx,
        opens, highs, lows, closes, volumes,
        minute_of_day,
        intraday_pnl, position,
        halted, daily_trade_count,
        daily_target, max_trades,
        PULLBACK_DISTANCE_TICKS * MNQ_TICK_SIZE,
        PULLBACK_SMA_PERIOD,
    )
    if pullback_signal > 0:
        return SIGNAL_PULLBACK_LONG
    if pullback_signal < 0:
        return SIGNAL_PULLBACK_SHORT

    breakout_signal = _vwap_poc_breakout_signal_impl(
        bar_idx,
        opens, highs, lows, closes, volumes,
        minute_of_day,
        intraday_pnl, position,
        halted, daily_trade_count,
        daily_target, max_trades,
        POC_BREAKOUT_SMA_PERIOD,
        POC_BREAKOUT_BAND_MULTIPLIER,
        POC_BREAKOUT_POC_LOOKBACK,
    )
    if breakout_signal > 0:
        return SIGNAL_POC_BREAKOUT_LONG
    if breakout_signal < 0:
        return SIGNAL_POC_BREAKOUT_SHORT

    return SIGNAL_NONE


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
        session_date = session_data["session_dates"][day_idx]

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
            vwap_poc_breakout_signal,
            all_trades[total_trade_count:],
            0, state.equity, 0.0, active_params,
        )
        total_trade_count += n_trades
        result = state.update_eod(pnl, state.equity + pnl, had_trade=n_trades > 0, session_date=session_date)
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
        if result == "blown" or state.live_transition_ready:
            break

    return {
        "trade_log": all_trades[:total_trade_count],
        "daily_log": daily_log[:total_day_count],
    }


def _compute_trade_stats(trade_log: np.ndarray) -> dict:
    if len(trade_log) == 0:
        return {
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
            "trade_count": 0,
        }

    net = trade_log["net_pnl"]
    wins = net[net > 0]
    losses = net[net < 0]
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    win_rate = float(len(wins) / len(net))
    expectancy = float(net.mean())
    return {
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "trade_count": int(len(net)),
    }


def _summarize_backtest_result(artifacts: dict) -> dict:
    trade_log = artifacts["trade_log"]
    daily_log = artifacts["daily_log"]
    trade_stats = _compute_trade_stats(trade_log)
    net_payouts = float(daily_log["net_payout"].sum()) if len(daily_log) else 0.0
    total_outcome = float(artifacts["final_equity"] + net_payouts)
    return {
        "trade_count": trade_stats["trade_count"],
        "win_rate": trade_stats["win_rate"],
        "avg_win": trade_stats["avg_win"],
        "avg_loss": trade_stats["avg_loss"],
        "expectancy": trade_stats["expectancy"],
        "final_equity": float(artifacts["final_equity"]),
        "net_payouts": net_payouts,
        "total_outcome": total_outcome,
        "payouts_completed": int(artifacts["payouts_completed"]),
        "funded_days": int(artifacts["funded_days"]),
        "eval_passed": bool(artifacts["eval_passed"]),
        "eval_pass_day": artifacts["eval_pass_day"],
        "live_transition_ready": bool(artifacts["live_transition_ready"]),
        "blown": bool(artifacts["blown"]),
        "status": artifacts["status"],
    }


def _backtest_portfolio_window(
    session_data: dict,
    day_range: tuple[int, int],
    params_eval: np.ndarray,
    params_funded: np.ndarray,
    strategy_profiles_eval: np.ndarray,
    strategy_profiles_funded: np.ndarray,
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
    eval_pass_day = -1
    blown_day = -1

    for rel_day_idx, day_idx in enumerate(range(start_day, end_day)):
        session_date = session_data["session_dates"][day_idx]
        start, end = session_data["day_boundaries"][day_idx]
        active_params = params_eval.copy() if state.phase == "eval" else params_funded.copy()
        strategy_profiles = strategy_profiles_eval.copy() if state.phase == "eval" else strategy_profiles_funded.copy()
        phase_id = 0 if state.phase == "eval" else 1
        payout_cycle_id = -1 if state.phase == "eval" else funded_payout_cycle_id
        active_params[PARAMS_CONTRACTS] = float(
            state.get_max_contracts() if state.phase == "funded" else mff_config["eval"]["max_contracts"]
        )

        n_trades, _, pnl = run_day_kernel_portfolio(
            session_data["open"][start:end],
            session_data["high"][start:end],
            session_data["low"][start:end],
            session_data["close"][start:end],
            session_data["volume"][start:end],
            session_data["timestamps"][start:end],
            session_data["minute_of_day"][start:end],
            session_data["bar_atr"][start:end],
            session_data["trailing_median_atr"][start:end],
            session_data["daily_atr_ratio"][start:end],
            session_data["rvol"][start:end],
            session_data["close_sma_50"][start:end],
            session_data["daily_regime_bias"][start:end],
            session_data["donchian_high_5"][start:end],
            session_data["donchian_low_5"][start:end],
            session_data["day_of_week"][start:end],
            slippage_lookup,
            day_idx,
            phase_id,
            payout_cycle_id,
            state.get_liquidation_floor_equity(),
            _combined_portfolio_signal_wfo,
            all_trades[total_trade_count:],
            0,
            state.equity,
            0.0,
            active_params,
            strategy_profiles,
        )
        total_trade_count += n_trades
        result = state.update_eod(pnl, state.equity + pnl, had_trade=n_trades > 0, session_date=session_date)
        net_payout = 0.0

        if result == "passed":
            eval_pass_day = rel_day_idx + 1
            state.transition_to_funded()
            funded_payout_cycle_id = 0
        elif result == "blown":
            blown_day = rel_day_idx + 1
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

        if result == "blown" or state.live_transition_ready:
            break

    daily_view = daily_log[:total_day_count]
    return {
        "trade_log": all_trades[:total_trade_count],
        "daily_log": daily_view,
        "final_equity": float(state.equity),
        "payouts_completed": int(state.payouts_completed),
        "funded_days": int((daily_view["phase_id"] == 1).sum()),
        "eval_passed": eval_pass_day != -1,
        "eval_pass_day": eval_pass_day if eval_pass_day != -1 else None,
        "live_transition_ready": bool(state.live_transition_ready),
        "blown": blown_day != -1,
        "blown_day": blown_day if blown_day != -1 else None,
        "status": (
            "live_ready" if state.live_transition_ready else
            "blown" if blown_day != -1 else
            "funded" if state.phase == "funded" else
            "eval"
        ),
    }


def _build_params_array(base_params: np.ndarray, overrides: dict, phase: str) -> np.ndarray:
    p = base_params.copy()
    for idx, val in overrides.items():
        override_phase, override_idx = idx
        if override_phase == phase or override_phase == "shared":
            p[override_idx] = float(val)
    return p


def _serialize_param_overrides(overrides: dict | None) -> dict | None:
    if overrides is None:
        return None
    result = {"eval": {}, "funded": {}, "shared": {}}
    for (phase, param_idx), value in overrides.items():
        result[phase][PARAM_INDEX_TO_NAME.get(param_idx, f"param_{param_idx}")] = float(value)
    if not result["eval"]:
        result.pop("eval")
    if not result["funded"]:
        result.pop("funded")
    if not result["shared"]:
        result.pop("shared")
    return result


def _serialize_param_grid(param_grid: dict[tuple[str, int], list[float]]) -> dict:
    result = {"eval": {}, "funded": {}, "shared": {}}
    for (phase, param_idx), values in param_grid.items():
        result[phase][PARAM_INDEX_TO_NAME.get(param_idx, f"param_{param_idx}")] = [
            float(value) for value in values
        ]
    if not result["eval"]:
        result.pop("eval")
    if not result["funded"]:
        result.pop("funded")
    if not result["shared"]:
        result.pop("shared")
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
    mc_eval_target_length: int = 200,
    mc_funded_target_length: int = 300,
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
        candidate_pool = []

        for combo in all_combos:
            overrides = dict(zip(grid_keys, combo))
            params_eval = _build_params_array(base_params_eval, overrides, "eval")
            params_funded = _build_params_array(base_params_funded, overrides, "funded")

            artifacts = _backtest_param_set(
                session_data, (0, train_end), params_eval, params_funded, slippage_lookup, mff_config)

            if len(artifacts["daily_log"]) < 10:
                continue

            trade_stats = _compute_trade_stats(artifacts["trade_log"])
            candidate_pool.append({
                "overrides": overrides,
                "artifacts": artifacts,
                "trade_stats": trade_stats,
            })

        candidate_pool.sort(
            key=lambda item: (
                item["trade_stats"]["expectancy"],
                item["trade_stats"]["win_rate"],
                item["trade_stats"]["trade_count"],
            ),
            reverse=True,
        )
        candidate_pool = candidate_pool[:MAX_MC_CANDIDATES_PER_WINDOW]

        for candidate in candidate_pool:
            overrides = candidate["overrides"]
            artifacts = candidate["artifacts"]

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
                eval_target_length=mc_eval_target_length,
                funded_target_length=mc_funded_target_length,
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
                        eval_target_length=mc_eval_target_length,
                        funded_target_length=mc_funded_target_length,
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
            oos_trade_stats = _compute_trade_stats(oos_artifacts["trade_log"])
        else:
            oos_nve = None
            oos_payout_rate = None
            oos_status = "not_scored"
            oos_trade_stats = {
                "win_rate": None,
                "avg_win": None,
                "avg_loss": None,
                "expectancy": None,
                "trade_count": 0,
            }

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
            "oos_win_rate": oos_trade_stats["win_rate"],
            "oos_avg_win": oos_trade_stats["avg_win"],
            "oos_avg_loss": oos_trade_stats["avg_loss"],
            "oos_expectancy": oos_trade_stats["expectancy"],
            "oos_trade_count": oos_trade_stats["trade_count"],
            "oos_nve": oos_nve,
            "oos_payout_rate": oos_payout_rate,
            "status": "ok" if best_combo else "not_scored",
            "oos_status": oos_status if best_combo else "not_scored",
        })

        train_end += step_days
        window_idx += 1

    return results


def run_walk_forward_portfolio_validation(
    session_data: dict,
    slippage_lookup: np.ndarray,
    params_eval: np.ndarray,
    params_funded: np.ndarray,
    strategy_profiles_eval: np.ndarray,
    strategy_profiles_funded: np.ndarray,
    mff_config: dict,
    window_train_days: int = 60,
    window_test_days: int = 30,
    step_days: int = 30,
) -> list[dict]:
    n_total_days = len(session_data["day_boundaries"])
    results = []
    window_idx = 0
    train_start = 0

    while train_start + window_train_days + window_test_days <= n_total_days:
        train_end = train_start + window_train_days
        test_start = train_end
        test_end = test_start + window_test_days

        train_artifacts = _backtest_portfolio_window(
            session_data,
            (train_start, train_end),
            params_eval,
            params_funded,
            strategy_profiles_eval,
            strategy_profiles_funded,
            slippage_lookup,
            mff_config,
        )
        oos_artifacts = _backtest_portfolio_window(
            session_data,
            (test_start, test_end),
            params_eval,
            params_funded,
            strategy_profiles_eval,
            strategy_profiles_funded,
            slippage_lookup,
            mff_config,
        )
        train_summary = _summarize_backtest_result(train_artifacts)
        oos_summary = _summarize_backtest_result(oos_artifacts)

        results.append({
            "window": window_idx,
            "train_date_range": (
                session_data["session_dates"][train_start],
                session_data["session_dates"][train_end - 1],
            ),
            "test_date_range": (
                session_data["session_dates"][test_start],
                session_data["session_dates"][test_end - 1],
            ),
            "train": train_summary,
            "oos": oos_summary,
        })

        train_start += step_days
        window_idx += 1

    return results
