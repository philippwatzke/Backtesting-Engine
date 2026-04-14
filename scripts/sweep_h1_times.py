#!/usr/bin/env python
"""Sweep MGC H1 trend entry windows across candidate start hours."""
import argparse
import copy
from pathlib import Path

import numpy as np

from propfirm.core.engine import run_day_kernel_portfolio
from propfirm.core.types import (
    PARAMS_CONTRACTS,
    PARAMS_EXTRA_SLIPPAGE_TICKS,
    PARAMS_MAX_TRADES,
    PROFILE_ARRAY_LENGTH,
    PROFILE_BREAKEVEN_TRIGGER_TICKS,
    PROFILE_RISK_BUFFER_FRACTION,
    PROFILE_RISK_PER_TRADE_USD,
    PROFILE_STOP_ATR_MULTIPLIER,
    PROFILE_TARGET_ATR_MULTIPLIER,
    TRADE_LOG_DTYPE,
)
from propfirm.io.config import build_phase_params, load_mff_config, load_params_config
from propfirm.market.data_loader import load_session_data
from propfirm.market.slippage import build_slippage_lookup
from propfirm.rules.mff import MFFState
from propfirm.strategy.portfolio import combined_portfolio_signal


def _parse_hhmm(value: str) -> int:
    hours, minutes = value.split(":")
    return int(hours) * 60 + int(minutes)


def _clock_to_session_minute(clock_hhmm: str, session_start_hhmm: str) -> int:
    return _parse_hhmm(clock_hhmm) - _parse_hhmm(session_start_hhmm)


def _compute_raw_metrics(trade_log: np.ndarray, ending_account_equity: float) -> dict:
    total_trades = int(len(trade_log))
    if total_trades == 0:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "net_profit": 0.0,
            "final_equity": 0.0,
            "ending_account_equity": float(ending_account_equity),
        }
    net_pnl = trade_log["net_pnl"].astype(np.float64)
    gross_profit = float(net_pnl[net_pnl > 0.0].sum())
    gross_loss = float(-net_pnl[net_pnl < 0.0].sum())
    net_profit = float(net_pnl.sum())
    return {
        "total_trades": total_trades,
        "win_rate": float(np.mean(net_pnl > 0.0)),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0.0 else float("inf"),
        "net_profit": net_profit,
        "final_equity": net_profit,
        "ending_account_equity": float(ending_account_equity),
    }


def _build_strategy_profiles(shared_cfg: dict, phase_cfg: dict, risk_buffer_fraction: float) -> np.ndarray:
    profiles = np.zeros((1, PROFILE_ARRAY_LENGTH), dtype=np.float64)
    profiles[0, PROFILE_RISK_PER_TRADE_USD] = float(phase_cfg["risk_per_trade_usd"])
    profiles[0, PROFILE_STOP_ATR_MULTIPLIER] = float(phase_cfg["stop_atr_multiplier"])
    profiles[0, PROFILE_TARGET_ATR_MULTIPLIER] = float(phase_cfg["target_atr_multiplier"])
    profiles[0, PROFILE_BREAKEVEN_TRIGGER_TICKS] = float(shared_cfg["breakeven_trigger_ticks"])
    profiles[0, PROFILE_RISK_BUFFER_FRACTION] = float(risk_buffer_fraction)
    return profiles


def _run_case(
    data: dict,
    slippage_lookup: np.ndarray,
    mff_cfg: dict,
    shared_cfg: dict,
    eval_cfg: dict,
    funded_cfg: dict,
    slip_cfg: dict,
    risk_buffer_fraction: float,
    use_daily_regime_filter: bool,
) -> dict:
    params_eval = build_phase_params(
        shared_cfg,
        eval_cfg,
        slip_cfg,
        mff_cfg["instrument"]["commission_per_side"],
        strategy_name="mgc_h1_trend",
        instrument_cfg=mff_cfg["instrument"],
    )
    params_funded = build_phase_params(
        shared_cfg,
        funded_cfg,
        slip_cfg,
        mff_cfg["instrument"]["commission_per_side"],
        strategy_name="mgc_h1_trend",
        instrument_cfg=mff_cfg["instrument"],
    )
    params_eval[PARAMS_EXTRA_SLIPPAGE_TICKS] = 0.0
    params_funded[PARAMS_EXTRA_SLIPPAGE_TICKS] = 0.0

    strategy_profiles_eval = _build_strategy_profiles(shared_cfg, eval_cfg, risk_buffer_fraction)
    strategy_profiles_funded = _build_strategy_profiles(shared_cfg, funded_cfg, risk_buffer_fraction)

    max_trades_per_day = int(max(params_eval[PARAMS_MAX_TRADES], params_funded[PARAMS_MAX_TRADES]))
    max_possible_trades = max(1, len(data["day_boundaries"]) * max_trades_per_day)
    all_trades = np.zeros(max_possible_trades, dtype=TRADE_LOG_DTYPE)

    state = MFFState(mff_cfg)
    funded_payout_cycle_id = 0
    total_trade_count = 0

    for day_idx, (start, end) in enumerate(data["day_boundaries"]):
        session_date = data["session_dates"][day_idx]
        active_params = params_eval.copy() if state.phase == "eval" else params_funded.copy()
        strategy_profiles = strategy_profiles_eval if state.phase == "eval" else strategy_profiles_funded
        phase_id = 0 if state.phase == "eval" else 1
        payout_cycle_id = -1 if state.phase == "eval" else funded_payout_cycle_id
        active_params[PARAMS_CONTRACTS] = float(
            state.get_max_contracts() if state.phase == "funded" else mff_cfg["eval"]["max_contracts"]
        )

        n_trades, _, pnl = run_day_kernel_portfolio(
            data["open"][start:end],
            data["high"][start:end],
            data["low"][start:end],
            data["close"][start:end],
            data["volume"][start:end],
            data["timestamps"][start:end],
            data["minute_of_day"][start:end],
            data["bar_atr"][start:end],
            data["trailing_median_atr"][start:end],
            data["daily_atr_ratio"][start:end],
            data["rvol"][start:end],
            data["close_sma_50"][start:end],
            data["daily_regime_bias"][start:end] if use_daily_regime_filter else np.full(end - start, np.nan, dtype=np.float64),
            data["donchian_high_5"][start:end],
            data["donchian_low_5"][start:end],
            data["day_of_week"][start:end],
            slippage_lookup,
            day_idx,
            phase_id,
            payout_cycle_id,
            state.get_liquidation_floor_equity(),
            combined_portfolio_signal,
            all_trades[total_trade_count:],
            0,
            state.equity,
            0.0,
            active_params,
            strategy_profiles,
        )

        total_trade_count += n_trades
        result = state.update_eod(pnl, state.equity + pnl, had_trade=n_trades > 0, session_date=session_date)

        if result == "passed":
            state.transition_to_funded()
            funded_payout_cycle_id = 0
        elif state.phase == "funded" and state.payout_eligible:
            net_payout = state.process_payout()
            if net_payout > 0:
                funded_payout_cycle_id += 1

        if result == "blown" or state.live_transition_ready:
            break

    trade_log = all_trades[:total_trade_count]
    return _compute_raw_metrics(trade_log, state.equity)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/MGC_1m_full_train.parquet"))
    parser.add_argument("--mff-config", type=Path, default=Path("configs/mff_flex_50k.toml"))
    parser.add_argument("--params-config", type=Path, default=Path("configs/default_params.toml"))
    parser.add_argument("--use-daily-regime-filter", action="store_true")
    args = parser.parse_args()

    mff_cfg = load_mff_config(args.mff_config)
    params_cfg = load_params_config(args.params_config)
    base_shared_cfg = params_cfg["strategy"]["mgc_h1_trend"]["shared"]
    eval_cfg = params_cfg["strategy"]["mgc_h1_trend"]["eval"]
    funded_cfg = params_cfg["strategy"]["mgc_h1_trend"]["funded"]
    slip_cfg = params_cfg["slippage"]
    risk_buffer_fraction = float(params_cfg.get("portfolio", {}).get("shared", {}).get("risk_buffer_fraction", 0.25))

    data = load_session_data(
        args.data,
        atr_period=slip_cfg["atr_period"],
        trailing_atr_days=slip_cfg["trailing_atr_days"],
        timeframe_minutes=60,
        session_start=base_shared_cfg["session_start"],
        session_end=base_shared_cfg["session_end"],
    )
    slippage_lookup = build_slippage_lookup(
        None,
        require_file=False,
        session_minutes=int(data["session_minutes"]),
    )

    trigger_end_minute = _clock_to_session_minute("14:00", base_shared_cfg["session_start"])
    candidates = ["08:00", "09:00", "10:00", "11:00"]

    print("Start-Stunde | Gesamttrades | Winrate | Profit Factor | Final Equity")
    print("------------ | ------------ | ------- | ------------- | ------------")
    for start_hhmm in candidates:
        shared_cfg = copy.deepcopy(base_shared_cfg)
        shared_cfg["trigger_start_minute"] = _clock_to_session_minute(start_hhmm, base_shared_cfg["session_start"])
        shared_cfg["trigger_end_minute"] = trigger_end_minute
        metrics = _run_case(
            data,
            slippage_lookup,
            mff_cfg,
            shared_cfg,
            eval_cfg,
            funded_cfg,
            slip_cfg,
            risk_buffer_fraction,
            args.use_daily_regime_filter,
        )
        print(
            f"{start_hhmm} ET | "
            f"{metrics['total_trades']:12d} | "
            f"{metrics['win_rate']:.2%} | "
            f"{metrics['profit_factor']:.4f} | "
            f"{metrics['final_equity']:.2f}"
        )


if __name__ == "__main__":
    main()
