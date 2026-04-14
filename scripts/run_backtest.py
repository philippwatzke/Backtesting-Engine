#!/usr/bin/env python
"""Run the MGC H1 trend backtest on the full train split."""
import argparse
from pathlib import Path

import numpy as np

from propfirm.core.engine import run_day_kernel_portfolio
from propfirm.core.types import (
    DAILY_LOG_DTYPE,
    PARAMS_CONTRACTS,
    PARAMS_EXTRA_SLIPPAGE_TICKS,
    PARAMS_MAX_TRADES,
    PARAMS_TIME_STOP_MINUTE,
    PROFILE_ARRAY_LENGTH,
    PROFILE_BREAKEVEN_TRIGGER_TICKS,
    PROFILE_RISK_BUFFER_FRACTION,
    PROFILE_RISK_PER_TRADE_USD,
    PROFILE_STOP_ATR_MULTIPLIER,
    PROFILE_TARGET_ATR_MULTIPLIER,
    TRADE_LOG_DTYPE,
)
from propfirm.io.config import build_phase_params, load_mff_config, load_params_config
from propfirm.io.reporting import build_report, save_report
from propfirm.market.data_loader import load_session_data
from propfirm.market.slippage import build_slippage_lookup
from propfirm.rules.mff import MFFState
from propfirm.strategy.portfolio import combined_portfolio_signal


def _compute_raw_metrics(trade_log: np.ndarray) -> dict:
    total_trades = int(len(trade_log))
    if total_trades == 0:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "net_profit": 0.0,
            "final_equity": 0.0,
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
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/MGC_1m_full_train.parquet"))
    parser.add_argument("--mff-config", type=Path, default=Path("configs/mff_flex_50k.toml"))
    parser.add_argument("--params-config", type=Path, default=Path("configs/default_params.toml"))
    parser.add_argument("--output", type=Path, default=Path("output/backtests_mgc_h1_trend"))
    parser.add_argument("--timeframe-minutes", type=int, default=60)
    parser.add_argument("--use-daily-regime-filter", action="store_true")
    parser.add_argument("--extra-slippage-ticks", type=float, default=0.0)
    args = parser.parse_args()

    mff_cfg = load_mff_config(args.mff_config)
    params_cfg = load_params_config(args.params_config)
    shared_cfg = params_cfg["strategy"]["mgc_h1_trend"]["shared"]
    eval_cfg = params_cfg["strategy"]["mgc_h1_trend"]["eval"]
    funded_cfg = params_cfg["strategy"]["mgc_h1_trend"]["funded"]
    portfolio_shared = params_cfg.get("portfolio", {}).get("shared", {})
    slip_cfg = params_cfg["slippage"]
    risk_buffer_fraction = float(portfolio_shared.get("risk_buffer_fraction", 0.25))

    data = load_session_data(
        args.data,
        atr_period=slip_cfg["atr_period"],
        trailing_atr_days=slip_cfg["trailing_atr_days"],
        timeframe_minutes=args.timeframe_minutes,
        session_start=shared_cfg["session_start"],
        session_end=shared_cfg["session_end"],
    )
    daily_regime_bias = (
        data["daily_regime_bias"]
        if args.use_daily_regime_filter
        else np.full(len(data["close"]), np.nan, dtype=np.float64)
    )

    slippage_lookup = build_slippage_lookup(
        None,
        require_file=False,
        session_minutes=int(data["session_minutes"]),
    )

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
    params_eval[PARAMS_EXTRA_SLIPPAGE_TICKS] = float(args.extra_slippage_ticks)
    params_funded[PARAMS_EXTRA_SLIPPAGE_TICKS] = float(args.extra_slippage_ticks)

    strategy_profiles_eval = np.zeros((1, PROFILE_ARRAY_LENGTH), dtype=np.float64)
    strategy_profiles_eval[0, PROFILE_RISK_PER_TRADE_USD] = float(eval_cfg["risk_per_trade_usd"])
    strategy_profiles_eval[0, PROFILE_STOP_ATR_MULTIPLIER] = float(eval_cfg["stop_atr_multiplier"])
    strategy_profiles_eval[0, PROFILE_TARGET_ATR_MULTIPLIER] = float(eval_cfg["target_atr_multiplier"])
    strategy_profiles_eval[0, PROFILE_BREAKEVEN_TRIGGER_TICKS] = float(shared_cfg["breakeven_trigger_ticks"])
    strategy_profiles_eval[0, PROFILE_RISK_BUFFER_FRACTION] = risk_buffer_fraction

    strategy_profiles_funded = np.zeros((1, PROFILE_ARRAY_LENGTH), dtype=np.float64)
    strategy_profiles_funded[0, PROFILE_RISK_PER_TRADE_USD] = float(funded_cfg["risk_per_trade_usd"])
    strategy_profiles_funded[0, PROFILE_STOP_ATR_MULTIPLIER] = float(funded_cfg["stop_atr_multiplier"])
    strategy_profiles_funded[0, PROFILE_TARGET_ATR_MULTIPLIER] = float(funded_cfg["target_atr_multiplier"])
    strategy_profiles_funded[0, PROFILE_BREAKEVEN_TRIGGER_TICKS] = float(shared_cfg["breakeven_trigger_ticks"])
    strategy_profiles_funded[0, PROFILE_RISK_BUFFER_FRACTION] = risk_buffer_fraction

    max_trades_per_day = int(max(params_eval[PARAMS_MAX_TRADES], params_funded[PARAMS_MAX_TRADES]))
    max_possible_trades = max(1, len(data["day_boundaries"]) * max_trades_per_day)

    state = MFFState(mff_cfg)
    funded_payout_cycle_id = 0
    all_trades = np.zeros(max_possible_trades, dtype=TRADE_LOG_DTYPE)
    daily_log = np.zeros(len(data["day_boundaries"]), dtype=DAILY_LOG_DTYPE)
    total_trade_count = 0
    total_day_count = 0

    for day_idx, (start, end) in enumerate(data["day_boundaries"]):
        session_date = data["session_dates"][day_idx]

        active_params = params_eval.copy() if state.phase == "eval" else params_funded.copy()
        strategy_profiles = strategy_profiles_eval.copy() if state.phase == "eval" else strategy_profiles_funded.copy()
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
            daily_regime_bias[start:end],
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
        net_payout = 0.0

        if result == "passed":
            print(f"EVAL PASSED on day {day_idx + 1} ({state.trading_days} trading days)")
            state.transition_to_funded()
            funded_payout_cycle_id = 0
        elif result == "blown":
            print(f"BLOWN on day {day_idx + 1}")
            print(f"Equity: ${state.equity:.2f}")
        elif state.phase == "funded" and state.payout_eligible:
            net_payout = state.process_payout()
            if net_payout > 0:
                print(f"PAYOUT #{state.payouts_completed}: net=${net_payout:.2f}")
                funded_payout_cycle_id += 1
                if state.live_transition_ready:
                    print(f"LIVE READY after payout #{state.payouts_completed} ({state.live_transition_reason})")

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

    trade_log = all_trades[:total_trade_count]
    daily_log = daily_log[:total_day_count]
    raw_metrics = _compute_raw_metrics(trade_log)

    print(f"Total trades: {raw_metrics['total_trades']}")
    print(f"Win rate: {raw_metrics['win_rate']:.2%}")
    print(f"Profit factor: {raw_metrics['profit_factor']:.2f}")
    print(f"Final equity: ${raw_metrics['final_equity']:.2f}")

    args.output.mkdir(parents=True, exist_ok=True)
    trade_log_path = args.output / "latest_trade_log.npy"
    daily_log_path = args.output / "latest_daily_log.npy"
    np.save(trade_log_path, trade_log)
    print(f"Trade log saved to {trade_log_path} ({total_trade_count} trades)")
    np.save(daily_log_path, daily_log)
    print(f"Daily lifecycle log saved to {daily_log_path} ({total_day_count} days)")

    pnl_path = args.output / "latest_trade_pnls.npy"
    np.save(pnl_path, trade_log["net_pnl"])
    print(f"Trade PNLs saved to {pnl_path}")

    report = build_report(
        params={
            "portfolio": {
                "shared": {
                    "max_trades_day": int(params_eval[PARAMS_MAX_TRADES]),
                    "strategies": ["mgc_h1_trend"],
                    "risk_buffer_fraction": risk_buffer_fraction,
                    "session_start": shared_cfg["session_start"],
                    "session_end": shared_cfg["session_end"],
                    "trigger_start_minute": int(shared_cfg["trigger_start_minute"]),
                    "trigger_end_minute": int(shared_cfg["trigger_end_minute"]),
                    "timeframe_minutes": int(args.timeframe_minutes),
                    "use_daily_regime_filter": bool(args.use_daily_regime_filter),
                },
                "eval": {
                    "daily_stop": eval_cfg["daily_stop"],
                    "daily_target": eval_cfg["daily_target"],
                },
                "funded": {
                    "daily_stop": funded_cfg["daily_stop"],
                    "daily_target": funded_cfg["daily_target"],
                },
                "profiles_eval": {
                    "mgc_h1_trend": {
                        "risk_per_trade_usd": float(eval_cfg["risk_per_trade_usd"]),
                        "stop_atr_multiplier": float(eval_cfg["stop_atr_multiplier"]),
                        "target_atr_multiplier": float(eval_cfg["target_atr_multiplier"]),
                        "donchian_lookback": int(shared_cfg["donchian_lookback"]),
                        "trigger_start_minute": int(shared_cfg["trigger_start_minute"]),
                        "trigger_end_minute": int(shared_cfg["trigger_end_minute"]),
                        "time_stop_minute": int(shared_cfg["time_stop_minute"]),
                        "breakeven_trigger_ticks": float(shared_cfg["breakeven_trigger_ticks"]),
                    },
                },
                "profiles_funded": {
                    "mgc_h1_trend": {
                        "risk_per_trade_usd": float(funded_cfg["risk_per_trade_usd"]),
                        "stop_atr_multiplier": float(funded_cfg["stop_atr_multiplier"]),
                        "target_atr_multiplier": float(funded_cfg["target_atr_multiplier"]),
                        "donchian_lookback": int(shared_cfg["donchian_lookback"]),
                        "trigger_start_minute": int(shared_cfg["trigger_start_minute"]),
                        "trigger_end_minute": int(shared_cfg["trigger_end_minute"]),
                        "time_stop_minute": int(shared_cfg["time_stop_minute"]),
                        "breakeven_trigger_ticks": float(shared_cfg["breakeven_trigger_ticks"]),
                    },
                },
            }
        },
        mc_result=None,
        config_snapshot={"params_cfg": params_cfg, "mff_cfg": mff_cfg},
        data_split=args.data.stem,
        data_date_range=(data["session_dates"][0], data["session_dates"][-1]),
        seed=params_cfg["general"]["random_seed"],
        stress_test={
            "extra_slippage_ticks": float(args.extra_slippage_ticks),
            "enabled": bool(args.extra_slippage_ticks > 0.0),
        },
    )
    report["artifacts"] = {
        "daily_log": str(daily_log_path),
        "trade_log": str(trade_log_path),
        "trade_pnls": str(pnl_path),
    }
    report["runtime_meta"] = {
        "raw_metrics": {
            **raw_metrics,
            "ending_account_equity": float(state.equity),
        },
        "mc_mode_recommended": (
            "daily"
            if (
                np.any(daily_log["phase_id"] == 0)
                and np.any((daily_log["phase_id"] == 1) & (daily_log["payout_cycle_id"] == 0))
            )
            else "not_ready"
        ),
        "mc_daily_lifecycle_ready": bool(
            np.any(daily_log["phase_id"] == 0)
            and np.any((daily_log["phase_id"] == 1) & (daily_log["payout_cycle_id"] == 0))
        ),
        "lifecycle_aware_daily_log": True,
        "payouts_completed": state.payouts_completed,
        "live_transition_ready": state.live_transition_ready,
        "live_transition_reason": state.live_transition_reason,
        "inactive_day": -1,
        "inactive": False,
    }
    save_report(report, args.output / "latest_backtest.json")
    print(f"Report saved to {args.output / 'latest_backtest.json'}")


if __name__ == "__main__":
    main()
