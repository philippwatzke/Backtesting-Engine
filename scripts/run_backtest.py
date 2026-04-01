#!/usr/bin/env python
"""Run a single strategy backtest."""
import argparse
import numpy as np
from pathlib import Path

from propfirm.io.config import load_mff_config, load_params_config, build_phase_params
from propfirm.market.data_loader import load_session_data
from propfirm.market.slippage import build_slippage_lookup
from propfirm.core.engine import run_day_kernel
from propfirm.core.types import TRADE_LOG_DTYPE, DAILY_LOG_DTYPE, PARAMS_MAX_TRADES, PARAMS_CONTRACTS
from propfirm.risk.risk import validate_position_size
from propfirm.rules.mff import MFFState
from propfirm.strategy.orb import orb_signal
from propfirm.io.reporting import build_report, save_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/MNQ_1m_train.parquet"))
    parser.add_argument("--mff-config", type=Path, default=Path("configs/mff_flex_50k.toml"))
    parser.add_argument("--params-config", type=Path, default=Path("configs/default_params.toml"))
    parser.add_argument("--output", type=Path, default=Path("output/backtests"))
    args = parser.parse_args()

    mff_cfg = load_mff_config(args.mff_config)
    params_cfg = load_params_config(args.params_config)
    orb_shared = params_cfg["strategy"]["orb"]["shared"]
    orb_eval = params_cfg["strategy"]["orb"]["eval"]
    orb_funded = params_cfg["strategy"]["orb"]["funded"]
    slip_cfg = params_cfg["slippage"]

    data = load_session_data(
        args.data,
        atr_period=slip_cfg["atr_period"],
        trailing_atr_days=slip_cfg["trailing_atr_days"],
    )

    slippage_lookup = build_slippage_lookup(
        Path("data/slippage/slippage_profile.parquet"),
        require_file=True,
    )

    params_eval = build_phase_params(
        orb_shared, orb_eval, slip_cfg, mff_cfg["instrument"]["commission_per_side"]
    )
    params_funded = build_phase_params(
        orb_shared, orb_funded, slip_cfg, mff_cfg["instrument"]["commission_per_side"]
    )

    max_trades_per_day = int(max(params_eval[PARAMS_MAX_TRADES], params_funded[PARAMS_MAX_TRADES]))
    max_possible_trades = max(1, len(data["day_boundaries"]) * max_trades_per_day)

    state = MFFState(mff_cfg)
    funded_payout_cycle_id = 0
    all_trades = np.zeros(max_possible_trades, dtype=TRADE_LOG_DTYPE)
    daily_log = np.zeros(len(data["day_boundaries"]), dtype=DAILY_LOG_DTYPE)
    total_trade_count = 0
    total_day_count = 0

    for day_idx, (start, end) in enumerate(data["day_boundaries"]):
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
            data["open"][start:end],
            data["high"][start:end],
            data["low"][start:end],
            data["close"][start:end],
            data["volume"][start:end],
            data["timestamps"][start:end],
            data["minute_of_day"][start:end],
            data["bar_atr"][start:end],
            data["trailing_median_atr"][start:end],
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

        if result == "passed":
            print(f"EVAL PASSED on day {day_idx + 1} ({state.trading_days} trading days)")
            state.transition_to_funded()
            funded_payout_cycle_id = 0
        elif result == "blown":
            print(f"BLOWN on day {day_idx + 1}")
            print(f"Equity: ${state.equity:.2f}")
            pass
        elif state.phase == "funded" and state.payout_eligible:
            net_payout = state.process_payout()
            if net_payout > 0:
                print(f"PAYOUT #{state.payouts_completed}: net=${net_payout:.2f}")
                funded_payout_cycle_id += 1

        daily_log[total_day_count]["day_id"] = day_idx
        daily_log[total_day_count]["phase_id"] = phase_id
        daily_log[total_day_count]["payout_cycle_id"] = payout_cycle_id
        daily_log[total_day_count]["had_trade"] = 1 if n_trades > 0 else 0
        daily_log[total_day_count]["n_trades"] = n_trades
        daily_log[total_day_count]["day_pnl"] = pnl
        daily_log[total_day_count]["net_payout"] = net_payout
        total_day_count += 1

        if result == "blown":
            break

    print(f"Total trades: {total_trade_count}")
    print(f"Final equity: ${state.equity:.2f}")

    args.output.mkdir(parents=True, exist_ok=True)
    trade_log = all_trades[:total_trade_count]
    daily_log = daily_log[:total_day_count]
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
        params={"eval": orb_eval, "funded": orb_funded, "shared": orb_shared},
        mc_result=None,
        config_snapshot={"params_cfg": params_cfg, "mff_cfg": mff_cfg},
        data_split=args.data.stem,
        data_date_range=(data["session_dates"][0], data["session_dates"][-1]),
        seed=params_cfg["general"]["random_seed"],
    )
    report["artifacts"] = {
        "daily_log": str(daily_log_path),
        "trade_log": str(trade_log_path),
        "trade_pnls": str(pnl_path),
    }
    report["runtime_meta"] = {
        "mc_mode_recommended": (
            "daily" if (
                np.any(daily_log["phase_id"] == 0)
                and np.any((daily_log["phase_id"] == 1) & (daily_log["payout_cycle_id"] == 0))
            ) else "not_ready"
        ),
        "mc_daily_lifecycle_ready": bool(
            np.any(daily_log["phase_id"] == 0)
            and np.any((daily_log["phase_id"] == 1) & (daily_log["payout_cycle_id"] == 0))
        ),
        "lifecycle_aware_daily_log": True,
        "payouts_completed": state.payouts_completed,
    }
    save_report(report, args.output / "latest_backtest.json")
    print(f"Report saved to {args.output / 'latest_backtest.json'}")


if __name__ == "__main__":
    main()
