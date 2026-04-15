#!/usr/bin/env python
"""Build a fresh deterministic MGC+MNQ Donchian portfolio OOS artifact set."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from propfirm.core.multi_engine import run_multi_asset_day_kernel
from propfirm.core.types import EXIT_CIRCUIT_BREAKER, TRADE_LOG_DTYPE


DEFAULT_MGC_DATA = Path("data/processed/MGC_1m_full_test.parquet")
DEFAULT_MNQ_DATA = Path("data/processed/MNQ_1m_full_test.parquet")
DEFAULT_MGC_MFF_CONFIG = Path("configs/mff_flex_50k_mgc.toml")
DEFAULT_MNQ_MFF_CONFIG = Path("configs/mff_flex_50k_mnq.toml")
DEFAULT_MGC_PARAMS_CONFIG = Path("configs/default_params.toml")
DEFAULT_MNQ_PARAMS_CONFIG = Path("configs/default_params_mnq.toml")
DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_OUTPUT_STEM = "donchian_portfolio_oos"


_RUN_PORTFOLIO_SPEC = importlib.util.spec_from_file_location(
    "run_portfolio_module",
    Path(__file__).resolve().with_name("run_portfolio.py"),
)
if _RUN_PORTFOLIO_SPEC is None or _RUN_PORTFOLIO_SPEC.loader is None:
    raise RuntimeError("Unable to load scripts/run_portfolio.py")
_RUN_PORTFOLIO_MODULE = importlib.util.module_from_spec(_RUN_PORTFOLIO_SPEC)
_RUN_PORTFOLIO_SPEC.loader.exec_module(_RUN_PORTFOLIO_MODULE)

PORTFOLIO_DAILY_LOG_DTYPE = _RUN_PORTFOLIO_MODULE.PORTFOLIO_DAILY_LOG_DTYPE
PORTFOLIO_TRADE_LOG_DTYPE = _RUN_PORTFOLIO_MODULE.PORTFOLIO_TRADE_LOG_DTYPE
_build_asset_setup = _RUN_PORTFOLIO_MODULE._build_asset_setup
_compute_raw_metrics = _RUN_PORTFOLIO_MODULE._compute_raw_metrics
_slice_asset_config = _RUN_PORTFOLIO_MODULE._slice_asset_config


def _compute_sanity_metrics(trade_log: np.ndarray) -> dict:
    total_trades = int(len(trade_log))
    if total_trades == 0:
        return {
            "total_trades": 0,
            "average_trade_pnl": 0.0,
            "win_rate": 0.0,
        }

    net_pnl = trade_log["net_pnl"].astype(np.float64)
    return {
        "total_trades": total_trades,
        "average_trade_pnl": float(np.mean(net_pnl)),
        "win_rate": float(np.mean(net_pnl > 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mgc-data", type=Path, default=DEFAULT_MGC_DATA)
    parser.add_argument("--mnq-data", type=Path, default=DEFAULT_MNQ_DATA)
    parser.add_argument("--mgc-mff-config", type=Path, default=DEFAULT_MGC_MFF_CONFIG)
    parser.add_argument("--mnq-mff-config", type=Path, default=DEFAULT_MNQ_MFF_CONFIG)
    parser.add_argument("--mgc-params-config", type=Path, default=DEFAULT_MGC_PARAMS_CONFIG)
    parser.add_argument("--mnq-params-config", type=Path, default=DEFAULT_MNQ_PARAMS_CONFIG)
    parser.add_argument("--mgc-timeframe-minutes", type=int, default=60)
    parser.add_argument("--mnq-timeframe-minutes", type=int, default=30)
    parser.add_argument("--circuit-breaker-threshold", type=float, default=-800.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", type=str, default=DEFAULT_OUTPUT_STEM)
    args = parser.parse_args()

    mgc = _build_asset_setup(
        "MGC",
        args.mgc_data,
        args.mgc_mff_config,
        args.mgc_params_config,
        args.mgc_timeframe_minutes,
    )
    mnq = _build_asset_setup(
        "MNQ",
        args.mnq_data,
        args.mnq_mff_config,
        args.mnq_params_config,
        args.mnq_timeframe_minutes,
    )

    master_calendar = sorted(set(mgc["data"]["session_dates"]) | set(mnq["data"]["session_dates"]))
    portfolio_daily = np.zeros(len(master_calendar), dtype=PORTFOLIO_DAILY_LOG_DTYPE)
    portfolio_trade_rows = []

    running_equity = 0.0
    peak_equity = 0.0
    circuit_breaker_days = 0

    for day_id, session_date in enumerate(master_calendar):
        mgc_cfg = _slice_asset_config(mgc, session_date, day_id)
        mnq_cfg = _slice_asset_config(mnq, session_date, day_id)
        result = run_multi_asset_day_kernel(
            [mgc_cfg, mnq_cfg],
            circuit_breaker_threshold=float(args.circuit_breaker_threshold),
        )

        mgc_day_pnl = float(result["assets"]["MGC"]["realized_pnl"])
        mnq_day_pnl = float(result["assets"]["MNQ"]["realized_pnl"])
        portfolio_day_pnl = mgc_day_pnl + mnq_day_pnl
        running_equity += portfolio_day_pnl
        if running_equity > peak_equity:
            peak_equity = running_equity
        drawdown = running_equity - peak_equity

        mgc_log = mgc_cfg["trade_log"][: result["assets"]["MGC"]["n_trades"]]
        mnq_log = mnq_cfg["trade_log"][: result["assets"]["MNQ"]["n_trades"]]
        day_cb_exits = int(
            np.sum(mgc_log["exit_reason"] == EXIT_CIRCUIT_BREAKER)
            + np.sum(mnq_log["exit_reason"] == EXIT_CIRCUIT_BREAKER)
        )
        if result["global_halt"]:
            circuit_breaker_days += 1

        for asset_setup, asset_cfg, asset_name, day_pnl in (
            (mgc, mgc_cfg, "MGC", mgc_day_pnl),
            (mnq, mnq_cfg, "MNQ", mnq_day_pnl),
        ):
            if len(asset_cfg["timestamps"]) == 0 or asset_setup["disabled"]:
                continue

            state = asset_setup["state"]
            n_trades = int(result["assets"][asset_name]["n_trades"])
            lifecycle = state.update_eod(
                day_pnl,
                state.equity + day_pnl,
                had_trade=n_trades > 0,
                session_date=session_date,
            )
            if lifecycle == "passed":
                state.transition_to_funded()
                asset_setup["funded_payout_cycle_id"] = 0
            elif lifecycle == "blown":
                asset_setup["disabled"] = True
            elif state.phase == "funded" and state.payout_eligible:
                net_payout = state.process_payout()
                if net_payout > 0:
                    asset_setup["funded_payout_cycle_id"] += 1

        portfolio_daily[day_id]["session_date"] = session_date
        portfolio_daily[day_id]["mgc_day_pnl"] = mgc_day_pnl
        portfolio_daily[day_id]["mnq_day_pnl"] = mnq_day_pnl
        portfolio_daily[day_id]["portfolio_day_pnl"] = portfolio_day_pnl
        portfolio_daily[day_id]["portfolio_equity"] = running_equity
        portfolio_daily[day_id]["portfolio_drawdown"] = drawdown
        portfolio_daily[day_id]["circuit_breaker_triggered"] = 1 if result["global_halt"] else 0
        portfolio_daily[day_id]["circuit_breaker_exit_trades"] = day_cb_exits

        for asset_name, day_log in (("MGC", mgc_log), ("MNQ", mnq_log)):
            for row in day_log:
                combined_row = np.zeros((), dtype=PORTFOLIO_TRADE_LOG_DTYPE)
                combined_row["asset"] = asset_name
                for field in TRADE_LOG_DTYPE.names:
                    combined_row[field] = row[field]
                portfolio_trade_rows.append(combined_row)

    portfolio_trade_log = (
        np.array(portfolio_trade_rows, dtype=PORTFOLIO_TRADE_LOG_DTYPE)
        if portfolio_trade_rows
        else np.zeros(0, dtype=PORTFOLIO_TRADE_LOG_DTYPE)
    )
    portfolio_pnls = portfolio_trade_log["net_pnl"].astype(np.float64)
    raw_metrics = _compute_raw_metrics(portfolio_trade_log)
    sanity_metrics = _compute_sanity_metrics(portfolio_trade_log)
    combined_max_drawdown = (
        float(portfolio_daily["portfolio_drawdown"].min()) if len(portfolio_daily) else 0.0
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    trade_log_path = args.output_dir / f"{stem}_trade_log.npy"
    daily_log_path = args.output_dir / f"{stem}_daily_log.npy"
    pnl_path = args.output_dir / f"{stem}_pnls.npy"
    report_path = args.output_dir / f"{stem}_report.json"

    np.save(trade_log_path, portfolio_trade_log)
    np.save(daily_log_path, portfolio_daily)
    np.save(pnl_path, portfolio_pnls)

    report = {
        "portfolio": {
            "assets": ["MGC", "MNQ"],
            "data": {
                "mgc": str(args.mgc_data),
                "mnq": str(args.mnq_data),
            },
            "session_dates": {
                "start": master_calendar[0] if master_calendar else None,
                "end": master_calendar[-1] if master_calendar else None,
                "count": len(master_calendar),
            },
            "metrics": {
                "combined_final_net_equity": raw_metrics["final_equity"],
                "combined_max_drawdown": combined_max_drawdown,
                "combined_win_rate": raw_metrics["win_rate"],
                "profit_factor": raw_metrics["profit_factor"],
                "total_trades": raw_metrics["total_trades"],
                "average_trade_pnl": sanity_metrics["average_trade_pnl"],
                "circuit_breaker_trigger_days": circuit_breaker_days,
            },
            "artifacts": {
                "trade_log": str(trade_log_path),
                "daily_log": str(daily_log_path),
                "trade_pnls": str(pnl_path),
            },
        }
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved trade log: {trade_log_path}")
    print(f"Saved daily log: {daily_log_path}")
    print(f"Saved PnL array: {pnl_path}")
    print(f"Saved report: {report_path}")
    print(f"Combined trades: {sanity_metrics['total_trades']}")
    print(f"Average Trade PnL: {sanity_metrics['average_trade_pnl']:.2f}")
    print(f"Winrate: {sanity_metrics['win_rate']:.2%}")


if __name__ == "__main__":
    main()
