#!/usr/bin/env python
"""Run the NT8 dual-feed breakout backtest with strict IS/OOS reporting."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from propfirm.execution.nt8_dual_feed import (
    BacktestMetrics,
    compute_performance_metrics,
    load_nt8_raw_feed,
    load_nt8_signal_feed,
    run_nt8_dual_feed_backtest,
    split_is_oos_metrics,
)
from propfirm.io.config import load_mff_config, load_params_config
from propfirm.io.reporting import build_report, save_report


DEFAULTS = {
    "MNQ": {
        "signal_csv": Path("data/ninjatrader/NT8_Dump_MNQ_5000_days.csv"),
        "raw_csv": Path("data/ninjatrader/NT8_RawDump_MNQ_5000_days.csv"),
        "params_config": Path("configs/default_params_mnq_legacy_frozen.toml"),
        "mff_config": Path("configs/mff_flex_50k_mnq.toml"),
    },
    "MGC": {
        "signal_csv": Path("data/ninjatrader/NT8_Dump_MGC_5000_days.csv"),
        "raw_csv": Path("data/ninjatrader/NT8_RawDump_MGC_5000_days.csv"),
        "params_config": Path("configs/default_params.toml"),
        "mff_config": Path("configs/mff_flex_50k_mgc.toml"),
    },
}


def _metrics_to_dict(metrics: BacktestMetrics) -> dict:
    return {
        "trade_count": metrics.trade_count,
        "win_rate": metrics.win_rate,
        "profit_factor": metrics.profit_factor,
        "net_profit": metrics.net_profit,
        "max_drawdown": metrics.max_drawdown,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", choices=["MNQ", "MGC", "BOTH"], default="BOTH")
    parser.add_argument("--phase", choices=["eval", "funded"], default="eval")
    parser.add_argument("--is-end-date", default="2024-12-31")
    parser.add_argument("--oos-start-date", default="2025-01-01")
    parser.add_argument("--output", type=Path, default=Path("output/nt8_dual_feed"))
    parser.add_argument("--disable-daily-regime-filter", action="store_true")
    return parser.parse_args()


def _run_single_instrument(
    instrument: str,
    phase: str,
    is_end_date: str,
    oos_start_date: str,
    use_daily_regime_filter: bool,
) -> tuple[dict, Path]:
    defaults = DEFAULTS[instrument]
    signal_df = load_nt8_signal_feed(defaults["signal_csv"])
    raw_df = load_nt8_raw_feed(defaults["raw_csv"])
    params_cfg = load_params_config(defaults["params_config"])
    mff_cfg = load_mff_config(defaults["mff_config"])

    trades = run_nt8_dual_feed_backtest(
        signal_df=signal_df,
        raw_df=raw_df,
        params_cfg=params_cfg,
        mff_cfg=mff_cfg,
        phase=phase,
        use_daily_regime_filter=use_daily_regime_filter,
    )
    overall = compute_performance_metrics(trades)
    split = split_is_oos_metrics(trades, is_end_date=is_end_date, oos_start_date=oos_start_date)
    return {
        "instrument": instrument,
        "phase": phase,
        "signal_csv": str(defaults["signal_csv"]),
        "raw_csv": str(defaults["raw_csv"]),
        "params_config": str(defaults["params_config"]),
        "mff_config": str(defaults["mff_config"]),
        "use_daily_regime_filter": use_daily_regime_filter,
        "overall": _metrics_to_dict(overall),
        "is": _metrics_to_dict(split["is"]),
        "oos": _metrics_to_dict(split["oos"]),
        "trades": trades,
    }, defaults["params_config"]


def main() -> None:
    args = parse_args()
    instruments = ["MNQ", "MGC"] if args.instrument == "BOTH" else [args.instrument]
    use_daily_regime_filter = not args.disable_daily_regime_filter

    reports: dict[str, dict] = {}
    combined_trades = []
    trades_by_instrument: dict[str, object] = {}

    for instrument in instruments:
        result, params_config_path = _run_single_instrument(
            instrument=instrument,
            phase=args.phase,
            is_end_date=args.is_end_date,
            oos_start_date=args.oos_start_date,
            use_daily_regime_filter=use_daily_regime_filter,
        )
        reports[instrument] = {k: v for k, v in result.items() if k != "trades"}
        trades = result["trades"].copy()
        trades["instrument"] = instrument
        combined_trades.append(trades)
        trades_by_instrument[instrument] = trades

    if combined_trades:
        import pandas as pd

        combined = pd.concat(combined_trades, ignore_index=True).sort_values(["entry_time", "exit_time"], kind="stable")
    else:
        import pandas as pd

        combined = pd.DataFrame()

    combined_overall = compute_performance_metrics(combined)
    combined_split = split_is_oos_metrics(combined, is_end_date=args.is_end_date, oos_start_date=args.oos_start_date)

    args.output.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output / "trades_combined.csv", index=False)
    for instrument, result in reports.items():
        df = trades_by_instrument[instrument]
        df.to_csv(args.output / f"trades_{instrument.lower()}.csv", index=False)

    report = build_report(
        params={
            "architecture": {
                "mode": "nt8_dual_feed",
                "signal_feed": "nt8_htf_dump",
                "execution_feed": "nt8_raw_1m",
                "phase": args.phase,
                "daily_regime_filter": use_daily_regime_filter,
            }
        },
        mc_result=None,
        config_snapshot={"per_instrument": reports},
        data_split="nt8_dual_feed_is_oos",
        data_date_range=(args.is_end_date, "latest_available"),
        seed=42,
        diagnostics={
            "per_instrument": reports,
            "combined": {
                "overall": _metrics_to_dict(combined_overall),
                "is": _metrics_to_dict(combined_split["is"]),
                "oos": _metrics_to_dict(combined_split["oos"]),
            },
        },
    )
    report["artifacts"] = {
        "combined_trades": str(args.output / "trades_combined.csv"),
        **{f"{instrument.lower()}_trades": str(args.output / f"trades_{instrument.lower()}.csv") for instrument in reports},
    }
    save_report(report, args.output / "nt8_dual_feed_report.json")

    print("NT8 dual-feed engine ready.")
    for instrument, result in reports.items():
        print(
            f"{instrument} {args.phase}: "
            f"IS trades={result['is']['trade_count']} PF={result['is']['profit_factor']:.2f} NP={result['is']['net_profit']:.2f} | "
            f"OOS trades={result['oos']['trade_count']} PF={result['oos']['profit_factor']:.2f} NP={result['oos']['net_profit']:.2f}"
        )
    print(
        f"Combined {args.phase}: "
        f"IS trades={combined_split['is'].trade_count} PF={combined_split['is'].profit_factor:.2f} NP={combined_split['is'].net_profit:.2f} | "
        f"OOS trades={combined_split['oos'].trade_count} PF={combined_split['oos'].profit_factor:.2f} NP={combined_split['oos'].net_profit:.2f}"
    )
    print(f"Report saved to {args.output / 'nt8_dual_feed_report.json'}")


if __name__ == "__main__":
    main()
