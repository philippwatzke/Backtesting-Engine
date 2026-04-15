#!/usr/bin/env python
"""Measure where NT8 dual-feed signals are filtered out."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from propfirm.execution.nt8_dual_feed import (
    _build_signal_arrays,
    _find_entry_index,
    _prepare_raw_execution_feed,
    load_nt8_raw_feed,
    load_nt8_signal_feed,
    run_nt8_dual_feed_backtest,
    split_is_oos_metrics,
)
from propfirm.io.config import load_mff_config, load_params_config


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", choices=["MNQ", "MGC", "BOTH"], default="BOTH")
    parser.add_argument("--phase", choices=["eval", "funded"], default="eval")
    parser.add_argument("--is-end-date", default="2024-12-31")
    parser.add_argument("--oos-start-date", default="2025-01-01")
    parser.add_argument("--output", type=Path, default=Path("output/nt8_signal_census"))
    return parser.parse_args()


def _ts_list(series: pd.Series, limit: int = 3) -> list[str]:
    if series.empty:
        return []
    return [ts.isoformat() for ts in series.iloc[:limit]]


def _run_single_instrument(
    instrument: str,
    phase: str,
    is_end_date: str,
    oos_start_date: str,
) -> dict:
    defaults = DEFAULTS[instrument]
    signal_df = load_nt8_signal_feed(defaults["signal_csv"])
    raw_df = load_nt8_raw_feed(defaults["raw_csv"])
    params_cfg = load_params_config(defaults["params_config"])
    mff_cfg = load_mff_config(defaults["mff_config"])

    shared_cfg = params_cfg["strategy"]["mgc_h1_trend"]["shared"]
    phase_cfg = params_cfg["strategy"]["mgc_h1_trend"][phase]

    signal_arrays, signal_prepared = _build_signal_arrays(signal_df, shared_cfg)
    raw_prepared, raw_ns = _prepare_raw_execution_feed(raw_df, shared_cfg)

    closes = signal_arrays["closes"]
    sma = signal_arrays["sma"]
    regime = signal_arrays["regime"]
    donchian_high = signal_arrays["donchian_high"]
    donchian_low = signal_arrays["donchian_low"]
    minute_of_day = signal_arrays["minute_of_day"]
    day_of_week = signal_arrays["day_of_week"]

    raw_long = np.isfinite(donchian_high) & (closes > donchian_high)
    raw_short = np.isfinite(donchian_low) & (closes < donchian_low)
    stage1 = raw_long | raw_short

    long_with_regime = raw_long & np.isfinite(sma) & (closes > sma) & (regime > 0.0)
    short_with_regime = raw_short & np.isfinite(sma) & (closes < sma) & (regime < 0.0)
    stage2 = long_with_regime | short_with_regime

    trigger_start = int(shared_cfg["trigger_start_minute"])
    trigger_end = int(shared_cfg["trigger_end_minute"])
    blocked_weekday = int(shared_cfg.get("blocked_weekday", -1))
    weekday_allowed = np.ones(len(signal_prepared), dtype=bool)
    if blocked_weekday >= 0:
        weekday_allowed = day_of_week != blocked_weekday

    stage3 = (
        stage2
        & signal_prepared["inside_research_session"].to_numpy(dtype=bool)
        & (minute_of_day >= trigger_start)
        & (minute_of_day <= trigger_end)
        & weekday_allowed
    )

    stage3_idx = np.flatnonzero(stage3)
    matched_entry_indices: list[int] = []
    unmatched_signal_indices: list[int] = []
    for bar_idx in stage3_idx:
        row = signal_prepared.iloc[bar_idx]
        entry_idx = _find_entry_index(raw_prepared, raw_ns, row["timestamp_utc"], str(row["session_date_et"]))
        if entry_idx >= 0:
            matched_entry_indices.append(int(entry_idx))
        else:
            unmatched_signal_indices.append(int(bar_idx))

    trades = run_nt8_dual_feed_backtest(
        signal_df=signal_df,
        raw_df=raw_df,
        params_cfg=params_cfg,
        mff_cfg=mff_cfg,
        phase=phase,
        use_daily_regime_filter=True,
    )
    split = split_is_oos_metrics(trades, is_end_date=is_end_date, oos_start_date=oos_start_date)

    unmatched_signal_times = signal_prepared.iloc[unmatched_signal_indices]["timestamp_utc"] if unmatched_signal_indices else pd.Series([], dtype="datetime64[ns, UTC]")
    first_stage3_signal_times = signal_prepared.iloc[stage3_idx]["timestamp_utc"] if len(stage3_idx) > 0 else pd.Series([], dtype="datetime64[ns, UTC]")

    return {
        "instrument": instrument,
        "phase": phase,
        "paths": {
            "signal_csv": str(defaults["signal_csv"]),
            "raw_csv": str(defaults["raw_csv"]),
            "params_config": str(defaults["params_config"]),
            "mff_config": str(defaults["mff_config"]),
        },
        "funnel": {
            "rows_signal_feed": int(len(signal_prepared)),
            "rows_raw_feed": int(len(raw_prepared)),
            "stage1_raw_breakout": int(stage1.sum()),
            "stage1_raw_breakout_long": int(raw_long.sum()),
            "stage1_raw_breakout_short": int(raw_short.sum()),
            "stage2_after_sma_regime": int(stage2.sum()),
            "stage2_long": int(long_with_regime.sum()),
            "stage2_short": int(short_with_regime.sum()),
            "stage3_after_time_window": int(stage3.sum()),
            "stage4_execution_timestamp_match": int(len(matched_entry_indices)),
        },
        "warmup_diagnostics": {
            "sma_nan_rows": int(np.isnan(sma).sum()),
            "regime_zero_rows": int((regime == 0.0).sum()),
            "regime_positive_rows": int((regime > 0.0).sum()),
            "regime_negative_rows": int((regime < 0.0).sum()),
        },
        "timestamp_forensics": {
            "first_stage3_signal_timestamps_utc": _ts_list(first_stage3_signal_times),
            "first_raw_timestamps_utc": _ts_list(raw_prepared["timestamp_utc"]),
            "first_unmatched_stage3_signal_timestamps_utc": _ts_list(unmatched_signal_times),
            "unmatched_stage3_count": int(len(unmatched_signal_indices)),
        },
        "trade_output": {
            "total_trades": int(len(trades)),
            "is_trades": int(split["is"].trade_count),
            "oos_trades": int(split["oos"].trade_count),
        },
        "filters_after_stage4": {
            "dropped_after_execution_match": int(max(0, len(matched_entry_indices) - len(trades))),
            "reasons": [
                "max_trades_day",
                "daily_stop_or_daily_target",
                "contracts_lt_1_from_risk_sizing",
            ],
        },
    }


def main() -> None:
    args = parse_args()
    instruments = ["MNQ", "MGC"] if args.instrument == "BOTH" else [args.instrument]
    args.output.mkdir(parents=True, exist_ok=True)

    report = {
        "phase": args.phase,
        "is_end_date": args.is_end_date,
        "oos_start_date": args.oos_start_date,
        "results": {},
    }

    for instrument in instruments:
        report["results"][instrument] = _run_single_instrument(
            instrument=instrument,
            phase=args.phase,
            is_end_date=args.is_end_date,
            oos_start_date=args.oos_start_date,
        )

    output_path = args.output / "signal_census_report.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for instrument, result in report["results"].items():
        funnel = result["funnel"]
        trades = result["trade_output"]
        print(
            f"{instrument}: "
            f"raw={funnel['stage1_raw_breakout']} "
            f"regime={funnel['stage2_after_sma_regime']} "
            f"time={funnel['stage3_after_time_window']} "
            f"exec_match={funnel['stage4_execution_timestamp_match']} "
            f"trades={trades['total_trades']} "
            f"(IS={trades['is_trades']} OOS={trades['oos_trades']})"
        )
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    main()
