#!/usr/bin/env python
"""Autopsy IS loser trades via MFE/MAE on the NT8 1m execution feed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from propfirm.execution.nt8_dual_feed import _prepare_raw_execution_feed, load_nt8_raw_feed


DEFAULTS = {
    "MNQ": {
        "trades_csv": Path("output/nt8_dual_feed_smoke/trades_mnq.csv"),
        "raw_csv": Path("data/ninjatrader/NT8_RawDump_MNQ_5000_days.csv"),
        "tick_size": 0.25,
    },
    "MGC": {
        "trades_csv": Path("output/nt8_dual_feed_smoke/trades_mgc.csv"),
        "raw_csv": Path("data/ninjatrader/NT8_RawDump_MGC_5000_days.csv"),
        "tick_size": 0.1,
    },
}

STOP_EXIT_REASONS = {"stop", "stop_gap", "stop_same_bar_priority"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", choices=["MNQ", "MGC", "BOTH"], default="BOTH")
    parser.add_argument("--is-end-date", default="2024-12-31")
    parser.add_argument("--output", type=Path, default=Path("output/nt8_loser_excursion_autopsy"))
    return parser.parse_args()


def load_trades(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ["signal_time", "entry_time", "exit_time"]:
        df[col] = pd.to_datetime(df[col], utc=True)
    return df.sort_values(["entry_time", "exit_time"], kind="stable").reset_index(drop=True)


def _update_excursions(
    direction: str,
    entry_price: float,
    bar_open: float,
    bar_high: float,
    bar_low: float,
    current_mfe: float,
    current_mae: float,
    include_extremes: bool,
) -> tuple[float, float]:
    if direction == "LONG":
        favorable_open = max(0.0, bar_open - entry_price)
        adverse_open = max(0.0, entry_price - bar_open)
        current_mfe = max(current_mfe, favorable_open)
        current_mae = max(current_mae, adverse_open)
        if include_extremes:
            current_mfe = max(current_mfe, max(0.0, bar_high - entry_price))
            current_mae = max(current_mae, max(0.0, entry_price - bar_low))
    else:
        favorable_open = max(0.0, entry_price - bar_open)
        adverse_open = max(0.0, bar_open - entry_price)
        current_mfe = max(current_mfe, favorable_open)
        current_mae = max(current_mae, adverse_open)
        if include_extremes:
            current_mfe = max(current_mfe, max(0.0, entry_price - bar_low))
            current_mae = max(current_mae, max(0.0, bar_high - entry_price))
    return current_mfe, current_mae


def analyze_trade_path(
    trade: pd.Series,
    raw_df: pd.DataFrame,
    raw_ns: np.ndarray,
    tick_size: float,
) -> dict:
    entry_time = pd.Timestamp(trade["entry_time"])
    exit_time = pd.Timestamp(trade["exit_time"])
    entry_ns = entry_time.value
    entry_idx = int(np.searchsorted(raw_ns, entry_ns, side="left"))
    if entry_idx >= len(raw_df) or raw_df.iloc[entry_idx]["timestamp_utc"] != entry_time:
        raise ValueError(f"Entry timestamp not found in raw feed: {entry_time.isoformat()}")

    direction = str(trade["direction"])
    exit_reason = str(trade["exit_reason"])
    entry_session_date = str(trade["entry_session_date"])
    entry_price = float(trade["entry_price"])

    mfe_price = 0.0
    mae_price = 0.0
    bars_held = 0

    for raw_idx in range(entry_idx, len(raw_df)):
        row = raw_df.iloc[raw_idx]
        if str(row["session_date_et"]) != entry_session_date:
            break

        row_time = pd.Timestamp(row["timestamp_utc"])
        bar_open = float(row["open"])
        bar_high = float(row["high"])
        bar_low = float(row["low"])

        if raw_idx > entry_idx and row_time == exit_time and exit_reason in {"stop_gap", "target_gap"}:
            mfe_price, mae_price = _update_excursions(
                direction=direction,
                entry_price=entry_price,
                bar_open=bar_open,
                bar_high=bar_open,
                bar_low=bar_open,
                current_mfe=mfe_price,
                current_mae=mae_price,
                include_extremes=False,
            )
            bars_held += 1
            break

        mfe_price, mae_price = _update_excursions(
            direction=direction,
            entry_price=entry_price,
            bar_open=bar_open,
            bar_high=bar_high,
            bar_low=bar_low,
            current_mfe=mfe_price,
            current_mae=mae_price,
            include_extremes=True,
        )
        bars_held += 1

        if row_time == exit_time:
            break

    mfe_ticks = mfe_price / tick_size
    mae_ticks = mae_price / tick_size
    half_target_ticks = 0.5 * float(trade["target_ticks"])
    category = "B_stop_hunted" if mfe_ticks > half_target_ticks else "A_dead_on_arrival"

    return {
        "mfe_ticks": float(mfe_ticks),
        "mae_ticks": float(mae_ticks),
        "half_target_ticks": float(half_target_ticks),
        "bars_held_raw": int(bars_held),
        "autopsy_category": category,
        "is_stop_exit": exit_reason in STOP_EXIT_REASONS,
    }


def summarize_counts(df: pd.DataFrame) -> dict:
    total = int(len(df))
    if total == 0:
        return {
            "count": 0,
            "category_a_count": 0,
            "category_b_count": 0,
            "category_a_pct": 0.0,
            "category_b_pct": 0.0,
        }
    category_a = int((df["autopsy_category"] == "A_dead_on_arrival").sum())
    category_b = int((df["autopsy_category"] == "B_stop_hunted").sum())
    return {
        "count": total,
        "category_a_count": category_a,
        "category_b_count": category_b,
        "category_a_pct": 100.0 * category_a / total,
        "category_b_pct": 100.0 * category_b / total,
    }


def run_single_instrument(instrument: str, is_end_date: str, output_dir: Path) -> dict:
    defaults = DEFAULTS[instrument]
    trades = load_trades(defaults["trades_csv"])
    raw_df = load_nt8_raw_feed(defaults["raw_csv"])
    raw_prepared, raw_ns = _prepare_raw_execution_feed(raw_df, {"session_start": "08:00", "session_end": "15:59"})

    losers = trades.loc[
        (pd.to_datetime(trades["entry_session_date"]) <= pd.Timestamp(is_end_date))
        & (pd.to_numeric(trades["net_pnl"], errors="coerce") < 0.0)
    ].copy()

    analyses: list[dict] = []
    for _, trade in losers.iterrows():
        analysis = analyze_trade_path(trade, raw_prepared, raw_ns, tick_size=float(defaults["tick_size"]))
        row = trade.to_dict()
        row.update(analysis)
        analyses.append(row)

    autopsy = pd.DataFrame(analyses)
    autopsy_path = output_dir / f"{instrument.lower()}_is_loser_autopsy.csv"
    autopsy.to_csv(autopsy_path, index=False)

    stop_losers = autopsy.loc[autopsy["is_stop_exit"]].copy()
    non_stop_losers = autopsy.loc[~autopsy["is_stop_exit"]].copy()

    summary = {
        "instrument": instrument,
        "paths": {
            "trades_csv": str(defaults["trades_csv"]),
            "raw_csv": str(defaults["raw_csv"]),
            "autopsy_csv": str(autopsy_path),
        },
        "is_loser_trades_all": summarize_counts(autopsy),
        "is_loser_trades_stop_only": summarize_counts(stop_losers),
        "excluded_non_stop_losers": {
            "count": int(len(non_stop_losers)),
            "exit_reason_breakdown": non_stop_losers["exit_reason"].value_counts().to_dict(),
        },
        "excursion_stats_all": {
            "median_mfe_ticks": float(autopsy["mfe_ticks"].median()) if not autopsy.empty else 0.0,
            "median_mae_ticks": float(autopsy["mae_ticks"].median()) if not autopsy.empty else 0.0,
        },
        "excursion_stats_stop_only": {
            "median_mfe_ticks": float(stop_losers["mfe_ticks"].median()) if not stop_losers.empty else 0.0,
            "median_mae_ticks": float(stop_losers["mae_ticks"].median()) if not stop_losers.empty else 0.0,
        },
    }
    return summary


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    instruments = ["MNQ", "MGC"] if args.instrument == "BOTH" else [args.instrument]

    report = {
        "is_end_date": args.is_end_date,
        "results": {},
    }
    for instrument in instruments:
        report["results"][instrument] = run_single_instrument(instrument, args.is_end_date, args.output)

    report_path = args.output / "loser_excursion_autopsy_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for instrument, summary in report["results"].items():
        all_losers = summary["is_loser_trades_all"]
        stop_losers = summary["is_loser_trades_stop_only"]
        print(
            f"{instrument}: "
            f"all losers A={all_losers['category_a_count']} ({all_losers['category_a_pct']:.2f}%) "
            f"B={all_losers['category_b_count']} ({all_losers['category_b_pct']:.2f}%) | "
            f"stop-only A={stop_losers['category_a_count']} ({stop_losers['category_a_pct']:.2f}%) "
            f"B={stop_losers['category_b_count']} ({stop_losers['category_b_pct']:.2f}%)"
        )
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
