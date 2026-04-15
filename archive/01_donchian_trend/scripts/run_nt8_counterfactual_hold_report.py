#!/usr/bin/env python
"""Counterfactual fixed-hold report for NT8 dual-feed entries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from propfirm.execution.nt8_dual_feed import load_nt8_signal_feed
from propfirm.io.config import load_mff_config


DEFAULTS = {
    "MNQ": {
        "signal_csv": Path("data/ninjatrader/NT8_Dump_MNQ_5000_days.csv"),
        "trades_csv": Path("output/nt8_dual_feed_smoke/trades_mnq.csv"),
        "mff_config": Path("configs/mff_flex_50k_mnq.toml"),
    },
    "MGC": {
        "signal_csv": Path("data/ninjatrader/NT8_Dump_MGC_5000_days.csv"),
        "trades_csv": Path("output/nt8_dual_feed_smoke/trades_mgc.csv"),
        "mff_config": Path("configs/mff_flex_50k_mgc.toml"),
    },
}

HOLD_BARS = (1, 3, 5, 10)
INVERT_HOLD_BARS = (1, 3, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", choices=["MNQ", "MGC", "BOTH"], default="BOTH")
    parser.add_argument("--is-end-date", default="2024-12-31")
    parser.add_argument("--output", type=Path, default=Path("output/nt8_counterfactual_hold"))
    return parser.parse_args()


def load_trades(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ["signal_time", "entry_time", "exit_time"]:
        df[col] = pd.to_datetime(df[col], utc=True)
    return df.sort_values(["entry_time", "exit_time"], kind="stable").reset_index(drop=True)


def _signed_ticks(direction: str, entry_price: float, exit_price: float, tick_size: float) -> float:
    if direction == "LONG":
        return (exit_price - entry_price) / tick_size
    return (entry_price - exit_price) / tick_size


def _summarize_ev(df: pd.DataFrame, hold_bars: int, tick_value: float) -> dict:
    col_ticks = f"hold_{hold_bars}_ticks"
    col_usd = f"hold_{hold_bars}_usd_1lot"
    used = df[col_ticks].notna()
    if not bool(used.any()):
        return {"count": 0, "ev_ticks": 0.0, "ev_usd_1lot": 0.0, "win_rate": 0.0, "median_ticks": 0.0}

    subset = df.loc[used]
    return {
        "count": int(len(subset)),
        "ev_ticks": float(subset[col_ticks].mean()),
        "ev_usd_1lot": float(subset[col_usd].mean()),
        "win_rate": float(100.0 * (subset[col_ticks] > 0.0).mean()),
        "median_ticks": float(subset[col_ticks].median()),
    }


def _summarize_inverted_ev(df: pd.DataFrame, hold_bars: int, tick_value: float) -> dict:
    col_ticks = f"invert_hold_{hold_bars}_ticks"
    col_usd = f"invert_hold_{hold_bars}_usd_1lot"
    used = df[col_ticks].notna()
    if not bool(used.any()):
        return {"count": 0, "ev_ticks": 0.0, "ev_usd_1lot": 0.0, "win_rate": 0.0, "median_ticks": 0.0}

    subset = df.loc[used]
    return {
        "count": int(len(subset)),
        "ev_ticks": float(subset[col_ticks].mean()),
        "ev_usd_1lot": float(subset[col_usd].mean()),
        "win_rate": float(100.0 * (subset[col_ticks] > 0.0).mean()),
        "median_ticks": float(subset[col_ticks].median()),
    }


def run_single_instrument(instrument: str, is_end_date: str, output_dir: Path) -> dict:
    defaults = DEFAULTS[instrument]
    signal_df = load_nt8_signal_feed(defaults["signal_csv"])
    trades = load_trades(defaults["trades_csv"])
    mff_cfg = load_mff_config(defaults["mff_config"])
    tick_size = float(mff_cfg["instrument"]["tick_size"])
    tick_value = float(mff_cfg["instrument"]["tick_value"])

    is_trades = trades.loc[pd.to_datetime(trades["entry_session_date"]) <= pd.Timestamp(is_end_date)].copy()
    signal_index = signal_df.index[signal_df["timestamp_utc"].isin(is_trades["signal_time"])].tolist()
    signal_lookup = dict(zip(signal_df.loc[signal_index, "timestamp_utc"], signal_index, strict=False))

    analysis_rows: list[dict] = []
    for _, trade in is_trades.iterrows():
        signal_time = pd.Timestamp(trade["signal_time"])
        signal_idx = signal_lookup.get(signal_time)
        if signal_idx is None:
            raise ValueError(f"Signal timestamp not found in signal feed: {signal_time.isoformat()}")

        row = trade.to_dict()
        entry_price = float(trade["entry_price"])
        direction = str(trade["direction"])

        for hold_bars in HOLD_BARS:
            exit_idx = signal_idx + hold_bars
            if exit_idx >= len(signal_df):
                row[f"hold_{hold_bars}_exit_time"] = pd.NaT
                row[f"hold_{hold_bars}_exit_close"] = pd.NA
                row[f"hold_{hold_bars}_ticks"] = pd.NA
                row[f"hold_{hold_bars}_usd_1lot"] = pd.NA
            else:
                exit_row = signal_df.iloc[exit_idx]
                exit_time = pd.Timestamp(exit_row["timestamp_utc"])
                exit_close = float(exit_row["close"])
                pnl_ticks = _signed_ticks(direction, entry_price, exit_close, tick_size)
                row[f"hold_{hold_bars}_exit_time"] = exit_time
                row[f"hold_{hold_bars}_exit_close"] = exit_close
                row[f"hold_{hold_bars}_ticks"] = pnl_ticks
                row[f"hold_{hold_bars}_usd_1lot"] = pnl_ticks * tick_value

            if hold_bars in INVERT_HOLD_BARS:
                if exit_idx >= len(signal_df):
                    row[f"invert_hold_{hold_bars}_ticks"] = pd.NA
                    row[f"invert_hold_{hold_bars}_usd_1lot"] = pd.NA
                else:
                    row[f"invert_hold_{hold_bars}_ticks"] = -row[f"hold_{hold_bars}_ticks"]
                    row[f"invert_hold_{hold_bars}_usd_1lot"] = -row[f"hold_{hold_bars}_usd_1lot"]

        analysis_rows.append(row)

    analysis_df = pd.DataFrame(analysis_rows)
    analysis_csv = output_dir / f"{instrument.lower()}_counterfactual_holds.csv"
    analysis_df.to_csv(analysis_csv, index=False)

    normal_ev = {f"hold_{n}": _summarize_ev(analysis_df, n, tick_value) for n in HOLD_BARS}
    inverted_ev = {f"hold_{n}": _summarize_inverted_ev(analysis_df, n, tick_value) for n in INVERT_HOLD_BARS}

    return {
        "instrument": instrument,
        "paths": {
            "signal_csv": str(defaults["signal_csv"]),
            "trades_csv": str(defaults["trades_csv"]),
            "analysis_csv": str(analysis_csv),
        },
        "is_trade_count": int(len(analysis_df)),
        "assumption": "Entry uses the actual 1m dual-feed entry price; counterfactual exit is the close of the n-th HTF bar after the signal bar.",
        "normal_hold_ev": normal_ev,
        "inverted_hold_ev": inverted_ev,
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    instruments = ["MNQ", "MGC"] if args.instrument == "BOTH" else [args.instrument]

    report = {"is_end_date": args.is_end_date, "results": {}}
    for instrument in instruments:
        report["results"][instrument] = run_single_instrument(instrument, args.is_end_date, args.output)

    report_path = args.output / "counterfactual_hold_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for instrument, summary in report["results"].items():
        h1 = summary["normal_hold_ev"]["hold_1"]
        h3 = summary["normal_hold_ev"]["hold_3"]
        inv3 = summary["inverted_hold_ev"]["hold_3"]
        print(
            f"{instrument}: "
            f"hold1 EV={h1['ev_ticks']:.2f} ticks, "
            f"hold3 EV={h3['ev_ticks']:.2f} ticks, "
            f"invert3 EV={inv3['ev_ticks']:.2f} ticks"
        )
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
