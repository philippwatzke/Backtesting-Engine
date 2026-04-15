#!/usr/bin/env python
"""Diagnose entry quality on NT8 dual-feed trades without changing exits."""

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
)
from propfirm.io.config import load_mff_config, load_params_config


DEFAULTS = {
    "MNQ": {
        "signal_csv": Path("data/ninjatrader/NT8_Dump_MNQ_5000_days.csv"),
        "raw_csv": Path("data/ninjatrader/NT8_RawDump_MNQ_5000_days.csv"),
        "params_config": Path("configs/default_params_mnq_legacy_frozen.toml"),
        "mff_config": Path("configs/mff_flex_50k_mnq.toml"),
        "trades_csv": Path("output/nt8_dual_feed_smoke/trades_mnq.csv"),
    },
    "MGC": {
        "signal_csv": Path("data/ninjatrader/NT8_Dump_MGC_5000_days.csv"),
        "raw_csv": Path("data/ninjatrader/NT8_RawDump_MGC_5000_days.csv"),
        "params_config": Path("configs/default_params.toml"),
        "mff_config": Path("configs/mff_flex_50k_mgc.toml"),
        "trades_csv": Path("output/nt8_dual_feed_smoke/trades_mgc.csv"),
    },
}

FOLLOW_THROUGH_BARS = (1, 3, 5, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", choices=["MNQ", "MGC", "BOTH"], default="BOTH")
    parser.add_argument("--is-end-date", default="2024-12-31")
    parser.add_argument("--output", type=Path, default=Path("output/nt8_entry_quality_autopsy"))
    return parser.parse_args()


def load_trades(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ["signal_time", "entry_time", "exit_time"]:
        df[col] = pd.to_datetime(df[col], utc=True)
    return df.sort_values(["entry_time", "exit_time"], kind="stable").reset_index(drop=True)


def _signed_ticks(delta_price: float, direction: int, tick_size: float) -> float:
    return (delta_price / tick_size) * direction


def _prepare_signal_frame(signal_df: pd.DataFrame, shared_cfg: dict, tick_size: float) -> tuple[pd.DataFrame, np.ndarray]:
    arrays, prepared = _build_signal_arrays(signal_df, shared_cfg)
    prepared = prepared.copy()
    prepared["signal_direction"] = np.where(
        (arrays["closes"] > arrays["donchian_high"]) & (arrays["closes"] > arrays["sma"]) & (arrays["regime"] > 0.0),
        1,
        np.where(
            (arrays["closes"] < arrays["donchian_low"]) & (arrays["closes"] < arrays["sma"]) & (arrays["regime"] < 0.0),
            -1,
            0,
        ),
    )

    trigger_start = int(shared_cfg["trigger_start_minute"])
    trigger_end = int(shared_cfg["trigger_end_minute"])
    prepared["stage3_candidate"] = (
        prepared["inside_research_session"]
        & (prepared["minute_of_day"] >= trigger_start)
        & (prepared["minute_of_day"] <= trigger_end)
        & (prepared["signal_direction"] != 0)
    )

    session_key = prepared["session_date_et"]
    prepared["session_high_so_far"] = prepared.groupby(session_key, sort=False)["high"].cummax()
    prepared["session_low_so_far"] = prepared.groupby(session_key, sort=False)["low"].cummin()
    session_range = prepared["session_high_so_far"] - prepared["session_low_so_far"]
    prepared["session_range_ticks"] = session_range / tick_size
    prepared["atr_ticks"] = prepared["atr_14_wilder"] / tick_size
    prepared["atr_pct"] = prepared["atr_14_wilder"] / prepared["close"]
    prepared["breakout_distance_ticks"] = np.where(
        prepared["signal_direction"] > 0,
        (prepared["close"] - prepared["donchian_high_5"]) / tick_size,
        np.where(
            prepared["signal_direction"] < 0,
            (prepared["donchian_low_5"] - prepared["close"]) / tick_size,
            np.nan,
        ),
    )
    prepared["sma_gap_ticks"] = np.where(
        prepared["signal_direction"] > 0,
        (prepared["close"] - prepared["sma_50"]) / tick_size,
        np.where(
            prepared["signal_direction"] < 0,
            (prepared["sma_50"] - prepared["close"]) / tick_size,
            np.nan,
        ),
    )
    prepared["distance_from_session_low_ticks"] = (prepared["close"] - prepared["session_low_so_far"]) / tick_size
    prepared["distance_from_session_high_ticks"] = (prepared["session_high_so_far"] - prepared["close"]) / tick_size
    prepared["range_position"] = np.where(
        session_range > 0.0,
        (prepared["close"] - prepared["session_low_so_far"]) / session_range,
        np.nan,
    )
    signal_ns = prepared["timestamp_utc"].array.asi8.astype(np.int64, copy=False)
    return prepared, signal_ns


def _follow_through_metrics(
    raw_df: pd.DataFrame,
    raw_ns: np.ndarray,
    entry_time: pd.Timestamp,
    entry_session_date: str,
    direction_label: str,
    entry_price: float,
    tick_size: float,
) -> dict[str, float]:
    direction = 1 if direction_label == "LONG" else -1
    entry_ns = entry_time.value
    entry_idx = int(np.searchsorted(raw_ns, entry_ns, side="left"))
    if entry_idx >= len(raw_df) or pd.Timestamp(raw_df.iloc[entry_idx]["timestamp_utc"]) != entry_time:
        raise ValueError(f"Entry timestamp not found in raw feed: {entry_time.isoformat()}")

    result: dict[str, float] = {}
    session_slice = raw_df.iloc[entry_idx:].copy()
    session_slice = session_slice.loc[session_slice["session_date_et"] == entry_session_date].reset_index(drop=True)

    for n_bars in FOLLOW_THROUGH_BARS:
        window = session_slice.iloc[:n_bars]
        if len(window) < n_bars:
            result[f"follow_close_{n_bars}_ticks"] = np.nan
            result[f"follow_mfe_{n_bars}_ticks"] = np.nan
            result[f"follow_mae_{n_bars}_ticks"] = np.nan
            result[f"follow_positive_{n_bars}"] = np.nan
            continue

        close_n = float(window.iloc[-1]["close"])
        if direction > 0:
            favorable = (window["high"].to_numpy(dtype=np.float64) - entry_price) / tick_size
            adverse = (entry_price - window["low"].to_numpy(dtype=np.float64)) / tick_size
        else:
            favorable = (entry_price - window["low"].to_numpy(dtype=np.float64)) / tick_size
            adverse = (window["high"].to_numpy(dtype=np.float64) - entry_price) / tick_size

        close_ticks = _signed_ticks(close_n - entry_price, direction, tick_size)
        result[f"follow_close_{n_bars}_ticks"] = float(close_ticks)
        result[f"follow_mfe_{n_bars}_ticks"] = float(np.max(favorable))
        result[f"follow_mae_{n_bars}_ticks"] = float(np.max(adverse))
        result[f"follow_positive_{n_bars}"] = float(close_ticks > 0.0)
    return result


def _summarize_subset(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"count": 0}

    summary = {"count": int(len(df))}
    metric_columns = [
        "atr_ticks",
        "atr_pct",
        "breakout_distance_ticks",
        "sma_gap_ticks",
        "distance_from_session_low_ticks",
        "distance_from_session_high_ticks",
        "session_range_ticks",
        "range_position",
    ]
    for col in metric_columns:
        summary[f"median_{col}"] = float(df[col].median())

    for n_bars in FOLLOW_THROUGH_BARS:
        summary[f"median_follow_close_{n_bars}_ticks"] = float(df[f"follow_close_{n_bars}_ticks"].median())
        summary[f"median_follow_mfe_{n_bars}_ticks"] = float(df[f"follow_mfe_{n_bars}_ticks"].median())
        summary[f"median_follow_mae_{n_bars}_ticks"] = float(df[f"follow_mae_{n_bars}_ticks"].median())
        summary[f"pct_positive_after_{n_bars}_bars"] = float(100.0 * df[f"follow_positive_{n_bars}"].mean())
    return summary


def run_single_instrument(instrument: str, is_end_date: str, output_dir: Path) -> dict:
    defaults = DEFAULTS[instrument]
    signal_df = load_nt8_signal_feed(defaults["signal_csv"])
    raw_df = load_nt8_raw_feed(defaults["raw_csv"])
    trades = load_trades(defaults["trades_csv"])
    params_cfg = load_params_config(defaults["params_config"])
    mff_cfg = load_mff_config(defaults["mff_config"])

    tick_size = float(mff_cfg["instrument"]["tick_size"])
    shared_cfg = params_cfg["strategy"]["mgc_h1_trend"]["shared"]
    signal_prepared, signal_ns = _prepare_signal_frame(signal_df, shared_cfg, tick_size)
    raw_prepared, raw_ns = _prepare_raw_execution_feed(raw_df, shared_cfg)

    is_trades = trades.loc[pd.to_datetime(trades["entry_session_date"]) <= pd.Timestamp(is_end_date)].copy()
    is_trades["is_winner"] = is_trades["net_pnl"] > 0.0

    analysis_rows: list[dict] = []
    for _, trade in is_trades.iterrows():
        signal_idx = int(np.searchsorted(signal_ns, trade["signal_time"].value, side="left"))
        if signal_idx >= len(signal_prepared) or pd.Timestamp(signal_prepared.iloc[signal_idx]["timestamp_utc"]) != trade["signal_time"]:
            raise ValueError(f"Signal timestamp not found in signal feed: {trade['signal_time'].isoformat()}")

        signal_row = signal_prepared.iloc[signal_idx]
        follow = _follow_through_metrics(
            raw_df=raw_prepared,
            raw_ns=raw_ns,
            entry_time=pd.Timestamp(trade["entry_time"]),
            entry_session_date=str(trade["entry_session_date"]),
            direction_label=str(trade["direction"]),
            entry_price=float(trade["entry_price"]),
            tick_size=tick_size,
        )
        analysis_row = trade.to_dict()
        analysis_row.update(
            {
                "atr_ticks": float(signal_row["atr_ticks"]),
                "atr_pct": float(signal_row["atr_pct"]),
                "breakout_distance_ticks": float(signal_row["breakout_distance_ticks"]),
                "sma_gap_ticks": float(signal_row["sma_gap_ticks"]),
                "distance_from_session_low_ticks": float(signal_row["distance_from_session_low_ticks"]),
                "distance_from_session_high_ticks": float(signal_row["distance_from_session_high_ticks"]),
                "session_range_ticks": float(signal_row["session_range_ticks"]),
                "range_position": float(signal_row["range_position"]),
            }
        )
        analysis_row.update(follow)
        analysis_rows.append(analysis_row)

    analysis_df = pd.DataFrame(analysis_rows)
    analysis_csv = output_dir / f"{instrument.lower()}_is_entry_quality.csv"
    analysis_df.to_csv(analysis_csv, index=False)

    is_sessions = signal_prepared.loc[
        pd.to_datetime(signal_prepared["session_date_et"]) <= pd.Timestamp(is_end_date), "session_date_et"
    ].drop_duplicates()
    signal_sessions = signal_prepared.loc[
        signal_prepared["stage3_candidate"] & (pd.to_datetime(signal_prepared["session_date_et"]) <= pd.Timestamp(is_end_date)),
        "session_date_et",
    ].drop_duplicates()
    trade_sessions = is_trades["entry_session_date"].drop_duplicates()

    winners = analysis_df.loc[analysis_df["is_winner"]].copy()
    losers = analysis_df.loc[~analysis_df["is_winner"]].copy()

    return {
        "instrument": instrument,
        "paths": {
            "signal_csv": str(defaults["signal_csv"]),
            "raw_csv": str(defaults["raw_csv"]),
            "trades_csv": str(defaults["trades_csv"]),
            "analysis_csv": str(analysis_csv),
        },
        "session_census": {
            "is_sessions_total": int(len(is_sessions)),
            "is_signal_sessions": int(len(signal_sessions)),
            "is_no_signal_sessions": int(len(is_sessions) - len(signal_sessions)),
            "is_trade_sessions": int(len(trade_sessions)),
        },
        "all_is_trades": _summarize_subset(analysis_df),
        "is_winners": _summarize_subset(winners),
        "is_losers": _summarize_subset(losers),
    }


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

    report_path = args.output / "entry_quality_autopsy_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for instrument, summary in report["results"].items():
        losers = summary["is_losers"]
        winners = summary["is_winners"]
        print(
            f"{instrument}: "
            f"IS sessions={summary['session_census']['is_sessions_total']} "
            f"signal_sessions={summary['session_census']['is_signal_sessions']} "
            f"trade_sessions={summary['session_census']['is_trade_sessions']} | "
            f"loser median close@3={losers.get('median_follow_close_3_ticks', 0.0):.2f} "
            f"winner median close@3={winners.get('median_follow_close_3_ticks', 0.0):.2f}"
        )
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
