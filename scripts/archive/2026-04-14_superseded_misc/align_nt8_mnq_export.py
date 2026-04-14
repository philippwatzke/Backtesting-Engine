#!/usr/bin/env python
"""Import an NT8 MNQ text export, align it, and run the Python Donchian backtest on that exact bar stream."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from propfirm.core.engine import run_day_kernel_portfolio
from propfirm.core.types import (
    DAILY_LOG_DTYPE,
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


DEFAULT_INPUT = Path("data/raw/MNQ 06-26.Last.txt")
DEFAULT_OUTPUT_PARQUET = Path("data/processed/MNQ_06_26_nt8.parquet")
DEFAULT_OUTPUT_DIR = Path("output/nt8_mnq_alignment")
DEFAULT_REFERENCE = Path("data/processed/MNQ_1m_adjusted_NT8.parquet")
DEFAULT_MFF_CONFIG = Path("configs/mff_flex_50k_mnq.toml")
DEFAULT_PARAMS_CONFIG = Path("configs/default_params_mnq.toml")
TIMESTAMP_FORMAT = "%Y%m%d %H%M%S"
SESSION_TZ = "America/New_York"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-parquet", type=Path, default=DEFAULT_OUTPUT_PARQUET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-tz", type=str, default=SESSION_TZ)
    parser.add_argument("--timeframe-minutes", type=int, default=30)
    parser.add_argument("--mff-config", type=Path, default=DEFAULT_MFF_CONFIG)
    parser.add_argument("--params-config", type=Path, default=DEFAULT_PARAMS_CONFIG)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--warmup-calendar-days", type=int, default=120)
    parser.add_argument("--extra-slippage-ticks", type=float, default=0.0)
    parser.add_argument("--skip-reference-compare", action="store_true")
    return parser.parse_args()


def load_nt8_txt(filepath: Path, source_tz: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep=";", header=None, names=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], format=TIMESTAMP_FORMAT, errors="raise")
    df["timestamp"] = df["timestamp"].dt.tz_localize(source_tz)
    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="raise")
    df = df.sort_values("timestamp", kind="stable").drop_duplicates(subset=["timestamp"], keep="first")
    df = df.set_index("timestamp")
    return df


def compare_with_reference(nt_df: pd.DataFrame, reference_path: Path) -> dict | None:
    if not reference_path.exists():
        return None

    ref = pd.read_parquet(reference_path).sort_index()
    if ref.index.tz is None:
        raise ValueError(f"Reference parquet must be timezone-aware: {reference_path}")
    ref = ref.tz_convert(SESSION_TZ)

    results: list[dict] = []
    nt_et = nt_df.tz_convert(SESSION_TZ)
    for hour_shift in range(-8, 9):
        shifted = nt_et.copy()
        shifted.index = shifted.index + pd.Timedelta(hours=hour_shift)
        common = shifted.index.intersection(ref.index)
        if len(common) < 100:
            continue
        diff = (shifted.loc[common, "close"] - ref.loc[common, "close"]).abs()
        results.append(
            {
                "hour_shift": hour_shift,
                "common_rows": int(len(common)),
                "mean_abs_close_diff": float(diff.mean()),
                "median_abs_close_diff": float(diff.median()),
                "max_abs_close_diff": float(diff.max()),
            }
        )

    if not results:
        return None
    return min(results, key=lambda item: item["mean_abs_close_diff"])


def build_combined_frame(nt_df: pd.DataFrame, reference_path: Path, warmup_calendar_days: int) -> pd.DataFrame:
    if not reference_path.exists():
        return nt_df

    ref = pd.read_parquet(reference_path).sort_index()
    if ref.index.tz is None:
        raise ValueError(f"Reference parquet must be timezone-aware: {reference_path}")
    ref = ref.tz_convert(SESSION_TZ)
    nt_et = nt_df.tz_convert(SESSION_TZ)

    warmup_start = nt_et.index.min() - pd.Timedelta(days=warmup_calendar_days)
    ref_warmup = ref.loc[(ref.index >= warmup_start) & (ref.index < nt_et.index.min()), ["open", "high", "low", "close", "volume"]]
    combined = pd.concat([ref_warmup, nt_et[["open", "high", "low", "close", "volume"]]], axis=0)
    combined = combined.loc[~combined.index.duplicated(keep="last")].sort_index(kind="stable")
    return combined


def slice_loaded_data(data: dict, export_session_dates: set[str]) -> dict:
    selected_boundaries: list[tuple[int, int]] = []
    selected_session_dates: list[str] = []
    for session_date, (start, end) in zip(data["session_dates"], data["day_boundaries"]):
        if session_date in export_session_dates:
            selected_boundaries.append((start, end))
            selected_session_dates.append(session_date)

    if not selected_boundaries:
        raise ValueError("No overlapping session dates found after loading combined parquet")

    start_idx = selected_boundaries[0][0]
    end_idx = selected_boundaries[-1][1]

    sliced = {}
    array_keys = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "timestamps",
        "minute_of_day",
        "day_of_week",
        "bar_atr",
        "trailing_median_atr",
        "daily_atr_ratio",
        "rvol",
        "close_sma_50",
        "daily_regime_bias",
        "donchian_high_5",
        "donchian_low_5",
    ]
    for key in array_keys:
        sliced[key] = data[key][start_idx:end_idx]

    sliced["day_boundaries"] = [(start - start_idx, end - start_idx) for start, end in selected_boundaries]
    sliced["session_dates"] = selected_session_dates
    sliced["session_minutes"] = data["session_minutes"]
    sliced["timeframe_minutes"] = data["timeframe_minutes"]
    sliced["bars_per_session"] = data["bars_per_session"]
    return sliced


def build_strategy_profiles(shared_cfg: dict, eval_cfg: dict, funded_cfg: dict, risk_buffer_fraction: float) -> tuple[np.ndarray, np.ndarray]:
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

    return strategy_profiles_eval, strategy_profiles_funded


def compute_trade_metrics(trade_log: np.ndarray) -> dict:
    if len(trade_log) == 0:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "net_profit": 0.0,
            "average_trade_pnl": 0.0,
            "overnight_exits": 0,
        }

    net_pnl = trade_log["net_pnl"].astype(np.float64)
    gross_profit = float(net_pnl[net_pnl > 0.0].sum())
    gross_loss = float(-net_pnl[net_pnl < 0.0].sum())
    entry_ts = pd.to_datetime(trade_log["entry_time"].astype(np.int64), utc=True).tz_convert(SESSION_TZ)
    exit_ts = pd.to_datetime(trade_log["exit_time"].astype(np.int64), utc=True).tz_convert(SESSION_TZ)
    return {
        "total_trades": int(len(trade_log)),
        "win_rate": float(np.mean(net_pnl > 0.0)),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0.0 else float("inf"),
        "net_profit": float(net_pnl.sum()),
        "average_trade_pnl": float(np.mean(net_pnl)),
        "overnight_exits": int(np.sum(exit_ts.date > entry_ts.date)),
    }


def run_python_backtest(data: dict, mff_cfg: dict, params_cfg: dict, extra_slippage_ticks: float) -> tuple[np.ndarray, np.ndarray, dict]:
    shared_cfg = params_cfg["strategy"]["mgc_h1_trend"]["shared"]
    eval_cfg = params_cfg["strategy"]["mgc_h1_trend"]["eval"]
    funded_cfg = params_cfg["strategy"]["mgc_h1_trend"]["funded"]
    portfolio_shared = params_cfg.get("portfolio", {}).get("shared", {})
    slip_cfg = params_cfg["slippage"]
    risk_buffer_fraction = float(portfolio_shared.get("risk_buffer_fraction", 0.25))

    slippage_lookup = build_slippage_lookup(None, require_file=False, session_minutes=int(data["session_minutes"]))

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
    params_eval[PARAMS_EXTRA_SLIPPAGE_TICKS] = float(extra_slippage_ticks)
    params_funded[PARAMS_EXTRA_SLIPPAGE_TICKS] = float(extra_slippage_ticks)

    strategy_profiles_eval, strategy_profiles_funded = build_strategy_profiles(
        shared_cfg, eval_cfg, funded_cfg, risk_buffer_fraction
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
            data["daily_regime_bias"][start:end],
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
            state.transition_to_funded()
            funded_payout_cycle_id = 0
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

    trade_log = all_trades[:total_trade_count]
    daily_log = daily_log[:total_day_count]
    metrics = compute_trade_metrics(trade_log)
    return trade_log, daily_log, metrics


def main() -> None:
    args = parse_args()
    nt_df = load_nt8_txt(args.input, args.source_tz)
    nt_df = nt_df.tz_convert(SESSION_TZ)

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    nt_df.to_parquet(args.output_parquet)

    compare_result = None
    if not args.skip_reference_compare:
        compare_result = compare_with_reference(nt_df, args.reference)

    combined_df = build_combined_frame(nt_df, args.reference, args.warmup_calendar_days)
    combined_path = args.output_dir / "mnq_nt8_with_warmup.parquet"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined_df.to_parquet(combined_path)

    mff_cfg = load_mff_config(args.mff_config)
    params_cfg = load_params_config(args.params_config)
    shared_cfg = params_cfg["strategy"]["mgc_h1_trend"]["shared"]
    slip_cfg = params_cfg["slippage"]

    loaded = load_session_data(
        combined_path,
        atr_period=slip_cfg["atr_period"],
        trailing_atr_days=slip_cfg["trailing_atr_days"],
        timeframe_minutes=args.timeframe_minutes,
        session_start=shared_cfg["session_start"],
        session_end=shared_cfg["session_end"],
    )

    export_session_dates = {str(ts.date()) for ts in nt_df.index}
    sliced = slice_loaded_data(loaded, export_session_dates)
    trade_log, daily_log, metrics = run_python_backtest(sliced, mff_cfg, params_cfg, args.extra_slippage_ticks)

    trade_log_path = args.output_dir / "mnq_nt8_trade_log.npy"
    daily_log_path = args.output_dir / "mnq_nt8_daily_log.npy"
    pnl_path = args.output_dir / "mnq_nt8_trade_pnls.npy"
    report_path = args.output_dir / "mnq_nt8_alignment_report.json"

    np.save(trade_log_path, trade_log)
    np.save(daily_log_path, daily_log)
    np.save(pnl_path, trade_log["net_pnl"].astype(np.float64))

    report = {
        "input_txt": str(args.input),
        "output_parquet": str(args.output_parquet),
        "combined_parquet": str(combined_path),
        "source_tz": args.source_tz,
        "timeframe_minutes": args.timeframe_minutes,
        "nt_export": {
            "rows": int(len(nt_df)),
            "start": str(nt_df.index.min()),
            "end": str(nt_df.index.max()),
            "session_dates": sorted(export_session_dates),
        },
        "reference_compare": compare_result,
        "python_backtest_metrics": metrics,
        "artifacts": {
            "trade_log": str(trade_log_path),
            "daily_log": str(daily_log_path),
            "trade_pnls": str(pnl_path),
        },
    }
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(f"NT8 input: {args.input}")
    print(f"Saved aligned parquet: {args.output_parquet}")
    print(f"Saved warmup parquet: {combined_path}")
    print(f"NT rows: {len(nt_df)}")
    print(f"Date range: {nt_df.index.min()} to {nt_df.index.max()}")
    if compare_result is not None:
        print(
            "Best reference shift diagnostic:"
            f" shift={compare_result['hour_shift']}h"
            f", common_rows={compare_result['common_rows']}"
            f", mean_abs_close_diff={compare_result['mean_abs_close_diff']:.2f}"
            f", median_abs_close_diff={compare_result['median_abs_close_diff']:.2f}"
        )
    print(f"Python trades on NT8 bars: {metrics['total_trades']}")
    print(f"Win rate: {metrics['win_rate']:.2%}")
    print(f"Profit factor: {metrics['profit_factor']:.2f}")
    print(f"Net profit: ${metrics['net_profit']:.2f}")
    print(f"Average trade PnL: ${metrics['average_trade_pnl']:.2f}")
    print(f"Overnight exits: {metrics['overnight_exits']}")
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
