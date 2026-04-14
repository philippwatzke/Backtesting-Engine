#!/usr/bin/env python
"""Full-range MNQ Donchian backtest on adjusted NT8 data + reconciliation."""
from __future__ import annotations

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

# ---------------------------------------------------------------------------
DATA_PATH = Path("data/processed/MNQ_1m_adjusted_NT8.parquet")
MFF_CONFIG = Path("configs/mff_flex_50k_mnq.toml")
PARAMS_CONFIG = Path("configs/default_params_mnq.toml")
TRADE_LOG_OUT = Path("output/nt8_mnq_alignment/mnq_full_trade_log.npy")
SESSION_TZ = "America/New_York"
TIMEFRAME = 30
# ---------------------------------------------------------------------------


def run_full_backtest() -> np.ndarray:
    mff_cfg = load_mff_config(MFF_CONFIG)
    params_cfg = load_params_config(PARAMS_CONFIG)
    shared_cfg = params_cfg["strategy"]["mgc_h1_trend"]["shared"]
    eval_cfg = params_cfg["strategy"]["mgc_h1_trend"]["eval"]
    funded_cfg = params_cfg["strategy"]["mgc_h1_trend"]["funded"]
    slip_cfg = params_cfg["slippage"]
    portfolio_shared = params_cfg.get("portfolio", {}).get("shared", {})
    risk_buffer_fraction = float(portfolio_shared.get("risk_buffer_fraction", 0.25))

    print(f"Loading data from {DATA_PATH} ...")
    data = load_session_data(
        DATA_PATH,
        atr_period=slip_cfg["atr_period"],
        trailing_atr_days=slip_cfg["trailing_atr_days"],
        timeframe_minutes=TIMEFRAME,
        session_start=shared_cfg["session_start"],
        session_end=shared_cfg["session_end"],
    )
    n_days = len(data["day_boundaries"])
    print(f"Sessions loaded: {n_days}")
    print(f"Date range: {data['session_dates'][0]} to {data['session_dates'][-1]}")

    slippage_lookup = build_slippage_lookup(
        None, require_file=False, session_minutes=int(data["session_minutes"])
    )

    params_eval = build_phase_params(
        shared_cfg, eval_cfg, slip_cfg,
        mff_cfg["instrument"]["commission_per_side"],
        strategy_name="mgc_h1_trend",
        instrument_cfg=mff_cfg["instrument"],
    )
    params_funded = build_phase_params(
        shared_cfg, funded_cfg, slip_cfg,
        mff_cfg["instrument"]["commission_per_side"],
        strategy_name="mgc_h1_trend",
        instrument_cfg=mff_cfg["instrument"],
    )
    params_eval[PARAMS_EXTRA_SLIPPAGE_TICKS] = 0.0
    params_funded[PARAMS_EXTRA_SLIPPAGE_TICKS] = 0.0

    # Strategy profiles
    sp_eval = np.zeros((1, PROFILE_ARRAY_LENGTH), dtype=np.float64)
    sp_eval[0, PROFILE_RISK_PER_TRADE_USD] = float(eval_cfg["risk_per_trade_usd"])
    sp_eval[0, PROFILE_STOP_ATR_MULTIPLIER] = float(eval_cfg["stop_atr_multiplier"])
    sp_eval[0, PROFILE_TARGET_ATR_MULTIPLIER] = float(eval_cfg["target_atr_multiplier"])
    sp_eval[0, PROFILE_BREAKEVEN_TRIGGER_TICKS] = float(shared_cfg["breakeven_trigger_ticks"])
    sp_eval[0, PROFILE_RISK_BUFFER_FRACTION] = risk_buffer_fraction

    sp_funded = np.zeros((1, PROFILE_ARRAY_LENGTH), dtype=np.float64)
    sp_funded[0, PROFILE_RISK_PER_TRADE_USD] = float(funded_cfg["risk_per_trade_usd"])
    sp_funded[0, PROFILE_STOP_ATR_MULTIPLIER] = float(funded_cfg["stop_atr_multiplier"])
    sp_funded[0, PROFILE_TARGET_ATR_MULTIPLIER] = float(funded_cfg["target_atr_multiplier"])
    sp_funded[0, PROFILE_BREAKEVEN_TRIGGER_TICKS] = float(shared_cfg["breakeven_trigger_ticks"])
    sp_funded[0, PROFILE_RISK_BUFFER_FRACTION] = risk_buffer_fraction

    max_trades_per_day = int(max(params_eval[PARAMS_MAX_TRADES], params_funded[PARAMS_MAX_TRADES]))
    max_possible = max(1, n_days * max_trades_per_day)

    # Raw backtest: bypass MFF rules to run every session unconditionally
    all_trades = np.zeros(max_possible, dtype=TRADE_LOG_DTYPE)
    total_trades = 0

    for day_idx, (start, end) in enumerate(data["day_boundaries"]):
        active_params = params_eval.copy()
        strategy_profiles = sp_eval.copy()
        active_params[PARAMS_CONTRACTS] = 1.0

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
            0,   # phase_id = eval
            -1,  # payout_cycle_id
            0.0, # no liquidation floor
            combined_portfolio_signal,
            all_trades[total_trades:],
            0,
            100_000.0,  # large equity so daily limits don't bite
            0.0,
            active_params,
            strategy_profiles,
        )
        total_trades += n_trades

    trade_log = all_trades[:total_trades]
    TRADE_LOG_OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(TRADE_LOG_OUT, trade_log)
    return trade_log


def print_metrics(trade_log: np.ndarray) -> None:
    n = len(trade_log)
    if n == 0:
        print("\n  NO TRADES.")
        return
    net = trade_log["net_pnl"].astype(np.float64)
    gross = trade_log["gross_pnl"].astype(np.float64)
    wins = gross[gross > 0]
    losses = gross[gross < 0]
    pf = float(wins.sum() / -losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float("inf")
    wr = float(np.mean(gross > 0))
    entry_ts = pd.to_datetime(trade_log["entry_time"].astype(np.int64), utc=True).tz_convert(SESSION_TZ)
    exit_ts = pd.to_datetime(trade_log["exit_time"].astype(np.int64), utc=True).tz_convert(SESSION_TZ)
    longs = trade_log[trade_log["signal_type"] > 0]
    shorts = trade_log[trade_log["signal_type"] < 0]

    print(f"\n{'='*60}")
    print("PYTHON BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"  Total trades:     {n}")
    print(f"  Longs / Shorts:   {len(longs)} / {len(shorts)}")
    print(f"  Win rate:         {wr:.1%}")
    print(f"  Profit factor:    {pf:.2f}")
    print(f"  Gross PnL:        ${gross.sum():,.2f}")
    print(f"  Net PnL:          ${net.sum():,.2f}")
    print(f"  Avg trade (net):  ${net.mean():,.2f}")
    print(f"  Date range:       {entry_ts.min().date()} to {entry_ts.max().date()}")
    print(f"  First 3 entries:")
    for i in range(min(3, n)):
        sig = "LONG" if trade_log[i]["signal_type"] > 0 else "SHORT"
        print(f"    {entry_ts[i]}  {sig}  @ {trade_log[i]['entry_price']:.2f}")


def main() -> None:
    print("=" * 60)
    print("STEP 1: Full-range MNQ Donchian backtest")
    print("=" * 60)
    trade_log = run_full_backtest()
    print_metrics(trade_log)

    print(f"\n{'='*60}")
    print("STEP 2: Reconciliation vs NT8")
    print(f"{'='*60}")
    # Import and run reconciliation
    from scripts.reconcile_trades import parse_nt8, parse_python_npy, match_by_date
    nt8_path = Path("data/raw/ninjatrader_backtest_log/NT8_MNQ_Executions.csv")
    nt8_trades = parse_nt8(nt8_path)
    print(f"\nNT8: {len(nt8_trades)} round-trips")
    py_trades = parse_python_npy([TRADE_LOG_OUT])
    print(f"Python: {len(py_trades)} round-trips")
    match_by_date(nt8_trades, py_trades)


if __name__ == "__main__":
    main()
