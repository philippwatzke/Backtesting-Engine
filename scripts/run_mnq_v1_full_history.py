#!/usr/bin/env python
"""Run the verified MNQ v1.0 breakout over the full clean Databento history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from propfirm.core.engine import run_day_kernel_portfolio
from propfirm.core.types import (
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
from propfirm.io.config import build_phase_params, load_params_config
from propfirm.market.data_loader import load_session_data
from propfirm.market.slippage import build_slippage_lookup
from propfirm.strategy.portfolio import combined_portfolio_signal


TRAIN_PATH = Path("data/processed/MNQ_1m_full_train.parquet")
VAL_PATH = Path("data/processed/MNQ_1m_full_val.parquet")
TEST_PATH = Path("data/processed/MNQ_1m_full_test.parquet")
DEFAULT_PARAMS = Path("configs/default_params_mnq.toml")
DEFAULT_OUTPUT = Path("output/mnq_v1_full_history")
SESSION_TZ = "America/New_York"
TIMEFRAME_MINUTES = 30
TICK_SIZE = 0.25
TICK_VALUE = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params-config", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--combined-data", type=Path, default=DEFAULT_OUTPUT / "mnq_full_history_v1.parquet")
    return parser.parse_args()


def build_combined_history(output_path: Path) -> pd.DataFrame:
    frames = []
    for path in (TRAIN_PATH, VAL_PATH, TEST_PATH):
        frame = pd.read_parquet(path).sort_index(kind="stable")
        frame.index = frame.index.tz_convert(SESSION_TZ)
        frames.append(frame)

    combined = pd.concat(frames, axis=0)
    combined = combined.loc[~combined.index.duplicated(keep="first")].sort_index(kind="stable")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path)
    return combined


def compute_metrics(trade_log: np.ndarray) -> dict[str, float | int | str]:
    if len(trade_log) == 0:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "net_profit": 0.0,
            "avg_trade_net": 0.0,
            "max_drawdown": 0.0,
        }

    net = trade_log["net_pnl"].astype(np.float64)
    gross = trade_log["gross_pnl"].astype(np.float64)
    wins = int(np.sum(net > 0.0))
    losses = int(np.sum(net < 0.0))
    gross_profit = float(gross[gross > 0.0].sum())
    gross_loss = float(-gross[gross < 0.0].sum())
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0.0 else float("inf")

    equity_curve = np.cumsum(net)
    running_peak = np.maximum.accumulate(np.concatenate(([0.0], equity_curve)))
    drawdowns = running_peak[1:] - equity_curve
    max_drawdown = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0

    return {
        "total_trades": int(len(trade_log)),
        "wins": wins,
        "losses": losses,
        "win_rate": float(np.mean(net > 0.0)),
        "profit_factor": profit_factor,
        "net_profit": float(net.sum()),
        "avg_trade_net": float(net.mean()),
        "max_drawdown": max_drawdown,
    }


def main() -> None:
    args = parse_args()
    params_cfg = load_params_config(args.params_config)
    shared_cfg = params_cfg["strategy"]["mgc_h1_trend"]["shared"]
    eval_cfg = params_cfg["strategy"]["mgc_h1_trend"]["eval"]
    slip_cfg = params_cfg["slippage"]
    portfolio_shared = params_cfg.get("portfolio", {}).get("shared", {})
    risk_buffer_fraction = float(portfolio_shared.get("risk_buffer_fraction", 0.25))

    combined = build_combined_history(args.combined_data)
    print("=" * 72)
    print("MNQ V1.0 FULL HISTORY RUN")
    print("=" * 72)
    print(f"Combined data: {args.combined_data}")
    print(f"Rows:          {len(combined):,}")
    print(f"Date range:    {combined.index.min()} to {combined.index.max()}")

    data = load_session_data(
        args.combined_data,
        atr_period=slip_cfg["atr_period"],
        trailing_atr_days=slip_cfg["trailing_atr_days"],
        timeframe_minutes=TIMEFRAME_MINUTES,
        session_start=shared_cfg["session_start"],
        session_end=shared_cfg["session_end"],
    )

    params_eval = build_phase_params(
        shared_cfg,
        eval_cfg,
        slip_cfg,
        commission_per_side=0.0,
        strategy_name="mgc_h1_trend",
        instrument_cfg={"tick_size": TICK_SIZE, "tick_value": TICK_VALUE},
    )
    params_eval[PARAMS_EXTRA_SLIPPAGE_TICKS] = 0.0
    params_eval[PARAMS_CONTRACTS] = 1.0

    strategy_profiles = np.zeros((1, PROFILE_ARRAY_LENGTH), dtype=np.float64)
    strategy_profiles[0, PROFILE_RISK_PER_TRADE_USD] = float(eval_cfg["risk_per_trade_usd"])
    strategy_profiles[0, PROFILE_STOP_ATR_MULTIPLIER] = float(eval_cfg["stop_atr_multiplier"])
    strategy_profiles[0, PROFILE_TARGET_ATR_MULTIPLIER] = float(eval_cfg["target_atr_multiplier"])
    strategy_profiles[0, PROFILE_BREAKEVEN_TRIGGER_TICKS] = float(shared_cfg["breakeven_trigger_ticks"])
    strategy_profiles[0, PROFILE_RISK_BUFFER_FRACTION] = risk_buffer_fraction

    slippage_lookup = build_slippage_lookup(None, require_file=False, session_minutes=int(data["session_minutes"]))
    daily_regime_bias = np.full(len(data["close"]), np.nan, dtype=np.float64)

    max_trades_per_day = int(params_eval[PARAMS_MAX_TRADES])
    all_trades = np.zeros(max(1, len(data["day_boundaries"]) * max_trades_per_day), dtype=TRADE_LOG_DTYPE)
    total_trades = 0

    for day_idx, (start, end) in enumerate(data["day_boundaries"]):
        n_trades, _, _ = run_day_kernel_portfolio(
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
            0,
            -1,
            0.0,
            combined_portfolio_signal,
            all_trades[total_trades:],
            0,
            100_000.0,
            0.0,
            params_eval,
            strategy_profiles,
        )
        total_trades += n_trades

    trade_log = all_trades[:total_trades]
    metrics = compute_metrics(trade_log)

    args.output.mkdir(parents=True, exist_ok=True)
    trade_log_path = args.output / "mnq_v1_full_history_trade_log.npy"
    report_path = args.output / "mnq_v1_full_history_report.json"
    np.save(trade_log_path, trade_log)

    report = {
        "version": "v1.0",
        "data_path": str(args.combined_data),
        "params_config": str(args.params_config),
        "timeframe_minutes": TIMEFRAME_MINUTES,
        "session_start": shared_cfg["session_start"],
        "session_end": shared_cfg["session_end"],
        "trigger_start_minute": int(shared_cfg["trigger_start_minute"]),
        "trigger_end_minute": int(shared_cfg["trigger_end_minute"]),
        "time_stop_minute": int(shared_cfg["time_stop_minute"]),
        "donchian_lookback": int(shared_cfg["donchian_lookback"]),
        "stop_atr_multiplier": float(eval_cfg["stop_atr_multiplier"]),
        "target_atr_multiplier": float(eval_cfg["target_atr_multiplier"]),
        "entry_on_close": True,
        "commission_per_side": 0.0,
        "extra_slippage_ticks": 0.0,
        "date_range": {
            "first_session": data["session_dates"][0],
            "last_session": data["session_dates"][-1],
        },
        "metrics": metrics,
        "artifacts": {
            "trade_log": str(trade_log_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print("Performance")
    print(f"  Trades:            {metrics['total_trades']} ({metrics['wins']} win / {metrics['losses']} loss)")
    print(f"  Win rate:          {metrics['win_rate']:.2%}")
    print(f"  Profit factor:     {metrics['profit_factor']:.2f}")
    print(f"  Net profit:        ${metrics['net_profit']:,.2f}")
    print(f"  Max drawdown:      ${metrics['max_drawdown']:,.2f}")
    print(f"  Avg trade net:     ${metrics['avg_trade_net']:,.2f}")
    print()
    print(f"Trade log saved to {trade_log_path}")
    print(f"Report saved to    {report_path}")


if __name__ == "__main__":
    main()
