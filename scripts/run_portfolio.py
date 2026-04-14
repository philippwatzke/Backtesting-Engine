#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np

from propfirm.core.multi_engine import run_multi_asset_day_kernel
from propfirm.core.types import (
    EXIT_CIRCUIT_BREAKER,
    PARAMS_CONTRACTS,
    PARAMS_EXTRA_SLIPPAGE_TICKS,
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


PORTFOLIO_DAILY_LOG_DTYPE = np.dtype([
    ("session_date", "U10"),
    ("mgc_day_pnl", "f8"),
    ("mnq_day_pnl", "f8"),
    ("portfolio_day_pnl", "f8"),
    ("portfolio_equity", "f8"),
    ("portfolio_drawdown", "f8"),
    ("circuit_breaker_triggered", "i1"),
    ("circuit_breaker_exit_trades", "i2"),
])

PORTFOLIO_TRADE_LOG_DTYPE = np.dtype([("asset", "U8")] + TRADE_LOG_DTYPE.descr)


def _compute_raw_metrics(trade_log: np.ndarray) -> dict:
    total_trades = int(len(trade_log))
    if total_trades == 0:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "final_equity": 0.0,
            "max_drawdown": 0.0,
        }
    net_pnl = trade_log["net_pnl"].astype(np.float64)
    equity = np.cumsum(net_pnl)
    gross_profit = float(net_pnl[net_pnl > 0.0].sum())
    gross_loss = float(-net_pnl[net_pnl < 0.0].sum())
    drawdown = equity - np.maximum.accumulate(equity)
    return {
        "total_trades": total_trades,
        "win_rate": float(np.mean(net_pnl > 0.0)),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0.0 else float("inf"),
        "final_equity": float(net_pnl.sum()),
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
    }


def _build_strategy_profiles(shared_cfg: dict, phase_cfg: dict, risk_buffer_fraction: float) -> np.ndarray:
    profiles = np.zeros((1, PROFILE_ARRAY_LENGTH), dtype=np.float64)
    profiles[0, PROFILE_RISK_PER_TRADE_USD] = float(phase_cfg["risk_per_trade_usd"])
    profiles[0, PROFILE_STOP_ATR_MULTIPLIER] = float(phase_cfg["stop_atr_multiplier"])
    profiles[0, PROFILE_TARGET_ATR_MULTIPLIER] = float(phase_cfg["target_atr_multiplier"])
    profiles[0, PROFILE_BREAKEVEN_TRIGGER_TICKS] = float(shared_cfg["breakeven_trigger_ticks"])
    profiles[0, PROFILE_RISK_BUFFER_FRACTION] = float(risk_buffer_fraction)
    return profiles


def _build_asset_setup(name: str, data_path: Path, mff_config: Path, params_config: Path, timeframe_minutes: int) -> dict:
    mff_cfg = load_mff_config(mff_config)
    params_cfg = load_params_config(params_config)
    shared_cfg = params_cfg["strategy"]["mgc_h1_trend"]["shared"]
    eval_cfg = params_cfg["strategy"]["mgc_h1_trend"]["eval"]
    funded_cfg = params_cfg["strategy"]["mgc_h1_trend"]["funded"]
    portfolio_shared = params_cfg.get("portfolio", {}).get("shared", {})
    risk_buffer_fraction = float(portfolio_shared.get("risk_buffer_fraction", 0.25))
    slip_cfg = params_cfg["slippage"]

    data = load_session_data(
        data_path,
        atr_period=slip_cfg["atr_period"],
        trailing_atr_days=slip_cfg["trailing_atr_days"],
        timeframe_minutes=timeframe_minutes,
        session_start=shared_cfg["session_start"],
        session_end=shared_cfg["session_end"],
    )
    slippage_lookup = build_slippage_lookup(
        None,
        require_file=False,
        session_minutes=int(data["session_minutes"]),
    )
    params_eval = build_phase_params(
        shared_cfg,
        eval_cfg,
        slip_cfg,
        mff_cfg["instrument"]["commission_per_side"],
        strategy_name="mgc_h1_trend",
        instrument_cfg=mff_cfg["instrument"],
    )
    params_eval[PARAMS_EXTRA_SLIPPAGE_TICKS] = 0.0
    params_funded = build_phase_params(
        shared_cfg,
        funded_cfg,
        slip_cfg,
        mff_cfg["instrument"]["commission_per_side"],
        strategy_name="mgc_h1_trend",
        instrument_cfg=mff_cfg["instrument"],
    )
    params_funded[PARAMS_EXTRA_SLIPPAGE_TICKS] = 0.0
    profiles_eval = _build_strategy_profiles(shared_cfg, eval_cfg, risk_buffer_fraction)
    profiles_funded = _build_strategy_profiles(shared_cfg, funded_cfg, risk_buffer_fraction)

    day_map = {
        session_date: (idx, bounds[0], bounds[1])
        for idx, (session_date, bounds) in enumerate(zip(data["session_dates"], data["day_boundaries"]))
    }
    return {
        "name": name,
        "data": data,
        "day_map": day_map,
        "params_eval": params_eval,
        "params_funded": params_funded,
        "strategy_profiles_eval": profiles_eval,
        "strategy_profiles_funded": profiles_funded,
        "slippage_lookup": slippage_lookup,
        "shared_cfg": shared_cfg,
        "eval_cfg": eval_cfg,
        "funded_cfg": funded_cfg,
        "mff_cfg": mff_cfg,
        "state": MFFState(mff_cfg),
        "funded_payout_cycle_id": 0,
        "disabled": False,
    }


def _resolve_phase_context(asset_setup: dict) -> tuple[np.ndarray, np.ndarray, int, int]:
    state = asset_setup["state"]
    if state.phase == "eval":
        params = asset_setup["params_eval"].copy()
        params[PARAMS_CONTRACTS] = float(asset_setup["mff_cfg"]["eval"]["max_contracts"])
        strategy_profiles = asset_setup["strategy_profiles_eval"].copy()
        return params, strategy_profiles, 0, -1

    params = asset_setup["params_funded"].copy()
    params[PARAMS_CONTRACTS] = float(state.get_max_contracts())
    strategy_profiles = asset_setup["strategy_profiles_funded"].copy()
    return params, strategy_profiles, 1, int(asset_setup["funded_payout_cycle_id"])


def _empty_asset_config(name: str, template: dict) -> dict:
    params, strategy_profiles, phase_id, payout_cycle_id = _resolve_phase_context(template)
    return {
        "name": name,
        "opens": np.empty(0, dtype=np.float64),
        "highs": np.empty(0, dtype=np.float64),
        "lows": np.empty(0, dtype=np.float64),
        "closes": np.empty(0, dtype=np.float64),
        "volumes": np.empty(0, dtype=np.uint64),
        "timestamps": np.empty(0, dtype=np.int64),
        "minute_of_day": np.empty(0, dtype=np.int16),
        "bar_atr": np.empty(0, dtype=np.float64),
        "trailing_atr": np.empty(0, dtype=np.float64),
        "daily_atr_ratio": np.empty(0, dtype=np.float64),
        "rvol": np.empty(0, dtype=np.float64),
        "close_sma_50": np.empty(0, dtype=np.float64),
        "daily_regime_bias": np.empty(0, dtype=np.float64),
        "donchian_high_5": np.empty(0, dtype=np.float64),
        "donchian_low_5": np.empty(0, dtype=np.float64),
        "day_of_week": np.empty(0, dtype=np.int8),
        "slippage_lookup": template["slippage_lookup"],
        "strategy_fn": combined_portfolio_signal,
        "strategy_profiles": strategy_profiles,
        "trade_log": np.zeros(1, dtype=TRADE_LOG_DTYPE),
        "trade_log_offset": 0,
        "current_day_id": -1,
        "current_phase_id": phase_id,
        "current_payout_cycle_id": payout_cycle_id,
        "liquidation_floor_equity": float(template["state"].get_liquidation_floor_equity()),
        "starting_equity": float(template["state"].equity),
        "starting_pnl": 0.0,
        "params": params,
    }


def _slice_asset_config(asset_setup: dict, session_date: str, day_id: int) -> dict:
    if asset_setup["disabled"]:
        return _empty_asset_config(asset_setup["name"], asset_setup)
    info = asset_setup["day_map"].get(session_date)
    if info is None:
        return _empty_asset_config(asset_setup["name"], asset_setup)

    _, start, end = info
    data = asset_setup["data"]
    params, strategy_profiles, phase_id, payout_cycle_id = _resolve_phase_context(asset_setup)
    return {
        "name": asset_setup["name"],
        "opens": data["open"][start:end],
        "highs": data["high"][start:end],
        "lows": data["low"][start:end],
        "closes": data["close"][start:end],
        "volumes": data["volume"][start:end],
        "timestamps": data["timestamps"][start:end],
        "minute_of_day": data["minute_of_day"][start:end],
        "bar_atr": data["bar_atr"][start:end],
        "trailing_atr": data["trailing_median_atr"][start:end],
        "daily_atr_ratio": data["daily_atr_ratio"][start:end],
        "rvol": data["rvol"][start:end],
        "close_sma_50": data["close_sma_50"][start:end],
        "daily_regime_bias": data["daily_regime_bias"][start:end],
        "donchian_high_5": data["donchian_high_5"][start:end],
        "donchian_low_5": data["donchian_low_5"][start:end],
        "day_of_week": data["day_of_week"][start:end],
        "slippage_lookup": asset_setup["slippage_lookup"],
        "strategy_fn": combined_portfolio_signal,
        "strategy_profiles": strategy_profiles,
        "trade_log": np.zeros(4, dtype=TRADE_LOG_DTYPE),
        "trade_log_offset": 0,
        "current_day_id": day_id,
        "current_phase_id": phase_id,
        "current_payout_cycle_id": payout_cycle_id,
        "liquidation_floor_equity": float(asset_setup["state"].get_liquidation_floor_equity()),
        "starting_equity": float(asset_setup["state"].equity),
        "starting_pnl": 0.0,
        "params": params,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mgc-data", type=Path, default=Path("data/processed/MGC_1m_full_test.parquet"))
    parser.add_argument("--mnq-data", type=Path, default=Path("data/processed/MNQ_1m_full_test.parquet"))
    parser.add_argument("--mgc-mff-config", type=Path, default=Path("configs/mff_flex_50k_mgc.toml"))
    parser.add_argument("--mnq-mff-config", type=Path, default=Path("configs/mff_flex_50k_mnq.toml"))
    parser.add_argument("--mgc-params-config", type=Path, default=Path("configs/default_params.toml"))
    parser.add_argument("--mnq-params-config", type=Path, default=Path("configs/default_params_mnq.toml"))
    parser.add_argument("--mgc-timeframe-minutes", type=int, default=60)
    parser.add_argument("--mnq-timeframe-minutes", type=int, default=30)
    parser.add_argument("--circuit-breaker-threshold", type=float, default=-800.0)
    parser.add_argument("--output", type=Path, default=Path("output/backtests_portfolio_mgc_mnq_test"))
    args = parser.parse_args()

    mgc = _build_asset_setup("MGC", args.mgc_data, args.mgc_mff_config, args.mgc_params_config, args.mgc_timeframe_minutes)
    mnq = _build_asset_setup("MNQ", args.mnq_data, args.mnq_mff_config, args.mnq_params_config, args.mnq_timeframe_minutes)

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
        day_cb_exits = int(np.sum(mgc_log["exit_reason"] == EXIT_CIRCUIT_BREAKER) + np.sum(mnq_log["exit_reason"] == EXIT_CIRCUIT_BREAKER))
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

    raw_metrics = _compute_raw_metrics(portfolio_trade_log)
    combined_max_drawdown = float(portfolio_daily["portfolio_drawdown"].min()) if len(portfolio_daily) else 0.0
    circuit_breaker_exit_trades = int(np.sum(portfolio_trade_log["exit_reason"] == EXIT_CIRCUIT_BREAKER)) if len(portfolio_trade_log) else 0

    args.output.mkdir(parents=True, exist_ok=True)
    trade_log_path = args.output / "latest_portfolio_trade_log.npy"
    daily_log_path = args.output / "latest_portfolio_daily_log.npy"
    np.save(trade_log_path, portfolio_trade_log)
    np.save(daily_log_path, portfolio_daily)

    report = {
        "portfolio": {
            "assets": ["MGC", "MNQ"],
            "session_dates": {
                "start": master_calendar[0] if master_calendar else None,
                "end": master_calendar[-1] if master_calendar else None,
                "count": len(master_calendar),
            },
            "metrics": {
                "combined_final_net_equity": raw_metrics["final_equity"],
                "combined_max_drawdown": combined_max_drawdown,
                "combined_win_rate": raw_metrics["win_rate"],
                "total_trades": raw_metrics["total_trades"],
                "profit_factor": raw_metrics["profit_factor"],
                "circuit_breaker_trigger_days": circuit_breaker_days,
                "circuit_breaker_exit_trades": circuit_breaker_exit_trades,
            },
            "artifacts": {
                "trade_log": str(trade_log_path),
                "daily_log": str(daily_log_path),
            },
        }
    }
    with open(args.output / "latest_portfolio_backtest.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Total trades: {raw_metrics['total_trades']}")
    print(f"Win rate: {raw_metrics['win_rate']:.2%}")
    print(f"Profit factor: {raw_metrics['profit_factor']:.4f}")
    print(f"Combined final net equity: {raw_metrics['final_equity']:.2f}")
    print(f"Combined max drawdown: {combined_max_drawdown:.2f}")
    print(f"Circuit breaker trigger days: {circuit_breaker_days}")
    print(f"Circuit breaker exit trades: {circuit_breaker_exit_trades}")


if __name__ == "__main__":
    main()
