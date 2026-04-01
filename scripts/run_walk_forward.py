#!/usr/bin/env python
"""Run walk-forward analysis on real historical data."""
import argparse
from pathlib import Path

from propfirm.io.config import load_mff_config, load_params_config, build_phase_params
from propfirm.market.data_loader import load_session_data
from propfirm.market.slippage import build_slippage_lookup
from propfirm.optim.walk_forward import run_walk_forward, _serialize_param_grid
from propfirm.io.reporting import build_report, save_report
from propfirm.core.types import (
    PARAMS_STOP_TICKS, PARAMS_TARGET_TICKS, PARAMS_CONTRACTS,
    PARAMS_DAILY_STOP,
)

MC_EVAL_TARGET_LENGTH = 200
MC_FUNDED_TARGET_LENGTH = 300


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/MNQ_1m_train.parquet"))
    parser.add_argument("--mff-config", type=Path, default=Path("configs/mff_flex_50k.toml"))
    parser.add_argument("--params-config", type=Path, default=Path("configs/default_params.toml"))
    parser.add_argument("--output", type=Path, default=Path("output/walk_forward"))
    parser.add_argument("--train-days", type=int, default=120)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--step-days", type=int, default=60)
    parser.add_argument("--n-mc-sims", type=int, default=None,
                        help="Optional override for monte_carlo.n_simulations")
    parser.add_argument("--n-workers", type=int, default=1)
    args = parser.parse_args()

    mff_cfg = load_mff_config(args.mff_config)
    params_cfg = load_params_config(args.params_config)
    orb_shared = params_cfg["strategy"]["orb"]["shared"]
    orb_eval = params_cfg["strategy"]["orb"]["eval"]
    orb_funded = params_cfg["strategy"]["orb"]["funded"]
    slip_cfg = params_cfg["slippage"]
    mc_cfg = params_cfg["monte_carlo"]
    seed = params_cfg["general"]["random_seed"]
    if mc_cfg["block_mode"] != "daily":
        raise ValueError("run_walk_forward.py requires monte_carlo.block_mode = 'daily'")
    n_mc_sims = args.n_mc_sims if args.n_mc_sims is not None else mc_cfg["n_simulations"]

    data = load_session_data(
        args.data,
        atr_period=slip_cfg["atr_period"],
        trailing_atr_days=slip_cfg["trailing_atr_days"],
    )
    slippage_lookup = build_slippage_lookup(
        Path("data/slippage/slippage_profile.parquet"),
        require_file=True,
    )

    base_params_eval = build_phase_params(
        orb_shared, orb_eval, slip_cfg, mff_cfg["instrument"]["commission_per_side"]
    )
    base_params_funded = build_phase_params(
        orb_shared, orb_funded, slip_cfg, mff_cfg["instrument"]["commission_per_side"]
    )

    param_grid = {
        ("eval", PARAMS_STOP_TICKS): [30.0, 40.0, 50.0],
        ("eval", PARAMS_TARGET_TICKS): [45.0, 60.0, 75.0],
        ("eval", PARAMS_CONTRACTS): [5.0, 10.0, 15.0],
        ("funded", PARAMS_TARGET_TICKS): [60.0, 80.0, 100.0],
        ("funded", PARAMS_CONTRACTS): [10.0, 20.0, 30.0],
        ("funded", PARAMS_DAILY_STOP): [-750.0, -1000.0, -1250.0],
    }

    n_combos = 1
    for v in param_grid.values():
        n_combos *= len(v)
    n_days = len(data["day_boundaries"])
    print(f"Walk-forward: {n_combos} param combos, {n_days} total days")
    print(f"Windows: train={args.train_days}d, test={args.test_days}d, step={args.step_days}d")

    results = run_walk_forward(
        session_data=data,
        slippage_lookup=slippage_lookup,
        base_params_eval=base_params_eval,
        base_params_funded=base_params_funded,
        param_grid=param_grid,
        mff_config=mff_cfg,
        window_train_days=args.train_days,
        window_test_days=args.test_days,
        step_days=args.step_days,
        n_mc_sims=n_mc_sims,
        mc_block_min=mc_cfg["block_size_min"],
        mc_block_max=mc_cfg["block_size_max"],
        mc_eval_target_length=MC_EVAL_TARGET_LENGTH,
        mc_funded_target_length=MC_FUNDED_TARGET_LENGTH,
        seed=seed,
        n_workers=args.n_workers,
    )

    def fmt_money(value):
        return "N/A" if value is None else f"${value:.0f}"

    def fmt_pct(value):
        return "N/A" if value is None else f"{value:.1%}"

    print(f"\nWalk-forward results ({len(results)} windows):")
    for r in results:
        print(f"  Window {r['window']}: "
              f"Status={r['status']} | "
              f"OOS Status={r['oos_status']} | "
              f"IS NVE={fmt_money(r['in_sample_nve'])} | "
              f"OOS NVE={fmt_money(r['oos_nve'])} | "
              f"OOS Payout Rate={fmt_pct(r['oos_payout_rate'])} | "
              f"Params={r['best_params']}")

    report = build_report(
        params={
            "eval": orb_eval,
            "funded": orb_funded,
            "shared": orb_shared,
            "walk_forward_param_grid": _serialize_param_grid(param_grid),
            "walk_forward_results": results,
        },
        mc_result=None,
        config_snapshot={"params_cfg": params_cfg, "mff_cfg": mff_cfg},
        data_split=args.data.stem,
        data_date_range=(data["session_dates"][0], data["session_dates"][-1]),
        seed=seed,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "walk_forward_results.json"
    report["artifacts"] = {
        "report": str(report_path),
        "input_data": str(args.data),
    }
    report["runtime_meta"] = {
        "mc_mode": mc_cfg["block_mode"],
        "lifecycle_aware_walk_forward": True,
        "mc_n_sims": n_mc_sims,
        "mc_block_size_min": mc_cfg["block_size_min"],
        "mc_block_size_max": mc_cfg["block_size_max"],
        "mc_eval_target_length": MC_EVAL_TARGET_LENGTH,
        "mc_funded_target_length": MC_FUNDED_TARGET_LENGTH,
        "train_days": args.train_days,
        "test_days": args.test_days,
        "step_days": args.step_days,
    }
    save_report(report, report_path)
    print(f"\nResults saved to {report_path}")


if __name__ == "__main__":
    main()
