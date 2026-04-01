#!/usr/bin/env python
"""Run Monte-Carlo simulation on lifecycle PNL artifacts."""
import argparse
import numpy as np
from pathlib import Path

from propfirm.io.config import load_mff_config, load_params_config
from propfirm.monte_carlo.bootstrap import run_monte_carlo, split_daily_log_for_mc
from propfirm.io.reporting import build_report, save_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", type=Path, required=True,
                        help="Path to .npy file (flat PNLs or structured daily lifecycle log)")
    parser.add_argument("--mff-config", type=Path, default=Path("configs/mff_flex_50k.toml"))
    parser.add_argument("--params-config", type=Path, default=Path("configs/default_params.toml"))
    parser.add_argument("--output", type=Path, default=Path("output/monte_carlo"))
    parser.add_argument("--n-workers", type=int, default=1)
    args = parser.parse_args()

    mff_cfg = load_mff_config(args.mff_config)
    params_cfg = load_params_config(args.params_config)
    mc_cfg = params_cfg["monte_carlo"]
    seed = params_cfg["general"]["random_seed"]

    raw = np.load(args.trades, allow_pickle=False)
    block_mode = mc_cfg["block_mode"]
    if raw.dtype.names is not None and "day_pnl" in raw.dtype.names:
        if block_mode != "daily":
            raise ValueError("Structured daily lifecycle logs require monte_carlo.block_mode = 'daily'")
        try:
            phase_pools = split_daily_log_for_mc(raw)
        except ValueError as exc:
            raise ValueError(
                "Structured daily log is not lifecycle-ready for daily MC. "
                "Need eval days plus funded payout_cycle_id==0 days."
            ) from exc
        eval_day_pnls = phase_pools["eval_day_pnls"]
        funded_day_pnls = phase_pools["funded_day_pnls"]
        print(f"Loaded structured daily log: eval={len(eval_day_pnls)} funded={len(funded_day_pnls)} days")
    else:
        if block_mode != "fixed":
            raise ValueError("Flat PNL arrays are only allowed for fixed-block legacy/synthetic MC runs")
        eval_day_pnls = raw.astype(np.float64)
        funded_day_pnls = eval_day_pnls
        print(f"Loaded flat PNL array: {len(eval_day_pnls)} trades")

    result = run_monte_carlo(
        eval_day_pnls, mff_cfg,
        funded_pnls=funded_day_pnls,
        n_sims=mc_cfg["n_simulations"],
        block_mode=block_mode,
        block_min=mc_cfg["block_size_min"],
        block_max=mc_cfg["block_size_max"],
        seed=seed,
        n_workers=args.n_workers,
    )

    print(f"Eval Pass Rate: {result.eval_pass_rate:.1%} "
          f"[{result.eval_pass_rate_ci_5:.1%} - {result.eval_pass_rate_ci_95:.1%}]")
    print(f"Funded Survival Rate: {result.funded_survival_rate:.1%}")
    print(f"Payout Rate: {result.payout_rate:.1%}")
    print(f"Mean Payout Net: ${result.mean_payout_net:.2f}")
    print(f"NVE: ${result.nve:.2f}")
    print(f"Mean days to eval pass: {result.mean_days_to_eval_pass:.1f}")
    print(f"Mean funded days to payout: {result.mean_funded_days_to_payout:.1f}")

    report = build_report(
        params={},
        mc_result=result,
        config_snapshot={"params_cfg": params_cfg, "mff_cfg": mff_cfg},
        data_split="custom",
        data_date_range=("", ""),
        seed=seed,
    )
    report["artifacts"] = {"input_artifact": str(args.trades)}
    report["runtime_meta"] = {
        "mc_mode": block_mode,
        "lifecycle_aware_input": bool(raw.dtype.names is not None and "day_pnl" in raw.dtype.names),
    }
    save_report(report, args.output / "mc_results.json")
    print(f"Report saved to {args.output / 'mc_results.json'}")


if __name__ == "__main__":
    main()
