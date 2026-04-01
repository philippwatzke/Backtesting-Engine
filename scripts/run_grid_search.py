#!/usr/bin/env python
"""Run synthetic grid search."""
import argparse
import json
from pathlib import Path

from propfirm.io.config import load_mff_config, load_params_config
from propfirm.optim.grid_search import run_synthetic_grid_search
from propfirm.io.reporting import build_report, save_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mff-config", type=Path, default=Path("configs/mff_flex_50k.toml"))
    parser.add_argument("--params-config", type=Path, default=Path("configs/default_params.toml"))
    parser.add_argument("--output", type=Path, default=Path("output/grid_search"))
    parser.add_argument("--n-mc-sims", type=int, default=1000)
    parser.add_argument("--n-workers", type=int, default=1)
    args = parser.parse_args()

    mff_cfg = load_mff_config(args.mff_config)
    params_cfg = load_params_config(args.params_config)
    seed = params_cfg["general"]["random_seed"]

    param_grid = {
        "win_rate": [0.55, 0.60, 0.65, 0.70],
        "risk_reward": [1.0, 1.2, 1.5, 2.0],
        "contracts": [5, 10, 15, 20],
    }

    n_combos = (len(param_grid['win_rate']) * len(param_grid['risk_reward'])
                * len(param_grid['contracts']))
    print(f"Running grid search: {len(param_grid['win_rate'])} x "
          f"{len(param_grid['risk_reward'])} x {len(param_grid['contracts'])} = "
          f"{n_combos} combinations")

    results = run_synthetic_grid_search(
        param_grid, mff_cfg,
        n_mc_sims=args.n_mc_sims,
        seed=seed,
        n_workers=args.n_workers,
    )

    print(f"\nTop 10 by NVE:")
    for i, r in enumerate(results[:10]):
        p = r["params"]
        print(f"  {i+1}. NVE=${r['nve']:.0f} | WR={p['win_rate']:.0%} "
              f"RR={p['risk_reward']:.1f} C={p['contracts']} | "
              f"EvalPass={r['eval_pass_rate']:.1%} "
              f"PayoutRate={r['payout_rate']:.1%}")

    report = build_report(
        params={"grid": param_grid, "top_10": results[:10]},
        mc_result=None,
        config_snapshot={"params_cfg": params_cfg, "mff_cfg": mff_cfg},
        data_split="synthetic",
        data_date_range=("N/A", "N/A"),
        seed=seed,
    )
    report_path = args.output / "grid_search_results.json"
    report["artifacts"] = {"report": str(report_path)}
    report["runtime_meta"] = {
        "mc_mode": "fixed",
        "optimization_path": "synthetic_prestudy",
        "lifecycle_aware": False,
    }
    save_report(report, report_path)
    print(f"\nFull results saved to {report_path}")


if __name__ == "__main__":
    main()
