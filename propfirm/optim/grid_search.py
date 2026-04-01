import numpy as np
from itertools import product
from propfirm.monte_carlo.bootstrap import run_monte_carlo
from propfirm.core.types import MNQ_TICK_VALUE


def generate_synthetic_trades(win_rate, reward_ticks, risk_ticks, contracts, n_trades, seed):
    rng = np.random.RandomState(seed)
    wins = rng.rand(n_trades) < win_rate
    pnls = np.where(
        wins,
        reward_ticks * MNQ_TICK_VALUE * contracts,
        -risk_ticks * MNQ_TICK_VALUE * contracts,
    )
    return pnls.astype(np.float64)


def run_synthetic_grid_search(param_grid, mff_config, n_mc_sims=1000,
                               seed=42, n_workers=1, n_synthetic_trades=200):
    results = []
    stop_ticks = 40.0

    combos = list(product(
        param_grid["win_rate"],
        param_grid["risk_reward"],
        param_grid["contracts"],
    ))

    for i, (wr, rr, contracts) in enumerate(combos):
        target_ticks = stop_ticks * rr
        trades = generate_synthetic_trades(
            win_rate=wr, reward_ticks=target_ticks, risk_ticks=stop_ticks,
            contracts=contracts, n_trades=n_synthetic_trades, seed=seed + i,
        )

        mc_result = run_monte_carlo(
            eval_pnls=trades, mff_config=mff_config, funded_pnls=trades,
            n_sims=n_mc_sims, block_mode="fixed", block_min=5, block_max=10,
            seed=seed + i + 10000, n_workers=n_workers,
            eval_target_length=n_synthetic_trades,
            funded_target_length=n_synthetic_trades,
        )

        results.append({
            "params": {
                "win_rate": wr, "risk_reward": rr,
                "stop_ticks": stop_ticks, "target_ticks": target_ticks,
                "contracts": contracts,
            },
            "nve": mc_result.nve,
            "eval_pass_rate": mc_result.eval_pass_rate,
            "payout_rate": mc_result.payout_rate,
            "funded_survival_rate": mc_result.funded_survival_rate,
            "mean_days_to_eval_pass": mc_result.mean_days_to_eval_pass,
            "mean_drawdown": mc_result.mean_drawdown,
        })

    results.sort(key=lambda r: r["nve"], reverse=True)
    return results
