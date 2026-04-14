"""Monte-Carlo pass/fail simulation for asymmetric MFF sizing candidates."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.asymmetrical_sizing_matrix import _apply_combo, _build_base_repriced


N_SIMS = 10_000
PROFIT_TARGET = 3_000.0
TRAILING_DRAWDOWN_LIMIT = 5_000.0
MAX_TRADES_PER_PATH = 10_000
OUTPUT_DIR = Path("output/monte_carlo_asymmetrical_setups")


@dataclass(frozen=True)
class SetupSpec:
    name: str
    mnq_contracts: int
    mgc_contracts: int


SETUPS = (
    SetupSpec(name="A", mnq_contracts=2, mgc_contracts=3),
    SetupSpec(name="B", mnq_contracts=3, mgc_contracts=1),
)


def _simulate_paths(
    trade_pnls: np.ndarray,
    n_sims: int,
    profit_target: float,
    trailing_drawdown_limit: float,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    pass_count = 0
    successful_trades_to_pass: list[int] = []
    successful_max_drawdowns: list[float] = []

    for _ in range(n_sims):
        equity = 0.0
        peak = 0.0
        max_drawdown = 0.0

        for trade_idx in range(1, MAX_TRADES_PER_PATH + 1):
            pnl = float(rng.choice(trade_pnls))
            equity += pnl
            if equity > peak:
                peak = equity
            drawdown = peak - equity
            if drawdown > max_drawdown:
                max_drawdown = drawdown

            if equity >= profit_target:
                pass_count += 1
                successful_trades_to_pass.append(trade_idx)
                successful_max_drawdowns.append(max_drawdown)
                break

            if drawdown >= trailing_drawdown_limit:
                break
        else:
            # Guardrail against non-terminating paths: treat as unresolved failure.
            continue

    probability = pass_count / n_sims if n_sims else 0.0
    return {
        "n_sims": int(n_sims),
        "pass_count": int(pass_count),
        "probability_of_passing": probability,
        "average_trades_to_pass": (
            float(np.mean(successful_trades_to_pass)) if successful_trades_to_pass else None
        ),
        "average_max_drawdown_successful_paths": (
            float(np.mean(successful_max_drawdowns)) if successful_max_drawdowns else None
        ),
        "median_trades_to_pass": (
            float(np.median(successful_trades_to_pass)) if successful_trades_to_pass else None
        ),
    }


def main() -> None:
    base = _build_base_repriced()
    rows = []

    for idx, setup in enumerate(SETUPS):
        combo = _apply_combo(base, setup.mnq_contracts, setup.mgc_contracts)
        pnl = combo["net_pnl_repriced"].to_numpy(dtype=np.float64)
        stats = _simulate_paths(
            trade_pnls=pnl,
            n_sims=N_SIMS,
            profit_target=PROFIT_TARGET,
            trailing_drawdown_limit=TRAILING_DRAWDOWN_LIMIT,
            seed=42 + idx,
        )
        row = {
            "setup": setup.name,
            "mnq_contracts": setup.mnq_contracts,
            "mgc_contracts": setup.mgc_contracts,
            **stats,
        }
        rows.append(row)

    result = pd.DataFrame(rows).sort_values(
        ["probability_of_passing", "average_trades_to_pass"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "monte_carlo_results.csv"
    json_path = OUTPUT_DIR / "monte_carlo_results.json"
    result.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(result.to_dict(orient="records"), indent=2), encoding="utf-8")

    print("Monte-Carlo Pass Probability")
    print(
        f"{'Setup':<6} {'MNQ':>4} {'MGC':>4} {'Pass Prob':>12} "
        f"{'Avg Trades to Pass':>20} {'Avg Max DD (Winners)':>22}"
    )
    print("-" * 78)
    for row in result.itertuples(index=False):
        avg_trades = f"{row.average_trades_to_pass:.2f}" if pd.notna(row.average_trades_to_pass) else "n/a"
        avg_dd = (
            f"${row.average_max_drawdown_successful_paths:,.2f}"
            if pd.notna(row.average_max_drawdown_successful_paths)
            else "n/a"
        )
        print(
            f"{row.setup:<6} "
            f"{int(row.mnq_contracts):>4} "
            f"{int(row.mgc_contracts):>4} "
            f"{float(row.probability_of_passing):>11.2%} "
            f"{avg_trades:>20} "
            f"{avg_dd:>22}"
        )

    winner = result.iloc[0]
    print()
    print(
        f"Hoehere statistische Pass-Wahrscheinlichkeit: Setup {winner['setup']} "
        f"({int(winner['mnq_contracts'])} MNQ / {int(winner['mgc_contracts'])} MGC)"
    )
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
