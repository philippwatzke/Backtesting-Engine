"""Block-bootstrap Monte Carlo with additional slippage stress for top setups."""

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
from scripts.final_portfolio_backtest import INSTRUMENT_SPECS


N_SIMS = 10_000
PROFIT_TARGET = 3_000.0
TRAILING_DRAWDOWN_LIMIT = 5_000.0
BLOCK_DAYS = 5
MAX_BLOCKS_PER_PATH = 500
OUTPUT_DIR = Path("output/monte_carlo_block_slippage_stress")


@dataclass(frozen=True)
class SetupSpec:
    name: str
    mnq_contracts: int
    mgc_contracts: int


SETUPS = (
    SetupSpec(name="2x3", mnq_contracts=2, mgc_contracts=3),
    SetupSpec(name="1x3", mnq_contracts=1, mgc_contracts=3),
    SetupSpec(name="2x2", mnq_contracts=2, mgc_contracts=2),
)


def _prepare_combo_trades(setup: SetupSpec) -> pd.DataFrame:
    base = _build_base_repriced()
    combo = _apply_combo(base, setup.mnq_contracts, setup.mgc_contracts)
    combo = combo.sort_values(["exit_dt", "entry_dt", "asset"], kind="stable").reset_index(drop=True)
    combo["exit_date"] = pd.to_datetime(combo["exit_date"])
    combo["contracts"] = combo["contracts"].astype(int)
    combo["trade_cost_per_tick"] = combo.apply(
        lambda row: INSTRUMENT_SPECS[str(row["asset"])].tick_value * int(row["contracts"]),
        axis=1,
    )
    return combo


def _build_blocks(trades: pd.DataFrame, block_days: int) -> list[pd.DataFrame]:
    unique_dates = sorted(trades["exit_date"].drop_duplicates().tolist())
    blocks: list[pd.DataFrame] = []
    if len(unique_dates) < block_days:
        return [trades.copy()]
    for start_idx in range(0, len(unique_dates) - block_days + 1):
        dates = unique_dates[start_idx : start_idx + block_days]
        block = trades.loc[trades["exit_date"].isin(dates)].copy()
        if not block.empty:
            block = block.sort_values(["exit_dt", "entry_dt", "asset"], kind="stable").reset_index(drop=True)
            blocks.append(block)
    return blocks


def _simulate_setup(blocks: list[pd.DataFrame], seed: int) -> dict[str, float | int | None]:
    rng = np.random.default_rng(seed)
    pass_count = 0
    successful_trades_to_pass: list[int] = []
    successful_max_drawdowns: list[float] = []

    for _ in range(N_SIMS):
        equity = 0.0
        peak = 0.0
        max_drawdown = 0.0
        trades_seen = 0
        passed = False

        for _block_idx in range(MAX_BLOCKS_PER_PATH):
            block = blocks[int(rng.integers(0, len(blocks)))]
            extra_ticks = rng.integers(0, 3, size=len(block), endpoint=False)
            stressed_pnl = (
                block["net_pnl_repriced"].to_numpy(dtype=np.float64)
                - extra_ticks.astype(np.float64) * block["trade_cost_per_tick"].to_numpy(dtype=np.float64)
            )

            for pnl in stressed_pnl:
                trades_seen += 1
                equity += float(pnl)
                if equity > peak:
                    peak = equity
                drawdown = peak - equity
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

                if equity >= PROFIT_TARGET:
                    pass_count += 1
                    successful_trades_to_pass.append(trades_seen)
                    successful_max_drawdowns.append(max_drawdown)
                    passed = True
                    break

                if drawdown >= TRAILING_DRAWDOWN_LIMIT:
                    passed = False
                    break

            if passed or drawdown >= TRAILING_DRAWDOWN_LIMIT:
                break

    return {
        "probability_of_passing": pass_count / N_SIMS,
        "average_trades_to_pass": float(np.mean(successful_trades_to_pass)) if successful_trades_to_pass else None,
        "average_max_drawdown_successful_paths": (
            float(np.mean(successful_max_drawdowns)) if successful_max_drawdowns else None
        ),
        "median_trades_to_pass": float(np.median(successful_trades_to_pass)) if successful_trades_to_pass else None,
        "successful_paths": int(pass_count),
    }


def main() -> None:
    rows: list[dict[str, float | int | None | str]] = []
    for idx, setup in enumerate(SETUPS):
        combo = _prepare_combo_trades(setup)
        blocks = _build_blocks(combo, BLOCK_DAYS)
        stats = _simulate_setup(blocks, seed=42 + idx)
        rows.append(
            {
                "setup": setup.name,
                "mnq_contracts": setup.mnq_contracts,
                "mgc_contracts": setup.mgc_contracts,
                "block_days": BLOCK_DAYS,
                **stats,
            }
        )

    result = pd.DataFrame(rows).sort_values(
        ["probability_of_passing", "average_trades_to_pass"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "block_slippage_stress_results.csv"
    json_path = OUTPUT_DIR / "block_slippage_stress_results.json"
    result.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(result.to_dict(orient="records"), indent=2), encoding="utf-8")

    print("Block Bootstrap + Slippage Stress")
    print(f"Blocks: {BLOCK_DAYS} trading days | Sims per setup: {N_SIMS}")
    print("Extra slippage stress: random 0, 1, or 2 additional roundtrip ticks per trade")
    print(
        f"{'Setup':<6} {'MNQ':>4} {'MGC':>4} {'Pass Prob':>12} "
        f"{'Avg Trades to Pass':>20} {'Avg Max DD (Winners)':>22}"
    )
    print("-" * 82)
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
        f"Spitzenposition unter Stress: Setup {winner['setup']} "
        f"({int(winner['mnq_contracts'])} MNQ / {int(winner['mgc_contracts'])} MGC)"
    )
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
