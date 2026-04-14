"""Grid-search asymmetric MNQ/MGC sizing against MFF safety buffers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.final_portfolio_backtest import (
    DONCHIAN_LOG,
    MCL_LOG,
    _build_portfolio_curves,
    _load_log,
    _mff_timeline_stats,
    _reprice_trade_log,
    _scale_contracts,
)


MNQ_GRID = (1, 2, 3)
MGC_GRID = (1, 2, 3, 4)
MAX_DRAWDOWN_BUFFER_LIMIT = 4_500.0
MAX_DAILY_LOSS_BUFFER_LIMIT = 2_000.0
PROFIT_TARGET = 3_000.0
OUTPUT_DIR = Path("output/asymmetrical_sizing_matrix")


def _build_base_repriced() -> pd.DataFrame:
    donchian = _load_log(DONCHIAN_LOG)
    _ = MCL_LOG  # Explicitly ignored for this matrix.
    base = donchian.loc[donchian["asset"].isin(["MNQ", "MGC"])].copy()
    repriced = _reprice_trade_log(base)
    return repriced.sort_values(["exit_dt", "entry_dt", "asset"], kind="stable").reset_index(drop=True)


def _apply_combo(base: pd.DataFrame, mnq_contracts: int, mgc_contracts: int) -> pd.DataFrame:
    parts = []
    for asset, contracts in (("MNQ", mnq_contracts), ("MGC", mgc_contracts)):
        asset_df = base.loc[base["asset"] == asset].copy()
        parts.append(_scale_contracts(asset_df, contracts))
    combined = pd.concat(parts, axis=0, ignore_index=True)
    return combined.sort_values(["exit_dt", "entry_dt", "asset"], kind="stable").reset_index(drop=True)


def main() -> None:
    base = _build_base_repriced()
    rows: list[dict[str, float | int | bool]] = []

    for mnq_contracts in MNQ_GRID:
        for mgc_contracts in MGC_GRID:
            combo = _apply_combo(base, mnq_contracts, mgc_contracts)
            _, daily_curve, metrics = _build_portfolio_curves(combo)
            timeline = _mff_timeline_stats(daily_curve, PROFIT_TARGET)

            row = {
                "mnq_contracts": mnq_contracts,
                "mgc_contracts": mgc_contracts,
                "net_profit": float(metrics["net_profit"]),
                "max_drawdown": float(metrics["max_drawdown"]),
                "max_daily_loss": float(metrics["max_daily_loss"]),
                "days_below_2000": int((daily_curve["daily_pnl"] <= -MAX_DAILY_LOSS_BUFFER_LIMIT).sum()),
                "safe_drawdown": float(metrics["max_drawdown"]) <= MAX_DRAWDOWN_BUFFER_LIMIT,
                "safe_daily": float(metrics["max_daily_loss"]) <= MAX_DAILY_LOSS_BUFFER_LIMIT,
                "passes_buffer": (
                    float(metrics["max_drawdown"]) <= MAX_DRAWDOWN_BUFFER_LIMIT
                    and float(metrics["max_daily_loss"]) <= MAX_DAILY_LOSS_BUFFER_LIMIT
                ),
                "avg_months_to_3000": timeline["average_months_to_target"],
                "median_months_to_3000": timeline["median_months_to_target"],
                "successful_start_count": int(timeline["successful_start_count"]),
            }
            rows.append(row)

    result = pd.DataFrame(rows).sort_values(
        ["passes_buffer", "net_profit", "mnq_contracts", "mgc_contracts"],
        ascending=[False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)

    safe = result.loc[result["passes_buffer"]].copy()
    top3 = safe.nlargest(3, "net_profit").reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "sizing_matrix.csv"
    json_path = OUTPUT_DIR / "sizing_matrix_report.json"
    result.to_csv(csv_path, index=False)

    report = {
        "grid": {
            "mnq_contracts": list(MNQ_GRID),
            "mgc_contracts": list(MGC_GRID),
        },
        "safety_filter": {
            "max_drawdown_limit": MAX_DRAWDOWN_BUFFER_LIMIT,
            "max_daily_loss_limit": MAX_DAILY_LOSS_BUFFER_LIMIT,
        },
        "safe_combinations": safe.to_dict(orient="records"),
        "top_3": top3.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Asymmetrical Sizing Matrix")
    print(f"Grid: MNQ {list(MNQ_GRID)} x MGC {list(MGC_GRID)}")
    print(
        f"Safety filter: max drawdown <= ${MAX_DRAWDOWN_BUFFER_LIMIT:,.0f} | "
        f"max daily loss <= ${MAX_DAILY_LOSS_BUFFER_LIMIT:,.0f}"
    )
    print()

    if top3.empty:
        print("Keine Kombination besteht den Sicherheitsfilter.")
    else:
        print("Top 3 sichere Kombinationen")
        print(
            f"{'Rank':<6} {'MNQ':>4} {'MGC':>4} {'Net Profit':>14} "
            f"{'Max DD':>12} {'Max Day Loss':>14} {'Avg Months to $3k':>18}"
        )
        print("-" * 78)
        for idx, row in enumerate(top3.itertuples(index=False), start=1):
            avg_months = row.avg_months_to_3000
            avg_months_text = f"{avg_months:.2f}" if pd.notna(avg_months) else "n/a"
            print(
                f"{idx:<6} "
                f"{int(row.mnq_contracts):>4} "
                f"{int(row.mgc_contracts):>4} "
                f"${float(row.net_profit):>13,.2f} "
                f"${float(row.max_drawdown):>11,.2f} "
                f"${float(row.max_daily_loss):>13,.2f} "
                f"{avg_months_text:>18}"
            )

        winner = top3.iloc[0]
        print()
        print("Gewinner-Kombination")
        print(f"  MNQ / MGC: {int(winner['mnq_contracts'])} / {int(winner['mgc_contracts'])}")
        print(f"  Net Profit: ${float(winner['net_profit']):,.2f}")
        print(f"  Max Drawdown: ${float(winner['max_drawdown']):,.2f}")
        print(f"  Max Daily Loss: ${float(winner['max_daily_loss']):,.2f}")
        if pd.notna(winner["avg_months_to_3000"]):
            print(f"  Durchschnitt bis $3,000: {float(winner['avg_months_to_3000']):.2f} Monate")
        else:
            print("  Durchschnitt bis $3,000: n/a")

    print()
    print(f"Matrix CSV: {csv_path}")
    print(f"Report JSON: {json_path}")


if __name__ == "__main__":
    main()
