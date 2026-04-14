"""Run a fresh canonical MGC+MNQ portfolio validation pipeline.

This script is the single entry point for:
1. building a fresh deterministic portfolio OOS artifact set
2. repricing the resulting trade log under the production cost model
3. running a fresh asymmetrical sizing matrix on that repriced base
4. running a fresh block-bootstrap slippage stress Monte Carlo on that same base
5. archiving stale legacy output artifacts into a dated archive folder

The goal is to keep one canonical output structure and stop mixing historical
artifact generations when evaluating the strategy.
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from propfirm.core.multi_engine import run_multi_asset_day_kernel
from propfirm.core.types import EXIT_CIRCUIT_BREAKER, TRADE_LOG_DTYPE
from scripts.export_donchian_portfolio_oos import (
    DEFAULT_MGC_DATA,
    DEFAULT_MGC_MFF_CONFIG,
    DEFAULT_MGC_PARAMS_CONFIG,
    DEFAULT_MNQ_DATA,
    DEFAULT_MNQ_MFF_CONFIG,
    DEFAULT_MNQ_PARAMS_CONFIG,
    PORTFOLIO_DAILY_LOG_DTYPE,
    PORTFOLIO_TRADE_LOG_DTYPE,
    _build_asset_setup,
    _compute_raw_metrics,
    _slice_asset_config,
)
from scripts.final_portfolio_backtest import (
    INSTRUMENT_SPECS,
    _build_portfolio_curves,
    _load_log,
    _mff_timeline_stats,
    _reprice_trade_log,
    _scale_contracts,
)


ROOT = Path("output/canonical_portfolio_validation")
RUNS_DIR = ROOT / "runs"
LATEST_DIR = ROOT / "latest"
ARCHIVE_ROOT = Path("output/archive")

PROFIT_TARGET = 3_000.0
TRAILING_DRAWDOWN_LIMIT = 5_000.0
BLOCK_DAYS = 5
MAX_BLOCKS_PER_PATH = 500
N_SIMS = 10_000


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


LEGACY_OUTPUT_DIRS = (
    Path("output/asymmetrical_sizing_matrix"),
    Path("output/donchian_portfolio_oos_fleet"),
    Path("output/donchian_portfolio_oos_fleet_10y"),
    Path("output/donchian_portfolio_oos_fleet_v2"),
    Path("output/fleet_simulator"),
    Path("output/fleet_simulator_oos"),
    Path("output/fleet_simulator_smoke"),
    Path("output/final_portfolio_backtest"),
    Path("output/final_portfolio_backtest_mnq_mgc_3x"),
    Path("output/monte_carlo_asymmetrical_setups"),
    Path("output/monte_carlo_block_slippage_stress"),
    Path("output/tmp_tv_parity_check"),
)

LEGACY_OUTPUT_FILES = (
    Path("output/donchian_portfolio_long_daily_log.npy"),
    Path("output/donchian_portfolio_long_pnls.npy"),
    Path("output/donchian_portfolio_long_report.json"),
    Path("output/donchian_portfolio_long_trade_log.npy"),
    Path("output/donchian_portfolio_oos_daily_log.npy"),
    Path("output/donchian_portfolio_oos_pnls.npy"),
    Path("output/donchian_portfolio_oos_report.json"),
    Path("output/donchian_portfolio_oos_trade_log.npy"),
    Path("output/portfolio_equity_curve.csv"),
    Path("output/portfolio_equity_curve_long.csv"),
)


def _now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _write_readme(root: Path) -> None:
    readme = root / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Canonical Portfolio Validation",
                "",
                "Use this folder as the single source of current Python validation artifacts.",
                "",
                "Structure:",
                "- `latest/portfolio`: fresh deterministic MGC+MNQ OOS trade log and report",
                "- `latest/repriced`: repriced MNQ+MGC base trade log under the cost model",
                "- `latest/sizing`: fresh asymmetrical sizing matrix on the repriced base",
                "- `latest/monte_carlo`: fresh block-bootstrap slippage stress on the repriced base",
                "- `runs/<timestamp>/...`: immutable snapshots of each refresh run",
                "",
                "Policy:",
                "- Python remains the research and backtest source of truth.",
                "- TradingView should be evaluated against these canonical Python outputs, not legacy `output/*` files.",
                "- Legacy outputs are archived under `output/archive/`.",
            ]
        ),
        encoding="utf-8",
    )


def _archive_legacy_outputs(run_stamp: str) -> dict[str, list[str]]:
    archive_dir = ARCHIVE_ROOT / f"{run_stamp}_legacy_outputs"
    archive_dir.mkdir(parents=True, exist_ok=True)

    moved_dirs: list[str] = []
    moved_files: list[str] = []

    for path in LEGACY_OUTPUT_DIRS:
        if path.exists():
            target = archive_dir / path.name
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(path), str(target))
            moved_dirs.append(str(path))

    for path in LEGACY_OUTPUT_FILES:
        if path.exists():
            target = archive_dir / path.name
            if target.exists():
                target.unlink()
            shutil.move(str(path), str(target))
            moved_files.append(str(path))

    return {
        "archive_dir": [str(archive_dir)],
        "moved_dirs": moved_dirs,
        "moved_files": moved_files,
    }


def _compute_sanity_metrics(trade_log: np.ndarray) -> dict[str, float | int]:
    total_trades = int(len(trade_log))
    if total_trades == 0:
        return {
            "total_trades": 0,
            "average_trade_pnl": 0.0,
            "win_rate": 0.0,
        }

    net_pnl = trade_log["net_pnl"].astype(np.float64)
    return {
        "total_trades": total_trades,
        "average_trade_pnl": float(np.mean(net_pnl)),
        "win_rate": float(np.mean(net_pnl > 0.0)),
    }


def _build_fresh_portfolio_artifacts(output_dir: Path) -> dict[str, str | dict]:
    mgc = _build_asset_setup(
        "MGC",
        DEFAULT_MGC_DATA,
        DEFAULT_MGC_MFF_CONFIG,
        DEFAULT_MGC_PARAMS_CONFIG,
        60,
    )
    mnq = _build_asset_setup(
        "MNQ",
        DEFAULT_MNQ_DATA,
        DEFAULT_MNQ_MFF_CONFIG,
        DEFAULT_MNQ_PARAMS_CONFIG,
        30,
    )

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
            circuit_breaker_threshold=-800.0,
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
        day_cb_exits = int(
            np.sum(mgc_log["exit_reason"] == EXIT_CIRCUIT_BREAKER)
            + np.sum(mnq_log["exit_reason"] == EXIT_CIRCUIT_BREAKER)
        )
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
    portfolio_pnls = portfolio_trade_log["net_pnl"].astype(np.float64)
    raw_metrics = _compute_raw_metrics(portfolio_trade_log)
    sanity_metrics = _compute_sanity_metrics(portfolio_trade_log)
    combined_max_drawdown = (
        float(portfolio_daily["portfolio_drawdown"].min()) if len(portfolio_daily) else 0.0
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    trade_log_path = output_dir / "portfolio_oos_trade_log.npy"
    daily_log_path = output_dir / "portfolio_oos_daily_log.npy"
    pnl_path = output_dir / "portfolio_oos_pnls.npy"
    report_path = output_dir / "portfolio_oos_report.json"

    np.save(trade_log_path, portfolio_trade_log)
    np.save(daily_log_path, portfolio_daily)
    np.save(pnl_path, portfolio_pnls)

    report = {
        "portfolio": {
            "assets": ["MGC", "MNQ"],
            "data": {
                "mgc": str(DEFAULT_MGC_DATA),
                "mnq": str(DEFAULT_MNQ_DATA),
            },
            "session_dates": {
                "start": master_calendar[0] if master_calendar else None,
                "end": master_calendar[-1] if master_calendar else None,
                "count": len(master_calendar),
            },
            "metrics": {
                "combined_final_net_equity": raw_metrics["final_equity"],
                "combined_max_drawdown": combined_max_drawdown,
                "combined_win_rate": raw_metrics["win_rate"],
                "profit_factor": raw_metrics["profit_factor"],
                "total_trades": raw_metrics["total_trades"],
                "average_trade_pnl": sanity_metrics["average_trade_pnl"],
                "circuit_breaker_trigger_days": circuit_breaker_days,
            },
            "artifacts": {
                "trade_log": str(trade_log_path),
                "daily_log": str(daily_log_path),
                "trade_pnls": str(pnl_path),
            },
        }
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return {
        "trade_log_path": str(trade_log_path),
        "daily_log_path": str(daily_log_path),
        "pnl_path": str(pnl_path),
        "report_path": str(report_path),
        "report": report,
    }


def _build_repriced_base(trade_log_path: Path, output_dir: Path) -> pd.DataFrame:
    base = _load_log(trade_log_path)
    base = base.loc[base["asset"].isin(["MNQ", "MGC"])].copy()
    repriced = _reprice_trade_log(base)
    repriced = repriced.sort_values(["exit_dt", "entry_dt", "asset"], kind="stable").reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    repriced.to_csv(output_dir / "repriced_base_trade_log.csv", index=False)
    return repriced


def _apply_combo(base: pd.DataFrame, mnq_contracts: int, mgc_contracts: int) -> pd.DataFrame:
    parts = []
    for asset, contracts in (("MNQ", mnq_contracts), ("MGC", mgc_contracts)):
        asset_df = base.loc[base["asset"] == asset].copy()
        parts.append(_scale_contracts(asset_df, contracts))
    combined = pd.concat(parts, axis=0, ignore_index=True)
    return combined.sort_values(["exit_dt", "entry_dt", "asset"], kind="stable").reset_index(drop=True)


def _build_sizing_matrix(base: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    rows: list[dict[str, float | int | bool | None]] = []
    for mnq_contracts in (1, 2, 3):
        for mgc_contracts in (1, 2, 3, 4):
            combo = _apply_combo(base, mnq_contracts, mgc_contracts)
            _, daily_curve, metrics = _build_portfolio_curves(combo)
            timeline = _mff_timeline_stats(daily_curve, PROFIT_TARGET)
            rows.append(
                {
                    "mnq_contracts": mnq_contracts,
                    "mgc_contracts": mgc_contracts,
                    "net_profit": float(metrics["net_profit"]),
                    "max_drawdown": float(metrics["max_drawdown"]),
                    "max_daily_loss": float(metrics["max_daily_loss"]),
                    "passes_buffer": (
                        float(metrics["max_drawdown"]) <= 4_500.0
                        and float(metrics["max_daily_loss"]) <= 2_000.0
                    ),
                    "avg_months_to_3000": timeline["average_months_to_target"],
                    "median_months_to_3000": timeline["median_months_to_target"],
                    "successful_start_count": int(timeline["successful_start_count"]),
                }
            )

    result = pd.DataFrame(rows).sort_values(
        ["passes_buffer", "net_profit", "mnq_contracts", "mgc_contracts"],
        ascending=[False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sizing_matrix.csv"
    json_path = output_dir / "sizing_matrix_report.json"
    result.to_csv(csv_path, index=False)
    report = {
        "safe_combinations": result.loc[result["passes_buffer"]].to_dict(orient="records"),
        "all_combinations": result.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"csv_path": str(csv_path), "json_path": str(json_path)}


def _build_blocks(trades: pd.DataFrame, block_days: int) -> list[pd.DataFrame]:
    unique_dates = sorted(trades["exit_date"].drop_duplicates().tolist())
    if len(unique_dates) < block_days:
        return [trades.copy()]
    blocks: list[pd.DataFrame] = []
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
        drawdown = 0.0

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


def _build_monte_carlo(base: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    rows: list[dict[str, float | int | None | str]] = []
    for idx, setup in enumerate(SETUPS):
        combo = _apply_combo(base, setup.mnq_contracts, setup.mgc_contracts)
        combo["trade_cost_per_tick"] = combo.apply(
            lambda row: INSTRUMENT_SPECS[str(row["asset"])].tick_value * int(row["contracts"]),
            axis=1,
        )
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

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "block_slippage_stress_results.csv"
    json_path = output_dir / "block_slippage_stress_results.json"
    result.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(result.to_dict(orient="records"), indent=2), encoding="utf-8")
    return {"csv_path": str(csv_path), "json_path": str(json_path)}


def _copy_run_to_latest(run_dir: Path) -> None:
    if LATEST_DIR.exists():
        shutil.rmtree(LATEST_DIR)
    shutil.copytree(run_dir, LATEST_DIR)


def main() -> None:
    run_stamp = _now_stamp()
    run_dir = RUNS_DIR / run_stamp
    portfolio_dir = run_dir / "portfolio"
    repriced_dir = run_dir / "repriced"
    sizing_dir = run_dir / "sizing"
    monte_carlo_dir = run_dir / "monte_carlo"

    ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    _write_readme(ROOT)

    archive_manifest = _archive_legacy_outputs(run_stamp)
    fresh_portfolio = _build_fresh_portfolio_artifacts(portfolio_dir)
    repriced_base = _build_repriced_base(Path(fresh_portfolio["trade_log_path"]), repriced_dir)
    sizing_outputs = _build_sizing_matrix(repriced_base, sizing_dir)
    monte_carlo_outputs = _build_monte_carlo(repriced_base, monte_carlo_dir)

    manifest = {
        "run_stamp": run_stamp,
        "portfolio": fresh_portfolio,
        "repriced_base_trade_log": str(repriced_dir / "repriced_base_trade_log.csv"),
        "sizing": sizing_outputs,
        "monte_carlo": monte_carlo_outputs,
        "archived_legacy_outputs": archive_manifest,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _copy_run_to_latest(run_dir)

    print(f"Canonical run: {run_dir}")
    print(f"Latest alias: {LATEST_DIR}")
    print(f"Portfolio report: {fresh_portfolio['report_path']}")
    print(f"Sizing report: {sizing_outputs['json_path']}")
    print(f"Monte Carlo report: {monte_carlo_outputs['json_path']}")
    print(f"Archived legacy outputs under: {archive_manifest['archive_dir'][0]}")


if __name__ == "__main__":
    main()
