"""Reprice final validated strategy logs and analyze combined portfolio risk."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ACCOUNT_SIZE = 50_000.0
TRAILING_DRAWDOWN_LIMIT = 5_000.0
DAILY_LOSS_LIMIT = 2_500.0
COMMISSION_PER_SIDE = 1.10
SLIPPAGE_TICKS_PER_SIDE = 1.0
SESSION_TZ = "America/New_York"

DONCHIAN_LOG = Path("output/donchian_portfolio_long_trade_log.npy")
MCL_LOG = Path("output/vwap_mcl_trade_log_full.npy")
OUTPUT_DIR = Path("output/final_portfolio_backtest")
PROFIT_TARGET = 3_000.0


@dataclass(frozen=True)
class InstrumentSpec:
    asset: str
    tick_size: float
    tick_value: float

    @property
    def slippage_points(self) -> float:
        return self.tick_size * SLIPPAGE_TICKS_PER_SIDE


INSTRUMENT_SPECS: dict[str, InstrumentSpec] = {
    "MNQ": InstrumentSpec(asset="MNQ", tick_size=0.25, tick_value=0.50),
    "MGC": InstrumentSpec(asset="MGC", tick_size=0.10, tick_value=1.00),
    "MCL": InstrumentSpec(asset="MCL", tick_size=0.01, tick_value=1.00),
}


def _infer_epoch_unit(values: np.ndarray) -> str:
    max_abs = int(np.max(np.abs(values))) if len(values) else 0
    if max_abs >= 10**17:
        return "ns"
    if max_abs >= 10**14:
        return "us"
    if max_abs >= 10**11:
        return "ms"
    return "s"


def _to_ny_timestamp(values: np.ndarray) -> pd.DatetimeIndex:
    unit = _infer_epoch_unit(values.astype(np.int64))
    return pd.to_datetime(values.astype(np.int64), unit=unit, utc=True).tz_convert(SESSION_TZ)


def _load_log(path: Path, asset: str | None = None) -> pd.DataFrame:
    arr = np.load(path, allow_pickle=False)
    if arr.dtype.names is None:
        raise TypeError(f"{path} is not a structured trade log")
    df = pd.DataFrame.from_records(arr)
    if asset is not None:
        df["asset"] = asset
    elif "asset" not in df.columns:
        raise ValueError(f"{path} is missing asset labels")
    df["entry_dt"] = _to_ny_timestamp(df["entry_time"].to_numpy(dtype=np.int64))
    df["exit_dt"] = _to_ny_timestamp(df["exit_time"].to_numpy(dtype=np.int64))
    df["contracts"] = df["contracts"].astype(int)
    df["signal_type"] = df["signal_type"].astype(int)
    for col in (
        "entry_price",
        "exit_price",
        "entry_slippage",
        "exit_slippage",
        "entry_commission",
        "exit_commission",
        "gross_pnl",
        "net_pnl",
    ):
        df[col] = df[col].astype(float)
    return df


def _reprice_trade_log(df: pd.DataFrame) -> pd.DataFrame:
    repriced_parts: list[pd.DataFrame] = []
    for asset, asset_df in df.groupby("asset", sort=False):
        spec = INSTRUMENT_SPECS[asset]
        out = asset_df.copy()

        sign = out["signal_type"].to_numpy(dtype=np.int8)
        contracts = out["contracts"].to_numpy(dtype=np.int32).astype(np.float64)

        raw_entry = out["entry_price"].to_numpy(dtype=np.float64) - sign * out["entry_slippage"].to_numpy(dtype=np.float64)
        raw_exit = out["exit_price"].to_numpy(dtype=np.float64) + sign * out["exit_slippage"].to_numpy(dtype=np.float64)

        entry_price = raw_entry + sign * spec.slippage_points
        exit_price = raw_exit - sign * spec.slippage_points

        gross_pnl = np.where(
            sign > 0,
            (exit_price - entry_price) * contracts / spec.tick_size * spec.tick_value,
            (entry_price - exit_price) * contracts / spec.tick_size * spec.tick_value,
        )
        total_commission = 2.0 * COMMISSION_PER_SIDE * contracts
        net_pnl = gross_pnl - total_commission

        out["entry_price_repriced"] = entry_price
        out["exit_price_repriced"] = exit_price
        out["entry_slippage_repriced"] = spec.slippage_points
        out["exit_slippage_repriced"] = spec.slippage_points
        out["entry_commission_repriced"] = COMMISSION_PER_SIDE * contracts
        out["exit_commission_repriced"] = COMMISSION_PER_SIDE * contracts
        out["gross_pnl_repriced"] = gross_pnl
        out["net_pnl_repriced"] = net_pnl
        out["cost_delta_vs_logged"] = net_pnl - out["net_pnl"].to_numpy(dtype=np.float64)
        repriced_parts.append(out)

    combined = pd.concat(repriced_parts, axis=0, ignore_index=True)
    combined = combined.sort_values(["exit_dt", "entry_dt", "asset"], kind="stable").reset_index(drop=True)
    combined["exit_date"] = combined["exit_dt"].dt.normalize()
    return combined


def _scale_contracts(df: pd.DataFrame, contracts_multiplier: int) -> pd.DataFrame:
    if contracts_multiplier == 1:
        return df.copy()

    out = df.copy()
    scale_cols = [
        "contracts",
        "entry_commission_repriced",
        "exit_commission_repriced",
        "gross_pnl_repriced",
        "net_pnl_repriced",
        "cost_delta_vs_logged",
    ]
    for col in scale_cols:
        out[col] = out[col].astype(float) * contracts_multiplier
    out["contracts"] = out["contracts"].round().astype(int)
    return out


def _profit_factor(pnl: pd.Series) -> float:
    gross_profit = float(pnl[pnl > 0.0].sum())
    gross_loss = float(-pnl[pnl < 0.0].sum())
    return float(gross_profit / gross_loss) if gross_loss > 0.0 else float("inf")


def _asset_metrics(df: pd.DataFrame) -> dict[str, float | int]:
    pnl = df["net_pnl_repriced"].astype(float)
    return {
        "trades": int(len(df)),
        "win_rate": float((pnl > 0.0).mean()) if len(df) else 0.0,
        "profit_factor": _profit_factor(pnl),
        "net_profit": float(pnl.sum()),
    }


def _build_portfolio_curves(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | bool | str]]:
    trades = df.copy()
    trades["equity"] = ACCOUNT_SIZE + trades["net_pnl_repriced"].cumsum()
    trades["running_peak"] = trades["equity"].cummax()
    trades["drawdown"] = trades["running_peak"] - trades["equity"]

    daily = (
        trades.groupby("exit_date", sort=True)["net_pnl_repriced"]
        .sum()
        .rename("daily_pnl")
        .to_frame()
    )
    daily["equity"] = ACCOUNT_SIZE + daily["daily_pnl"].cumsum()
    daily["running_peak"] = daily["equity"].cummax()
    daily["drawdown"] = daily["running_peak"] - daily["equity"]

    worst_day_idx = daily["daily_pnl"].idxmin() if len(daily) else None
    worst_day_pnl = float(daily["daily_pnl"].min()) if len(daily) else 0.0
    max_dd = float(trades["drawdown"].max()) if len(trades) else 0.0
    trailing_breach = bool(max_dd > TRAILING_DRAWDOWN_LIMIT)
    daily_breach = bool((daily["daily_pnl"] <= -DAILY_LOSS_LIMIT).any()) if len(daily) else False

    metrics = {
        "total_trades": int(len(trades)),
        "win_rate": float((trades["net_pnl_repriced"] > 0.0).mean()) if len(trades) else 0.0,
        "profit_factor": _profit_factor(trades["net_pnl_repriced"]),
        "net_profit": float(trades["net_pnl_repriced"].sum()),
        "ending_equity": float(trades["equity"].iloc[-1]) if len(trades) else ACCOUNT_SIZE,
        "max_drawdown": max_dd,
        "max_daily_loss": abs(worst_day_pnl),
        "worst_day_pnl": worst_day_pnl,
        "worst_day": worst_day_idx.strftime("%Y-%m-%d") if worst_day_idx is not None else None,
        "daily_loss_breach": daily_breach,
        "trailing_drawdown_breach": trailing_breach,
        "passes_mff": not daily_breach and not trailing_breach,
    }
    return trades, daily, metrics


def _annual_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    yearly = trades.copy()
    yearly["year"] = yearly["exit_dt"].dt.year

    rows = []
    for year, year_df in yearly.groupby("year", sort=True):
        pnl = year_df["net_pnl_repriced"].astype(float)
        rows.append(
            {
                "year": int(year),
                "trades": int(len(year_df)),
                "win_rate": float((pnl > 0.0).mean()) if len(year_df) else 0.0,
                "profit_factor": _profit_factor(pnl),
                "net_profit": float(pnl.sum()),
            }
        )
    return pd.DataFrame(rows)


def _underwater_stats(daily: pd.DataFrame) -> dict[str, float | int | str | None]:
    if daily.empty:
        return {
            "longest_underwater_trading_days": 0,
            "longest_underwater_calendar_days": 0,
            "longest_underwater_start": None,
            "longest_underwater_end": None,
        }

    underwater = daily["equity"] < daily["running_peak"]
    longest_len = 0
    longest_start = None
    longest_end = None
    current_start = None
    current_len = 0

    for idx, is_underwater in zip(daily.index, underwater.to_numpy()):
        if is_underwater:
            if current_start is None:
                current_start = idx
                current_len = 1
            else:
                current_len += 1
            if current_len > longest_len:
                longest_len = current_len
                longest_start = current_start
                longest_end = idx
        else:
            current_start = None
            current_len = 0

    if longest_start is None or longest_end is None:
        return {
            "longest_underwater_trading_days": 0,
            "longest_underwater_calendar_days": 0,
            "longest_underwater_start": None,
            "longest_underwater_end": None,
        }

    calendar_days = int((longest_end.date() - longest_start.date()).days) + 1
    return {
        "longest_underwater_trading_days": int(longest_len),
        "longest_underwater_calendar_days": calendar_days,
        "longest_underwater_start": longest_start.strftime("%Y-%m-%d"),
        "longest_underwater_end": longest_end.strftime("%Y-%m-%d"),
    }


def _mff_timeline_stats(daily: pd.DataFrame, profit_target: float) -> dict[str, float | int | None]:
    if daily.empty:
        return {
            "successful_start_count": 0,
            "average_trading_days_to_target": None,
            "average_calendar_days_to_target": None,
            "average_months_to_target": None,
            "median_months_to_target": None,
        }

    dates = list(daily.index)
    pnl = daily["daily_pnl"].to_numpy(dtype=np.float64)
    trading_days_to_target = []
    calendar_days_to_target = []
    months_to_target = []

    for start_idx, start_date in enumerate(dates):
        future = np.cumsum(pnl[start_idx:])
        hit = np.where(future >= profit_target)[0]
        if len(hit) == 0:
            continue
        offset = int(hit[0])
        end_date = dates[start_idx + offset]
        trading_days = offset + 1
        calendar_days = int((end_date.date() - start_date.date()).days) + 1
        months = calendar_days / 30.4375
        trading_days_to_target.append(trading_days)
        calendar_days_to_target.append(calendar_days)
        months_to_target.append(months)

    if not months_to_target:
        return {
            "successful_start_count": 0,
            "average_trading_days_to_target": None,
            "average_calendar_days_to_target": None,
            "average_months_to_target": None,
            "median_months_to_target": None,
        }

    return {
        "successful_start_count": int(len(months_to_target)),
        "average_trading_days_to_target": float(np.mean(trading_days_to_target)),
        "average_calendar_days_to_target": float(np.mean(calendar_days_to_target)),
        "average_months_to_target": float(np.mean(months_to_target)),
        "median_months_to_target": float(np.median(months_to_target)),
    }


def _print_table_1(metrics_by_asset: dict[str, dict[str, float | int]], assets: list[str]) -> None:
    print("Tabelle 1: Individuelle Netto-Metriken nach Kosten")
    print(f"{'Asset':<6} {'Trades':>8} {'Win-Rate':>10} {'PF':>8} {'Net Profit':>14}")
    print("-" * 52)
    for asset in assets:
        row = metrics_by_asset[asset]
        print(
            f"{asset:<6} "
            f"{int(row['trades']):>8} "
            f"{float(row['win_rate']):>9.2%} "
            f"{float(row['profit_factor']):>8.2f} "
            f"${float(row['net_profit']):>13,.2f}"
        )


def _print_table_2(metrics: dict[str, float | bool | str], daily: pd.DataFrame) -> None:
    print()
    print("Tabelle 2: Portfolio Gesamt-Metriken")
    print(f"{'Startkapital':<28} ${ACCOUNT_SIZE:,.2f}")
    print(f"{'Net Profit':<28} ${float(metrics['net_profit']):,.2f}")
    print(f"{'End Equity':<28} ${float(metrics['ending_equity']):,.2f}")
    print(f"{'Total Trades':<28} {int(metrics['total_trades'])}")
    print(f"{'Win-Rate':<28} {float(metrics['win_rate']):.2%}")
    print(f"{'Profit Factor':<28} {float(metrics['profit_factor']):.2f}")
    print(f"{'Max Peak-to-Valley DD':<28} ${float(metrics['max_drawdown']):,.2f}")
    print(f"{'Worst Day':<28} {metrics['worst_day']} (${float(metrics['worst_day_pnl']):,.2f})")
    print(f"{'Max Daily Loss':<28} ${float(metrics['max_daily_loss']):,.2f}")
    print(f"{'Days <= -$2,500':<28} {int((daily['daily_pnl'] <= -DAILY_LOSS_LIMIT).sum())}")


def _print_annual_table(annual: pd.DataFrame) -> None:
    print()
    print("Jahres-Splits")
    print(f"{'Jahr':<6} {'Trades':>8} {'Win-Rate':>10} {'PF':>8} {'Net Profit':>14}")
    print("-" * 52)
    for row in annual.itertuples(index=False):
        print(
            f"{int(row.year):<6} "
            f"{int(row.trades):>8} "
            f"{float(row.win_rate):>9.2%} "
            f"{float(row.profit_factor):>8.2f} "
            f"${float(row.net_profit):>13,.2f}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assets",
        nargs="+",
        default=["MNQ", "MGC", "MCL"],
        choices=["MNQ", "MGC", "MCL"],
    )
    parser.add_argument("--contracts-multiplier", type=int, default=1)
    parser.add_argument("--profit-target", type=float, default=PROFIT_TARGET)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    donchian = _load_log(DONCHIAN_LOG)
    mcl = _load_log(MCL_LOG, asset="MCL")

    validated = pd.concat(
        [
            donchian.loc[donchian["asset"].isin(["MNQ", "MGC"])].copy(),
            mcl.copy(),
        ],
        axis=0,
        ignore_index=True,
    )
    repriced = _reprice_trade_log(validated)
    repriced = repriced.loc[repriced["asset"].isin(args.assets)].copy()
    repriced = _scale_contracts(repriced, args.contracts_multiplier)
    repriced = repriced.sort_values(["exit_dt", "entry_dt", "asset"], kind="stable").reset_index(drop=True)

    metrics_by_asset = {
        asset: _asset_metrics(repriced[repriced["asset"] == asset].copy())
        for asset in args.assets
    }
    trade_curve, daily_curve, portfolio_metrics = _build_portfolio_curves(repriced)
    annual = _annual_metrics(repriced)
    underwater = _underwater_stats(daily_curve)
    timeline = _mff_timeline_stats(daily_curve, args.profit_target)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trade_csv = args.output_dir / "repriced_trade_log.csv"
    daily_csv = args.output_dir / "portfolio_daily_curve.csv"
    equity_csv = args.output_dir / "portfolio_trade_curve.csv"
    report_json = args.output_dir / "final_portfolio_report.json"

    repriced.to_csv(trade_csv, index=False)
    daily_curve.to_csv(daily_csv, index_label="date")
    trade_curve.to_csv(equity_csv, index=False)

    report = {
        "cost_model": {
            "commission_per_side_per_contract": COMMISSION_PER_SIDE,
            "slippage_ticks_per_side": SLIPPAGE_TICKS_PER_SIDE,
            "contracts_multiplier": args.contracts_multiplier,
            "account_size": ACCOUNT_SIZE,
            "daily_loss_limit": DAILY_LOSS_LIMIT,
            "trailing_drawdown_limit": TRAILING_DRAWDOWN_LIMIT,
        },
        "assets": args.assets,
        "source_logs": {
            "mnq_mgc": str(DONCHIAN_LOG),
            "mcl": str(MCL_LOG),
        },
        "date_range": {
            "first_trade": repriced["entry_dt"].min().isoformat() if len(repriced) else None,
            "last_trade": repriced["exit_dt"].max().isoformat() if len(repriced) else None,
        },
        "asset_metrics": metrics_by_asset,
        "portfolio_metrics": portfolio_metrics,
        "annual_metrics": annual.to_dict(orient="records"),
        "underwater": underwater,
        "mff_timeline": timeline,
    }
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    _print_table_1(metrics_by_asset, args.assets)
    _print_table_2(portfolio_metrics, daily_curve)
    _print_annual_table(annual)
    print()
    print("Underwater-Phasen")
    print(
        f"  Laengste Stagnation: {int(underwater['longest_underwater_trading_days'])} Handelstage "
        f"({int(underwater['longest_underwater_calendar_days'])} Kalendertage) "
        f"von {underwater['longest_underwater_start']} bis {underwater['longest_underwater_end']}"
    )
    print()
    print("MFF-Timeline")
    if timeline["successful_start_count"]:
        print(
            f"  Durchschnitt bis $3,000 Profit: {float(timeline['average_months_to_target']):.2f} Monate "
            f"({float(timeline['average_trading_days_to_target']):.1f} Handelstage)"
        )
        print(f"  Median bis $3,000 Profit: {float(timeline['median_months_to_target']):.2f} Monate")
        print(f"  Erfolgreiche historische Startpunkte: {int(timeline['successful_start_count'])}")
    else:
        print("  Profitziel wurde historisch von keinem Startpunkt aus erreicht.")
    print()
    print("Kostenmodell")
    print(
        "  Die Kosten wurden realistisch pro Kontrakt umgerechnet: "
        "$1.10 je Seite plus 1 Tick je Seite."
    )
    print("  Mehrkontrakt-Trades skalieren die Kosten proportional mit der Kontraktzahl.")
    print()
    if bool(portfolio_metrics["passes_mff"]):
        print("Fazit: Das Portfolio hätte das $50k MFF-Profil historisch bestanden.")
    else:
        print("Fazit: Das Portfolio hätte das $50k MFF-Profil historisch gebrochen.")
    print(f"  Daily loss breach: {'YES' if portfolio_metrics['daily_loss_breach'] else 'NO'}")
    print(f"  Trailing drawdown breach: {'YES' if portfolio_metrics['trailing_drawdown_breach'] else 'NO'}")
    print()
    print(f"Trade log CSV: {trade_csv}")
    print(f"Daily equity CSV: {daily_csv}")
    print(f"Report JSON: {report_json}")


if __name__ == "__main__":
    main()
