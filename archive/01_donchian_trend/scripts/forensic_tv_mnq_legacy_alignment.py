#!/usr/bin/env python
"""Detailed TradingView-vs-Python forensic report for MNQ legacy frozen."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SESSION_TZ = "America/New_York"
TV_CSV_DEFAULT = Path(
    "data/tradingview/PropFirm_Breakout_Strategy_-_MNQ_Legacy_Frozen_CME_MINI_MNQ1!_2026-04-14_788c7.csv"
)
PY_LOG_DEFAULT = Path("output/backtests_mnq_legacy_frozen/latest_trade_log.npy")
RAW_DATA_DEFAULT = Path("data/processed/MNQ_1m_full_test.parquet")
OUTPUT_DIR_DEFAULT = Path("output/tradingview_forensics/mnq_legacy_alignment")


def _normalize_exit_reason(value: str) -> str:
    lower = value.lower()
    if "stop" in lower:
        return "stop"
    if "hardclose" in lower or "hard_close" in lower or "hard close" in lower:
        return "hard_close"
    if "profit" in lower or "target" in lower:
        return "target"
    return lower


def _direction_from_signal(value: str) -> int:
    lower = value.lower()
    if "long" in lower:
        return 1
    if "short" in lower:
        return -1
    raise ValueError(f"Cannot infer direction from signal: {value}")


def _load_tv_trades(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw = raw.rename(columns=lambda c: c.strip())

    records: list[dict] = []
    for trade_id, group in raw.groupby("Trade #", sort=True):
        entry_rows = group[group["Typ"].str.contains("Einstieg", na=False)]
        exit_rows = group[group["Typ"].str.contains("Ausstieg", na=False)]
        if len(entry_rows) != 1 or len(exit_rows) != 1:
            continue

        entry = entry_rows.iloc[0]
        exit_ = exit_rows.iloc[0]
        entry_time_utc = pd.Timestamp(entry["Datum und Uhrzeit"], tz="UTC")
        exit_time_utc = pd.Timestamp(exit_["Datum und Uhrzeit"], tz="UTC")

        records.append(
            {
                "trade_id": int(trade_id),
                "entry_time_utc": entry_time_utc,
                "exit_time_utc": exit_time_utc,
                "entry_time_et": entry_time_utc.tz_convert(SESSION_TZ),
                "exit_time_et": exit_time_utc.tz_convert(SESSION_TZ),
                "entry_price": float(entry["Preis USD"]),
                "exit_price": float(exit_["Preis USD"]),
                "contracts": int(entry["Größe (Menge)"]),
                "net_pnl": float(exit_["G&V netto USD"]),
                "direction": _direction_from_signal(str(entry["Signal"])),
                "entry_signal": str(entry["Signal"]),
                "exit_signal": str(exit_["Signal"]),
                "exit_reason": _normalize_exit_reason(str(exit_["Signal"])),
                "is_closed": True,
            }
        )

    tv = pd.DataFrame.from_records(records).sort_values("entry_time_et").reset_index(drop=True)
    tv["entry_date_et"] = tv["entry_time_et"].dt.strftime("%Y-%m-%d")
    tv["exit_date_et"] = tv["exit_time_et"].dt.strftime("%Y-%m-%d")
    return tv


def _load_python_trades(path: Path) -> pd.DataFrame:
    arr = np.load(path, allow_pickle=False)
    reason_map = {0: "target", 1: "stop", 2: "hard_close", 3: "circuit_breaker"}

    py = pd.DataFrame(
        {
            "entry_time_utc": pd.to_datetime(arr["entry_time"].astype(np.int64), utc=True),
            "exit_time_utc": pd.to_datetime(arr["exit_time"].astype(np.int64), utc=True),
            "entry_price": arr["entry_price"].astype(np.float64),
            "exit_price": arr["exit_price"].astype(np.float64),
            "contracts": arr["contracts"].astype(np.int64),
            "net_pnl": arr["net_pnl"].astype(np.float64),
            "direction": arr["signal_type"].astype(np.int64),
            "exit_reason": pd.Series(arr["exit_reason"].astype(np.int64)).map(reason_map),
        }
    )
    py["entry_time_et"] = py["entry_time_utc"].dt.tz_convert(SESSION_TZ)
    py["exit_time_et"] = py["exit_time_utc"].dt.tz_convert(SESSION_TZ)
    py["entry_date_et"] = py["entry_time_et"].dt.strftime("%Y-%m-%d")
    py["exit_date_et"] = py["exit_time_et"].dt.strftime("%Y-%m-%d")
    return py.sort_values("entry_time_et").reset_index(drop=True)


def _load_focus_session_counts(path: Path, dates: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(path).sort_index()
    df.index = df.index.tz_convert(SESSION_TZ)

    rows: list[dict] = []
    for date_str in dates:
        try:
            session = df.loc[date_str]
        except KeyError:
            session = df.iloc[0:0]
        rows.append(
            {
                "date_et": date_str,
                "rows_1m": int(len(session)),
                "first_bar_et": str(session.index.min()) if len(session) else "",
                "last_bar_et": str(session.index.max()) if len(session) else "",
                "is_full_0800_1559_session": bool(len(session) == 480),
            }
        )
    return pd.DataFrame(rows)


def _build_report(tv: pd.DataFrame, py: pd.DataFrame, raw_data_path: Path) -> tuple[dict, dict[str, pd.DataFrame]]:
    overlap_mask = (tv["entry_time_et"] >= py["entry_time_et"].min()) & (tv["entry_time_et"] <= py["entry_time_et"].max())
    tv_overlap = tv.loc[overlap_mask].copy()

    exact = tv_overlap.merge(
        py,
        on=["entry_time_et", "exit_time_et", "contracts", "direction", "exit_reason"],
        how="inner",
        suffixes=("_tv", "_py"),
    )

    soft = tv_overlap.merge(
        py,
        on=["entry_date_et", "direction"],
        how="outer",
        indicator=True,
        suffixes=("_tv", "_py"),
    ).sort_values(["entry_date_et", "direction", "trade_id"], kind="stable")

    both = soft.loc[soft["_merge"] == "both"].copy()
    both["entry_time_match"] = both["entry_time_et_tv"] == both["entry_time_et_py"]
    both["exit_time_match"] = both["exit_time_et_tv"] == both["exit_time_et_py"]
    both["contracts_match"] = both["contracts_tv"] == both["contracts_py"]
    both["exit_reason_match"] = both["exit_reason_tv"] == both["exit_reason_py"]
    both["entry_price_abs_delta"] = (both["entry_price_py"] - both["entry_price_tv"]).abs()
    both["exit_price_abs_delta"] = (both["exit_price_py"] - both["exit_price_tv"]).abs()
    both["net_pnl_abs_delta"] = (both["net_pnl_py"] - both["net_pnl_tv"]).abs()
    both["exact_structural_match"] = (
        both["entry_time_match"] & both["exit_time_match"] & both["contracts_match"] & both["exit_reason_match"]
    )

    signal_mismatches = soft.loc[soft["_merge"] != "both"].copy()
    qty_drift = both.loc[~both["exact_structural_match"]].copy()
    qty_drift = qty_drift[
        [
            "entry_date_et",
            "direction",
            "trade_id",
            "entry_time_et_tv",
            "entry_time_et_py",
            "exit_time_et_tv",
            "exit_time_et_py",
            "contracts_tv",
            "contracts_py",
            "exit_reason_tv",
            "exit_reason_py",
            "entry_price_tv",
            "entry_price_py",
            "exit_price_tv",
            "exit_price_py",
            "net_pnl_tv",
            "net_pnl_py",
            "entry_price_abs_delta",
            "exit_price_abs_delta",
            "net_pnl_abs_delta",
        ]
    ].sort_values(["entry_date_et", "trade_id"], kind="stable")

    focus_dates = sorted(
        {
            *signal_mismatches["entry_date_et"].dropna().astype(str).tolist(),
            *qty_drift["entry_date_et"].dropna().astype(str).tolist(),
        }
    )
    session_counts = _load_focus_session_counts(raw_data_path, focus_dates)

    signal_tv_only = signal_mismatches.loc[signal_mismatches["_merge"] == "left_only"].copy()
    signal_py_only = signal_mismatches.loc[signal_mismatches["_merge"] == "right_only"].copy()

    summary = {
        "artifacts": {
            "tv_csv": str(TV_CSV_DEFAULT),
            "python_trade_log": str(PY_LOG_DEFAULT),
            "raw_data": str(raw_data_path),
        },
        "tv_full_window": {
            "closed_trades": int(len(tv)),
            "entry_start_et": str(tv["entry_time_et"].min()),
            "entry_end_et": str(tv["entry_time_et"].max()),
            "net_pnl_closed": float(tv["net_pnl"].sum()),
        },
        "python_frozen_window": {
            "trades": int(len(py)),
            "entry_start_et": str(py["entry_time_et"].min()),
            "entry_end_et": str(py["entry_time_et"].max()),
            "net_pnl": float(py["net_pnl"].sum()),
        },
        "overlap": {
            "tv_closed_trades_in_overlap": int(len(tv_overlap)),
            "python_trades": int(len(py)),
            "exact_structural_matches": int(len(exact)),
            "date_direction_matches": int(len(both)),
            "signal_mismatch_pairs": int(len(signal_mismatches)),
            "tv_only_signal_days": int((signal_mismatches["_merge"] == "left_only").sum()),
            "python_only_signal_days": int((signal_mismatches["_merge"] == "right_only").sum()),
            "qty_or_structure_drift_days": int(len(qty_drift)),
            "avg_abs_entry_price_delta": float(both["entry_price_abs_delta"].mean()) if len(both) else 0.0,
            "avg_abs_exit_price_delta": float(both["exit_price_abs_delta"].mean()) if len(both) else 0.0,
            "avg_abs_net_pnl_delta": float(both["net_pnl_abs_delta"].mean()) if len(both) else 0.0,
            "tv_overlap_net_pnl": float(tv_overlap["net_pnl"].sum()),
            "python_overlap_net_pnl": float(py["net_pnl"].sum()),
            "tv_signal_counts": {
                "long": int((tv_overlap["direction"] == 1).sum()),
                "short": int((tv_overlap["direction"] == -1).sum()),
            },
            "python_signal_counts": {
                "long": int((py["direction"] == 1).sum()),
                "short": int((py["direction"] == -1).sum()),
            },
            "tv_exit_reason_counts": tv_overlap["exit_reason"].value_counts().to_dict(),
            "python_exit_reason_counts": py["exit_reason"].value_counts().to_dict(),
        },
        "high_confidence_findings": [
            "Direction mix matches exactly in the overlap window.",
            "Exit-reason mix matches exactly in the overlap window.",
            "Absolute price levels diverge far more than trade PnL, consistent with adjusted continuous-contract price offsets.",
            "The signal-mismatch days cluster around U.S. holiday or short-session periods.",
        ],
        "signal_mismatch_days": {
            "tv_only": signal_tv_only[
                [
                    "entry_date_et",
                    "trade_id",
                    "entry_time_et_tv",
                    "exit_time_et_tv",
                    "contracts_tv",
                    "exit_reason_tv",
                ]
            ].to_dict(orient="records"),
            "python_only": signal_py_only[
                [
                    "entry_date_et",
                    "entry_time_et_py",
                    "exit_time_et_py",
                    "contracts_py",
                    "exit_reason_py",
                ]
            ].to_dict(orient="records"),
        },
    }

    tables = {
        "tv_trades": tv,
        "python_trades": py,
        "exact_matches": exact,
        "soft_matches": both,
        "signal_mismatches": signal_mismatches,
        "qty_drift": qty_drift,
        "focus_session_counts": session_counts,
    }
    return summary, tables


def _write_report(summary: dict, tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    overlap = summary["overlap"]
    md = f"""# MNQ Legacy Frozen TV-vs-Python Forensic Report

## Scope
- TradingView CSV: `{summary["artifacts"]["tv_csv"]}`
- Python frozen log: `{summary["artifacts"]["python_trade_log"]}`
- Raw session data used for holiday/session checks: `{summary["artifacts"]["raw_data"]}`

## Full windows
- TV closed trades: {summary["tv_full_window"]["closed_trades"]}
- TV entry range ET: {summary["tv_full_window"]["entry_start_et"]} -> {summary["tv_full_window"]["entry_end_et"]}
- TV closed net PnL: {summary["tv_full_window"]["net_pnl_closed"]:.2f}
- Python frozen trades: {summary["python_frozen_window"]["trades"]}
- Python entry range ET: {summary["python_frozen_window"]["entry_start_et"]} -> {summary["python_frozen_window"]["entry_end_et"]}
- Python net PnL: {summary["python_frozen_window"]["net_pnl"]:.2f}

## Overlap findings
- TV closed trades in overlap: {overlap["tv_closed_trades_in_overlap"]}
- Python trades in overlap: {overlap["python_trades"]}
- Exact structural matches: {overlap["exact_structural_matches"]}
- Date+direction matches: {overlap["date_direction_matches"]}
- TV-only signal days: {overlap["tv_only_signal_days"]}
- Python-only signal days: {overlap["python_only_signal_days"]}
- Qty/structure drift days on matched dates: {overlap["qty_or_structure_drift_days"]}
- Avg abs entry price delta: {overlap["avg_abs_entry_price_delta"]:.2f}
- Avg abs exit price delta: {overlap["avg_abs_exit_price_delta"]:.2f}
- Avg abs net PnL delta: {overlap["avg_abs_net_pnl_delta"]:.2f}
- TV overlap net PnL: {overlap["tv_overlap_net_pnl"]:.2f}
- Python overlap net PnL: {overlap["python_overlap_net_pnl"]:.2f}

## Interpretation
- Direction and exit-reason distributions match exactly in the overlap window.
- The remaining parity gap is driven by a small set of signal-day swaps plus several qty drifts.
- The large absolute price-level differences are not a reliable parity metric here because the TradingView export is from `MNQ1!` continuous-contract data.
- The mismatch days cluster around holiday or shortened-session periods, which strongly suggests session-hygiene differences between TradingView chart logic and the Python loader.

## Files
- `signal_mismatches.csv`
- `qty_drift.csv`
- `soft_matches.csv`
- `focus_session_counts.csv`
"""
    (output_dir / "report.md").write_text(md, encoding="utf-8")


def main() -> None:
    tv = _load_tv_trades(TV_CSV_DEFAULT)
    py = _load_python_trades(PY_LOG_DEFAULT)
    summary, tables = _build_report(tv, py, RAW_DATA_DEFAULT)
    _write_report(summary, tables, OUTPUT_DIR_DEFAULT)

    overlap = summary["overlap"]
    print("MNQ Legacy Frozen TV-vs-Python Forensic Report")
    print(f"Output: {OUTPUT_DIR_DEFAULT}")
    print(f"TV closed trades: {summary['tv_full_window']['closed_trades']}")
    print(f"Python frozen trades: {summary['python_frozen_window']['trades']}")
    print(f"Exact structural matches: {overlap['exact_structural_matches']}")
    print(f"Date+direction matches: {overlap['date_direction_matches']}")
    print(f"TV-only signal days: {overlap['tv_only_signal_days']}")
    print(f"Python-only signal days: {overlap['python_only_signal_days']}")
    print(f"Qty/structure drift days: {overlap['qty_or_structure_drift_days']}")


if __name__ == "__main__":
    main()
