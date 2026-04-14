#!/usr/bin/env python
"""Forensic reconciliation between NT8 execution log and Python Donchian engine trade log."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NT8_DEFAULT = Path("data/raw/ninjatrader_backtest_log/NT8_MNQ_Executions.csv")
PYTHON_DEFAULTS = [
    Path("output/backtests_mnq_tf30_regime_val/latest_trade_log.npy"),
    Path("output/backtests_mnq_tf30_regime_test/latest_trade_log.npy"),
]
SESSION_TZ = "America/New_York"
MATCH_TOLERANCE = pd.Timedelta(minutes=1)
TICK_SIZE = 0.25  # MNQ tick


@dataclass
class RoundTrip:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    direction: int  # +1 long, -1 short
    exit_reason: str
    source: str


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_nt8(path: Path) -> list[RoundTrip]:
    """Parse NT8 semicolon-separated execution CSV into round-trip trades."""
    df = pd.read_csv(path, sep=";", encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]

    trades: list[RoundTrip] = []
    i = 0
    rows = df.to_dict("records")
    while i < len(rows) - 1:
        entry_row = rows[i]
        exit_row = rows[i + 1]

        action_entry = str(entry_row["Action"]).strip()
        action_exit = str(exit_row["Action"]).strip()

        # Determine direction
        if action_entry in ("Buy",):
            direction = 1
            expected_exit = "Sell"
        elif action_entry in ("SellShort",):
            direction = -1
            expected_exit = "BuyToCover"
        else:
            # Not an entry row — skip
            i += 1
            continue

        if action_exit != expected_exit:
            # Orphaned entry — still record it, advance by 1
            i += 1
            continue

        def _parse_price(val: object) -> float:
            s = str(val).replace(",", ".")
            return float(s)

        entry_time = pd.Timestamp(str(entry_row["Time"]).strip())
        exit_time = pd.Timestamp(str(exit_row["Time"]).strip())
        entry_price = _parse_price(entry_row["Price"])
        exit_price = _parse_price(exit_row["Price"])

        exit_name = str(exit_row["Name"]).strip()
        if "stop" in exit_name.lower():
            exit_reason = "stop"
        elif "hard" in exit_name.lower() or "close" in exit_name.lower():
            exit_reason = "hard_close"
        elif "target" in exit_name.lower() or "profit" in exit_name.lower():
            exit_reason = "target"
        else:
            exit_reason = exit_name

        trades.append(RoundTrip(
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            direction=direction,
            exit_reason=exit_reason,
            source="NT8",
        ))
        i += 2

    trades.sort(key=lambda t: t.entry_time)
    return trades


def parse_python_npy(paths: list[Path]) -> list[RoundTrip]:
    """Load one or more structured .npy trade logs and merge into round-trips."""
    arrays = []
    for p in paths:
        if not p.exists():
            print(f"  [WARN] Python trade log not found: {p}")
            continue
        arr = np.load(p, allow_pickle=True)
        if len(arr) > 0:
            arrays.append(arr)
            print(f"  Loaded {len(arr)} trades from {p}")

    if not arrays:
        return []

    combined = np.concatenate(arrays)
    # Sort by entry_time
    combined = combined[np.argsort(combined["entry_time"])]

    signal_map = {1: 1, -1: -1, 2: 1, -2: -1}
    reason_map = {0: "target", 1: "stop", 2: "hard_close", 3: "circuit_breaker"}

    trades: list[RoundTrip] = []
    for row in combined:
        entry_ts = pd.Timestamp(int(row["entry_time"]), unit="ns", tz="UTC").tz_convert(SESSION_TZ)
        exit_ts = pd.Timestamp(int(row["exit_time"]), unit="ns", tz="UTC").tz_convert(SESSION_TZ)
        trades.append(RoundTrip(
            entry_time=entry_ts,
            exit_time=exit_ts,
            entry_price=float(row["entry_price"]),
            exit_price=float(row["exit_price"]),
            direction=signal_map.get(int(row["signal_type"]), 0),
            exit_reason=reason_map.get(int(row["exit_reason"]), f"code_{row['exit_reason']}"),
            source="PY",
        ))

    trades.sort(key=lambda t: t.entry_time)
    return trades


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_by_date(nt8: list[RoundTrip], py: list[RoundTrip]) -> None:
    """Match trades by calendar date, then report first discrepancy of each type."""

    # Index by date
    nt8_by_date: dict[str, list[RoundTrip]] = {}
    for t in nt8:
        d = str(t.entry_time.date())
        nt8_by_date.setdefault(d, []).append(t)

    py_by_date: dict[str, list[RoundTrip]] = {}
    for t in py:
        d = str(t.entry_time.date())
        py_by_date.setdefault(d, []).append(t)

    all_dates = sorted(set(nt8_by_date.keys()) | set(py_by_date.keys()))

    # Restrict to overlapping date range
    nt8_dates = sorted(nt8_by_date.keys())
    py_dates = sorted(py_by_date.keys())
    if not py_dates:
        print("\n[ERROR] No Python trades loaded — cannot reconcile.")
        return

    overlap_start = max(nt8_dates[0], py_dates[0])
    overlap_end = min(nt8_dates[-1], py_dates[-1])
    overlap_dates = [d for d in all_dates if overlap_start <= d <= overlap_end]

    print(f"\n{'='*72}")
    print(f"DATE OVERLAP: {overlap_start} to {overlap_end} ({len(overlap_dates)} calendar dates with trades)")
    print(f"  NT8 trades in overlap: {sum(len(nt8_by_date.get(d, [])) for d in overlap_dates)}")
    print(f"  PY  trades in overlap: {sum(len(py_by_date.get(d, [])) for d in overlap_dates)}")
    print(f"{'='*72}")

    first_missing: dict[str, str | None] = {"nt8_only": None, "py_only": None}
    first_price_mismatch: str | None = None
    first_time_mismatch: str | None = None
    first_direction_mismatch: str | None = None
    first_exit_reason_mismatch: str | None = None

    matched_count = 0
    nt8_only_dates: list[str] = []
    py_only_dates: list[str] = []

    for date in overlap_dates:
        nt8_trades = nt8_by_date.get(date, [])
        py_trades = py_by_date.get(date, [])

        # --- Missing trades ---
        if nt8_trades and not py_trades:
            nt8_only_dates.append(date)
            if first_missing["nt8_only"] is None:
                t = nt8_trades[0]
                first_missing["nt8_only"] = (
                    f"\n>>> FIRST MISSING TRADE (exists in NT8, absent in Python):\n"
                    f"    Date:        {date}\n"
                    f"    NT8 Entry:   {t.entry_time}  @ {t.entry_price:.2f}\n"
                    f"    NT8 Exit:    {t.exit_time}  @ {t.exit_price:.2f}\n"
                    f"    Direction:   {'LONG' if t.direction == 1 else 'SHORT'}\n"
                    f"    Exit Reason: {t.exit_reason}"
                )
            continue

        if py_trades and not nt8_trades:
            py_only_dates.append(date)
            if first_missing["py_only"] is None:
                t = py_trades[0]
                first_missing["py_only"] = (
                    f"\n>>> FIRST MISSING TRADE (exists in Python, absent in NT8):\n"
                    f"    Date:        {date}\n"
                    f"    PY  Entry:   {t.entry_time}  @ {t.entry_price:.2f}\n"
                    f"    PY  Exit:    {t.exit_time}  @ {t.exit_price:.2f}\n"
                    f"    Direction:   {'LONG' if t.direction == 1 else 'SHORT'}\n"
                    f"    Exit Reason: {t.exit_reason}"
                )
            continue

        # --- Pairwise comparison (same date) ---
        for idx in range(max(len(nt8_trades), len(py_trades))):
            if idx >= len(nt8_trades):
                if first_missing["py_only"] is None:
                    t = py_trades[idx]
                    first_missing["py_only"] = (
                        f"\n>>> FIRST EXTRA PY TRADE (Python has more trades on this date):\n"
                        f"    Date:        {date}\n"
                        f"    PY  Entry:   {t.entry_time}  @ {t.entry_price:.2f}\n"
                        f"    NT8 trades on date: {len(nt8_trades)}, PY trades: {len(py_trades)}"
                    )
                continue
            if idx >= len(py_trades):
                if first_missing["nt8_only"] is None:
                    t = nt8_trades[idx]
                    first_missing["nt8_only"] = (
                        f"\n>>> FIRST EXTRA NT8 TRADE (NT8 has more trades on this date):\n"
                        f"    Date:        {date}\n"
                        f"    NT8 Entry:   {t.entry_time}  @ {t.entry_price:.2f}\n"
                        f"    NT8 trades on date: {len(nt8_trades)}, PY trades: {len(py_trades)}"
                    )
                continue

            nt = nt8_trades[idx]
            pt = py_trades[idx]
            matched_count += 1

            # Direction mismatch
            if nt.direction != pt.direction and first_direction_mismatch is None:
                first_direction_mismatch = (
                    f"\n>>> FIRST DIRECTION MISMATCH:\n"
                    f"    Date:     {date}\n"
                    f"    NT8:      {'LONG' if nt.direction == 1 else 'SHORT'}  entry {nt.entry_time} @ {nt.entry_price:.2f}\n"
                    f"    Python:   {'LONG' if pt.direction == 1 else 'SHORT'}  entry {pt.entry_time} @ {pt.entry_price:.2f}"
                )

            # Time mismatch (entry)
            # Make both tz-naive for comparison if needed
            nt_entry_naive = nt.entry_time.tz_localize(None) if nt.entry_time.tzinfo is not None else nt.entry_time
            pt_entry_naive = pt.entry_time.tz_localize(None) if pt.entry_time.tzinfo is not None else pt.entry_time
            nt_exit_naive = nt.exit_time.tz_localize(None) if nt.exit_time.tzinfo is not None else nt.exit_time
            pt_exit_naive = pt.exit_time.tz_localize(None) if pt.exit_time.tzinfo is not None else pt.exit_time

            entry_delta = abs(nt_entry_naive - pt_entry_naive)
            exit_delta = abs(nt_exit_naive - pt_exit_naive)

            if entry_delta > MATCH_TOLERANCE and first_time_mismatch is None:
                first_time_mismatch = (
                    f"\n>>> FIRST TIME MISMATCH (entry delta > {MATCH_TOLERANCE}):\n"
                    f"    Date:        {date}\n"
                    f"    NT8 Entry:   {nt.entry_time}\n"
                    f"    PY  Entry:   {pt.entry_time}\n"
                    f"    Entry Delta: {entry_delta}\n"
                    f"    NT8 Exit:    {nt.exit_time}\n"
                    f"    PY  Exit:    {pt.exit_time}\n"
                    f"    Exit Delta:  {exit_delta}"
                )

            # Price mismatch (> 1 tick)
            entry_price_diff = abs(nt.entry_price - pt.entry_price)
            exit_price_diff = abs(nt.exit_price - pt.exit_price)
            if (entry_price_diff > TICK_SIZE or exit_price_diff > TICK_SIZE) and first_price_mismatch is None:
                first_price_mismatch = (
                    f"\n>>> FIRST PRICE MISMATCH (> 1 tick = {TICK_SIZE}):\n"
                    f"    Date:             {date}\n"
                    f"    NT8 Entry Price:  {nt.entry_price:.2f}\n"
                    f"    PY  Entry Price:  {pt.entry_price:.2f}\n"
                    f"    Entry Diff:       {entry_price_diff:.2f}  ({entry_price_diff / TICK_SIZE:.1f} ticks)\n"
                    f"    NT8 Exit Price:   {nt.exit_price:.2f}\n"
                    f"    PY  Exit Price:   {pt.exit_price:.2f}\n"
                    f"    Exit Diff:        {exit_price_diff:.2f}  ({exit_price_diff / TICK_SIZE:.1f} ticks)"
                )

            # Exit reason mismatch
            if nt.exit_reason != pt.exit_reason and first_exit_reason_mismatch is None:
                first_exit_reason_mismatch = (
                    f"\n>>> FIRST EXIT REASON MISMATCH:\n"
                    f"    Date:       {date}\n"
                    f"    NT8:        {nt.exit_reason}  (exit @ {nt.exit_time} / {nt.exit_price:.2f})\n"
                    f"    Python:     {pt.exit_reason}  (exit @ {pt.exit_time} / {pt.exit_price:.2f})"
                )

    # ---------------------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------------------
    print(f"\n{'='*72}")
    print("TRADE COUNT SUMMARY")
    print(f"{'='*72}")
    print(f"  NT8 total round-trips:    {len(nt8)}")
    print(f"  Python total round-trips: {len(py)}")
    print(f"  Absolute difference:      {abs(len(nt8) - len(py))}")
    print(f"  Matched (same date):      {matched_count}")
    print(f"  NT8-only dates (in overlap): {len(nt8_only_dates)}")
    print(f"  PY-only dates (in overlap):  {len(py_only_dates)}")

    if nt8_only_dates:
        print(f"\n  NT8-only dates: {', '.join(nt8_only_dates[:10])}{'...' if len(nt8_only_dates) > 10 else ''}")
    if py_only_dates:
        print(f"  PY-only dates:  {', '.join(py_only_dates[:10])}{'...' if len(py_only_dates) > 10 else ''}")

    print(f"\n{'='*72}")
    print("FIRST DISCREPANCIES")
    print(f"{'='*72}")

    any_found = False
    for label, msg in [
        ("Missing (NT8 only)", first_missing["nt8_only"]),
        ("Missing (PY only)", first_missing["py_only"]),
        ("Direction mismatch", first_direction_mismatch),
        ("Time mismatch", first_time_mismatch),
        ("Price mismatch", first_price_mismatch),
        ("Exit reason mismatch", first_exit_reason_mismatch),
    ]:
        if msg is not None:
            print(msg)
            any_found = True

    if not any_found:
        print("\n  No discrepancies found in overlapping trades.")

    # ---------------------------------------------------------------------------
    # Timezone diagnostic
    # ---------------------------------------------------------------------------
    print(f"\n{'='*72}")
    print("TIMEZONE DIAGNOSTIC")
    print(f"{'='*72}")
    # Check if NT8 times are consistently offset from Python times
    offsets: list[pd.Timedelta] = []
    for date in overlap_dates:
        nt8_trades = nt8_by_date.get(date, [])
        py_trades = py_by_date.get(date, [])
        if nt8_trades and py_trades:
            nt_t = nt8_trades[0].entry_time
            py_t = py_trades[0].entry_time
            # NT8 is tz-naive, Python is tz-aware
            nt_naive = nt_t.tz_localize(None) if nt_t.tzinfo is not None else nt_t
            py_naive = py_t.tz_localize(None) if py_t.tzinfo is not None else py_t
            offsets.append(nt_naive - py_naive)

    if offsets:
        unique_offsets = sorted(set(str(o) for o in offsets))
        print(f"  NT8_entry - PY_entry offsets across {len(offsets)} matched dates:")
        for uo in unique_offsets:
            count = sum(1 for o in offsets if str(o) == uo)
            print(f"    {uo:>20s}  ({count} trades)")

    # ---------------------------------------------------------------------------
    # Hard-close time diagnostic
    # ---------------------------------------------------------------------------
    print(f"\n{'='*72}")
    print("HARD-CLOSE TIME DIAGNOSTIC")
    print(f"{'='*72}")
    nt8_hc_times = set()
    py_hc_times = set()
    for t in nt8:
        if t.exit_reason == "hard_close":
            nt8_hc_times.add(t.exit_time.strftime("%H:%M"))
    for t in py:
        if t.exit_reason == "hard_close":
            py_hc_times.add(t.exit_time.strftime("%H:%M"))
    print(f"  NT8 hard-close exit times: {sorted(nt8_hc_times)}")
    print(f"  PY  hard-close exit times: {sorted(py_hc_times)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile NT8 vs Python Donchian trades")
    parser.add_argument("--nt8", type=Path, default=NT8_DEFAULT, help="NT8 execution CSV")
    parser.add_argument(
        "--python-trades",
        type=Path,
        nargs="+",
        default=None,
        help="One or more Python .npy trade log files (default: val + test)",
    )
    parser.add_argument("--nt8-tz", type=str, default=None,
                        help="Timezone to assign to NT8 timestamps (e.g. 'US/Central'). "
                             "If set, NT8 times are localized and converted to ET for comparison.")
    args = parser.parse_args()

    if not args.nt8.exists():
        raise FileNotFoundError(f"NT8 file not found: {args.nt8}")

    python_paths = args.python_trades if args.python_trades else PYTHON_DEFAULTS

    print(f"NT8 source: {args.nt8}")
    print(f"Python sources: {[str(p) for p in python_paths]}")
    if args.nt8_tz:
        print(f"NT8 timezone override: {args.nt8_tz}")

    # Parse
    nt8_trades = parse_nt8(args.nt8)
    print(f"\nNT8: parsed {len(nt8_trades)} round-trip trades")
    if nt8_trades:
        print(f"  Range: {nt8_trades[0].entry_time} to {nt8_trades[-1].entry_time}")

    # If NT8 timezone override, localize and convert
    if args.nt8_tz:
        for t in nt8_trades:
            t.entry_time = t.entry_time.tz_localize(args.nt8_tz).tz_convert(SESSION_TZ)
            t.exit_time = t.exit_time.tz_localize(args.nt8_tz).tz_convert(SESSION_TZ)
        print(f"  After TZ conversion: {nt8_trades[0].entry_time} to {nt8_trades[-1].entry_time}")

    print()
    py_trades = parse_python_npy(python_paths)
    print(f"Python: loaded {len(py_trades)} round-trip trades")
    if py_trades:
        print(f"  Range: {py_trades[0].entry_time} to {py_trades[-1].entry_time}")

    # Match and compare
    match_by_date(nt8_trades, py_trades)


if __name__ == "__main__":
    main()
