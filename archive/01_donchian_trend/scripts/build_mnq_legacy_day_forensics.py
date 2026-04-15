#!/usr/bin/env python
"""Build per-day forensic report for remaining MNQ legacy TV-vs-Python mismatches."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from propfirm.core.types import (
    PARAMS_CONTRACTS,
    PARAMS_ENTRY_ON_CLOSE,
    PARAMS_EXTRA_SLIPPAGE_TICKS,
    PARAMS_TICK_SIZE,
    PARAMS_TICK_VALUE,
    PROFILE_BREAKEVEN_TRIGGER_TICKS,
    PROFILE_RISK_BUFFER_FRACTION,
    PROFILE_RISK_PER_TRADE_USD,
    PROFILE_STOP_ATR_MULTIPLIER,
    PROFILE_TARGET_ATR_MULTIPLIER,
)
from propfirm.io.config import build_phase_params, load_mff_config, load_params_config
from propfirm.rules.mff import MFFState
from propfirm.strategy.portfolio import combined_portfolio_signal
from run_mnq_legacy_frozen import _build_profiles, _load_legacy_mnq_data


SESSION_TZ = "America/New_York"
XLSX_PATH = Path("data/tradingview/PropFirm_Breakout_Strategy_-_MNQ_Legacy_Frozen_CME_MINI_MNQ1!_2026-04-14_d7d39.xlsx")
PY_LOG_PATH = Path("output/backtests_mnq_legacy_frozen/latest_trade_log.npy")
PY_DAILY_PATH = Path("output/backtests_mnq_legacy_frozen/latest_daily_log.npy")
RAW_DATA_PATH = Path("data/processed/MNQ_1m_full_test.parquet")
MFF_CONFIG = Path("configs/mff_flex_50k_mnq.toml")
PARAMS_CONFIG = Path("configs/default_params_mnq_legacy_frozen.toml")
OUTPUT_DIR = Path("output/tradingview_forensics/mnq_legacy_day_forensics")

NS = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _col_idx(cell_ref: str) -> int:
    letters = re.match(r"[A-Z]+", cell_ref).group(0)
    value = 0
    for ch in letters:
        value = value * 26 + (ord(ch) - 64)
    return value - 1


def _load_xlsx_rows(path: Path) -> list[list[str]]:
    with ZipFile(path) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("main:si", NS):
                shared_strings.append("".join(t.text or "" for t in si.iterfind(".//main:t", NS)))

        sheet = ET.fromstring(zf.read("xl/worksheets/sheet4.xml"))
        rows: list[list[str]] = []
        for row in sheet.findall(".//main:sheetData/main:row", NS):
            vals: dict[int, str] = {}
            for cell in row.findall("main:c", NS):
                idx = _col_idx(cell.attrib.get("r", "A1"))
                cell_type = cell.attrib.get("t")
                value_node = cell.find("main:v", NS)
                value = ""
                if cell_type == "s" and value_node is not None:
                    value = shared_strings[int(value_node.text)]
                elif cell_type == "inlineStr":
                    inline = cell.find("main:is", NS)
                    if inline is not None:
                        value = "".join(t.text or "" for t in inline.iterfind(".//main:t", NS))
                elif value_node is not None:
                    value = value_node.text or ""
                vals[idx] = value
            max_idx = max(vals.keys()) if vals else -1
            rows.append([vals.get(i, "") for i in range(max_idx + 1)])
    return rows


def _load_tv_trades(path: Path) -> pd.DataFrame:
    rows = _load_xlsx_rows(path)
    raw = pd.DataFrame(rows[1:])

    dt = pd.to_datetime(pd.to_numeric(raw[2]), unit="D", origin="1899-12-30", utc=True).dt.round("min")
    raw["dt_utc"] = dt
    raw["price"] = pd.to_numeric(raw[4])
    raw["qty"] = pd.to_numeric(raw[5])
    raw["pnl"] = pd.to_numeric(raw[7])

    records: list[dict] = []
    for trade_id, group in raw.groupby(0, sort=True):
        entry = group[group[1].str.contains("Einstieg", na=False)].iloc[0]
        exit_ = group[group[1].str.contains("Ausstieg", na=False)].iloc[0]
        entry_signal = str(entry[3])
        exit_signal = str(exit_[3])
        records.append(
            {
                "trade_id": int(trade_id),
                "entry_time_utc": entry["dt_utc"],
                "exit_time_utc": exit_["dt_utc"],
                "entry_time_et": entry["dt_utc"].tz_convert(SESSION_TZ),
                "exit_time_et": exit_["dt_utc"].tz_convert(SESSION_TZ),
                "entry_price": float(entry["price"]),
                "exit_price": float(exit_["price"]),
                "contracts": int(entry["qty"]),
                "net_pnl": float(exit_["pnl"]),
                "direction": 1 if "Long" in entry_signal else -1,
                "entry_signal": entry_signal,
                "exit_signal": exit_signal,
                "exit_reason": "stop" if "Stop" in exit_signal else "hard_close" if "HardClose" in exit_signal else "target",
            }
        )
    tv = pd.DataFrame(records).sort_values("entry_time_et").reset_index(drop=True)
    tv["entry_date_et"] = tv["entry_time_et"].dt.strftime("%Y-%m-%d")
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
    py["signal_time_et"] = py["entry_time_et"] - pd.Timedelta(minutes=30)
    return py.sort_values("entry_time_et").reset_index(drop=True)


def _load_daily_frame(path: Path, session_dates: list[str]) -> pd.DataFrame:
    arr = np.load(path, allow_pickle=False)
    df = pd.DataFrame(
        {
            "day_id": arr["day_id"].astype(np.int64),
            "phase_id": arr["phase_id"].astype(np.int64),
            "had_trade": arr["had_trade"].astype(np.int64),
            "n_trades": arr["n_trades"].astype(np.int64),
            "day_pnl": arr["day_pnl"].astype(np.float64),
            "net_payout": arr["net_payout"].astype(np.float64),
        }
    )
    df["session_date"] = session_dates[: len(df)]
    return df


def _signal_label(signal: int) -> str:
    if signal > 0:
        return "LONG"
    if signal < 0:
        return "SHORT"
    return "NONE"


def _build_issue_dates(tv: pd.DataFrame, py: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    overlap = tv[(tv["entry_time_et"] >= py["entry_time_et"].min()) & (tv["entry_time_et"] <= py["entry_time_et"].max())].copy()
    soft = overlap.merge(py, on=["entry_date_et", "direction"], how="outer", indicator=True, suffixes=("_tv", "_py"))
    both = soft.loc[soft["_merge"] == "both"].copy()
    both["entry_time_match"] = both["entry_time_et_tv"] == both["entry_time_et_py"]
    both["exit_time_match"] = both["exit_time_et_tv"] == both["exit_time_et_py"]
    both["contracts_match"] = both["contracts_tv"] == both["contracts_py"]
    both["exit_reason_match"] = both["exit_reason_tv"] == both["exit_reason_py"]
    both["exact_structural_match"] = (
        both["entry_time_match"] & both["exit_time_match"] & both["contracts_match"] & both["exit_reason_match"]
    )

    issue_rows: list[dict] = []
    for _, row in both.loc[~both["exact_structural_match"]].iterrows():
        issue_rows.append(
            {
                "issue_type": "qty_or_structure_drift",
                "entry_date_et": row["entry_date_et"],
                "direction": int(row["direction"]),
                "tv_trade_id": int(row["trade_id"]),
            }
        )
    for _, row in soft.loc[soft["_merge"] == "left_only"].iterrows():
        issue_rows.append(
            {
                "issue_type": "tv_only_signal",
                "entry_date_et": row["entry_date_et"],
                "direction": int(row["direction"]),
                "tv_trade_id": int(row["trade_id"]),
            }
        )
    for _, row in soft.loc[soft["_merge"] == "right_only"].iterrows():
        issue_rows.append(
            {
                "issue_type": "python_only_signal",
                "entry_date_et": row["entry_date_et"],
                "direction": int(row["direction"]),
                "tv_trade_id": np.nan,
            }
        )

    issue_df = pd.DataFrame(issue_rows).drop_duplicates(subset=["issue_type", "entry_date_et", "direction"]).sort_values(
        ["entry_date_et", "issue_type", "direction"], kind="stable"
    )
    issue_dates = issue_df["entry_date_et"].drop_duplicates().tolist()
    return issue_df.reset_index(drop=True), issue_dates


def _build_python_focus(issue_dates: list[str], data: dict, daily_df: pd.DataFrame, mff_cfg: dict, params_cfg: dict) -> pd.DataFrame:
    shared_cfg = params_cfg["strategy"]["mgc_h1_trend"]["shared"]
    eval_cfg = params_cfg["strategy"]["mgc_h1_trend"]["eval"]
    funded_cfg = params_cfg["strategy"]["mgc_h1_trend"]["funded"]
    slip_cfg = params_cfg["slippage"]
    risk_buffer_fraction = float(params_cfg.get("portfolio", {}).get("shared", {}).get("risk_buffer_fraction", 0.25))

    params_eval = build_phase_params(
        shared_cfg,
        eval_cfg,
        slip_cfg,
        mff_cfg["instrument"]["commission_per_side"],
        strategy_name="mgc_h1_trend",
        instrument_cfg=mff_cfg["instrument"],
    )
    params_funded = build_phase_params(
        shared_cfg,
        funded_cfg,
        slip_cfg,
        mff_cfg["instrument"]["commission_per_side"],
        strategy_name="mgc_h1_trend",
        instrument_cfg=mff_cfg["instrument"],
    )
    for params in (params_eval, params_funded):
        params[PARAMS_EXTRA_SLIPPAGE_TICKS] = 0.0
        params[PARAMS_ENTRY_ON_CLOSE] = 0.0

    strategy_profiles_eval, strategy_profiles_funded = _build_profiles(
        shared_cfg,
        eval_cfg,
        funded_cfg,
        risk_buffer_fraction,
    )

    state = MFFState(mff_cfg)
    rows: list[dict] = []

    for day_idx, (start, end) in enumerate(data["day_boundaries"]):
        session_date = data["session_dates"][day_idx]
        active_params = params_eval.copy() if state.phase == "eval" else params_funded.copy()
        strategy_profiles = strategy_profiles_eval.copy() if state.phase == "eval" else strategy_profiles_funded.copy()
        active_params[PARAMS_CONTRACTS] = float(
            state.get_max_contracts() if state.phase == "funded" else mff_cfg["eval"]["max_contracts"]
        )

        if session_date in issue_dates:
            for local_idx in range(end - start):
                global_idx = start + local_idx
                signal = combined_portfolio_signal(
                    local_idx,
                    data["open"][start:end],
                    data["high"][start:end],
                    data["low"][start:end],
                    data["close"][start:end],
                    data["volume"][start:end],
                    data["bar_atr"][start:end],
                    data["trailing_median_atr"][start:end],
                    data["daily_atr_ratio"][start:end],
                    data["rvol"][start:end],
                    data["close_sma_50"][start:end],
                    data["daily_regime_bias"][start:end],
                    data["donchian_high_5"][start:end],
                    data["donchian_low_5"][start:end],
                    data["minute_of_day"][start:end],
                    data["day_of_week"][start:end],
                    state.equity,
                    0.0,
                    0,
                    0.0,
                    False,
                    0,
                    active_params,
                )
                if signal == 0:
                    continue

                profile_idx = abs(int(signal)) - 1
                risk_per_trade_usd = float(strategy_profiles[profile_idx, PROFILE_RISK_PER_TRADE_USD])
                stop_atr_multiplier = float(strategy_profiles[profile_idx, PROFILE_STOP_ATR_MULTIPLIER])
                target_atr_multiplier = float(strategy_profiles[profile_idx, PROFILE_TARGET_ATR_MULTIPLIER])
                risk_buffer_fraction_bar = float(strategy_profiles[profile_idx, PROFILE_RISK_BUFFER_FRACTION])
                _ = float(strategy_profiles[profile_idx, PROFILE_BREAKEVEN_TRIGGER_TICKS])

                liquidation_floor_equity = float(state.get_liquidation_floor_equity())
                drawdown_buffer = float(state.equity - liquidation_floor_equity)
                risk_cap = drawdown_buffer * risk_buffer_fraction_bar
                effective_risk = min(risk_per_trade_usd, risk_cap)

                current_atr = float(data["bar_atr"][global_idx])
                tick_size = float(active_params[PARAMS_TICK_SIZE])
                tick_value = float(active_params[PARAMS_TICK_VALUE])
                dynamic_stop_ticks = (current_atr * stop_atr_multiplier) / tick_size
                dynamic_target_ticks = (current_atr * target_atr_multiplier) / tick_size
                risk_per_contract = dynamic_stop_ticks * tick_value
                contracts = int(np.floor(effective_risk / risk_per_contract)) if risk_per_contract > 0 else 0
                max_contracts = int(active_params[PARAMS_CONTRACTS])
                if max_contracts > 0 and contracts > max_contracts:
                    contracts = max_contracts

                ts_et = pd.Timestamp(int(data["timestamps"][global_idx]), unit="ns", tz="UTC").tz_convert(SESSION_TZ)
                rows.append(
                    {
                        "session_date": session_date,
                        "signal_time_et": ts_et,
                        "entry_time_et": ts_et + pd.Timedelta(minutes=30),
                        "minute_of_day": int(data["minute_of_day"][global_idx]),
                        "signal": int(signal),
                        "signal_label": _signal_label(int(signal)),
                        "close": float(data["close"][global_idx]),
                        "high": float(data["high"][global_idx]),
                        "low": float(data["low"][global_idx]),
                        "sma50": float(data["close_sma_50"][global_idx]),
                        "atr14_legacy": current_atr,
                        "donchian_high_5": float(data["donchian_high_5"][global_idx]),
                        "donchian_low_5": float(data["donchian_low_5"][global_idx]),
                        "daily_regime_bias": float(data["daily_regime_bias"][global_idx]),
                        "equity_before_day": float(state.equity),
                        "liquidation_floor_equity": liquidation_floor_equity,
                        "drawdown_buffer": drawdown_buffer,
                        "effective_risk_usd": effective_risk,
                        "stop_ticks": dynamic_stop_ticks,
                        "target_ticks": dynamic_target_ticks,
                        "contracts_python_signal": contracts,
                    }
                )

        if day_idx < len(daily_df):
            day_row = daily_df.iloc[day_idx]
            result = state.update_eod(
                float(day_row["day_pnl"]),
                float(state.equity + day_row["day_pnl"]),
                had_trade=bool(day_row["had_trade"]),
                session_date=session_date,
            )
            if result == "passed":
                state.start_funded()
            elif state.phase == "funded" and state.payout_eligible:
                state.process_payout()
            if result == "blown" or state.live_transition_ready:
                break

    return pd.DataFrame(rows).sort_values(["session_date", "signal_time_et"], kind="stable").reset_index(drop=True)


def _load_session_counts(path: Path, dates: list[str]) -> pd.DataFrame:
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
                "session_date": date_str,
                "rows_1m": int(len(session)),
                "first_bar_et": str(session.index.min()) if len(session) else "",
                "last_bar_et": str(session.index.max()) if len(session) else "",
                "is_full_0800_1559_session": bool(len(session) == 480),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    tv = _load_tv_trades(XLSX_PATH)
    py = _load_python_trades(PY_LOG_PATH)
    issue_df, issue_dates = _build_issue_dates(tv, py)

    params_cfg = load_params_config(PARAMS_CONFIG)
    mff_cfg = load_mff_config(MFF_CONFIG)
    data = _load_legacy_mnq_data(RAW_DATA_PATH, params_cfg)
    daily_df = _load_daily_frame(PY_DAILY_PATH, data["session_dates"])
    py_focus = _build_python_focus(issue_dates, data, daily_df, mff_cfg, params_cfg)
    session_counts = _load_session_counts(RAW_DATA_PATH, issue_dates)

    tv_by_day = tv.groupby(["entry_date_et", "direction"], sort=True).first().reset_index()
    py_by_day = py.groupby(["entry_date_et", "direction"], sort=True).first().reset_index()
    py_focus_by_day = py_focus.groupby(["session_date", "signal"], sort=True).first().reset_index()

    merged = issue_df.merge(
        tv_by_day,
        left_on=["entry_date_et", "direction"],
        right_on=["entry_date_et", "direction"],
        how="left",
        suffixes=("", "_tv"),
    ).merge(
        py_by_day,
        left_on=["entry_date_et", "direction"],
        right_on=["entry_date_et", "direction"],
        how="left",
        suffixes=("_tv", "_py"),
    ).merge(
        py_focus_by_day,
        left_on=["entry_date_et", "direction"],
        right_on=["session_date", "signal"],
        how="left",
    ).merge(
        session_counts,
        left_on="entry_date_et",
        right_on="session_date",
        how="left",
        suffixes=("", "_session"),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    issue_df.to_csv(OUTPUT_DIR / "issue_dates.csv", index=False)
    py_focus.to_csv(OUTPUT_DIR / "python_signal_focus.csv", index=False)
    session_counts.to_csv(OUTPUT_DIR / "session_counts.csv", index=False)
    merged.to_csv(OUTPUT_DIR / "day_forensics.csv", index=False)

    summary = {
        "artifacts": {
            "xlsx": str(XLSX_PATH),
            "python_trade_log": str(PY_LOG_PATH),
            "python_daily_log": str(PY_DAILY_PATH),
            "raw_data": str(RAW_DATA_PATH),
        },
        "issue_dates": issue_dates,
        "issue_count": int(len(issue_df)),
        "qty_or_structure_drift_count": int((issue_df["issue_type"] == "qty_or_structure_drift").sum()),
        "tv_only_signal_count": int((issue_df["issue_type"] == "tv_only_signal").sum()),
        "python_only_signal_count": int((issue_df["issue_type"] == "python_only_signal").sum()),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    qty_rows = merged.loc[merged["issue_type"] == "qty_or_structure_drift"].copy()
    tv_only_rows = merged.loc[merged["issue_type"] == "tv_only_signal"].copy()
    py_only_rows = merged.loc[merged["issue_type"] == "python_only_signal"].copy()
    short_session_rows = merged.loc[merged["is_full_0800_1559_session"] == False, ["entry_date_et", "rows_1m"]].drop_duplicates()

    md_lines = [
        "# MNQ Legacy Day Forensics",
        "",
        f"- Issue count: {summary['issue_count']}",
        f"- Qty/structure drift days: {summary['qty_or_structure_drift_count']}",
        f"- TV-only signal days: {summary['tv_only_signal_count']}",
        f"- Python-only signal days: {summary['python_only_signal_count']}",
        "",
        "## Issue Dates",
    ]
    for date_str in issue_dates:
        md_lines.append(f"- {date_str}")
    md_lines.extend(
        [
            "",
            "## Qty / Structure Drift",
        ]
    )
    if len(qty_rows):
        for _, row in qty_rows.iterrows():
            md_lines.append(
                "- "
                + f"{row['entry_date_et']}: TV qty {int(row['contracts_tv'])} vs Python qty {int(row['contracts_py'])}, "
                + f"exit {row['exit_reason_tv']}, stopTicks {float(row['stop_ticks']):.2f}, "
                + f"effRisk {float(row['effective_risk_usd']):.2f}"
            )
    else:
        md_lines.append("- none")

    md_lines.extend(
        [
            "",
            "## TV-Only Signal Days",
        ]
    )
    if len(tv_only_rows):
        for _, row in tv_only_rows.iterrows():
            session_flag = "short session" if not bool(row["is_full_0800_1559_session"]) else "full session"
            md_lines.append(
                "- "
                + f"{row['entry_date_et']}: TV qty {int(row['contracts_tv'])}, exit {row['exit_reason_tv']}, "
                + f"{session_flag}, rows_1m {int(row['rows_1m'])}"
            )
    else:
        md_lines.append("- none")

    md_lines.extend(
        [
            "",
            "## Python-Only Signal Days",
        ]
    )
    if len(py_only_rows):
        for _, row in py_only_rows.iterrows():
            md_lines.append(
                "- "
                + f"{row['entry_date_et']}: Python qty {int(row['contracts_py'])}, exit {row['exit_reason_py']}, "
                + f"minute {int(row['minute_of_day'])}, stopTicks {float(row['stop_ticks']):.2f}"
            )
    else:
        md_lines.append("- none")

    md_lines.extend(
        [
            "",
            "## Short Sessions In Issue Set",
        ]
    )
    if len(short_session_rows):
        for _, row in short_session_rows.iterrows():
            md_lines.append(f"- {row['entry_date_et']}: {int(row['rows_1m'])} one-minute bars")
    else:
        md_lines.append("- none")

    md_lines.extend(
        [
            "",
            "## Files",
            "- `issue_dates.csv`",
            "- `day_forensics.csv`",
            "- `python_signal_focus.csv`",
            "- `session_counts.csv`",
        ]
    )
    (OUTPUT_DIR / "report.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Wrote day forensics to {OUTPUT_DIR}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
