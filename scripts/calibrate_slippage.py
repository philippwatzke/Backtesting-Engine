#!/usr/bin/env python
"""Calibrate slippage profile from normalized training data."""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from propfirm.io.config import load_mff_config
from propfirm.market.data_loader import _prepare_session_frame, compute_minute_of_day


def calibrate(train_path: Path, output_path: Path, tick_size: float):
    df = _prepare_session_frame(pd.read_parquet(train_path))
    print(f"Loaded {len(df)} bars from {train_path}")

    df["spread_ticks"] = (df["high"] - df["low"]) / tick_size
    df["minute_of_day"] = compute_minute_of_day(df.index)

    vol_q25 = df["volume"].quantile(0.25)
    low_vol_mask = df["volume"] <= vol_q25

    buckets = []
    for start in range(0, 390, 15):
        end = min(start + 15, 390)
        mask = (df["minute_of_day"] >= start) & (df["minute_of_day"] < end)
        subset = df[mask & low_vol_mask]["spread_ticks"]
        baseline = float(subset.median()) if len(subset) > 0 else 1.0
        buckets.append({"bucket_start": start, "bucket_end": end,
                        "baseline_ticks": baseline})

    result = pd.DataFrame(buckets)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    print(f"Saved slippage profile to {output_path}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate slippage profile")
    parser.add_argument("--train", type=Path,
                        default=Path("data/processed/MNQ_1m_train.parquet"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/slippage/slippage_profile.parquet"))
    parser.add_argument("--mff-config", type=Path,
                        default=Path("configs/mff_flex_50k.toml"))
    args = parser.parse_args()
    mff_cfg = load_mff_config(args.mff_config)
    calibrate(args.train, args.output, tick_size=mff_cfg["instrument"]["tick_size"])
