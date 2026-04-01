#!/usr/bin/env python
"""Calibrate slippage profile from normalized training data."""
import argparse
import pandas as pd
from pathlib import Path
from propfirm.io.config import load_mff_config
from propfirm.market.data_loader import _prepare_session_frame, compute_minute_of_day
from propfirm.market.slippage import estimate_baseline_ticks


def calibrate(
    train_path: Path,
    output_path: Path,
    tick_size: float,
    quantile: float,
    fraction: float,
    floor_ticks: float,
    cap_ticks: float,
):
    df = _prepare_session_frame(pd.read_parquet(train_path))
    print(f"Loaded {len(df)} bars from {train_path}")

    # With OHLCV data we only observe the bar range, not the executable spread.
    # Estimate baseline slippage as a small fraction of the lower-tail minute
    # range, bucketed by time-of-day and clipped to a plausible tick band.
    df["range_ticks"] = (df["high"] - df["low"]) / tick_size
    df["minute_of_day"] = compute_minute_of_day(df.index)

    buckets = []
    for start in range(0, 390, 15):
        end = min(start + 15, 390)
        mask = (df["minute_of_day"] >= start) & (df["minute_of_day"] < end)
        subset = df.loc[mask, "range_ticks"].to_numpy()
        baseline = estimate_baseline_ticks(
            subset,
            quantile=quantile,
            fraction=fraction,
            floor_ticks=floor_ticks,
            cap_ticks=cap_ticks,
        )
        buckets.append(
            {
                "bucket_start": start,
                "bucket_end": end,
                "baseline_ticks": baseline,
            }
        )

    result = pd.DataFrame(buckets)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    print(f"Saved slippage profile to {output_path}")
    print(
        "Calibration params:",
        {
            "quantile": quantile,
            "fraction": fraction,
            "floor_ticks": floor_ticks,
            "cap_ticks": cap_ticks,
        },
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate slippage profile")
    parser.add_argument("--train", type=Path,
                        default=Path("data/processed/MNQ_1m_train.parquet"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/slippage/slippage_profile.parquet"))
    parser.add_argument("--mff-config", type=Path,
                        default=Path("configs/mff_flex_50k.toml"))
    parser.add_argument("--quantile", type=float, default=0.10)
    parser.add_argument("--fraction", type=float, default=0.08)
    parser.add_argument("--floor-ticks", type=float, default=1.0)
    parser.add_argument("--cap-ticks", type=float, default=4.0)
    args = parser.parse_args()
    mff_cfg = load_mff_config(args.mff_config)
    calibrate(
        args.train,
        args.output,
        tick_size=mff_cfg["instrument"]["tick_size"],
        quantile=args.quantile,
        fraction=args.fraction,
        floor_ticks=args.floor_ticks,
        cap_ticks=args.cap_ticks,
    )
