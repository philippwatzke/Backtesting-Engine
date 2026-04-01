import numpy as np
from numba import njit
from pathlib import Path
from propfirm.core.types import MNQ_TICK_SIZE, SLIPPAGE_FLOOR_POINTS


@njit(cache=True)
def compute_slippage(
    minute_of_day: int,
    bar_atr: float,
    trailing_median_atr: float,
    slippage_lookup: np.ndarray,
    is_stop_order: bool,
    stop_penalty: float,
) -> float:
    """Compute hybrid slippage for a single bar.

    Returns slippage in points (not ticks). Minimum 0.25 (1 tick).
    """
    baseline = slippage_lookup[minute_of_day]
    if trailing_median_atr > 0.0:
        atr_mult = bar_atr / trailing_median_atr
    else:
        atr_mult = 1.0
    penalty = stop_penalty if is_stop_order else 1.0
    raw = baseline * atr_mult * penalty * MNQ_TICK_SIZE
    if raw < SLIPPAGE_FLOOR_POINTS:
        raw = SLIPPAGE_FLOOR_POINTS
    return raw


_DEFAULT_BUCKETS = [
    (0, 15, 3.0),
    (15, 60, 1.5),
    (60, 120, 1.0),
    (120, 240, 0.75),
    (240, 330, 1.0),
    (330, 375, 1.25),
    (375, 390, 2.0),
]


def build_slippage_lookup(profile_path: Path | None, require_file: bool = False) -> np.ndarray:
    """Build a 390-element slippage lookup array (one per RTH minute)."""
    lookup = np.ones(390, dtype=np.float64)
    if profile_path is not None:
        profile_path = Path(profile_path)
        if profile_path.exists():
            import pandas as pd
            df = pd.read_parquet(profile_path)
            for _, row in df.iterrows():
                start = int(row["bucket_start"])
                end = int(row["bucket_end"])
                lookup[start:end] = row["baseline_ticks"]
            return lookup
        if require_file:
            raise FileNotFoundError(
                f"Slippage profile not found: {profile_path}. "
                "Run scripts/calibrate_slippage.py first."
            )
    for start, end, baseline in _DEFAULT_BUCKETS:
        lookup[start:end] = baseline
    return lookup
