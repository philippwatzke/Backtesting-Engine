import numpy as np
from numba import njit
from pathlib import Path
from propfirm.core.types import SLIPPAGE_FLOOR_POINTS


@njit(cache=True)
def compute_slippage(
    minute_of_day: int,
    bar_atr: float,
    trailing_median_atr: float,
    slippage_lookup: np.ndarray,
    is_stop_order: bool,
    stop_penalty: float,
    tick_size: float,
    extra_slippage_points: float = 0.0,
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
    raw = baseline * atr_mult * penalty * tick_size
    floor_points = tick_size if tick_size > 0.0 else SLIPPAGE_FLOOR_POINTS
    if raw < floor_points:
        raw = floor_points
    raw += extra_slippage_points
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


def _build_scaled_default_lookup(session_minutes: int, multiplier: float = 1.25) -> np.ndarray:
    """Scale the default RTH slippage profile to arbitrary session lengths."""
    lookup = np.ones(session_minutes, dtype=np.float64)
    if session_minutes <= 0:
        return lookup
    for start, end, baseline in _DEFAULT_BUCKETS:
        scaled_start = int(np.floor(start / 390.0 * session_minutes))
        scaled_end = int(np.ceil(end / 390.0 * session_minutes))
        if scaled_end <= scaled_start:
            scaled_end = min(session_minutes, scaled_start + 1)
        lookup[scaled_start:scaled_end] = baseline * multiplier
    return lookup


def estimate_baseline_ticks(
    range_ticks: np.ndarray,
    quantile: float = 0.10,
    fraction: float = 0.08,
    floor_ticks: float = 1.0,
    cap_ticks: float = 4.0,
) -> float:
    """Estimate a realistic baseline slippage in ticks from 1-minute ranges.

    OHLCV bars do not expose bid/ask spread or queue position, so the full
    high-low range is far too large to use directly as slippage. A more stable
    proxy is a small fraction of a lower-tail range quantile, clipped to a
    defensible tick band.
    """
    values = np.asarray(range_ticks, dtype=np.float64)
    if values.size == 0:
        return floor_ticks
    if not (0.0 < quantile <= 1.0):
        raise ValueError("quantile must be in (0, 1]")
    if fraction <= 0.0:
        raise ValueError("fraction must be > 0")
    if floor_ticks <= 0.0:
        raise ValueError("floor_ticks must be > 0")
    if cap_ticks < floor_ticks:
        raise ValueError("cap_ticks must be >= floor_ticks")

    baseline = float(np.quantile(values, quantile)) * fraction
    if baseline < floor_ticks:
        baseline = floor_ticks
    if baseline > cap_ticks:
        baseline = cap_ticks
    return baseline


def build_slippage_lookup(
    profile_path: Path | None,
    require_file: bool = False,
    session_minutes: int = 390,
) -> np.ndarray:
    """Build a per-minute slippage lookup array for the active session."""
    lookup = np.ones(session_minutes, dtype=np.float64)
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
    if session_minutes != 390:
        return _build_scaled_default_lookup(session_minutes)
    for start, end, baseline in _DEFAULT_BUCKETS:
        lookup[start:end] = baseline
    return lookup
