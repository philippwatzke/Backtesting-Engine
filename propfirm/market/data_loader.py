import numpy as np
import pandas as pd
from pathlib import Path
from propfirm.core.types import BARS_PER_RTH_SESSION


SESSION_TZ = "America/New_York"
RTH_OPEN_MINUTE = 9 * 60 + 30
REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def _prepare_session_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize bars into a strict RTH-only America/New_York session frame."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Parquet data must be indexed by DatetimeIndex")
    if df.index.tz is None:
        raise ValueError("Parquet index must be timezone-aware")
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Parquet data missing required columns: {missing}")

    df = df.sort_index(kind="stable").copy()
    df.index = df.index.tz_convert(SESSION_TZ)
    df = df.loc[~df.index.duplicated(keep="first")]
    df = df.between_time("09:30", "15:59")

    if df.empty:
        raise ValueError("No RTH bars remain after timezone conversion/filtering")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Prepared session index must be monotonic increasing")
    return df


def compute_minute_of_day(index: pd.DatetimeIndex) -> np.ndarray:
    """Convert timestamps to minute-of-day (0=09:30, 389=15:59)."""
    if index.tz is None:
        raise ValueError("minute_of_day requires timezone-aware timestamps")
    if str(index.tz) != SESSION_TZ:
        raise ValueError("minute_of_day requires America/New_York timestamps")
    hours = index.hour
    minutes = index.minute
    total_minutes = hours * 60 + minutes
    minute_of_day = total_minutes - RTH_OPEN_MINUTE
    if np.any(minute_of_day < 0) or np.any(minute_of_day >= BARS_PER_RTH_SESSION):
        raise ValueError("minute_of_day received timestamps outside 09:30-15:59 ET")
    return np.array(minute_of_day, dtype=np.int16)


def compute_trailing_atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    day_boundaries: list[tuple[int, int]],
    period: int = 14,
    trailing_days: int = 5,
) -> np.ndarray:
    """Compute trailing N-day median ATR - causally correct (no look-ahead)."""
    n_bars = len(highs)
    result = np.zeros(n_bars, dtype=np.float64)

    tr = np.maximum(highs - lows, np.zeros(n_bars))
    if n_bars > 1:
        prev_close = np.roll(closes, 1)
        prev_close[0] = closes[0]
        tr = np.maximum(tr, np.abs(highs - prev_close))
        tr = np.maximum(tr, np.abs(lows - prev_close))

    session_atrs = []
    for start, end in day_boundaries:
        session_tr = tr[start:end]
        if len(session_tr) >= period:
            atr_values = np.convolve(session_tr, np.ones(period) / period, mode="valid")
            session_atrs.append(np.median(atr_values))
        elif len(session_tr) > 0:
            session_atrs.append(np.mean(session_tr))
        else:
            session_atrs.append(0.0)

    for day_idx, (start, end) in enumerate(day_boundaries):
        if day_idx == 0:
            trailing_val = session_atrs[0] if session_atrs[0] > 0 else 1.0
        else:
            lookback = session_atrs[max(0, day_idx - trailing_days):day_idx]
            trailing_val = float(np.median(lookback)) if lookback else 1.0
        if trailing_val <= 0:
            trailing_val = 1.0
        result[start:end] = trailing_val

    return result


def _find_day_boundaries(index: pd.DatetimeIndex) -> list[tuple[int, int]]:
    """Find (start_idx, end_idx) for each trading day."""
    if len(index) == 0:
        raise ValueError("Cannot compute day boundaries for empty index")
    dates = index.date
    boundaries = []
    current_date = dates[0]
    start = 0
    for i in range(1, len(dates)):
        if dates[i] != current_date:
            boundaries.append((start, i))
            start = i
            current_date = dates[i]
    boundaries.append((start, len(dates)))
    return boundaries


def load_session_data(
    parquet_path: Path,
    atr_period: int = 14,
    trailing_atr_days: int = 5,
) -> dict:
    """Load parquet data and compute all precomputed arrays."""
    df = pd.read_parquet(parquet_path)
    df = _prepare_session_frame(df)

    day_boundaries = _find_day_boundaries(df.index)

    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)

    n = len(df)
    tr = np.maximum(highs - lows, np.zeros(n))
    if n > 1:
        prev_close = np.roll(closes, 1)
        prev_close[0] = closes[0]
        tr = np.maximum(tr, np.abs(highs - prev_close))
        tr = np.maximum(tr, np.abs(lows - prev_close))

    bar_atr = np.zeros(n, dtype=np.float64)
    if n >= atr_period:
        kernel = np.ones(atr_period) / atr_period
        convolved = np.convolve(tr, kernel, mode="full")[:n]
        bar_atr[atr_period - 1:] = convolved[atr_period - 1:]
        bar_atr[:atr_period - 1] = convolved[:atr_period - 1]
    else:
        bar_atr[:] = np.mean(tr)

    trailing_median_atr = compute_trailing_atr(
        highs, lows, closes, day_boundaries, atr_period, trailing_atr_days
    )

    return {
        "open": df["open"].values.astype(np.float64),
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": df["volume"].values,
        "timestamps": df.index.asi8,
        "minute_of_day": compute_minute_of_day(df.index),
        "bar_atr": bar_atr,
        "trailing_median_atr": trailing_median_atr,
        "day_boundaries": day_boundaries,
        "session_dates": [str(df.index[start].date()) for start, _ in day_boundaries],
    }
