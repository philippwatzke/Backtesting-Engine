import numpy as np
import pandas as pd
from pathlib import Path
from propfirm.core.types import BARS_PER_RTH_SESSION


SESSION_TZ = "America/New_York"
RTH_OPEN_MINUTE = 9 * 60 + 30
REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
RESAMPLE_TIMEFRAME_MINUTES = 5
RESAMPLED_BARS_PER_SESSION = BARS_PER_RTH_SESSION // RESAMPLE_TIMEFRAME_MINUTES


def _session_wraps_overnight(session_start: str, session_end: str) -> bool:
    return _session_open_minute(session_end) <= _session_open_minute(session_start)


def _session_minutes(session_start: str, session_end: str) -> int:
    start = _session_open_minute(session_start)
    end = _session_open_minute(session_end)
    if end < start:
        end += 24 * 60
    return end - start + 1


def _session_open_minute(session_start: str) -> int:
    ts = pd.Timestamp(f"2000-01-01 {session_start}")
    return ts.hour * 60 + ts.minute


def _session_labels(index: pd.DatetimeIndex, session_start: str, session_end: str) -> np.ndarray:
    session_open_minute = _session_open_minute(session_start)
    wraps_overnight = _session_wraps_overnight(session_start, session_end)
    labels = index.normalize()
    if wraps_overnight:
        total_minutes = index.hour * 60 + index.minute
        labels = labels - pd.to_timedelta((total_minutes < session_open_minute).astype(np.int8), unit="D")
    return np.asarray(labels.date)


def _filter_complete_sessions(
    df: pd.DataFrame,
    session_start: str,
    session_end: str,
    bars_per_session: int,
) -> pd.DataFrame:
    """Keep only complete, gap-free sessions."""
    start_hour, start_minute = map(int, session_start.split(":"))
    labels = _session_labels(df.index, session_start, session_end)
    keep = []
    for session_date, session_df in df.groupby(labels, sort=True):
        expected_start = (
            pd.Timestamp(session_date)
            .tz_localize(SESSION_TZ)
            + pd.Timedelta(hours=start_hour, minutes=start_minute)
        )
        expected_index = pd.date_range(
            start=expected_start,
            periods=bars_per_session,
            freq="min",
        )
        if len(session_df) == bars_per_session and session_df.index.equals(expected_index):
            keep.append(session_df)
    if not keep:
        raise ValueError("No complete sessions remain after filtering")
    return pd.concat(keep, axis=0)


def _prepare_session_frame(
    df: pd.DataFrame,
    session_start: str = "09:30",
    session_end: str = "15:59",
) -> pd.DataFrame:
    """Normalize bars into a strict America/New_York session frame."""
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
    df = df.between_time(session_start, session_end)

    if df.empty:
        raise ValueError("No session bars remain after timezone conversion/filtering")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Prepared session index must be monotonic increasing")
    return _filter_complete_sessions(
        df,
        session_start=session_start,
        session_end=session_end,
        bars_per_session=_session_minutes(session_start, session_end),
    )


def _resample_session_frame(
    df: pd.DataFrame,
    session_start: str,
    session_end: str,
    bars_per_session: int,
    timeframe_minutes: int = RESAMPLE_TIMEFRAME_MINUTES,
) -> pd.DataFrame:
    """Aggregate complete 1-minute sessions into a higher timeframe."""
    if timeframe_minutes <= 1:
        return df

    rule = f"{timeframe_minutes}min"
    start_hour, start_minute = map(int, session_start.split(":"))
    resampled_bars_per_session = bars_per_session // timeframe_minutes
    keep = []
    labels = _session_labels(df.index, session_start, session_end)
    for session_date, session_df in df.groupby(labels, sort=True):
        resampled = session_df.resample(rule, label="left", closed="left").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        resampled = resampled.dropna(subset=["open", "high", "low", "close"])
        expected_start = (
            pd.Timestamp(session_date)
            .tz_localize(SESSION_TZ)
            + pd.Timedelta(hours=start_hour, minutes=start_minute)
        )
        expected_index = pd.date_range(
            start=expected_start,
            periods=resampled_bars_per_session,
            freq=rule,
        )
        resampled = resampled.loc[resampled.index.intersection(expected_index)]
        if len(resampled) == resampled_bars_per_session and resampled.index.equals(expected_index):
            keep.append(resampled)
    if not keep:
        raise ValueError("No complete resampled sessions remain after aggregation")
    return pd.concat(keep, axis=0)


def compute_minute_of_day(
    index: pd.DatetimeIndex,
    session_open_minute: int = RTH_OPEN_MINUTE,
    wraps_overnight: bool = False,
    session_minutes: int | None = None,
) -> np.ndarray:
    """Convert timestamps to minute-of-day relative to session open."""
    if index.tz is None:
        raise ValueError("minute_of_day requires timezone-aware timestamps")
    if str(index.tz) != SESSION_TZ:
        raise ValueError("minute_of_day requires America/New_York timestamps")
    hours = index.hour
    minutes = index.minute
    total_minutes = hours * 60 + minutes
    if wraps_overnight:
        total_minutes = np.where(total_minutes < session_open_minute, total_minutes + 24 * 60, total_minutes)
    minute_of_day = total_minutes - session_open_minute
    if np.any(minute_of_day < 0):
        raise ValueError("minute_of_day received timestamps before session open")
    if session_minutes is not None and np.any(minute_of_day >= session_minutes):
        raise ValueError("minute_of_day received timestamps after session end")
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
    result = np.full(n_bars, np.nan, dtype=np.float64)

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
            continue
        else:
            lookback = session_atrs[max(0, day_idx - trailing_days):day_idx]
            trailing_val = float(np.median(lookback)) if lookback else 1.0
        if trailing_val <= 0:
            trailing_val = 1.0
        result[start:end] = trailing_val

    return result


def compute_daily_atr_ratio(day_highs: np.ndarray, day_lows: np.ndarray, window: int = 20) -> np.ndarray:
    """Compute daily ATR(1) to prior-window mean ratio on a per-day basis."""
    if len(day_highs) != len(day_lows):
        raise ValueError("day_highs and day_lows must have the same length")
    if window <= 0:
        raise ValueError("window must be positive")
    daily_atr = np.maximum(day_highs - day_lows, 0.0)
    ratios = np.ones(len(daily_atr), dtype=np.float64)
    for day_idx in range(1, len(daily_atr)):
        start = max(0, day_idx - window)
        mean_atr = float(np.mean(daily_atr[start:day_idx]))
        if mean_atr > 0.0:
            ratios[day_idx] = daily_atr[day_idx] / mean_atr
        else:
            ratios[day_idx] = 1.0
    return ratios


def compute_rvol(
    volumes: np.ndarray,
    minute_of_day: np.ndarray,
    day_boundaries: list[tuple[int, int]],
    session_minutes: int,
    lookback_days: int = 20,
) -> np.ndarray:
    """Compute relative volume per minute-of-day using a causal rolling 20-day baseline."""
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive")

    n_days = len(day_boundaries)
    day_slot_volumes = np.full((n_days, session_minutes), np.nan, dtype=np.float64)
    for day_idx, (start, end) in enumerate(day_boundaries):
        slots = minute_of_day[start:end].astype(np.int64)
        day_slot_volumes[day_idx, slots] = volumes[start:end].astype(np.float64)

    rvol = np.ones(len(volumes), dtype=np.float64)
    for day_idx, (start, end) in enumerate(day_boundaries):
        if day_idx < lookback_days:
            continue
        slots = minute_of_day[start:end].astype(np.int64)
        history = day_slot_volumes[day_idx - lookback_days:day_idx, :][:, slots]
        history_sum = np.nansum(history, axis=0)
        history_count = np.sum(~np.isnan(history), axis=0)
        baseline = np.ones(len(slots), dtype=np.float64)
        valid = history_count > 0
        baseline[valid] = history_sum[valid] / history_count[valid]
        current_volume = volumes[start:end].astype(np.float64)
        valid = baseline > 0.0
        day_rvol = np.ones(len(slots), dtype=np.float64)
        day_rvol[valid] = current_volume[valid] / baseline[valid]
        rvol[start:end] = day_rvol
    return rvol


def compute_daily_regime_bias(
    closes: np.ndarray,
    day_boundaries: list[tuple[int, int]],
    window: int = 50,
) -> np.ndarray:
    """Compute prior-day close vs daily SMA(50) regime bias per bar.

    Bias values:
    -  1.0: prior-day close > prior-day daily SMA(50), long regime
    - -1.0: prior-day close < prior-day daily SMA(50), short regime
    -  0.0: no valid daily regime yet (insufficient history) or exact tie
    """
    if window <= 0:
        raise ValueError("window must be positive")

    n_bars = day_boundaries[-1][1] if day_boundaries else 0
    regime = np.zeros(n_bars, dtype=np.float64)
    if not day_boundaries:
        return regime

    daily_closes = np.array([closes[end - 1] for _, end in day_boundaries], dtype=np.float64)
    daily_sma = (
        pd.Series(daily_closes)
        .rolling(window=window, min_periods=window)
        .mean()
        .to_numpy(dtype=np.float64)
    )
    daily_bias = np.zeros(len(day_boundaries), dtype=np.float64)
    for day_idx in range(1, len(day_boundaries)):
        prev_close = daily_closes[day_idx - 1]
        prev_sma = daily_sma[day_idx - 1]
        if not np.isfinite(prev_sma):
            continue
        if prev_close > prev_sma:
            daily_bias[day_idx] = 1.0
        elif prev_close < prev_sma:
            daily_bias[day_idx] = -1.0

    for day_idx, (start, end) in enumerate(day_boundaries):
        regime[start:end] = daily_bias[day_idx]
    return regime


def _compute_wilder_rma(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder-style running moving average with alpha = 1 / period."""
    if period <= 0:
        raise ValueError("period must be positive")

    result = np.full(len(values), np.nan, dtype=np.float64)
    if len(values) == 0:
        return result

    series = pd.Series(values.astype(np.float64))
    rma = series.ewm(alpha=1.0 / period, adjust=False).mean().to_numpy(dtype=np.float64)
    if len(values) >= period:
        seed = float(series.iloc[:period].mean())
        result[period - 1] = seed
        for idx in range(period, len(values)):
            result[idx] = (result[idx - 1] * (period - 1) + values[idx]) / period
        return result

    result[:] = rma
    return result


def _find_day_boundaries(session_labels: np.ndarray) -> list[tuple[int, int]]:
    """Find (start_idx, end_idx) for each trading session."""
    if len(session_labels) == 0:
        raise ValueError("Cannot compute day boundaries for empty index")
    boundaries = []
    current_date = session_labels[0]
    start = 0
    for i in range(1, len(session_labels)):
        if session_labels[i] != current_date:
            boundaries.append((start, i))
            start = i
            current_date = session_labels[i]
    boundaries.append((start, len(session_labels)))
    return boundaries


def load_session_data(
    parquet_path: Path,
    atr_period: int = 14,
    trailing_atr_days: int = 5,
    timeframe_minutes: int = RESAMPLE_TIMEFRAME_MINUTES,
    session_start: str = "09:30",
    session_end: str = "15:59",
) -> dict:
    """Load parquet data and compute all precomputed arrays."""
    df = pd.read_parquet(parquet_path)
    bars_per_session = _session_minutes(session_start, session_end)
    session_open_minute = _session_open_minute(session_start)
    wraps_overnight = _session_wraps_overnight(session_start, session_end)
    df = _prepare_session_frame(df, session_start=session_start, session_end=session_end)
    df = _resample_session_frame(
        df,
        session_start=session_start,
        session_end=session_end,
        bars_per_session=bars_per_session,
        timeframe_minutes=timeframe_minutes,
    )

    session_labels = _session_labels(df.index, session_start, session_end)
    day_boundaries = _find_day_boundaries(session_labels)
    minute_of_day = compute_minute_of_day(
        df.index,
        session_open_minute=session_open_minute,
        wraps_overnight=wraps_overnight,
        session_minutes=bars_per_session,
    )

    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    volumes = df["volume"].values

    n = len(df)
    tr = np.maximum(highs - lows, np.zeros(n))
    if n > 1:
        prev_close = np.roll(closes, 1)
        prev_close[0] = closes[0]
        tr = np.maximum(tr, np.abs(highs - prev_close))
        tr = np.maximum(tr, np.abs(lows - prev_close))

    bar_atr = _compute_wilder_rma(tr, atr_period)
    if n < atr_period and n > 0:
        bar_atr[:] = np.mean(tr)

    trailing_median_atr = compute_trailing_atr(
        highs, lows, closes, day_boundaries, atr_period, trailing_atr_days
    )
    day_highs = np.array([highs[start:end].max() for start, end in day_boundaries], dtype=np.float64)
    day_lows = np.array([lows[start:end].min() for start, end in day_boundaries], dtype=np.float64)
    daily_atr_ratio_by_day = compute_daily_atr_ratio(day_highs, day_lows, window=20)
    daily_atr_ratio = np.zeros(n, dtype=np.float64)
    for day_idx, (start, end) in enumerate(day_boundaries):
        daily_atr_ratio[start:end] = daily_atr_ratio_by_day[day_idx]
    rvol = compute_rvol(
        volumes,
        minute_of_day,
        day_boundaries,
        session_minutes=bars_per_session,
        lookback_days=20,
    )
    close_sma_50 = (
        df["close"]
        .rolling(window=50, min_periods=50)
        .mean()
        .to_numpy(dtype=np.float64)
    )
    donchian_high_5 = (
        df["high"]
        .shift(1)
        .rolling(window=5, min_periods=5)
        .max()
        .to_numpy(dtype=np.float64)
    )
    donchian_low_5 = (
        df["low"]
        .shift(1)
        .rolling(window=5, min_periods=5)
        .min()
        .to_numpy(dtype=np.float64)
    )
    daily_regime_bias = compute_daily_regime_bias(closes, day_boundaries, window=50)
    day_of_week = np.zeros(n, dtype=np.int8)
    for day_idx, (start, end) in enumerate(day_boundaries):
        day_of_week[start:end] = pd.Timestamp(session_labels[start]).weekday()

    return {
        "open": df["open"].values.astype(np.float64),
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "timestamps": df.index.asi8,
        "minute_of_day": minute_of_day,
        "day_of_week": day_of_week,
        "bar_atr": bar_atr,
        "trailing_median_atr": trailing_median_atr,
        "daily_atr_ratio": daily_atr_ratio,
        "rvol": rvol,
        "close_sma_50": close_sma_50,
        "daily_regime_bias": daily_regime_bias,
        "donchian_high_5": donchian_high_5,
        "donchian_low_5": donchian_low_5,
        "day_boundaries": day_boundaries,
        "session_dates": [str(session_labels[start]) for start, _ in day_boundaries],
        "session_minutes": bars_per_session,
        "timeframe_minutes": timeframe_minutes,
        "bars_per_session": bars_per_session // timeframe_minutes,
    }


# ---------------------------------------------------------------------------
# KAMA / MACD — standalone feature helpers (do not alter existing functions)
# ---------------------------------------------------------------------------

def compute_kama(closes: np.ndarray, period: int) -> np.ndarray:
    """Kaufman's Adaptive Moving Average (KAMA).

    Uses fast EMA period = 2, slow EMA period = 30.
    Returns NaN for bars before the first fully-formed period window.
    """
    if period <= 0:
        raise ValueError("period must be > 0")
    n = len(closes)
    kama = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return kama

    fast_sc = 2.0 / (2.0 + 1.0)   # smoothing constant for 2-period EMA
    slow_sc = 2.0 / (30.0 + 1.0)  # smoothing constant for 30-period EMA

    kama[period - 1] = closes[period - 1]
    for i in range(period, n):
        direction = abs(closes[i] - closes[i - period])
        path = np.sum(np.abs(np.diff(closes[i - period : i + 1])))
        er = direction / path if path > 0.0 else 0.0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (closes[i] - kama[i - 1])

    return kama


def compute_macd(
    closes: np.ndarray,
    fast: int,
    slow: int,
    signal_period: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Standard MACD using exponential moving averages.

    Returns (macd_line, signal_line) as float64 arrays.
    Uses pandas EWM with adjust=False (standard charting convention).
    """
    if fast <= 0 or slow <= 0 or signal_period <= 0:
        raise ValueError("fast, slow, and signal_period must all be > 0")
    if fast >= slow:
        raise ValueError("fast must be < slow")
    s = pd.Series(closes.astype(np.float64))
    ema_fast = s.ewm(span=fast, adjust=False).mean().to_numpy(dtype=np.float64)
    ema_slow = s.ewm(span=slow, adjust=False).mean().to_numpy(dtype=np.float64)
    macd_line = ema_fast - ema_slow
    signal_line = (
        pd.Series(macd_line).ewm(span=signal_period, adjust=False).mean().to_numpy(dtype=np.float64)
    )
    return macd_line, signal_line
