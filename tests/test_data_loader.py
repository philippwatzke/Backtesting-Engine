import numpy as np
import pytest
from pathlib import Path
import pandas as pd

from propfirm.market.data_loader import (
    load_session_data,
    compute_minute_of_day,
    compute_trailing_atr,
    compute_daily_atr_ratio,
)


DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
TRAIN_PATH = DATA_DIR / "MNQ_1m_train.parquet"


@pytest.fixture
def train_data():
    if not TRAIN_PATH.exists():
        pytest.skip("Training data not available")
    return load_session_data(TRAIN_PATH, atr_period=14, trailing_atr_days=5)


class TestLoadSessionData:
    def test_returns_dict_with_required_keys(self, train_data):
        required = ["open", "high", "low", "close", "volume",
                     "timestamps", "minute_of_day", "day_of_week", "bar_atr",
                     "trailing_median_atr", "daily_atr_ratio", "rvol", "close_sma_50",
                     "daily_regime_bias",
                     "donchian_high_5", "donchian_low_5",
                     "day_boundaries", "session_dates",
                     "timeframe_minutes", "bars_per_session", "session_minutes"]
        for key in required:
            assert key in train_data, f"Missing key: {key}"

    def test_arrays_are_numpy(self, train_data):
        assert isinstance(train_data["open"], np.ndarray)
        assert isinstance(train_data["minute_of_day"], np.ndarray)

    def test_minute_of_day_range(self, train_data):
        mod = train_data["minute_of_day"]
        assert mod.min() >= 0
        assert mod.max() <= 389
        assert np.all(mod % 5 == 0)

    def test_no_look_ahead_in_trailing_atr(self, train_data):
        atr = train_data["trailing_median_atr"]
        assert not np.any(np.isnan(atr))
        assert not np.any(np.isinf(atr))

    def test_daily_atr_ratio_is_finite_and_positive(self, train_data):
        ratio = train_data["daily_atr_ratio"]
        assert len(ratio) == len(train_data["close"])
        assert not np.any(np.isnan(ratio))
        assert not np.any(np.isinf(ratio))
        assert np.all(ratio > 0.0)

    def test_day_boundaries_are_sorted(self, train_data):
        bounds = train_data["day_boundaries"]
        assert len(bounds) > 0
        for i in range(1, len(bounds)):
            assert bounds[i][0] > bounds[i - 1][0]

    def test_session_dates_align_with_day_boundaries(self, train_data):
        assert len(train_data["session_dates"]) == len(train_data["day_boundaries"])

    def test_rejects_naive_timestamps(self, tmp_path):
        df = pd.DataFrame(
            {
                "open": [1.0, 1.1],
                "high": [1.2, 1.3],
                "low": [0.9, 1.0],
                "close": [1.1, 1.2],
                "volume": [10, 20],
            },
            index=pd.DatetimeIndex(["2022-01-03 09:30:00", "2022-01-03 09:31:00"]),
        )
        path = tmp_path / "naive.parquet"
        df.to_parquet(path)
        with pytest.raises(ValueError):
            load_session_data(path)

    def test_sorts_converts_and_filters_to_rth(self, tmp_path):
        idx = pd.date_range(
            "2022-01-03 14:30:00+00:00",
            periods=390,
            freq="min",
            tz="UTC",
        )
        df = pd.DataFrame(
            {
                "open": np.linspace(1.0, 2.0, 390),
                "high": np.linspace(1.1, 2.1, 390),
                "low": np.linspace(0.9, 1.9, 390),
                "close": np.linspace(1.0, 2.0, 390),
                "volume": np.full(390, 10),
            },
            index=idx,
        )
        extra = pd.DataFrame(
            {
                "open": [0.8, 2.2],
                "high": [0.9, 2.3],
                "low": [0.7, 2.1],
                "close": [0.8, 2.2],
                "volume": [5, 5],
            },
            index=pd.DatetimeIndex(
                ["2022-01-03 13:00:00+00:00", "2022-01-03 21:05:00+00:00"]
            ),
        )
        combined = pd.concat([df.iloc[[1, 0]], extra, df.iloc[2:]])
        path = tmp_path / "utc_unsorted.parquet"
        combined.to_parquet(path)
        loaded = load_session_data(path)
        np.testing.assert_array_equal(loaded["minute_of_day"], np.arange(0, 390, 5, dtype=np.int16))
        assert len(loaded["open"]) == 78
        assert loaded["timeframe_minutes"] == 5
        assert loaded["bars_per_session"] == 78
        assert loaded["session_dates"] == ["2022-01-03"]
        np.testing.assert_allclose(loaded["daily_atr_ratio"], np.ones(78, dtype=np.float64))

    def test_resamples_ohlcv_into_five_minute_bars(self, tmp_path):
        idx = pd.date_range(
            "2022-01-03 14:30:00+00:00",
            periods=390,
            freq="min",
            tz="UTC",
        )
        opens = np.arange(390, dtype=np.float64) + 100.0
        highs = opens + 1.0
        lows = opens - 1.0
        closes = opens + 0.5
        volumes = np.arange(390, dtype=np.int64) + 1
        df = pd.DataFrame(
            {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
            },
            index=idx,
        )
        path = tmp_path / "five_minute_resample.parquet"
        df.to_parquet(path)

        loaded = load_session_data(path)

        assert loaded["open"][0] == opens[0]
        assert loaded["high"][0] == highs[:5].max()
        assert loaded["low"][0] == lows[:5].min()
        assert loaded["close"][0] == closes[4]
        assert loaded["volume"][0] == volumes[:5].sum()

    def test_rejects_incomplete_rth_session(self, tmp_path):
        idx = pd.DatetimeIndex(
            [
                "2022-01-03 14:31:00+00:00",  # 09:31 ET
                "2022-01-03 14:30:00+00:00",  # 09:30 ET
                "2022-01-03 13:00:00+00:00",  # 08:00 ET -> filtered
                "2022-01-03 21:05:00+00:00",  # 16:05 ET -> filtered
                "2022-01-03 20:59:00+00:00",  # 15:59 ET
            ]
        )
        df = pd.DataFrame(
            {
                "open": [1.1, 1.0, 0.8, 1.4, 1.3],
                "high": [1.2, 1.1, 0.9, 1.5, 1.4],
                "low": [1.0, 0.9, 0.7, 1.3, 1.2],
                "close": [1.1, 1.0, 0.8, 1.4, 1.3],
                "volume": [20, 10, 5, 30, 25],
            },
            index=idx,
        )
        path = tmp_path / "utc_unsorted.parquet"
        df.to_parquet(path)
        with pytest.raises(ValueError, match="No complete sessions remain"):
            load_session_data(path)

    def test_supports_overnight_session_resample(self, tmp_path):
        idx = pd.date_range(
            "2022-01-03 23:00:00+00:00",
            periods=481,
            freq="min",
            tz="UTC",
        )
        df = pd.DataFrame(
            {
                "open": np.linspace(1.0, 2.0, 481),
                "high": np.linspace(1.1, 2.1, 481),
                "low": np.linspace(0.9, 1.9, 481),
                "close": np.linspace(1.0, 2.0, 481),
                "volume": np.full(481, 10),
            },
            index=idx,
        )
        path = tmp_path / "overnight.parquet"
        df.to_parquet(path)
        loaded = load_session_data(
            path,
            timeframe_minutes=5,
            session_start="18:00",
            session_end="02:00",
        )
        assert loaded["timeframe_minutes"] == 5
        assert loaded["bars_per_session"] == 96
        assert loaded["session_minutes"] == 481
        assert loaded["session_dates"] == ["2022-01-03"]
        assert loaded["minute_of_day"][0] == 0
        assert loaded["minute_of_day"][-1] == 475


class TestComputeMinuteOfDay:
    def test_known_timestamps(self):
        idx = pd.DatetimeIndex([
            "2022-01-03 09:30:00",
            "2022-01-03 09:31:00",
            "2022-01-03 15:59:00",
        ], tz="America/New_York")
        result = compute_minute_of_day(idx)
        assert result[0] == 0
        assert result[1] == 1
        assert result[2] == 389

    def test_known_overnight_timestamps(self):
        idx = pd.DatetimeIndex([
            "2022-01-03 18:00:00",
            "2022-01-03 23:55:00",
            "2022-01-04 01:55:00",
        ], tz="America/New_York")
        result = compute_minute_of_day(
            idx,
            session_open_minute=18 * 60,
            wraps_overnight=True,
            session_minutes=481,
        )
        assert result[0] == 0
        assert result[1] == 355
        assert result[2] == 475


class TestComputeTrailingATR:
    def test_output_length_matches_input(self):
        highs = np.random.rand(1000) * 10 + 100
        lows = highs - np.random.rand(1000) * 2
        closes = (highs + lows) / 2
        day_bounds = [(i * 390, min((i + 1) * 390, 1000)) for i in range(3)]
        result = compute_trailing_atr(highs, lows, closes, day_bounds, period=14, trailing_days=5)
        assert len(result) == len(highs)

    def test_first_day_is_nan_then_history_becomes_finite(self):
        n = 2000
        highs = np.random.rand(n) * 10 + 100
        lows = highs - np.random.rand(n) * 2
        closes = (highs + lows) / 2
        day_bounds = [(i * 390, min((i + 1) * 390, n)) for i in range(6)]
        result = compute_trailing_atr(highs, lows, closes, day_bounds, period=14, trailing_days=5)
        assert np.all(np.isnan(result[:390]))
        assert np.all(np.isfinite(result[390:]))


class TestComputeDailyATRRatio:
    def test_uses_only_prior_days(self):
        highs = np.array([110.0, 120.0, 140.0], dtype=np.float64)
        lows = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        result = compute_daily_atr_ratio(highs, lows, window=20)
        np.testing.assert_allclose(result, np.array([1.0, 2.0, 40.0 / 15.0], dtype=np.float64))
