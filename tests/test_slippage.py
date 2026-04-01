import numpy as np
import pytest
from numba import njit

from propfirm.market.slippage import compute_slippage, build_slippage_lookup


class TestComputeSlippage:
    def test_returns_float(self):
        lookup = np.ones(390, dtype=np.float64)
        result = compute_slippage(0, 5.0, 5.0, lookup, False, 1.5)
        assert isinstance(result, float)

    def test_floor_at_one_tick(self):
        lookup = np.full(390, 0.01, dtype=np.float64)  # Very low baseline
        result = compute_slippage(100, 0.1, 10.0, lookup, False, 1.0)
        assert result == 0.25  # Floor = 1 tick = 0.25 points

    def test_scales_with_atr(self):
        lookup = np.ones(390, dtype=np.float64)
        low_vol = compute_slippage(100, 2.0, 10.0, lookup, False, 1.0)
        high_vol = compute_slippage(100, 20.0, 10.0, lookup, False, 1.0)
        assert high_vol > low_vol

    def test_stop_penalty_increases_slippage(self):
        lookup = np.ones(390, dtype=np.float64)
        no_penalty = compute_slippage(100, 5.0, 5.0, lookup, False, 1.5)
        with_penalty = compute_slippage(100, 5.0, 5.0, lookup, True, 1.5)
        assert with_penalty == no_penalty * 1.5

    def test_zero_trailing_atr_returns_floor(self):
        lookup = np.ones(390, dtype=np.float64)
        result = compute_slippage(100, 5.0, 0.0, lookup, False, 1.0)
        assert result >= 0.25

    def test_open_minute_higher_than_midday(self):
        lookup = np.zeros(390, dtype=np.float64)
        lookup[0] = 3.0
        lookup[120] = 0.75
        open_slip = compute_slippage(0, 5.0, 5.0, lookup, False, 1.0)
        mid_slip = compute_slippage(120, 5.0, 5.0, lookup, False, 1.0)
        assert open_slip > mid_slip


class TestBuildSlippageLookup:
    def test_returns_390_element_array(self):
        lookup = build_slippage_lookup(None)
        assert lookup.shape == (390,)
        assert lookup.dtype == np.float64

    def test_all_values_positive(self):
        lookup = build_slippage_lookup(None)
        assert np.all(lookup > 0)

    def test_open_higher_than_midday(self):
        lookup = build_slippage_lookup(None)
        assert lookup[0] > lookup[120]

    def test_require_file_raises_when_profile_missing(self, tmp_path):
        missing = tmp_path / "missing_profile.parquet"
        with pytest.raises(FileNotFoundError):
            build_slippage_lookup(missing, require_file=True)
