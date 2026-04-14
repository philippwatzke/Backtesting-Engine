import numpy as np

from propfirm.core.types import (
    PARAMS_ARRAY_LENGTH,
    PARAMS_BLOCKED_WEEKDAY,
    PARAMS_DAILY_TARGET,
    PARAMS_MAX_TRADES,
    PARAMS_POC_LOOKBACK,
    PARAMS_TRIGGER_END_MINUTE,
    PARAMS_TRIGGER_START_MINUTE,
)
from propfirm.strategy.mgc_h1_trend_strategy import mgc_h1_trend_signal


class TestMGCH1TrendSignal:
    def _make_params(self):
        params = np.zeros(PARAMS_ARRAY_LENGTH, dtype=np.float64)
        params[PARAMS_DAILY_TARGET] = 600.0
        params[PARAMS_MAX_TRADES] = 1.0
        params[PARAMS_POC_LOOKBACK] = 5.0
        params[PARAMS_TRIGGER_START_MINUTE] = 60.0
        params[PARAMS_TRIGGER_END_MINUTE] = 360.0
        return params

    def _make_minutes(self, n, active_minute=120):
        return np.full(n, active_minute, dtype=np.int16)

    def _make_day_of_week(self, n, weekday=1):
        return np.full(n, weekday, dtype=np.int8)

    def _make_sma(self, closes):
        sma = np.full(len(closes), np.nan, dtype=np.float64)
        for idx in range(49, len(closes)):
            sma[idx] = np.mean(closes[idx - 49:idx + 1])
        return sma

    def _make_donchian(self, highs, lows):
        d_high = np.full(len(highs), np.nan, dtype=np.float64)
        d_low = np.full(len(lows), np.nan, dtype=np.float64)
        for idx in range(5, len(highs)):
            d_high[idx] = np.max(highs[idx - 5:idx])
            d_low[idx] = np.min(lows[idx - 5:idx])
        return d_high, d_low

    def test_long_signal_on_close_above_prior_5_bar_high_and_sma50(self):
        n = 60
        o = np.full(n, 2400.0, dtype=np.float64)
        c = 2400.0 + np.arange(n, dtype=np.float64) * 0.5
        h = c + 0.4
        l = c - 0.6
        c[55] = c[54] + 3.0
        h[55] = c[55] + 0.4
        l[55] = c[55] - 0.6
        v = np.full(n, 1000, dtype=np.uint64)
        atr = np.ones(n, dtype=np.float64)
        trail = np.ones(n, dtype=np.float64)
        daily_ratio = np.ones(n, dtype=np.float64)
        rvol = np.ones(n, dtype=np.float64)
        close_sma_50 = self._make_sma(c)
        daily_regime_bias = np.full(n, np.nan, dtype=np.float64)
        donchian_high_5, donchian_low_5 = self._make_donchian(h, l)
        mod = self._make_minutes(n, 120)
        dow = self._make_day_of_week(n)

        sig = mgc_h1_trend_signal(
            55, o, h, l, c, v, atr, trail, daily_ratio, rvol, close_sma_50,
            daily_regime_bias,
            donchian_high_5, donchian_low_5, mod, dow,
            0.0, 0.0, 0, 0.0, False, 0, self._make_params()
        )
        assert sig == 1

    def test_short_signal_on_close_below_prior_5_bar_low_and_sma50(self):
        n = 60
        o = np.full(n, 2400.0, dtype=np.float64)
        c = 2450.0 - np.arange(n, dtype=np.float64) * 0.5
        h = c + 0.6
        l = c - 0.4
        c[55] = c[54] - 3.0
        h[55] = c[55] + 0.6
        l[55] = c[55] - 0.4
        v = np.full(n, 1000, dtype=np.uint64)
        atr = np.ones(n, dtype=np.float64)
        trail = np.ones(n, dtype=np.float64)
        daily_ratio = np.ones(n, dtype=np.float64)
        rvol = np.ones(n, dtype=np.float64)
        close_sma_50 = self._make_sma(c)
        daily_regime_bias = np.full(n, np.nan, dtype=np.float64)
        donchian_high_5, donchian_low_5 = self._make_donchian(h, l)
        mod = self._make_minutes(n, 120)
        dow = self._make_day_of_week(n)

        sig = mgc_h1_trend_signal(
            55, o, h, l, c, v, atr, trail, daily_ratio, rvol, close_sma_50,
            daily_regime_bias,
            donchian_high_5, donchian_low_5, mod, dow,
            0.0, 0.0, 0, 0.0, False, 0, self._make_params()
        )
        assert sig == -1

    def test_blocks_before_sma50_history_exists(self):
        n = 40
        o = np.full(n, 2400.0, dtype=np.float64)
        h = np.full(n, 2401.0, dtype=np.float64)
        l = np.full(n, 2399.0, dtype=np.float64)
        c = np.full(n, 2400.5, dtype=np.float64)
        v = np.full(n, 1000, dtype=np.uint64)
        atr = np.ones(n, dtype=np.float64)
        trail = np.ones(n, dtype=np.float64)
        daily_ratio = np.ones(n, dtype=np.float64)
        rvol = np.ones(n, dtype=np.float64)
        close_sma_50 = self._make_sma(c)
        daily_regime_bias = np.full(n, np.nan, dtype=np.float64)
        donchian_high_5, donchian_low_5 = self._make_donchian(h, l)
        mod = self._make_minutes(n, 120)
        dow = self._make_day_of_week(n)

        sig = mgc_h1_trend_signal(
            39, o, h, l, c, v, atr, trail, daily_ratio, rvol, close_sma_50,
            daily_regime_bias,
            donchian_high_5, donchian_low_5, mod, dow,
            0.0, 0.0, 0, 0.0, False, 0, self._make_params()
        )
        assert sig == 0

    def test_blocks_breakout_when_close_is_below_sma50(self):
        n = 60
        o = np.full(n, 2400.0, dtype=np.float64)
        c = np.full(n, 2400.0, dtype=np.float64)
        h = np.full(n, 2400.5, dtype=np.float64)
        l = np.full(n, 2399.5, dtype=np.float64)
        c[:55] = 2500.0
        h[:55] = 2500.5
        l[:55] = 2490.0
        c[55] = 2496.0
        h[55] = 2501.0
        l[55] = 2495.5
        v = np.full(n, 1000, dtype=np.uint64)
        atr = np.ones(n, dtype=np.float64)
        trail = np.ones(n, dtype=np.float64)
        daily_ratio = np.ones(n, dtype=np.float64)
        rvol = np.ones(n, dtype=np.float64)
        close_sma_50 = self._make_sma(c)
        daily_regime_bias = np.full(n, np.nan, dtype=np.float64)
        donchian_high_5, donchian_low_5 = self._make_donchian(h, l)
        mod = self._make_minutes(n, 120)
        dow = self._make_day_of_week(n)

        sig = mgc_h1_trend_signal(
            55, o, h, l, c, v, atr, trail, daily_ratio, rvol, close_sma_50,
            daily_regime_bias,
            donchian_high_5, donchian_low_5, mod, dow,
            0.0, 0.0, 0, 0.0, False, 0, self._make_params()
        )
        assert sig == 0

    def test_blocks_entries_before_0900_et(self):
        n = 60
        o = np.full(n, 2400.0, dtype=np.float64)
        c = 2400.0 + np.arange(n, dtype=np.float64) * 0.5
        h = c + 0.4
        l = c - 0.6
        c[55] = c[54] + 3.0
        h[55] = c[55] + 0.4
        l[55] = c[55] - 0.6
        v = np.full(n, 1000, dtype=np.uint64)
        atr = np.ones(n, dtype=np.float64)
        trail = np.ones(n, dtype=np.float64)
        daily_ratio = np.ones(n, dtype=np.float64)
        rvol = np.ones(n, dtype=np.float64)
        close_sma_50 = self._make_sma(c)
        daily_regime_bias = np.full(n, np.nan, dtype=np.float64)
        donchian_high_5, donchian_low_5 = self._make_donchian(h, l)
        mod = self._make_minutes(n, 0)
        dow = self._make_day_of_week(n)

        sig = mgc_h1_trend_signal(
            55, o, h, l, c, v, atr, trail, daily_ratio, rvol, close_sma_50,
            daily_regime_bias,
            donchian_high_5, donchian_low_5, mod, dow,
            0.0, 0.0, 0, 0.0, False, 0, self._make_params()
        )
        assert sig == 0

    def test_blocks_entries_after_1400_et(self):
        n = 60
        o = np.full(n, 2400.0, dtype=np.float64)
        c = 2400.0 + np.arange(n, dtype=np.float64) * 0.5
        h = c + 0.4
        l = c - 0.6
        c[55] = c[54] + 3.0
        h[55] = c[55] + 0.4
        l[55] = c[55] - 0.6
        v = np.full(n, 1000, dtype=np.uint64)
        atr = np.ones(n, dtype=np.float64)
        trail = np.ones(n, dtype=np.float64)
        daily_ratio = np.ones(n, dtype=np.float64)
        rvol = np.ones(n, dtype=np.float64)
        close_sma_50 = self._make_sma(c)
        daily_regime_bias = np.full(n, np.nan, dtype=np.float64)
        donchian_high_5, donchian_low_5 = self._make_donchian(h, l)
        mod = self._make_minutes(n, 420)
        dow = self._make_day_of_week(n)

        sig = mgc_h1_trend_signal(
            55, o, h, l, c, v, atr, trail, daily_ratio, rvol, close_sma_50,
            daily_regime_bias,
            donchian_high_5, donchian_low_5, mod, dow,
            0.0, 0.0, 0, 0.0, False, 0, self._make_params()
        )
        assert sig == 0

    def test_blocks_long_when_daily_regime_is_short(self):
        n = 60
        o = np.full(n, 2400.0, dtype=np.float64)
        c = 2400.0 + np.arange(n, dtype=np.float64) * 0.5
        h = c + 0.4
        l = c - 0.6
        c[55] = c[54] + 3.0
        h[55] = c[55] + 0.4
        l[55] = c[55] - 0.6
        v = np.full(n, 1000, dtype=np.uint64)
        atr = np.ones(n, dtype=np.float64)
        trail = np.ones(n, dtype=np.float64)
        daily_ratio = np.ones(n, dtype=np.float64)
        rvol = np.ones(n, dtype=np.float64)
        close_sma_50 = self._make_sma(c)
        daily_regime_bias = np.full(n, -1.0, dtype=np.float64)
        donchian_high_5, donchian_low_5 = self._make_donchian(h, l)
        mod = self._make_minutes(n, 120)
        dow = self._make_day_of_week(n)

        sig = mgc_h1_trend_signal(
            55, o, h, l, c, v, atr, trail, daily_ratio, rvol, close_sma_50,
            daily_regime_bias, donchian_high_5, donchian_low_5, mod, dow,
            0.0, 0.0, 0, 0.0, False, 0, self._make_params()
        )
        assert sig == 0

    def test_blocks_short_when_daily_regime_is_long(self):
        n = 60
        o = np.full(n, 2400.0, dtype=np.float64)
        c = 2450.0 - np.arange(n, dtype=np.float64) * 0.5
        h = c + 0.6
        l = c - 0.4
        c[55] = c[54] - 3.0
        h[55] = c[55] + 0.6
        l[55] = c[55] - 0.4
        v = np.full(n, 1000, dtype=np.uint64)
        atr = np.ones(n, dtype=np.float64)
        trail = np.ones(n, dtype=np.float64)
        daily_ratio = np.ones(n, dtype=np.float64)
        rvol = np.ones(n, dtype=np.float64)
        close_sma_50 = self._make_sma(c)
        daily_regime_bias = np.full(n, 1.0, dtype=np.float64)
        donchian_high_5, donchian_low_5 = self._make_donchian(h, l)
        mod = self._make_minutes(n, 120)
        dow = self._make_day_of_week(n)

        sig = mgc_h1_trend_signal(
            55, o, h, l, c, v, atr, trail, daily_ratio, rvol, close_sma_50,
            daily_regime_bias, donchian_high_5, donchian_low_5, mod, dow,
            0.0, 0.0, 0, 0.0, False, 0, self._make_params()
        )
        assert sig == 0

    def test_blocks_signals_on_blocked_weekday(self):
        n = 60
        o = np.full(n, 2400.0, dtype=np.float64)
        c = 2400.0 + np.arange(n, dtype=np.float64) * 0.5
        h = c + 0.4
        l = c - 0.6
        c[55] = c[54] + 3.0
        h[55] = c[55] + 0.4
        l[55] = c[55] - 0.6
        v = np.full(n, 1000, dtype=np.uint64)
        atr = np.ones(n, dtype=np.float64)
        trail = np.ones(n, dtype=np.float64)
        daily_ratio = np.ones(n, dtype=np.float64)
        rvol = np.ones(n, dtype=np.float64)
        close_sma_50 = self._make_sma(c)
        daily_regime_bias = np.full(n, np.nan, dtype=np.float64)
        donchian_high_5, donchian_low_5 = self._make_donchian(h, l)
        mod = self._make_minutes(n, 120)
        dow = self._make_day_of_week(n, 2)
        params = self._make_params()
        params[PARAMS_BLOCKED_WEEKDAY] = 2.0

        sig = mgc_h1_trend_signal(
            55, o, h, l, c, v, atr, trail, daily_ratio, rvol, close_sma_50,
            daily_regime_bias, donchian_high_5, donchian_low_5, mod, dow,
            0.0, 0.0, 0, 0.0, False, 0, params
        )
        assert sig == 0
