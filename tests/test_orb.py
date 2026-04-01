import numpy as np
import pytest
from propfirm.strategy.orb import orb_signal


class TestORBSignal:
    def _make_arrays(self, n, base=20000.0):
        opens = np.full(n, base, dtype=np.float64)
        highs = np.full(n, base + 2.0, dtype=np.float64)
        lows = np.full(n, base - 2.0, dtype=np.float64)
        closes = np.full(n, base, dtype=np.float64)
        volumes = np.full(n, 1000, dtype=np.uint64)
        mod = np.arange(n, dtype=np.int16)
        params = np.array([15.0, 40.0, 60.0, 10.0, -750.0, 600.0, 2.0, 2.0, 0.0, 1.5, 0.54],
                          dtype=np.float64)
        return opens, highs, lows, closes, volumes, mod, params

    def test_no_signal_during_range_buildup(self):
        n = 390
        o, h, l, c, v, mod, p = self._make_arrays(n)
        sig = orb_signal(5, o, h, l, c, v, mod, 0.0, 0.0, 0, 0.0, False, 0, p)
        assert sig == 0

    def test_long_breakout(self):
        n = 390
        o, h, l, c, v, mod, p = self._make_arrays(n, 20000.0)
        c[15] = 20005.0
        h[15] = 20006.0
        sig = orb_signal(15, o, h, l, c, v, mod, 0.0, 0.0, 0, 0.0, False, 0, p)
        assert sig == 1

    def test_short_breakout(self):
        n = 390
        o, h, l, c, v, mod, p = self._make_arrays(n, 20000.0)
        c[15] = 19995.0
        l[15] = 19994.0
        sig = orb_signal(15, o, h, l, c, v, mod, 0.0, 0.0, 0, 0.0, False, 0, p)
        assert sig == -1

    def test_no_signal_when_halted(self):
        n = 390
        o, h, l, c, v, mod, p = self._make_arrays(n)
        c[15] = 20010.0
        h[15] = 20010.0
        sig = orb_signal(15, o, h, l, c, v, mod, 0.0, 0.0, 0, 0.0, True, 0, p)
        assert sig == 0

    def test_no_signal_when_daily_target_reached(self):
        n = 390
        o, h, l, c, v, mod, p = self._make_arrays(n)
        c[15] = 20010.0
        h[15] = 20010.0
        sig = orb_signal(15, o, h, l, c, v, mod, 0.0, 700.0, 0, 0.0, False, 0, p)
        assert sig == 0

    def test_no_signal_when_max_trades_reached(self):
        n = 390
        o, h, l, c, v, mod, p = self._make_arrays(n)
        c[15] = 20010.0
        h[15] = 20010.0
        sig = orb_signal(15, o, h, l, c, v, mod, 0.0, 0.0, 0, 0.0, False, 2, p)
        assert sig == 0

    def test_no_signal_when_position_open(self):
        n = 390
        o, h, l, c, v, mod, p = self._make_arrays(n)
        c[15] = 20010.0
        sig = orb_signal(15, o, h, l, c, v, mod, 0.0, 0.0, 5, 0.0, False, 0, p)
        assert sig == 0
