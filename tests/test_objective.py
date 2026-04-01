import pytest
from propfirm.optim.objective import compute_capped_nve, compute_single_payout


class TestComputeSinglePayout:
    def test_basic_payout(self):
        result = compute_single_payout(2000.0, 0.50, 5000.0, 250.0, 0.80)
        assert result == 1000.0 * 0.80

    def test_capped_at_5000(self):
        result = compute_single_payout(20000.0, 0.50, 5000.0, 250.0, 0.80)
        assert result == 5000.0 * 0.80

    def test_below_minimum_returns_zero(self):
        result = compute_single_payout(400.0, 0.50, 5000.0, 250.0, 0.80)
        assert result == 0.0

    def test_at_exactly_minimum(self):
        result = compute_single_payout(500.0, 0.50, 5000.0, 250.0, 0.80)
        assert result == 250.0 * 0.80

    def test_zero_profit(self):
        result = compute_single_payout(0.0, 0.50, 5000.0, 250.0, 0.80)
        assert result == 0.0


class TestComputeCappedNVE:
    def test_positive_nve(self):
        nve = compute_capped_nve(payout_rate=0.70, mean_payout_net=800.0, eval_cost=107.0)
        assert abs(nve - 453.0) < 0.01

    def test_negative_nve(self):
        nve = compute_capped_nve(payout_rate=0.10, mean_payout_net=500.0, eval_cost=107.0)
        assert abs(nve - (-57.0)) < 0.01

    def test_zero_pass_rate(self):
        nve = compute_capped_nve(0.0, 800.0, 107.0)
        assert nve == -107.0
