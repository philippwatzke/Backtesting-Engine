import pytest
from propfirm.risk.risk import check_circuit_breaker, validate_position_size


class TestCircuitBreaker:
    def test_not_halted_within_limit(self):
        assert check_circuit_breaker(-500.0, -750.0) == False

    def test_halted_at_limit(self):
        assert check_circuit_breaker(-750.0, -750.0) == True

    def test_halted_beyond_limit(self):
        assert check_circuit_breaker(-1000.0, -750.0) == True

    def test_not_halted_positive_pnl(self):
        assert check_circuit_breaker(500.0, -750.0) == False


class TestPositionSizing:
    def test_within_limits(self):
        assert validate_position_size(10, 20) == 10

    def test_clamped_to_max(self):
        assert validate_position_size(30, 20) == 20

    def test_zero_contracts(self):
        assert validate_position_size(0, 20) == 0

    def test_negative_contracts_clamped(self):
        assert validate_position_size(-5, 20) == 0
