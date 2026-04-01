from numba import njit


@njit(cache=True)
def check_circuit_breaker(intraday_pnl: float, daily_stop: float) -> bool:
    """Check if circuit breaker should fire.

    Args:
        intraday_pnl: Current day's PNL (negative = loss)
        daily_stop: Daily stop level (negative, e.g. -750.0)

    Returns:
        True if halted (PNL at or beyond stop), False otherwise.
    """
    return intraday_pnl <= daily_stop


@njit(cache=True)
def validate_position_size(requested: int, max_contracts: int) -> int:
    """Clamp position size to valid range [0, max_contracts]."""
    if requested < 0:
        return 0
    return min(requested, max_contracts)
