def compute_single_payout(
    total_profit: float,
    max_pct: float,
    payout_cap: float,
    payout_min_gross: float,
    profit_split: float,
) -> float:
    """Compute net payout for a single payout event."""
    gross = min(total_profit * max_pct, payout_cap)
    if gross < payout_min_gross:
        return 0.0
    return gross * profit_split


def compute_capped_nve(
    payout_rate: float,
    mean_payout_net: float,
    eval_cost: float,
) -> float:
    """Compute Net Expected Value.

    NVE = payout_rate * E[payout_net] - eval_cost
    """
    return payout_rate * mean_payout_net - eval_cost
