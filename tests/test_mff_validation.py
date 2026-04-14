"""Comprehensive MFF Flex $50k rule validation.

These 21 scenarios validate every rule path in the MFF state machine
before any optimization code touches the simulator. This is the firewall
between the simulator core and the optimization layer.
"""
import pytest
from propfirm.rules.mff import MFFState


def make_config():
    return {
        "eval": {
            "profit_target": 3000.0,
            "max_loss_limit": 2000.0,
            "consistency_max_pct": 0.50,
            "min_trading_days": 2,
            "max_contracts": 50,
        },
        "funded": {
            "max_loss_limit": 2000.0,
            "mll_frozen_value": 100.0,
            "winning_day_threshold": 150.0,
            "payout_winning_days_required": 5,
            "payout_max_pct": 0.50,
            "payout_cap": 5000.0,
            "payout_min_gross": 250.0,
            "payout_min_net_profit_between_payouts": 500.0,
            "profit_split_trader": 0.80,
            "eval_cost": 107.0,
            "inactivity_rule_calendar_days": 7,
            "live_transition_payouts_required": 5,
            "live_sim_cap_profit": 100000.0,
            "scaling": {
                "tiers": [
                    {"min_profit": -1e9, "max_profit": 1500.0, "max_contracts": 20},
                    {"min_profit": 1500.0, "max_profit": 2000.0, "max_contracts": 30},
                    {"min_profit": 2000.0, "max_profit": 1e9, "max_contracts": 50},
                ],
            },
        },
    }


# === EVAL PHASE: Pass Conditions ===

class TestEvalPass:
    def test_01_pass_after_exactly_2_days(self):
        """Minimum trading days: pass on day 2 with $3000+."""
        s = MFFState(make_config())
        s.update_eod(1550.0, 1550.0)
        result = s.update_eod(1550.0, 3100.0)
        assert result == "passed"
        assert s.trading_days == 2

    def test_02_no_pass_on_day_1_even_with_target(self):
        """Must have >= 2 trading days."""
        s = MFFState(make_config())
        result = s.update_eod(3500.0, 3500.0)
        assert result == "continue"

    def test_03_pass_exactly_at_target(self):
        """Edge: total profit == $3000 exactly."""
        s = MFFState(make_config())
        s.update_eod(1500.0, 1500.0)
        result = s.update_eod(1500.0, 3000.0)
        assert result == "passed"

    def test_03b_flat_day_does_not_count_toward_minimum_trading_days(self):
        """Only real trading days should satisfy the eval minimum-day rule."""
        s = MFFState(make_config())
        s.update_eod(3000.0, 3000.0, had_trade=True)
        result = s.update_eod(0.0, 3000.0, had_trade=False)
        assert result == "continue"
        assert s.trading_days == 1


# === EVAL PHASE: Blown Conditions ===

class TestEvalBlown:
    def test_04_blown_by_trailing_dd(self):
        """Equity drops >= $2000 below EOD high."""
        s = MFFState(make_config())
        s.update_eod(1800.0, 1800.0)
        result = s.update_eod(-2100.0, -300.0)
        assert result == "blown"

    def test_05_blown_at_exactly_mll(self):
        """Edge: drawdown == $2000 exactly SHOULD blow."""
        s = MFFState(make_config())
        s.update_eod(2000.0, 2000.0)
        result = s.update_eod(-2000.0, 0.0)
        assert result == "blown"

    def test_06_trailing_dd_tracks_eod_high(self):
        """EOD high never decreases, even after losing days."""
        s = MFFState(make_config())
        s.update_eod(1000.0, 1000.0)
        s.update_eod(-200.0, 800.0)
        assert s.eod_high == 1000.0
        s.update_eod(500.0, 1300.0)
        assert s.eod_high == 1300.0


# === CONSISTENCY RULE ===

class TestConsistency:
    def test_07_consistency_blocks_pass(self):
        """One day > 50% of total profit blocks eval pass."""
        s = MFFState(make_config())
        s.update_eod(100.0, 100.0)
        result = s.update_eod(2950.0, 3050.0)
        assert s.consistency_ok() == False
        assert result == "continue"

    def test_08_consistency_ok_with_even_days(self):
        """Two equal days: 50/50 should pass."""
        s = MFFState(make_config())
        s.update_eod(1500.0, 1500.0)
        result = s.update_eod(1500.0, 3000.0)
        assert s.consistency_ok() == True
        assert result == "passed"


# === FUNDED PHASE: Transition Reset ===

class TestFundedTransition:
    def test_09_transition_resets_eval_state(self):
        """Funded account starts completely fresh after eval pass."""
        s = MFFState(make_config())
        s.update_eod(1500.0, 1500.0)
        s.update_eod(1600.0, 3100.0)
        s.transition_to_funded()
        assert s.phase == "funded"
        assert s.equity == 0.0
        assert s.eod_high == 0.0
        assert s.total_profit == 0.0
        assert s.winning_days == 0
        assert s.trading_days == 0
        assert s.daily_profits == []


# === FUNDED PHASE: Scaling ===

class TestFundedScaling:
    def test_10_tier_1_below_1500(self):
        s = MFFState(make_config())
        s.transition_to_funded()
        s.total_profit = 500.0
        assert s.get_max_contracts() == 20

    def test_11_tier_2_at_1500(self):
        s = MFFState(make_config())
        s.transition_to_funded()
        s.total_profit = 1500.0
        assert s.get_max_contracts() == 30

    def test_12_tier_3_at_2000(self):
        s = MFFState(make_config())
        s.transition_to_funded()
        s.total_profit = 2000.0
        assert s.get_max_contracts() == 50


# === FUNDED PHASE: Payout Logic ===

class TestFundedPayout:
    def test_13_payout_basic(self):
        """Standard payout: 50% of profit, 80/20 split."""
        s = MFFState(make_config())
        s.transition_to_funded()
        s.total_profit = 2000.0
        s.cycle_net_profit = 2000.0
        s.winning_days = 5
        net = s.process_payout()
        assert net == 800.0
        assert s.total_profit == 1000.0
        assert s.cycle_net_profit == 0.0
        assert s.winning_days == 0

    def test_14_payout_below_minimum_returns_zero(self):
        s = MFFState(make_config())
        s.transition_to_funded()
        s.total_profit = 400.0
        s.cycle_net_profit = 400.0
        s.winning_days = 5
        net = s.process_payout()
        assert net == 0.0

    def test_15_payout_capped_at_5000(self):
        s = MFFState(make_config())
        s.transition_to_funded()
        s.total_profit = 20000.0
        s.cycle_net_profit = 20000.0
        s.winning_days = 5
        net = s.process_payout()
        assert net == 5000.0 * 0.80

    def test_16_mll_freezes_after_first_payout(self):
        s = MFFState(make_config())
        s.transition_to_funded()
        s.total_profit = 2000.0
        s.cycle_net_profit = 2000.0
        s.winning_days = 5
        s.process_payout()
        assert s.mll_frozen == True
        assert s.mll == 100.0
        assert s.static_floor_equity == 100.0

    def test_17_mll_stays_frozen_after_second_payout(self):
        s = MFFState(make_config())
        s.transition_to_funded()
        s.total_profit = 4000.0
        s.cycle_net_profit = 4000.0
        s.winning_days = 10
        s.process_payout()
        s.total_profit = 3000.0
        s.cycle_net_profit = 3000.0
        s.winning_days = 5
        s.process_payout()
        assert s.mll == 100.0


# === WINNING DAYS ===

class TestWinningDays:
    def test_18_winning_day_at_threshold(self):
        """Day with exactly $150 PNL counts."""
        s = MFFState(make_config())
        s.update_eod(150.0, 150.0)
        assert s.winning_days == 1

    def test_19_day_at_149_does_not_count(self):
        s = MFFState(make_config())
        s.update_eod(149.0, 149.0)
        assert s.winning_days == 0


# === PAYOUT ELIGIBILITY ===

class TestPayoutEligibility:
    def test_20_eligible_with_5_days_and_enough_profit(self):
        s = MFFState(make_config())
        s.transition_to_funded()
        s.winning_days = 5
        s.cycle_net_profit = 600.0
        assert s.payout_eligible == True

    def test_21_not_eligible_with_4_days(self):
        s = MFFState(make_config())
        s.transition_to_funded()
        s.winning_days = 4
        s.cycle_net_profit = 10000.0
        assert s.payout_eligible == False

    def test_22_not_eligible_without_new_cycle_profit_after_payout(self):
        s = MFFState(make_config())
        s.transition_to_funded()
        s.total_profit = 2000.0
        s.cycle_net_profit = 2000.0
        s.winning_days = 5
        s.process_payout()
        s.winning_days = 5
        assert s.payout_eligible == False


class TestLifecycleCompliance:
    def test_23_live_ready_after_five_payouts(self):
        s = MFFState(make_config())
        s.transition_to_funded()
        for _ in range(5):
            s.total_profit += 2000.0
            s.cycle_net_profit = 2000.0
            s.winning_days = 5
            s.process_payout()
        assert s.live_transition_ready == True
        assert s.live_transition_reason == "payouts"

    def test_24_inactivity_breaches_after_seven_calendar_days(self):
        s = MFFState(make_config())
        s.transition_to_funded()
        s.update_eod(200.0, 200.0, had_trade=True, session_date="2024-01-02")
        assert s.check_inactivity_before_session("2024-01-08") == "continue"
        assert s.check_inactivity_before_session("2024-01-09") == "inactive"
