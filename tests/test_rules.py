import pytest
from propfirm.rules.mff import MFFState


def make_eval_config():
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
            "profit_split_trader": 0.80,
            "eval_cost": 107.0,
            "scaling": {
                "tiers": [
                    {"min_profit": -1e9, "max_profit": 1500.0, "max_contracts": 20},
                    {"min_profit": 1500.0, "max_profit": 2000.0, "max_contracts": 30},
                    {"min_profit": 2000.0, "max_profit": 1e9, "max_contracts": 50},
                ],
            },
        },
    }


class TestMFFStateInit:
    def test_starts_in_eval_phase(self):
        state = MFFState(make_eval_config())
        assert state.phase == "eval"
        assert state.equity == 0.0
        assert state.trading_days == 0
        assert state.drawdown_high_watermark == 0.0
        assert state.static_floor_equity is None

    def test_initial_mll(self):
        state = MFFState(make_eval_config())
        assert state.mll == 2000.0
        assert state.mll_frozen == False


class TestEODTrailingDrawdown:
    def test_eod_high_updates_on_new_high(self):
        state = MFFState(make_eval_config())
        state.update_eod(500.0, 500.0)
        assert state.eod_high == 500.0
        state.update_eod(300.0, 800.0)
        assert state.eod_high == 800.0

    def test_eod_high_does_not_decrease(self):
        state = MFFState(make_eval_config())
        state.update_eod(500.0, 500.0)
        state.update_eod(-200.0, 300.0)
        assert state.eod_high == 500.0

    def test_blown_when_equity_below_trailing_dd(self):
        state = MFFState(make_eval_config())
        state.update_eod(1500.0, 1500.0)  # eod_high = 1500
        result = state.update_eod(-2100.0, -600.0)  # 1500 - (-600) = 2100 > 2000
        assert result == "blown"

    def test_continue_when_within_drawdown(self):
        state = MFFState(make_eval_config())
        state.update_eod(1000.0, 1000.0)
        result = state.update_eod(-500.0, 500.0)  # 1000 - 500 = 500 < 2000
        assert result == "continue"


class TestEvalPassCondition:
    def test_passed_when_target_reached_with_enough_days(self):
        state = MFFState(make_eval_config())
        state.update_eod(1500.0, 1500.0)
        result = state.update_eod(1500.0, 3000.0)  # total >= 3000, 2 days, consistency 0.50 OK
        assert result == "passed"

    def test_not_passed_with_only_one_day(self):
        state = MFFState(make_eval_config())
        result = state.update_eod(3500.0, 3500.0)  # > 3000 but only 1 day
        assert result == "continue"


class TestConsistencyRule:
    def test_consistency_violation_detected(self):
        state = MFFState(make_eval_config())
        state.update_eod(100.0, 100.0)
        state.update_eod(100.0, 200.0)
        result = state.update_eod(2900.0, 3100.0)
        # Profit target met, min days met, but consistency fails -> not passed
        assert state.consistency_ok() == False
        assert result == "continue"


class TestFundedPhase:
    def test_negative_profit_uses_first_scaling_tier(self):
        state = MFFState(make_eval_config())
        state.phase = "funded"
        state.total_profit = -250.0
        assert state.get_max_contracts() == 20

    def test_scaling_tier_0_to_1499(self):
        state = MFFState(make_eval_config())
        state.phase = "funded"
        state.equity = 1000.0
        state.total_profit = 1000.0
        assert state.get_max_contracts() == 20

    def test_scaling_tier_1500_to_1999(self):
        state = MFFState(make_eval_config())
        state.phase = "funded"
        state.total_profit = 1700.0
        assert state.get_max_contracts() == 30

    def test_scaling_tier_2000_plus(self):
        state = MFFState(make_eval_config())
        state.phase = "funded"
        state.total_profit = 2500.0
        assert state.get_max_contracts() == 50

    def test_mll_freezes_after_first_payout(self):
        state = MFFState(make_eval_config())
        state.transition_to_funded()
        state.equity = 2000.0
        state.total_profit = 2000.0
        state.winning_days = 5
        state.process_payout()
        assert state.mll == 100.0
        assert state.static_floor_equity == 900.0

    def test_payout_eligibility(self):
        state = MFFState(make_eval_config())
        state.phase = "funded"
        state.winning_days = 5
        state.total_profit = 600.0
        assert state.payout_eligible == True

    def test_payout_not_eligible_insufficient_days(self):
        state = MFFState(make_eval_config())
        state.phase = "funded"
        state.winning_days = 4
        state.total_profit = 600.0
        assert state.payout_eligible == False


class TestPhaseShift:
    def test_get_active_params_eval(self):
        state = MFFState(make_eval_config())
        assert state.get_active_params() == "params_eval"

    def test_get_active_params_funded(self):
        state = MFFState(make_eval_config())
        state.transition_to_funded()
        assert state.get_active_params() == "params_funded"

    def test_transition_resets_all_funded_fields(self):
        state = MFFState(make_eval_config())
        state.update_eod(1500.0, 1500.0)
        state.update_eod(1600.0, 3100.0)
        assert state.total_profit == 3100.0
        assert state.eod_high == 3100.0
        assert state.trading_days == 2
        assert state.winning_days == 2

        state.transition_to_funded()
        assert state.phase == "funded"
        assert state.equity == 0.0
        assert state.eod_high == 0.0
        assert state.total_profit == 0.0
        assert state.max_single_day_profit == 0.0
        assert state.daily_profits == []
        assert state.trading_days == 0
        assert state.winning_days == 0
        assert state.mll_frozen == False
        assert state.payouts_completed == 0
        assert state.mll == 2000.0
        assert state.drawdown_high_watermark == 0.0
        assert state.static_floor_equity is None


class TestWinningDayTracking:
    def test_winning_day_counted(self):
        state = MFFState(make_eval_config())
        state.update_eod(200.0, 200.0)
        assert state.winning_days == 1

    def test_day_below_threshold_not_counted(self):
        state = MFFState(make_eval_config())
        state.update_eod(100.0, 100.0)
        assert state.winning_days == 0

    def test_losing_day_not_counted(self):
        state = MFFState(make_eval_config())
        state.update_eod(-300.0, -300.0)
        assert state.winning_days == 0


class TestStaticFloorAfterPayout:
    def test_static_floor_does_not_rise_with_new_highs(self):
        state = MFFState(make_eval_config())
        state.transition_to_funded()
        state.equity = 2000.0
        state.total_profit = 2000.0
        state.winning_days = 5
        state.process_payout()
        assert state.static_floor_equity == 900.0

        result = state.update_eod(400.0, 1400.0)
        assert result == "continue"
        assert state.static_floor_equity == 900.0

        result = state.update_eod(-550.0, 850.0)
        assert result == "blown"

    def test_process_payout_reduces_equity_and_total_profit(self):
        state = MFFState(make_eval_config())
        state.transition_to_funded()
        state.equity = 2000.0
        state.total_profit = 2000.0
        state.winning_days = 5
        net = state.process_payout()
        assert net == 800.0
        assert state.equity == 1000.0
        assert state.total_profit == 1000.0
