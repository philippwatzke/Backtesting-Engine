import pytest
from pathlib import Path

from propfirm.io.config import load_mff_config, load_params_config


CONFIGS_DIR = Path(__file__).parent.parent / "configs"


class TestLoadMFFConfig:
    def test_loads_eval_profit_target(self):
        cfg = load_mff_config(CONFIGS_DIR / "mff_flex_50k.toml")
        assert cfg["eval"]["profit_target"] == 3000.0

    def test_loads_funded_scaling_tiers(self):
        cfg = load_mff_config(CONFIGS_DIR / "mff_flex_50k.toml")
        tiers = cfg["funded"]["scaling"]["tiers"]
        assert len(tiers) == 3
        assert tiers[0]["max_contracts"] == 20
        assert tiers[2]["max_contracts"] == 50

    def test_loads_instrument_tick_size(self):
        cfg = load_mff_config(CONFIGS_DIR / "mff_flex_50k.toml")
        assert cfg["instrument"]["tick_size"] == 0.25
        assert cfg["instrument"]["commission_per_side"] == 0.54

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_mff_config(Path("/nonexistent/path.toml"))

    def test_rejects_invalid_profit_split(self, tmp_path):
        bad_cfg = tmp_path / "bad_mff.toml"
        bad_cfg.write_text(
            "[eval]\n"
            "profit_target = 3000.0\n"
            "max_loss_limit = 2000.0\n"
            "consistency_max_pct = 0.50\n"
            "min_trading_days = 2\n"
            "max_contracts = 50\n\n"
            "[funded]\n"
            "max_loss_limit = 2000.0\n"
            "mll_frozen_value = 100.0\n"
            "winning_day_threshold = 150.0\n"
            "payout_winning_days_required = 5\n"
            "payout_max_pct = 0.50\n"
            "payout_cap = 5000.0\n"
            "payout_min_gross = 250.0\n"
            "profit_split_trader = 1.20\n"
            "eval_cost = 107.0\n\n"
            "[[funded.scaling.tiers]]\n"
            "min_profit = 0.0\n"
            "max_profit = 1e9\n"
            "max_contracts = 20\n\n"
            "[instrument]\n"
            "name = \"MNQ\"\n"
            "tick_size = 0.25\n"
            "tick_value = 0.50\n"
            "commission_per_side = 0.54\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            load_mff_config(bad_cfg)

    def test_rejects_scaling_gap(self, tmp_path):
        bad_cfg = tmp_path / "bad_gap.toml"
        bad_cfg.write_text(
            "[eval]\n"
            "profit_target = 3000.0\n"
            "max_loss_limit = 2000.0\n"
            "consistency_max_pct = 0.50\n"
            "min_trading_days = 2\n"
            "max_contracts = 50\n\n"
            "[funded]\n"
            "max_loss_limit = 2000.0\n"
            "mll_frozen_value = 100.0\n"
            "winning_day_threshold = 150.0\n"
            "payout_winning_days_required = 5\n"
            "payout_max_pct = 0.50\n"
            "payout_cap = 5000.0\n"
            "payout_min_gross = 250.0\n"
            "profit_split_trader = 0.80\n"
            "eval_cost = 107.0\n\n"
            "[[funded.scaling.tiers]]\n"
            "min_profit = -1e9\n"
            "max_profit = 1500.0\n"
            "max_contracts = 20\n\n"
            "[[funded.scaling.tiers]]\n"
            "min_profit = 1600.0\n"
            "max_profit = 2000.0\n"
            "max_contracts = 30\n\n"
            "[instrument]\n"
            "name = \"MNQ\"\n"
            "tick_size = 0.25\n"
            "tick_value = 0.50\n"
            "commission_per_side = 0.54\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            load_mff_config(bad_cfg)

    def test_rejects_scaling_without_nonpositive_coverage(self, tmp_path):
        bad_cfg = tmp_path / "bad_floor.toml"
        bad_cfg.write_text(
            "[eval]\n"
            "profit_target = 3000.0\n"
            "max_loss_limit = 2000.0\n"
            "consistency_max_pct = 0.50\n"
            "min_trading_days = 2\n"
            "max_contracts = 50\n\n"
            "[funded]\n"
            "max_loss_limit = 2000.0\n"
            "mll_frozen_value = 100.0\n"
            "winning_day_threshold = 150.0\n"
            "payout_winning_days_required = 5\n"
            "payout_max_pct = 0.50\n"
            "payout_cap = 5000.0\n"
            "payout_min_gross = 250.0\n"
            "profit_split_trader = 0.80\n"
            "eval_cost = 107.0\n\n"
            "[[funded.scaling.tiers]]\n"
            "min_profit = 100.0\n"
            "max_profit = 1500.0\n"
            "max_contracts = 20\n\n"
            "[[funded.scaling.tiers]]\n"
            "min_profit = 1500.0\n"
            "max_profit = 1e9\n"
            "max_contracts = 30\n\n"
            "[instrument]\n"
            "name = \"MNQ\"\n"
            "tick_size = 0.25\n"
            "tick_value = 0.50\n"
            "commission_per_side = 0.54\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            load_mff_config(bad_cfg)

    def test_rejects_scaling_without_open_ended_last_tier(self, tmp_path):
        bad_cfg = tmp_path / "bad_ceiling.toml"
        bad_cfg.write_text(
            "[eval]\n"
            "profit_target = 3000.0\n"
            "max_loss_limit = 2000.0\n"
            "consistency_max_pct = 0.50\n"
            "min_trading_days = 2\n"
            "max_contracts = 50\n\n"
            "[funded]\n"
            "max_loss_limit = 2000.0\n"
            "mll_frozen_value = 100.0\n"
            "winning_day_threshold = 150.0\n"
            "payout_winning_days_required = 5\n"
            "payout_max_pct = 0.50\n"
            "payout_cap = 5000.0\n"
            "payout_min_gross = 250.0\n"
            "profit_split_trader = 0.80\n"
            "eval_cost = 107.0\n\n"
            "[[funded.scaling.tiers]]\n"
            "min_profit = -1e9\n"
            "max_profit = 1500.0\n"
            "max_contracts = 20\n\n"
            "[[funded.scaling.tiers]]\n"
            "min_profit = 1500.0\n"
            "max_profit = 2000.0\n"
            "max_contracts = 30\n\n"
            "[[funded.scaling.tiers]]\n"
            "min_profit = 2000.0\n"
            "max_profit = 2500.0\n"
            "max_contracts = 50\n\n"
            "[instrument]\n"
            "name = \"MNQ\"\n"
            "tick_size = 0.25\n"
            "tick_value = 0.50\n"
            "commission_per_side = 0.54\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            load_mff_config(bad_cfg)

    def test_rejects_non_integer_min_trading_days(self, tmp_path):
        bad_cfg = tmp_path / "bad_mff_float_day.toml"
        bad_cfg.write_text(
            "[eval]\n"
            "profit_target = 3000.0\n"
            "max_loss_limit = 2000.0\n"
            "consistency_max_pct = 0.50\n"
            "min_trading_days = 2.5\n"
            "max_contracts = 50\n\n"
            "[funded]\n"
            "max_loss_limit = 2000.0\n"
            "mll_frozen_value = 100.0\n"
            "winning_day_threshold = 150.0\n"
            "payout_winning_days_required = 5\n"
            "payout_max_pct = 0.50\n"
            "payout_cap = 5000.0\n"
            "payout_min_gross = 250.0\n"
            "profit_split_trader = 0.80\n"
            "eval_cost = 107.0\n\n"
            "[[funded.scaling.tiers]]\n"
            "min_profit = -1e9\n"
            "max_profit = 1e9\n"
            "max_contracts = 20\n\n"
            "[instrument]\n"
            "name = \"MNQ\"\n"
            "tick_size = 0.25\n"
            "tick_value = 0.50\n"
            "commission_per_side = 0.54\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="integer"):
            load_mff_config(bad_cfg)


class TestLoadParamsConfig:
    def test_loads_random_seed(self):
        cfg = load_params_config(CONFIGS_DIR / "default_params.toml")
        assert cfg["general"]["random_seed"] == 42

    def test_loads_monte_carlo_settings(self):
        cfg = load_params_config(CONFIGS_DIR / "default_params.toml")
        mc = cfg["monte_carlo"]
        assert mc["n_simulations"] == 10_000
        assert mc["block_size_min"] == 5

    def test_loads_strategy_orb_eval_and_funded(self):
        cfg = load_params_config(CONFIGS_DIR / "default_params.toml")
        shared = cfg["strategy"]["orb"]["shared"]
        eval_cfg = cfg["strategy"]["orb"]["eval"]
        funded_cfg = cfg["strategy"]["orb"]["funded"]
        assert shared["range_minutes"] == 15
        assert eval_cfg["daily_stop"] == -750.0
        assert funded_cfg["contracts"] == 20

    def test_rejects_invalid_block_mode(self, tmp_path):
        bad_cfg = tmp_path / "bad_params.toml"
        bad_cfg.write_text(
            "[general]\n"
            "random_seed = 42\n\n"
            "[strategy.orb.shared]\n"
            "range_minutes = 15\n"
            "max_trades_day = 2\n"
            "buffer_ticks = 2.0\n"
            "volume_threshold = 0.0\n\n"
            "[strategy.orb.eval]\n"
            "stop_ticks = 40.0\n"
            "target_ticks = 60.0\n"
            "contracts = 10\n"
            "daily_stop = -750.0\n"
            "daily_target = 600.0\n\n"
            "[strategy.orb.funded]\n"
            "stop_ticks = 35.0\n"
            "target_ticks = 80.0\n"
            "contracts = 20\n"
            "daily_stop = -1000.0\n"
            "daily_target = 900.0\n\n"
            "[slippage]\n"
            "stop_penalty = 1.5\n"
            "atr_period = 14\n"
            "trailing_atr_days = 5\n\n"
            "[monte_carlo]\n"
            "n_simulations = 1000\n"
            "block_mode = \"hourly\"\n"
            "block_size_min = 5\n"
            "block_size_max = 10\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            load_params_config(bad_cfg)

    def test_rejects_block_size_inversion(self, tmp_path):
        bad_cfg = tmp_path / "bad_params.toml"
        bad_cfg.write_text(
            "[general]\n"
            "random_seed = 42\n\n"
            "[strategy.orb.shared]\n"
            "range_minutes = 15\n"
            "max_trades_day = 2\n"
            "buffer_ticks = 2.0\n"
            "volume_threshold = 0.0\n\n"
            "[strategy.orb.eval]\n"
            "stop_ticks = 40.0\n"
            "target_ticks = 60.0\n"
            "contracts = 10\n"
            "daily_stop = -750.0\n"
            "daily_target = 600.0\n\n"
            "[strategy.orb.funded]\n"
            "stop_ticks = 35.0\n"
            "target_ticks = 80.0\n"
            "contracts = 20\n"
            "daily_stop = -1000.0\n"
            "daily_target = 900.0\n\n"
            "[slippage]\n"
            "stop_penalty = 1.5\n"
            "atr_period = 14\n"
            "trailing_atr_days = 5\n\n"
            "[monte_carlo]\n"
            "n_simulations = 1000\n"
            "block_mode = \"fixed\"\n"
            "block_size_min = 10\n"
            "block_size_max = 5\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            load_params_config(bad_cfg)

    def test_rejects_missing_funded_phase_block(self, tmp_path):
        bad_cfg = tmp_path / "bad_params.toml"
        bad_cfg.write_text(
            "[general]\n"
            "random_seed = 42\n\n"
            "[strategy.orb.shared]\n"
            "range_minutes = 15\n"
            "max_trades_day = 2\n"
            "buffer_ticks = 2.0\n"
            "volume_threshold = 0.0\n\n"
            "[strategy.orb.eval]\n"
            "stop_ticks = 40.0\n"
            "target_ticks = 60.0\n"
            "contracts = 10\n"
            "daily_stop = -750.0\n"
            "daily_target = 600.0\n\n"
            "[slippage]\n"
            "stop_penalty = 1.5\n"
            "atr_period = 14\n"
            "trailing_atr_days = 5\n\n"
            "[monte_carlo]\n"
            "n_simulations = 1000\n"
            "block_mode = \"daily\"\n"
            "block_size_min = 5\n"
            "block_size_max = 10\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            load_params_config(bad_cfg)

    def test_rejects_non_integer_contracts(self, tmp_path):
        bad_cfg = tmp_path / "bad_params_float_contracts.toml"
        bad_cfg.write_text(
            "[general]\n"
            "random_seed = 42\n\n"
            "[strategy.orb.shared]\n"
            "range_minutes = 15\n"
            "max_trades_day = 2\n"
            "buffer_ticks = 2.0\n"
            "volume_threshold = 0.0\n\n"
            "[strategy.orb.eval]\n"
            "stop_ticks = 40.0\n"
            "target_ticks = 60.0\n"
            "contracts = 10.5\n"
            "daily_stop = -750.0\n"
            "daily_target = 600.0\n\n"
            "[strategy.orb.funded]\n"
            "stop_ticks = 35.0\n"
            "target_ticks = 80.0\n"
            "contracts = 20\n"
            "daily_stop = -1000.0\n"
            "daily_target = 900.0\n\n"
            "[slippage]\n"
            "stop_penalty = 1.5\n"
            "atr_period = 14\n"
            "trailing_atr_days = 5\n\n"
            "[monte_carlo]\n"
            "n_simulations = 1000\n"
            "block_mode = \"daily\"\n"
            "block_size_min = 5\n"
            "block_size_max = 10\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="integer"):
            load_params_config(bad_cfg)
