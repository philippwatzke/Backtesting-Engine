from pathlib import Path
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def _load_toml(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "rb") as f:
        return tomllib.load(f)


def _require_keys(mapping: dict, keys: list[str], ctx: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"{ctx} missing required keys: {missing}")


def _require_positive(value, ctx: str) -> None:
    if value <= 0:
        raise ValueError(f"{ctx} must be > 0")


def _require_non_negative(value, ctx: str) -> None:
    if value < 0:
        raise ValueError(f"{ctx} must be >= 0")


def _require_pct(value, ctx: str) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{ctx} must be between 0 and 1")


def _validate_mff_config(cfg: dict) -> None:
    _require_keys(cfg, ["eval", "funded", "instrument"], "root")
    eval_cfg = cfg["eval"]
    funded_cfg = cfg["funded"]
    instrument_cfg = cfg["instrument"]

    _require_keys(
        eval_cfg,
        ["profit_target", "max_loss_limit", "consistency_max_pct", "min_trading_days", "max_contracts"],
        "eval",
    )
    _require_positive(eval_cfg["profit_target"], "eval.profit_target")
    _require_positive(eval_cfg["max_loss_limit"], "eval.max_loss_limit")
    _require_pct(eval_cfg["consistency_max_pct"], "eval.consistency_max_pct")
    _require_positive(eval_cfg["min_trading_days"], "eval.min_trading_days")
    _require_positive(eval_cfg["max_contracts"], "eval.max_contracts")

    _require_keys(
        funded_cfg,
        [
            "max_loss_limit", "mll_frozen_value", "winning_day_threshold",
            "payout_winning_days_required", "payout_max_pct", "payout_cap",
            "payout_min_gross", "profit_split_trader", "eval_cost", "scaling",
        ],
        "funded",
    )
    _require_positive(funded_cfg["max_loss_limit"], "funded.max_loss_limit")
    _require_positive(funded_cfg["mll_frozen_value"], "funded.mll_frozen_value")
    _require_non_negative(funded_cfg["winning_day_threshold"], "funded.winning_day_threshold")
    _require_positive(funded_cfg["payout_winning_days_required"], "funded.payout_winning_days_required")
    _require_pct(funded_cfg["payout_max_pct"], "funded.payout_max_pct")
    _require_positive(funded_cfg["payout_cap"], "funded.payout_cap")
    _require_non_negative(funded_cfg["payout_min_gross"], "funded.payout_min_gross")
    _require_pct(funded_cfg["profit_split_trader"], "funded.profit_split_trader")
    _require_non_negative(funded_cfg["eval_cost"], "funded.eval_cost")

    scaling_cfg = funded_cfg["scaling"]
    _require_keys(scaling_cfg, ["tiers"], "funded.scaling")
    tiers = scaling_cfg["tiers"]
    if not tiers:
        raise ValueError("funded.scaling.tiers must be non-empty")
    if tiers[0]["min_profit"] > 0:
        raise ValueError("funded.scaling.tiers[0] must cover non-positive funded profit")
    prev_max = None
    for idx, tier in enumerate(tiers):
        _require_keys(tier, ["min_profit", "max_profit", "max_contracts"], f"funded.scaling.tiers[{idx}]")
        if tier["min_profit"] > tier["max_profit"]:
            raise ValueError(f"funded.scaling.tiers[{idx}] has min_profit > max_profit")
        _require_positive(tier["max_contracts"], f"funded.scaling.tiers[{idx}].max_contracts")
        if prev_max is not None and tier["min_profit"] < prev_max:
            raise ValueError("funded.scaling.tiers must be sorted and non-overlapping")
        if prev_max is not None and tier["min_profit"] > prev_max:
            raise ValueError("funded.scaling.tiers must be contiguous with no uncovered profit gaps")
        prev_max = tier["max_profit"]
    if tiers[-1]["max_profit"] < 1e9:
        raise ValueError("funded.scaling.tiers[-1] must provide open-ended upper profit coverage")

    _require_keys(instrument_cfg, ["name", "tick_size", "tick_value", "commission_per_side"], "instrument")
    _require_positive(instrument_cfg["tick_size"], "instrument.tick_size")
    _require_positive(instrument_cfg["tick_value"], "instrument.tick_value")
    _require_non_negative(instrument_cfg["commission_per_side"], "instrument.commission_per_side")


def _validate_phase_block(phase_cfg: dict, ctx: str) -> None:
    _require_keys(phase_cfg, ["stop_ticks", "target_ticks", "contracts", "daily_stop", "daily_target"], ctx)
    _require_positive(phase_cfg["stop_ticks"], f"{ctx}.stop_ticks")
    _require_positive(phase_cfg["target_ticks"], f"{ctx}.target_ticks")
    _require_positive(phase_cfg["contracts"], f"{ctx}.contracts")
    if phase_cfg["daily_stop"] >= 0:
        raise ValueError(f"{ctx}.daily_stop must be negative")
    _require_positive(phase_cfg["daily_target"], f"{ctx}.daily_target")


def _validate_params_config(cfg: dict) -> None:
    _require_keys(cfg, ["general", "strategy", "slippage", "monte_carlo"], "root")
    _require_keys(cfg["general"], ["random_seed"], "general")
    _require_non_negative(cfg["general"]["random_seed"], "general.random_seed")

    strategy_cfg = cfg["strategy"]
    _require_keys(strategy_cfg, ["orb"], "strategy")
    orb_cfg = strategy_cfg["orb"]
    _require_keys(orb_cfg, ["shared", "eval", "funded"], "strategy.orb")

    shared_cfg = orb_cfg["shared"]
    _require_keys(shared_cfg, ["range_minutes", "max_trades_day", "buffer_ticks", "volume_threshold"], "strategy.orb.shared")
    _require_positive(shared_cfg["range_minutes"], "strategy.orb.shared.range_minutes")
    _require_positive(shared_cfg["max_trades_day"], "strategy.orb.shared.max_trades_day")
    _require_non_negative(shared_cfg["buffer_ticks"], "strategy.orb.shared.buffer_ticks")
    _require_non_negative(shared_cfg["volume_threshold"], "strategy.orb.shared.volume_threshold")

    _validate_phase_block(orb_cfg["eval"], "strategy.orb.eval")
    _validate_phase_block(orb_cfg["funded"], "strategy.orb.funded")

    slippage_cfg = cfg["slippage"]
    _require_keys(slippage_cfg, ["stop_penalty", "atr_period", "trailing_atr_days"], "slippage")
    _require_positive(slippage_cfg["stop_penalty"], "slippage.stop_penalty")
    _require_positive(slippage_cfg["atr_period"], "slippage.atr_period")
    _require_positive(slippage_cfg["trailing_atr_days"], "slippage.trailing_atr_days")

    mc_cfg = cfg["monte_carlo"]
    _require_keys(mc_cfg, ["n_simulations", "block_mode", "block_size_min", "block_size_max"], "monte_carlo")
    _require_positive(mc_cfg["n_simulations"], "monte_carlo.n_simulations")
    if mc_cfg["block_mode"] not in {"daily", "fixed"}:
        raise ValueError("monte_carlo.block_mode must be 'daily' or 'fixed'")
    _require_positive(mc_cfg["block_size_min"], "monte_carlo.block_size_min")
    _require_positive(mc_cfg["block_size_max"], "monte_carlo.block_size_max")
    if mc_cfg["block_size_min"] > mc_cfg["block_size_max"]:
        raise ValueError("monte_carlo.block_size_min must be <= block_size_max")


def load_mff_config(path: Path) -> dict:
    """Load MFF rules configuration from TOML file."""
    cfg = _load_toml(path)
    _validate_mff_config(cfg)
    return cfg


def load_params_config(path: Path) -> dict:
    """Load strategy/simulator parameters from TOML file."""
    cfg = _load_toml(path)
    _validate_params_config(cfg)
    return cfg


def build_phase_params(shared_cfg: dict, phase_cfg: dict, slip_cfg: dict, commission_per_side: float):
    """Build the flat float64 params array for a single phase (eval or funded).

    Shared across all CLI scripts to prevent positional index drift.
    Uses PARAMS_* constants from propfirm.core.types.
    """
    import numpy as np
    from propfirm.core.types import (
        PARAMS_RANGE_MINUTES, PARAMS_STOP_TICKS, PARAMS_TARGET_TICKS,
        PARAMS_CONTRACTS, PARAMS_DAILY_STOP, PARAMS_DAILY_TARGET,
        PARAMS_MAX_TRADES, PARAMS_BUFFER_TICKS, PARAMS_VOL_THRESHOLD,
        PARAMS_STOP_PENALTY, PARAMS_COMMISSION, PARAMS_ARRAY_LENGTH,
    )
    params = np.zeros(PARAMS_ARRAY_LENGTH, dtype=np.float64)
    params[PARAMS_RANGE_MINUTES] = float(shared_cfg["range_minutes"])
    params[PARAMS_STOP_TICKS] = phase_cfg["stop_ticks"]
    params[PARAMS_TARGET_TICKS] = phase_cfg["target_ticks"]
    params[PARAMS_CONTRACTS] = float(phase_cfg["contracts"])
    params[PARAMS_DAILY_STOP] = phase_cfg["daily_stop"]
    params[PARAMS_DAILY_TARGET] = phase_cfg["daily_target"]
    params[PARAMS_MAX_TRADES] = float(shared_cfg["max_trades_day"])
    params[PARAMS_BUFFER_TICKS] = shared_cfg["buffer_ticks"]
    params[PARAMS_VOL_THRESHOLD] = shared_cfg["volume_threshold"]
    params[PARAMS_STOP_PENALTY] = slip_cfg["stop_penalty"]
    params[PARAMS_COMMISSION] = commission_per_side
    return params
