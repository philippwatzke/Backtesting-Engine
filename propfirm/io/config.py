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


def _require_int_like(value, ctx: str) -> None:
    if isinstance(value, bool) or int(value) != value:
        raise ValueError(f"{ctx} must be an integer")


def _require_positive_int(value, ctx: str) -> None:
    _require_positive(value, ctx)
    _require_int_like(value, ctx)


def _require_non_negative_int(value, ctx: str) -> None:
    _require_non_negative(value, ctx)
    _require_int_like(value, ctx)


def _require_pct(value, ctx: str) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{ctx} must be between 0 and 1")


def _require_hhmm(value: str, ctx: str) -> None:
    try:
        pd = __import__("pandas")
        pd.Timestamp(f"2000-01-01 {value}")
    except Exception as exc:
        raise ValueError(f"{ctx} must be in HH:MM format") from exc


def _validate_portfolio_config(portfolio_cfg: dict) -> None:
    if "shared" not in portfolio_cfg:
        raise ValueError("portfolio missing required keys: ['shared']")
    shared_cfg = portfolio_cfg["shared"]
    _require_keys(shared_cfg, ["risk_buffer_fraction"], "portfolio.shared")
    if not (0.0 < shared_cfg["risk_buffer_fraction"] <= 1.0):
        raise ValueError("portfolio.shared.risk_buffer_fraction must be > 0 and <= 1")


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
    _require_positive_int(eval_cfg["min_trading_days"], "eval.min_trading_days")
    _require_positive_int(eval_cfg["max_contracts"], "eval.max_contracts")

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
    _require_positive_int(funded_cfg["payout_winning_days_required"], "funded.payout_winning_days_required")
    _require_pct(funded_cfg["payout_max_pct"], "funded.payout_max_pct")
    _require_positive(funded_cfg["payout_cap"], "funded.payout_cap")
    _require_non_negative(funded_cfg["payout_min_gross"], "funded.payout_min_gross")
    _require_pct(funded_cfg["profit_split_trader"], "funded.profit_split_trader")
    _require_non_negative(funded_cfg["eval_cost"], "funded.eval_cost")
    if "payout_min_net_profit_between_payouts" in funded_cfg:
        _require_non_negative(
            funded_cfg["payout_min_net_profit_between_payouts"],
            "funded.payout_min_net_profit_between_payouts",
        )
    if "inactivity_rule_calendar_days" in funded_cfg:
        _require_positive_int(
            funded_cfg["inactivity_rule_calendar_days"],
            "funded.inactivity_rule_calendar_days",
        )
    if "live_transition_payouts_required" in funded_cfg:
        _require_positive_int(
            funded_cfg["live_transition_payouts_required"],
            "funded.live_transition_payouts_required",
        )
    if "live_sim_cap_profit" in funded_cfg:
        _require_positive(
            funded_cfg["live_sim_cap_profit"],
            "funded.live_sim_cap_profit",
        )

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
        _require_positive_int(tier["max_contracts"], f"funded.scaling.tiers[{idx}].max_contracts")
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
    _require_positive_int(phase_cfg["contracts"], f"{ctx}.contracts")
    if phase_cfg["daily_stop"] >= 0:
        raise ValueError(f"{ctx}.daily_stop must be negative")
    _require_positive(phase_cfg["daily_target"], f"{ctx}.daily_target")


def _validate_dynamic_phase_block(phase_cfg: dict, ctx: str) -> None:
    _require_keys(
        phase_cfg,
        ["risk_per_trade_usd", "stop_atr_multiplier", "target_atr_multiplier", "daily_stop", "daily_target"],
        ctx,
    )
    _require_positive(phase_cfg["risk_per_trade_usd"], f"{ctx}.risk_per_trade_usd")
    _require_positive(phase_cfg["stop_atr_multiplier"], f"{ctx}.stop_atr_multiplier")
    _require_positive(phase_cfg["target_atr_multiplier"], f"{ctx}.target_atr_multiplier")
    if phase_cfg["daily_stop"] >= 0:
        raise ValueError(f"{ctx}.daily_stop must be negative")
    _require_positive(phase_cfg["daily_target"], f"{ctx}.daily_target")


def _validate_orb_strategy(orb_cfg: dict) -> None:
    _require_keys(orb_cfg, ["shared", "eval", "funded"], "strategy.orb")
    shared_cfg = orb_cfg["shared"]
    _require_keys(shared_cfg, ["range_minutes", "max_trades_day", "buffer_ticks", "volume_threshold"], "strategy.orb.shared")
    _require_positive_int(shared_cfg["range_minutes"], "strategy.orb.shared.range_minutes")
    _require_positive_int(shared_cfg["max_trades_day"], "strategy.orb.shared.max_trades_day")
    _require_non_negative(shared_cfg["buffer_ticks"], "strategy.orb.shared.buffer_ticks")
    _require_non_negative(shared_cfg["volume_threshold"], "strategy.orb.shared.volume_threshold")
    _validate_phase_block(orb_cfg["eval"], "strategy.orb.eval")
    _validate_phase_block(orb_cfg["funded"], "strategy.orb.funded")


def _validate_mcl_orb_strategy(orb_cfg: dict) -> None:
    _require_keys(orb_cfg, ["shared", "eval", "funded"], "strategy.mcl_orb")
    shared_cfg = orb_cfg["shared"]
    _require_keys(
        shared_cfg,
        ["range_minutes", "trigger_start_minute", "trigger_end_minute", "time_stop_minute", "max_trades_day"],
        "strategy.mcl_orb.shared",
    )
    _require_positive_int(shared_cfg["range_minutes"], "strategy.mcl_orb.shared.range_minutes")
    _require_non_negative_int(shared_cfg["trigger_start_minute"], "strategy.mcl_orb.shared.trigger_start_minute")
    _require_non_negative_int(shared_cfg["trigger_end_minute"], "strategy.mcl_orb.shared.trigger_end_minute")
    _require_non_negative_int(shared_cfg["time_stop_minute"], "strategy.mcl_orb.shared.time_stop_minute")
    _require_positive_int(shared_cfg["max_trades_day"], "strategy.mcl_orb.shared.max_trades_day")
    if shared_cfg["trigger_start_minute"] < shared_cfg["range_minutes"]:
        raise ValueError("strategy.mcl_orb.shared.trigger_start_minute must be >= range_minutes")
    if shared_cfg["trigger_end_minute"] < shared_cfg["trigger_start_minute"]:
        raise ValueError("strategy.mcl_orb.shared.trigger_end_minute must be >= trigger_start_minute")
    if shared_cfg["time_stop_minute"] < shared_cfg["trigger_end_minute"]:
        raise ValueError("strategy.mcl_orb.shared.time_stop_minute must be >= trigger_end_minute")
    _validate_dynamic_phase_block(orb_cfg["eval"], "strategy.mcl_orb.eval")
    _validate_dynamic_phase_block(orb_cfg["funded"], "strategy.mcl_orb.funded")


def _validate_vwap_pullback_strategy(vwap_cfg: dict) -> None:
    _require_keys(vwap_cfg, ["shared", "eval", "funded"], "strategy.vwap_pullback")
    shared_cfg = vwap_cfg["shared"]
    _require_keys(
        shared_cfg,
        ["max_trades_day", "distance_ticks", "sma_period", "breakeven_trigger_ticks"],
        "strategy.vwap_pullback.shared",
    )
    _require_positive_int(shared_cfg["max_trades_day"], "strategy.vwap_pullback.shared.max_trades_day")
    _require_non_negative(shared_cfg["distance_ticks"], "strategy.vwap_pullback.shared.distance_ticks")
    _require_positive_int(shared_cfg["sma_period"], "strategy.vwap_pullback.shared.sma_period")
    _require_non_negative(
        shared_cfg["breakeven_trigger_ticks"],
        "strategy.vwap_pullback.shared.breakeven_trigger_ticks",
    )
    _validate_dynamic_phase_block(vwap_cfg["eval"], "strategy.vwap_pullback.eval")
    _validate_dynamic_phase_block(vwap_cfg["funded"], "strategy.vwap_pullback.funded")


def _validate_m6a_fade_strategy(fade_cfg: dict) -> None:
    _require_keys(fade_cfg, ["shared", "eval", "funded"], "strategy.m6a_fade")
    shared_cfg = fade_cfg["shared"]
    _require_keys(
        shared_cfg,
        ["max_trades_day", "bb_period", "band_multiplier", "session_start", "session_end"],
        "strategy.m6a_fade.shared",
    )
    _require_positive_int(shared_cfg["max_trades_day"], "strategy.m6a_fade.shared.max_trades_day")
    _require_positive_int(shared_cfg["bb_period"], "strategy.m6a_fade.shared.bb_period")
    _require_positive(shared_cfg["band_multiplier"], "strategy.m6a_fade.shared.band_multiplier")
    _require_hhmm(shared_cfg["session_start"], "strategy.m6a_fade.shared.session_start")
    _require_hhmm(shared_cfg["session_end"], "strategy.m6a_fade.shared.session_end")
    if shared_cfg["session_start"] == shared_cfg["session_end"]:
        raise ValueError("strategy.m6a_fade.shared.session_start must differ from session_end")
    _validate_dynamic_phase_block(fade_cfg["eval"], "strategy.m6a_fade.eval")
    _validate_dynamic_phase_block(fade_cfg["funded"], "strategy.m6a_fade.funded")


def _validate_mgc_macro_orb_strategy(orb_cfg: dict) -> None:
    _require_keys(orb_cfg, ["shared", "eval", "funded"], "strategy.mgc_macro_orb")
    shared_cfg = orb_cfg["shared"]
    _require_keys(
        shared_cfg,
        [
            "session_start",
            "session_end",
            "range_minutes",
            "trigger_start_minute",
            "trigger_end_minute",
            "time_stop_minute",
            "max_trades_day",
            "breakeven_trigger_ticks",
            "min_rvol",
            "max_rvol",
        ],
        "strategy.mgc_macro_orb.shared",
    )
    _require_hhmm(shared_cfg["session_start"], "strategy.mgc_macro_orb.shared.session_start")
    _require_hhmm(shared_cfg["session_end"], "strategy.mgc_macro_orb.shared.session_end")
    _require_positive_int(shared_cfg["range_minutes"], "strategy.mgc_macro_orb.shared.range_minutes")
    _require_non_negative_int(shared_cfg["trigger_start_minute"], "strategy.mgc_macro_orb.shared.trigger_start_minute")
    _require_non_negative_int(shared_cfg["trigger_end_minute"], "strategy.mgc_macro_orb.shared.trigger_end_minute")
    _require_non_negative_int(shared_cfg["time_stop_minute"], "strategy.mgc_macro_orb.shared.time_stop_minute")
    _require_positive_int(shared_cfg["max_trades_day"], "strategy.mgc_macro_orb.shared.max_trades_day")
    _require_non_negative(
        shared_cfg["breakeven_trigger_ticks"],
        "strategy.mgc_macro_orb.shared.breakeven_trigger_ticks",
    )
    _require_positive(shared_cfg["min_rvol"], "strategy.mgc_macro_orb.shared.min_rvol")
    _require_positive(shared_cfg["max_rvol"], "strategy.mgc_macro_orb.shared.max_rvol")
    if shared_cfg["session_start"] == shared_cfg["session_end"]:
        raise ValueError("strategy.mgc_macro_orb.shared.session_start must differ from session_end")
    if shared_cfg["max_rvol"] <= shared_cfg["min_rvol"]:
        raise ValueError("strategy.mgc_macro_orb.shared.max_rvol must be > min_rvol")
    if shared_cfg["trigger_start_minute"] < shared_cfg["range_minutes"]:
        raise ValueError("strategy.mgc_macro_orb.shared.trigger_start_minute must be >= range_minutes")
    if shared_cfg["trigger_end_minute"] < shared_cfg["trigger_start_minute"]:
        raise ValueError("strategy.mgc_macro_orb.shared.trigger_end_minute must be >= trigger_start_minute")
    if shared_cfg["time_stop_minute"] < shared_cfg["trigger_end_minute"]:
        raise ValueError("strategy.mgc_macro_orb.shared.time_stop_minute must be >= trigger_end_minute")
    _validate_dynamic_phase_block(orb_cfg["eval"], "strategy.mgc_macro_orb.eval")
    _validate_dynamic_phase_block(orb_cfg["funded"], "strategy.mgc_macro_orb.funded")


def _validate_mgc_h1_trend_strategy(trend_cfg: dict) -> None:
    _require_keys(trend_cfg, ["shared", "eval", "funded"], "strategy.mgc_h1_trend")
    shared_cfg = trend_cfg["shared"]
    _require_keys(
        shared_cfg,
        [
            "session_start",
            "session_end",
            "max_trades_day",
            "donchian_lookback",
            "trigger_start_minute",
            "trigger_end_minute",
            "time_stop_minute",
            "breakeven_trigger_ticks",
        ],
        "strategy.mgc_h1_trend.shared",
    )
    _require_hhmm(shared_cfg["session_start"], "strategy.mgc_h1_trend.shared.session_start")
    _require_hhmm(shared_cfg["session_end"], "strategy.mgc_h1_trend.shared.session_end")
    _require_positive_int(shared_cfg["max_trades_day"], "strategy.mgc_h1_trend.shared.max_trades_day")
    _require_positive_int(shared_cfg["donchian_lookback"], "strategy.mgc_h1_trend.shared.donchian_lookback")
    _require_non_negative_int(shared_cfg["trigger_start_minute"], "strategy.mgc_h1_trend.shared.trigger_start_minute")
    _require_non_negative_int(shared_cfg["trigger_end_minute"], "strategy.mgc_h1_trend.shared.trigger_end_minute")
    _require_non_negative_int(shared_cfg["time_stop_minute"], "strategy.mgc_h1_trend.shared.time_stop_minute")
    _require_non_negative(
        shared_cfg["breakeven_trigger_ticks"],
        "strategy.mgc_h1_trend.shared.breakeven_trigger_ticks",
    )
    if "blocked_weekday" in shared_cfg:
        _require_non_negative_int(shared_cfg["blocked_weekday"], "strategy.mgc_h1_trend.shared.blocked_weekday")
        if shared_cfg["blocked_weekday"] > 6:
            raise ValueError("strategy.mgc_h1_trend.shared.blocked_weekday must be between 0 and 6")
    if shared_cfg["session_start"] == shared_cfg["session_end"]:
        raise ValueError("strategy.mgc_h1_trend.shared.session_start must differ from session_end")
    if shared_cfg["trigger_end_minute"] < shared_cfg["trigger_start_minute"]:
        raise ValueError("strategy.mgc_h1_trend.shared.trigger_end_minute must be >= trigger_start_minute")
    if shared_cfg["time_stop_minute"] < shared_cfg["trigger_end_minute"]:
        raise ValueError("strategy.mgc_h1_trend.shared.time_stop_minute must be >= trigger_end_minute")
    _validate_dynamic_phase_block(trend_cfg["eval"], "strategy.mgc_h1_trend.eval")
    _validate_dynamic_phase_block(trend_cfg["funded"], "strategy.mgc_h1_trend.funded")


def _validate_vwap_poc_breakout_strategy(vwap_cfg: dict) -> None:
    _require_keys(vwap_cfg, ["shared", "eval", "funded"], "strategy.vwap_poc_breakout")
    shared_cfg = vwap_cfg["shared"]
    _require_keys(
        shared_cfg,
        ["max_trades_day", "sma_period", "band_multiplier", "poc_lookback"],
        "strategy.vwap_poc_breakout.shared",
    )
    _require_positive_int(shared_cfg["max_trades_day"], "strategy.vwap_poc_breakout.shared.max_trades_day")
    _require_positive_int(shared_cfg["sma_period"], "strategy.vwap_poc_breakout.shared.sma_period")
    _require_non_negative(shared_cfg["band_multiplier"], "strategy.vwap_poc_breakout.shared.band_multiplier")
    _require_positive_int(shared_cfg["poc_lookback"], "strategy.vwap_poc_breakout.shared.poc_lookback")
    _validate_dynamic_phase_block(vwap_cfg["eval"], "strategy.vwap_poc_breakout.eval")
    _validate_dynamic_phase_block(vwap_cfg["funded"], "strategy.vwap_poc_breakout.funded")


def _validate_moc_flow_strategy(moc_cfg: dict) -> None:
    _require_keys(moc_cfg, ["shared", "eval", "funded"], "strategy.moc_flow")
    shared_cfg = moc_cfg["shared"]
    _require_keys(
        shared_cfg,
        ["max_trades_day", "trend_threshold_pct", "breakeven_trigger_ticks"],
        "strategy.moc_flow.shared",
    )
    _require_positive_int(shared_cfg["max_trades_day"], "strategy.moc_flow.shared.max_trades_day")
    _require_non_negative(shared_cfg["trend_threshold_pct"], "strategy.moc_flow.shared.trend_threshold_pct")
    _require_non_negative(shared_cfg["breakeven_trigger_ticks"], "strategy.moc_flow.shared.breakeven_trigger_ticks")
    _validate_dynamic_phase_block(moc_cfg["eval"], "strategy.moc_flow.eval")
    _validate_dynamic_phase_block(moc_cfg["funded"], "strategy.moc_flow.funded")


def _validate_params_config(cfg: dict) -> None:
    _require_keys(cfg, ["general", "strategy", "slippage", "monte_carlo"], "root")
    _require_keys(cfg["general"], ["random_seed"], "general")
    _require_non_negative_int(cfg["general"]["random_seed"], "general.random_seed")
    if "portfolio" in cfg:
        _validate_portfolio_config(cfg["portfolio"])

    strategy_cfg = cfg["strategy"]
    if (
        "orb" not in strategy_cfg
        and "mcl_orb" not in strategy_cfg
        and "m6a_fade" not in strategy_cfg
        and "mgc_macro_orb" not in strategy_cfg
        and "mgc_h1_trend" not in strategy_cfg
        and "vwap_pullback" not in strategy_cfg
        and "vwap_poc_breakout" not in strategy_cfg
        and "moc_flow" not in strategy_cfg
    ):
        raise ValueError("strategy must define at least one supported strategy block")
    if "orb" in strategy_cfg:
        _validate_orb_strategy(strategy_cfg["orb"])
    if "mcl_orb" in strategy_cfg:
        _validate_mcl_orb_strategy(strategy_cfg["mcl_orb"])
    if "m6a_fade" in strategy_cfg:
        _validate_m6a_fade_strategy(strategy_cfg["m6a_fade"])
    if "mgc_macro_orb" in strategy_cfg:
        _validate_mgc_macro_orb_strategy(strategy_cfg["mgc_macro_orb"])
    if "mgc_h1_trend" in strategy_cfg:
        _validate_mgc_h1_trend_strategy(strategy_cfg["mgc_h1_trend"])
    if "vwap_pullback" in strategy_cfg:
        _validate_vwap_pullback_strategy(strategy_cfg["vwap_pullback"])
    if "vwap_poc_breakout" in strategy_cfg:
        _validate_vwap_poc_breakout_strategy(strategy_cfg["vwap_poc_breakout"])
    if "moc_flow" in strategy_cfg:
        _validate_moc_flow_strategy(strategy_cfg["moc_flow"])
    slippage_cfg = cfg["slippage"]
    _require_keys(slippage_cfg, ["stop_penalty", "atr_period", "trailing_atr_days"], "slippage")
    _require_positive(slippage_cfg["stop_penalty"], "slippage.stop_penalty")
    _require_positive_int(slippage_cfg["atr_period"], "slippage.atr_period")
    _require_positive_int(slippage_cfg["trailing_atr_days"], "slippage.trailing_atr_days")

    mc_cfg = cfg["monte_carlo"]
    _require_keys(mc_cfg, ["n_simulations", "block_mode", "block_size_min", "block_size_max"], "monte_carlo")
    _require_positive_int(mc_cfg["n_simulations"], "monte_carlo.n_simulations")
    if mc_cfg["block_mode"] not in {"daily", "fixed"}:
        raise ValueError("monte_carlo.block_mode must be 'daily' or 'fixed'")
    _require_positive_int(mc_cfg["block_size_min"], "monte_carlo.block_size_min")
    _require_positive_int(mc_cfg["block_size_max"], "monte_carlo.block_size_max")
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


def build_phase_params(
    shared_cfg: dict,
    phase_cfg: dict,
    slip_cfg: dict,
    commission_per_side: float,
    strategy_name: str = "orb",
    instrument_cfg: dict | None = None,
):
    """Build the flat float64 params array for a single phase (eval or funded).

    Shared across all CLI scripts to prevent positional index drift.
    Uses PARAMS_* constants from propfirm.core.types.
    """
    import numpy as np
    from propfirm.core.types import (
        PARAMS_RANGE_MINUTES, PARAMS_STOP_TICKS, PARAMS_TARGET_TICKS,
        PARAMS_CONTRACTS, PARAMS_DAILY_STOP, PARAMS_DAILY_TARGET,
        PARAMS_MAX_TRADES, PARAMS_BUFFER_TICKS, PARAMS_VOL_THRESHOLD,
        PARAMS_STOP_PENALTY, PARAMS_COMMISSION, PARAMS_DISTANCE_TICKS,
        PARAMS_SMA_PERIOD, PARAMS_BREAKEVEN_TRIGGER_TICKS, PARAMS_BAND_MULTIPLIER,
        PARAMS_POC_LOOKBACK, PARAMS_EXTRA_SLIPPAGE_TICKS, PARAMS_ENTRY_MINUTE,
        PARAMS_TREND_THRESHOLD_PCT, PARAMS_TRIGGER_START_MINUTE,
        PARAMS_TRIGGER_END_MINUTE, PARAMS_TIME_STOP_MINUTE,
        PARAMS_TICK_SIZE, PARAMS_TICK_VALUE, PARAMS_MIN_RVOL, PARAMS_MAX_RVOL,
        PARAMS_ENTRY_ON_CLOSE,
        PARAMS_BLOCKED_WEEKDAY,
        PARAMS_TRAIL_BAR_EXTREME, PARAMS_ARRAY_LENGTH,
        HARD_CLOSE_MINUTE, MNQ_TICK_SIZE, MNQ_TICK_VALUE,
    )
    params = np.zeros(PARAMS_ARRAY_LENGTH, dtype=np.float64)
    tick_size = float((instrument_cfg or {}).get("tick_size", MNQ_TICK_SIZE))
    tick_value = float((instrument_cfg or {}).get("tick_value", MNQ_TICK_VALUE))
    params[PARAMS_DAILY_STOP] = phase_cfg["daily_stop"]
    params[PARAMS_DAILY_TARGET] = phase_cfg["daily_target"]
    params[PARAMS_STOP_PENALTY] = slip_cfg["stop_penalty"]
    params[PARAMS_COMMISSION] = commission_per_side
    params[PARAMS_EXTRA_SLIPPAGE_TICKS] = 0.0
    params[PARAMS_ENTRY_MINUTE] = -1.0
    params[PARAMS_TIME_STOP_MINUTE] = float(shared_cfg.get("time_stop_minute", HARD_CLOSE_MINUTE))
    params[PARAMS_TICK_SIZE] = tick_size
    params[PARAMS_TICK_VALUE] = tick_value
    params[PARAMS_BLOCKED_WEEKDAY] = float(shared_cfg.get("blocked_weekday", -1))
    params[PARAMS_ENTRY_ON_CLOSE] = 0.0
    if strategy_name == "orb":
        params[PARAMS_STOP_TICKS] = phase_cfg["stop_ticks"]
        params[PARAMS_TARGET_TICKS] = phase_cfg["target_ticks"]
        params[PARAMS_CONTRACTS] = float(phase_cfg["contracts"])
        params[PARAMS_RANGE_MINUTES] = float(shared_cfg["range_minutes"])
        params[PARAMS_MAX_TRADES] = float(shared_cfg["max_trades_day"])
        params[PARAMS_BUFFER_TICKS] = shared_cfg["buffer_ticks"]
        params[PARAMS_VOL_THRESHOLD] = shared_cfg["volume_threshold"]
    elif strategy_name == "mcl_orb":
        params[PARAMS_RANGE_MINUTES] = float(shared_cfg["range_minutes"])
        params[PARAMS_MAX_TRADES] = float(shared_cfg["max_trades_day"])
        params[PARAMS_TRIGGER_START_MINUTE] = float(shared_cfg["trigger_start_minute"])
        params[PARAMS_TRIGGER_END_MINUTE] = float(shared_cfg["trigger_end_minute"])
        params[PARAMS_TIME_STOP_MINUTE] = float(shared_cfg["time_stop_minute"])
    elif strategy_name == "vwap_pullback":
        params[PARAMS_MAX_TRADES] = float(shared_cfg["max_trades_day"])
        params[PARAMS_DISTANCE_TICKS] = float(shared_cfg["distance_ticks"])
        params[PARAMS_SMA_PERIOD] = float(shared_cfg["sma_period"])
        params[PARAMS_BREAKEVEN_TRIGGER_TICKS] = float(shared_cfg["breakeven_trigger_ticks"])
    elif strategy_name == "m6a_fade":
        params[PARAMS_MAX_TRADES] = float(shared_cfg["max_trades_day"])
        params[PARAMS_SMA_PERIOD] = float(shared_cfg["bb_period"])
        params[PARAMS_BAND_MULTIPLIER] = float(shared_cfg["band_multiplier"])
    elif strategy_name == "mgc_macro_orb":
        params[PARAMS_RANGE_MINUTES] = float(shared_cfg["range_minutes"])
        params[PARAMS_MAX_TRADES] = float(shared_cfg["max_trades_day"])
        params[PARAMS_TRIGGER_START_MINUTE] = float(shared_cfg["trigger_start_minute"])
        params[PARAMS_TRIGGER_END_MINUTE] = float(shared_cfg["trigger_end_minute"])
        params[PARAMS_TIME_STOP_MINUTE] = float(shared_cfg["time_stop_minute"])
        params[PARAMS_MIN_RVOL] = float(shared_cfg["min_rvol"])
        params[PARAMS_MAX_RVOL] = float(shared_cfg["max_rvol"])
    elif strategy_name == "mgc_h1_trend":
        params[PARAMS_MAX_TRADES] = float(shared_cfg["max_trades_day"])
        params[PARAMS_POC_LOOKBACK] = float(shared_cfg["donchian_lookback"])
        params[PARAMS_TRIGGER_START_MINUTE] = float(shared_cfg["trigger_start_minute"])
        params[PARAMS_TRIGGER_END_MINUTE] = float(shared_cfg["trigger_end_minute"])
        params[PARAMS_TIME_STOP_MINUTE] = float(shared_cfg["time_stop_minute"])
        params[PARAMS_BREAKEVEN_TRIGGER_TICKS] = float(shared_cfg["breakeven_trigger_ticks"])
        params[PARAMS_TRAIL_BAR_EXTREME] = 0.0
        params[PARAMS_ENTRY_ON_CLOSE] = 1.0
    elif strategy_name == "vwap_poc_breakout":
        params[PARAMS_MAX_TRADES] = float(shared_cfg["max_trades_day"])
        params[PARAMS_SMA_PERIOD] = float(shared_cfg["sma_period"])
        params[PARAMS_BAND_MULTIPLIER] = float(shared_cfg["band_multiplier"])
        params[PARAMS_POC_LOOKBACK] = float(shared_cfg["poc_lookback"])
    elif strategy_name == "moc_flow":
        from propfirm.core.types import MOC_ENTRY_MINUTE

        params[PARAMS_MAX_TRADES] = float(shared_cfg["max_trades_day"])
        params[PARAMS_ENTRY_MINUTE] = float(MOC_ENTRY_MINUTE)
        params[PARAMS_TREND_THRESHOLD_PCT] = float(shared_cfg["trend_threshold_pct"])
    else:
        raise ValueError(f"Unsupported strategy_name: {strategy_name}")
    return params
