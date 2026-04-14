from dataclasses import dataclass, field
from datetime import date


@dataclass
class MFFState:
    """MFF Flex $50k state machine tracking eval and funded phases."""

    _config: dict = field(repr=False)

    phase: str = "eval"
    equity: float = 0.0
    eod_high: float = 0.0
    mll: float = 0.0
    mll_frozen: bool = False
    drawdown_high_watermark: float = 0.0
    static_floor_equity: float | None = None
    trading_days: int = 0
    winning_days: int = 0
    total_profit: float = 0.0
    max_single_day_profit: float = 0.0
    daily_profits: list = field(default_factory=list)
    payouts_completed: int = 0
    cycle_net_profit: float = 0.0
    funded_profit_generated: float = 0.0
    live_transition_ready: bool = False
    live_transition_reason: str | None = None
    phase_start_date: date | None = None
    last_trade_date: date | None = None

    def __init__(self, config: dict):
        self._config = config
        self.phase = "eval"
        self.equity = 0.0
        self.eod_high = 0.0
        self.mll = config["eval"]["max_loss_limit"]
        self.mll_frozen = False
        self.drawdown_high_watermark = 0.0
        self.static_floor_equity = None
        self.trading_days = 0
        self.winning_days = 0
        self.total_profit = 0.0
        self.max_single_day_profit = 0.0
        self.daily_profits = []
        self.payouts_completed = 0
        self.cycle_net_profit = 0.0
        self.funded_profit_generated = 0.0
        self.live_transition_ready = False
        self.live_transition_reason = None
        self.phase_start_date = None
        self.last_trade_date = None

    @staticmethod
    def _coerce_session_date(session_date) -> date | None:
        if session_date is None:
            return None
        if isinstance(session_date, date):
            return session_date
        if isinstance(session_date, str):
            return date.fromisoformat(session_date[:10])
        raise TypeError(f"Unsupported session_date type: {type(session_date)!r}")

    @property
    def funded_cfg(self) -> dict:
        return self._config["funded"]

    @property
    def payout_min_net_profit_between_payouts(self) -> float:
        return float(self.funded_cfg.get("payout_min_net_profit_between_payouts", 500.0))

    @property
    def inactivity_rule_calendar_days(self) -> int:
        return int(self.funded_cfg.get("inactivity_rule_calendar_days", 7))

    @property
    def live_transition_payouts_required(self) -> int:
        return int(self.funded_cfg.get("live_transition_payouts_required", 5))

    @property
    def live_sim_cap_profit(self) -> float:
        return float(self.funded_cfg.get("live_sim_cap_profit", 100000.0))

    @property
    def payout_eligible(self) -> bool:
        funded_cfg = self.funded_cfg
        min_gross = funded_cfg["payout_min_gross"]
        max_pct = funded_cfg["payout_max_pct"]
        potential_gross = min(self.cycle_net_profit * max_pct, funded_cfg["payout_cap"])
        return (
            self.phase == "funded"
            and
            self.winning_days >= funded_cfg["payout_winning_days_required"]
            and self.cycle_net_profit >= self.payout_min_net_profit_between_payouts
            and potential_gross >= min_gross
        )

    def check_inactivity_before_session(self, session_date) -> str:
        if self.phase != "funded":
            return "continue"
        current_date = self._coerce_session_date(session_date)
        if current_date is None:
            return "continue"
        if self.phase_start_date is None:
            self.phase_start_date = current_date
        reference_date = self.last_trade_date if self.last_trade_date is not None else self.phase_start_date
        if reference_date is None:
            return "continue"
        if (current_date - reference_date).days >= self.inactivity_rule_calendar_days:
            return "inactive"
        return "continue"

    def _refresh_live_transition_status(self) -> None:
        if self.phase != "funded" or self.live_transition_ready:
            return
        if self.payouts_completed >= self.live_transition_payouts_required:
            self.live_transition_ready = True
            self.live_transition_reason = "payouts"
            return
        if self.funded_profit_generated >= self.live_sim_cap_profit:
            self.live_transition_ready = True
            self.live_transition_reason = "sim_cap"

    def consistency_ok(self) -> bool:
        if self.total_profit <= 0:
            return True
        max_pct = self._config["eval"]["consistency_max_pct"]
        return self.max_single_day_profit / self.total_profit <= max_pct

    def update_eod(self, day_pnl: float, eod_equity: float, had_trade: bool = True, session_date=None) -> str:
        """Process end-of-day update. Returns 'continue', 'passed', or 'blown'."""
        current_date = self._coerce_session_date(session_date)
        if had_trade:
            self.trading_days += 1
        self.equity = eod_equity
        self.total_profit += day_pnl
        self.daily_profits.append(day_pnl)

        if self.phase == "funded":
            if self.phase_start_date is None and current_date is not None:
                self.phase_start_date = current_date
            self.cycle_net_profit += day_pnl
            self.funded_profit_generated += day_pnl
            if had_trade and current_date is not None:
                self.last_trade_date = current_date

        threshold = self.funded_cfg["winning_day_threshold"]
        if day_pnl >= threshold:
            self.winning_days += 1

        if day_pnl > self.max_single_day_profit:
            self.max_single_day_profit = day_pnl

        if eod_equity > self.eod_high:
            self.eod_high = eod_equity

        if not self.mll_frozen:
            if eod_equity > self.drawdown_high_watermark:
                self.drawdown_high_watermark = eod_equity
            if eod_equity <= self.drawdown_high_watermark - self.mll:
                return "blown"
        else:
            if self.static_floor_equity is None:
                raise RuntimeError("static_floor_equity must be set when mll_frozen is True")
            if eod_equity <= self.static_floor_equity:
                return "blown"

        if self.phase == "eval":
            eval_cfg = self._config["eval"]
            if (
                self.total_profit >= eval_cfg["profit_target"]
                and self.trading_days >= eval_cfg["min_trading_days"]
                and self.consistency_ok()
            ):
                return "passed"

        self._refresh_live_transition_status()
        return "continue"

    def get_active_params(self) -> str:
        return "params_eval" if self.phase == "eval" else "params_funded"

    def get_max_contracts(self) -> int:
        if self.phase == "eval":
            return self._config["eval"]["max_contracts"]
        tiers = self._config["funded"]["scaling"]["tiers"]
        for idx, tier in enumerate(tiers):
            upper_ok = (
                self.total_profit < tier["max_profit"]
                or (idx == len(tiers) - 1 and self.total_profit <= tier["max_profit"])
            )
            if tier["min_profit"] <= self.total_profit and upper_ok:
                return tier["max_contracts"]
        raise RuntimeError(f"No funded scaling tier covers total_profit={self.total_profit}")

    def get_liquidation_floor_equity(self) -> float:
        if self.mll_frozen:
            if self.static_floor_equity is None:
                raise RuntimeError("static_floor_equity must be set when mll_frozen is True")
            return float(self.static_floor_equity)
        return float(self.drawdown_high_watermark - self.mll)

    def transition_to_funded(self):
        self.phase = "funded"
        self.mll = self.funded_cfg["max_loss_limit"]
        self.equity = 0.0
        self.eod_high = 0.0
        self.total_profit = 0.0
        self.max_single_day_profit = 0.0
        self.daily_profits = []
        self.trading_days = 0
        self.winning_days = 0
        self.mll_frozen = False
        self.payouts_completed = 0
        self.drawdown_high_watermark = 0.0
        self.static_floor_equity = None
        self.cycle_net_profit = 0.0
        self.funded_profit_generated = 0.0
        self.live_transition_ready = False
        self.live_transition_reason = None
        self.phase_start_date = None
        self.last_trade_date = None

    def process_payout(self) -> float:
        funded_cfg = self.funded_cfg
        gross = min(
            self.cycle_net_profit * funded_cfg["payout_max_pct"],
            funded_cfg["payout_cap"],
        )
        if gross < funded_cfg["payout_min_gross"]:
            return 0.0
        net = gross * funded_cfg["profit_split_trader"]
        self.equity -= gross
        self.total_profit -= gross
        self.payouts_completed += 1
        if self.payouts_completed >= 1 and not self.mll_frozen:
            self.mll_frozen = True
            self.mll = funded_cfg["mll_frozen_value"]
            self.static_floor_equity = funded_cfg["mll_frozen_value"]
        self.cycle_net_profit = 0.0
        self.winning_days = 0
        self._refresh_live_transition_status()
        return net
