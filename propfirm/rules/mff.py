from dataclasses import dataclass, field


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

    @property
    def payout_eligible(self) -> bool:
        funded_cfg = self._config["funded"]
        min_gross = funded_cfg["payout_min_gross"]
        max_pct = funded_cfg["payout_max_pct"]
        potential_gross = min(self.total_profit * max_pct, funded_cfg["payout_cap"])
        return (
            self.winning_days >= funded_cfg["payout_winning_days_required"]
            and potential_gross >= min_gross
        )

    def consistency_ok(self) -> bool:
        if self.total_profit <= 0:
            return True
        max_pct = self._config["eval"]["consistency_max_pct"]
        return self.max_single_day_profit / self.total_profit <= max_pct

    def update_eod(self, day_pnl: float, eod_equity: float) -> str:
        """Process end-of-day update. Returns 'continue', 'passed', or 'blown'."""
        self.trading_days += 1
        self.equity = eod_equity
        self.total_profit += day_pnl
        self.daily_profits.append(day_pnl)

        threshold = self._config["funded"]["winning_day_threshold"]
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

    def transition_to_funded(self):
        self.phase = "funded"
        self.mll = self._config["funded"]["max_loss_limit"]
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

    def process_payout(self) -> float:
        funded_cfg = self._config["funded"]
        gross = min(
            self.total_profit * funded_cfg["payout_max_pct"],
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
            self.static_floor_equity = self.equity - self.mll
        self.winning_days = 0
        return net
