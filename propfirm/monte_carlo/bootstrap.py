import numpy as np
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from propfirm.rules.mff import MFFState


@dataclass
class MCResult:
    eval_pass_rate: float
    eval_pass_rate_ci_5: float
    eval_pass_rate_ci_95: float
    funded_survival_rate: float
    payout_rate: float
    mean_payout_net: float
    mean_days_to_eval_pass: float
    mean_funded_days_to_payout: float
    mean_drawdown: float
    nve: float
    n_simulations: int


def split_daily_log_for_mc(daily_log: np.ndarray) -> dict:
    if daily_log.dtype.names is None:
        raise ValueError("Structured daily log required for daily Monte-Carlo")
    required = {"day_id", "phase_id", "payout_cycle_id", "day_pnl"}
    if not required.issubset(set(daily_log.dtype.names)):
        raise ValueError("Structured daily log missing required lifecycle fields")
    eval_log = daily_log[daily_log["phase_id"] == 0]
    funded_log = daily_log[
        (daily_log["phase_id"] == 1) & (daily_log["payout_cycle_id"] == 0)
    ]
    if len(eval_log) == 0 or len(funded_log) == 0:
        raise ValueError("Lifecycle Monte-Carlo requires eval days and funded payout_cycle_id==0 days")
    return {
        "eval_day_pnls": eval_log["day_pnl"].astype(np.float64),
        "funded_day_pnls": funded_log["day_pnl"].astype(np.float64),
    }


def block_bootstrap_single(values, target_length, seed, block_min=5, block_max=10):
    rng = np.random.RandomState(seed)
    result = []
    n = len(values)
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    if n == 0:
        raise ValueError("Cannot bootstrap an empty input sequence")
    while len(result) < target_length:
        block_size = rng.randint(block_min, block_max + 1)
        start = rng.randint(0, max(1, n - block_size + 1))
        end = min(start + block_size, n)
        result.extend(values[start:end].tolist())
    return np.array(result[:target_length], dtype=np.float64)


def _run_days(state, sequence, trades_per_day):
    n = len(sequence)
    max_drawdown = 0.0
    days = 0
    i = 0
    while i < n:
        day_end = min(i + trades_per_day, n)
        day_pnl = float(np.sum(sequence[i:day_end]))
        eod_equity = state.equity + day_pnl
        result = state.update_eod(day_pnl, eod_equity)
        days += 1
        dd = state.eod_high - state.equity
        if dd > max_drawdown:
            max_drawdown = dd
        if result in ("passed", "blown"):
            return result, days, max_drawdown
        i = day_end
    return "exhausted", days, max_drawdown


def _simulate_single_path(eval_sequence, funded_sequence, mff_config,
                           eval_trades_per_day, funded_trades_per_day):
    state = MFFState(mff_config)
    eval_result, eval_days, eval_dd = _run_days(state, eval_sequence, eval_trades_per_day)
    eval_cfg = mff_config["eval"]
    if eval_result == "exhausted":
        if (state.total_profit >= eval_cfg["profit_target"]
                and state.trading_days >= eval_cfg["min_trading_days"]
                and state.consistency_ok()):
            eval_result = "passed"
    if eval_result != "passed":
        return {"eval_passed": False, "funded_survived": False,
                "payout_net": 0.0, "eval_days": eval_days,
                "funded_days": 0, "drawdown": eval_dd}

    state.transition_to_funded()

    funded_days = 0
    max_funded_dd = 0.0
    lifecycle_dd = eval_dd
    i = 0
    while i < len(funded_sequence):
        day_end = min(i + funded_trades_per_day, len(funded_sequence))
        day_pnl = float(np.sum(funded_sequence[i:day_end]))
        eod_equity = state.equity + day_pnl
        result = state.update_eod(day_pnl, eod_equity)
        funded_days += 1
        dd = state.eod_high - state.equity
        if dd > max_funded_dd:
            max_funded_dd = dd
        if max_funded_dd > lifecycle_dd:
            lifecycle_dd = max_funded_dd

        if result == "blown":
            return {"eval_passed": True, "funded_survived": False,
                    "payout_net": 0.0, "eval_days": eval_days,
                    "funded_days": funded_days, "drawdown": lifecycle_dd}

        if state.payout_eligible:
            net = state.process_payout()
            if net > 0:
                return {"eval_passed": True, "funded_survived": True,
                        "payout_net": net, "eval_days": eval_days,
                        "funded_days": funded_days, "drawdown": lifecycle_dd}
        i = day_end

    return {"eval_passed": True, "funded_survived": True,
            "payout_net": 0.0, "eval_days": eval_days,
            "funded_days": funded_days, "drawdown": lifecycle_dd}


def _run_chunk(args):
    (eval_pnls, funded_pnls, mff_config, seeds,
     block_mode, block_min, block_max,
     eval_target_length, funded_target_length,
     trades_per_day_fixed) = args
    results = []
    for seed in seeds:
        eval_seq = block_bootstrap_single(eval_pnls, eval_target_length, seed,
                                          block_min=block_min, block_max=block_max)
        funded_seq = block_bootstrap_single(funded_pnls, funded_target_length, seed + 1_000_000,
                                            block_min=block_min, block_max=block_max)
        if block_mode == "daily":
            eval_trades_per_day = 1
            funded_trades_per_day = 1
        else:
            eval_trades_per_day = trades_per_day_fixed
            funded_trades_per_day = trades_per_day_fixed
        result = _simulate_single_path(eval_seq, funded_seq, mff_config,
                                        eval_trades_per_day=eval_trades_per_day,
                                        funded_trades_per_day=funded_trades_per_day)
        results.append(result)
    return results


def run_monte_carlo(eval_pnls, mff_config, funded_pnls=None, n_sims=10_000,
                    seed=42, n_workers=1, eval_target_length=200,
                    funded_target_length=300, block_mode="daily",
                    block_min=5, block_max=10, trades_per_day_fixed=2):
    if block_mode not in {"daily", "fixed"}:
        raise ValueError(f"Unsupported block_mode: {block_mode}")
    if len(eval_pnls) == 0:
        raise ValueError("eval_pnls must be non-empty")
    if block_mode == "daily" and funded_pnls is None:
        raise ValueError("daily block mode requires explicit eval/funded day-level inputs")
    if funded_pnls is None:
        funded_pnls = eval_pnls
    if len(funded_pnls) == 0:
        raise ValueError("funded_pnls must be non-empty")

    all_seeds = [seed + i for i in range(n_sims)]

    if n_workers <= 1:
        all_results = _run_chunk(
            (eval_pnls, funded_pnls, mff_config, all_seeds,
             block_mode, block_min, block_max,
             eval_target_length, funded_target_length,
             trades_per_day_fixed)
        )
    else:
        chunks = np.array_split(all_seeds, n_workers)
        args = [(eval_pnls, funded_pnls, mff_config, chunk.tolist(),
                 block_mode, block_min, block_max,
                 eval_target_length, funded_target_length,
                 trades_per_day_fixed)
                for chunk in chunks]
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            chunk_results = list(pool.map(_run_chunk, args))
        all_results = []
        for chunk in chunk_results:
            all_results.extend(chunk)

    n_total = len(all_results)
    eval_passed = [r for r in all_results if r["eval_passed"]]
    funded_survived = [r for r in eval_passed if r["funded_survived"]]
    got_payout = [r for r in all_results if r["payout_net"] > 0]

    eval_pass_rate = len(eval_passed) / n_total
    funded_survival_rate = (len(funded_survived) / len(eval_passed) if eval_passed else 0.0)
    payout_rate = len(got_payout) / n_total

    pass_flags = np.array([1 if r["eval_passed"] else 0 for r in all_results])
    rng_ci = np.random.RandomState(seed)
    n_flags = len(pass_flags)
    bootstrap_rates = np.array([
        np.mean(rng_ci.choice(pass_flags, size=n_flags, replace=True))
        for _ in range(1000)
    ])
    ci_5 = float(np.percentile(bootstrap_rates, 5))
    ci_95 = float(np.percentile(bootstrap_rates, 95))

    mean_eval_days = (float(np.mean([r["eval_days"] for r in eval_passed]))
                      if eval_passed else 0.0)
    mean_funded_days = (float(np.mean([r["funded_days"] for r in got_payout]))
                        if got_payout else 0.0)
    mean_dd = float(np.mean([r["drawdown"] for r in all_results]))

    mean_payout_net = (float(np.mean([r["payout_net"] for r in got_payout]))
                       if got_payout else 0.0)
    funded_cfg = mff_config["funded"]
    from propfirm.optim.objective import compute_capped_nve
    nve = compute_capped_nve(payout_rate, mean_payout_net, funded_cfg["eval_cost"])

    return MCResult(
        eval_pass_rate=eval_pass_rate,
        eval_pass_rate_ci_5=ci_5,
        eval_pass_rate_ci_95=ci_95,
        funded_survival_rate=funded_survival_rate,
        payout_rate=payout_rate,
        mean_payout_net=mean_payout_net,
        mean_days_to_eval_pass=mean_eval_days,
        mean_funded_days_to_payout=mean_funded_days,
        mean_drawdown=mean_dd,
        nve=nve,
        n_simulations=n_sims,
    )
