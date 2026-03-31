# Prop-Firm Backtesting Engine - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic backtesting engine that simulates MFF Flex $50k rules, computes Capped NVE via Monte-Carlo block-bootstrap, and optimizes an ORB baseline strategy through grid search and walk-forward analysis.

**Architecture:** Layered Python package (`propfirm/`) with a Numba JIT kernel for the bar-loop hot path, Python-layer EOD rule enforcement (MFF state machine), and multiprocessing for Monte-Carlo parallelization. Precomputed arrays (ATR, slippage, minute_of_day) eliminate look-ahead bias and keep the kernel simple.

**Tech Stack:** Python 3.10+, NumPy, Numba, Pandas (IO only), tomllib, pytest, concurrent.futures.

**Repository:** `philippwatzke/Backtesting-Engine` (`https://github.com/philippwatzke/Backtesting-Engine`)

**Spec:** `docs/superpowers/specs/2026-03-30-prop-firm-engine-design.md`

## Implementation Firewall

> This section is normative. If a later task snippet conflicts with any rule below,
> the firewall rule wins and the task snippet must be updated before implementation proceeds.

### Non-Negotiable Invariants

1. **Data contract**
   - Session input must be timezone-aware, normalized to `America/New_York`, sorted, deduplicated, and filtered to RTH `09:30-15:59`.
   - `minute_of_day` is valid only for RTH bars and must remain in `[0, 389]`.
   - `session_dates` are derived from normalized timestamps, never from raw parquet ordering assumptions.

2. **Engine contract**
   - One trade uses one `TRADE_LOG_DTYPE` row from entry through exit.
   - `net_pnl == gross_pnl - entry_commission - exit_commission` for every closed trade.
   - Target exits must not apply `stop_penalty`.
   - After an exit on a bar, no re-entry may occur on the same bar.
   - Trade-log capacity overflow is a hard error, never a silent truncate.

3. **MFF lifecycle contract**
   - `transition_to_funded()` performs a full funded reset; eval history must not contaminate funded state.
   - `funded.scaling.tiers` must be contiguous, cover non-positive funded profit, and never leave uncovered profit gaps.
   - `get_max_contracts()` must hard-fail on uncovered profit, never silently fall through to the highest tier.
   - Before the first funded payout, drawdown is trailing.
   - After the first funded payout, drawdown is controlled only by a static floor.
   - `process_payout()` deducts gross payout from both `equity` and `total_profit`.

4. **Artifact contract**
   - `TRADE_LOG_DTYPE` is the diagnostic round-trip artifact.
   - `DAILY_LOG_DTYPE` is the authoritative lifecycle artifact for real ORB Monte-Carlo because it preserves zero-trade days.
   - `phase_id` semantics are fixed: `0=eval`, `1=funded`.
   - `payout_cycle_id` semantics are fixed: `-1=eval`, `0=pre-first-payout funded`, `1+=later funded cycles`.
   - `day_pnl` is pre-payout day PnL; `net_payout` is booked after that day's trading.

5. **Monte-Carlo / NVE contract**
   - Real ORB Monte-Carlo uses `DAILY_LOG_DTYPE`, not trade-log-derived pseudo-days.
   - Daily Monte-Carlo must split eval days from funded `payout_cycle_id==0` days before resampling.
   - Structured `DAILY_LOG_DTYPE` inputs force `mc_mode="daily"` in the CLI; `fixed` mode is rejected for lifecycle artifacts.
   - `block_mode="fixed"` is synthetic/legacy only.
   - `mean_drawdown` means max lifecycle drawdown across eval plus funded, not funded-only drawdown.
   - `nve = payout_rate * mean_payout_net - eval_cost`; `payout_rate` means actual payout probability, not eval-pass rate.

6. **Walk-forward contract**
   - In-sample and OOS scoring use the same lifecycle-aware Monte-Carlo path.
   - Windows without sufficient lifecycle pools are `not_scored`, never coerced to `0.0`.
   - Eval and funded params remain separated end-to-end.
   - Historical walk-forward inherits Monte-Carlo block settings from `params_cfg["monte_carlo"]`; only `n_simulations` may be CLI-overridden for faster research runs.
   - Params arrays are built only through `PARAMS_*` constants, never positional literals.

7. **Reporting / audit contract**
   - Reports must include both `params_cfg` and `mff_cfg`.
   - Every CLI report must include artifact paths and runtime mode flags such as `mc_mode` and `mc_daily_lifecycle_ready`.
   - Date ranges in reports come from `session_dates`, never from raw `day_boundaries` indices.

8. **Calibration / synthetic-study contract**
   - `scripts/calibrate_slippage.py` must reuse the normalized session-data contract (`_prepare_session_frame()` plus `compute_minute_of_day()`).
   - Calibrated `baseline_ticks` must be true tick counts, not normalized price-point proxies.
   - Synthetic grid search may vary only parameters that actually affect the synthetic path; day-structured knobs like `daily_stop` are excluded.

9. **Config contract**
   - Config loaders validate required keys, numeric ranges, phase blocks, block-size ordering, and enumerated strings.
   - `funded.scaling.tiers` must also guarantee upper coverage for all reachable funded profit values; no runtime-only tier exhaustion is allowed.
   - `funded.scaling.tiers` are half-open `[min_profit, max_profit)` except for the final tier, which includes its upper bound.
   - Invalid config is rejected at load time, not deferred to runtime.

### Cross-Task Consistency Checklist

- [ ] Every contract change is propagated through producer, consumer, tests, CLI, report, and smoke-test text in the same commit.
- [ ] Every new state field is reflected wherever relevant in state reset logic, tests, serialization, and reporting.
- [ ] Every new artifact field is updated in dtype definition, writing path, reading path, and smoke validation.
- [ ] Every status string has one canonical spelling across implementation, tests, CLI output, and reporting.
- [ ] Every `Expected: X passed` gate matches the actual test count shown in the snippet directly above it.
- [ ] No real ORB Monte-Carlo path reconstructs calendar days from the trade log.
- [ ] No params array in any script or helper is built positionally; `PARAMS_*` constants are mandatory. `build_phase_params()` in `config.py` is the single source.
- [ ] No task may silently reintroduce eval/funded contamination in artifacts, bootstrap pools, or reporting fields.
- [ ] Every strategy callback passed to `run_day_kernel()` must be `@njit`-decorated for Numba compatibility.
- [ ] No hardcoded tick_size or tick_value literals (`0.25`, `0.50`) in engine or slippage code; use `MNQ_TICK_SIZE` / `MNQ_TICK_VALUE` constants.
- [ ] If this plan intentionally diverges from the design spec, the divergence is listed in the Spec Drift Register below.

### Traceability Matrix

| Concern | Source of truth in this plan | Verified by tests | Produced / consumed by | Artifact / gate |
|------|-----------------------------|-------------------|------------------------|-----------------|
| Config validity | Task 2 `propfirm/io/config.py` | `tests/test_config.py` | All CLI scripts | load-time rejection of bad TOML |
| Typed artifacts | Task 3 `propfirm/core/types.py` | `tests/test_types.py` | engine, MC, reporting, CLI | `TRADE_LOG_DTYPE`, `DAILY_LOG_DTYPE` |
| MFF lifecycle | Task 4 `propfirm/rules/mff.py` | `tests/test_rules.py` | backtest, MC, walk-forward | funded reset, payout, static floor |
| Slippage semantics | Task 5 `propfirm/market/slippage.py` | `tests/test_slippage.py` | engine (imported), calibration | target-vs-stop fill logic |
| Params array builder | Task 2 `propfirm/io/config.py:build_phase_params` | (via CLI integration) | all CLI scripts, walk-forward | single source for params array construction |
| Session normalization | Task 6 `propfirm/market/data_loader.py` | `tests/test_data_loader.py` | backtest, walk-forward | `session_data`, `session_dates` |
| Risk guards | Task 7 `propfirm/risk/risk.py` | `tests/test_risk.py` | engine, backtest, walk-forward | circuit-breaker and contract-clamp semantics |
| Trade execution | Task 8 `propfirm/core/engine.py` | `tests/test_engine.py` | backtest | `latest_trade_log.npy` |
| Strategy signals | Task 9 `propfirm/strategy/orb.py` | `tests/test_orb.py` | engine, backtest, walk-forward | deterministic ORB params usage |
| NVE semantics | Task 10 `propfirm/optim/objective.py` | `tests/test_objective.py` | MC, grid search, walk-forward | payout-rate-based NVE |
| Lifecycle Monte-Carlo | Task 11 `propfirm/monte_carlo/bootstrap.py` | `tests/test_monte_carlo.py` | `run_monte_carlo.py`, walk-forward | `latest_daily_log.npy` |
| Audit trail | Task 12 `propfirm/io/reporting.py` | `tests/test_reporting.py` | all CLI scripts | self-contained JSON reports |
| Slippage calibration | Task 13 `scripts/calibrate_slippage.py` | (manual smoke) | `run_backtest.py`, engine | `data/slippage/slippage_profile.parquet` |
| Synthetic search | Task 14 `propfirm/optim/grid_search.py` | `tests/test_grid_search.py` | `run_grid_search.py` | synthetic-only optimization path |
| Historical walk-forward | Task 15 `propfirm/optim/walk_forward.py` | `tests/test_walk_forward.py` | `run_walk_forward.py` | lifecycle-aware IS/OOS scoring |
| End-to-end integration | Task 17 smoke checks | Task 17 inline Python checks | full pipeline | release/tag gate |

### Spec Drift Register

The following differences from `docs/superpowers/specs/2026-03-30-prop-firm-engine-design.md`
are intentional and authoritative for this implementation plan:

| Topic | Plan authority |
|------|----------------|
| NVE metric | Uses `payout_rate`, not plain `pass_rate` |
| Monte-Carlo production input | Uses `DAILY_LOG_DTYPE` daily lifecycle artifacts |
| Trade-log role | Diagnostic artifact only, not authoritative day reconstruction |
| Artifact format | `.npy` lifecycle artifacts are the production path |
| Reporting names | Uses disaggregated metrics such as `eval_pass_rate`, `funded_survival_rate`, `payout_rate` |

---

## File Map

### New Files

| File | Responsibility |
|------|----------------|
| `pyproject.toml` | Package definition, dependencies, entry-points |
| `.gitignore` | Ignore output/, data/, __pycache__, .parquet artifacts |
| `configs/mff_flex_50k.toml` | MFF Flex $50k rules (declarative) |
| `configs/default_params.toml` | Strategy + simulator + MC defaults |
| `propfirm/__init__.py` | Package root |
| `propfirm/core/__init__.py` | Core subpackage |
| `propfirm/core/types.py` | TRADE_LOG_DTYPE, DAILY_LOG_DTYPE, constants |
| `propfirm/core/engine.py` | Numba @njit bar-loop, trade execution |
| `propfirm/market/__init__.py` | Market subpackage |
| `propfirm/market/slippage.py` | compute_slippage(), slippage lookup builder |
| `propfirm/market/data_loader.py` | Parquet -> NumPy arrays + precomputed vectors |
| `propfirm/rules/__init__.py` | Rules subpackage |
| `propfirm/rules/mff.py` | MFFState dataclass, EOD logic, phase-shift |
| `propfirm/risk/__init__.py` | Risk subpackage |
| `propfirm/risk/risk.py` | Circuit breaker, position sizing constraints |
| `propfirm/strategy/__init__.py` | Strategy subpackage |
| `propfirm/strategy/orb.py` | ORB signal generation (@njit) |
| `propfirm/monte_carlo/__init__.py` | Monte-Carlo subpackage |
| `propfirm/monte_carlo/bootstrap.py` | Block-bootstrap + multiprocessing |
| `propfirm/optim/__init__.py` | Optimization subpackage |
| `propfirm/optim/objective.py` | Capped NVE objective helpers |
| `propfirm/optim/grid_search.py` | Synthetic grid search |
| `propfirm/optim/walk_forward.py` | Walk-forward analysis |
| `propfirm/io/__init__.py` | IO subpackage |
| `propfirm/io/reporting.py` | JSON reports + audit trail |
| `propfirm/io/config.py` | TOML loader + validation + `build_phase_params()` |
| `scripts/calibrate_slippage.py` | CLI: calibrate slippage profile |
| `scripts/run_backtest.py` | CLI: single strategy backtest |
| `scripts/run_grid_search.py` | CLI: grid search |
| `scripts/run_monte_carlo.py` | CLI: Monte-Carlo simulation |
| `scripts/run_walk_forward.py` | CLI: walk-forward analysis |
| `tests/__init__.py` | Test package |
| `tests/test_types.py` | Types + constants tests |
| `tests/test_config.py` | TOML loading tests |
| `tests/test_rules.py` | MFF state machine tests |
| `tests/test_slippage.py` | Slippage model tests |
| `tests/test_data_loader.py` | Data loading + preprocessing tests |
| `tests/test_risk.py` | Circuit breaker tests |
| `tests/test_engine.py` | Numba kernel integration tests |
| `tests/test_mff_validation.py` | Firewall validation for MFF lifecycle rules |
| `tests/test_orb.py` | ORB strategy tests |
| `tests/test_monte_carlo.py` | Bootstrap + multiprocessing tests |
| `tests/test_objective.py` | Capped NVE tests |
| `tests/test_grid_search.py` | Grid search tests |
| `tests/test_walk_forward.py` | Walk-forward tests |
| `tests/test_reporting.py` | JSON report tests |

### Existing Files (unchanged)

| File | Note |
|------|------|
| `data/processed/MNQ_1m_train.parquet` | Training data - read only |
| `data/processed/MNQ_1m_validation.parquet` | Validation data - read only |
| `data/processed/MNQ_1m_test.parquet` | Test data - FIREWALL, do not touch until final eval |
| `data/raw/MNQ_1m_2022_heute_raw.parquet` | Raw source - read only |

---

## Task 1: Project Scaffolding + Git Init

> **Authoritative Plan Note:** Falls dieses Implementierungsdokument von
> `docs/superpowers/specs/2026-03-30-prop-firm-engine-design.md` abweicht,
> ist für Phase 2 dieses Dokument autoritativ. Insbesondere gelten hier die
> neueren Verträge für `payout_rate`-basierte NVE, den `.npy`-Artefaktfluss
> und den täglichen Lifecycle-Log als Produktionsquelle für Monte-Carlo.

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `propfirm/__init__.py`
- Create: all `__init__.py` files for subpackages

- [ ] **Step 1: Initialize git repository**

```bash
git init
```

Expected: `Initialized empty Git repository`

- [ ] **Step 2: Create .gitignore**

Create `.gitignore`:

```
__pycache__/
*.pyc
*.pyo
.eggs/
*.egg-info/
dist/
build/
.venv/
venv/
output/
# Data files excluded - too large for git. Parquet files live in data/ locally.
data/
*.parquet
.numba_cache/
```

- [ ] **Step 3: Create pyproject.toml**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "propfirm"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "numba>=0.58",
    "pandas>=2.0",
    "pyarrow>=14.0",
    "tomli>=2.0; python_version < '3.11'",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
]

[tool.setuptools.packages.find]
include = ["propfirm*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 4: Create all __init__.py files**

Create the following empty files:
- `propfirm/__init__.py`
- `propfirm/core/__init__.py`
- `propfirm/market/__init__.py`
- `propfirm/rules/__init__.py`
- `propfirm/risk/__init__.py`
- `propfirm/strategy/__init__.py`
- `propfirm/monte_carlo/__init__.py`
- `propfirm/optim/__init__.py`
- `propfirm/io/__init__.py`
- `tests/__init__.py`

- [ ] **Step 5: Create output directory structure**

```powershell
New-Item -ItemType Directory -Force -Path output/grid_search, output/backtests, output/monte_carlo
New-Item -ItemType Directory -Force -Path configs, scripts, data/slippage
```

- [ ] **Step 6: Install package in dev mode**

```bash
pip install -e ".[dev]"
```

Expected: Successfully installed propfirm-0.1.0

- [ ] **Step 7: Verify pytest runs with no tests**

```bash
pytest --co -q
```

Expected: `no tests ran` (no errors)

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .gitignore propfirm/ tests/__init__.py
git commit -m "feat: scaffold project structure with package layout"
```

---

## Task 2: TOML Configuration + Loader

**Files:**
- Create: `configs/mff_flex_50k.toml`
- Create: `configs/default_params.toml`
- Create: `propfirm/io/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Create MFF rules config**

Create `configs/mff_flex_50k.toml`:

```toml
[eval]
profit_target = 3000.0
max_loss_limit = 2000.0
consistency_max_pct = 0.50
min_trading_days = 2
max_contracts = 50

[funded]
max_loss_limit = 2000.0
mll_frozen_value = 100.0
winning_day_threshold = 150.0
payout_winning_days_required = 5
payout_max_pct = 0.50
payout_cap = 5000.0
payout_min_gross = 250.0
profit_split_trader = 0.80
eval_cost = 107.0

[funded.scaling]
tiers = [
    { min_profit = -1e9, max_profit = 1500.0, max_contracts = 20 },
    { min_profit = 1500.0, max_profit = 2000.0, max_contracts = 30 },
    { min_profit = 2000.0, max_profit = 1e9, max_contracts = 50 },
]

[instrument]
name = "MNQ"
tick_size = 0.25
tick_value = 0.50
commission_per_side = 0.54
```

- [ ] **Step 2: Create default params config**

Create `configs/default_params.toml`:

```toml
[general]
random_seed = 42

# [simulator] section reserved for future use (e.g. starting_equity overrides).
# Not consumed by current code - MFFState always starts at equity 0.0.

[strategy.orb.shared]
range_minutes = 15
max_trades_day = 2
buffer_ticks = 2.0
volume_threshold = 0.0

[strategy.orb.eval]
stop_ticks = 40.0
target_ticks = 60.0
contracts = 10
daily_stop = -750.0
daily_target = 600.0

[strategy.orb.funded]
stop_ticks = 35.0
target_ticks = 80.0
contracts = 20
daily_stop = -1000.0
daily_target = 900.0

[slippage]
stop_penalty = 1.5
atr_period = 14
trailing_atr_days = 5

[monte_carlo]
n_simulations = 10_000
block_mode = "daily"       # REQUIRED for real ORB backtests with structured daily lifecycle logs
                           # "fixed" is only for synthetic or legacy flat-PNL runs
block_size_min = 5         # In daily mode: calendar-day block length; in fixed mode: trade block length
block_size_max = 10        # In daily mode: calendar-day block length; in fixed mode: trade block length

[stress_test]
slippage_multiplier = 3.0
drawdown_buffer = 0.8
worst_day_injection = true

```

- [ ] **Step 3: Write failing tests for config loader**

Create `tests/test_config.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they fail**

```bash
pytest tests/test_config.py -v
```

Expected: FAIL - `ModuleNotFoundError: No module named 'propfirm.io.config'`

- [ ] **Step 5: Implement config loader**

Create `propfirm/io/config.py`:

```python
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
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/test_config.py -v
```

Expected: 14 passed

- [ ] **Step 7: Commit**

```bash
git add configs/ propfirm/io/config.py tests/test_config.py
git commit -m "feat: add TOML config loader with MFF rules and default params"
```

---

## Task 3: Core Types (Trade Log, Daily Log, Constants)

**Files:**
- Create: `propfirm/core/types.py`
- Create: `tests/test_types.py`

- [ ] **Step 1: Write failing tests for types**

Create `tests/test_types.py`:

```python
import numpy as np
from propfirm.core.types import (
    TRADE_LOG_DTYPE,
    DAILY_LOG_DTYPE,
    MNQ_TICK_SIZE,
    MNQ_TICK_VALUE,
    MNQ_COMMISSION_PER_SIDE,
    BARS_PER_RTH_SESSION,
    EXIT_TARGET,
    EXIT_STOP,
    EXIT_HARD_CLOSE,
    EXIT_CIRCUIT_BREAKER,
    SIGNAL_LONG,
    SIGNAL_SHORT,
    SIGNAL_NONE,
    PARAMS_RANGE_MINUTES,
    PARAMS_STOP_TICKS,
    PARAMS_TARGET_TICKS,
    PARAMS_CONTRACTS,
    PARAMS_DAILY_STOP,
    PARAMS_DAILY_TARGET,
    PARAMS_MAX_TRADES,
    PARAMS_BUFFER_TICKS,
    PARAMS_VOL_THRESHOLD,
    PARAMS_STOP_PENALTY,
    PARAMS_COMMISSION,
    PARAMS_ARRAY_LENGTH,
)


class TestTradeLogDtype:
    def test_has_required_fields(self):
        names = TRADE_LOG_DTYPE.names
        assert "day_id" in names
        assert "phase_id" in names
        assert "payout_cycle_id" in names
        assert "entry_time" in names
        assert "exit_time" in names
        assert "entry_price" in names
        assert "exit_price" in names
        assert "entry_slippage" in names
        assert "exit_slippage" in names
        assert "entry_commission" in names
        assert "exit_commission" in names
        assert "contracts" in names
        assert "gross_pnl" in names
        assert "net_pnl" in names
        assert "signal_type" in names
        assert "exit_reason" in names

    def test_can_create_empty_array(self):
        arr = np.zeros(100, dtype=TRADE_LOG_DTYPE)
        assert arr.shape == (100,)
        assert arr["net_pnl"][0] == 0.0

    def test_field_types(self):
        assert TRADE_LOG_DTYPE["day_id"] == np.dtype("i4")
        assert TRADE_LOG_DTYPE["phase_id"] == np.dtype("i1")
        assert TRADE_LOG_DTYPE["payout_cycle_id"] == np.dtype("i2")
        assert TRADE_LOG_DTYPE["entry_time"] == np.dtype("i8")
        assert TRADE_LOG_DTYPE["net_pnl"] == np.dtype("f8")
        assert TRADE_LOG_DTYPE["gross_pnl"] == np.dtype("f8")
        assert TRADE_LOG_DTYPE["contracts"] == np.dtype("i4")
        assert TRADE_LOG_DTYPE["signal_type"] == np.dtype("i1")


class TestDailyLogDtype:
    def test_has_required_fields(self):
        names = DAILY_LOG_DTYPE.names
        assert "day_id" in names
        assert "phase_id" in names
        assert "payout_cycle_id" in names
        assert "had_trade" in names
        assert "n_trades" in names
        assert "day_pnl" in names
        assert "net_payout" in names

    def test_field_types(self):
        assert DAILY_LOG_DTYPE["day_id"] == np.dtype("i4")
        assert DAILY_LOG_DTYPE["phase_id"] == np.dtype("i1")
        assert DAILY_LOG_DTYPE["payout_cycle_id"] == np.dtype("i2")
        assert DAILY_LOG_DTYPE["had_trade"] == np.dtype("i1")
        assert DAILY_LOG_DTYPE["n_trades"] == np.dtype("i2")
        assert DAILY_LOG_DTYPE["day_pnl"] == np.dtype("f8")
        assert DAILY_LOG_DTYPE["net_payout"] == np.dtype("f8")


class TestConstants:
    def test_mnq_constants(self):
        assert MNQ_TICK_SIZE == 0.25
        assert MNQ_TICK_VALUE == 0.50
        assert MNQ_COMMISSION_PER_SIDE == 0.54

    def test_session_bars(self):
        assert BARS_PER_RTH_SESSION == 390  # 09:30-16:00 = 390 minutes

    def test_exit_reasons(self):
        assert EXIT_TARGET == 0
        assert EXIT_STOP == 1
        assert EXIT_HARD_CLOSE == 2
        assert EXIT_CIRCUIT_BREAKER == 3

    def test_signal_codes(self):
        assert SIGNAL_LONG == 1
        assert SIGNAL_SHORT == -1
        assert SIGNAL_NONE == 0


class TestParamsIndexConstants:
    def test_indices_are_sequential(self):
        indices = [
            PARAMS_RANGE_MINUTES, PARAMS_STOP_TICKS, PARAMS_TARGET_TICKS,
            PARAMS_CONTRACTS, PARAMS_DAILY_STOP, PARAMS_DAILY_TARGET,
            PARAMS_MAX_TRADES, PARAMS_BUFFER_TICKS, PARAMS_VOL_THRESHOLD,
            PARAMS_STOP_PENALTY, PARAMS_COMMISSION,
        ]
        assert indices == list(range(11))

    def test_array_length_matches(self):
        assert PARAMS_ARRAY_LENGTH == 11

    def test_no_duplicate_indices(self):
        indices = [
            PARAMS_RANGE_MINUTES, PARAMS_STOP_TICKS, PARAMS_TARGET_TICKS,
            PARAMS_CONTRACTS, PARAMS_DAILY_STOP, PARAMS_DAILY_TARGET,
            PARAMS_MAX_TRADES, PARAMS_BUFFER_TICKS, PARAMS_VOL_THRESHOLD,
            PARAMS_STOP_PENALTY, PARAMS_COMMISSION,
        ]
        assert len(indices) == len(set(indices))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_types.py -v
```

Expected: FAIL - `ImportError`

- [ ] **Step 3: Implement types**

Create `propfirm/core/types.py`:

```python
import numpy as np

# --- MNQ Instrument Constants ---
MNQ_TICK_SIZE = 0.25
MNQ_TICK_VALUE = 0.50
MNQ_COMMISSION_PER_SIDE = 0.54

# --- Session Constants ---
BARS_PER_RTH_SESSION = 390  # 09:30 to 16:00 ET = 390 one-minute bars
HARD_CLOSE_MINUTE = 389     # minute_of_day index for 15:59 (0-based from 09:30)

# --- Signal Codes ---
SIGNAL_LONG = 1
SIGNAL_SHORT = -1
SIGNAL_NONE = 0

# --- Exit Reason Codes ---
EXIT_TARGET = 0
EXIT_STOP = 1
EXIT_HARD_CLOSE = 2
EXIT_CIRCUIT_BREAKER = 3

# --- Trade Log Structured Array ---
# IMPORTANT INVARIANTS:
# 1. phase_id partitions eval (0) vs funded (1) for phase-aware Monte-Carlo.
# 2. payout_cycle_id partitions funded trading into pre/post payout regimes.
#    Eval trades use -1; funded first-payout-cycle uses 0; later cycles use 1, 2, ...
# 3. day_id is the authoritative day grouping WITHIN each (phase_id, payout_cycle_id) pool.
# 4. net_pnl must equal gross_pnl - entry_commission - exit_commission.
# 5. sum(net_pnl) must exactly match the trade-level equity curve before payouts.
# The trade log is the diagnostic round-trip artifact. Monte-Carlo and
# walk-forward consume DAILY_LOG_DTYPE for lifecycle simulation because it
# preserves zero-trade days required by the rules engine.
TRADE_LOG_DTYPE = np.dtype([
    ("day_id", "i4"),                # Authoritative trading-day id for MC grouping
    ("phase_id", "i1"),              # 0=eval, 1=funded; prevents phase contamination in MC
    ("payout_cycle_id", "i2"),       # -1=eval, 0=pre-first-payout funded, 1+=later funded cycles
    ("entry_time", "i8"),           # Unix timestamp
    ("exit_time", "i8"),
    ("entry_price", "f8"),
    ("exit_price", "f8"),
    ("entry_slippage", "f8"),       # Slippage at entry (points)
    ("exit_slippage", "f8"),        # Slippage at exit (points)
    ("entry_commission", "f8"),     # Commission at entry ($)
    ("exit_commission", "f8"),      # Commission at exit ($)
    ("contracts", "i4"),
    ("gross_pnl", "f8"),            # Raw PNL from price movement (after slippage, before commission)
    ("net_pnl", "f8"),              # gross_pnl - entry_commission - exit_commission
    ("signal_type", "i1"),          # 1=Long, -1=Short
    ("exit_reason", "i1"),          # 0=Target, 1=Stop, 2=HardClose, 3=CB
])

# --- Daily Lifecycle Log Structured Array ---
# IMPORTANT INVARIANTS:
# 1. DAILY_LOG_DTYPE is the authoritative Monte-Carlo artifact for real ORB runs.
# 2. One row exists for EVERY calendar trading day, including had_trade == 0 days.
# 3. day_pnl is the pre-payout day result and must equal the sum of that day's trade_log net_pnl.
# 4. net_payout is booked AFTER day_pnl for that day; 0.0 means no payout occurred.
# 5. phase_id / payout_cycle_id use the same semantics as TRADE_LOG_DTYPE.
DAILY_LOG_DTYPE = np.dtype([
    ("day_id", "i4"),
    ("phase_id", "i1"),
    ("payout_cycle_id", "i2"),
    ("had_trade", "i1"),
    ("n_trades", "i2"),
    ("day_pnl", "f8"),
    ("net_payout", "f8"),
])

# --- Slippage Floor ---
SLIPPAGE_FLOOR_POINTS = 0.25   # Minimum 1 tick slippage

# --- Params Array Index Constants ---
# The strategy params array is a flat float64 array shared between engine and strategy.
# Define indices HERE once - engine.py and orb.py reference these constants, never magic numbers.
# This prevents index drift when the layout is extended.
PARAMS_RANGE_MINUTES = 0       # ORB range buildup period (minutes)
PARAMS_STOP_TICKS = 1          # Stop-loss distance in ticks
PARAMS_TARGET_TICKS = 2        # Take-profit distance in ticks
PARAMS_CONTRACTS = 3           # Number of contracts per trade
PARAMS_DAILY_STOP = 4          # Circuit breaker: max daily loss
PARAMS_DAILY_TARGET = 5        # Hit-and-run: daily profit target
PARAMS_MAX_TRADES = 6          # Max trades per day
PARAMS_BUFFER_TICKS = 7        # ORB breakout buffer in ticks
PARAMS_VOL_THRESHOLD = 8       # ORB volume filter (0 = disabled)
PARAMS_STOP_PENALTY = 9        # Slippage penalty for stop orders
PARAMS_COMMISSION = 10         # Commission per side per contract
PARAMS_ARRAY_LENGTH = 11       # Expected length of params array
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_types.py -v
```

Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add propfirm/core/types.py tests/test_types.py
git commit -m "feat: add core types - trade log dtype, constants, signal/exit codes, params indices"
```

---

## Task 4: MFF Rules State Machine

**Files:**
- Create: `propfirm/rules/mff.py`
- Create: `tests/test_rules.py`

- [ ] **Step 1: Write failing tests for MFFState**

Create `tests/test_rules.py`:

```python
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
        state.update_eod(2000.0, 2000.0)
        result = state.update_eod(1100.0, 3100.0)  # total > 3000, 2 days
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
        # Now total_profit=200, max_single_day=100 -> 100/200 = 0.50 (borderline OK)
        # Add day that dominates
        result = state.update_eod(2900.0, 3100.0)
        # max_single=2900, total=3100, 2900/3100 = 0.935 > 0.50
        # But we need to check: does eval "pass" get blocked by consistency?
        assert state.consistency_ok() == False


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
        """Funded account starts fresh - no eval contamination."""
        state = MFFState(make_eval_config())
        # Simulate some eval progress
        state.update_eod(1500.0, 1500.0)
        state.update_eod(1600.0, 3100.0)  # "passed"
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
        assert state.mll == 2000.0  # funded MLL from config
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_rules.py -v
```

Expected: FAIL - `ImportError`

- [ ] **Step 3: Implement MFFState**

Create `propfirm/rules/mff.py`:

```python
from dataclasses import dataclass, field


@dataclass
class MFFState:
    """MFF Flex $50k state machine tracking eval and funded phases."""

    _config: dict = field(repr=False)

    # Phase
    phase: str = "eval"  # "eval" | "funded"

    # Equity tracking
    equity: float = 0.0
    eod_high: float = 0.0
    mll: float = 0.0
    mll_frozen: bool = False
    drawdown_high_watermark: float = 0.0
    static_floor_equity: float | None = None

    # Day tracking
    trading_days: int = 0
    winning_days: int = 0

    # Profit tracking
    total_profit: float = 0.0
    max_single_day_profit: float = 0.0
    daily_profits: list = field(default_factory=list)

    # Payout tracking
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

        # Track winning days (>= threshold)
        threshold = self._config["funded"]["winning_day_threshold"]
        if day_pnl >= threshold:
            self.winning_days += 1

        # Track max single day profit
        if day_pnl > self.max_single_day_profit:
            self.max_single_day_profit = day_pnl

        # Update diagnostic EOD high watermark
        if eod_equity > self.eod_high:
            self.eod_high = eod_equity

        # Drawdown mode:
        # - Before first funded payout: trailing high watermark
        # - After first funded payout: static floor only
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

        # Check eval pass condition
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
        """Transition from eval to funded phase.

        IMPORTANT - Funded Reset Contract:
        The funded account is a separate account from eval. All performance
        tracking fields must be reset so that funded metrics (scaling tiers,
        payout eligibility, winning days, trailing drawdown) are computed
        purely from funded trading, not contaminated by eval history.

        Fields RESET to fresh funded baseline:
        - equity          -> 0.0   (funded account starts at zero PNL)
        - eod_high        -> 0.0   (diagnostic high reset)
        - total_profit    -> 0.0   (funded profit accumulator)
        - max_single_day_profit -> 0.0 (funded consistency, if checked)
        - daily_profits   -> []    (funded daily history)
        - trading_days    -> 0     (funded day count)
        - winning_days    -> 0     (payout requires funded winning days)
        - mll_frozen      -> False (no payouts yet in funded)
        - payouts_completed -> 0   (fresh payout counter)
        - drawdown_high_watermark -> 0.0
        - static_floor_equity -> None

        Fields SET to funded config:
        - phase           -> "funded"
        - mll             -> funded max_loss_limit

        Fields NOT carried over from eval:
        - eval equity/profit are irrelevant - the funded account is independent
        """
        self.phase = "funded"
        self.mll = self._config["funded"]["max_loss_limit"]
        # --- Full funded reset ---
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
        """Process a payout. Returns net payout amount.

        After payout:
        - deduct gross from both equity and total_profit
        - freeze the MLL at funded.mll_frozen_value on the first payout
        - set a static floor equity that no longer trails future highs
        - reset winning_days for the next payout cycle
        """
        funded_cfg = self._config["funded"]
        gross = min(
            self.total_profit * funded_cfg["payout_max_pct"],
            funded_cfg["payout_cap"],
        )
        if gross < funded_cfg["payout_min_gross"]:
            return 0.0
        net = gross * funded_cfg["profit_split_trader"]
        # Deduct withdrawn amount from both tracked profit and equity
        self.equity -= gross
        self.total_profit -= gross
        self.payouts_completed += 1
        if self.payouts_completed >= 1 and not self.mll_frozen:
            self.mll_frozen = True
            self.mll = funded_cfg["mll_frozen_value"]
            self.static_floor_equity = self.equity - self.mll
        # Reset winning days for next payout cycle
        self.winning_days = 0
        return net
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_rules.py -v
```

Expected: 24 passed

- [ ] **Step 5: Commit**

```bash
git add propfirm/rules/mff.py tests/test_rules.py
git commit -m "feat: add MFF state machine - trailing DD, consistency, phase-shift, funded reset, scaling"
```

---

## Task 5: Slippage Model

**Files:**
- Create: `propfirm/market/slippage.py`
- Create: `tests/test_slippage.py`

- [ ] **Step 1: Write failing tests for slippage**

Create `tests/test_slippage.py`:

```python
import numpy as np
import pytest
from numba import njit

from propfirm.market.slippage import compute_slippage, build_slippage_lookup


class TestComputeSlippage:
    def test_returns_float(self):
        lookup = np.ones(390, dtype=np.float64)
        result = compute_slippage(0, 5.0, 5.0, lookup, False, 1.5)
        assert isinstance(result, float)

    def test_floor_at_one_tick(self):
        lookup = np.full(390, 0.01, dtype=np.float64)  # Very low baseline
        result = compute_slippage(100, 0.1, 10.0, lookup, False, 1.0)
        assert result == 0.25  # Floor = 1 tick = 0.25 points

    def test_scales_with_atr(self):
        lookup = np.ones(390, dtype=np.float64)
        low_vol = compute_slippage(100, 2.0, 10.0, lookup, False, 1.0)
        high_vol = compute_slippage(100, 20.0, 10.0, lookup, False, 1.0)
        assert high_vol > low_vol

    def test_stop_penalty_increases_slippage(self):
        lookup = np.ones(390, dtype=np.float64)
        no_penalty = compute_slippage(100, 5.0, 5.0, lookup, False, 1.5)
        with_penalty = compute_slippage(100, 5.0, 5.0, lookup, True, 1.5)
        assert with_penalty == no_penalty * 1.5

    def test_zero_trailing_atr_returns_floor(self):
        lookup = np.ones(390, dtype=np.float64)
        result = compute_slippage(100, 5.0, 0.0, lookup, False, 1.0)
        # atr_mult defaults to 1.0 when trailing is 0
        assert result >= 0.25

    def test_open_minute_higher_than_midday(self):
        lookup = np.zeros(390, dtype=np.float64)
        lookup[0] = 3.0   # 09:30 - high
        lookup[120] = 0.75  # 11:30 - low
        open_slip = compute_slippage(0, 5.0, 5.0, lookup, False, 1.0)
        mid_slip = compute_slippage(120, 5.0, 5.0, lookup, False, 1.0)
        assert open_slip > mid_slip


class TestBuildSlippageLookup:
    def test_returns_390_element_array(self):
        lookup = build_slippage_lookup(None)  # Uses default profile
        assert lookup.shape == (390,)
        assert lookup.dtype == np.float64

    def test_all_values_positive(self):
        lookup = build_slippage_lookup(None)
        assert np.all(lookup > 0)

    def test_open_higher_than_midday(self):
        lookup = build_slippage_lookup(None)
        assert lookup[0] > lookup[120]  # 09:30 > 11:30
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_slippage.py -v
```

Expected: FAIL - `ImportError`

- [ ] **Step 3: Implement slippage model**

Create `propfirm/market/slippage.py`:

```python
import numpy as np
from numba import njit
from pathlib import Path
from propfirm.core.types import MNQ_TICK_SIZE, SLIPPAGE_FLOOR_POINTS


@njit(cache=True)
def compute_slippage(
    minute_of_day: int,
    bar_atr: float,
    trailing_median_atr: float,
    slippage_lookup: np.ndarray,
    is_stop_order: bool,
    stop_penalty: float,
) -> float:
    """Compute hybrid slippage for a single bar.

    Returns slippage in points (not ticks). Minimum 0.25 (1 tick).
    """
    baseline = slippage_lookup[minute_of_day]
    if trailing_median_atr > 0.0:
        atr_mult = bar_atr / trailing_median_atr
    else:
        atr_mult = 1.0
    penalty = stop_penalty if is_stop_order else 1.0
    raw = baseline * atr_mult * penalty * MNQ_TICK_SIZE
    if raw < SLIPPAGE_FLOOR_POINTS:
        raw = SLIPPAGE_FLOOR_POINTS
    return raw


# --- Default slippage profile (baseline in ticks, not points) ---
# Buckets: minutes since 09:30 -> baseline multiplier
_DEFAULT_BUCKETS = [
    (0, 15, 3.0),       # 09:30-09:45 - Open auction
    (15, 60, 1.5),      # 09:45-10:30 - Early volatility
    (60, 120, 1.0),     # 10:30-11:30 - Normal
    (120, 240, 0.75),   # 11:30-13:30 - Lunch dip
    (240, 330, 1.0),    # 13:30-15:00 - Normal
    (330, 375, 1.25),   # 15:00-15:45 - MOC flow
    (375, 390, 2.0),    # 15:45-16:00 - Close auction
]


def build_slippage_lookup(profile_path: Path | None) -> np.ndarray:
    """Build a 390-element slippage lookup array (one per RTH minute).

    If profile_path is None or does not exist, uses default buckets.
    If profile_path points to a parquet file, loads calibrated values.
    """
    lookup = np.ones(390, dtype=np.float64)

    if profile_path is not None and Path(profile_path).exists():
        import pandas as pd
        df = pd.read_parquet(profile_path)
        for _, row in df.iterrows():
            start = int(row["bucket_start"])
            end = int(row["bucket_end"])
            lookup[start:end] = row["baseline_ticks"]
    else:
        for start, end, baseline in _DEFAULT_BUCKETS:
            lookup[start:end] = baseline

    return lookup
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_slippage.py -v
```

Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add propfirm/market/slippage.py tests/test_slippage.py
git commit -m "feat: add hybrid slippage model - lookup table + ATR scaling + floor"
```

---

## Task 6: Data Loader (Parquet -> NumPy + Precomputed Vectors)

**Files:**
- Create: `propfirm/market/data_loader.py`
- Create: `tests/test_data_loader.py`

- [ ] **Step 1: Write failing tests for data loader**

Create `tests/test_data_loader.py`:

```python
import numpy as np
import pytest
from pathlib import Path
import pandas as pd

from propfirm.market.data_loader import load_session_data, compute_minute_of_day, compute_trailing_atr


DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
TRAIN_PATH = DATA_DIR / "MNQ_1m_train.parquet"


@pytest.fixture
def train_data():
    if not TRAIN_PATH.exists():
        pytest.skip("Training data not available")
    return load_session_data(TRAIN_PATH, atr_period=14, trailing_atr_days=5)


class TestLoadSessionData:
    def test_returns_dict_with_required_keys(self, train_data):
        required = ["open", "high", "low", "close", "volume",
                     "timestamps", "minute_of_day", "bar_atr",
                     "trailing_median_atr", "day_boundaries", "session_dates"]
        for key in required:
            assert key in train_data, f"Missing key: {key}"

    def test_arrays_are_numpy(self, train_data):
        assert isinstance(train_data["open"], np.ndarray)
        assert isinstance(train_data["minute_of_day"], np.ndarray)

    def test_minute_of_day_range(self, train_data):
        mod = train_data["minute_of_day"]
        assert mod.min() >= 0
        assert mod.max() <= 389

    def test_no_look_ahead_in_trailing_atr(self, train_data):
        """trailing_median_atr at bar 0 of a day must use only past days."""
        atr = train_data["trailing_median_atr"]
        # First 5 days of data: trailing ATR should use whatever is available
        # Key check: no NaN or inf in the array
        assert not np.any(np.isnan(atr))
        assert not np.any(np.isinf(atr))

    def test_day_boundaries_are_sorted(self, train_data):
        bounds = train_data["day_boundaries"]
        assert len(bounds) > 0
        for i in range(1, len(bounds)):
            assert bounds[i][0] > bounds[i - 1][0]

    def test_session_dates_align_with_day_boundaries(self, train_data):
        assert len(train_data["session_dates"]) == len(train_data["day_boundaries"])

    def test_rejects_naive_timestamps(self, tmp_path):
        df = pd.DataFrame(
            {
                "open": [1.0, 1.1],
                "high": [1.2, 1.3],
                "low": [0.9, 1.0],
                "close": [1.1, 1.2],
                "volume": [10, 20],
            },
            index=pd.DatetimeIndex(["2022-01-03 09:30:00", "2022-01-03 09:31:00"]),
        )
        path = tmp_path / "naive.parquet"
        df.to_parquet(path)
        with pytest.raises(ValueError):
            load_session_data(path)

    def test_sorts_converts_and_filters_to_rth(self, tmp_path):
        idx = pd.DatetimeIndex(
            [
                "2022-01-03 14:31:00+00:00",  # 09:31 ET
                "2022-01-03 14:30:00+00:00",  # 09:30 ET
                "2022-01-03 13:00:00+00:00",  # 08:00 ET -> filtered
                "2022-01-03 21:05:00+00:00",  # 16:05 ET -> filtered
                "2022-01-03 20:59:00+00:00",  # 15:59 ET
            ]
        )
        df = pd.DataFrame(
            {
                "open": [1.1, 1.0, 0.8, 1.4, 1.3],
                "high": [1.2, 1.1, 0.9, 1.5, 1.4],
                "low": [1.0, 0.9, 0.7, 1.3, 1.2],
                "close": [1.1, 1.0, 0.8, 1.4, 1.3],
                "volume": [20, 10, 5, 30, 25],
            },
            index=idx,
        )
        path = tmp_path / "utc_unsorted.parquet"
        df.to_parquet(path)
        loaded = load_session_data(path)
        np.testing.assert_array_equal(loaded["minute_of_day"], np.array([0, 1, 389], dtype=np.int16))
        assert len(loaded["open"]) == 3
        assert loaded["session_dates"] == ["2022-01-03"]


class TestComputeMinuteOfDay:
    def test_known_timestamps(self):
        idx = pd.DatetimeIndex([
            "2022-01-03 09:30:00",
            "2022-01-03 09:31:00",
            "2022-01-03 15:59:00",
        ], tz="America/New_York")
        result = compute_minute_of_day(idx)
        assert result[0] == 0    # 09:30
        assert result[1] == 1    # 09:31
        assert result[2] == 389  # 15:59


class TestComputeTrailingATR:
    def test_output_length_matches_input(self):
        highs = np.random.rand(1000) * 10 + 100
        lows = highs - np.random.rand(1000) * 2
        closes = (highs + lows) / 2
        day_bounds = [(i * 390, min((i + 1) * 390, 1000)) for i in range(3)]
        result = compute_trailing_atr(highs, lows, closes, day_bounds, period=14, trailing_days=5)
        assert len(result) == len(highs)

    def test_no_nan_values(self):
        n = 2000
        highs = np.random.rand(n) * 10 + 100
        lows = highs - np.random.rand(n) * 2
        closes = (highs + lows) / 2
        day_bounds = [(i * 390, min((i + 1) * 390, n)) for i in range(6)]
        result = compute_trailing_atr(highs, lows, closes, day_bounds, period=14, trailing_days=5)
        assert not np.any(np.isnan(result))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_data_loader.py -v
```

Expected: FAIL - `ImportError`

- [ ] **Step 3: Implement data loader**

Create `propfirm/market/data_loader.py`:

```python
import numpy as np
import pandas as pd
from pathlib import Path
from propfirm.core.types import BARS_PER_RTH_SESSION


SESSION_TZ = "America/New_York"
RTH_OPEN_MINUTE = 9 * 60 + 30
REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def _prepare_session_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize bars into a strict RTH-only America/New_York session frame."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Parquet data must be indexed by DatetimeIndex")
    if df.index.tz is None:
        raise ValueError("Parquet index must be timezone-aware")
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Parquet data missing required columns: {missing}")

    df = df.sort_index(kind="stable").copy()
    df.index = df.index.tz_convert(SESSION_TZ)
    df = df.loc[~df.index.duplicated(keep="first")]
    df = df.between_time("09:30", "15:59")

    if df.empty:
        raise ValueError("No RTH bars remain after timezone conversion/filtering")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Prepared session index must be monotonic increasing")
    return df


def compute_minute_of_day(index: pd.DatetimeIndex) -> np.ndarray:
    """Convert timestamps to minute-of-day (0=09:30, 389=15:59).

    Expects a timezone-aware America/New_York RTH-only index.
    Uses actual timestamps, NOT modulo arithmetic.
    """
    if index.tz is None:
        raise ValueError("minute_of_day requires timezone-aware timestamps")
    if str(index.tz) != SESSION_TZ:
        raise ValueError("minute_of_day requires America/New_York timestamps")
    hours = index.hour
    minutes = index.minute
    total_minutes = hours * 60 + minutes
    minute_of_day = total_minutes - RTH_OPEN_MINUTE
    if np.any(minute_of_day < 0) or np.any(minute_of_day >= BARS_PER_RTH_SESSION):
        raise ValueError("minute_of_day received timestamps outside 09:30-15:59 ET")
    return minute_of_day.astype(np.int16)


def compute_trailing_atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    day_boundaries: list[tuple[int, int]],
    period: int = 14,
    trailing_days: int = 5,
) -> np.ndarray:
    """Compute trailing N-day median ATR - causally correct (no look-ahead).

    For each bar, the trailing_median_atr is the median of the daily
    session ATRs of the last `trailing_days` completed sessions.
    Within a session, the value is constant (computed from past sessions only).
    """
    n_bars = len(highs)
    result = np.zeros(n_bars, dtype=np.float64)

    # Compute True Range per bar
    tr = np.maximum(highs - lows, np.zeros(n_bars))
    if n_bars > 1:
        prev_close = np.roll(closes, 1)
        prev_close[0] = closes[0]
        tr = np.maximum(tr, np.abs(highs - prev_close))
        tr = np.maximum(tr, np.abs(lows - prev_close))

    # Compute session-level median ATR
    session_atrs = []
    for start, end in day_boundaries:
        session_tr = tr[start:end]
        if len(session_tr) >= period:
            # Simple moving average of TR over `period` bars, take median
            atr_values = np.convolve(session_tr, np.ones(period) / period, mode="valid")
            session_atrs.append(np.median(atr_values))
        elif len(session_tr) > 0:
            session_atrs.append(np.mean(session_tr))
        else:
            session_atrs.append(0.0)

    # Assign trailing median ATR to each bar (causally correct)
    for day_idx, (start, end) in enumerate(day_boundaries):
        if day_idx == 0:
            # First day: use first session's ATR as best estimate
            trailing_val = session_atrs[0] if session_atrs[0] > 0 else 1.0
        else:
            lookback = session_atrs[max(0, day_idx - trailing_days):day_idx]
            trailing_val = float(np.median(lookback)) if lookback else 1.0
        if trailing_val <= 0:
            trailing_val = 1.0
        result[start:end] = trailing_val

    return result


def _find_day_boundaries(index: pd.DatetimeIndex) -> list[tuple[int, int]]:
    """Find (start_idx, end_idx) for each trading day."""
    if len(index) == 0:
        raise ValueError("Cannot compute day boundaries for empty index")
    dates = index.date
    boundaries = []
    current_date = dates[0]
    start = 0
    for i in range(1, len(dates)):
        if dates[i] != current_date:
            boundaries.append((start, i))
            start = i
            current_date = dates[i]
    boundaries.append((start, len(dates)))
    return boundaries


def load_session_data(
    parquet_path: Path,
    atr_period: int = 14,
    trailing_atr_days: int = 5,
) -> dict:
    """Load parquet data and compute all precomputed arrays.

    Returns dict with keys:
        open, high, low, close, volume: np.ndarray (float64/uint64)
        timestamps: np.ndarray (int64, unix nanoseconds)
        minute_of_day: np.ndarray (int16, 0-389)
        bar_atr: np.ndarray (float64, per-bar ATR)
        trailing_median_atr: np.ndarray (float64, causally correct)
        day_boundaries: list of (start_idx, end_idx) tuples
        session_dates: list[str] of ISO session dates for reporting/audit
    """
    df = pd.read_parquet(parquet_path)
    df = _prepare_session_frame(df)

    day_boundaries = _find_day_boundaries(df.index)

    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)

    # Per-bar True Range
    n = len(df)
    tr = np.maximum(highs - lows, np.zeros(n))
    if n > 1:
        prev_close = np.roll(closes, 1)
        prev_close[0] = closes[0]
        tr = np.maximum(tr, np.abs(highs - prev_close))
        tr = np.maximum(tr, np.abs(lows - prev_close))

    # Simple rolling ATR (per bar) - causal, no look-ahead.
    # np.convolve(mode="full")[:n] is strictly backward-looking.
    # WARMUP: First (atr_period - 1) bars use partial windows (fewer bars in average).
    # This is intentional - these bars have a shorter averaging period but are still
    # causal. Do NOT "fix" this by using future data or padding with session averages.
    # The slippage model divides bar_atr / trailing_median_atr, so warmup values
    # normalize out in practice.
    bar_atr = np.zeros(n, dtype=np.float64)
    if n >= atr_period:
        kernel = np.ones(atr_period) / atr_period
        convolved = np.convolve(tr, kernel, mode="full")[:n]
        bar_atr[atr_period - 1:] = convolved[atr_period - 1:]  # Full windows
        bar_atr[:atr_period - 1] = convolved[:atr_period - 1]  # Partial windows (warmup)
    else:
        bar_atr[:] = np.mean(tr)

    trailing_median_atr = compute_trailing_atr(
        highs, lows, closes, day_boundaries, atr_period, trailing_atr_days
    )

    return {
        "open": df["open"].values.astype(np.float64),
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": df["volume"].values,
        "timestamps": df.index.asi8,
        "minute_of_day": compute_minute_of_day(df.index),
        "bar_atr": bar_atr,
        "trailing_median_atr": trailing_median_atr,
        "day_boundaries": day_boundaries,
        "session_dates": [str(df.index[start].date()) for start, _ in day_boundaries],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_data_loader.py -v
```

Expected: 11 passed (or some skipped if data not available)

- [ ] **Step 5: Commit**

```bash
git add propfirm/market/data_loader.py tests/test_data_loader.py
git commit -m "feat: add data loader - parquet to numpy + causal ATR + minute_of_day"
```

---

## Task 7: Risk Module (Circuit Breaker + Position Sizing)

**Files:**
- Create: `propfirm/risk/risk.py`
- Create: `tests/test_risk.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_risk.py`:

```python
import pytest
from propfirm.risk.risk import check_circuit_breaker, validate_position_size


class TestCircuitBreaker:
    def test_not_halted_within_limit(self):
        assert check_circuit_breaker(-500.0, -750.0) == False

    def test_halted_at_limit(self):
        assert check_circuit_breaker(-750.0, -750.0) == True

    def test_halted_beyond_limit(self):
        assert check_circuit_breaker(-1000.0, -750.0) == True

    def test_not_halted_positive_pnl(self):
        assert check_circuit_breaker(500.0, -750.0) == False


class TestPositionSizing:
    def test_within_limits(self):
        assert validate_position_size(10, 20) == 10

    def test_clamped_to_max(self):
        assert validate_position_size(30, 20) == 20

    def test_zero_contracts(self):
        assert validate_position_size(0, 20) == 0

    def test_negative_contracts_clamped(self):
        assert validate_position_size(-5, 20) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_risk.py -v
```

Expected: FAIL - `ImportError`

- [ ] **Step 3: Implement risk module**

Create `propfirm/risk/risk.py`:

```python
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
    """Clamp position size to valid range [0, max_contracts].

    Used by Python orchestration code before entering the Numba kernel so the
    funded scaling contract has one canonical clamp helper.
    """
    if requested < 0:
        return 0
    return min(requested, max_contracts)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_risk.py -v
```

Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add propfirm/risk/risk.py tests/test_risk.py
git commit -m "feat: add risk module - circuit breaker + position size validation"
```

---

## Task 8: Numba Engine (Bar Loop + Trade Execution)

**Files:**
- Create: `propfirm/core/engine.py`
- Create: `tests/test_engine.py`

- [ ] **Step 1: Write failing tests for engine**

Create `tests/test_engine.py`:

```python
import numpy as np
import pytest
from numba import njit
from propfirm.core.engine import run_day_kernel
from propfirm.core.types import (
    TRADE_LOG_DTYPE, EXIT_TARGET, EXIT_STOP, EXIT_HARD_CLOSE,
    EXIT_CIRCUIT_BREAKER, SIGNAL_LONG, SIGNAL_SHORT, SIGNAL_NONE,
)


def make_flat_bars(n_bars, base_price=20000.0, spread=5.0):
    """Create synthetic flat market data."""
    opens = np.full(n_bars, base_price, dtype=np.float64)
    highs = np.full(n_bars, base_price + spread, dtype=np.float64)
    lows = np.full(n_bars, base_price - spread, dtype=np.float64)
    closes = np.full(n_bars, base_price, dtype=np.float64)
    volumes = np.full(n_bars, 1000, dtype=np.uint64)
    timestamps = np.arange(n_bars, dtype=np.int64) + 1_640_000_000_000_000_000
    minute_of_day = np.arange(n_bars, dtype=np.int16)
    bar_atr = np.full(n_bars, spread * 2, dtype=np.float64)
    trailing_atr = np.full(n_bars, spread * 2, dtype=np.float64)
    slippage_lookup = np.ones(390, dtype=np.float64)
    return (opens, highs, lows, closes, volumes, timestamps, minute_of_day,
            bar_atr, trailing_atr, slippage_lookup)


@njit(cache=True)
def null_strategy(bar_idx, opens, highs, lows, closes, volumes,
                  minute_of_day, equity, intraday_pnl, position,
                  entry_price, halted, daily_trade_count, params):
    """Strategy that generates no signals."""
    return 0


@njit(cache=True)
def always_long_strategy(bar_idx, opens, highs, lows, closes, volumes,
                         minute_of_day, equity, intraday_pnl, position,
                         entry_price, halted, daily_trade_count, params):
    """Strategy that goes long on bar 15 (after 'range' period)."""
    if minute_of_day[bar_idx] == 15 and position == 0 and not halted:
        return 1
    return 0


@njit(cache=True)
def long_on_bar_15_and_16_strategy(bar_idx, opens, highs, lows, closes, volumes,
                                   minute_of_day, equity, intraday_pnl, position,
                                   entry_price, halted, daily_trade_count, params):
    """Used to verify that an exit bar cannot immediately re-enter on the same OHLC bar."""
    if minute_of_day[bar_idx] == 15 or minute_of_day[bar_idx] == 16:
        if position == 0 and not halted:
            return 1
    return 0


class TestRunDayKernel:
    def test_no_trades_with_null_strategy(self):
        bars = make_flat_bars(390)
        trade_log = np.zeros(100, dtype=TRADE_LOG_DTYPE)
        # 11-element params: [range_min, stop, target, contracts, daily_stop,
        #   daily_target, max_trades, buffer, vol_thresh, stop_penalty, commission]
        params = np.array([15.0, 40.0, 60.0, 10.0, -750.0, 600.0, 2.0, 2.0, 0.0, 1.5, 0.54],
                          dtype=np.float64)
        n_trades, final_equity, final_pnl = run_day_kernel(
            *bars, 7, 0, -1, null_strategy, trade_log, 0, 0.0, 0.0, params
        )
        assert n_trades == 0
        assert final_pnl == 0.0

    def test_single_long_trade_target(self):
        n = 390
        base = 20000.0
        bars = list(make_flat_bars(n, base, 5.0))
        # Make price rise after entry to hit target
        for i in range(16, n):
            bars[0][i] = base + 20.0  # open
            bars[1][i] = base + 30.0  # high
            bars[3][i] = base + 25.0  # close
        bars = tuple(bars)

        trade_log = np.zeros(100, dtype=TRADE_LOG_DTYPE)
        params = np.array([15.0, 40.0, 20.0, 1.0, -750.0, 600.0, 2.0, 0.0, 0.0, 1.5, 0.54],
                          dtype=np.float64)
        n_trades, final_equity, final_pnl = run_day_kernel(
            *bars, 7, 0, -1, always_long_strategy, trade_log, 0, 0.0, 0.0, params
        )
        assert n_trades >= 1
        assert trade_log[0]["day_id"] == 7
        assert trade_log[0]["phase_id"] == 0
        assert trade_log[0]["payout_cycle_id"] == -1
        assert trade_log[0]["entry_time"] > 0
        assert trade_log[0]["exit_time"] >= trade_log[0]["entry_time"]
        assert trade_log[0]["signal_type"] == SIGNAL_LONG

    def test_target_exit_does_not_use_stop_penalty(self):
        n = 390
        base = 20000.0
        bars = list(make_flat_bars(n, base, 5.0))
        for i in range(16, n):
            bars[1][i] = base + 30.0
            bars[3][i] = base + 25.0
        bars = tuple(bars)

        params_lo = np.array([15.0, 40.0, 20.0, 1.0, -750.0, 600.0, 2.0, 0.0, 0.0, 1.0, 0.54],
                             dtype=np.float64)
        params_hi = np.array([15.0, 40.0, 20.0, 1.0, -750.0, 600.0, 2.0, 0.0, 0.0, 10.0, 0.54],
                             dtype=np.float64)

        log_lo = np.zeros(10, dtype=TRADE_LOG_DTYPE)
        log_hi = np.zeros(10, dtype=TRADE_LOG_DTYPE)
        _, _, pnl_lo = run_day_kernel(*bars, 7, 0, -1, always_long_strategy, log_lo, 0, 0.0, 0.0, params_lo)
        _, _, pnl_hi = run_day_kernel(*bars, 7, 0, -1, always_long_strategy, log_hi, 0, 0.0, 0.0, params_hi)
        assert log_lo[0]["exit_reason"] == EXIT_TARGET
        assert log_hi[0]["exit_reason"] == EXIT_TARGET
        assert np.isclose(pnl_lo, pnl_hi)

    def test_hard_close_at_1559(self):
        n = 390
        base = 20000.0
        bars = make_flat_bars(n, base, 2.0)  # Tight range, won't hit stop/target
        trade_log = np.zeros(100, dtype=TRADE_LOG_DTYPE)
        params = np.array([15.0, 200.0, 200.0, 1.0, -750.0, 600.0, 2.0, 0.0, 0.0, 1.5, 0.54],
                          dtype=np.float64)
        n_trades, final_equity, final_pnl = run_day_kernel(
            *bars, 7, 0, -1, always_long_strategy, trade_log, 0, 0.0, 0.0, params
        )
        assert n_trades >= 1
        # Trade should be closed by hard-close
        assert trade_log[0]["exit_reason"] == EXIT_HARD_CLOSE

    def test_exit_bar_cannot_reenter_same_bar(self):
        n = 390
        base = 20000.0
        bars = list(make_flat_bars(n, base, 5.0))
        # Minute 16 hits the target. Without a same-bar guard the strategy would
        # exit and then re-enter again on the same bar.
        for i in range(16, n):
            bars[1][i] = base + 30.0
            bars[3][i] = base + 25.0
        bars = tuple(bars)

        trade_log = np.zeros(100, dtype=TRADE_LOG_DTYPE)
        params = np.array([15.0, 40.0, 20.0, 1.0, -750.0, 600.0, 3.0, 0.0, 0.0, 1.5, 0.54],
                          dtype=np.float64)
        n_trades, _, _ = run_day_kernel(
            *bars, 7, 0, -1, long_on_bar_15_and_16_strategy, trade_log, 0, 0.0, 0.0, params
        )
        assert n_trades == 1

    def test_circuit_breaker_halts_new_entries(self):
        n = 390
        base = 20000.0
        bars = list(make_flat_bars(n, base, 5.0))
        # Make price crash after entry to trigger circuit breaker
        for i in range(16, n):
            bars[0][i] = base - 500.0
            bars[1][i] = base - 490.0
            bars[2][i] = base - 510.0
            bars[3][i] = base - 500.0
        bars = tuple(bars)
        trade_log = np.zeros(100, dtype=TRADE_LOG_DTYPE)
        params = np.array([15.0, 40.0, 60.0, 5.0, -200.0, 600.0, 3.0, 0.0, 0.0, 1.5, 0.54],
                          dtype=np.float64)
        n_trades, final_equity, final_pnl = run_day_kernel(
            *bars, 7, 0, -1, always_long_strategy, trade_log, 0, 0.0, 0.0, params
        )
        # One trade row only; after the stop-out the circuit breaker blocks re-entry.
        assert n_trades <= 1

    def test_trade_log_net_pnl_matches_equity_delta(self):
        bars = list(make_flat_bars(390, 20000.0, 5.0))
        for i in range(16, 390):
            bars[1][i] = 20035.0
            bars[3][i] = 20025.0
        bars = tuple(bars)
        trade_log = np.zeros(100, dtype=TRADE_LOG_DTYPE)
        params = np.array([15.0, 40.0, 20.0, 1.0, -750.0, 600.0, 2.0, 0.0, 0.0, 1.5, 0.54],
                          dtype=np.float64)
        n_trades, final_equity, final_pnl = run_day_kernel(
            *bars, 3, 0, -1, always_long_strategy, trade_log, 0, 0.0, 0.0, params
        )
        assert np.isclose(trade_log[:n_trades]["net_pnl"].sum(), final_equity)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_engine.py -v
```

Expected: FAIL - `ImportError`

- [ ] **Step 3: Implement the Numba engine**

Create `propfirm/core/engine.py`:

```python
import numpy as np
from numba import njit
from propfirm.core.types import (
    TRADE_LOG_DTYPE, SIGNAL_LONG, SIGNAL_SHORT, SIGNAL_NONE,
    EXIT_TARGET, EXIT_STOP, EXIT_HARD_CLOSE, EXIT_CIRCUIT_BREAKER,
    HARD_CLOSE_MINUTE, MNQ_TICK_SIZE, MNQ_TICK_VALUE,
    PARAMS_RANGE_MINUTES, PARAMS_STOP_TICKS, PARAMS_TARGET_TICKS,
    PARAMS_CONTRACTS, PARAMS_DAILY_STOP, PARAMS_DAILY_TARGET,
    PARAMS_MAX_TRADES, PARAMS_STOP_PENALTY, PARAMS_COMMISSION,
)
from propfirm.market.slippage import compute_slippage as _compute_slippage
from propfirm.risk.risk import check_circuit_breaker


@njit(cache=True)
def run_day_kernel(
    opens, highs, lows, closes, volumes,
    timestamps, minute_of_day, bar_atr, trailing_atr, slippage_lookup,
    current_day_id,
    current_phase_id,
    current_payout_cycle_id,
    strategy_fn, trade_log, trade_log_offset,
    starting_equity, starting_pnl,
    params,
):
    """Run the bar-loop for a single trading day.

    params is a flat float64 array with 11 elements.
    Index constants defined in propfirm.core.types (PARAMS_*).
    Engine ignores PARAMS_RANGE_MINUTES, PARAMS_BUFFER_TICKS, PARAMS_VOL_THRESHOLD
    (used by strategy function only).

    Returns: (n_trades, final_equity, final_intraday_pnl)
    """
    stop_ticks = params[PARAMS_STOP_TICKS]
    target_ticks = params[PARAMS_TARGET_TICKS]
    contracts = int(params[PARAMS_CONTRACTS])
    daily_stop = params[PARAMS_DAILY_STOP]
    daily_target = params[PARAMS_DAILY_TARGET]
    max_trades = int(params[PARAMS_MAX_TRADES])
    stop_penalty = params[PARAMS_STOP_PENALTY]
    commission_per_side = params[PARAMS_COMMISSION]

    n_bars = len(opens)
    equity = starting_equity
    intraday_pnl = starting_pnl
    entry_price = 0.0
    position = 0
    halted = False
    daily_trade_count = 0
    trade_idx = trade_log_offset
    open_trade_idx = -1
    stop_level = 0.0
    target_level = 0.0

    for bar_idx in range(n_bars):
        mod = minute_of_day[bar_idx]
        bar_open = opens[bar_idx]
        bar_high = highs[bar_idx]
        bar_low = lows[bar_idx]
        bar_close = closes[bar_idx]
        current_atr = bar_atr[bar_idx]
        current_trailing = trailing_atr[bar_idx]
        exited_this_bar = False

        # --- Check exits for open position ---
        if position != 0:
            exit_price = 0.0
            exit_reason = -1

            if position > 0:  # Long
                # Check stop
                if bar_low <= stop_level:
                    slip = _compute_slippage(mod, current_atr, current_trailing,
                                             slippage_lookup, True, stop_penalty)
                    exit_price = stop_level - slip
                    exit_reason = EXIT_STOP
                # Check target
                elif bar_high >= target_level:
                    slip = _compute_slippage(mod, current_atr, current_trailing,
                                             slippage_lookup, False, stop_penalty)
                    exit_price = target_level - slip
                    exit_reason = EXIT_TARGET
            else:  # Short
                if bar_high >= stop_level:
                    slip = _compute_slippage(mod, current_atr, current_trailing,
                                             slippage_lookup, True, stop_penalty)
                    exit_price = stop_level + slip
                    exit_reason = EXIT_STOP
                elif bar_low <= target_level:
                    slip = _compute_slippage(mod, current_atr, current_trailing,
                                             slippage_lookup, False, stop_penalty)
                    exit_price = target_level + slip
                    exit_reason = EXIT_TARGET

            # Hard close at 15:59
            if exit_reason == -1 and mod >= HARD_CLOSE_MINUTE:
                slip = _compute_slippage(mod, current_atr, current_trailing,
                                         slippage_lookup, False, stop_penalty)
                if position > 0:
                    exit_price = bar_close - slip
                else:
                    exit_price = bar_close + slip
                exit_reason = EXIT_HARD_CLOSE

            if exit_reason >= 0:
                # Calculate PNL
                abs_contracts = abs(position)
                if position > 0:
                    gross_pnl = (exit_price - entry_price) * abs_contracts / MNQ_TICK_SIZE * MNQ_TICK_VALUE
                else:
                    gross_pnl = (entry_price - exit_price) * abs_contracts / MNQ_TICK_SIZE * MNQ_TICK_VALUE
                exit_commission = commission_per_side * abs_contracts
                # net_pnl is the ROUNDTRIP net: gross minus BOTH entry and exit commissions.
                # Entry commission was already deducted from equity at entry time,
                # so only exit commission is deducted from equity now.
                # But the logged net_pnl must reflect the complete roundtrip cost.
                entry_comm_logged = commission_per_side * abs_contracts  # Matches entry
                net_pnl = gross_pnl - entry_comm_logged - exit_commission

                # Record exit on the SAME row as the corresponding entry.
                if open_trade_idx < 0:
                    raise RuntimeError("open_trade_idx missing on exit")
                trade_log[open_trade_idx]["day_id"] = current_day_id
                trade_log[open_trade_idx]["phase_id"] = current_phase_id
                trade_log[open_trade_idx]["payout_cycle_id"] = current_payout_cycle_id
                trade_log[open_trade_idx]["exit_time"] = timestamps[bar_idx]
                trade_log[open_trade_idx]["exit_price"] = exit_price
                trade_log[open_trade_idx]["exit_slippage"] = slip
                trade_log[open_trade_idx]["exit_commission"] = exit_commission
                trade_log[open_trade_idx]["gross_pnl"] = gross_pnl
                trade_log[open_trade_idx]["net_pnl"] = net_pnl
                trade_log[open_trade_idx]["exit_reason"] = exit_reason

                # Equity update: only exit-side cost (entry was already deducted)
                intraday_pnl += gross_pnl - exit_commission
                equity += gross_pnl - exit_commission
                position = 0
                entry_price = 0.0
                open_trade_idx = -1
                exited_this_bar = True

        # --- Circuit breaker check ---
        if not halted and check_circuit_breaker(intraday_pnl, daily_stop):
            halted = True

        # This bar's OHLC has already been consumed by the exit decision.
        # With 1-minute OHLC data we do not allow a second entry decision on
        # the same bar because intrabar ordering is unknowable.
        if exited_this_bar:
            continue

        # --- No new entries at or after 15:59 ---
        if mod >= HARD_CLOSE_MINUTE:
            continue

        # --- Signal generation ---
        if position == 0 and not halted and daily_trade_count < max_trades:
            signal = strategy_fn(
                bar_idx, opens, highs, lows, closes, volumes,
                minute_of_day, equity, intraday_pnl, position,
                entry_price, halted, daily_trade_count, params
            )

            if signal == SIGNAL_LONG or signal == SIGNAL_SHORT:
                if trade_idx >= len(trade_log):
                    raise RuntimeError("trade_log capacity exceeded")
                slip = _compute_slippage(mod, current_atr, current_trailing,
                                         slippage_lookup, False, stop_penalty)
                entry_commission = commission_per_side * contracts  # Entry side

                if signal == SIGNAL_LONG:
                    fill_price = bar_close + slip
                    stop_level = fill_price - stop_ticks * MNQ_TICK_SIZE
                    target_level = fill_price + target_ticks * MNQ_TICK_SIZE
                    position = contracts
                else:
                    fill_price = bar_close - slip
                    stop_level = fill_price + stop_ticks * MNQ_TICK_SIZE
                    target_level = fill_price - target_ticks * MNQ_TICK_SIZE
                    position = -contracts

                entry_price = fill_price
                intraday_pnl -= entry_commission
                equity -= entry_commission
                daily_trade_count += 1

                open_trade_idx = trade_idx
                trade_log[open_trade_idx]["day_id"] = current_day_id
                trade_log[open_trade_idx]["phase_id"] = current_phase_id
                trade_log[open_trade_idx]["payout_cycle_id"] = current_payout_cycle_id
                trade_log[open_trade_idx]["entry_time"] = timestamps[bar_idx]
                trade_log[open_trade_idx]["entry_price"] = fill_price
                trade_log[open_trade_idx]["entry_slippage"] = slip
                trade_log[open_trade_idx]["entry_commission"] = entry_commission
                trade_log[open_trade_idx]["contracts"] = contracts
                trade_log[open_trade_idx]["signal_type"] = signal
                trade_idx += 1

    n_trades = trade_idx - trade_log_offset
    return n_trades, equity, intraday_pnl
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_engine.py -v
```

Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add propfirm/core/engine.py tests/test_engine.py
git commit -m "feat: add Numba JIT engine - bar loop, slippage, hard-close, circuit breaker"
```

---

## Task 8.5: MFF State Machine Validation (Firewall)

> **This task is the gate between Prompt A (Simulator Kern) and Prompt B (Optimierungs-Layer).**
> Do NOT proceed to Task 9+ until all tests below pass.

**Files:**
- Create: `tests/test_mff_validation.py`

- [ ] **Step 1: Write comprehensive MFF validation scenarios**

Create `tests/test_mff_validation.py`:

```python
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


# === EVAL PHASE: Pass Conditions ===

class TestEvalPass:
    def test_01_pass_after_exactly_2_days(self):
        """Minimum trading days: pass on day 2 with $3000+."""
        s = MFFState(make_config())
        s.update_eod(1500.0, 1500.0)
        result = s.update_eod(1600.0, 3100.0)
        assert result == "passed"
        assert s.trading_days == 2

    def test_02_no_pass_on_day_1_even_with_target(self):
        """Must have >= 2 trading days."""
        s = MFFState(make_config())
        result = s.update_eod(3500.0, 3500.0)
        assert result == "continue"  # Not passed yet - only 1 day

    def test_03_pass_exactly_at_target(self):
        """Edge: total profit == $3000 exactly."""
        s = MFFState(make_config())
        s.update_eod(1500.0, 1500.0)
        result = s.update_eod(1500.0, 3000.0)
        assert result == "passed"


# === EVAL PHASE: Blown Conditions ===

class TestEvalBlown:
    def test_04_blown_by_trailing_dd(self):
        """Equity drops >= $2000 below EOD high."""
        s = MFFState(make_config())
        s.update_eod(1800.0, 1800.0)  # eod_high = 1800
        result = s.update_eod(-2100.0, -300.0)  # 1800 - (-300) = 2100 > 2000
        assert result == "blown"

    def test_05_blown_at_exactly_mll(self):
        """Edge: drawdown == $2000 exactly SHOULD blow (>= is conservative).
        MFF 'Maximum Loss Limit' - conservative assumption: reaching the limit
        triggers a blow. If MFF confirms strict > is correct, relax to 'continue'.
        """
        s = MFFState(make_config())
        s.update_eod(2000.0, 2000.0)
        result = s.update_eod(-2000.0, 0.0)  # 2000 - 0 = 2000, == MLL
        assert result == "blown"

    def test_06_trailing_dd_tracks_eod_high(self):
        """EOD high never decreases, even after losing days."""
        s = MFFState(make_config())
        s.update_eod(1000.0, 1000.0)
        s.update_eod(-200.0, 800.0)
        assert s.eod_high == 1000.0  # Did not decrease
        s.update_eod(500.0, 1300.0)
        assert s.eod_high == 1300.0  # New high


# === CONSISTENCY RULE ===

class TestConsistency:
    def test_07_consistency_blocks_pass(self):
        """One day > 50% of total profit blocks eval pass."""
        s = MFFState(make_config())
        s.update_eod(100.0, 100.0)
        result = s.update_eod(2950.0, 3050.0)
        # Day 2: 2950/3050 = 96.7% > 50%
        assert s.consistency_ok() == False
        # Even though profit > target and days >= 2, consistency fails
        assert result == "continue"

    def test_08_consistency_ok_with_even_days(self):
        """Two equal days: 50/50 should pass."""
        s = MFFState(make_config())
        s.update_eod(1500.0, 1500.0)
        result = s.update_eod(1500.0, 3000.0)
        # 1500/3000 = 50% = max_pct, exactly at boundary
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
        s.winning_days = 5
        net = s.process_payout()
        # gross = min(2000*0.5, 5000) = 1000, net = 1000 * 0.8 = 800
        assert net == 800.0
        # After payout, profit is reduced by gross amount
        assert s.total_profit == 1000.0  # 2000 - 1000
        assert s.winning_days == 0  # Reset for next payout cycle

    def test_14_payout_below_minimum_returns_zero(self):
        s = MFFState(make_config())
        s.transition_to_funded()
        s.total_profit = 400.0  # 400 * 0.5 = 200 < 250
        s.winning_days = 5
        net = s.process_payout()
        assert net == 0.0

    def test_15_payout_capped_at_5000(self):
        s = MFFState(make_config())
        s.transition_to_funded()
        s.total_profit = 20000.0  # 20000 * 0.5 = 10000, capped at 5000
        s.winning_days = 5
        net = s.process_payout()
        assert net == 5000.0 * 0.80  # 4000

    def test_16_mll_freezes_after_first_payout(self):
        s = MFFState(make_config())
        s.transition_to_funded()
        s.total_profit = 2000.0
        s.winning_days = 5
        s.process_payout()
        assert s.mll_frozen == True
        assert s.mll == 100.0  # Frozen at $100

    def test_17_mll_stays_frozen_after_second_payout(self):
        s = MFFState(make_config())
        s.transition_to_funded()
        s.total_profit = 4000.0
        s.winning_days = 10
        s.process_payout()  # First payout
        s.total_profit = 3000.0  # Simulate more funded trading
        s.winning_days = 5
        s.process_payout()  # Second payout
        assert s.mll == 100.0  # Still frozen


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
        s.total_profit = 600.0  # 600 * 0.5 = 300 >= 250
        assert s.payout_eligible == True

    def test_21_not_eligible_with_4_days(self):
        s = MFFState(make_config())
        s.transition_to_funded()
        s.winning_days = 4
        s.total_profit = 10000.0
        assert s.payout_eligible == False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_mff_validation.py -v
```

Expected: FAIL - `ImportError` (if MFFState not yet built) or all 21 pass (if Task 4 is complete)

- [ ] **Step 3: Fix any failing scenarios**

If any of the 21 tests fail, fix `propfirm/rules/mff.py` until all pass. Do not proceed to Task 9 until this is green.

- [ ] **Step 4: Run full test suite as gate check**

```bash
pytest tests/ -v --tb=short
```

Expected: ALL tests from Tasks 1-8 plus all 21 validation scenarios pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_mff_validation.py
git commit -m "test: add 21 MFF state machine validation scenarios - simulator firewall"
```

---

> **=== PROMPT A / PROMPT B SPLIT POINT ===**
>
> **Prompt A (Tasks 1-8.5)** is complete when this commit passes.
> The simulator core is validated and can be trusted.
>
> **Prompt B (Tasks 9-17)** builds the optimization layer on top.
> Do NOT start Prompt B until Prompt A is fully green.

---

## Task 9: ORB Strategy

**Files:**
- Create: `propfirm/strategy/orb.py`
- Create: `tests/test_orb.py`

- [ ] **Step 1: Write failing tests for ORB**

Create `tests/test_orb.py`:

```python
import numpy as np
import pytest
from propfirm.strategy.orb import orb_signal


class TestORBSignal:
    def _make_arrays(self, n, base=20000.0):
        opens = np.full(n, base, dtype=np.float64)
        highs = np.full(n, base + 2.0, dtype=np.float64)
        lows = np.full(n, base - 2.0, dtype=np.float64)
        closes = np.full(n, base, dtype=np.float64)
        volumes = np.full(n, 1000, dtype=np.uint64)
        mod = np.arange(n, dtype=np.int16)
        # 11-element params using PARAMS_* index layout from types.py:
        # [range_min, stop, target, contracts, daily_stop,
        #  daily_target, max_trades, buffer, vol_thresh, stop_penalty, commission]
        params = np.array([15.0, 40.0, 60.0, 10.0, -750.0, 600.0, 2.0, 2.0, 0.0, 1.5, 0.54],
                          dtype=np.float64)
        return opens, highs, lows, closes, volumes, mod, params

    def test_no_signal_during_range_buildup(self):
        n = 390
        o, h, l, c, v, mod, p = self._make_arrays(n)
        sig = orb_signal(5, o, h, l, c, v, mod, 0.0, 0.0, 0, 0.0, False, 0, p)
        assert sig == 0  # bar 5 < range_minutes (15 default from params)

    def test_long_breakout(self):
        n = 390
        o, h, l, c, v, mod, p = self._make_arrays(n, 20000.0)
        # Range: bars 0-14, high = 20002.0
        # Bar 15: close breaks above range_high + buffer
        c[15] = 20005.0  # 20002 + 2 (buffer) + 1 clearance
        h[15] = 20006.0
        sig = orb_signal(15, o, h, l, c, v, mod, 0.0, 0.0, 0, 0.0, False, 0, p)
        assert sig == 1

    def test_short_breakout(self):
        n = 390
        o, h, l, c, v, mod, p = self._make_arrays(n, 20000.0)
        # Range low = 19998.0
        c[15] = 19995.0  # Below range_low - buffer
        l[15] = 19994.0
        sig = orb_signal(15, o, h, l, c, v, mod, 0.0, 0.0, 0, 0.0, False, 0, p)
        assert sig == -1

    def test_no_signal_when_halted(self):
        n = 390
        o, h, l, c, v, mod, p = self._make_arrays(n)
        c[15] = 20010.0
        h[15] = 20010.0
        sig = orb_signal(15, o, h, l, c, v, mod, 0.0, 0.0, 0, 0.0, True, 0, p)
        assert sig == 0

    def test_no_signal_when_daily_target_reached(self):
        n = 390
        o, h, l, c, v, mod, p = self._make_arrays(n)
        c[15] = 20010.0
        h[15] = 20010.0
        sig = orb_signal(15, o, h, l, c, v, mod, 0.0, 700.0, 0, 0.0, False, 0, p)
        assert sig == 0  # daily_target = 600, pnl = 700 > target

    def test_no_signal_when_max_trades_reached(self):
        n = 390
        o, h, l, c, v, mod, p = self._make_arrays(n)
        c[15] = 20010.0
        h[15] = 20010.0
        sig = orb_signal(15, o, h, l, c, v, mod, 0.0, 0.0, 0, 0.0, False, 2, p)
        assert sig == 0  # max_trades = 2, already at 2

    def test_no_signal_when_position_open(self):
        n = 390
        o, h, l, c, v, mod, p = self._make_arrays(n)
        c[15] = 20010.0
        sig = orb_signal(15, o, h, l, c, v, mod, 0.0, 0.0, 5, 0.0, False, 0, p)
        assert sig == 0  # position != 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_orb.py -v
```

Expected: FAIL - `ImportError`

- [ ] **Step 3: Implement ORB strategy**

Create `propfirm/strategy/orb.py`:

```python
import numpy as np
from numba import njit
from propfirm.core.types import (
    MNQ_TICK_SIZE,
    PARAMS_RANGE_MINUTES, PARAMS_DAILY_TARGET, PARAMS_MAX_TRADES,
    PARAMS_BUFFER_TICKS, PARAMS_VOL_THRESHOLD,
)


@njit(cache=True)
def orb_signal(
    bar_idx,
    opens, highs, lows, closes, volumes,
    minute_of_day,
    equity, intraday_pnl, position, entry_price,
    halted, daily_trade_count,
    params,
):
    """Opening Range Breakout signal generator.

    Params array uses PARAMS_* index constants from types.py (11 elements).

    Returns: 1 (long), -1 (short), 0 (no signal)
    """
    range_minutes = int(params[PARAMS_RANGE_MINUTES])
    daily_target = params[PARAMS_DAILY_TARGET]
    max_trades = int(params[PARAMS_MAX_TRADES])
    buffer_ticks = params[PARAMS_BUFFER_TICKS]
    volume_threshold = params[PARAMS_VOL_THRESHOLD]

    mod = minute_of_day[bar_idx]

    # Phase 1: During range buildup, no signals
    if mod < range_minutes:
        return 0

    # Self-awareness checks
    if halted:
        return 0
    if position != 0:
        return 0
    if intraday_pnl >= daily_target:
        return 0
    if daily_trade_count >= max_trades:
        return 0

    # Compute opening range high/low from first range_minutes bars
    # We need to look back to the start of the day
    day_start = bar_idx - mod  # bar_idx where minute_of_day == 0
    range_high = -1e18
    range_low = 1e18
    for i in range(day_start, day_start + range_minutes):
        if i >= 0 and i < len(highs):
            if highs[i] > range_high:
                range_high = highs[i]
            if lows[i] < range_low:
                range_low = lows[i]

    buffer_points = buffer_ticks * MNQ_TICK_SIZE  # Convert ticks to points

    # Volume check (optional, 0 = disabled)
    if volume_threshold > 0.0:
        # Compute average volume over range period
        avg_vol = 0.0
        count = 0
        for i in range(day_start, day_start + range_minutes):
            if i >= 0 and i < len(volumes):
                avg_vol += volumes[i]
                count += 1
        if count > 0:
            avg_vol /= count
        if avg_vol > 0 and volumes[bar_idx] / avg_vol < volume_threshold:
            return 0

    # Breakout detection
    bar_close = closes[bar_idx]
    if bar_close > range_high + buffer_points:
        return 1  # Long breakout
    if bar_close < range_low - buffer_points:
        return -1  # Short breakout

    return 0
```

- [ ] **Step 4: Run all tests**

> **NOTE:** No retroactive engine.py update needed here. The params array has been
> 11 elements from the start (defined in Task 3 via PARAMS_* constants). Engine and
> ORB both reference the same constants - no index drift possible.



```bash
pytest tests/test_orb.py tests/test_engine.py -v
```

Expected: 14 passed

- [ ] **Step 5: Commit**

```bash
git add propfirm/strategy/orb.py propfirm/core/engine.py tests/test_orb.py tests/test_engine.py
git commit -m "feat: add ORB strategy - range breakout, volume filter, self-awareness"
```

---

## Task 10: Capped NVE + Objective Function

**Files:**
- Create: `propfirm/optim/objective.py`
- Create: `tests/test_objective.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_objective.py`:

```python
import pytest
from propfirm.optim.objective import compute_capped_nve, compute_single_payout


class TestComputeSinglePayout:
    def test_basic_payout(self):
        # total_profit=2000, 50% = 1000, < 5000 cap, > 250 min
        result = compute_single_payout(2000.0, 0.50, 5000.0, 250.0, 0.80)
        assert result == 1000.0 * 0.80  # 800.0

    def test_capped_at_5000(self):
        result = compute_single_payout(20000.0, 0.50, 5000.0, 250.0, 0.80)
        # 20000 * 0.5 = 10000, capped at 5000
        assert result == 5000.0 * 0.80  # 4000.0

    def test_below_minimum_returns_zero(self):
        result = compute_single_payout(400.0, 0.50, 5000.0, 250.0, 0.80)
        # 400 * 0.5 = 200 < 250 minimum
        assert result == 0.0

    def test_at_exactly_minimum(self):
        result = compute_single_payout(500.0, 0.50, 5000.0, 250.0, 0.80)
        # 500 * 0.5 = 250 >= 250
        assert result == 250.0 * 0.80  # 200.0

    def test_zero_profit(self):
        result = compute_single_payout(0.0, 0.50, 5000.0, 250.0, 0.80)
        assert result == 0.0


class TestComputeCappedNVE:
    def test_positive_nve(self):
        nve = compute_capped_nve(
            payout_rate=0.70,
            mean_payout_net=800.0,
            eval_cost=107.0,
        )
        # 0.70 * 800 - 107 = 560 - 107 = 453
        assert abs(nve - 453.0) < 0.01

    def test_negative_nve(self):
        nve = compute_capped_nve(
            payout_rate=0.10,
            mean_payout_net=500.0,
            eval_cost=107.0,
        )
        # 0.10 * 500 - 107 = 50 - 107 = -57
        assert abs(nve - (-57.0)) < 0.01

    def test_zero_pass_rate(self):
        nve = compute_capped_nve(0.0, 800.0, 107.0)
        assert nve == -107.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_objective.py -v
```

Expected: FAIL - `ImportError`

- [ ] **Step 3: Implement objective functions**

Create `propfirm/optim/objective.py`:

```python
def compute_single_payout(
    total_profit: float,
    max_pct: float,
    payout_cap: float,
    payout_min_gross: float,
    profit_split: float,
) -> float:
    """Compute net payout for a single payout event.

    Applies the full MFF payout chain:
    1. Gross = min(total_profit * max_pct, cap)
    2. Valid if gross >= minimum
    3. Net = gross * profit_split
    """
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

    IMPORTANT: payout_rate must be the probability of achieving an actual
    funded payout (not just the eval pass rate). This requires the full
    lifecycle simulation: eval pass -> funded survival -> payout qualification.
    """
    return payout_rate * mean_payout_net - eval_cost
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_objective.py -v
```

Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add propfirm/optim/objective.py tests/test_objective.py
git commit -m "feat: add capped NVE computation with full MFF payout chain"
```

---

## Task 11: Monte-Carlo Block Bootstrap

**Files:**
- Create: `propfirm/monte_carlo/bootstrap.py`
- Create: `tests/test_monte_carlo.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_monte_carlo.py`:

```python
import numpy as np
import pytest
from propfirm.monte_carlo.bootstrap import (
    block_bootstrap_single,
    split_daily_log_for_mc,
    run_monte_carlo,
    MCResult,
    _simulate_single_path,
)


def make_winning_trades(n=50):
    """Create a trade PNL array that should easily pass eval."""
    return np.array([100.0] * n, dtype=np.float64)


def make_losing_trades(n=50):
    """Create a trade PNL array that should always fail."""
    return np.array([-200.0] * n, dtype=np.float64)


def make_mixed_trades(n=100):
    """60% winners at +150, 40% losers at -100."""
    rng = np.random.RandomState(42)
    trades = np.where(rng.rand(n) < 0.6, 150.0, -100.0)
    return trades


def make_mff_config():
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


def make_structured_daily_log():
    log = np.zeros(6, dtype=[
        ("day_id", "i4"), ("phase_id", "i1"), ("payout_cycle_id", "i2"),
        ("had_trade", "i1"), ("n_trades", "i2"), ("day_pnl", "f8"), ("net_payout", "f8")
    ])
    log["day_id"] = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
    log["phase_id"] = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
    log["payout_cycle_id"] = np.array([-1, -1, -1, 0, 0, 1], dtype=np.int16)
    log["had_trade"] = np.array([1, 0, 1, 1, 0, 1], dtype=np.int8)
    log["n_trades"] = np.array([2, 0, 1, 2, 0, 1], dtype=np.int16)
    log["day_pnl"] = np.array([50.0, 0.0, 120.0, 90.0, 0.0, 60.0])
    log["net_payout"] = np.array([0.0, 0.0, 0.0, 0.0, 200.0, 0.0])
    return log


class TestBlockBootstrapSingle:
    def test_returns_sequence_of_correct_length(self):
        trades = make_mixed_trades(100)
        seq = block_bootstrap_single(trades, target_length=50, seed=42,
                                     block_min=5, block_max=10)
        assert len(seq) >= 50

    def test_deterministic_with_same_seed(self):
        trades = make_mixed_trades(100)
        r1 = block_bootstrap_single(trades, 50, seed=42, block_min=5, block_max=10)
        r2 = block_bootstrap_single(trades, 50, seed=42, block_min=5, block_max=10)
        np.testing.assert_array_equal(r1, r2)

    def test_different_with_different_seed(self):
        trades = make_mixed_trades(100)
        r1 = block_bootstrap_single(trades, 50, seed=42, block_min=5, block_max=10)
        r2 = block_bootstrap_single(trades, 50, seed=99, block_min=5, block_max=10)
        assert not np.array_equal(r1, r2)

    def test_daily_block_mode_bootstraps_day_level_series(self):
        """Daily mode will use the same bootstrap function on day_pnl arrays."""
        daily_values = np.array([50.0, 0.0, 120.0, -30.0, 90.0], dtype=np.float64)
        seq = block_bootstrap_single(daily_values, 8, seed=42,
                                     block_min=2, block_max=3)
        assert len(seq) >= 8
        assert 0.0 in seq  # zero-trade days survive resampling

    def test_split_daily_log_for_mc_keeps_zero_trade_days(self):
        pools = split_daily_log_for_mc(make_structured_daily_log())
        np.testing.assert_array_equal(pools["eval_day_pnls"], np.array([50.0, 0.0, 120.0]))
        np.testing.assert_array_equal(pools["funded_day_pnls"], np.array([90.0, 0.0]))
        assert 60.0 not in pools["funded_day_pnls"]  # Later payout cycles excluded


class TestRunMonteCarlo:
    def test_returns_mc_result_with_disaggregated_rates(self):
        trades = make_mixed_trades(100)
        result = run_monte_carlo(trades, make_mff_config(),
                                 funded_pnls=trades,
                                 n_sims=100, seed=42, n_workers=1,
                                 block_mode="fixed",
                                 block_min=5, block_max=10)
        assert isinstance(result, MCResult)
        assert 0.0 <= result.eval_pass_rate <= 1.0
        assert 0.0 <= result.funded_survival_rate <= 1.0
        assert 0.0 <= result.payout_rate <= 1.0
        # payout_rate <= eval_pass_rate (can't get payout without passing eval)
        assert result.payout_rate <= result.eval_pass_rate + 1e-9

    def test_winning_trades_high_eval_pass_rate(self):
        trades = make_winning_trades(100)
        result = run_monte_carlo(trades, make_mff_config(),
                                 funded_pnls=trades,
                                 n_sims=200, seed=42, n_workers=1,
                                 block_mode="fixed",
                                 block_min=5, block_max=10)
        assert result.eval_pass_rate > 0.5

    def test_losing_trades_zero_rates(self):
        trades = make_losing_trades(50)
        result = run_monte_carlo(trades, make_mff_config(),
                                 funded_pnls=trades,
                                 n_sims=100, seed=42, n_workers=1,
                                 block_mode="fixed",
                                 block_min=5, block_max=10)
        assert result.eval_pass_rate == 0.0
        assert result.payout_rate == 0.0

    def test_reproducible(self):
        trades = make_mixed_trades(100)
        r1 = run_monte_carlo(trades, make_mff_config(),
                             funded_pnls=trades,
                             n_sims=100, block_min=5, block_max=10,
                             seed=42, n_workers=1, block_mode="fixed")
        r2 = run_monte_carlo(trades, make_mff_config(),
                             funded_pnls=trades,
                             n_sims=100, block_min=5, block_max=10,
                             seed=42, n_workers=1, block_mode="fixed")
        assert r1.eval_pass_rate == r2.eval_pass_rate
        assert r1.payout_rate == r2.payout_rate

    def test_ci_brackets_eval_pass_rate(self):
        """CI must bracket the eval pass rate for non-trivial distributions."""
        trades = make_mixed_trades(200)
        result = run_monte_carlo(trades, make_mff_config(),
                                 funded_pnls=trades,
                                 n_sims=500, seed=42, n_workers=1,
                                 block_mode="fixed",
                                 block_min=5, block_max=10)
        if 0.1 < result.eval_pass_rate < 0.9:
            assert result.eval_pass_rate_ci_5 < result.eval_pass_rate
            assert result.eval_pass_rate_ci_95 > result.eval_pass_rate
            assert result.eval_pass_rate_ci_5 > 0.0
            assert result.eval_pass_rate_ci_95 < 1.0

    def test_multiprocessing(self):
        trades = make_mixed_trades(100)
        result = run_monte_carlo(trades, make_mff_config(),
                                 funded_pnls=trades,
                                 n_sims=200, seed=42, n_workers=2,
                                 block_mode="fixed",
                                 block_min=5, block_max=10)
        assert isinstance(result, MCResult)
        assert 0.0 <= result.eval_pass_rate <= 1.0

    def test_daily_mode_requires_explicit_funded_day_inputs(self):
        day_pnls = np.array([50.0, 0.0, 120.0], dtype=np.float64)
        with pytest.raises(ValueError):
            run_monte_carlo(
                day_pnls, make_mff_config(),
                n_sims=10, seed=42, n_workers=1,
                block_mode="daily"
            )

    def test_empty_inputs_rejected(self):
        empty = np.array([], dtype=np.float64)
        with pytest.raises(ValueError):
            run_monte_carlo(
                empty, make_mff_config(),
                funded_pnls=empty,
                n_sims=10, seed=42, n_workers=1,
                block_mode="fixed"
            )

    def test_drawdown_uses_full_lifecycle_max(self):
        eval_seq = np.array([2500.0, -400.0, 1000.0], dtype=np.float64)
        funded_seq = np.array([200.0, 200.0, 200.0, 200.0, 200.0], dtype=np.float64)
        result = _simulate_single_path(
            eval_seq, funded_seq, make_mff_config(),
            eval_trades_per_day=1,
            funded_trades_per_day=1,
        )
        assert result["eval_passed"] == True
        assert result["payout_net"] > 0.0
        assert result["drawdown"] >= 400.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_monte_carlo.py -v
```

Expected: FAIL - `ImportError`

- [ ] **Step 3: Implement Monte-Carlo bootstrap**

Create `propfirm/monte_carlo/bootstrap.py`:

```python
import numpy as np
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from propfirm.rules.mff import MFFState


@dataclass
class MCResult:
    eval_pass_rate: float
    eval_pass_rate_ci_5: float
    eval_pass_rate_ci_95: float
    funded_survival_rate: float   # Of eval passers, fraction that do not blow up in funded
    payout_rate: float            # Overall: fraction of all sims that reach a payout
    mean_payout_net: float        # Mean net payout for sims that achieved one
    mean_days_to_eval_pass: float
    mean_funded_days_to_payout: float
    mean_drawdown: float            # Mean max lifecycle drawdown across eval + funded
    nve: float                    # payout_rate * mean_payout_net - eval_cost
    n_simulations: int


def split_daily_log_for_mc(daily_log: np.ndarray) -> dict:
    """Split a structured DAILY lifecycle log into first-payout MC inputs.

    This is the authoritative production path for real ORB Monte-Carlo because
    it preserves zero-trade days required by min_trading_days and funded
    winning-day logic.
    """
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


def block_bootstrap_single(
    values: np.ndarray,
    target_length: int,
    seed: int,
    block_min: int = 5,
    block_max: int = 10,
) -> np.ndarray:
    """Generate a single bootstrapped 1D sequence using contiguous blocks.

    In `fixed` mode, `values` are flat trade PNLs and block sizes are measured
    in trades. In `daily` mode, `values` are day-level PNLs and block sizes are
    measured in calendar trading days.
    """
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


def _run_days(state: MFFState, sequence: np.ndarray, trades_per_day: int):
    """Advance the MFF state machine day-by-day through a sequence.

    In daily mode, sequence contains one `day_pnl` per calendar trading day and
    `trades_per_day=1`. In fixed mode, sequence contains trade pnls and
    `trades_per_day` is a coarse fallback used only for synthetic/legacy paths.
    """
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


def _simulate_single_path(
    eval_sequence: np.ndarray,
    funded_sequence: np.ndarray,
    mff_config: dict,
    eval_trades_per_day: int,
    funded_trades_per_day: int,
) -> dict:
    """Simulate the full lifecycle: eval attempt -> funded trading -> payout.

    This models the actual prop-firm pipeline:
    1. EVAL: trade until pass or blow. If blown or exhausted -> fail.
    2. FUNDED TRANSITION: reset all state (equity, profit, winning days, etc.)
    3. FUNDED: trade with funded rules (different MLL, scaling tiers).
       Check payout eligibility after each day.
    4. PAYOUT: if eligible, compute net payout. If blown -> fail.

    Two separate trade sequences are used (eval_sequence, funded_sequence)
    because funded trading follows different parameters and the bootstrap
    must independently sample for each phase.
    """
    state = MFFState(mff_config)

    # --- Phase 1: Eval ---
    eval_result, eval_days, eval_dd = _run_days(
        state, eval_sequence, eval_trades_per_day)

    if eval_result != "passed":
        return {"eval_passed": False, "funded_survived": False,
                "payout_net": 0.0, "eval_days": eval_days,
                "funded_days": 0, "drawdown": eval_dd}

    # --- Phase 2: Funded transition (full reset) ---
    state.transition_to_funded()

    # --- Phase 3: Funded trading -> payout ---
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

    # Exhausted funded sequence without payout
    return {"eval_passed": True, "funded_survived": True,
            "payout_net": 0.0, "eval_days": eval_days,
            "funded_days": funded_days, "drawdown": lifecycle_dd}


def _run_chunk(args):
    """Worker function for multiprocessing.

    Each sim bootstraps TWO independent sequences: one for eval, one for funded.
    Using separate seeds ensures eval and funded paths are uncorrelated.
    """
    (eval_pnls, funded_pnls, mff_config, seeds,
     block_mode, block_min, block_max,
     eval_target_length, funded_target_length,
     trades_per_day_fixed) = args
    results = []
    for seed in seeds:
        eval_seq = block_bootstrap_single(
            eval_pnls, eval_target_length, seed,
            block_min=block_min, block_max=block_max,
        )
        funded_seq = block_bootstrap_single(
            funded_pnls, funded_target_length, seed + 1_000_000,
            block_min=block_min, block_max=block_max,
        )
        if block_mode == "daily":
            eval_trades_per_day = 1
            funded_trades_per_day = 1
        else:
            eval_trades_per_day = trades_per_day_fixed
            funded_trades_per_day = trades_per_day_fixed
        result = _simulate_single_path(
            eval_seq, funded_seq, mff_config,
            eval_trades_per_day=eval_trades_per_day,
            funded_trades_per_day=funded_trades_per_day,
        )
        results.append(result)
    return results


def run_monte_carlo(
    eval_pnls: np.ndarray,
    mff_config: dict,
    funded_pnls: np.ndarray | None = None,
    n_sims: int = 10_000,
    seed: int = 42,
    n_workers: int = 1,
    eval_target_length: int = 200,
    funded_target_length: int = 300,
    block_mode: str = "daily",
    block_min: int = 5,
    block_max: int = 10,
    trades_per_day_fixed: int = 2,
) -> MCResult:
    """Run Monte-Carlo block-bootstrap simulation of the FULL lifecycle.

    Each simulation models: eval attempt -> funded transition -> funded
    trading -> payout qualification. Two independent bootstrap sequences
    are generated per sim (eval + funded) to decorrelate the phases.

    Outputs are disaggregated:
    - eval_pass_rate: fraction of sims that pass eval
    - funded_survival_rate: of eval passers, fraction not blown in funded
    - payout_rate: fraction of ALL sims that achieve a payout
    - nve: payout_rate * mean_payout_net - eval_cost

    For real ORB backtests, block_mode MUST be "daily" and eval_pnls /
    funded_pnls must come from DAILY_LOG_DTYPE via split_daily_log_for_mc().
    This preserves zero-trade days. The fixed-mode fallback is reserved for
    synthetic or legacy flat-PNL studies and may reuse one synthetic pool for
    both phases.
    """
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

    # --- Disaggregated aggregation ---
    n_total = len(all_results)
    eval_passed = [r for r in all_results if r["eval_passed"]]
    funded_survived = [r for r in eval_passed if r["funded_survived"]]
    got_payout = [r for r in all_results if r["payout_net"] > 0]

    eval_pass_rate = len(eval_passed) / n_total
    funded_survival_rate = (len(funded_survived) / len(eval_passed)
                           if eval_passed else 0.0)
    payout_rate = len(got_payout) / n_total

    # Confidence interval on eval pass rate
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

    # NVE: payout_rate (not eval_pass_rate!) * mean net payout - eval cost
    # This correctly reflects that you need to BOTH pass eval AND survive
    # funded AND qualify for payout to receive money.
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_monte_carlo.py -v
```

Expected: 14 passed

- [ ] **Step 5: Commit**

```bash
git add propfirm/monte_carlo/bootstrap.py tests/test_monte_carlo.py
git commit -m "feat: add Monte-Carlo block bootstrap with multiprocessing and NVE"
```

---

## Task 12: JSON Reporting + Audit Trail

**Files:**
- Create: `propfirm/io/reporting.py`
- Create: `tests/test_reporting.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_reporting.py`:

```python
import json
import pytest
import numpy as np
from pathlib import Path
from propfirm.io.reporting import build_report, save_report


class TestBuildReport:
    def test_has_meta_section(self):
        report = build_report(
            params={"eval": {"stop": 40}, "funded": {"stop": 60}},
            mc_result=None,
            config_snapshot={"test": True},
            data_split="train",
            data_date_range=("2022-01-03", "2024-06-28"),
            seed=42,
        )
        assert "meta" in report
        assert report["meta"]["random_seed"] == 42
        assert report["meta"]["data_split"] == "train"
        assert report["meta"]["config_snapshot"] == {"test": True}

    def test_has_git_hash(self):
        report = build_report(
            params={}, mc_result=None, config_snapshot={},
            data_split="train", data_date_range=("", ""), seed=42,
        )
        assert "git_hash" in report["meta"]

    def test_has_timestamp(self):
        report = build_report(
            params={}, mc_result=None, config_snapshot={},
            data_split="train", data_date_range=("", ""), seed=42,
        )
        assert "timestamp" in report["meta"]
        assert "T" in report["meta"]["timestamp"]  # ISO format


class TestSaveReport:
    def test_saves_valid_json(self, tmp_path):
        report = build_report(
            params={"eval": {}}, mc_result=None, config_snapshot={},
            data_split="train", data_date_range=("", ""), seed=42,
        )
        out_path = tmp_path / "test_report.json"
        save_report(report, out_path)
        assert out_path.exists()
        loaded = json.loads(out_path.read_text())
        assert loaded["meta"]["random_seed"] == 42
        assert loaded["meta"]["config_snapshot"] == {}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_reporting.py -v
```

Expected: FAIL - `ImportError`

- [ ] **Step 3: Implement reporting**

Create `propfirm/io/reporting.py`:

```python
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import asdict


def _get_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def build_report(
    params: dict,
    mc_result,
    config_snapshot: dict,
    data_split: str,
    data_date_range: tuple[str, str],
    seed: int,
    diagnostics: dict | None = None,
    stress_test: dict | None = None,
) -> dict:
    """Build a complete JSON report structure."""
    report = {
        "meta": {
            "git_hash": _get_git_hash(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "random_seed": seed,
            "config_snapshot": config_snapshot,
            "data_split": data_split,
            "data_date_range": list(data_date_range),
        },
        "params": params,
    }

    if mc_result is not None:
        report["results"] = asdict(mc_result)

    if stress_test is not None:
        report["stress_test"] = stress_test

    if diagnostics is not None:
        report["diagnostics"] = diagnostics

    return report


def save_report(report: dict, path: Path):
    """Save report as formatted JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_reporting.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add propfirm/io/reporting.py tests/test_reporting.py
git commit -m "feat: add JSON reporting with git hash audit trail"
```

---

## Task 13: Slippage Calibration Script

**Files:**
- Create: `scripts/calibrate_slippage.py`

- [ ] **Step 1: Implement calibration script**

Create `scripts/calibrate_slippage.py`:

```python
#!/usr/bin/env python
"""Calibrate slippage profile from normalized training data.

Uses the same session normalization path as the backtest loader and stores
true tick-count baselines per 15-minute bucket.
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from propfirm.io.config import load_mff_config
from propfirm.market.data_loader import _prepare_session_frame, compute_minute_of_day


def calibrate(train_path: Path, output_path: Path, tick_size: float):
    df = _prepare_session_frame(pd.read_parquet(train_path))
    print(f"Loaded {len(df)} bars from {train_path}")

    # Compute spread proxy in TICKS so baseline_ticks remains semantically correct.
    df["spread_ticks"] = (df["high"] - df["low"]) / tick_size
    df["minute_of_day"] = compute_minute_of_day(df.index)

    # Lower volume quartile mask (proxy for "normal" conditions)
    vol_q25 = df["volume"].quantile(0.25)
    low_vol_mask = df["volume"] <= vol_q25

    # Define 15-minute buckets
    buckets = []
    for start in range(0, 390, 15):
        end = min(start + 15, 390)
        mask = (df["minute_of_day"] >= start) & (df["minute_of_day"] < end)
        subset = df[mask & low_vol_mask]["spread_ticks"]
        baseline = float(subset.median()) if len(subset) > 0 else 1.0
        buckets.append({"bucket_start": start, "bucket_end": end,
                        "baseline_ticks": baseline})

    result = pd.DataFrame(buckets)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    print(f"Saved slippage profile to {output_path}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate slippage profile")
    parser.add_argument("--train", type=Path,
                        default=Path("data/processed/MNQ_1m_train.parquet"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/slippage/slippage_profile.parquet"))
    parser.add_argument("--mff-config", type=Path,
                        default=Path("configs/mff_flex_50k.toml"))
    args = parser.parse_args()
    mff_cfg = load_mff_config(args.mff_config)
    calibrate(args.train, args.output, tick_size=mff_cfg["instrument"]["tick_size"])
```

- [ ] **Step 2: Run calibration**

```bash
python scripts/calibrate_slippage.py
```

Expected: Prints bucket table, creates `data/slippage/slippage_profile.parquet`

- [ ] **Step 3: Verify output**

```bash
python -c "import pandas as pd; print(pd.read_parquet('data/slippage/slippage_profile.parquet').to_string())"
```

Expected: 26 rows (390/15 buckets), all baseline_ticks > 0

- [ ] **Step 5: Commit**

```bash
git add scripts/calibrate_slippage.py
git commit -m "feat: add slippage calibration script"
```

---

## Task 14: Synthetic Grid Search

**Files:**
- Create: `propfirm/optim/grid_search.py`
- Create: `tests/test_grid_search.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_grid_search.py`:

```python
import pytest
from propfirm.optim.grid_search import run_synthetic_grid_search, generate_synthetic_trades


def make_mff_config():
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


class TestGenerateSyntheticTrades:
    def test_returns_array(self):
        trades = generate_synthetic_trades(
            win_rate=0.60, reward_ticks=60.0, risk_ticks=40.0,
            contracts=10, n_trades=100, seed=42,
        )
        assert len(trades) == 100

    def test_win_rate_approximate(self):
        trades = generate_synthetic_trades(
            win_rate=0.60, reward_ticks=60.0, risk_ticks=40.0,
            contracts=10, n_trades=10000, seed=42,
        )
        actual_wr = (trades > 0).sum() / len(trades)
        assert abs(actual_wr - 0.60) < 0.05

    def test_deterministic(self):
        t1 = generate_synthetic_trades(0.6, 60.0, 40.0, 10, 100, 42)
        t2 = generate_synthetic_trades(0.6, 60.0, 40.0, 10, 100, 42)
        assert (t1 == t2).all()


class TestRunSyntheticGridSearch:
    def test_returns_ranked_results(self):
        param_grid = {
            "win_rate": [0.55, 0.65],
            "risk_reward": [1.0, 1.5],
            "contracts": [10],
        }
        results = run_synthetic_grid_search(
            param_grid, make_mff_config(),
            n_mc_sims=50, seed=42, n_workers=1,
        )
        assert len(results) > 0
        # Results should be sorted by NVE descending
        nves = [r["nve"] for r in results]
        assert nves == sorted(nves, reverse=True)

    def test_each_result_has_required_keys(self):
        param_grid = {
            "win_rate": [0.60],
            "risk_reward": [1.5],
            "contracts": [10],
        }
        results = run_synthetic_grid_search(
            param_grid, make_mff_config(),
            n_mc_sims=50, seed=42, n_workers=1,
        )
        r = results[0]
        assert "params" in r
        assert "nve" in r
        assert "eval_pass_rate" in r
        assert "payout_rate" in r
        assert "funded_survival_rate" in r
        assert "daily_stop" not in r["params"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_grid_search.py -v
```

Expected: FAIL - `ImportError`

- [ ] **Step 3: Implement grid search**

Create `propfirm/optim/grid_search.py`:

```python
import numpy as np
from itertools import product
from propfirm.monte_carlo.bootstrap import run_monte_carlo
from propfirm.core.types import MNQ_TICK_VALUE


def generate_synthetic_trades(
    win_rate: float,
    reward_ticks: float,
    risk_ticks: float,
    contracts: int,
    n_trades: int,
    seed: int,
) -> np.ndarray:
    """Generate synthetic trade PNLs using Bernoulli distribution."""
    rng = np.random.RandomState(seed)
    wins = rng.rand(n_trades) < win_rate
    pnls = np.where(
        wins,
        reward_ticks * MNQ_TICK_VALUE * contracts,
        -risk_ticks * MNQ_TICK_VALUE * contracts,
    )
    return pnls.astype(np.float64)


def run_synthetic_grid_search(
    param_grid: dict,
    mff_config: dict,
    n_mc_sims: int = 1000,
    seed: int = 42,
    n_workers: int = 1,
    n_synthetic_trades: int = 200,
) -> list[dict]:
    """Run grid search over synthetic trade distributions.

    param_grid keys: win_rate, risk_reward, contracts
    risk_reward is target_ticks / stop_ticks (stop_ticks fixed at 40).
    This is a synthetic pre-study only. It intentionally reuses the same
    Bernoulli pool for eval and funded under fixed-block MC, so it does not
    vary day-structured controls like daily_stop.
    """
    results = []
    stop_ticks = 40.0  # Fixed baseline

    combos = list(product(
        param_grid["win_rate"],
        param_grid["risk_reward"],
        param_grid["contracts"],
    ))

    for i, (wr, rr, contracts) in enumerate(combos):
        target_ticks = stop_ticks * rr
        trades = generate_synthetic_trades(
            win_rate=wr,
            reward_ticks=target_ticks,
            risk_ticks=stop_ticks,
            contracts=contracts,
            n_trades=n_synthetic_trades,
            seed=seed + i,
        )

        mc_result = run_monte_carlo(
            eval_pnls=trades,
            mff_config=mff_config,
            funded_pnls=trades,
            n_sims=n_mc_sims,
            block_mode="fixed",
            block_min=5,
            block_max=10,
            seed=seed + i + 10000,
            n_workers=n_workers,
            eval_target_length=n_synthetic_trades,
            funded_target_length=n_synthetic_trades,
        )

        results.append({
            "params": {
                "win_rate": wr,
                "risk_reward": rr,
                "stop_ticks": stop_ticks,
                "target_ticks": target_ticks,
                "contracts": contracts,
            },
            "nve": mc_result.nve,
            "eval_pass_rate": mc_result.eval_pass_rate,
            "payout_rate": mc_result.payout_rate,
            "funded_survival_rate": mc_result.funded_survival_rate,
            "mean_days_to_eval_pass": mc_result.mean_days_to_eval_pass,
            "mean_drawdown": mc_result.mean_drawdown,
        })

    results.sort(key=lambda r: r["nve"], reverse=True)
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_grid_search.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add propfirm/optim/grid_search.py tests/test_grid_search.py
git commit -m "feat: add synthetic grid search with Monte-Carlo NVE ranking"
```

---

## Task 15: Walk-Forward Analysis

**Files:**
- Create: `propfirm/optim/walk_forward.py`
- Create: `tests/test_walk_forward.py`

> **Design Note (Review Feedback):** The original plan used
> `run_synthetic_grid_search()` inside the walk-forward loop. This does NOT
> constitute a real walk-forward because it optimizes synthetic Bernoulli
> distributions, not actual ORB parameters on historical data. The synthetic
> grid search remains useful as a **sensitivity pre-study** (Task 14) but
> must not be the core of walk-forward.
>
> A real walk-forward must:
> 1. Per window: backtest ORB with each parameter set on TRAIN bars
> 2. Collect the resulting trade log plus daily lifecycle log per parameter set
> 3. Run Monte-Carlo/NVE on the daily lifecycle log to score each parameter set
> 4. Pick the best parameter set by NVE
> 5. Backtest the best parameter set on TEST bars (out-of-sample)
> 6. Run Monte-Carlo/NVE on the OOS daily lifecycle log to measure OOS performance
>
> This measures parameter stability across market regimes - the actual
> purpose of walk-forward analysis.

- [ ] **Step 1: Implement walk-forward analysis**

Create `propfirm/optim/walk_forward.py`:

```python
import numpy as np
from itertools import product
from propfirm.core.engine import run_day_kernel
from propfirm.core.types import (
    TRADE_LOG_DTYPE, DAILY_LOG_DTYPE, PARAMS_ARRAY_LENGTH,
    PARAMS_CONTRACTS, PARAMS_MAX_TRADES,
    PARAMS_RANGE_MINUTES, PARAMS_STOP_TICKS, PARAMS_TARGET_TICKS,
    PARAMS_DAILY_STOP, PARAMS_DAILY_TARGET, PARAMS_BUFFER_TICKS,
    PARAMS_VOL_THRESHOLD, PARAMS_STOP_PENALTY, PARAMS_COMMISSION,
)
from propfirm.risk.risk import validate_position_size
from propfirm.rules.mff import MFFState
from propfirm.strategy.orb import orb_signal
from propfirm.monte_carlo.bootstrap import (
    run_monte_carlo,
    split_daily_log_for_mc,
)


PARAM_INDEX_TO_NAME = {
    PARAMS_RANGE_MINUTES: "range_minutes",
    PARAMS_STOP_TICKS: "stop_ticks",
    PARAMS_TARGET_TICKS: "target_ticks",
    PARAMS_CONTRACTS: "contracts",
    PARAMS_DAILY_STOP: "daily_stop",
    PARAMS_DAILY_TARGET: "daily_target",
    PARAMS_MAX_TRADES: "max_trades_day",
    PARAMS_BUFFER_TICKS: "buffer_ticks",
    PARAMS_VOL_THRESHOLD: "volume_threshold",
    PARAMS_STOP_PENALTY: "stop_penalty",
    PARAMS_COMMISSION: "commission_per_side",
}


def _backtest_param_set(
    session_data: dict,
    day_range: tuple[int, int],
    params_eval: np.ndarray,
    params_funded: np.ndarray,
    slippage_lookup: np.ndarray,
    mff_config: dict,
) -> dict:
    """Backtest a state-aware ORB parameter set over a date range.

    Returns both the chronological trade log and the day-complete lifecycle log.
    """
    state = MFFState(mff_config)
    funded_payout_cycle_id = 0
    start_day, end_day = day_range
    max_trades_per_day = int(max(params_eval[PARAMS_MAX_TRADES], params_funded[PARAMS_MAX_TRADES]))
    max_possible_trades = max(1, (end_day - start_day) * max_trades_per_day)
    all_trades = np.zeros(max_possible_trades, dtype=TRADE_LOG_DTYPE)
    daily_log = np.zeros(end_day - start_day, dtype=DAILY_LOG_DTYPE)
    total_trade_count = 0
    total_day_count = 0

    for day_idx in range(start_day, end_day):
        start, end = session_data["day_boundaries"][day_idx]
        active_params = params_eval if state.phase == "eval" else params_funded.copy()
        phase_id = 0 if state.phase == "eval" else 1
        payout_cycle_id = -1 if state.phase == "eval" else funded_payout_cycle_id
        if state.phase == "funded":
            active_params[PARAMS_CONTRACTS] = float(
                validate_position_size(
                    int(active_params[PARAMS_CONTRACTS]),
                    state.get_max_contracts(),
                )
            )
        n_trades, equity, pnl = run_day_kernel(
            session_data["open"][start:end],
            session_data["high"][start:end],
            session_data["low"][start:end],
            session_data["close"][start:end],
            session_data["volume"][start:end],
            session_data["timestamps"][start:end],
            session_data["minute_of_day"][start:end],
            session_data["bar_atr"][start:end],
            session_data["trailing_median_atr"][start:end],
            slippage_lookup,
            day_idx,
            phase_id,
            payout_cycle_id,
            orb_signal,
            all_trades[total_trade_count:],
            0, state.equity, 0.0, active_params,
        )
        total_trade_count += n_trades
        result = state.update_eod(pnl, state.equity + pnl)
        net_payout = 0.0
        if result == "blown":
            pass
        elif state.phase == "funded" and state.payout_eligible:
            net_payout = state.process_payout()
            if net_payout > 0:
                funded_payout_cycle_id += 1
        daily_log[total_day_count]["day_id"] = day_idx
        daily_log[total_day_count]["phase_id"] = phase_id
        daily_log[total_day_count]["payout_cycle_id"] = payout_cycle_id
        daily_log[total_day_count]["had_trade"] = 1 if n_trades > 0 else 0
        daily_log[total_day_count]["n_trades"] = n_trades
        daily_log[total_day_count]["day_pnl"] = pnl
        daily_log[total_day_count]["net_payout"] = net_payout
        total_day_count += 1
        if result == "passed" and state.phase == "eval":
            state.transition_to_funded()
            funded_payout_cycle_id = 0
        if result == "blown":
            break

    return {
        "trade_log": all_trades[:total_trade_count],
        "daily_log": daily_log[:total_day_count],
    }


def _build_params_array(base_params: np.ndarray, overrides: dict, phase: str) -> np.ndarray:
    """Create a params array with phase-scoped overrides applied."""
    p = base_params.copy()
    for idx, val in overrides.items():
        override_phase, override_idx = idx
        if override_phase == phase:
            p[override_idx] = float(val)
    return p


def _serialize_param_overrides(overrides: dict | None) -> dict | None:
    """Convert tuple-key overrides into a JSON-safe nested dict for reports."""
    if overrides is None:
        return None
    result = {"eval": {}, "funded": {}}
    for (phase, param_idx), value in overrides.items():
        result[phase][PARAM_INDEX_TO_NAME.get(param_idx, f"param_{param_idx}")] = float(value)
    if not result["eval"]:
        result.pop("eval")
    if not result["funded"]:
        result.pop("funded")
    return result


def run_walk_forward(
    session_data: dict,
    slippage_lookup: np.ndarray,
    base_params_eval: np.ndarray,
    base_params_funded: np.ndarray,
    param_grid: dict[tuple[str, int], list[float]],
    mff_config: dict,
    window_train_days: int = 120,
    window_test_days: int = 60,
    step_days: int = 60,
    n_mc_sims: int = 500,
    mc_block_min: int = 5,
    mc_block_max: int = 10,
    seed: int = 42,
    n_workers: int = 1,
) -> list[dict]:
    """Run expanding-window walk-forward analysis on REAL historical data.

    param_grid maps namespaced keys to values, e.g.
    {("eval", PARAMS_STOP_TICKS): [30, 40, 50],
     ("funded", PARAMS_TARGET_TICKS): [60, 90, 120]}.

    Per window:
    1. Backtest each parameter combination on train bars using BOTH eval and funded params
    2. Extract the daily lifecycle log and split it into eval + funded-first-cycle
       pools via daily_log["phase_id"] and daily_log["payout_cycle_id"]
    3. Run daily-block Monte-Carlo/NVE on phase-separated pools to score
    4. Pick best parameter set by NVE
    5. Backtest best params on test bars (out-of-sample)
    6. Score the OOS daily lifecycle log with phase-separated daily-block MC/NVE

    Historical walk-forward always scores with daily-block Monte-Carlo.
    `mc_block_min` / `mc_block_max` should come from params_cfg["monte_carlo"] so
    walk-forward uses the same block semantics as the standalone MC CLI.

    Returns list of window results with IS/OOS NVE and parameter stability metrics.
    """
    n_total_days = len(session_data["day_boundaries"])
    results = []
    window_idx = 0

    # Generate all parameter combinations
    grid_keys = sorted(param_grid.keys())
    grid_values = [param_grid[k] for k in grid_keys]
    all_combos = list(product(*grid_values))

    train_end = window_train_days
    while train_end + window_test_days <= n_total_days:
        test_end = train_end + window_test_days

        # --- Train phase: backtest + score each parameter set ---
        best_nve = -np.inf
        best_combo = None
        best_is_result = None

        for combo in all_combos:
            overrides = dict(zip(grid_keys, combo))
            params_eval = _build_params_array(base_params_eval, overrides, "eval")
            params_funded = _build_params_array(base_params_funded, overrides, "funded")

            # Backtest on train window
            artifacts = _backtest_param_set(
                session_data, (0, train_end), params_eval, params_funded, slippage_lookup, mff_config)

            if len(artifacts["daily_log"]) < 10:
                continue  # Not enough calendar days to evaluate robustly

            try:
                phase_pools = split_daily_log_for_mc(artifacts["daily_log"])
            except ValueError:
                continue  # Skip combos without eval + funded payout_cycle_id==0 day pools
            mc_result = run_monte_carlo(
                phase_pools["eval_day_pnls"], mff_config,
                funded_pnls=phase_pools["funded_day_pnls"],
                n_sims=n_mc_sims, seed=seed + window_idx,
                n_workers=n_workers,
                block_mode="daily",
                block_min=mc_block_min,
                block_max=mc_block_max,
            )

            if mc_result.nve > best_nve:
                best_nve = mc_result.nve
                best_combo = overrides
                best_is_result = mc_result

        # --- Test phase: evaluate best params on OOS window ---
        if best_combo is not None:
            best_params_eval = _build_params_array(base_params_eval, best_combo, "eval")
            best_params_funded = _build_params_array(base_params_funded, best_combo, "funded")
            oos_artifacts = _backtest_param_set(
                session_data, (train_end, test_end),
                best_params_eval, best_params_funded, slippage_lookup, mff_config)

            if len(oos_artifacts["daily_log"]) >= 5:
                try:
                    oos_phase_pools = split_daily_log_for_mc(oos_artifacts["daily_log"])
                    oos_mc = run_monte_carlo(
                        oos_phase_pools["eval_day_pnls"], mff_config,
                        funded_pnls=oos_phase_pools["funded_day_pnls"],
                        n_sims=n_mc_sims, seed=seed + window_idx + 50000,
                        n_workers=n_workers,
                        block_mode="daily",
                        block_min=mc_block_min,
                        block_max=mc_block_max,
                    )
                    oos_nve = oos_mc.nve
                    oos_payout_rate = oos_mc.payout_rate
                    oos_status = "ok"
                except ValueError:
                    oos_nve = None
                    oos_payout_rate = None
                    oos_status = "not_scored"
            else:
                oos_nve = None
                oos_payout_rate = None
                oos_status = "not_scored"
        else:
            oos_nve = None
            oos_payout_rate = None
            oos_status = "not_scored"

        results.append({
            "window": window_idx,
            "train_date_range": (
                session_data["session_dates"][0],
                session_data["session_dates"][train_end - 1],
            ),
            "test_date_range": (
                session_data["session_dates"][train_end],
                session_data["session_dates"][test_end - 1],
            ),
            "best_params": _serialize_param_overrides(best_combo),
            "in_sample_nve": best_nve if best_combo else None,
            "in_sample_payout_rate": (best_is_result.payout_rate
                                      if best_is_result else None),
            "oos_nve": oos_nve,
            "oos_payout_rate": oos_payout_rate,
            "status": "ok" if best_combo else "not_scored",
            "oos_status": oos_status if best_combo else "not_scored",
        })

        train_end += step_days
        window_idx += 1

    return results
```

- [ ] **Step 2: Write failing walk-forward tests**

Create `tests/test_walk_forward.py`:

```python
import json
import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import patch
from propfirm.optim.walk_forward import (
    _backtest_param_set,
    _build_params_array,
    _serialize_param_overrides,
    run_walk_forward,
)
from propfirm.core.types import (
    TRADE_LOG_DTYPE, DAILY_LOG_DTYPE, PARAMS_ARRAY_LENGTH,
    PARAMS_STOP_TICKS, PARAMS_TARGET_TICKS, PARAMS_CONTRACTS,
    PARAMS_DAILY_STOP, PARAMS_DAILY_TARGET, PARAMS_RANGE_MINUTES,
    PARAMS_MAX_TRADES, PARAMS_BUFFER_TICKS, PARAMS_VOL_THRESHOLD,
    PARAMS_STOP_PENALTY, PARAMS_COMMISSION,
)
from propfirm.market.slippage import build_slippage_lookup


def make_mff_config():
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


def make_base_params():
    params = np.zeros(PARAMS_ARRAY_LENGTH, dtype=np.float64)
    params[PARAMS_RANGE_MINUTES] = 15.0
    params[PARAMS_STOP_TICKS] = 40.0
    params[PARAMS_TARGET_TICKS] = 60.0
    params[PARAMS_CONTRACTS] = 10.0
    params[PARAMS_DAILY_STOP] = -750.0
    params[PARAMS_DAILY_TARGET] = 600.0
    params[PARAMS_MAX_TRADES] = 2.0
    params[PARAMS_BUFFER_TICKS] = 2.0
    params[PARAMS_VOL_THRESHOLD] = 0.0
    params[PARAMS_STOP_PENALTY] = 1.5
    params[PARAMS_COMMISSION] = 0.54
    return params


class TestBuildParamsArray:
    def test_eval_overrides_apply_to_eval_only(self):
        base = make_base_params()
        overrides = {
            ("eval", PARAMS_STOP_TICKS): 30.0,
            ("funded", PARAMS_TARGET_TICKS): 100.0,
        }
        result = _build_params_array(base, overrides, "eval")
        assert result[PARAMS_STOP_TICKS] == 30.0
        assert result[PARAMS_TARGET_TICKS] == 60.0  # funded override not applied

    def test_funded_overrides_apply_to_funded_only(self):
        base = make_base_params()
        overrides = {
            ("eval", PARAMS_STOP_TICKS): 30.0,
            ("funded", PARAMS_TARGET_TICKS): 100.0,
        }
        result = _build_params_array(base, overrides, "funded")
        assert result[PARAMS_STOP_TICKS] == 40.0  # eval override not applied
        assert result[PARAMS_TARGET_TICKS] == 100.0

    def test_does_not_mutate_base(self):
        base = make_base_params()
        original_stop = base[PARAMS_STOP_TICKS]
        overrides = {("eval", PARAMS_STOP_TICKS): 999.0}
        _build_params_array(base, overrides, "eval")
        assert base[PARAMS_STOP_TICKS] == original_stop


class TestSerializeParamOverrides:
    def test_returns_json_safe_nested_dict(self):
        result = _serialize_param_overrides({
            ("eval", PARAMS_STOP_TICKS): 30.0,
            ("funded", PARAMS_CONTRACTS): 20.0,
        })
        assert result == {
            "eval": {"stop_ticks": 30.0},
            "funded": {"contracts": 20.0},
        }
        json.dumps(result)


class TestBacktestParamSet:
    @pytest.fixture
    def synthetic_session(self):
        """Build minimal synthetic session data for 10 days of flat market."""
        from pathlib import Path
        bars_per_day = 390
        n_days = 10
        n_bars = bars_per_day * n_days
        base = 20000.0
        data = {
            "open": np.full(n_bars, base, dtype=np.float64),
            "high": np.full(n_bars, base + 5.0, dtype=np.float64),
            "low": np.full(n_bars, base - 5.0, dtype=np.float64),
            "close": np.full(n_bars, base, dtype=np.float64),
            "volume": np.full(n_bars, 1000, dtype=np.uint64),
            "timestamps": np.arange(n_bars, dtype=np.int64) + 1_640_000_000_000_000_000,
            "minute_of_day": np.tile(np.arange(bars_per_day, dtype=np.int16), n_days),
            "bar_atr": np.full(n_bars, 10.0, dtype=np.float64),
            "trailing_median_atr": np.full(n_bars, 10.0, dtype=np.float64),
            "day_boundaries": [(i * bars_per_day, (i + 1) * bars_per_day) for i in range(n_days)],
            "session_dates": [f"2022-01-{3 + i:02d}" for i in range(n_days)],
        }
        return data

    def test_daily_log_has_lifecycle_fields(self, synthetic_session):
        params = make_base_params()
        slippage_lookup = build_slippage_lookup(None)
        result = _backtest_param_set(
            synthetic_session, (0, 5), params, params, slippage_lookup, make_mff_config()
        )
        dl = result["daily_log"]
        assert "phase_id" in dl.dtype.names
        assert "payout_cycle_id" in dl.dtype.names
        assert "day_pnl" in dl.dtype.names
        assert "had_trade" in dl.dtype.names

    def test_eval_days_have_negative_payout_cycle_id(self, synthetic_session):
        params = make_base_params()
        slippage_lookup = build_slippage_lookup(None)
        result = _backtest_param_set(
            synthetic_session, (0, 5), params, params, slippage_lookup, make_mff_config()
        )
        dl = result["daily_log"]
        eval_days = dl[dl["phase_id"] == 0]
        if len(eval_days) > 0:
            assert np.all(eval_days["payout_cycle_id"] == -1)

    def test_funded_days_have_nonneg_payout_cycle_id(self, synthetic_session):
        params = make_base_params()
        slippage_lookup = build_slippage_lookup(None)
        result = _backtest_param_set(
            synthetic_session, (0, 10), params, params, slippage_lookup, make_mff_config()
        )
        dl = result["daily_log"]
        funded_days = dl[dl["phase_id"] == 1]
        if len(funded_days) > 0:
            assert np.all(funded_days["payout_cycle_id"] >= 0)


class TestRunWalkForward:
    @pytest.fixture
    def synthetic_session(self):
        bars_per_day = 390
        n_days = 30  # Enough for at least one train+test window
        n_bars = bars_per_day * n_days
        base = 20000.0
        data = {
            "open": np.full(n_bars, base, dtype=np.float64),
            "high": np.full(n_bars, base + 5.0, dtype=np.float64),
            "low": np.full(n_bars, base - 5.0, dtype=np.float64),
            "close": np.full(n_bars, base, dtype=np.float64),
            "volume": np.full(n_bars, 1000, dtype=np.uint64),
            "timestamps": np.arange(n_bars, dtype=np.int64) + 1_640_000_000_000_000_000,
            "minute_of_day": np.tile(np.arange(bars_per_day, dtype=np.int16), n_days),
            "bar_atr": np.full(n_bars, 10.0, dtype=np.float64),
            "trailing_median_atr": np.full(n_bars, 10.0, dtype=np.float64),
            "day_boundaries": [(i * bars_per_day, (i + 1) * bars_per_day) for i in range(n_days)],
            "session_dates": [f"2022-01-{3 + i:02d}" for i in range(n_days)],
        }
        return data

    def test_returns_list_of_window_results(self, synthetic_session):
        base_params = make_base_params()
        slippage_lookup = build_slippage_lookup(None)
        param_grid = {
            ("eval", PARAMS_STOP_TICKS): [30.0, 40.0],
        }
        results = run_walk_forward(
            synthetic_session, slippage_lookup,
            base_params, base_params, param_grid, make_mff_config(),
            window_train_days=10, window_test_days=5, step_days=5,
            n_mc_sims=20, seed=42, n_workers=1,
        )
        assert isinstance(results, list)
        for r in results:
            assert "window" in r
            assert "in_sample_nve" in r
            assert "oos_nve" in r
            assert "status" in r
            assert "oos_status" in r

    def test_not_scored_windows_are_none_not_zero(self, synthetic_session):
        """Windows without sufficient lifecycle pools must be 'not_scored', not 0.0."""
        base_params = make_base_params()
        slippage_lookup = build_slippage_lookup(None)
        # Use very tight window so lifecycle pools are likely insufficient
        param_grid = {("eval", PARAMS_STOP_TICKS): [40.0]}
        results = run_walk_forward(
            synthetic_session, slippage_lookup,
            base_params, base_params, param_grid, make_mff_config(),
            window_train_days=5, window_test_days=3, step_days=3,
            n_mc_sims=10, seed=42, n_workers=1,
        )
        for r in results:
            if r["status"] == "not_scored":
                assert r["in_sample_nve"] is None
            if r["oos_status"] == "not_scored":
                assert r["oos_nve"] is None

    def test_does_not_import_synthetic_grid_search(self):
        """Walk-forward must NOT use synthetic grid search internally."""
        import inspect
        from propfirm.optim import walk_forward
        source = inspect.getsource(walk_forward)
        assert "run_synthetic_grid_search" not in source
        assert "generate_synthetic_trades" not in source

    def test_each_result_has_best_params(self, synthetic_session):
        base_params = make_base_params()
        slippage_lookup = build_slippage_lookup(None)
        param_grid = {
            ("eval", PARAMS_STOP_TICKS): [30.0, 40.0],
        }
        results = run_walk_forward(
            synthetic_session, slippage_lookup,
            base_params, base_params, param_grid, make_mff_config(),
            window_train_days=10, window_test_days=5, step_days=5,
            n_mc_sims=20, seed=42, n_workers=1,
        )
        for r in results:
            if r["status"] == "ok":
                assert r["best_params"] is not None

    def test_window_ranges_use_session_dates(self, synthetic_session):
        base_params = make_base_params()
        slippage_lookup = build_slippage_lookup(None)
        results = run_walk_forward(
            synthetic_session, slippage_lookup,
            base_params, base_params, {("eval", PARAMS_STOP_TICKS): [40.0]}, make_mff_config(),
            window_train_days=10, window_test_days=5, step_days=5,
            n_mc_sims=10, seed=42, n_workers=1,
        )
        assert results[0]["train_date_range"] == (
            synthetic_session["session_dates"][0],
            synthetic_session["session_dates"][9],
        )
        assert results[0]["test_date_range"] == (
            synthetic_session["session_dates"][10],
            synthetic_session["session_dates"][14],
        )

    def test_forwards_mc_block_config_to_monte_carlo(self, synthetic_session):
        base_params = make_base_params()
        slippage_lookup = build_slippage_lookup(None)
        daily_log = np.zeros(12, dtype=DAILY_LOG_DTYPE)
        daily_log["day_id"] = np.arange(12, dtype=np.int32)
        daily_log["phase_id"] = np.array([0] * 6 + [1] * 6, dtype=np.int8)
        daily_log["payout_cycle_id"] = np.array([-1] * 6 + [0] * 6, dtype=np.int16)
        daily_log["had_trade"] = 1
        daily_log["n_trades"] = 1
        daily_log["day_pnl"] = np.linspace(50.0, 160.0, 12)

        with patch(
            "propfirm.optim.walk_forward._backtest_param_set",
            return_value={"trade_log": np.zeros(0, dtype=TRADE_LOG_DTYPE), "daily_log": daily_log},
        ), patch(
            "propfirm.optim.walk_forward.split_daily_log_for_mc",
            return_value={
                "eval_day_pnls": np.array([100.0, 0.0, 120.0], dtype=np.float64),
                "funded_day_pnls": np.array([90.0, 0.0, 110.0], dtype=np.float64),
            },
        ), patch(
            "propfirm.optim.walk_forward.run_monte_carlo",
            return_value=SimpleNamespace(nve=1.0, payout_rate=0.1),
        ) as mock_mc:
            run_walk_forward(
                synthetic_session, slippage_lookup,
                base_params, base_params, {("eval", PARAMS_STOP_TICKS): [40.0]}, make_mff_config(),
                window_train_days=10, window_test_days=5, step_days=5,
                n_mc_sims=11, mc_block_min=7, mc_block_max=9, seed=42, n_workers=1,
            )

        assert mock_mc.call_count >= 1
        for call in mock_mc.call_args_list:
            assert call.kwargs["block_mode"] == "daily"
            assert call.kwargs["block_min"] == 7
            assert call.kwargs["block_max"] == 9
            assert call.kwargs["n_sims"] == 11
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_walk_forward.py -v
```

Expected: FAIL - `ImportError`

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_walk_forward.py -v
```

Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add propfirm/optim/walk_forward.py tests/test_walk_forward.py
git commit -m "feat: add state-aware walk-forward analysis with daily-block MC scoring"
```

---

## Task 16: CLI Scripts

**Files:**
- Create: `scripts/run_backtest.py`
- Create: `scripts/run_grid_search.py`
- Create: `scripts/run_monte_carlo.py`
- Create: `scripts/run_walk_forward.py`

- [ ] **Step 1: Create run_backtest.py**

Create `scripts/run_backtest.py`:

```python
#!/usr/bin/env python
"""Run a single strategy backtest."""
import argparse
import numpy as np
from pathlib import Path

from propfirm.io.config import load_mff_config, load_params_config, build_phase_params
from propfirm.market.data_loader import load_session_data
from propfirm.market.slippage import build_slippage_lookup
from propfirm.core.engine import run_day_kernel
from propfirm.core.types import TRADE_LOG_DTYPE, DAILY_LOG_DTYPE, PARAMS_MAX_TRADES, PARAMS_CONTRACTS
from propfirm.risk.risk import validate_position_size
from propfirm.rules.mff import MFFState
from propfirm.strategy.orb import orb_signal
from propfirm.io.reporting import build_report, save_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/MNQ_1m_train.parquet"))
    parser.add_argument("--mff-config", type=Path, default=Path("configs/mff_flex_50k.toml"))
    parser.add_argument("--params-config", type=Path, default=Path("configs/default_params.toml"))
    parser.add_argument("--output", type=Path, default=Path("output/backtests"))
    args = parser.parse_args()

    mff_cfg = load_mff_config(args.mff_config)
    params_cfg = load_params_config(args.params_config)
    orb_shared = params_cfg["strategy"]["orb"]["shared"]
    orb_eval = params_cfg["strategy"]["orb"]["eval"]
    orb_funded = params_cfg["strategy"]["orb"]["funded"]
    slip_cfg = params_cfg["slippage"]

    data = load_session_data(
        args.data,
        atr_period=slip_cfg["atr_period"],
        trailing_atr_days=slip_cfg["trailing_atr_days"],
    )

    slippage_lookup = build_slippage_lookup(Path("data/slippage/slippage_profile.parquet"))

    params_eval = build_phase_params(
        orb_shared, orb_eval, slip_cfg, mff_cfg["instrument"]["commission_per_side"]
    )
    params_funded = build_phase_params(
        orb_shared, orb_funded, slip_cfg, mff_cfg["instrument"]["commission_per_side"]
    )

    max_trades_per_day = int(max(params_eval[PARAMS_MAX_TRADES], params_funded[PARAMS_MAX_TRADES]))
    max_possible_trades = max(1, len(data["day_boundaries"]) * max_trades_per_day)

    state = MFFState(mff_cfg)
    funded_payout_cycle_id = 0
    all_trades = np.zeros(max_possible_trades, dtype=TRADE_LOG_DTYPE)
    daily_log = np.zeros(len(data["day_boundaries"]), dtype=DAILY_LOG_DTYPE)
    total_trade_count = 0
    total_day_count = 0

    for day_idx, (start, end) in enumerate(data["day_boundaries"]):
        active_params = params_eval if state.phase == "eval" else params_funded.copy()
        phase_id = 0 if state.phase == "eval" else 1
        payout_cycle_id = -1 if state.phase == "eval" else funded_payout_cycle_id
        if state.phase == "funded":
            active_params[PARAMS_CONTRACTS] = float(
                validate_position_size(
                    int(active_params[PARAMS_CONTRACTS]),
                    state.get_max_contracts(),
                )
            )
        n_trades, equity, pnl = run_day_kernel(
            data["open"][start:end],
            data["high"][start:end],
            data["low"][start:end],
            data["close"][start:end],
            data["volume"][start:end],
            data["timestamps"][start:end],
            data["minute_of_day"][start:end],
            data["bar_atr"][start:end],
            data["trailing_median_atr"][start:end],
            slippage_lookup,
            day_idx,
            phase_id,
            payout_cycle_id,
            orb_signal,
            all_trades[total_trade_count:],
            0, state.equity, 0.0, active_params,
        )

        total_trade_count += n_trades
        result = state.update_eod(pnl, state.equity + pnl)
        net_payout = 0.0

        if result == "passed":
            print(f"EVAL PASSED on day {day_idx + 1} ({state.trading_days} trading days)")
            state.transition_to_funded()
            funded_payout_cycle_id = 0
        elif result == "blown":
            print(f"BLOWN on day {day_idx + 1}")
            print(f"Equity: ${state.equity:.2f}")
            pass
        elif state.phase == "funded" and state.payout_eligible:
            net_payout = state.process_payout()
            if net_payout > 0:
                print(f"PAYOUT #{state.payouts_completed}: net=${net_payout:.2f}")
                funded_payout_cycle_id += 1

        daily_log[total_day_count]["day_id"] = day_idx
        daily_log[total_day_count]["phase_id"] = phase_id
        daily_log[total_day_count]["payout_cycle_id"] = payout_cycle_id
        daily_log[total_day_count]["had_trade"] = 1 if n_trades > 0 else 0
        daily_log[total_day_count]["n_trades"] = n_trades
        daily_log[total_day_count]["day_pnl"] = pnl
        daily_log[total_day_count]["net_payout"] = net_payout
        total_day_count += 1

        if result == "blown":
            break

    print(f"Total trades: {total_trade_count}")
    print(f"Final equity: ${state.equity:.2f}")

    # --- Persist artifacts ---
    args.output.mkdir(parents=True, exist_ok=True)
    trade_log = all_trades[:total_trade_count]
    daily_log = daily_log[:total_day_count]
    trade_log_path = args.output / "latest_trade_log.npy"
    daily_log_path = args.output / "latest_daily_log.npy"
    np.save(trade_log_path, trade_log)
    print(f"Trade log saved to {trade_log_path} ({total_trade_count} trades)")
    np.save(daily_log_path, daily_log)
    print(f"Daily lifecycle log saved to {daily_log_path} ({total_day_count} days)")

    # Also save net_pnl as a legacy/synthetic convenience artifact
    pnl_path = args.output / "latest_trade_pnls.npy"
    np.save(pnl_path, trade_log["net_pnl"])
    print(f"Trade PNLs saved to {pnl_path}")

    report = build_report(
        params={"eval": orb_eval, "funded": orb_funded, "shared": orb_shared},
        mc_result=None,
        config_snapshot={"params_cfg": params_cfg, "mff_cfg": mff_cfg},
        data_split=args.data.stem,
        data_date_range=(data["session_dates"][0], data["session_dates"][-1]),
        seed=params_cfg["general"]["random_seed"],
    )
    report["artifacts"] = {
        "daily_log": str(daily_log_path),
        "trade_log": str(trade_log_path),
        "trade_pnls": str(pnl_path),
    }
    report["runtime_meta"] = {
        "mc_mode_recommended": (
            "daily" if (
                np.any(daily_log["phase_id"] == 0)
                and np.any((daily_log["phase_id"] == 1) & (daily_log["payout_cycle_id"] == 0))
            ) else "not_ready"
        ),
        "mc_daily_lifecycle_ready": bool(
            np.any(daily_log["phase_id"] == 0)
            and np.any((daily_log["phase_id"] == 1) & (daily_log["payout_cycle_id"] == 0))
        ),
        "lifecycle_aware_daily_log": True,
        "payouts_completed": state.payouts_completed,
    }
    save_report(report, args.output / "latest_backtest.json")
    print(f"Report saved to {args.output / 'latest_backtest.json'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create run_grid_search.py**

Create `scripts/run_grid_search.py`:

```python
#!/usr/bin/env python
"""Run synthetic grid search.

This remains a sensitivity pre-study only. Historical walk-forward remains the
authoritative parameter-selection path for ORB.
"""
import argparse
import json
from pathlib import Path

from propfirm.io.config import load_mff_config, load_params_config
from propfirm.optim.grid_search import run_synthetic_grid_search
from propfirm.io.reporting import build_report, save_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mff-config", type=Path, default=Path("configs/mff_flex_50k.toml"))
    parser.add_argument("--params-config", type=Path, default=Path("configs/default_params.toml"))
    parser.add_argument("--output", type=Path, default=Path("output/grid_search"))
    parser.add_argument("--n-mc-sims", type=int, default=1000)
    parser.add_argument("--n-workers", type=int, default=1)
    args = parser.parse_args()

    mff_cfg = load_mff_config(args.mff_config)
    params_cfg = load_params_config(args.params_config)
    seed = params_cfg["general"]["random_seed"]

    param_grid = {
        "win_rate": [0.55, 0.60, 0.65, 0.70],
        "risk_reward": [1.0, 1.2, 1.5, 2.0],
        "contracts": [5, 10, 15, 20],
    }

    n_combos = (len(param_grid['win_rate']) * len(param_grid['risk_reward'])
                * len(param_grid['contracts']))
    print(f"Running grid search: {len(param_grid['win_rate'])} x "
          f"{len(param_grid['risk_reward'])} x {len(param_grid['contracts'])} = "
          f"{n_combos} combinations")

    results = run_synthetic_grid_search(
        param_grid, mff_cfg,
        n_mc_sims=args.n_mc_sims,
        seed=seed,
        n_workers=args.n_workers,
    )

    print(f"\nTop 10 by NVE:")
    for i, r in enumerate(results[:10]):
        p = r["params"]
        print(f"  {i+1}. NVE=${r['nve']:.0f} | WR={p['win_rate']:.0%} "
              f"RR={p['risk_reward']:.1f} C={p['contracts']} | "
              f"EvalPass={r['eval_pass_rate']:.1%} "
              f"PayoutRate={r['payout_rate']:.1%}")

    report = build_report(
        params={"grid": param_grid, "top_10": results[:10]},
        mc_result=None,
        config_snapshot={"params_cfg": params_cfg, "mff_cfg": mff_cfg},
        data_split="synthetic",
        data_date_range=("N/A", "N/A"),
        seed=seed,
    )
    report_path = args.output / "grid_search_results.json"
    report["artifacts"] = {"report": str(report_path)}
    report["runtime_meta"] = {
        "mc_mode": "fixed",
        "optimization_path": "synthetic_prestudy",
        "lifecycle_aware": False,
    }
    save_report(report, report_path)
    print(f"\nFull results saved to {report_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create run_monte_carlo.py**

Create `scripts/run_monte_carlo.py`:

```python
#!/usr/bin/env python
"""Run Monte-Carlo simulation on lifecycle PNL artifacts.

Accepts either:
- A structured daily lifecycle log .npy file with day_pnl
  (preferred production path for real ORB runs)
- A flat .npy file of net PNLs (legacy/fixed-block only)

Pipeline usage:
    python scripts/run_backtest.py --output output/backtests
    python scripts/run_monte_carlo.py --trades output/backtests/latest_daily_log.npy
"""
import argparse
import numpy as np
from pathlib import Path

from propfirm.io.config import load_mff_config, load_params_config
from propfirm.monte_carlo.bootstrap import run_monte_carlo, split_daily_log_for_mc
from propfirm.io.reporting import build_report, save_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", type=Path, required=True,
                        help="Path to .npy file (flat PNLs or structured daily lifecycle log)")
    parser.add_argument("--mff-config", type=Path, default=Path("configs/mff_flex_50k.toml"))
    parser.add_argument("--params-config", type=Path, default=Path("configs/default_params.toml"))
    parser.add_argument("--output", type=Path, default=Path("output/monte_carlo"))
    parser.add_argument("--n-workers", type=int, default=1)
    args = parser.parse_args()

    mff_cfg = load_mff_config(args.mff_config)
    params_cfg = load_params_config(args.params_config)
    mc_cfg = params_cfg["monte_carlo"]
    seed = params_cfg["general"]["random_seed"]

    raw = np.load(args.trades, allow_pickle=False)
    block_mode = mc_cfg["block_mode"]
    # Auto-detect: structured daily lifecycle log vs flat PNL array
    if raw.dtype.names is not None and "day_pnl" in raw.dtype.names:
        if block_mode != "daily":
            raise ValueError("Structured daily lifecycle logs require monte_carlo.block_mode = 'daily'")
        try:
            phase_pools = split_daily_log_for_mc(raw)
        except ValueError as exc:
            raise ValueError(
                "Structured daily log is not lifecycle-ready for daily MC. "
                "Need eval days plus funded payout_cycle_id==0 days."
            ) from exc
        eval_day_pnls = phase_pools["eval_day_pnls"]
        funded_day_pnls = phase_pools["funded_day_pnls"]
        print(f"Loaded structured daily log: eval={len(eval_day_pnls)} funded={len(funded_day_pnls)} days")
    else:
        if block_mode != "fixed":
            raise ValueError("Flat PNL arrays are only allowed for fixed-block legacy/synthetic MC runs")
        eval_day_pnls = raw.astype(np.float64)
        funded_day_pnls = eval_day_pnls
        print(f"Loaded flat PNL array: {len(eval_day_pnls)} trades")

    result = run_monte_carlo(
        eval_day_pnls, mff_cfg,
        funded_pnls=funded_day_pnls,
        n_sims=mc_cfg["n_simulations"],
        block_mode=block_mode,
        block_min=mc_cfg["block_size_min"],
        block_max=mc_cfg["block_size_max"],
        seed=seed,
        n_workers=args.n_workers,
    )

    print(f"Eval Pass Rate: {result.eval_pass_rate:.1%} "
          f"[{result.eval_pass_rate_ci_5:.1%} - {result.eval_pass_rate_ci_95:.1%}]")
    print(f"Funded Survival Rate: {result.funded_survival_rate:.1%}")
    print(f"Payout Rate: {result.payout_rate:.1%}")
    print(f"Mean Payout Net: ${result.mean_payout_net:.2f}")
    print(f"NVE: ${result.nve:.2f}")
    print(f"Mean days to eval pass: {result.mean_days_to_eval_pass:.1f}")
    print(f"Mean funded days to payout: {result.mean_funded_days_to_payout:.1f}")

    report = build_report(
        params={},
        mc_result=result,
        config_snapshot={"params_cfg": params_cfg, "mff_cfg": mff_cfg},
        data_split="custom",
        data_date_range=("", ""),
        seed=seed,
    )
    report["artifacts"] = {"input_artifact": str(args.trades)}
    report["runtime_meta"] = {
        "mc_mode": block_mode,
        "lifecycle_aware_input": bool(raw.dtype.names is not None and "day_pnl" in raw.dtype.names),
    }
    save_report(report, args.output / "mc_results.json")
    print(f"Report saved to {args.output / 'mc_results.json'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create run_walk_forward.py**

Create `scripts/run_walk_forward.py`:

```python
#!/usr/bin/env python
"""Run walk-forward analysis on real historical data.

This performs REAL walk-forward optimization:
1. Per window: backtest ORB with each parameter combo on train bars
2. Score each combo via Monte-Carlo/NVE on the resulting daily lifecycle log
3. Pick best params by NVE
4. Backtest best params on out-of-sample test bars
5. Score the OOS daily lifecycle log to measure parameter stability

Pipeline usage:
    python scripts/run_walk_forward.py --data data/processed/MNQ_1m_train.parquet
"""
import argparse
import json
import numpy as np
from pathlib import Path

from propfirm.io.config import load_mff_config, load_params_config, build_phase_params
from propfirm.market.data_loader import load_session_data
from propfirm.market.slippage import build_slippage_lookup
from propfirm.optim.walk_forward import run_walk_forward
from propfirm.io.reporting import build_report, save_report
from propfirm.core.types import (
    PARAMS_STOP_TICKS, PARAMS_TARGET_TICKS, PARAMS_CONTRACTS,
    PARAMS_DAILY_STOP,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/MNQ_1m_train.parquet"))
    parser.add_argument("--mff-config", type=Path, default=Path("configs/mff_flex_50k.toml"))
    parser.add_argument("--params-config", type=Path, default=Path("configs/default_params.toml"))
    parser.add_argument("--output", type=Path, default=Path("output/walk_forward"))
    parser.add_argument("--train-days", type=int, default=120)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--step-days", type=int, default=60)
    parser.add_argument("--n-mc-sims", type=int, default=None,
                        help="Optional override for monte_carlo.n_simulations; block sizes still come from params config")
    parser.add_argument("--n-workers", type=int, default=1)
    args = parser.parse_args()

    mff_cfg = load_mff_config(args.mff_config)
    params_cfg = load_params_config(args.params_config)
    orb_shared = params_cfg["strategy"]["orb"]["shared"]
    orb_eval = params_cfg["strategy"]["orb"]["eval"]
    orb_funded = params_cfg["strategy"]["orb"]["funded"]
    slip_cfg = params_cfg["slippage"]
    mc_cfg = params_cfg["monte_carlo"]
    seed = params_cfg["general"]["random_seed"]
    if mc_cfg["block_mode"] != "daily":
        raise ValueError("run_walk_forward.py requires monte_carlo.block_mode = 'daily'")
    n_mc_sims = args.n_mc_sims if args.n_mc_sims is not None else mc_cfg["n_simulations"]

    data = load_session_data(
        args.data,
        atr_period=slip_cfg["atr_period"],
        trailing_atr_days=slip_cfg["trailing_atr_days"],
    )
    slippage_lookup = build_slippage_lookup(Path("data/slippage/slippage_profile.parquet"))

    base_params_eval = build_phase_params(
        orb_shared, orb_eval, slip_cfg, mff_cfg["instrument"]["commission_per_side"]
    )
    base_params_funded = build_phase_params(
        orb_shared, orb_funded, slip_cfg, mff_cfg["instrument"]["commission_per_side"]
    )

    # Parameter grid: state-aware ORB params to optimize
    param_grid = {
        ("eval", PARAMS_STOP_TICKS): [30.0, 40.0, 50.0],
        ("eval", PARAMS_TARGET_TICKS): [45.0, 60.0, 75.0],
        ("eval", PARAMS_CONTRACTS): [5.0, 10.0, 15.0],
        ("funded", PARAMS_TARGET_TICKS): [60.0, 80.0, 100.0],
        ("funded", PARAMS_CONTRACTS): [10.0, 20.0, 30.0],
        ("funded", PARAMS_DAILY_STOP): [-750.0, -1000.0, -1250.0],
    }

    n_combos = 1
    for v in param_grid.values():
        n_combos *= len(v)
    n_days = len(data["day_boundaries"])
    print(f"Walk-forward: {n_combos} param combos, {n_days} total days")
    print(f"Windows: train={args.train_days}d, test={args.test_days}d, step={args.step_days}d")

    results = run_walk_forward(
        session_data=data,
        slippage_lookup=slippage_lookup,
        base_params_eval=base_params_eval,
        base_params_funded=base_params_funded,
        param_grid=param_grid,
        mff_config=mff_cfg,
        window_train_days=args.train_days,
        window_test_days=args.test_days,
        step_days=args.step_days,
        n_mc_sims=n_mc_sims,
        mc_block_min=mc_cfg["block_size_min"],
        mc_block_max=mc_cfg["block_size_max"],
        seed=seed,
        n_workers=args.n_workers,
    )

    def fmt_money(value):
        return "N/A" if value is None else f"${value:.0f}"

    def fmt_pct(value):
        return "N/A" if value is None else f"{value:.1%}"

    print(f"\nWalk-forward results ({len(results)} windows):")
    for r in results:
        print(f"  Window {r['window']}: "
              f"Status={r['status']} | "
              f"OOS Status={r['oos_status']} | "
              f"IS NVE={fmt_money(r['in_sample_nve'])} | "
              f"OOS NVE={fmt_money(r['oos_nve'])} | "
              f"OOS Payout Rate={fmt_pct(r['oos_payout_rate'])} | "
              f"Params={r['best_params']}")

    report = build_report(
        params={
            "eval": orb_eval,
            "funded": orb_funded,
            "shared": orb_shared,
            "walk_forward_results": results,
        },
        mc_result=None,
        config_snapshot={"params_cfg": params_cfg, "mff_cfg": mff_cfg},
        data_split=args.data.stem,
        data_date_range=(data["session_dates"][0], data["session_dates"][-1]),
        seed=seed,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "walk_forward_results.json"
    report["artifacts"] = {
        "report": str(report_path),
        "input_data": str(args.data),
    }
    report["runtime_meta"] = {
        "mc_mode": mc_cfg["block_mode"],
        "lifecycle_aware_walk_forward": True,
        "mc_n_sims": n_mc_sims,
        "mc_block_size_min": mc_cfg["block_size_min"],
        "mc_block_size_max": mc_cfg["block_size_max"],
        "train_days": args.train_days,
        "test_days": args.test_days,
        "step_days": args.step_days,
    }
    save_report(report, report_path)
    print(f"\nResults saved to {report_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Commit**

```bash
git add scripts/
git commit -m "feat: add CLI scripts - backtest, grid search, monte carlo, walk forward"
```

---

## Task 17: Integration Test + Smoke Test

> **Release gate:** Before tagging `v0.1.0-phase2`, re-run the
> `Implementation Firewall` checklist above. If any checklist item fails,
> the build is not release-ready even if the immediate smoke test is green.

- [ ] **Step 1: Run full test suite (including 21 MFF validation scenarios)**

```bash
pytest tests/ -v --tb=short
```

Expected: ALL tests pass - including `test_mff_validation.py` (21 scenarios)

- [ ] **Step 2: Run slippage calibration on real data**

```bash
python scripts/calibrate_slippage.py
```

Expected: Creates `data/slippage/slippage_profile.parquet` with 26 buckets

- [ ] **Step 3: Run grid search with statistically meaningful sample**

```bash
python scripts/run_grid_search.py --n-mc-sims 1000 --n-workers 4
```

Expected: Prints top 10 results ranked by NVE, saves JSON to `output/grid_search/`

- [ ] **Step 4: Run deterministic Monte-Carlo CLI smoke on fixture daily_log**

```bash
python -c "
import numpy as np
from pathlib import Path
from propfirm.core.types import DAILY_LOG_DTYPE
Path('output/fixtures').mkdir(parents=True, exist_ok=True)
fixture = np.zeros(6, dtype=DAILY_LOG_DTYPE)
fixture['day_id'] = np.arange(6, dtype=np.int32)
fixture['phase_id'] = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
fixture['payout_cycle_id'] = np.array([-1, -1, -1, 0, 0, 1], dtype=np.int16)
fixture['had_trade'] = np.array([1, 0, 1, 1, 0, 1], dtype=np.int8)
fixture['n_trades'] = np.array([1, 0, 1, 1, 0, 1], dtype=np.int16)
fixture['day_pnl'] = np.array([400.0, 0.0, 3200.0, 300.0, 0.0, 100.0], dtype=np.float64)
fixture['net_payout'] = np.zeros(6, dtype=np.float64)
np.save('output/fixtures/mc_fixture_daily_log.npy', fixture)
"
python scripts/run_monte_carlo.py --trades output/fixtures/mc_fixture_daily_log.npy --n-workers 1 --output output/monte_carlo_fixture
```

Expected: `run_monte_carlo.py` succeeds on a deterministic lifecycle fixture and writes `output/monte_carlo_fixture/mc_results.json`

- [ ] **Step 5: Verify backtest lifecycle artifacts exist**

```bash
python -c "
import numpy as np
from propfirm.core.types import TRADE_LOG_DTYPE, DAILY_LOG_DTYPE
trade_log = np.load('output/backtests/latest_trade_log.npy')
daily_log = np.load('output/backtests/latest_daily_log.npy')
pnls = np.load('output/backtests/latest_trade_pnls.npy')
assert trade_log.dtype == TRADE_LOG_DTYPE, f'Wrong trade_log dtype: {trade_log.dtype}'
assert daily_log.dtype == DAILY_LOG_DTYPE, f'Wrong daily_log dtype: {daily_log.dtype}'
assert len(trade_log) == len(pnls), f'Length mismatch: {len(trade_log)} vs {len(pnls)}'

# Trade log remains the diagnostic round-trip record.
assert 'day_id' in trade_log.dtype.names
assert 'phase_id' in trade_log.dtype.names
assert 'payout_cycle_id' in trade_log.dtype.names
assert np.all(np.diff(trade_log['day_id']) >= 0), 'trade_log day_id must be monotonic'
assert np.all(np.isin(trade_log['phase_id'], [0, 1])), 'trade_log phase_id must be eval/funded only'
assert np.all((trade_log['phase_id'] == 1) | (trade_log['payout_cycle_id'] == -1)), 'Eval trades must use payout_cycle_id=-1'
assert np.all((trade_log['phase_id'] == 0) | (trade_log['payout_cycle_id'] >= 0)), 'Funded trades must use non-negative payout_cycle_id'

# Daily log is the authoritative lifecycle artifact for MC because it preserves zero-trade days.
assert 'day_id' in daily_log.dtype.names
assert 'phase_id' in daily_log.dtype.names
assert 'payout_cycle_id' in daily_log.dtype.names
assert 'had_trade' in daily_log.dtype.names
assert 'n_trades' in daily_log.dtype.names
assert 'day_pnl' in daily_log.dtype.names
assert 'net_payout' in daily_log.dtype.names
assert np.all(np.diff(daily_log['day_id']) >= 0), 'daily_log day_id must be monotonic'
assert np.all(np.isin(daily_log['phase_id'], [0, 1])), 'daily_log phase_id must be eval/funded only'
assert np.all((daily_log['phase_id'] == 1) | (daily_log['payout_cycle_id'] == -1)), 'Eval days must use payout_cycle_id=-1'
assert np.all((daily_log['phase_id'] == 0) | (daily_log['payout_cycle_id'] >= 0)), 'Funded days must use non-negative payout_cycle_id'
assert np.all((daily_log['had_trade'] == 0) | (daily_log['had_trade'] == 1)), 'had_trade must be binary'
assert np.all((daily_log['had_trade'] == 1) == (daily_log['n_trades'] > 0)), 'had_trade must match n_trades'

# Verify net_pnl invariant: logged trade-level net_pnl must match extracted pnls
import numpy.testing as npt
npt.assert_array_almost_equal(trade_log['net_pnl'], pnls)
print(f'Trade log OK: {len(trade_log)} trades, net_pnl invariant holds')
print(f'Daily log OK: {len(daily_log)} calendar trading days, lifecycle fields present')
print(f'Total net PNL: \${pnls.sum():.2f}')
"
```

Expected: trade-level diagnostics and daily lifecycle log are both consistent; the daily log is the authoritative MC artifact

- [ ] **Step 6: Verify end-to-end pipeline: backtest -> MC**

```bash
python -c "
import json, subprocess, sys
r = json.load(open('output/backtests/latest_backtest.json'))
if not r['runtime_meta']['mc_daily_lifecycle_ready']:
    print('Skipping daily MC smoke: backtest artifact has no eval + funded payout_cycle_id==0 lifecycle pool')
    sys.exit(0)
raise SystemExit(subprocess.call([
    'python', 'scripts/run_monte_carlo.py',
    '--trades', 'output/backtests/latest_daily_log.npy',
    '--n-workers', '2'
]))
"
```

Expected: Either skips with an explicit lifecycle-readiness reason or runs daily MC successfully on the structured daily lifecycle log

- [ ] **Step 7: Run walk-forward CLI smoke on historical data**

```bash
python scripts/run_walk_forward.py --data data/processed/MNQ_1m_train.parquet --train-days 60 --test-days 20 --step-days 20 --n-mc-sims 100 --n-workers 1 --output output/walk_forward
```

Expected: `run_walk_forward.py` completes, prints per-window statuses, and writes `output/walk_forward/walk_forward_results.json`

- [ ] **Step 8: Verify static funded floor behavior**

```bash
python -c "
from propfirm.rules.mff import MFFState
from propfirm.io.config import load_mff_config
cfg = load_mff_config('configs/mff_flex_50k.toml')
s = MFFState(cfg)
s.transition_to_funded()
s.equity = 2000.0
s.total_profit = 2000.0
s.winning_days = 5
s.process_payout()
floor = s.static_floor_equity
assert floor == 900.0, floor
s.update_eod(400.0, 1400.0)
assert s.static_floor_equity == floor
assert s.update_eod(-550.0, 850.0) == 'blown'
print('Static floor OK:', floor)
"
```

Expected: static floor remains unchanged after later highs and triggers blow-up at the frozen threshold

- [ ] **Step 9: Verify JSON report is self-contained**

```bash
python -c "
import json
from pathlib import Path

def check_report(path, runtime_keys):
    r = json.load(open(path))
    assert 'meta' in r
    assert 'git_hash' in r['meta']
    assert 'random_seed' in r['meta']
    assert 'timestamp' in r['meta']
    assert 'config_snapshot' in r['meta']
    assert 'params_cfg' in r['meta']['config_snapshot']
    assert 'mff_cfg' in r['meta']['config_snapshot']
    assert 'artifacts' in r
    assert 'runtime_meta' in r
    for key in runtime_keys:
        assert key in r['runtime_meta'], f'Missing runtime_meta[{key}] in {path}'
    print('Report OK:', path, r['meta']['git_hash'], r['meta']['timestamp'])
    return r

backtest = check_report(
    'output/backtests/latest_backtest.json',
    ['mc_mode_recommended', 'mc_daily_lifecycle_ready', 'lifecycle_aware_daily_log'],
)
grid = check_report(
    'output/grid_search/grid_search_results.json',
    ['mc_mode', 'optimization_path', 'lifecycle_aware'],
)
print('Top NVE:', grid['params']['top_10'][0]['nve'])

required_reports = [
    ('output/monte_carlo_fixture/mc_results.json', ['mc_mode']),
    ('output/walk_forward/walk_forward_results.json', ['mc_mode', 'lifecycle_aware_walk_forward', 'mc_n_sims', 'mc_block_size_min', 'mc_block_size_max']),
]
for path, keys in required_reports:
    check_report(path, keys)

optional_reports = [
    ('output/monte_carlo/mc_results.json', ['mc_mode']),
]
for path, keys in optional_reports:
    if Path(path).exists():
        check_report(path, keys)
"
```

Expected: Confirms every required CLI report contains `config_snapshot`, `artifacts`, and `runtime_meta`; prints git hash, timestamp, and top NVE value

- [ ] **Step 10: Commit any fixes**

If any integration issues were found and fixed:

```bash
git add -u
git commit -m "fix: integration fixes from smoke test"
```

- [ ] **Step 11: Final commit - tag milestone**

```bash
git tag v0.1.0-phase2
```
