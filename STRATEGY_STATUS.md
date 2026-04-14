# Strategy Status

Stand: `2026-04-14`

## Aktuelle Referenz

Die aktuelle Python-Referenz fuer die wiederhergestellte alte profitable MNQ-Strategie ist:
- [`scripts/run_mnq_legacy_frozen.py`](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/scripts/run_mnq_legacy_frozen.py)
- [`configs/default_params_mnq_legacy_frozen.toml`](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/configs/default_params_mnq_legacy_frozen.toml)
- Output: [`output/backtests_mnq_legacy_frozen`](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/output/backtests_mnq_legacy_frozen)

Das ist der einzig explizit als `exact legacy match` verifizierte MNQ-Pfad.

## Aktueller Portfolio-Pfad

Der aktuelle Portfolio-/Validierungs-Workflow ist:
- [`scripts/run_clean_portfolio_validation.py`](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/scripts/run_clean_portfolio_validation.py)
- MGC-Config: [`configs/default_params.toml`](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/configs/default_params.toml)
- MNQ-Portfolio-Default: [`configs/default_params_mnq.toml`](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/configs/default_params_mnq.toml)
- Canonical Output: [`output/canonical_portfolio_validation`](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/output/canonical_portfolio_validation)

Wichtig:
- `default_params_mnq.toml` ist die aktuelle profitable Legacy-Parametrisierung fuer den Portfolio-Pfad.
- Der exakte alte MNQ-Run wird trotzdem nur ueber `run_mnq_legacy_frozen.py` reproduziert.

## TradingView

Aktuelle Legacy-Zielversion fuer TradingView:
- [`tradingview/PropFirmBreakoutStrategy_MNQ_Legacy.pine`](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/tradingview/PropFirmBreakoutStrategy_MNQ_Legacy.pine)

Nicht als Source of Truth behandeln:
- [`tradingview/archive/2026-04-14_noncurrent_variants/PropFirmBreakoutStrategy_MNQ.pine`](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/tradingview/archive/2026-04-14_noncurrent_variants/PropFirmBreakoutStrategy_MNQ.pine)
- [`tradingview/archive/2026-04-14_noncurrent_variants/PropFirmBreakoutStrategy_Production.pine`](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/tradingview/archive/2026-04-14_noncurrent_variants/PropFirmBreakoutStrategy_Production.pine)

## Aktiv vs. Archiviert

Aktiv:
- `configs/default_params.toml`
- `configs/default_params_mnq.toml`
- `configs/default_params_mnq_legacy_frozen.toml`
- `configs/mff_flex_50k.toml`
- `configs/mff_flex_50k_mgc.toml`
- `configs/mff_flex_50k_mnq.toml`
- `scripts/run_mnq_legacy_frozen.py`
- `scripts/run_clean_portfolio_validation.py`
- `scripts/final_portfolio_backtest.py`
- `scripts/asymmetrical_sizing_matrix.py`
- `scripts/monte_carlo_block_slippage_stress.py`

Archiviert:
- alte/experimentelle Skripte unter [`scripts/archive`](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/scripts/archive)
- nicht aktuelle Config-Varianten unter [`configs/archive`](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/configs/archive)
- nicht aktuelle Strategy-Tests unter [`tests/archive`](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/tests/archive)
- alte Outputs und Temp-Artefakte unter [`output/archive`](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/output/archive)

## Nicht aktuelle Strategielinien

Aktuell nicht Referenz und nur noch historisch/experimentell:
- TV-aligned MNQ-Variante
- MCL-Portfolio-Pfad
- M6A Fade
- KAMA/MACD
- London Fade
- VWAP Pullback / VWAP POC Breakout
- Renko / Tick-Replay / Macro ORB Nebenpfade

## Arbeitsregel

Wenn unklar ist, welcher Stand gilt:
1. zuerst [`STRATEGY_STATUS.md`](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/STRATEGY_STATUS.md) lesen
2. dann nur mit den dort genannten aktiven Configs/Runners arbeiten
3. alles andere als Historie oder Experiment behandeln, bis es bewusst wieder aktiviert wird
