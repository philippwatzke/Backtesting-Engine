# Prop-Firm Backtesting Engine

Backtesting- und Research-Engine fuer intraday Futures-Strategien unter MFF-aehnlichen Prop-Firm-Regeln.

Aktueller Navigationspunkt:
- Siehe [STRATEGY_STATUS.md](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/STRATEGY_STATUS.md) fuer die aktuelle Referenzstrategie, aktive Runner und archivierte Varianten.

Der Schwerpunkt liegt auf:
- lifecycle-aware Backtests mit Eval- und Funded-Phase
- strukturierten Trade- und Daily-Logs
- Monte-Carlo auf Daily-Lifecycle-Artefakten
- Walk-Forward-Auswertung auf echten historischen Sessions

## Setup

Voraussetzungen:
- Python 3.10+
- lokale Parquet-Daten unter `data/processed/`

Installation:

```bash
python -m pip install -e .[dev]
```

Tests:

```bash
python -m pytest tests -q
```

## Daten

Erwartete Inputs:
- `data/processed/MNQ_1m_train.parquet`
- optional weitere Splits wie `MNQ_1m_validation.parquet` und `MNQ_1m_test.parquet`

Die Daten muessen:
- einen timezone-aware `DatetimeIndex` haben
- `open`, `high`, `low`, `close`, `volume` enthalten
- auf RTH `09:30-15:59 America/New_York` normalisierbar sein

Unvollstaendige oder gappige RTH-Sessions werden beim Laden aus dem Backtest-Universum entfernt.

## Standard-Workflow

### 1. Slippage kalibrieren

Historische Runs erwarten ein kalibriertes Slippage-Profil:

```bash
python scripts/calibrate_slippage.py
```

Erzeugt:
- `data/slippage/slippage_profile.parquet`

Hinweis:
- Das aktuelle Modell kalibriert `baseline_ticks` als robuste Funktion aus unteren 1-Minuten-Range-Quantilen.
- Ohne Bid/Ask-Daten bleibt Slippage heuristisch. Das Profil ist deshalb ein Research-Input, kein Marktgrundgesetz.

### 2. Einzelnen Backtest ausfuehren

```bash
python scripts/run_backtest.py --data data/processed/MNQ_1m_train.parquet --output output/backtests
```

Erzeugt:
- `output/backtests/latest_trade_log.npy`
- `output/backtests/latest_daily_log.npy`
- `output/backtests/latest_trade_pnls.npy`
- `output/backtests/latest_backtest.json`

Wichtige Pruefung direkt danach:
- `runtime_meta.mc_daily_lifecycle_ready` in `latest_backtest.json`

Wenn dieser Wert `false` ist:
- hat der Backtest keine ausreichenden Eval- plus funded-`payout_cycle_id == 0`-Tage erzeugt
- dann sind echte Daily-Monte-Carlo- und Walk-Forward-Scores noch nicht auswertbar

### 3. Monte-Carlo aus Daily-Log

Nur sinnvoll, wenn das Daily-Log lifecycle-ready ist:

```bash
python scripts/run_monte_carlo.py --trades output/backtests/latest_daily_log.npy --output output/monte_carlo
```

Erzeugt:
- `output/monte_carlo/mc_results.json`

### 4. Time Sweep

```bash
python scripts/sweep_h1_times.py --data data/processed/MGC_1m_full_train.parquet
```

Erzeugt:
- eine tabellarische H1-Startzeit-Sensitivitaet fuer den aktiven MGC-Trendpfad

## Wichtige Artefakte

### Trade Log

`latest_trade_log.npy` enthaelt pro Trade unter anderem:
- `phase_id`
- `payout_cycle_id`
- `entry_price`, `exit_price`
- `entry_slippage`, `exit_slippage`
- `gross_pnl`, `net_pnl`

### Daily Log

`latest_daily_log.npy` enthaelt pro Session:
- `phase_id`
- `payout_cycle_id`
- `had_trade`
- `n_trades`
- `day_pnl`
- `net_payout`

Semantik:
- `phase_id == 0`: Eval
- `phase_id == 1`: Funded
- `payout_cycle_id == -1`: Eval
- `payout_cycle_id == 0`: Funded vor dem ersten Payout
- `payout_cycle_id >= 1`: spaetere Funded-Payout-Zyklen

## Konfiguration

Wichtige Dateien:
- `configs/default_params.toml`
- `configs/mff_flex_50k.toml`

Empfehlung fuer Research:
- diese Dateien als Baseline einfrieren
- Aenderungen bewusst versionieren
- Backtests immer gegen klar benannte Parameterstaende vergleichen

## Typische Stolperfallen

### Fehlendes Slippage-Profil bei Legacy-Research

Einige aeltere Research-Skripte verlangen ein vorhandenes Profil unter:
- `data/slippage/slippage_profile.parquet`

Wenn es fehlt:
- zuerst `python scripts/calibrate_slippage.py` ausfuehren

### Monte-Carlo bricht mit Lifecycle-Fehler ab

Dann fehlt mindestens einer dieser Pools:
- Eval-Tage
- Funded-Tage mit `payout_cycle_id == 0`

Das bedeutet in der Regel:
- die aktuelle Strategie besteht die Eval auf echten Daten nicht robust genug

### Walk-Forward ist komplett `not_scored`

Dann erzeugt kein Parameterfenster ausreichend verwertbare Lifecycle-Artefakte.
Das ist ein brauchbares Research-Signal:
- entweder die Strategie ist aktuell zu schwach
- oder die Parameter-/Slippage-Annahmen sind noch nicht passend

## Empfehlung fuer Strategieentwicklung

Sinnvolle Reihenfolge:
1. Slippage kalibrieren
2. Backtest laufen lassen
3. Trade- und Daily-Log ansehen
4. erst danach Monte-Carlo oder Walk-Forward ernsthaft interpretieren

Pragmatisch:
- zuerst einen Baseline-Backtest stabil bekommen
- danach Entry, Exit, Tageslimits oder Position Sizing verbessern
- Monte-Carlo und Walk-Forward nur fuer Kandidaten verwenden, die den Lifecycle ueberhaupt erreichen
