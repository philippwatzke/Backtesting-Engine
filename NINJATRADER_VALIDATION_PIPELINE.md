# NinjaTrader Validation Pipeline

Stand: `2026-04-14`

Dieses Dokument ist die verbindliche Pipeline fuer die aktive Breakout-Strategielinie auf `MNQ` und `MGC`.

Verbindliche Datenspezifikation:
- [NT8_DATA_SPEC.md](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/NT8_DATA_SPEC.md)

## Aktive Architektur

- `Python + Databento` = Research, Hypothesen, Parametrisierung, historische Referenz
- `NinjaTrader + NinjaTrader-Daten` = plattformnative Revalidierung, OOS, Stress, Forward Test
- `TradingView` = archiviert, nicht mehr aktiver Entscheidungs- oder Ausfuehrungspfad

## Strategie-Scope

- `MNQ Breakout`
- `MGC Breakout`
- spaeter als kombiniertes Portfolio zu bewerten, aber zuerst je Instrument einzeln robust machen

## Grundsatz

Eine Strategie ist erst dann verantwortbar forward-testbar, wenn sie nicht nur auf Python-/Databento-Daten gut aussieht, sondern auch auf der echten NinjaTrader-Datenwelt stabil bleibt.

Python ist die Forschungswahrheit.  
NinjaTrader ist die Handelswahrheit.

## Phase 1: Python Truth Freezen

Ziel:
- die aktuelle Research-Referenz bewusst nicht mehr moving target werden lassen

Verbindliche Referenzen:
- [scripts/run_mnq_legacy_frozen.py](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/scripts/run_mnq_legacy_frozen.py)
- [configs/default_params_mnq_legacy_frozen.toml](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/configs/default_params_mnq_legacy_frozen.toml)
- [scripts/run_clean_portfolio_validation.py](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/scripts/run_clean_portfolio_validation.py)
- [configs/default_params.toml](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/configs/default_params.toml)
- [configs/default_params_mnq.toml](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/configs/default_params_mnq.toml)

Ergebnis:
- klare Python-Benchmark fuer `MNQ`, `MGC` und das kombinierte Breakout-Portfolio

## Phase 2: NinjaTrader Data Basis Freezen

Vor jeder Strategievalidierung muessen diese Punkte fixiert sein:

- Datenprovider
- konkretes Instrument-/Contract-Mapping
- Continuous- oder Einzelkontrakt-Regel
- Rollregel
- Trading-Hours-Template
- Zeitzone
- Bar-Konstruktion

Ohne diese Spezifikation ist jeder Backtest potenziell driftend.

## Phase 3: NinjaScript Rebuild Und Signalabgleich

Ziel:
- die Python-Logik in NinjaTrader sauber nachbauen
- nicht blind "ungefaehr gleich", sondern nachvollziehbar

Arbeitsschritte:

1. NinjaScript pro Instrument plausibilisieren
2. Kernserien gegen Python vergleichen:
- SMA
- ATR
- Donchian
- Regime
- Entry-Zeitfenster
- Exit-Logik

3. Kontrolltage dokumentieren:
- Tage mit Trade in beiden Systemen
- Python-only-Tage
- NT-only-Tage

Ergebnis:
- erklaerbare Drift statt blinder Abweichung

## Phase 4: NT8 OOS Validation

Erst wenn die Logik plausibel steht, folgt der echte Test:

- In-Sample nicht neu optimieren, bis OOS vorliegt
- OOS pro Instrument getrennt auswerten
- danach Portfolio auf NT8-Daten zusammenfuehren

Zu pruefen:
- Net Profit
- Profit Factor
- Max Drawdown
- Trefferquote
- Avg Trade
- Trade Count
- Equity-Form im juengsten Zeitraum

## Phase 5: Stress Und Robustheit

Pflicht vor Forward Test:

- Slippage-Stress
- Kosten-Stress
- Monte Carlo / Resampling
- Drawdown-Stress
- Tagesverlust- und Execution-Fail-Szenarien

Ziel:
- pruefen, ob die Edge robust ist oder nur auf idealisierter Historie lebt

## Phase 6: Forward Test Auf NinjaTrader

Forward Test erst, wenn die Phasen 1 bis 5 bestanden sind.

Forward-Test-Regeln:

- nur auf der echten spaeteren NT8-Umgebung
- idealerweise auf VPS
- gleiche Datenquelle wie im NT8-Backtest
- gleiche Sessiontemplates
- gleiche Strategieversion
- keine stillen Parameteraenderungen

Zu loggen:

- Signalzeit
- Fillzeit
- Fillpreis
- Slippage zur Backtest-Erwartung
- Session-/Disconnect-Probleme
- Daily-loss-breaker-Verhalten

## Go/No-Go Vor Forward Test

Go nur, wenn alle Punkte gelten:

- Python-Referenz ist eingefroren
- NT8-Datenbasis ist eingefroren
- MNQ und MGC sind einzeln plausibilisiert
- Portfolio auf NT8-Daten bleibt wirtschaftlich sinnvoll
- OOS kollabiert nicht
- Slippage- und Monte-Carlo-Stress zerstoeren die Edge nicht
- die NT8-Implementierung ist operational sauber fuer VPS/Sim

No-Go, wenn einer dieser Punkte zutrifft:

- Datenbasis noch unklar
- starke ungeklaerte Signaldrift
- OOS deutlich schwach oder negativ
- Strategie lebt nur von wenigen Clustertagen
- operative Umsetzung in NT8 ist noch nicht stabil

## Kritische Quant-Einschaetzung

Der aktuelle Pfad ist methodisch deutlich besser als TradingView.

Aber:
- Python-Erfolg allein ist keine Handelsfreigabe
- NT8-Revalidierung ist nicht kosmetisch, sondern der eigentliche Filter
- wenn die Edge auf NT8-Daten nicht stabil bleibt, war sie nicht robust genug fuer diese Ausfuehrungswelt

Wenn die Strategie jedoch auf NinjaTrader-Daten sauber repliziert, OOS besteht und die Stress-Tests ueberlebt, dann ist der Uebergang in den Forward Test fachlich gerechtfertigt.
