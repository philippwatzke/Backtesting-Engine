# NinjaTrader 8 Setup

Diese Datei beschreibt den produktiven Einsatz von [PropFirmBreakoutStrategy.cs](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/nt8/PropFirmBreakoutStrategy.cs) in NinjaTrader 8.

## Import

Es gibt zwei saubere Wege:

1. Direkter Source-Import
   - Kopiere [PropFirmBreakoutStrategy.cs](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/nt8/PropFirmBreakoutStrategy.cs) nach:
     - `Documents\NinjaTrader 8\bin\Custom\Strategies\`
   - Öffne in NinjaTrader:
     - `New > NinjaScript Editor`
   - Klicke `Compile`

2. Offizieller NinjaScript-ZIP-Import
   - Wenn du später aus NT8 heraus ein ZIP baust, importierst du es über:
     - `Tools > Import > NinjaScript...`

Wichtiger Hinweis:
- Die aktuelle Repo-Datei ist eine normale `.cs`-Quelldatei, kein exportiertes NT8-Archiv.
- Für den ersten lokalen Einsatz ist der direkte Copy- und Compile-Weg der einfachste.

## Chart Setup

Pro Instrument ein eigener Chart mit eigener Strategie-Instanz.

Gemeinsam:
- `Calculate = OnBarClose`
- Keine zweite Strategie auf demselben Chart
- Instrument-spezifische `TickSize` kommt vom Instrument selbst in NT8
- Trading Hours Template so wählen, dass dein Intraday-Chart die im Research verwendete Session sauber abbildet

Empfohlen:
- Session-Kontext auf dem Chart prüfen, bevor live gehandelt wird
- Zeitwerte im Strategie-Dialog gegen die tatsächlich sichtbare Chart-Zeit prüfen

## MGC Preset

Chart:
- Instrument: `MGC`
- Bar Type: `Minute`
- Value: `60`

Strategie-Parameter:
- `DonchianLookback = 5`
- `StopLossATR = 1.5`
- `TargetATR = 10.0`
- `StartTime = 090000`
- `EndTime = 110000`
- `DailyLossLimitUSD = 400`

Research-Herkunft:
- H1-Modell
- Entry-Fenster `09:00-11:00 ET`
- Daily-SMA50-Regime-Filter aktiv

## MNQ Preset

Chart:
- Instrument: `MNQ`
- Bar Type: `Minute`
- Value: `30`

Strategie-Parameter:
- `DonchianLookback = 10`
- `StopLossATR = 1.5`
- `TargetATR = 10.0`
- `StartTime = 103000`
- `EndTime = 110000`
- `DailyLossLimitUSD = 400`

Research-Herkunft:
- M30-Modell
- Entry-Fenster `10:30-11:00 ET`
- Daily-SMA50-Regime-Filter aktiv

## Live Hinweise

- Der Strategy-Code berechnet Signale auf `OnBarClose`, also analog zum Research-Modell mit nächster ausführbarer Kerze.
- Der Daily-Loss-Breaker ruft im `Realtime`-Pfad `CloseStrategy(...)` auf.
  - Das ist absichtlich hart.
  - Dadurch werden offene Positionen geschlossen und die Strategie deaktiviert.
- Im Historical/Analyzer-Fall verwendet der Code stattdessen einen normalen Exit-Fallback.

## Checkliste Vor Live

- Chart-Zeit prüfen: Stimmen `09:00` für MGC und `10:30` für MNQ wirklich mit deiner NT8-Zeitdarstellung überein?
- Instrument-Mapping prüfen: `MGC` und `MNQ` korrekter Front-/Micro-Kontrakt
- ATM aus: Die Strategie verwaltet Stop und Target selbst
- Positionsgröße in NT8 prüfen: Die aktuelle Version setzt keine dynamische Kontraktzahl aus dem Python-Risk-Engine-Modell um
- Sim-Konto zuerst mehrere Sessions laufen lassen

## Quellen

- NinjaTrader Import: https://ninjatrader.com/support/helpGuides/nt8/import.htm
- NinjaTrader Export: https://ninjatrader.com/support/helpguides/nt8/export.htm
- Forum-Hinweis zum manuellen `.cs`-Kopieren in `bin\\Custom\\Strategies`: https://forum.ninjatrader.com/forum/ninjatrader-8/strategy-development/1286343-strategy-builder-files
