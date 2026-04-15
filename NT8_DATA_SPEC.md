# NT8 Data Specification

Stand: `2026-04-14`

Dieses Dokument ist die verbindliche Datenspezifikation fuer die aktive NinjaTrader-Validierung von `MNQ Breakout` und `MGC Breakout`.

## Grundsatz

Vor jedem Export und vor jeder NT8-Validierung gelten diese Einstellungen als eingefroren.

Wenn spaeter einer dieser Punkte geaendert wird, ist das methodisch eine neue Datenwelt und muss als neuer Validierungslauf behandelt werden.

## Provider Policy

Primaerer Provider:
- `Tradovate`

Fallback fuer tiefere Historie:
- `Kinetick`

Regel:
- `Kinetick` darf nur dann verwendet werden, wenn die `Tradovate`-Historie fuer den benoetigten Backtestzeitraum nicht weit genug zurueckreicht.
- Bei Fallback muessen alle anderen Einstellungen identisch bleiben.
- Ergebnisse aus `Tradovate` und `Kinetick` duerfen nicht stillschweigend gemischt werden. Der verwendete Provider muss pro Export explizit protokolliert werden.

## Trading Hours Template

Verbindlich:
- `CME US Index Futures ETH`

Begruendung:
- volle elektronische Session
- korrekte CME-Pausen
- keine kuenstliche Verkuerzung der Session
- stabile Berechnung von `SMA`, `ATR`, `Donchian` und Sessionkontext

## Contract Handling

Verbindlich:
- `Continuous Contract`
- `Merge Policy = Merge Back Adjusted`
- Roll-Tage nach Standard-CME-Vorgaben von NinjaTrader

Begruendung:
- dieses Setup ist der beabsichtigte NT8-Analysestandard fuer den Datenabgleich gegen Python
- Abweichungen an dieser Stelle veraendern Preislevel, Indikatoren und Regime

## Timezone Policy

Chart-UI:
- darf in `America/New_York` betrieben werden

CSV-Export:
- muss in `UTC` erfolgen

Regel:
- Sommerzeit-/DST-Logik wird ausschliesslich auf der Python-Seite behandelt
- keine lokale Zeit als Export-Truth verwenden

## Export Format

Verbindlich:
- `OHLCV`
- Timestamp in `UTC`
- bevorzugt als eindeutiger UTC-Zeitstempel ohne UI-Ambiguitaet

Akzeptierte Timestamp-Formen:
- Unix Timestamp in Millisekunden
- oder `YYYY-MM-DD HH:MM:SS` in strikt dokumentiertem `UTC`

Preisdaten:
- keine UI-Rundungen
- Rohwerte aus dem Export verwenden

## Instrument Specs

### MNQ

- Instrument: `MNQ`
- Datenquelle: `Tradovate`, sonst `Kinetick` als dokumentierter Fallback
- Contract Handling: `Continuous`
- Merge Policy: `Merge Back Adjusted`
- Trading Hours Template: `CME US Index Futures ETH`
- Timezone im Export: `UTC`
- Bar Type: `Minute`
- Bar Size: `30`
- Exportinhalt: `timestamp, open, high, low, close, volume`

### MGC

- Instrument: `MGC`
- Datenquelle: `Tradovate`, sonst `Kinetick` als dokumentierter Fallback
- Contract Handling: `Continuous`
- Merge Policy: `Merge Back Adjusted`
- Trading Hours Template: `CME US Index Futures ETH`
- Timezone im Export: `UTC`
- Bar Type: `Minute`
- Bar Size: `60`
- Exportinhalt: `timestamp, open, high, low, close, volume`

## Export Package

Jeder Exportlauf muss mindestens diese Metadaten mitfuehren:

- `export_date`
- `provider`
- `instrument`
- `contract_handling = continuous`
- `merge_policy = back_adjusted`
- `trading_hours_template = CME US Index Futures ETH`
- `export_timezone = UTC`
- `bar_type`
- `bar_size`
- `date_range`

## Validation Levels

### Level 1: WYSIWYG Chart Truth

Artefakt:
- `NT8_Dump...csv` aus [DataDumpExporter.cs](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/nt8/DataDumpExporter.cs)

Zweck:
- den konkreten NT8-Chart so abgreifen, wie er wirklich gerechnet wird
- schneller erster Plausibilitaetscheck fuer Bars und Indikatoren

### Level 2: NT8-Native Source Truth

Artefakt:
- `NT8_RawDump...csv` aus [RawBarDumpExporter.cs](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/nt8/RawBarDumpExporter.cs)

Zweck:
- aktuelle NT8-native Rohbars exportieren
- hoehere Timeframes in Python selbst rekonstruieren
- Drift in Session-, DST-, Roll- oder Barlogik gezielt isolieren

## Small Control Run

Vor dem Voll-Export wird nur ein kleiner Kontrolllauf gemacht.

Ziel:
- pruefen, ob Datenbasis und Signalpfad zwischen `Python` und `NT8` grundsaetzlich sauber genug sind

### Kontrolllauf-Scope

- zuerst `MNQ` und `MGC` getrennt
- kleiner Zeitraum von `2 bis 4 Wochen`
- Zeitraum muss normale Handelstage enthalten, keine ausschliesslich atypischen Feiertagscluster
- wenn moeglich mindestens:
  - Tage mit Signal
  - Tage ohne Signal
  - mindestens ein Beispiel pro Long-/Short- oder Regimewechsel, falls vorhanden

### Kontrolllauf-Exports

Pro Instrument:
- ein OHLCV-Bar-Export gemaess dieser Spezifikation
- optional zusaetzlich ein NT8-Trade- oder Strategy-Analyzer-Export, falls bereits vorhanden

### Kontrolllauf-Abgleich Gegen Python

Pro Instrument vergleichen:
- `SMA`
- `ATR`
- `Donchian`
- `Regime`
- `Signalzeitpunkt`
- `Trade-Richtung`
- `Entry-Bar`

Ziel:
- nicht perfekte kosmetische Gleichheit erzwingen
- sondern klaeren, ob Drift erklaerbar und wirtschaftlich tragbar ist

### Go/No-Go Nach Dem Kontrolllauf

Go fuer Voll-Export:
- Exportschema ist stabil
- Zeitstempel kommen sauber in `UTC`
- keine offenkundige Session- oder Roll-Fehlkonfiguration
- Signalabweichungen sind klein oder klar erklaerbar

No-Go fuer Voll-Export:
- UTC-Export unklar oder inkonsistent
- Trading Hours Template weicht faktisch ab
- Continuous-/Merge-Setup ist nicht reproduzierbar
- Signale kippen unerklaert bereits im Kontrollfenster

## Arbeitsregel

Vor jedem neuen NT8-Export:

1. [STRATEGY_STATUS.md](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/STRATEGY_STATUS.md) lesen
2. [NINJATRADER_VALIDATION_PIPELINE.md](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/NINJATRADER_VALIDATION_PIPELINE.md) lesen
3. dieses Dokument gegen die geplanten NT8-Einstellungen abgleichen
4. erst dann exportieren
