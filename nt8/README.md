# NinjaTrader 8 Validation Path

Dieser Ordner ist jetzt der aktive plattformnative Validierungspfad fuer neue Hypothesen auf `NT8`-Truth-Daten.

Die verbindliche Gesamtpipeline steht in [NINJATRADER_VALIDATION_PIPELINE.md](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/NINJATRADER_VALIDATION_PIPELINE.md).

Die verbindliche Datenspezifikation steht in [NT8_DATA_SPEC.md](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/NT8_DATA_SPEC.md).

## Aktive Datei

- [DataDumpExporter.cs](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/nt8/DataDumpExporter.cs)
- [RawBarDumpExporter.cs](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/nt8/RawBarDumpExporter.cs)
- [run_nt8_dual_feed_backtest.py](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/scripts/run_nt8_dual_feed_backtest.py)
- [nt8_dual_feed.py](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/propfirm/execution/nt8_dual_feed.py)

Die fruehere Donchian/SMA-Trendfolge ist archiviert:

- [archive/01_donchian_trend](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/archive/01_donchian_trend)

## Rolle Von NinjaTrader

- `Python + Databento` bleibt der Research- und Hypothesenpfad.
- `NinjaTrader + NinjaTrader-Daten` ist die finale Validierungsumgebung fuer die spaetere Ausfuehrung.
- Eine Python-validierte Strategie ist noch keine freigegebene NT8-Strategie.

## Ziel-Setup

- Eigene Strategie-Instanz pro Instrument
- `Calculate = OnBarClose`
- Keine zweite Strategie auf demselben Chart
- Trading-Hours-Template und Sessionlogik explizit dokumentieren
- Datenprovider fuer die Revalidierung einfrieren, bevor OOS und Forward Test starten

## Validierungsreihenfolge

1. Python-Research-Truth einfrieren
2. NinjaTrader-Datenbasis einfrieren
3. `Feed A`: HTF-Chart-Dump mit `DataDumpExporter`
4. `Feed B`: 1m-Rohdaten-Dump mit `RawBarDumpExporter`
5. Dual-Feed-Runner aufsetzen:
   [run_nt8_dual_feed_backtest.py](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/scripts/run_nt8_dual_feed_backtest.py)
6. Signale nur aus `Feed A`, Intrabar-Routing nur aus `Feed B`
7. IS/OOS strikt getrennt auswerten
8. Erst danach Slippage-, Drawdown- und Monte-Carlo-Stress
9. Erst danach Sim/Forward Test auf VPS

## Dual-Feed Regel

- Keine aktive Pandas-HTF-Rebuild-Logik mehr fuer NT8-Validierung
- `DataDumpExporter` ist die alleinige Source of Truth fuer Signal-Bars und Indikatoren
- `RawBarDumpExporter` ist die alleinige Source of Truth fuer 1m-Ausfuehrungsrouting
- `nt8_dual_feed.py` enthaelt jetzt nur noch ein leeres `generate_signals()`-Template fuer neue Hypothesen

## Mindeststandard Vor Forward Test

- Datenprovider, Contract-Handling, Rollregel und Sessiontemplate sind dokumentiert
- MNQ und MGC sind einzeln gegen Python plausibilisiert
- Kombinierte Portfolio-Kennzahlen auf NT8-Daten sind akzeptabel
- OOS kollabiert nicht
- Kosten- und Slippage-Stress zerstoeren die Edge nicht
- Forward Test laeuft auf genau derselben NT8-Umgebung, die spaeter live genutzt wird
