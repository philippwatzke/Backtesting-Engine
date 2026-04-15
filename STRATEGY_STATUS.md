# Strategy Status

Stand: `2026-04-15`

## Hartes Fazit

Die Donchian/SMA-Trendfolge ist verworfen.

- Grund: mangelnder Follow-Through auf `NT8`-Truth-Daten
- Ergebnis: negativer Erwartungswert nach Dual-Feed-Validierung
- Status: archiviert unter [archive/01_donchian_trend](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/archive/01_donchian_trend)

Die `Dual-Feed`-Pipeline selbst ist verifiziert und bleibt aktiv.

- `Feed A`: HTF-Signal-Truth aus `NT8`
- `Feed B`: 1m-Execution-Truth aus `NT8`
- Status: einsatzbereit fuer neue Hypothesen aus `Market Microstructure` und `Mean Reversion`

## Aktiver Pfad

Der aktive Arbeits- und Entscheidungsweg ist jetzt:

- `Python + Databento` fuer Research und Referenz
- `NinjaTrader + NinjaTrader-Daten` fuer Revalidierung, OOS, Stress und Forward Test

Verbindliche Prozessdoku:
- [NINJATRADER_VALIDATION_PIPELINE.md](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/NINJATRADER_VALIDATION_PIPELINE.md)
- [NT8_DATA_SPEC.md](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/NT8_DATA_SPEC.md)

## Python Research Truth

Aktiv ist nur noch die Infrastruktur und nicht mehr die verworfene Donchian-Hypothese.

- neue Hypothesen sollen den `Dual-Feed`-Pfad verwenden
- historische Donchian-Artefakte liegen im Archiv

## NinjaTrader Active

- [nt8/README.md](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/nt8/README.md)
- [run_nt8_dual_feed_backtest.py](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/scripts/run_nt8_dual_feed_backtest.py)
- [nt8_dual_feed.py](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/propfirm/execution/nt8_dual_feed.py)
- [DataDumpExporter.cs](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/nt8/DataDumpExporter.cs)
- [RawBarDumpExporter.cs](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/nt8/RawBarDumpExporter.cs)

NinjaTrader ist jetzt der einzig aktive plattformnative Validierungspfad fuer die spaetere Handelsumgebung.
Der aktive Python-NT8-Pfad ist jetzt `Dual-Feed` mit leerem Signal-Template:

- `Feed A`: fertige NT8-HTF-Dumps fuer Signalbildung
- `Feed B`: NT8-1m-Rohdaten fuer Intrabar-Ausfuehrung
- keine aktive HTF-Rebuild- oder Resampling-Logik mehr im Handels-Pfad
- keine aktive Donchian/SMA-Logik mehr im Execution-Kern

## Archiv

Donchian-Trendfolge:

- [archive/01_donchian_trend](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/archive/01_donchian_trend)

## TradingView Archived

Historische TradingView-Artefakte sind archiviert und nicht mehr aktiver Referenzpfad:

- [tradingview/archive/2026-04-14_shelved_for_nt8_validation/PropFirmBreakoutStrategy_MNQ_Legacy.pine](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/tradingview/archive/2026-04-14_shelved_for_nt8_validation/PropFirmBreakoutStrategy_MNQ_Legacy.pine)
- [tradingview/archive/2026-04-14_shelved_for_nt8_validation/TRADINGVIEW_GO_NO_GO.md](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/tradingview/archive/2026-04-14_shelved_for_nt8_validation/TRADINGVIEW_GO_NO_GO.md)
- [data/archive/2026-04-14_tradingview_attempts](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/data/archive/2026-04-14_tradingview_attempts)
- [output/archive/2026-04-14_tradingview_forensics](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/output/archive/2026-04-14_tradingview_forensics)

## Aktiv Vs. Historisch

Aktiv:
- NinjaTrader-Rohdaten und `NT8`-Dumps
- Dual-Feed-Execution und Reporting
- neue Hypothesen auf `NT8`-Basis

Historisch:
- Donchian/SMA-Trendfolge inkl. Autopsien und Paritaets-Reports
- TradingView-Paritaets- und Alert-Pfad
- alte Experimente unter `scripts/archive`, `configs/archive`, `tests/archive`, `output/archive`

## Arbeitsregel

Wenn unklar ist, welcher Stand gilt:

1. zuerst [STRATEGY_STATUS.md](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/STRATEGY_STATUS.md) lesen
2. dann [NINJATRADER_VALIDATION_PIPELINE.md](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/NINJATRADER_VALIDATION_PIPELINE.md) lesen
3. nur mit aktiven Python-Referenzen und dem `nt8`-Pfad arbeiten
4. TradingView ausschliesslich als historische Dokumentation behandeln
