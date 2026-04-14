# TradingView Alert Setup

Aktueller TV-Referenzpfad:
- [`PropFirmBreakoutStrategy_MNQ_Legacy.pine`](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/tradingview/PropFirmBreakoutStrategy_MNQ_Legacy.pine)

Archivierte nicht aktuelle Varianten:
- [`archive/2026-04-14_noncurrent_variants/PropFirmBreakoutStrategy_MNQ.pine`](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/tradingview/archive/2026-04-14_noncurrent_variants/PropFirmBreakoutStrategy_MNQ.pine)
- [`archive/2026-04-14_noncurrent_variants/PropFirmBreakoutStrategy_Production.pine`](/c:/Users/phili/Prop-Firm%20Backtesting%20Engine/tradingview/archive/2026-04-14_noncurrent_variants/PropFirmBreakoutStrategy_Production.pine)

Diese Anleitung gilt fuer die Legacy-Datei.

## Voraussetzungen

- MNQ auf einem `1m`-Chart laden
- `ADJ` im Chart aktivieren
- Pine-Skript als `Strategy` hinzufuegen
- Bridge-Ziel kennen:
  - Webhook-Bridge wie PineConnector / eigener Receiver
  - oder Message-Box / Copy-Paste-Bridge

## Alert In TradingView

1. `Add Alert` auf dem Chart oeffnen.
2. Bei `Condition` die Strategie `PropFirm Breakout Strategy - MNQ (30m signals from 1m)` waehlen.
3. Als Event `Order fills only` oder `Order fills and alert() function calls` verwenden.
4. In das Feld `Message` exakt diesen Platzhalter setzen:

```text
{{strategy.order.alert_message}}
```

Das ist wichtig, weil das Pine-Skript die JSON-Nachricht dynamisch pro Entry/Exit erzeugt.

## Webhook vs Alert Box

`Webhook URL`:
- Verwenden, wenn deine Bridge einen HTTP-Endpoint erwartet.
- Die Webhook-URL in `Webhook URL` eintragen.
- Im `Message`-Feld trotzdem `{{strategy.order.alert_message}}` verwenden.

`Alert Box` / Message-only:
- Verwenden, wenn deine Bridge keine Webhook-URL nutzt.
- Kein Webhook noetig.
- Im `Message`-Feld ebenfalls `{{strategy.order.alert_message}}` verwenden.
- TradingView sendet dann das JSON in der normalen Alert-Nachricht.

## JSON Format

Beispiel fuer einen Entry:

```json
{"action":"buy","symbol":"MNQ1!","qty":"1","orderType":"market","sl":"24500.25","tp":"24950.25","reason":"LongBreakout","source":"tradingview","timestamp":"2026-04-13 10:00:00"}
```

Beispiel fuer einen Exit:

```json
{"action":"close","symbol":"MNQ1!","qty":"1","orderType":"market","reason":"HardCloseLong","source":"tradingview","timestamp":"2026-04-13 15:59:00"}
```

## Sicherheitslogik

Das Skript hat jetzt eine optionale Intraday-Loss-Sperre:

- Input: `Max Intraday Loss USD`
- `0` bedeutet deaktiviert
- Bei Ueberschreitung:
  - werden keine neuen Entries mehr zugelassen
  - offene Positionen werden per `strategy.close(...)` geschlossen
  - es wird ein eigener Exit-Alert mit `reason = MaxIntradayLossLong` oder `MaxIntradayLossShort` erzeugt
