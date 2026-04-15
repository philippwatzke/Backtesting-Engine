# TradingView Go / No-Go

## Ziel

Diese Datei trennt sauber:

- `Python legacy frozen` = Research Truth
- `TradingView MNQ legacy variant` = Execution Candidate

Die TradingView-Version muss nicht `1:1` Python sein.  
Sie darf aber nur dann gehandelt werden, wenn sie als **eigenstaendige Strategievariante** mit eigener Verifikation, eigener Sim-Phase und eigenen Risikoregeln freigegeben wurde.

## Aktueller Stand

- Python `mnq_legacy_frozen` ist die belastbare Referenz.
- TradingView ist aktuell **nicht** 1:1 deckungsgleich.
- Der grobe Signalcharakter stimmt oft.
- Die Restabweichung sitzt aktuell vor allem in:
  - Datenbasis / Continuous-Contract-Handling
  - Daily-Regime / SMA-Pfad
  - ATR / StopTicks / Qty
  - verkürzten Sessions / Feiertagen

## Harte Regel

Nicht sagen:

- "Python ist profitabel, also ist TradingView automatisch okay."

Nur sagen:

- "Python ist profitabel."
- "TradingView ist entweder ausreichend nah dran oder als eigene Variante separat profitabel."

## Go / No-Go Matrix

### Sofortiges No-Go

- TradingView-Signale werden gegen Python bewertet, obwohl TV auf anderer Datenbasis läuft.
- Daily-Regime / SMA / Donchian sind an Fokus-Tagen strukturell anders.
- Short Sessions werden inkonsistent mal gehandelt, mal ignoriert.
- Qty-Drift ist noch systematisch vorhanden.
- Es gibt noch keinen dokumentierten Sim-Run mit echten Alerts und Broker-Fills.

### Go fuer weitere Forensik

- Fokus-Tage sind identifiziert.
- TV-Chart-Einstellungen sind dokumentiert.
- Python-Referenz ist eingefroren.
- Restabweichungen sind auf wenige Tage reduziert.

### Go fuer Paper / Sim

Nur wenn alle Punkte erfuellt sind:

- TV-Datenbasis ist dokumentiert und bewusst akzeptiert.
- `2026-01-21` ist geklaert.
- Holiday-/Short-Session-Regel ist explizit entschieden.
- Die verbleibenden Qty-Drifts sind entweder beseitigt oder bewusst akzeptiert.
- Ein neuer TV-Export wurde gegen Python oder gegen die akzeptierte TV-Variante geprueft.

### Go fuer Echtgeld

Nur wenn alle Punkte erfuellt sind:

- Mindestens `20-30` Sim-Trades mit realem Alert-Flow
- keine doppelten Entries
- keine haengenden Positionen
- HardClose funktioniert sauber
- Symbolmapping und Qty-Mapping stimmen
- die tatsaechliche Sim-PnL passt zur erwarteten TV-Variante

## Konkreter Arbeitsplan

### Phase 1: Datenbasis entscheiden

1. TradingView-Chart exakt dokumentieren:
- Symbol
- Continuous-Contract-Modus
- Back-Adjustment / Adjustment
- Session-Template
- Timezone

2. Zwei Kontrolltage neu pruefen:
- `2026-01-21`
- `2026-01-23`

3. Auf beiden Tagen fuer `10:30 ET` und `11:00 ET` notieren:
- `Close`
- `SMA50`
- `Donchian H`
- `Regime`
- `Close > H`
- `Close > SMA`
- `Reg Long`
- `Raw Long`

Ziel:
- wenn diese Werte weiter strukturell von Python abweichen, ist das primaer ein Datenproblem, kein Pine-Problem.

### Phase 2: Session-Regel einfrieren

Entscheide genau eine Regel:

- verkuerzte Sessions nicht handeln
- oder verkuerzte Sessions bewusst handeln

Die Regel muss in Python-Vergleich und TV-Ausfuehrung gleich behandelt werden.

### Phase 3: Sizing sauber ziehen

Die verbleibenden Qty-Tage:

- `2025-09-03`
- `2025-09-22`
- `2025-10-01`
- `2025-10-27`
- `2026-01-27`

Fuer diese Tage vergleichen:

- `ATR14`
- `StopTicks`
- `EffRisk`
- `Qty`

Ziel:
- systematischen ATR-/Sizing-Drift entweder beseitigen oder explizit akzeptieren.

### Phase 4: Neue TV-Variante evaluieren

Erst jetzt:

1. TV-Export neu ziehen
2. denselben Zeitraum wie Python verwenden
3. Export archivieren
4. Soll-Ist-Abgleich fahren

Dann eine der zwei Aussagen treffen:

- `TV ist nah genug an Python`
- `TV ist eigene profitable Variante`

Keine Mischform.

## Was du konkret tun sollst

### Wenn du maximale methodische Sauberkeit willst

1. Python als alleinige Truth beibehalten
2. TV nur dann handeln, wenn Datenbasis und Restdrift dokumentiert sind
3. erst Sim, dann Echtgeld

### Wenn du maximal pragmatisch vorgehen willst

1. akzeptieren, dass TV eine eigene Variante ist
2. diese TV-Variante separat backtesten
3. dann `2-4` Wochen Sim mit CrossTrade
4. erst danach Echtgeld

## Mein professionelles Urteil

Wenn du zu einem Ergebnis kommen willst, das du mit gutem Gewissen handeln kannst, dann brauchst du nicht zwingend `1:1` Paritaet.  
Du brauchst aber zwingend **klare Systemidentitaet**.

Das bedeutet:

- entweder: "Ich handle Python-paritaetsnah"
- oder: "Ich handle eine eigenstaendig validierte TV-Variante"

Was du nicht tun solltest:

- eine teilweise abweichende TV-Version mit Python-Ergebnissen rechtfertigen

Das ist der eigentliche methodische Fehler, der spaeter teuer wird.
