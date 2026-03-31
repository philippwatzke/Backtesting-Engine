# Prop-Firm Backtesting Engine — Design Specification

**Date:** 2026-03-30
**Status:** Draft
**Scope:** Phases 0–2 (Data, Simulator, Grid Search, ORB Baseline). Phase 3 (LLM-assisted Strategy Search) deferred.

---

## 1. Business Context & Objective

Algorithmische Trading-Engine zur systematischen Ausnutzung der konvexen Auszahlungsstruktur von Prop-Firms.

- **Downside:** Begrenzt auf Evaluierungskosten
- **Upside:** Asymmetrische Payouts
- **Zielfunktion:** Maximierung des Net Expected Value (NVE) = (Pass Rate x E[Capped Payout]) - Evaluierungskosten

**Target:** MyFundedFutures (MFF) Flex Plan $50k.

---

## 2. MFF Flex $50k Regelwerk

### 2.1 Evaluierungs-Phase

| Regel | Wert |
|-------|------|
| Profit Target | $3.000 |
| Maximum Loss Limit (MLL) | $2.000 (EOD Trailing Drawdown, kein Intraday-Trailing) |
| Konsistenzregel | Kein einzelner Gewinntag > 50% des Gesamtprofits |
| Mindest-Handelstage | 2 |
| Max Kontrakte | 50 Micro (MNQ) |
| Tägliches Verlustlimit (Prop-Firm) | Keins |

### 2.2 Sim-Funded-Phase

| Regel | Wert |
|-------|------|
| Drawdown | $2.000 EOD Trailing |
| Drawdown nach erstem Payout | Friert statisch bei $100 ein |
| Scaling | $0–$1.499: 20 Micros, $1.500–$1.999: 30 Micros, $2.000+: 50 Micros |
| Payout-Qualifikation | 5 Gewinntage (je >= $150 Tages-PNL) |
| Auszahlbar | Max 50% des Gesamtprofits, gedeckelt bei $5.000/Payout |
| Mindest-Auszahlung | $250 |
| Profit-Split | 80% Trader / 20% Firma |

---

## 3. Daten-Fundament

### 3.1 Quelle & Instrument

- **Quelle:** Databento (Dataset: GLBX.MDP3)
- **Instrument:** MNQ.v.0 (Continuous Contract, volume-based roll)
- **Aufloesung:** 1-Minuten OHLCV
- **Zeitzone:** America/New_York (US-Ostkueste)
- **Session:** Regular Trading Hours (RTH: 09:30–16:00 ET)
- **Overnight:** Ausgeschlossen

### 3.2 Datentrennung (unveraenderlich)

| Split | Zeitraum | Bars | Handelstage |
|-------|----------|------|-------------|
| Train | 2022-01-03 bis 2024-06-28 | 247.035 | 643 |
| Validation | 2024-07-01 bis 2025-03-31 | 73.875 | 193 |
| Test | 2025-04-01 bis 2026-03-27 | 98.085 | 256 |

Test-Set ist eine absolute Firewall — wird nur einmal am Ende geoeffnet.

---

## 4. Architektur: Geschichtete Module mit Numba-Kernels

### 4.1 Package-Layout

```
prop-firm-engine/
├── pyproject.toml
├── configs/
│   ├── mff_flex_50k.toml           # MFF-Regelwerk (deklarativ)
│   └── default_params.toml         # Strategie + Simulator + Monte-Carlo Settings
├── data/
│   ├── raw/
│   │   └── MNQ_1m_2022_heute_raw.parquet
│   ├── processed/
│   │   ├── MNQ_1m_train.parquet
│   │   ├── MNQ_1m_validation.parquet
│   │   └── MNQ_1m_test.parquet
│   └── slippage/
│       └── slippage_profile.parquet
├── output/                          # .gitignore'd — Reports, Logs, Plots
│   ├── grid_search/
│   ├── backtests/
│   └── monte_carlo/
├── propfirm/                        # Importierbares Python-Package
│   ├── __init__.py
│   ├── core/
│   │   ├── engine.py               # Numba-JIT Bar-Loop (Hot Path)
│   │   └── types.py                # NumPy Structured Arrays, Numba NamedTuples
│   ├── market/
│   │   ├── slippage.py             # Hybrid-Slippage-Modell
│   │   └── data_loader.py          # Parquet -> NumPy Arrays + Preprocessing
│   ├── rules/
│   │   └── mff.py                  # MFF-Regelwerk + Phase-Shift State
│   ├── risk/
│   │   └── risk.py                 # Circuit Breaker, Position Sizing
│   ├── strategy/
│   │   └── orb.py                  # Opening Range Breakout Baseline
│   ├── monte_carlo/
│   │   └── bootstrap.py            # Block-Bootstrap + Multiprocessing
│   ├── optim/
│   │   ├── grid_search.py          # Synthetischer Grid Search (Phase 2)
│   │   ├── objective.py            # Optuna Objectives + Capped NVE
│   │   └── walk_forward.py         # Walk-Forward-Analyse
│   └── io/
│       ├── reporting.py            # JSON-Reports + Audit-Trail
│       └── config.py               # TOML-Loader, Validierung
├── scripts/
│   ├── calibrate_slippage.py       # Slippage-Profil kalibrieren
│   ├── run_grid_search.py          # CLI: Grid Search
│   ├── run_backtest.py             # CLI: Einzelne Strategie
│   ├── run_monte_carlo.py          # CLI: Monte-Carlo
│   └── run_walk_forward.py         # CLI: Walk-Forward-Analyse
└── tests/
    ├── test_engine.py
    ├── test_rules.py
    ├── test_slippage.py
    └── test_strategy.py
```

### 4.2 Design-Prinzipien

- **TOML fuer Konfiguration:** MFF-Regelwerk deklarativ, kein Code-Change bei Regelaenderungen
- **Numba nur auf dem Hot Path:** Bar-Iteration im JIT, EOD-Logik in Python
- **Precomputed Arrays:** ATR, Slippage-Vektoren, `minute_of_day` einmalig vorab berechnet, nicht im Loop
- **Time-Sync statt Modulo:** Uhrzeit wird NICHT ueber `bar_idx % bars_per_day` berechnet (versagt bei verkuerztem Handel/Datenluecken). Stattdessen liefert `data_loader.py` ein `minute_of_day`-Array (int16, 0–389) aus echten Timestamps. Der Kernel liest `minute_of_day[bar_idx]`.
- **Daten bleiben in `data/`, Artefakte in `output/`:** Saubere Trennung
- **Slippage-Profil als persistiertes Artefakt:** Einmal kalibriert, bei jedem Run wiederverwendet

---

## 5. Simulator-Architektur

### 5.1 Datenfluss

```
TOML Config --> config.py --> Parameter-Dict
                                    |
Parquet --> data_loader.py --> NumPy Arrays (OHLCV, precomputed ATR,
                               slippage_per_bar, trailing_median_atr)
                                    |
                    +------------------------------+
                    |     engine.py (@njit)         |
                    |                              |
                    |  Input:  price arrays         |
                    |          strategy_fn          |
                    |          params (flat)        |
                    |          slippage_lookup      |
                    |          precomputed arrays   |
                    |                              |
                    |  Output: trade_log (struct)   |
                    |          equity_curve         |
                    |          bar_states           |
                    +------------------------------+
                                    |
                    +------------------------------+
                    |  Python-Schicht (post-JIT)   |
                    |                              |
                    |  rules/mff.py -> EOD-Check   |
                    |  risk/risk.py -> Validierung |
                    |  io/reporting.py -> JSON     |
                    +------------------------------+
```

### 5.2 Numba-Boundary

| Im @njit-Kernel (engine.py) | In Python (rules/, risk/) |
|-----------------------------|---------------------------|
| Bar-Iteration ueber OHLCV | EOD Trailing Drawdown Update |
| Signal-Generierung (strategy_fn) | Konsistenzregel (50%-Check) |
| Entry/Exit-Ausfuehrung | Phase-Shift Eval -> Funded |
| Slippage + Commission anwenden | Payout-Berechnung |
| Intraday-PNL tracken | Scaling-Plan Enforcement |
| Circuit Breaker pruefen | Capped NVE-Berechnung |
| Hard-Close 15:59 ET | Reporting |
| Trade-Log in Structured Array schreiben | |

**Begruendung:** Kernel iteriert ~250k Bars (Hot Path). EOD-Logik laeuft ~643 mal pro Split und braucht komplexe Zustandsuebergaenge — in Python wartbarer.

### 5.3 State-Strukturen (core/types.py)

```python
# Numba-kompatible NamedTuples fuer den Kernel
SimState:
    equity:          float64   # Aktueller Kontostand
    intraday_pnl:    float64   # PNL des aktuellen Tages
    entry_price:     float64   # 0.0 = flat (keine Position)
    position:        int64     # -N / 0 / +N Kontrakte
    daily_stop_level: float64  # Circuit Breaker Level
    halted:          boolean   # True wenn CB gefeuert hat
    daily_trade_count: int64   # Trades heute

# Structured Array dtype fuer Trade-Log (vor-alloziert)
TRADE_LOG_DTYPE:
    entry_time:      int64     # Unix-Timestamp
    exit_time:       int64
    entry_price:     float64
    exit_price:      float64
    slippage:        float64   # Variable Markt-Slippage
    commission:      float64   # Fixe Gebuehren ($0.54/Seite/Kontrakt)
    contracts:       int32
    pnl:             float64   # Netto nach Slippage + Commission
    signal_type:     int8      # 1=Long, -1=Short
    exit_reason:     int8      # 0=Target, 1=Stop, 2=HardClose, 3=CircuitBreaker
```

### 5.4 Phase-Shift State (rules/mff.py)

```python
MFFState:
    phase:                   "eval" | "funded"
    equity:                  float
    eod_high:                float    # Hoechster EOD-Stand (fuer Trailing)
    mll:                     float    # Aktuelles Maximum Loss Limit
    mll_frozen:              bool     # True nach erstem Payout
    trading_days:            int
    winning_days:            int      # Tage mit PNL >= $150
    total_profit:            float
    max_single_day_profit:   float
    payouts_completed:       int
    payout_eligible:         bool     # True wenn winning_days >= 5 UND total_profit >= 500

    update_eod(day_pnl, eod_equity) -> "continue" | "passed" | "blown"
    get_active_params()              -> "params_eval" | "params_funded"
    get_max_contracts()              -> int (Scaling Plan)
```

### 5.5 Zwei-Ebenen State Machine — Tagesablauf

```
09:30   Kernel startet mit SimState
        Strategy-Funktion generiert Signale im Bar-Loop

09:30-15:58  Bar-Loop (@njit):
        ├── strategy_fn(bar_idx, ohlcv, sim_state, params) -> signal
        ├── Signal? -> Slippage + Commission berechnen -> Entry/Exit
        ├── Intraday-PNL updaten
        ├── Circuit Breaker Check (daily_stop_level)
        │   └── Halted? -> Keine weiteren Trades
        └── Trade in Structured Array loggen

15:59   Hard-Close: Offene Position zwangsweise glattstellen

16:00   Kernel gibt trade_log + final SimState zurueck
        |
        v  Python EOD-Schicht:
        MFFState.update_eod(day_pnl, eod_equity)
        ├── eod_high updaten (nur wenn neues High)
        ├── Trailing DD pruefen: eod_equity <= eod_high - mll? -> BLOWN (konservativ: >= MLL = blown)
        ├── Konsistenz-Check updaten
        ├── Winning Day? (>= $150)
        ├── Phase-Shift Check (Eval bestanden?)
        └── Return: continue / passed / blown
```

---

## 6. Slippage-Modell

### 6.1 Hybrid-Architektur

```
Effektive Slippage = Tageszeit-Baseline(bucket) x ATR-Multiplikator(bar) x Richtungs-Penalty
```

**Komponente 1 — Tageszeit-Lookup (kalibriert aus Trainingsdaten):**

Berechnet in `scripts/calibrate_slippage.py`, persistiert als `data/slippage/slippage_profile.parquet`.

| Bucket (ET) | Erwartete Tendenz |
|-------------|-------------------|
| 09:30–09:45 | Hoch (Open-Auction, Spread-Ausweitung) |
| 09:45–10:30 | Erhoet (Fruehe Volatilitaet) |
| 10:30–11:30 | Normal |
| 11:30–13:30 | Niedrig (Lunch-Dip) |
| 13:30–15:00 | Normal |
| 15:00–15:45 | Leicht erhoeht (MOC-Flow) |
| 15:45–16:00 | Hoch (Close-Auction) |

Konkrete Werte werden empirisch kalibriert (Spread-Proxy: High-Low bei Bars im unteren Volume-Quartil).

**Komponente 2 — ATR-Multiplikator (kausal korrekt):**

```
atr_multiplier = bar_atr / trailing_5d_median_atr
```

KRITISCH: `trailing_5d_median_atr` nutzt ausschliesslich den Median-ATR der letzten 5 abgeschlossenen Handelstage. Keine Zukunftsdaten, kein Look-Ahead Bias. Wird in `data_loader.py` als precomputed Array bereitgestellt.

**Komponente 3 — Richtungs-Penalty:**

Stop-Market-Orders bei schnellen Moves gegen die Position: konfigurierbarer Penalty-Faktor (z.B. 1.5x). Parameter in TOML.

### 6.2 Slippage vs. Commission — Getrennte Behandlung

| Kosten-Typ | Modell | Wert |
|------------|--------|------|
| Slippage | Variabel (Hybrid-Modell) | Dynamisch pro Bar |
| Commission | Fix pro Kontrakt/Seite | $0.54/Seite/MNQ |

Getrennte Felder im Trade-Log fuer Diagnostik: "Scheitert das System an der Ausfuehrung oder an Fixkosten?"

### 6.3 Im Numba-Kernel

Lookup-Table als flaches NumPy-Array (Index = Minuten seit 09:30):

```python
@njit
def compute_slippage(minute_of_day, bar_atr, trailing_median_atr,
                     slippage_lookup, is_stop_order, stop_penalty):
    baseline = slippage_lookup[minute_of_day]
    atr_mult = bar_atr / trailing_median_atr if trailing_median_atr > 0 else 1.0
    penalty = stop_penalty if is_stop_order else 1.0
    raw = baseline * atr_mult * penalty * 0.25  # MNQ Tick-Size
    return max(raw, 0.25)  # Floor: mindestens 1 Tick Slippage
```

### 6.4 Kalibrierungsprozess

`scripts/calibrate_slippage.py`:
1. Laed Trainingsdaten
2. Berechnet pro 15-Min-Bucket: Median Spread-Proxy (High - Low bei Bars im unteren Volume-Quartil)
3. Normalisiert relativ zum Gesamt-Median
4. Speichert als `slippage_profile.parquet` (Spalten: bucket_start, bucket_end, baseline_ticks)

---

## 7. Monte-Carlo-Engine

### 7.1 Block-Bootstrap

Zieht zufaellig zusammenhaengende Trade-Bloecke statt isolierter Einzeltrades. Erhaelt serielle Autokorrelation (Verlustserien).

```
Trade-Log (N Trades)
    -> Zerlege in Bloecke (Laenge: konfigurierbar, 5–10)
    -> Ziehe M Bloecke zufaellig mit Zuruecklegen
    -> Spiele durch MFFState (EOD-Checks, Trailing DD, Konsistenz)
    -> Ergebnis: passed / blown + Metriken
    -> Wiederhole S mal (default 10.000)
    -> Output: Pass-Rate-Verteilung
```

### 7.2 Konfiguration

```toml
[monte_carlo]
n_simulations = 10_000
block_size_min = 5
block_size_max = 10
random_seed = 42  # Default = global seed
```

### 7.3 Multiprocessing

Simulationen sind vollstaendig unabhaengig. Jede Simulation bekommt einen eigenen Seed (`global_seed + simulation_index`). Workers erhalten Chunks dieser Seed-Listen, nicht einen einzelnen Worker-Seed.

### 7.4 Stress-Test-Modus (separater Report)

Beantwortet: "Wie fragil ist der Edge?"

```toml
[stress_test]
slippage_multiplier = 3.0     # 3x normale Slippage
drawdown_buffer = 0.8         # Nur 80% des MLL nutzen
worst_day_injection = true    # Schlimmster Tag in jeden Pfad einfuegen
```

Separater Report in `output/monte_carlo/`. Kontaminiert nicht die NVE-Berechnung.

---

## 8. Grid Search & Optimierung

### 8.1 Synthetischer Grid Search (Phase 2)

Findet optimale Risiko-Geometrie ohne Marktdaten:

1. Fuer jede Parameter-Kombination:
   - Generiere synthetische Trade-Sequenz (Bernoulli mit Win-Rate, fixe R:R-Ratio)
   - Spiele durch MFFState (alle MFF-Regeln)
   - Monte-Carlo Block-Bootstrap
   - Berechne Capped NVE
2. Ranking nach NVE -> Top-K Parameter-Sets

### 8.2 State-Aware Parameter-Shift

Zwei separate Parameter-Sets, da sich das Ziel nach Eval aendert:

**Eval-Phase** (Ziel: Pass Rate maximieren):
- Win-Rate: 0.55–0.70
- Risk/Reward: 1.0–2.0
- Contracts: 5–20
- Daily-Stop: -$500 bis -$1.000

**Funded-Phase** (Ziel: Payouts maximieren):
- Win-Rate: 0.45–0.60
- Risk/Reward: 1.5–3.0
- Contracts: 10–50
- Daily-Stop: -$500 bis -$1.500

### 8.3 Capped NVE-Berechnung (objective.py)

Die NVE-Formel bildet die gesamte Payout-Kette ab:

```
payout_gross = min(total_profit * 0.5, 5000)     # 50%-Regel + $5k Cap
payout_valid = payout_gross if payout_gross >= 250 else 0  # $250 Minimum (brutto)
payout_net = payout_valid * 0.8                   # 80/20 Split

nve = pass_rate * E[payout_net] - eval_cost
```

Ohne diese Kette optimiert der Optimizer auf unrealistische Gewinne.

### 8.4 Optuna-Integration (Phase 3)

```python
objective(trial):
    params = trial.suggest(...)
    trade_log = run_backtest(strategy_fn, data, params, mff_config)
    mc_result = run_monte_carlo(trade_log, mff_config, mc_config)
    return mc_result.nve
```

- **Sampler:** TPESampler(seed=global_seed)
- **Pruner:** MedianPruner(n_startup_trials=20, n_warmup_steps=5)

### 8.5 Walk-Forward-Analyse (optim/walk_forward.py)

Testet Parameter-Stabilitaet ueber wechselnde Marktregime. Expandierendes Fenster:

```
Window 1: Train 2022-01 -> 2022-12 | Test 2023-01 -> 2023-03
Window 2: Train 2022-01 -> 2023-03 | Test 2023-04 -> 2023-06
...
Window N: Train 2022-01 -> 2024-03 | Test 2024-04 -> 2024-06
```

Output:
- Parameter-Drift ueber Fenster
- Out-of-Sample NVE pro Fenster
- Recalibration Frequency

---

## 9. ORB-Baseline-Strategie

### 9.1 Konzept

Opening Range Breakout: Die ersten N Minuten nach Eroeffnung bilden eine Range. Ausbruch ueber High / unter Low signalisiert Tagesrichtung.

### 9.2 Signal-Generierung (im Kernel)

```python
@njit
def orb_signal(bar_idx, ohlcv, sim_state, params):
    minute_of_day = bar_idx % bars_per_day

    # Phase 1: Range aufbauen
    if minute_of_day < params.range_minutes:
        # Track High/Low der Range
        return 0

    # Self-Awareness Checks
    if sim_state.intraday_pnl >= params.daily_target:
        return 0  # Hit and Run
    if sim_state.halted:
        return 0  # Circuit Breaker
    if sim_state.daily_trade_count >= params.max_trades_day:
        return 0

    # Phase 2: Breakout Detection
    if close > range_high + buffer:
        return +1  # Long
    if close < range_low - buffer:
        return -1  # Short

    return 0
```

### 9.3 Trade-Management

```
Entry:   Stop-Market am Breakout-Level + Slippage
Stop:    stop_ticks unterhalb/oberhalb Entry (Stop-Market)
Target:  target_ticks in Gewinnrichtung (Stop-Market, nicht Limit)
Exit:    Erstbestes von: Target | Stop | Hard-Close 15:59 | Circuit Breaker
```

Target als Stop-Market statt Limit-Order: konservativer, vermeidet Touch-vs-Through-Problem.

### 9.4 Optimierbare Parameter

| Parameter | Range | Typ |
|-----------|-------|-----|
| range_minutes | 5–30 | int |
| stop_ticks | 10–100 | float |
| target_ticks | 15–200 | float |
| contracts | 1–50 | int (getrennt Eval/Funded) |
| daily_stop | -300 bis -1500 | float (getrennt Eval/Funded) |
| daily_target | 300–1500 | float (getrennt Eval/Funded) |
| max_trades_day | 1–3 | int |
| buffer_ticks | 0–5 | float |
| volume_threshold | 0.0–2.0 | float (0 = deaktiviert) |

### 9.5 Strategie-Interface (fuer Phase 3 Drop-in-Replacements)

Jede Strategie muss eine @njit-kompatible Funktion liefern:

```
signature: (bar_idx, ohlcv_arrays, sim_state, strategy_params) -> signal (+1, -1, 0)
```

Plus eine deklarative Liste optimierbarer Parameter mit Ranges.

---

## 10. Reporting & Audit

### 10.1 JSON-Report-Struktur (pro Trial)

```json
{
    "meta": {
        "git_hash": "abc123",
        "timestamp": "2026-03-30T14:22:00Z",
        "random_seed": 42,
        "config_snapshot": { ... },
        "data_split": "train",
        "data_date_range": ["2022-01-03", "2024-06-28"]
    },
    "params": {
        "eval": { ... },
        "funded": { ... }
    },
    "results": {
        "pass_rate": 0.72,
        "pass_rate_ci_5": 0.68,
        "pass_rate_ci_95": 0.76,
        "nve": 1240.50,
        "mean_days_to_pass": 12.3,
        "mean_drawdown": -890.0,
        "consistency_margin": 0.18,
        "total_trades": 847,
        "win_rate": 0.61
    },
    "stress_test": {
        "pass_rate_stressed": 0.58,
        "nve_stressed": 620.30
    },
    "diagnostics": {
        "slippage_total": 2340.0,
        "commission_total": 915.0,
        "circuit_breaker_hits": 14,
        "hard_close_exits": 3,
        "blown_reason_distribution": {
            "trailing_dd": 0.85,
            "consistency_rule": 0.10,
            "circuit_breaker_cascade": 0.05
        }
    }
}
```

### 10.2 Trade-Log

Parquet-File pro Backtest-Run in `output/backtests/`. Schema entspricht TRADE_LOG_DTYPE (Sektion 5.3).

---

## 11. Tech Stack

| Komponente | Technologie |
|------------|-------------|
| Sprache | Python 3.11+ |
| Hot Path | Numba @njit |
| Daten | Pandas (IO), NumPy (Kernel) |
| Konfiguration | TOML (tomllib) |
| Optimierung | Optuna + TPESampler + MedianPruner |
| Parallelisierung | concurrent.futures.ProcessPoolExecutor |
| Serialisierung | Parquet (Daten), JSON (Reports) |
| Testing | pytest |
| Reproduzierbarkeit | Global Seed in Config, Git-Hash in Reports |

---

## 12. Scope & Abgrenzung

### In Scope (Phase 0–2)

- Daten-Pipeline (bereits vorhanden)
- Slippage-Kalibrierung
- Simulator (Engine + MFF-Rules + Slippage)
- Grid Search (synthetisch)
- ORB-Baseline-Strategie
- Monte-Carlo Block-Bootstrap + Stress-Test
- Walk-Forward-Analyse
- JSON-Reporting + Audit-Trail

### Explizit Out of Scope

- LLM-Agent Strategy Search (Phase 3 — spaeter, interaktiv)
- Limit-Order Touch-vs-Through Modell (Phase 3+)
- News-Event Calendar/Filter (Phase 3, Strategie-Entscheidung)
- Stationary Block-Bootstrap mit geometrischer Verteilung (Upgrade wenn noetig)
- Visuelle Plots (Phase 2+, nach strukturiertem Reporting)
- Live-Trading-Anbindung

---

## 13. Deferred Decisions

| Entscheidung | Aufgeschoben bis | Grund |
|-------------|-----------------|-------|
| Touch-vs-Through Limit Fill | Phase 3 | ORB nutzt Stop-Market-Orders |
| News-Blackout-Filter | Phase 3 | Strategie-Entscheidung, nicht Simulator |
| Variable Bootstrap-Blocklaenge | Bedarf | Feste Bloecke ausreichend |
| Plot-Generation | Nach Phase 2 | JSON-Reports haben Prioritaet |
| Multi-Instrument Support | Nicht geplant | Fokus auf MNQ |
