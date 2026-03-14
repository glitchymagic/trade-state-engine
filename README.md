# Trade State Engine

Atomic, file-locked state management for autonomous trading systems. Single source of truth pattern with self-healing derived state.

## The Problem

Trading systems that manage positions need:
- **Atomic writes** — a crash mid-write can't corrupt state
- **Exclusive locking** — scanner and position monitor can't write simultaneously
- **TOCTOU safety** — risk checks must happen under the same lock as the write
- **Self-healing** — if derived state drifts, it should auto-correct

## The Solution

### TradeStore — File-Locked Atomic State

```python
from trade_store import TradeStore

store = TradeStore(trade_log="trades.json", lock_file="trades.lock")

# All risk checks happen UNDER THE LOCK (no TOCTOU race):
# - Position cap enforcement
# - Duplicate symbol check
# - Correlation group risk limits
# - Asset consecutive loss cap
trade = store.open_trade("BTC", "LONG", 65000, 63500, 68000, 0.02, "RSI_Div", "A")
```

Key guarantees:
- **fcntl.flock()** exclusive locking (BSD semantics, per-FD)
- **Atomic writes** via temp file + `os.replace()`
- **Risk checks under lock** — position cap, correlation limits, loss caps all enforced atomically
- **Post-write integrity verification** — catches corruption immediately

### StateReconciler — Self-Healing Derived State

Every N minutes, rebuild ALL derived state from the single source of truth:

```python
from reconciler import StateReconciler

reconciler = StateReconciler("trades.json", output_dir="./state/")
reconciler.reconcile()
# Rebuilds: portfolio.json, equity.json, risk_state.json, setup_performance.json
```

If any derived file gets corrupted, it heals within one reconciliation cycle. The trade log is never modified by the reconciler — it's read-only.

## Architecture

```
                    WRITE PATH (locked)
                    ┌─────────────────┐
Scanner ──────────→ │   TradeStore     │
Position Monitor ─→ │  (fcntl.flock)  │ ──→ trade_log.json (SOURCE OF TRUTH)
                    │  atomic writes   │
                    └─────────────────┘

                    READ PATH (reconciler, every 15 min)
trade_log.json ──→ StateReconciler ──→ portfolio.json  (derived)
                                   ──→ equity.json     (derived)
                                   ──→ risk_state.json (derived)
                                   ──→ performance.json(derived)
```

## Key Design Decisions

1. **trade_log.json is the ONLY source of truth** — every other file is derived
2. **BSD flock() semantics** — locks are per file-descriptor, not per-process. Opening the same file twice and flocking both = self-deadlock (we learned this the hard way)
3. **Atomic writes everywhere** — write to .tmp, then `os.replace()`. Crash-safe.
4. **Risk checks under lock** — `open_trade()` checks position cap, correlation limits, and loss caps inside the locked section. No TOCTOU window.
5. **Corruption guard** — if trade count drops >50% between reconciliation cycles, abort and alert (prevents cascading data loss)
6. **Post-write integrity checks** — non-blocking verification after every write

## Requirements

- Python 3.11+
- macOS or Linux (uses `fcntl.flock`)
- No external dependencies (stdlib only)

## Bugs This Architecture Prevented

- **Duplicate trades** — scanner and position monitor both trying to write simultaneously (fixed by exclusive flock)
- **State drift** — 4+ derived files updated independently by different scripts (fixed by single source of truth + reconciler)
- **Corrupted state on crash** — process died mid-write leaving half-written JSON (fixed by atomic temp+replace)
- **Race condition in risk checks** — position cap checked before write, but another process opened a trade in between (fixed by TOCTOU-safe gateway)
