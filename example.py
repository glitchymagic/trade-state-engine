"""Example: Using the Trade State Engine."""
from pathlib import Path

from trade_store import TradeStore

# Initialize with a temp file
store = TradeStore(
    trade_log="/tmp/demo_trade_log.json",
    lock_file="/tmp/demo_trade.lock",
    max_positions=5,
    starting_capital=10000,
)

# Open a trade (all risk checks enforced under flock)
# size_usd is in units — 0.02 BTC at $65K = ~$1,300 notional
trade = store.open_trade(
    symbol="BTC",
    direction="LONG",
    entry_price=65000,
    stop_price=63500,
    target_price=68000,
    size_usd=0.02,
    setup_type="RSI_Divergence",
    grade="A",
    notes="Clean oversold bounce",
)
print(f"Opened: {trade['id']} — {trade['symbol']} {trade['direction']}")

# Check position count
print(f"Open positions: {store.position_count()}")
print(f"Can open another? {store.can_open_position('ETH')}")

# Partial close at 1R
result = store.partial_close(trade["id"], 0.5, 66500, "Partial TP at 1R")
print(f"Partial close: 50% at $66,500")

# Close the remainder
trade = store.close_trade(trade["id"], 67000, "Final TP hit")
print(f"Closed at $67,000 — P&L: ${trade.get('pnl_usd', 0):.2f}")

# Verify integrity
violations = store.verify_integrity()
print(f"Integrity: {'CLEAN' if not violations else violations}")

# Clean up
Path("/tmp/demo_trade_log.json").unlink(missing_ok=True)
Path("/tmp/demo_trade.lock").unlink(missing_ok=True)
