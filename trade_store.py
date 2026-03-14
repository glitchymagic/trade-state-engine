"""
Trade Store — Atomic, file-locked interface to a trade log JSON file.

All reads/writes go through this module. Prevents duplicate trades via
fcntl exclusive locking and atomic writes (tmp + os.replace).

Usage:
    from trade_store import TradeStore
    store = TradeStore()
    open_trades = store.get_open_trades()
    store.close_trade("T001", exit_price=70000, exit_reason="Stop loss hit")
"""

import fcntl
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger("trade_store")

# Default correlation groups (can be overridden via constructor)
DEFAULT_CORRELATION_GROUPS = {
    "crypto_major": {"BTC", "ETH"},
    "crypto_alt": {"SOL", "XRP"},
    "tech_megacap": {"AAPL", "MSFT", "AMZN", "META", "NFLX"},
    "tech_semi": {"NVDA", "AMD"},
}


def _net_pnl(trade: dict) -> float:
    """Total P&L for a trade including partial exit P&L."""
    pnl = trade.get("pnl_usd", 0) or 0
    for pe in trade.get("partial_exits", []):
        pnl += pe.get("pnl_usd", 0) or 0
    return pnl


def _compute_equity(trades: list, starting_capital: float) -> float:
    """Compute current equity from trade data (starting_capital + realized P&L)."""
    realized = 0.0
    for t in trades:
        if t.get("status") == "CLOSED":
            realized += _net_pnl(t)
        elif t.get("status") == "OPEN":
            # Count partial exit P&L from open trades
            for pe in t.get("partial_exits", []):
                realized += pe.get("pnl_usd", 0) or 0
    return starting_capital + realized


class TradeStore:
    """Atomic, file-locked interface to a trade log JSON file.

    All risk checks (position cap, correlation limits, loss caps) are
    enforced under the exclusive flock — no TOCTOU window.

    Args:
        trade_log: Path to the trade log JSON file.
        lock_file: Path to the lock file for exclusive locking.
        max_positions: Maximum number of concurrent open positions.
        starting_capital: Starting capital for equity calculations.
        correlation_groups: Dict mapping group name to set of symbols.
        max_corr_group_risk_pct: Max risk per correlation group as fraction
            of equity (e.g., 4.0 means 4%).
        asset_consecutive_loss_cap: Block trading an asset after N consecutive
            losses. Set to 0 to disable.
        on_integrity_violation: Optional callback(violations: list[str]) called
            when post-write integrity checks find issues. Defaults to logging.
    """

    def __init__(
        self,
        trade_log=None,
        lock_file=None,
        max_positions=7,
        starting_capital=5000,
        correlation_groups=None,
        max_corr_group_risk_pct=4.0,
        asset_consecutive_loss_cap=3,
        on_integrity_violation=None,
    ):
        self.trade_log = Path(trade_log) if trade_log else Path("trades/trade_log.json")
        self.lock_file = Path(lock_file) if lock_file else Path("trades/trade.lock")
        self.max_positions = max_positions
        self.starting_capital = starting_capital
        self.correlation_groups = correlation_groups or DEFAULT_CORRELATION_GROUPS
        self.max_corr_group_risk_pct = max_corr_group_risk_pct / 100.0  # Store as fraction
        self.asset_consecutive_loss_cap = asset_consecutive_loss_cap
        self.on_integrity_violation = on_integrity_violation

        # Ensure parent dirs exist
        self.trade_log.parent.mkdir(parents=True, exist_ok=True)
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)

    # ─── Reading (no lock needed for reads) ───

    def _load(self) -> dict:
        """Load trade log. Returns default structure if missing/corrupt."""
        try:
            return json.loads(self.trade_log.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {"trades": [], "next_id": 1}

    def get_all_trades(self) -> list:
        """Return all trades (open + closed)."""
        return self._load().get("trades", [])

    def get_open_trades(self) -> list:
        """Return only OPEN trades."""
        return [t for t in self.get_all_trades() if t.get("status") == "OPEN"]

    def get_closed_trades(self) -> list:
        """Return only CLOSED trades."""
        return [t for t in self.get_all_trades() if t.get("status") == "CLOSED"]

    def get_trade(self, trade_id: str) -> dict | None:
        """Return a specific trade by ID, or None."""
        for t in self.get_all_trades():
            if t.get("id") == trade_id:
                return t
        return None

    def get_open_symbols(self) -> set:
        """Return set of symbols with OPEN positions."""
        return {t["symbol"] for t in self.get_open_trades()}

    def get_next_id(self) -> str:
        """Return next trade ID (e.g., T006)."""
        data = self._load()
        return f"T{data.get('next_id', 1):03d}"

    def position_count(self) -> int:
        """Return number of open positions."""
        return len(self.get_open_trades())

    def can_open_position(self, symbol: str) -> tuple:
        """Check if a new position can be opened (non-authoritative fast check).

        Returns (allowed: bool, reason: str).
        Note: The authoritative check happens inside open_trade() under lock.
        """
        open_trades = self.get_open_trades()
        open_symbols = {t["symbol"] for t in open_trades}

        if symbol in open_symbols:
            return False, f"Already have open position in {symbol}"
        if len(open_trades) >= self.max_positions:
            return False, f"Position cap reached: {len(open_trades)}/{self.max_positions}"
        return True, "OK"

    # ─── Writing (exclusive file lock) ───

    def _atomic_save(self, data: dict):
        """Write to tmp file then os.replace for atomic update."""
        tmp = str(self.trade_log) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
            f.write("\n")
        os.replace(tmp, str(self.trade_log))
        # Post-write integrity check (non-blocking)
        self._post_write_check()

    def _post_write_check(self):
        """Run integrity check after writes. Log violations, never block."""
        try:
            violations = self.verify_integrity()
            if violations:
                logger.warning("INTEGRITY VIOLATIONS: %s", violations)
                if self.on_integrity_violation:
                    try:
                        self.on_integrity_violation(violations)
                    except Exception:
                        pass
        except Exception:
            pass  # Never let integrity checks break trading

    def _with_lock(self, fn):
        """Execute fn under exclusive file lock. Returns fn's return value."""
        lock_fd = open(self.lock_file, "w")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            return fn()
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()

    def open_trade(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_price: float,
        target_price: float,
        size_usd: float,
        setup_type: str = "",
        grade: str = "",
        notes: str = "",
    ) -> dict:
        """Open a new trade under exclusive lock.

        All risk checks are enforced under the lock (TOCTOU-safe):
        - Position cap
        - Duplicate symbol check
        - Correlation group risk limits
        - Asset consecutive loss cap

        Args:
            symbol: Trading symbol (e.g., "BTC", "AAPL").
            direction: "LONG" or "SHORT".
            entry_price: Entry price.
            stop_price: Initial stop loss price.
            target_price: Take profit target price.
            size_usd: Position size in units (price-denominated).
            setup_type: Name of the setup/strategy that triggered the trade.
            grade: Setup quality grade (e.g., "A+", "A", "B").
            notes: Free-form notes about the trade.

        Returns:
            The new trade dict on success, or dict with "error" key on rejection.
        """
        def _do():
            data = self._load()
            open_trades = [t for t in data["trades"] if t.get("status") == "OPEN"]
            open_symbols = {t["symbol"] for t in open_trades}

            # Enforce position cap under lock
            if symbol in open_symbols:
                return {"error": f"Duplicate blocked: {symbol} already has OPEN position"}
            if len(open_trades) >= self.max_positions:
                return {"error": f"Position cap: {len(open_trades)}/{self.max_positions}"}

            # Correlation group risk limit (authoritative — under lock)
            new_group = None
            for group, members in self.correlation_groups.items():
                if symbol in members:
                    new_group = group
                    break
            if new_group:
                group_members = self.correlation_groups[new_group]
                equity = _compute_equity(data["trades"], self.starting_capital)
                # Sum risk of open trades in this group
                group_risk = 0.0
                for t in open_trades:
                    if t.get("symbol") in group_members:
                        orig_stop = t.get("original_stop") or t.get("stop_price", 0)
                        orig_size = t.get("original_size") or t.get("size", 0)
                        group_risk += abs(t.get("entry_price", 0) - orig_stop) * orig_size
                new_risk = abs(entry_price - stop_price) * size_usd
                max_group_risk = equity * self.max_corr_group_risk_pct
                if group_risk + new_risk > max_group_risk:
                    return {
                        "error": (
                            f"Correlation risk limit: ${group_risk + new_risk:,.0f} "
                            f"exceeds ${max_group_risk:,.0f} "
                            f"({self.max_corr_group_risk_pct*100:.0f}% equity) "
                            f"in '{new_group}' group"
                        )
                    }

            # Asset consecutive loss cap (authoritative — under lock)
            if self.asset_consecutive_loss_cap > 0:
                closed_for_symbol = [
                    t for t in data["trades"]
                    if t.get("symbol") == symbol and t.get("status") == "CLOSED"
                ]
                if len(closed_for_symbol) >= self.asset_consecutive_loss_cap:
                    recent = closed_for_symbol[-self.asset_consecutive_loss_cap:]
                    all_losses = all(_net_pnl(t) <= 0 for t in recent)
                    if all_losses:
                        return {
                            "error": (
                                f"Asset loss cap: {self.asset_consecutive_loss_cap} "
                                f"consecutive losses on {symbol}"
                            )
                        }

            trade_id = f"T{data.get('next_id', 1):03d}"
            trade = {
                "id": trade_id,
                "symbol": symbol,
                "direction": direction,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "original_stop": stop_price,
                "target_price": target_price,
                "size": size_usd,
                "original_size": size_usd,
                "entry_time": datetime.now(UTC).isoformat(),
                "exit_price": None,
                "exit_time": None,
                "exit_reason": None,
                "pnl_usd": None,
                "pnl_pct": None,
                "r_multiple": None,
                "status": "OPEN",
                "setup_type": setup_type,
                "grade": grade,
                "notes": notes,
            }

            data["trades"].append(trade)
            data["next_id"] = int(trade_id.replace("T", "")) + 1
            self._atomic_save(data)
            return trade

        return self._with_lock(_do)

    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str = "",
        lessons: str = "",
    ) -> dict:
        """Close a trade under exclusive lock. Calculates P&L and R-multiple.

        Returns the updated trade dict, or dict with "error" key.
        """
        def _do():
            data = self._load()
            for trade in data["trades"]:
                if trade["id"] == trade_id and trade["status"] == "OPEN":
                    trade["exit_price"] = exit_price
                    trade["exit_time"] = datetime.now(UTC).isoformat()
                    trade["exit_reason"] = exit_reason
                    trade["status"] = "CLOSED"
                    trade["lessons"] = lessons

                    # P&L calculation
                    direction = trade["direction"].upper()
                    if direction == "LONG":
                        pnl_per_unit = exit_price - trade["entry_price"]
                    else:
                        pnl_per_unit = trade["entry_price"] - exit_price

                    trade["pnl_usd"] = round(pnl_per_unit * trade["size"], 2)
                    trade["pnl_pct"] = round(
                        (pnl_per_unit / trade["entry_price"]) * 100, 2
                    )

                    risk_per_unit = abs(
                        trade["entry_price"]
                        - (trade.get("original_stop") or trade["stop_price"])
                    )
                    if risk_per_unit > 0:
                        trade["r_multiple"] = round(pnl_per_unit / risk_per_unit, 2)
                    else:
                        trade["r_multiple"] = 0.0

                    self._atomic_save(data)
                    return trade

            return {"error": f"Trade {trade_id} not found or not open"}

        return self._with_lock(_do)

    def partial_close(
        self,
        trade_id: str,
        close_pct: float,
        exit_price: float,
        reason: str = "Partial TP",
    ) -> dict:
        """Partially close a trade — reduce size and log the partial exit.

        Args:
            trade_id: ID of the trade to partially close.
            close_pct: Fraction to close (0.0 to 1.0, e.g., 0.5 = 50%).
            exit_price: Price at which the partial close executes.
            reason: Reason for the partial close.

        Returns:
            Dict with partial exit details, or error dict.
        """
        def _do():
            data = self._load()
            for trade in data["trades"]:
                if trade["id"] == trade_id and trade["status"] == "OPEN":
                    old_size = trade["size"]
                    close_size = round(old_size * close_pct, 6)
                    remaining_size = round(old_size - close_size, 6)

                    if remaining_size <= 0:
                        return {
                            "error": "Partial close would close entire position — use close_trade instead"
                        }

                    # Calculate P&L on the closed portion
                    direction = trade["direction"].upper()
                    if direction == "LONG":
                        pnl_per_unit = exit_price - trade["entry_price"]
                    else:
                        pnl_per_unit = trade["entry_price"] - exit_price
                    partial_pnl = round(pnl_per_unit * close_size, 2)

                    # Record partial exit
                    partial_exit = {
                        "exit_price": exit_price,
                        "size_closed": close_size,
                        "pnl_usd": partial_pnl,
                        "reason": reason,
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                    if "partial_exits" not in trade:
                        trade["partial_exits"] = []
                    trade["partial_exits"].append(partial_exit)

                    # Reduce position size
                    trade["size"] = remaining_size

                    self._atomic_save(data)
                    return {
                        "trade_id": trade_id,
                        "symbol": trade["symbol"],
                        "direction": trade["direction"],
                        "size_closed": close_size,
                        "remaining_size": remaining_size,
                        "exit_price": exit_price,
                        "partial_pnl": partial_pnl,
                        "entry_price": trade["entry_price"],
                    }

            return {"error": f"Trade {trade_id} not found or not open"}

        return self._with_lock(_do)

    def update_trade(self, trade_id: str, updates: dict) -> dict:
        """Update arbitrary fields on a trade under exclusive lock.

        Used for trailing stop updates, adding notes, etc.
        Returns updated trade or error dict.
        """
        def _do():
            data = self._load()
            for trade in data["trades"]:
                if trade["id"] == trade_id:
                    trade.update(updates)
                    self._atomic_save(data)
                    return trade
            return {"error": f"Trade {trade_id} not found"}

        return self._with_lock(_do)

    def update_trades_batch(self, updates: list) -> int:
        """Batch update multiple trades under a single lock acquisition.

        Args:
            updates: List of (trade_id, updates_dict) tuples.

        Returns:
            Number of trades updated.
        """
        def _do():
            data = self._load()
            trade_map = {t["id"]: t for t in data["trades"]}
            count = 0
            for trade_id, upd in updates:
                if trade_id in trade_map:
                    trade_map[trade_id].update(upd)
                    count += 1
            if count > 0:
                self._atomic_save(data)
            return count

        return self._with_lock(_do)

    def save_raw(self, data: dict):
        """Save raw data under lock. Used by reconciler or migration scripts."""
        self._with_lock(lambda: self._atomic_save(data))

    def verify_integrity(self) -> list:
        """Check invariants on the trade log. Returns list of violations.

        Called after every write to detect corruption early.
        Non-blocking — logs violations but never prevents writes.

        Invariants checked:
        - INV-1: No duplicate trade IDs
        - INV-2: OPEN trades have required fields, no exit data
        - INV-3: CLOSED trades have exit data
        - INV-4: Position cap not exceeded
        - INV-5: No duplicate open symbols
        - INV-6: P&L sign consistency for closed trades
        """
        trades = self.get_all_trades()
        violations = []

        # INV-1: No duplicate IDs
        ids = [t.get("id") for t in trades if t.get("id")]
        if len(ids) != len(set(ids)):
            dupes = [i for i in ids if ids.count(i) > 1]
            violations.append(f"Duplicate trade IDs: {set(dupes)}")

        for t in trades:
            tid = t.get("id", "???")

            # INV-2: OPEN trades have required fields, no exit data
            if t.get("status") == "OPEN":
                for f in ("entry_price", "stop_price", "size", "symbol", "direction"):
                    if not t.get(f):
                        violations.append(f"{tid}: OPEN missing {f}")
                if t.get("exit_price") is not None:
                    violations.append(f"{tid}: OPEN but has exit_price")

            # INV-3: CLOSED trades have exit data
            if t.get("status") == "CLOSED":
                for f in ("exit_price", "pnl_usd"):
                    if t.get(f) is None:
                        violations.append(f"{tid}: CLOSED missing {f}")

            # INV-6: P&L sign consistency for closed trades
            if (
                t.get("status") == "CLOSED"
                and t.get("exit_price")
                and t.get("entry_price")
                and t.get("pnl_usd") is not None
            ):
                direction = 1 if t.get("direction", "").upper() == "LONG" else -1
                expected_sign = (t["exit_price"] - t["entry_price"]) * direction
                pnl = t.get("pnl_usd", 0) or 0
                if (expected_sign > 0 and pnl < -0.01) or (
                    expected_sign < 0 and pnl > 0.01
                ):
                    violations.append(
                        f"{tid}: P&L sign mismatch "
                        f"(expected {'profit' if expected_sign > 0 else 'loss'}, "
                        f"got ${pnl:+.2f})"
                    )

        # INV-4: Position cap
        open_count = sum(1 for t in trades if t.get("status") == "OPEN")
        if open_count > self.max_positions:
            violations.append(
                f"Position cap exceeded: {open_count}/{self.max_positions}"
            )

        # INV-5: No duplicate open symbols
        open_syms = [t["symbol"] for t in trades if t.get("status") == "OPEN"]
        if len(open_syms) != len(set(open_syms)):
            violations.append(f"Duplicate open symbols: {open_syms}")

        return violations
