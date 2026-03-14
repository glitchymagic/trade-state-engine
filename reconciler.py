"""
State Reconciler — Rebuilds all derived state from a trade log.

Single source of truth pattern: the trade log is NEVER modified by the
reconciler. All derived files (portfolio, equity, risk state, setup
performance) are rebuilt from scratch each cycle.

If any derived file gets corrupted, it auto-heals within one reconciliation
cycle. The corruption guard prevents cascading data loss if the trade log
itself is damaged.

Usage:
    from reconciler import StateReconciler

    reconciler = StateReconciler(
        trade_log_path="trades/trade_log.json",
        output_dir="./state/",
        starting_capital=10000,
    )
    reconciler.reconcile()
"""

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger("reconciler")


def _net_pnl(trade: dict) -> float:
    """Total P&L for a closed trade including partial exit P&L.

    close_trade() only records P&L on the remaining portion.
    If partial TPs were taken, their P&L is stored in partial_exits[].pnl_usd.
    This function sums both to get the true net P&L for win/loss classification.
    """
    pnl = trade.get("pnl_usd", 0) or 0
    for pe in trade.get("partial_exits", []):
        pnl += pe.get("pnl_usd", 0)
    return pnl


def _partial_exit_pnl(trades: list) -> float:
    """Sum P&L from partial exits on still-open trades."""
    total = 0
    for t in trades:
        if t.get("status") == "OPEN":
            for pe in t.get("partial_exits", []):
                total += pe.get("pnl_usd", 0)
    return total


def _atomic_save(path: Path, data):
    """Atomic write: tmp + os.replace. Crash-safe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    if isinstance(data, str):
        with open(tmp, "w") as f:
            f.write(data)
    else:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
            f.write("\n")
    os.replace(tmp, str(path))


class StateReconciler:
    """Rebuilds all derived state from the trade log.

    Derived files:
    - portfolio.json   — open positions, equity, win/loss stats
    - equity.json      — equity curve, peak, drawdown, metrics
    - risk_state.json  — drawdown tracking, halt flags
    - setup_performance.json — per-setup win rate, avg R, expectancy

    Args:
        trade_log_path: Path to the trade log JSON file (source of truth).
        output_dir: Directory to write derived state files.
        starting_capital: Starting capital for equity calculations.
        drawdown_limits: Dict with 'daily', 'weekly', 'warning', 'total' keys
            as negative fractions (e.g., -0.02 for -2%). Defaults provided.
        on_corruption: Optional callback(prev_count, curr_count) called when
            the corruption guard triggers. Defaults to logging.
        on_invariant_violation: Optional callback(violations: list[str]) called
            when post-reconcile checks find issues. Defaults to logging.
    """

    def __init__(
        self,
        trade_log_path,
        output_dir,
        starting_capital=5000,
        drawdown_limits=None,
        on_corruption=None,
        on_invariant_violation=None,
    ):
        self.trade_log_path = Path(trade_log_path)
        self.output_dir = Path(output_dir)
        self.starting_capital = starting_capital
        self.drawdown_limits = drawdown_limits or {
            "daily": -0.02,
            "warning": -0.05,
            "weekly": -0.10,
            "total": -0.20,
        }
        self.on_corruption = on_corruption
        self.on_invariant_violation = on_invariant_violation

        # Derived file paths
        self.portfolio_file = self.output_dir / "portfolio.json"
        self.equity_file = self.output_dir / "equity.json"
        self.risk_state_file = self.output_dir / "risk_state.json"
        self.setup_perf_file = self.output_dir / "setup_performance.json"

        # Ensure output dir exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_trade_log(self) -> dict:
        """Load the source of truth."""
        try:
            return json.loads(self.trade_log_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {"trades": [], "next_id": 1}

    def reconcile(self):
        """Main reconciliation: rebuild all derived state from the trade log.

        Steps:
        1. Load trade log (source of truth)
        2. Corruption guard — abort if trade count drops >50%
        3. Rebuild portfolio, equity, risk state, setup performance
        4. Atomic-save all derived files
        5. Post-reconcile invariant checks
        """
        data = self.load_trade_log()
        trades = data.get("trades", [])

        # Corruption guard — if trade count drops dramatically, abort
        try:
            existing_eq = json.loads(self.equity_file.read_text())
            prev_count = existing_eq.get("total_trades", 0)
            curr_count = len([t for t in trades if t.get("status") == "CLOSED"])
            if prev_count > 10 and curr_count < prev_count * 0.5:
                msg = (
                    f"CRITICAL: trade count dropped {prev_count} -> {curr_count}. "
                    "Possible trade log corruption. Skipping rebuild."
                )
                logger.critical(msg)
                if self.on_corruption:
                    try:
                        self.on_corruption(prev_count, curr_count)
                    except Exception:
                        pass
                return
        except (FileNotFoundError, json.JSONDecodeError):
            pass  # First run or no equity file yet — proceed normally

        # Load existing equity for curve preservation
        existing_equity = None
        try:
            existing_equity = json.loads(self.equity_file.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        # Rebuild everything
        portfolio = self.rebuild_portfolio(trades)
        equity = self.rebuild_equity(trades, existing_equity)
        risk_state = self.rebuild_risk_state(trades, equity)
        setup_perf = self.rebuild_setup_performance(trades)

        # Portfolio heat + streak detection (merged into equity data)
        heat = self._calculate_portfolio_heat(trades, equity["current_equity"])
        equity["portfolio_heat"] = heat["heat_pct"]
        equity["heat_per_position"] = heat["heat_per_position"]

        streaks = self._calculate_streaks(trades)
        equity["current_streak"] = streaks["current_streak"]
        equity["max_win_streak"] = streaks["max_win_streak"]
        equity["max_loss_streak"] = streaks["max_loss_streak"]
        equity["last_5_results"] = streaks["last_5_results"]

        # Atomic save all derived files
        _atomic_save(self.portfolio_file, portfolio)
        _atomic_save(self.equity_file, equity)
        _atomic_save(self.risk_state_file, risk_state)
        _atomic_save(self.setup_perf_file, setup_perf)

        # Post-reconcile invariant checks
        try:
            self._check_invariants(trades, portfolio, equity)
        except Exception as e:
            logger.warning("Invariant check failed: %s", e)

        open_count = len([t for t in trades if t.get("status") == "OPEN"])
        closed_count = len([t for t in trades if t.get("status") == "CLOSED"])
        logger.info(
            "Reconciled: %d open, %d closed, equity=$%s, dd=%.1f%%",
            open_count,
            closed_count,
            f"{equity['current_equity']:,.0f}",
            equity["drawdown_percent"],
        )

    def rebuild_portfolio(self, trades: list) -> dict:
        """Rebuild portfolio state from trade list."""
        open_trades = [t for t in trades if t.get("status") == "OPEN"]
        closed_trades = [t for t in trades if t.get("status") == "CLOSED"]

        realized_pnl = (
            sum(_net_pnl(t) for t in closed_trades) + _partial_exit_pnl(trades)
        )
        winning = sum(1 for t in closed_trades if _net_pnl(t) > 0)
        losing = sum(1 for t in closed_trades if _net_pnl(t) < 0)
        current_equity = self.starting_capital + realized_pnl

        open_positions = []
        for t in open_trades:
            open_positions.append(
                {
                    "trade_id": t["id"],
                    "symbol": t["symbol"],
                    "direction": t["direction"],
                    "size": t["size"],
                    "entry_price": t["entry_price"],
                }
            )

        return {
            "balance": round(current_equity, 2),
            "starting_capital": self.starting_capital,
            "realized_pnl": round(realized_pnl, 2),
            "current_equity": round(current_equity, 2),
            "total_trades": len(closed_trades),
            "winning_trades": winning,
            "losing_trades": losing,
            "open_positions": open_positions,
            "last_updated": datetime.now(UTC).isoformat(),
        }

    def rebuild_equity(self, trades: list, existing_equity: dict = None) -> dict:
        """Rebuild equity curve, metrics, and drawdown state."""
        closed_trades = [t for t in trades if t.get("status") == "CLOSED"]

        realized_pnl = (
            sum(_net_pnl(t) for t in closed_trades) + _partial_exit_pnl(trades)
        )
        current_equity = self.starting_capital + realized_pnl
        winning = sum(1 for t in closed_trades if _net_pnl(t) > 0)
        losing = sum(1 for t in closed_trades if _net_pnl(t) < 0)

        # Preserve equity curve from existing file if available
        existing_curve = []
        inception_date = datetime.now(UTC).strftime("%Y-%m-%d")
        if existing_equity:
            existing_curve = existing_equity.get("equity_curve", [])
            inception_date = existing_equity.get("inception_date", inception_date)

        # Update or add today's entry
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        curve = list(existing_curve)
        if curve and curve[-1].get("date") == today:
            curve[-1]["equity"] = round(current_equity, 2)
        else:
            curve.append({"date": today, "equity": round(current_equity, 2)})

        peak = max(
            [e.get("equity", self.starting_capital) for e in curve]
            + [self.starting_capital]
        )
        drawdown_pct = (
            round(((current_equity - peak) / peak) * 100, 2) if peak > 0 else 0
        )

        drawdown_status = "NORMAL"
        if drawdown_pct <= self.drawdown_limits["total"] * 100:
            drawdown_status = "STOP"
        elif drawdown_pct <= self.drawdown_limits["weekly"] * 100:
            drawdown_status = "REDUCE"
        elif drawdown_pct <= self.drawdown_limits["warning"] * 100:
            drawdown_status = "WARNING"

        # Profit factor and win rate (using net P&L including partial exits)
        gross_wins = sum(_net_pnl(t) for t in closed_trades if _net_pnl(t) > 0)
        gross_losses = abs(
            sum(_net_pnl(t) for t in closed_trades if _net_pnl(t) < 0)
        )
        profit_factor = (
            round(gross_wins / gross_losses, 2) if gross_losses > 0 else 0
        )
        win_rate = (
            round((winning / len(closed_trades)) * 100, 1) if closed_trades else 0
        )

        # Total return
        total_return_pct = (
            round(
                ((current_equity - self.starting_capital) / self.starting_capital)
                * 100,
                2,
            )
            if self.starting_capital > 0
            else 0
        )

        # Max drawdown (worst peak-to-trough %) and duration
        max_dd_pct = 0.0
        max_dd_days = 0
        current_dd_days = 0
        running_peak = self.starting_capital
        for entry in curve:
            eq = entry.get("equity", self.starting_capital)
            if eq >= running_peak:
                running_peak = eq
                current_dd_days = 0
            else:
                current_dd_days += 1
                max_dd_days = max(max_dd_days, current_dd_days)
                dd = round(((eq - running_peak) / running_peak) * 100, 2)
                if dd < max_dd_pct:
                    max_dd_pct = dd

        # Average R-multiple
        r_multiples = []
        for t in closed_trades:
            r = t.get("r_multiple")
            if r is not None and r != 0:
                r_multiples.append(r)
            else:
                entry = t.get("entry_price", 0)
                stop = t.get("original_stop") or t.get("stop_price", 0)
                risk = abs(entry - stop) if entry and stop else 0
                pnl = _net_pnl(t)
                size = t.get("original_size") or t.get("size", 0) or 0
                if risk > 0 and size > 0:
                    r_multiples.append(round(pnl / (risk * size), 2))
        avg_r = round(sum(r_multiples) / len(r_multiples), 2) if r_multiples else 0

        return {
            "starting_capital": self.starting_capital,
            "current_equity": round(current_equity, 2),
            "total_return_pct": total_return_pct,
            "peak_equity": round(peak, 2),
            "drawdown_percent": drawdown_pct,
            "max_drawdown_pct": max_dd_pct,
            "drawdown_status": drawdown_status,
            "current_drawdown_days": current_dd_days,
            "max_drawdown_days": max_dd_days,
            "total_trades": len(closed_trades),
            "winning_trades": winning,
            "losing_trades": losing,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_r": avg_r,
            "total_pnl": round(realized_pnl, 2),
            "inception_date": inception_date,
            "last_updated": datetime.now(UTC).isoformat(),
            "equity_curve": curve,
        }

    def rebuild_risk_state(self, trades: list, equity: dict) -> dict:
        """Rebuild risk state with drawdown tracking and halt flags."""
        open_trades = [t for t in trades if t.get("status") == "OPEN"]
        current_value = equity.get("current_equity", self.starting_capital)

        # Load existing risk state for daily/weekly reset dates
        existing = {}
        try:
            existing = json.loads(self.risk_state_file.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        today = datetime.now(UTC).strftime("%Y-%m-%d")
        weekday = datetime.now(UTC).weekday()  # 0=Monday

        # Reset daily/weekly start values on date rollover
        daily_start = existing.get("daily_start_value", self.starting_capital)
        weekly_start = existing.get("weekly_start_value", self.starting_capital)
        daily_reset = existing.get("daily_reset_date", today)
        weekly_reset = existing.get("weekly_reset_date", today)

        if daily_reset != today:
            daily_start = current_value
            daily_reset = today

        if weekday == 0 and weekly_reset != today:
            weekly_start = current_value
            weekly_reset = today

        # Calculate drawdowns
        daily_dd = (
            ((current_value - daily_start) / daily_start) if daily_start > 0 else 0
        )
        weekly_dd = (
            ((current_value - weekly_start) / weekly_start) if weekly_start > 0 else 0
        )
        total_dd = (
            ((current_value - self.starting_capital) / self.starting_capital)
            if self.starting_capital > 0
            else 0
        )

        # Check halt conditions
        halted = False
        halt_reason = None
        if daily_dd <= self.drawdown_limits["daily"]:
            halted = True
            halt_reason = (
                f"Daily drawdown {daily_dd*100:.1f}% exceeds "
                f"{self.drawdown_limits['daily']*100}% limit"
            )
        elif weekly_dd <= self.drawdown_limits["weekly"]:
            halted = True
            halt_reason = (
                f"Weekly drawdown {weekly_dd*100:.1f}% exceeds "
                f"{self.drawdown_limits['weekly']*100}% limit"
            )
        elif total_dd <= self.drawdown_limits["total"]:
            halted = True
            halt_reason = (
                f"Total drawdown {total_dd*100:.1f}% exceeds "
                f"{self.drawdown_limits['total']*100}% limit"
            )

        return {
            "starting_value": self.starting_capital,
            "current_value": round(current_value, 2),
            "daily_start_value": round(daily_start, 2),
            "weekly_start_value": round(weekly_start, 2),
            "daily_reset_date": daily_reset,
            "weekly_reset_date": weekly_reset,
            "open_positions": [
                {
                    "symbol": t["symbol"],
                    "direction": t["direction"],
                    "trade_id": t["id"],
                }
                for t in open_trades
            ],
            "halted": halted,
            "halt_reason": halt_reason,
        }

    def rebuild_setup_performance(self, trades: list) -> dict:
        """Rebuild per-setup metrics: win rate, avg R, expectancy."""
        closed = [t for t in trades if t.get("status") == "CLOSED"]
        if not closed:
            return {"setups": {}, "last_updated": datetime.now(UTC).isoformat()}

        # Group closed trades by setup type
        by_setup = {}
        for t in closed:
            setup = t.get("setup_type") or t.get("setup", "Unknown")
            by_setup.setdefault(setup, []).append(t)

        setups = {}
        for setup, setup_trades in by_setup.items():
            wins = [t for t in setup_trades if _net_pnl(t) > 0]
            losses = [t for t in setup_trades if _net_pnl(t) < 0]
            breakeven = [t for t in setup_trades if _net_pnl(t) == 0]

            total = len(setup_trades)
            win_rate = len(wins) / total if total > 0 else 0

            # Collect R-multiples
            r_multiples = []
            for t in setup_trades:
                r = t.get("r_multiple")
                if r is not None and r != 0:
                    r_multiples.append(round(r, 2))
                else:
                    entry = t.get("entry_price", 0)
                    stop = t.get("original_stop") or t.get("stop_price", 0)
                    risk_per_unit = abs(entry - stop) if entry and stop else 0
                    pnl = t.get("pnl_usd", 0) or 0
                    size = t.get("original_size") or t.get("size", 0) or 0
                    if risk_per_unit > 0 and size > 0:
                        r_multiples.append(round(pnl / (risk_per_unit * size), 2))

            avg_r = (
                round(sum(r_multiples) / len(r_multiples), 2) if r_multiples else 0
            )
            avg_win_r = 0
            avg_loss_r = 0
            win_rs = [r for r in r_multiples if r > 0]
            loss_rs = [r for r in r_multiples if r < 0]
            if win_rs:
                avg_win_r = round(sum(win_rs) / len(win_rs), 2)
            if loss_rs:
                avg_loss_r = round(sum(loss_rs) / len(loss_rs), 2)

            # Expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
            expectancy = round(
                (win_rate * avg_win_r) + ((1 - win_rate) * avg_loss_r), 2
            )

            total_pnl = sum(_net_pnl(t) for t in setup_trades)

            setups[setup] = {
                "total_trades": total,
                "wins": len(wins),
                "losses": len(losses),
                "breakeven": len(breakeven),
                "win_rate": round(win_rate * 100, 1),
                "avg_r": avg_r,
                "avg_win_r": avg_win_r,
                "avg_loss_r": avg_loss_r,
                "expectancy": expectancy,
                "total_pnl": round(total_pnl, 2),
                "trades": [t.get("id") for t in setup_trades],
            }

        # Sort by expectancy descending
        sorted_setups = dict(
            sorted(setups.items(), key=lambda x: x[1]["expectancy"], reverse=True)
        )

        return {
            "setups": sorted_setups,
            "total_closed": len(closed),
            "last_updated": datetime.now(UTC).isoformat(),
        }

    def _calculate_portfolio_heat(self, trades: list, equity: float) -> dict:
        """Calculate total portfolio risk exposure (% of equity at risk)."""
        open_trades = [t for t in trades if t.get("status") == "OPEN"]
        if not open_trades or equity <= 0:
            return {"heat_pct": 0.0, "heat_per_position": []}

        per_position = []
        total_risk_usd = 0

        for t in open_trades:
            entry = t.get("entry_price", 0)
            stop = t.get("original_stop") or t.get("stop_price", 0)
            size = t.get("size", 0)

            if entry <= 0 or size <= 0:
                continue

            risk_per_unit = abs(entry - stop)
            risk_usd = risk_per_unit * size
            risk_pct = round((risk_usd / equity) * 100, 2)
            total_risk_usd += risk_usd

            per_position.append(
                {
                    "trade_id": t["id"],
                    "symbol": t["symbol"],
                    "risk_pct": risk_pct,
                }
            )

        heat_pct = round((total_risk_usd / equity) * 100, 2)
        return {"heat_pct": heat_pct, "heat_per_position": per_position}

    def _calculate_streaks(self, trades: list) -> dict:
        """Track consecutive win/loss streaks from closed trades."""
        closed = [t for t in trades if t.get("status") == "CLOSED"]
        if not closed:
            return {
                "current_streak": 0,
                "max_win_streak": 0,
                "max_loss_streak": 0,
                "last_5_results": [],
            }

        # Sort by close time ascending
        closed_sorted = sorted(closed, key=lambda t: t.get("exit_time", ""))

        results = []
        for t in closed_sorted:
            pnl = _net_pnl(t)
            results.append("W" if pnl > 0 else ("BE" if pnl == 0 else "L"))

        # Walk results to find streaks
        max_win = 0
        max_loss = 0
        streak = 0
        prev = None

        for r in results:
            if r == "BE":
                continue  # Breakeven doesn't extend or break streaks
            if r == prev:
                streak += 1 if r == "W" else -1
            else:
                streak = 1 if r == "W" else -1
            prev = r
            if streak > 0:
                max_win = max(max_win, streak)
            else:
                max_loss = max(max_loss, abs(streak))

        return {
            "current_streak": streak,
            "max_win_streak": max_win,
            "max_loss_streak": max_loss,
            "last_5_results": results[-5:],
        }

    def _check_invariants(self, trades: list, portfolio: dict, equity: dict):
        """Post-reconcile invariant checks. Log violations, never block."""
        violations = []

        open_trades = [t for t in trades if t.get("status") == "OPEN"]
        closed_trades = [t for t in trades if t.get("status") == "CLOSED"]

        # INV-R1: Open count matches between trade log and portfolio
        portfolio_open = len(portfolio.get("open_positions", []))
        actual_open = len(open_trades)
        if portfolio_open != actual_open:
            violations.append(
                f"Open count mismatch: trade_log={actual_open}, "
                f"portfolio={portfolio_open}"
            )

        # INV-R2: Equity = starting_capital + realized_pnl (within rounding)
        realized_pnl = (
            sum(_net_pnl(t) for t in closed_trades) + _partial_exit_pnl(trades)
        )
        expected_equity = round(self.starting_capital + realized_pnl, 2)
        actual_equity = round(equity.get("current_equity", 0), 2)
        if abs(expected_equity - actual_equity) > 0.02:
            violations.append(
                f"Equity mismatch: expected ${expected_equity:,.2f}, "
                f"got ${actual_equity:,.2f}"
            )

        # INV-R3: No CLOSED trade with pnl_usd=None
        for t in closed_trades:
            if t.get("pnl_usd") is None:
                violations.append(f"{t.get('id', '???')}: CLOSED with pnl_usd=None")

        # INV-R4: No duplicate open symbols
        open_syms = [t["symbol"] for t in open_trades]
        if len(open_syms) != len(set(open_syms)):
            violations.append(f"Duplicate open symbols: {open_syms}")

        if violations:
            msg = "RECONCILER INVARIANT VIOLATIONS:\n" + "\n".join(
                f"- {v}" for v in violations
            )
            logger.warning(msg)
            if self.on_invariant_violation:
                try:
                    self.on_invariant_violation(violations)
                except Exception:
                    pass
