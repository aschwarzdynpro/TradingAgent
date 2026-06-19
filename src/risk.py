"""Risk layer — the mandatory gate between strategy and execution.

No order leaves the system without passing through here. Every check is pure and
driven by ``RiskConfig`` / ``SizingConfig`` (from YAML), so the exact same logic
runs in backtest and live. The component is deliberately stateless: callers pass
in an ``AccountSnapshot`` describing the current world (equity, buying power,
open positions, cooldowns, daily P&L) and get back a decision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

from .config import RiskConfig, SizingConfig


@dataclass
class AccountSnapshot:
    """Everything the risk layer needs to judge an order, at one point in time."""

    equity: float
    buying_power: float
    cash: float
    daily_pnl: float = 0.0
    open_positions: set[str] = field(default_factory=set)
    cooldowns: dict[str, date] = field(default_factory=dict)  # symbol -> last exit date


@dataclass
class RiskDecision:
    approved: bool
    quantity: float
    reason: str


def compute_quantity(
    notional_base: float,
    price: float,
    fx_base_to_instrument: float,
    allow_fractional: bool,
) -> float:
    """Translate a base-currency target notional into a share quantity.

    ``fx_base_to_instrument`` converts the base currency into the instrument's
    currency (e.g. EUR->USD ~ 1.08; use 1.0 when they match). With fractional
    shares disabled the quantity is floored; a floor to 0 means "skip".
    """
    if price <= 0 or fx_base_to_instrument <= 0:
        return 0.0
    notional_instrument = notional_base * fx_base_to_instrument
    qty = notional_instrument / price
    if not allow_fractional:
        qty = math.floor(qty)
    return float(qty)


class RiskManager:
    def __init__(self, risk: RiskConfig, sizing: SizingConfig, base_currency: str = "EUR"):
        self.risk = risk
        self.sizing = sizing
        self.base_currency = base_currency

    # ── kill switch ────────────────────────────────────────────────────────
    def kill_switch_active(self, snap: AccountSnapshot) -> bool:
        """True when today's P&L has breached the daily loss limit.

        While active, NO new entries are allowed (existing positions are left
        alone unless auto_flatten_on_kill is explicitly enabled elsewhere).
        """
        if snap.equity <= 0:
            return False
        return snap.daily_pnl < -abs(self.risk.daily_loss_limit_pct) * snap.equity

    def cooldown_active(self, symbol: str, asof: date, snap: AccountSnapshot) -> bool:
        last_exit = snap.cooldowns.get(symbol)
        if last_exit is None:
            return False
        return (asof - last_exit).days < self.risk.cooldown_days

    # ── entry approval ───────────────────────────────────────────────────────
    def evaluate_entry(
        self,
        symbol: str,
        price: float,
        asof: date,
        snap: AccountSnapshot,
        fx_base_to_instrument: float = 1.0,
    ) -> RiskDecision:
        """Run every entry guard; return an approved quantity or a rejection."""
        if self.kill_switch_active(snap):
            return RiskDecision(False, 0.0, "kill_switch_active")

        if self.risk.one_trade_per_symbol and symbol in snap.open_positions:
            return RiskDecision(False, 0.0, "already_in_position")

        if len(snap.open_positions) >= self.risk.max_open_positions:
            return RiskDecision(False, 0.0, "max_open_positions")

        if self.cooldown_active(symbol, asof, snap):
            return RiskDecision(False, 0.0, "cooldown_active")

        if price <= 0:
            return RiskDecision(False, 0.0, "invalid_price")

        qty = compute_quantity(
            self.sizing.per_trade_notional, price, fx_base_to_instrument, self.sizing.allow_fractional
        )
        if qty <= 0:
            return RiskDecision(False, 0.0, "quantity_rounds_to_zero")

        # Cost in instrument currency converted back to base for the cash/BP checks.
        cost_instrument = qty * price
        cost_base = cost_instrument / fx_base_to_instrument

        # Respect the cash buffer: never deploy the last slice of equity.
        min_cash = self.risk.cash_buffer_pct * snap.equity
        if snap.cash - cost_base < min_cash:
            return RiskDecision(False, 0.0, "cash_buffer_breached")

        # Buying-power check against the real broker figure (base currency).
        if cost_base > snap.buying_power:
            return RiskDecision(False, 0.0, "insufficient_buying_power")

        return RiskDecision(True, qty, "approved")
