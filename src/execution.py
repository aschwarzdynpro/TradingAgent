"""Order routing and fill handling.

Builds the configured order type (limit-with-offset by default; never a blind
market order), submits via the broker, records everything to the store, and
reconciles fills. On startup it cleans up orphaned/working orders so the agent
starts from a known state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .broker import IBKRBroker
from .config import AppConfig
from .log import get_logger
from .store import Store

log = get_logger(__name__)


@dataclass
class OrderResult:
    symbol: str
    action: str
    quantity: float
    order_type: str
    limit_price: float | None
    status: str
    broker_order_id: int | None
    filled: float = 0.0
    avg_fill_price: float | None = None


def _round_qty(qty: float, allow_fractional: bool) -> float:
    if not allow_fractional:
        return float(math.floor(qty))
    # IBKR fractional shares: keep a sane precision.
    return round(qty, 4)


def _round_price(price: float) -> float:
    # US equities trade in $0.01 increments above $1; good enough as a default.
    return round(price, 2)


class ExecutionEngine:
    def __init__(self, broker: IBKRBroker, cfg: AppConfig, store: Store):
        self.broker = broker
        self.cfg = cfg
        self.ex = cfg.cfg.execution
        self.store = store
        self.mode = cfg.env.mode

    # ── order construction ─────────────────────────────────────────────────────
    def _build_order(self, action: str, qty: float, ref_price: float):
        from ib_async import LimitOrder, MarketOrder, Order

        otype = self.ex.order_type
        if otype == "LMT":
            # Buy slightly above / sell slightly below last to improve fill odds
            # without crossing blindly.
            offset = self.ex.limit_offset_pct
            lmt = ref_price * (1 + offset) if action == "BUY" else ref_price * (1 - offset)
            lmt = _round_price(lmt)
            order = LimitOrder(action, qty, lmt, tif=self.ex.tif, outsideRth=self.ex.outside_rth)
            return order, lmt
        if otype == "MKT":
            return MarketOrder(action, qty), None
        if otype == "MOO":
            o = MarketOrder(action, qty)
            o.tif = "OPG"  # market-on-open
            return o, None
        if otype == "MOC":
            o = Order(action=action, totalQuantity=qty, orderType="MOC")
            return o, None
        raise ValueError(f"Unsupported order_type: {otype}")

    # ── submission ─────────────────────────────────────────────────────────────
    def submit(self, spec, action: str, quantity: float, ref_price: float,
               reason: str) -> OrderResult:
        qty = _round_qty(quantity, self.cfg.cfg.sizing.allow_fractional)
        if qty <= 0:
            log.warning("submit_skipped_zero_qty", symbol=spec.symbol, action=action)
            self.store.record_order(spec.symbol, action, self.ex.order_type, 0,
                                    None, self.ex.tif, "skipped_zero_qty", self.mode, reason=reason)
            return OrderResult(spec.symbol, action, 0, self.ex.order_type, None, "skipped_zero_qty", None)

        contract = self.broker.qualify(spec)
        order, lmt = self._build_order(action, qty, ref_price)

        order_id = self.store.record_order(
            spec.symbol, action, self.ex.order_type, qty, lmt, self.ex.tif,
            "submitted", self.mode, reason=reason,
        )
        trade = self.broker.ib.placeOrder(contract, order)
        log.info("order_submitted", symbol=spec.symbol, action=action, qty=qty,
                 type=self.ex.order_type, limit=lmt, mode=self.mode, reason=reason)

        # Give the order a moment to ack / (for marketable orders) fill.
        self.broker.ib.sleep(3.0)
        return self._reconcile(order_id, spec.symbol, action, qty, lmt, trade)

    def _reconcile(self, order_id: int, symbol: str, action: str, qty: float,
                   lmt: float | None, trade) -> OrderResult:
        status = (trade.orderStatus.status or "submitted").lower()
        filled = float(trade.orderStatus.filled or 0.0)
        avg_px = float(trade.orderStatus.avgFillPrice or 0.0) or None
        broker_id = trade.order.orderId

        self.store.update_order_status(order_id, status, broker_order_id=broker_id)
        for f in trade.fills:
            self.store.record_fill(
                symbol, action, float(f.execution.shares), float(f.execution.price),
                commission=getattr(getattr(f, "commissionReport", None), "commission", None),
                broker_order_id=broker_id, broker_exec_id=f.execution.execId,
            )
        if filled > 0:
            log.info("order_fill", symbol=symbol, filled=filled, avg_price=avg_px, status=status)
        else:
            log.info("order_working", symbol=symbol, status=status)
        return OrderResult(symbol, action, qty, self.ex.order_type, lmt, status, broker_id,
                           filled=filled, avg_fill_price=avg_px)

    # ── startup hygiene ─────────────────────────────────────────────────────────
    def cleanup_orphan_orders(self) -> int:
        """Cancel any working orders left over from a previous run."""
        cancelled = 0
        for trade in self.broker.open_trades():
            try:
                self.broker.ib.cancelOrder(trade.order)
                cancelled += 1
                log.warning("orphan_order_cancelled", symbol=trade.contract.symbol,
                            order_id=trade.order.orderId)
            except Exception as e:  # pragma: no cover - network dependent
                log.error("orphan_cancel_failed", error=str(e))
        if cancelled:
            self.store.record_event("ORPHAN_CLEANUP", f"cancelled {cancelled} working orders",
                                    level="WARNING")
        return cancelled
