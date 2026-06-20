"""IBKR connectivity wrapper around ``ib_async``.

Responsibilities: connect (with the paper/live port chosen from config and a
mandatory live confirmation), qualify contracts, read account values / positions,
fetch a current price, and submit/track orders.

Safety: connecting to the LIVE port (env.live_trading=True) requires an explicit
confirmation callback to return True. There is no way to go live by accident.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .config import AppConfig, SymbolSpec
from .log import get_logger

log = get_logger(__name__)


@dataclass
class AccountValues:
    equity: float          # NetLiquidation
    buying_power: float     # BuyingPower / AvailableFunds
    cash: float             # TotalCashValue
    currency: str


class BrokerError(RuntimeError):
    pass


class IBKRBroker:
    def __init__(self, cfg: AppConfig, confirm_live: Callable[[], bool] | None = None):
        # Import here so the rest of the package (config, indicators, strategy,
        # backtest, tests) does not require ib_async to be installed.
        from ib_async import IB

        self.cfg = cfg
        self.env = cfg.env
        self.ib = IB()
        self._confirm_live = confirm_live
        self._contracts: dict[str, object] = {}

    # ── connection ───────────────────────────────────────────────────────────
    def connect(self, readonly: bool = False) -> None:
        """Connect to the gateway. ``readonly=True`` connects without order/account
        write privileges — use it for pure data work (e.g. the cache fetch) and
        against a gateway whose API is configured "Read-Only"."""
        env = self.env
        if env.live_trading:
            if self._confirm_live is None or not self._confirm_live():
                raise BrokerError(
                    "LIVE trading requested but not confirmed. Refusing to connect "
                    "to the live port. (Default is paper.)"
                )
            log.warning("connecting_live", port=env.port, host=env.host if hasattr(env, "host") else env.ib_host)
        else:
            log.info("connecting_paper", host=env.ib_host, port=env.port, readonly=readonly)

        self.ib.connect(
            host=env.ib_host,
            port=env.port,
            clientId=env.ib_client_id,
            timeout=env.ib_timeout,
            readonly=readonly,
        )
        log.info("connected", mode=env.mode, server_version=self.ib.client.serverVersion())

    def disconnect(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()
            log.info("disconnected")

    @property
    def connected(self) -> bool:
        return self.ib.isConnected()

    # ── contracts ─────────────────────────────────────────────────────────────
    def qualify(self, spec: SymbolSpec):
        from ib_async import Stock

        if spec.symbol in self._contracts:
            return self._contracts[spec.symbol]
        contract = Stock(
            symbol=spec.symbol,
            exchange=spec.exchange,
            currency=spec.currency,
            primaryExchange=spec.primary_exchange or "",
        )
        qualified = self.ib.qualifyContracts(contract)
        if not qualified:
            raise BrokerError(f"Could not qualify contract for {spec.symbol}")
        self._contracts[spec.symbol] = qualified[0]
        return qualified[0]

    # ── account ───────────────────────────────────────────────────────────────
    def account_values(self) -> AccountValues:
        rows = self.ib.accountSummary(self.env.ib_account or "")
        tags = {r.tag: r for r in rows if (not self.env.ib_account or r.account == self.env.ib_account)}

        def _val(tag: str, default: float = 0.0) -> float:
            r = tags.get(tag)
            try:
                return float(r.value) if r else default
            except (TypeError, ValueError):
                return default

        currency = tags["NetLiquidation"].currency if "NetLiquidation" in tags else self.cfg.cfg.account.base_currency
        # Prefer BuyingPower; fall back to AvailableFunds.
        bp = _val("BuyingPower") or _val("AvailableFunds")
        return AccountValues(
            equity=_val("NetLiquidation"),
            buying_power=bp,
            cash=_val("TotalCashValue"),
            currency=currency,
        )

    def positions(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for p in self.ib.positions(self.env.ib_account or ""):
            sym = p.contract.symbol
            out[sym] = {
                "symbol": sym,
                "quantity": float(p.position),
                "avg_price": float(p.avgCost),
                "currency": p.contract.currency,
                "contract": p.contract,
            }
        return out

    def open_orders(self) -> list:
        return list(self.ib.openOrders())

    def open_trades(self) -> list:
        return list(self.ib.openTrades())

    # ── market data ────────────────────────────────────────────────────────────
    def last_price(self, contract) -> float | None:
        """Best-effort current price: snapshot ticker, else last daily close."""
        ticker = self.ib.reqMktData(contract, "", snapshot=True, regulatorySnapshot=False)
        self.ib.sleep(2.0)
        for px in (ticker.last, ticker.close, ticker.marketPrice()):
            if px and px == px and px > 0:  # not NaN, positive
                self.ib.cancelMktData(contract)
                return float(px)
        self.ib.cancelMktData(contract)
        # Fallback: last completed daily bar close.
        from .data import fetch_history

        df = fetch_history(self.ib, contract, duration="5 D", bar_size="1 day")
        if len(df):
            return float(df["close"].iloc[-1])
        return None

    # ── FX ──────────────────────────────────────────────────────────────────────
    def fx_rate(self, base: str, quote: str) -> float:
        """Spot rate to convert ``base`` -> ``quote`` (1 base = X quote)."""
        if base == quote:
            return 1.0
        from ib_async import Forex

        pair = Forex(base + quote)
        try:
            self.ib.qualifyContracts(pair)
            ticker = self.ib.reqMktData(pair, "", snapshot=True)
            self.ib.sleep(2.0)
            px = ticker.marketPrice() or ticker.close or ticker.last
            self.ib.cancelMktData(pair)
            if px and px == px and px > 0:
                return float(px)
        except Exception as e:  # pragma: no cover - network dependent
            log.warning("fx_rate_failed", base=base, quote=quote, error=str(e))
        # Inverse pair fallback.
        try:
            inv = Forex(quote + base)
            self.ib.qualifyContracts(inv)
            ticker = self.ib.reqMktData(inv, "", snapshot=True)
            self.ib.sleep(2.0)
            px = ticker.marketPrice() or ticker.close or ticker.last
            self.ib.cancelMktData(inv)
            if px and px == px and px > 0:
                return 1.0 / float(px)
        except Exception:  # pragma: no cover
            pass
        log.warning("fx_rate_unavailable_default_1", base=base, quote=quote)
        return 1.0
