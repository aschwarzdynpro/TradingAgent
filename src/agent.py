"""Agent orchestration: Data -> Strategy -> Risk -> Execution -> Store/Log.

One ``run_once`` performs a full daily cycle on completed daily bars. The same
strategy and risk logic as the backtest is used. Live trading requires both
LIVE_TRADING=true in the environment AND an interactive confirmation at startup.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime

from .broker import AccountValues, IBKRBroker
from .config import AppConfig, SymbolSpec, load_config
from .data import get_bars
from .execution import ExecutionEngine
from .log import configure_logging, get_logger
from .risk import AccountSnapshot, RiskManager
from .store import Store, StoredPosition
from .strategy import PositionState, SignalType, compute_regime, evaluate

log = get_logger(__name__)


def confirm_live_interactive() -> bool:
    """Ask the operator to explicitly confirm LIVE trading."""
    print("\n" + "!" * 64)
    print(" LIVE TRADING IS ENABLED (LIVE_TRADING=true).")
    print(" This will place orders with REAL money on the LIVE port.")
    print("!" * 64)
    try:
        answer = input(" Type 'GO LIVE' exactly to proceed (anything else aborts): ")
    except EOFError:
        return False
    return answer.strip() == "GO LIVE"


class TradingAgent:
    def __init__(self, cfg: AppConfig, broker: IBKRBroker | None = None):
        self.cfg = cfg
        self.store = Store(cfg.env.db_path)
        self.broker = broker or IBKRBroker(cfg, confirm_live=confirm_live_interactive)
        self.execution = ExecutionEngine(self.broker, cfg, self.store)
        self.rm = RiskManager(cfg.cfg.risk, cfg.cfg.sizing, cfg.cfg.account.base_currency)
        self._fx_cache: dict[str, float] = {}
        self._market_risk_on: bool | None = None

    # ── helpers ────────────────────────────────────────────────────────────────
    def _fx(self, instrument_ccy: str) -> float:
        base = self.cfg.cfg.account.base_currency
        if instrument_ccy == base:
            return 1.0
        if instrument_ccy not in self._fx_cache:
            self._fx_cache[instrument_ccy] = self.broker.fx_rate(base, instrument_ccy)
        return self._fx_cache[instrument_ccy]

    def _reconcile_positions(self) -> None:
        """Sync the store's position table with the broker's truth.

        - Broker position we don't track -> adopt it (entry today, trail = avg cost).
        - Tracked position the broker no longer has -> it was closed elsewhere.
        """
        broker_pos = self.broker.positions()
        stored = self.store.get_positions()

        for sym, bp in broker_pos.items():
            if bp["quantity"] == 0:
                continue
            if sym not in stored:
                log.warning("adopting_untracked_position", symbol=sym, qty=bp["quantity"])
                self.store.upsert_position(StoredPosition(
                    symbol=sym, quantity=bp["quantity"], avg_price=bp["avg_price"],
                    entry_date=date.today(), highest_high=bp["avg_price"],
                    currency=bp.get("currency"),
                ))
                self.store.record_event("ADOPT_POSITION", f"adopted untracked {sym}", level="WARNING")

        for sym, sp in stored.items():
            if sym not in broker_pos or broker_pos[sym]["quantity"] == 0:
                log.warning("tracked_position_gone_at_broker", symbol=sym)
                self.store.close_position(sym, sp.avg_price, date.today(), "closed_externally")
                self.store.record_event("EXTERNAL_CLOSE", f"{sym} gone at broker", level="WARNING")

    def _snapshot(self) -> tuple[AccountSnapshot, AccountValues]:
        av = self.broker.account_values()
        prev_equity = self._last_equity()
        daily_pnl = (av.equity - prev_equity) if prev_equity is not None else 0.0
        snap = AccountSnapshot(
            equity=av.equity,
            buying_power=av.buying_power,
            cash=av.cash,
            daily_pnl=daily_pnl,
            open_positions=set(self.store.get_positions().keys()),
            cooldowns=self.store.last_exit_dates(),
        )
        return snap, av

    def _last_equity(self) -> float | None:
        import sqlite3

        conn = sqlite3.connect(self.store.db_path)
        try:
            row = conn.execute(
                "SELECT equity FROM equity_curve ORDER BY ts DESC LIMIT 1"
            ).fetchone()
        finally:
            conn.close()
        return float(row[0]) if row else None

    def _compute_market_regime(self) -> bool | None:
        """Market risk-on/off from the regime symbol (its close vs SMA). Returns
        None when the filter is off or the regime can't be determined (inert)."""
        sp = self.cfg.cfg.strategy
        if not sp.use_regime_filter:
            return None
        try:
            contract = self.broker.qualify(SymbolSpec(symbol=sp.regime_symbol))
            df = get_bars(
                sp.regime_symbol, ib=self.broker.ib, contract=contract,
                cache_dir=self.cfg.cfg.data.cache_dir,
                duration=self.cfg.cfg.data.history_duration,
                what_to_show=self.cfg.cfg.data.what_to_show, use_rth=self.cfg.cfg.data.use_rth,
            )
            reg = compute_regime(df["close"], sp.regime_sma)
            if not len(reg):
                return None
            return bool(reg.iloc[-1])
        except Exception as e:
            log.warning("regime_unavailable_inert", symbol=sp.regime_symbol, error=str(e))
            return None

    # ── main cycle ─────────────────────────────────────────────────────────────
    def run_once(self) -> None:
        if not self.broker.connected:
            self.broker.connect()
        self.store.record_event("RUN_START", f"daily cycle {datetime.utcnow().isoformat()}",
                                payload={"mode": self.cfg.env.mode})

        self.execution.cleanup_orphan_orders()
        self._reconcile_positions()

        self._market_risk_on = self._compute_market_regime()
        if self.cfg.cfg.strategy.use_regime_filter:
            log.info("market_regime", risk_on=self._market_risk_on,
                     symbol=self.cfg.cfg.strategy.regime_symbol)
            if self._market_risk_on is False:
                self.store.record_event("REGIME_RISK_OFF",
                                        f"{self.cfg.cfg.strategy.regime_symbol} below SMA"
                                        f"{self.cfg.cfg.strategy.regime_sma}; entries blocked")

        snap, av = self._snapshot()
        self.store.record_equity(av.equity, cash=av.cash, mode=self.cfg.env.mode)
        log.info("account", equity=av.equity, buying_power=av.buying_power, cash=av.cash,
                 currency=av.currency, mode=self.cfg.env.mode)

        kill = self.rm.kill_switch_active(snap)
        if kill:
            log.error("kill_switch_active", daily_pnl=snap.daily_pnl, equity=snap.equity)
            self.store.record_event("KILL_SWITCH", "daily loss limit breached; no new entries",
                                    level="ERROR", payload={"daily_pnl": snap.daily_pnl})
            if self.cfg.cfg.risk.auto_flatten_on_kill:
                self._flatten_all("kill_switch_flatten")

        for spec in self.cfg.cfg.symbols:
            try:
                self._process_symbol(spec, snap, kill)
            except Exception as e:  # one bad symbol must not stop the whole run
                log.error("symbol_failed", symbol=spec.symbol, error=str(e))
                self.store.record_event("SYMBOL_ERROR", f"{spec.symbol}: {e}", level="ERROR")

        self.store.record_event("RUN_END", "daily cycle complete")
        log.info("run_complete", mode=self.cfg.env.mode)

    def _process_symbol(self, spec, snap: AccountSnapshot, kill: bool) -> None:
        contract = self.broker.qualify(spec)
        df = get_bars(
            spec.symbol, ib=self.broker.ib, contract=contract,
            cache_dir=self.cfg.cfg.data.cache_dir,
            duration=self.cfg.cfg.data.history_duration,
            what_to_show=self.cfg.cfg.data.what_to_show,
            use_rth=self.cfg.cfg.data.use_rth,
        )
        df.attrs["symbol"] = spec.symbol

        stored = self.store.get_position(spec.symbol)
        pstate = (
            PositionState(spec.symbol, stored.quantity, stored.avg_price,
                          stored.entry_date, stored.highest_high)
            if stored else None
        )
        res = evaluate(df, pstate, self.cfg.cfg.strategy, market_risk_on=self._market_risk_on)
        self.store.record_signal(res.asof, spec.symbol, res.signal.value, res.reason,
                                 res.price, res.indicators)

        # Keep the trailing high ratcheting even on HOLD.
        if stored and res.new_highest_high is not None:
            self.store.update_highest_high(spec.symbol, res.new_highest_high)

        if res.signal is SignalType.EXIT and stored:
            self._do_exit(spec, stored, res)
        elif res.signal is SignalType.ENTER_LONG and not stored:
            if kill:
                log.info("entry_blocked_kill_switch", symbol=spec.symbol)
                return
            self._do_entry(spec, res, snap)

    def _do_entry(self, spec, res, snap: AccountSnapshot) -> None:
        fx = self._fx(spec.currency)
        ref_price = self.broker.last_price(self.broker.qualify(spec)) or res.price
        decision = self.rm.evaluate_entry(spec.symbol, ref_price, res.asof, snap, fx_base_to_instrument=fx)
        if not decision.approved:
            log.info("entry_rejected", symbol=spec.symbol, reason=decision.reason)
            self.store.record_event("ENTRY_REJECTED", f"{spec.symbol}: {decision.reason}")
            return

        result = self.execution.submit(spec, "BUY", decision.quantity, ref_price, reason=res.reason)
        if result.filled > 0:
            self.store.upsert_position(StoredPosition(
                symbol=spec.symbol, quantity=result.filled,
                avg_price=result.avg_fill_price or ref_price, entry_date=res.asof,
                highest_high=res.indicators.get("close", ref_price), currency=spec.currency,
            ))
            # Reserve cash so subsequent same-run entries see reduced capacity.
            snap.open_positions.add(spec.symbol)
            snap.cash -= (result.avg_fill_price or ref_price) * result.filled / fx
        else:
            # Order is working (e.g. a resting limit). It will be reconciled next run.
            log.info("entry_order_working", symbol=spec.symbol)

    def _do_exit(self, spec, stored: StoredPosition, res) -> None:
        ref_price = self.broker.last_price(self.broker.qualify(spec)) or res.price
        result = self.execution.submit(spec, "SELL", stored.quantity, ref_price, reason=res.reason)
        if result.filled > 0:
            self.store.close_position(spec.symbol, result.avg_fill_price or ref_price,
                                      res.asof, res.reason)
            log.info("position_closed", symbol=spec.symbol, reason=res.reason)
        else:
            log.info("exit_order_working", symbol=spec.symbol)

    def _flatten_all(self, reason: str) -> None:
        for sym, sp in self.store.get_positions().items():
            spec = next((s for s in self.cfg.cfg.symbols if s.symbol == sym), None)
            if spec is None:
                continue
            ref = self.broker.last_price(self.broker.qualify(spec)) or sp.avg_price
            result = self.execution.submit(spec, "SELL", sp.quantity, ref, reason=reason)
            if result.filled > 0:
                self.store.close_position(sym, result.avg_fill_price or ref, date.today(), reason)

    def shutdown(self) -> None:
        self.broker.disconnect()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one daily cycle of the trading agent.")
    parser.add_argument("--once", action="store_true", help="run a single cycle and exit (default)")
    parser.add_argument("--schedule", action="store_true", help="run continuously on the daily schedule")
    args = parser.parse_args(argv)

    cfg = load_config()
    configure_logging(cfg.env.log_level)
    log.info("starting", mode=cfg.env.mode, host=cfg.env.ib_host, port=cfg.env.port,
             universe=cfg.cfg.active_universe)

    # Confirm live ONCE here; downstream brokers are then pre-confirmed so the
    # operator is not prompted again per run/per connection.
    if cfg.env.live_trading and not confirm_live_interactive():
        log.error("live_not_confirmed_aborting")
        return 2

    def confirm() -> bool:
        # Live already confirmed once above; downstream brokers are pre-approved.
        return True

    if args.schedule:
        from .scheduler import run_scheduler

        run_scheduler(cfg, confirm_live=confirm)
        return 0

    broker = IBKRBroker(cfg, confirm_live=confirm)
    agent = TradingAgent(cfg, broker=broker)
    try:
        agent.run_once()
    finally:
        agent.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
