# Roadmap

Status of v1 and the planned path to a hardened, live-capable system. Ordered by
priority — each phase builds on the previous one. **Paper stays the default
throughout; live is only ever a deliberate, confirmed switch.**

## ✅ v1 — built

- Config (pydantic-settings: `.env` + `config.yaml`), paper-default, one-flag live.
- Indicators (SMA/RSI/ATR, Wilder), strategy (trend + RSI + ATR/trend exits).
- Risk layer (sizing/FX, caps, cooldown, kill-switch, cash buffer, buying power).
- Backtest engine (same strategy+risk, no look-ahead) with metrics + equity curve.
- Broker (`ib_async`), execution (limit/MOO/MOC/MKT, fill reconciliation, orphan
  cleanup), SQLite audit trail, daily scheduler, orchestration agent.
- Tests for indicators/strategy/risk (33 passing) and SessionStart hook.

---

## Phase 1 — Validate on real data (do this before any paper money matters)

Goal: trust the numbers before trusting the agent.

1. **Pull real history into the cache.** Run the agent once against the paper
   gateway (it caches every fetch) or import CSVs into `data/cache/`.
2. **Backtest the v1 universe on real bars**, then sweep parameters
   (`sma_trend`, `rsi_entry/exit`, `atr_mult`, `cooldown_days`) — but guard
   against overfitting (out-of-sample / walk-forward split).
3. **Add backtest realism:** per-share commission + slippage model already
   stubbed in `Backtester`; calibrate to IBKR's actual fees. Add a benchmark
   (buy-and-hold SPY) and report alpha.
4. **Backtest regression test:** lock a known result on a fixed CSV fixture so
   future refactors can't silently change strategy behaviour.

## Phase 2 — Paper hardening (run it for weeks, watch it)

1. **End-to-end paper soak test** for several weeks; reconcile the SQLite store
   against IBKR statements daily.
2. **Partial-fill & working-order lifecycle:** currently a resting limit is
   reconciled on the *next* run. Add intra-run polling / `ib.openTrades()`
   follow-up and a timeout→cancel/replace policy.
3. **Notifications/alerting:** push kill-switch trips, rejected entries, errors,
   and the daily summary to email/Telegram/Slack (the `events` table is the
   source).
4. **Daily P&L baseline fix:** kill-switch `daily_pnl` is derived from the last
   stored equity point. Anchor it to the session open (broker `DailyPnL`
   subscription) for a true intraday figure.
5. **Health checks:** detect stale data (last bar too old), gateway disconnects,
   and clock/timezone drift; refuse to trade on stale inputs.

## Phase 3 — Risk & correctness deepening

1. **FX correctness:** verify account-currency vs. base-currency handling when
   the IBKR account currency ≠ `base_currency`; convert equity/buying-power
   consistently. Add tests with a mocked broker.
2. **Tick-size rounding** via IBKR contract details (the current `round(px, 2)`
   is a US-equity assumption; EU/Xetra differ).
3. **Per-position stop orders at the broker** (optional): mirror the ATR stop as
   a resting stop order so protection survives an agent/gateway outage.
4. **Position-level risk caps:** max exposure per sector/currency, portfolio
   heat, correlation limits.
5. **Broker integration tests** with a fake `IB` (mock `ib_async`) so
   broker/execution/agent get coverage without a live gateway.

## Phase 4 — Operations & deployment

1. **Containerize** (Docker) the agent + IB Gateway (e.g. `ib-gateway-docker`)
   with auto-restart and 2FA handling for unattended runs.
2. **Run as a service** (systemd/supervisor) with log rotation and the WAL
   SQLite DB on durable storage; scheduled DB backups.
3. **CI** (GitHub Actions): run `pytest` on every push; optionally add
   `ruff`/`black`/`mypy` (none configured yet) and wire them into the
   SessionStart hook.
4. **Secrets management** beyond `.env` for production (vault / encrypted env).
5. **Dashboard:** small read-only view over the SQLite tables (equity curve,
   open positions, recent signals/events).

## Phase 5 — Strategy evolution (only after the above is solid)

1. **v2 EU/Xetra universe** (already in config) once multi-currency FX is proven.
2. **Volatility-scaled sizing** (size by ATR/risk-per-trade instead of fixed
   notional) and ATR-based initial stops.
3. **Additional regimes/filters** (e.g. market-breadth or index-trend gate to
   stand down in bear markets).
4. **Short side / pairs** — a much bigger risk surface; treat as a separate
   project with its own validation.

---

### Guiding principles (unchanged from v1)

- Correctness and risk control beat features.
- Backtest → paper → live, in that order; no logic divergence between them.
- No order bypasses the risk layer. No secrets in the repo. Paper is the default.
