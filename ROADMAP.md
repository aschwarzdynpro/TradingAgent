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
- Tests for indicators/strategy/risk/backtest (41 passing) and SessionStart hook.

---

## Phase 1 — Validate on real data (do this before any paper money matters)

Goal: trust the numbers before trusting the agent.

1. ✅ **Pull real history into the cache.** `python -m src.fetch` (fetch-only:
   read-only connect, no orders) caches the active universe + benchmark to
   `data/cache/`. `data.history_duration` raised `2 Y` → `15 Y` so the SMA-200
   warmup leaves enough live bars (3769 bars/symbol, 2011→2026).
2. ✅ **Backtest + walk-forward sweep on real bars.** `python -m src.sweep`
   (rolling train→test folds; params chosen in-sample, scored out-of-sample;
   warm-up-aware; min-trades guard; `--notional`/`--max-positions` overrides).
   **Verdict: v1 has no *competitive* edge, and is heavily overfit.** Coarse
   grid, 6 folds (2016→2025):
   - At the shipped `per_trade_notional: 500` (5% of €10k): stitched OOS
     **−1.22%** over ~9y, Sharpe −0.07. The strategy paid fees to tread water —
     the $1 min-commission is ~0.4% round-trip on a €500 notional.
   - **Deployment experiment** at €2500/trade (25%, min-commission now ~0.03%):
     stitched OOS flips to **+12.14%** (CAGR +1.28%, Sharpe 0.21, max DD −14%).
     So €500 was self-sabotage — *but* even sized properly it badly trails SPY
     (**+193.5%**, 12.7% CAGR) and overfits hard (mean IS +21.5% → OOS +2.0%).
   Conclusion: the €500 default is wrong (below cost-efficiency) and should be
   raised; sizing alone does **not** make v1 viable. **Do NOT run a paper soak on
   v1 as-is.** Realistic path is strategy rework (Phase 5), not more sweeping.
3. ✅ **Backtest realism:** IBKR-fixed cost model (`$0.005`/share, `$1` min,
   capped at 1% of trade value) + slippage, all in `config.yaml` under
   `backtest:`. Buy-and-hold benchmark (default SPY) with alpha (CAGR + total
   return) reported. *Next:* validate the fee assumption against a real IBKR
   statement and switch to tiered pricing if that is what the account uses.
4. ✅ **Backtest regression test:** locked metrics + full trade sequence on
   committed CSV fixtures (`tests/fixtures/cache/`) — see
   `tests/test_backtest.py`.

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
3. ✅ **CI** (GitHub Actions): `ruff check` + `pytest` on every push/PR across
   Python 3.11/3.12 (`.github/workflows/ci.yml`); `ruff` is in the `dev` extra
   so the SessionStart hook installs it too. *Optional next:* add `mypy`.
4. **Secrets management** beyond `.env` for production (vault / encrypted env).
5. **Dashboard:** small read-only view over the SQLite tables (equity curve,
   open positions, recent signals/events).

## Phase 5 — Strategy evolution (started — v1 thesis was not viable)

**Trend/momentum candidate (built).** A `strategy.mode: trend_momentum` was added
(enter when `Close > SMA(sma_trend)` AND 12-1 style momentum > 0; exit on trend
break / ATR trail; rides the trend, no RSI exit). Walk-forward (same harness,
€2500/trade, 2016→2025) vs mean-reversion: **OOS +66.5% / CAGR 5.83% / Sharpe
0.47 / max DD −29%**, vs MR's +12.1% / 1.28% / 0.21 / −14%. Far better, and
*parameter-stable* — every fold chose `mom126 sma200`, so the form is robust, not
a noise optimum. **But it still trails SPY** (+207% / 13.3% CAGR) and the −29% DD
exposes the missing bear-market filter. Net: the honest baseline to beat remains
"just hold SPY"; momentum is a real improvement but not yet compelling standalone.

**Regime filter (built).** `strategy.use_regime_filter` (+ `regime_symbol`,
`regime_sma`, `regime_exit`): no new entries while the regime symbol is below its
SMA; with `regime_exit`, open positions are also flattened. Wired through backtest,
sweep (`--regime`/`--regime-exit`) AND the live agent (no logic divergence).
Walk-forward (momentum, €2500, 2016→2025):
| variant | OOS total | CAGR | Sharpe | max DD |
|---|---|---|---|---|
| momentum, no regime | +66.5% | 5.83% | 0.47 | −29.1% |
| + regime entry-gate only | +46.3% | 4.32% | 0.39 | −34.7% |
| **+ regime_exit (flatten)** | +57.7% | 5.19% | **0.51** | **−17.7%** |
`regime_exit` is the winner — drawdown −29%→−17.7%, best Sharpe. Entry-gate-only is
a trap (worse): you must *exit* on risk-off, not just stop buying. Still trails SPY
(13.3% CAGR) on absolute return, but the risk profile is now much gentler.

Remaining levers:
1. **Volatility-scaled sizing** (ATR/risk-per-trade instead of fixed notional) +
   ATR-based initial stops. **Indicated next step.**
2. **v2 EU/Xetra universe** (already in config) once multi-currency FX is proven.
3. **Short side / pairs** — a much bigger risk surface; treat as a separate
   project with its own validation.

> Honest baseline check: even the best variant (Sharpe 0.51, −17.7% DD, ~5.2%
> CAGR) trails buy-and-hold SPY on return. Decide the mandate (beat SPY absolute
> vs. low-drawdown / uncorrelated stream) before investing more tuning.

---

### Guiding principles (unchanged from v1)

- Correctness and risk control beat features.
- Backtest → paper → live, in that order; no logic divergence between them.
- No order bypasses the risk layer. No secrets in the repo. Paper is the default.
