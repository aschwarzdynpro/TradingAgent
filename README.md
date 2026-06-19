# IBKR Trading Agent

An autonomous, **paper-first** trading agent for **Interactive Brokers (IBKR)**.
It trades a combined strategy — **trend regime filter + RSI mean-reversion entry +
ATR/trend exits** — on **daily bars**, long-only in v1.

> ⚠️ **Real money is involved.** Correctness and risk control come before
> features. The system defaults to **paper**. Going live is a single,
> documented flag change *plus* an explicit confirmation at startup. You cannot
> go live by accident.

---

## Table of contents

1. [How it works](#how-it-works)
2. [Strategy](#strategy)
3. [Risk layer](#risk-layer)
4. [Project layout](#project-layout)
5. [Setup](#setup)
6. [IB Gateway / TWS configuration](#ib-gateway--tws-configuration)
7. [Configuration reference](#configuration-reference)
8. [Running it](#running-it) — backtest → paper → live
9. [Going live (the one-flag switch)](#going-live-the-one-flag-switch)
10. [State & audit trail](#state--audit-trail)
11. [Tests](#tests)

---

## How it works

```
Scheduler (once per day, Europe/Berlin)
   → DataProvider     (completed daily bars per symbol)
   → Strategy         (ENTER_LONG / EXIT / HOLD)
   → RiskManager      (sizing + caps + kill-switch → approved orders)
   → ExecutionEngine  (order to IBKR; default limit-with-offset)
   → Store + Logger   (full SQLite audit trail)
```

The agent runs **once daily** (default 23:30 Europe/Berlin, after the US close)
on **completed** daily bars and places orders for the next session. The same
strategy and risk code paths are used by the backtest, paper and live — there is
deliberately **no logic divergence** between them.

## Strategy

Mean-reversion and momentum are decoupled by a regime filter: the trend sets the
direction, RSI sets the timing. **Long only, no shorting in v1.**

**Entry** (all required):
- Trend filter: `Close > SMA(200)` (uptrend regime)
- Timing: `RSI(14) < 30` (oversold pullback within the uptrend)
- No existing position in the symbol

**Exit** (any one):
- `RSI(14) > 55` (mean-reversion reached), **or**
- ATR trailing stop: price falls `k × ATR(14)` below the highest high since
  entry (`k = 3.0`), **or**
- Trend break: `Close < SMA(200)`

All thresholds are **parameters in `config.yaml`** (`sma_trend`, `rsi_period`,
`rsi_entry`, `rsi_exit`, `atr_period`, `atr_mult`). They are deliberate
**placeholders to calibrate in the backtest** — not optimised values. RSI and
ATR use Wilder's smoothing.

## Risk layer

Every candidate order passes through the `RiskManager`; nothing reaches execution
without it.

- **Position sizing:** quantity = `per_trade_notional` (converted to the
  instrument's currency via the live FX rate) / current price. With
  `allow_fractional: false` the quantity is floored; if that rounds to 0 shares,
  the trade is **skipped and logged**.
- **Max open positions:** default `12` (configurable).
- **One trade per symbol:** no pyramiding in v1.
- **Symbol cooldown:** no re-entry for `cooldown_days` (default 3) after an exit
  (anti-whipsaw).
- **Daily-loss kill-switch:** if intraday P&L drops below `-3%` of equity → **no
  new entries** for the rest of the day + an event is logged. Auto-flatten is
  **off by default** (whipsaw risk) and only available behind
  `auto_flatten_on_kill: true`.
- **Cash buffer + buying-power check:** keeps `cash_buffer_pct` of equity in cash
  and checks every order against the real IBKR buying power before submitting.

## Project layout

```
trading-agent/
├── config.yaml          # strategy/risk/sizing params + universe (NO secrets)
├── .env.example         # connection + LIVE_TRADING flag (copy to .env)
├── pyproject.toml
├── src/
│   ├── config.py        # pydantic-settings: .env + config.yaml
│   ├── broker.py        # ib_async wrapper: connect, qualify, data, orders, FX
│   ├── data.py          # daily bars, CSV cache, synthetic generator
│   ├── indicators.py    # SMA, RSI, ATR (pure, Wilder-smoothed, tested)
│   ├── strategy.py      # signal logic (ENTER_LONG / EXIT / HOLD)
│   ├── risk.py          # RiskManager: sizing, caps, cooldown, kill-switch
│   ├── execution.py     # order routing, order types, fills, orphan cleanup
│   ├── store.py         # SQLite: signals/orders/fills/positions/equity/events
│   ├── scheduler.py     # daily run (APScheduler, Europe/Berlin)
│   ├── backtest.py      # backtest engine (same strategy + risk as live)
│   └── agent.py         # orchestration loop
├── tests/               # test_indicators / test_strategy / test_risk
└── README.md
```

## Setup

Requires **Python 3.11+**.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .            # or: pip install -e ".[dev]" for pytest

cp .env.example .env        # then edit .env
```

`.env` and the SQLite DB are gitignored — **no secrets or account numbers are
ever committed.**

## IB Gateway / TWS configuration

The agent connects to a locally running **IB Gateway** (recommended) or **TWS**.

1. Install and log in to IB Gateway. Use your **paper** credentials first.
2. **Configure → Settings → API → Settings**:
   - Enable **"Enable ActiveX and Socket Clients"**.
   - **Socket port**:
     - **4002 = IB Gateway PAPER** (default for this agent)
     - **4001 = IB Gateway LIVE**
     - (TWS alternatives: 7497 paper / 7496 live — set these in `.env` if you use TWS)
   - Add `127.0.0.1` to **Trusted IPs**.
   - Leave **"Read-Only API"** unchecked (the agent needs to place orders).
3. For unattended operation, set the auto-logoff / auto-restart settings so the
   gateway stays connected when the scheduler fires.
4. (Optional, EU/Phase-2 universe) enable the relevant market-data subscriptions.

Set the matching values in `.env`:

```ini
LIVE_TRADING=false
IB_HOST=127.0.0.1
IB_PAPER_PORT=4002
IB_LIVE_PORT=4001
IB_CLIENT_ID=17
IB_ACCOUNT=            # optional, e.g. DU1234567 for paper
```

The agent picks the port automatically: `4002` while `LIVE_TRADING=false`,
`4001` once it is `true`.

## Configuration reference

All strategy/risk parameters live in **`config.yaml`** (never hard-coded):

| Section | Key | Meaning |
|---|---|---|
| `account` | `base_currency` | Reporting/sizing base currency (e.g. EUR) |
| | `total_capital` | Informational; real equity comes from IBKR |
| `sizing` | `per_trade_notional` | Target size per trade, in base currency |
| | `allow_fractional` | Use IBKR fractional shares; else floor (0 → skip) |
| `strategy` | `timeframe` | Bar size (`1 day`) |
| | `sma_trend` | Trend SMA period (200) |
| | `rsi_period` / `rsi_entry` / `rsi_exit` | RSI period (14), entry (<30), exit (>55) |
| | `atr_period` / `atr_mult` | ATR period (14), trailing-stop multiple k (3.0) |
| `risk` | `max_open_positions` | Concurrent position cap (12) |
| | `one_trade_per_symbol` | No pyramiding (true) |
| | `cooldown_days` | No re-entry for N days after exit (3) |
| | `daily_loss_limit_pct` | Kill-switch threshold (0.03 = -3%) |
| | `auto_flatten_on_kill` | Flatten on kill-switch (default false) |
| | `cash_buffer_pct` | Minimum cash kept (0.05 = 5%) |
| `execution` | `order_type` | `LMT` (default), `MOO`, `MOC`, `MKT` |
| | `limit_offset_pct` | Limit offset from last price (0.002 = 0.2%) |
| | `tif` / `outside_rth` | Time-in-force; allow fills outside RTH |
| `scheduler` | `timezone` / `run_time` | When to run (Europe/Berlin, 23:30) |
| | `run_on_weekends` | Skip Sat/Sun (false) |
| `data` | `history_duration` / `what_to_show` / `use_rth` | History request settings |
| | `cache_dir` | CSV cache location |
| `universe` | `active_universe` | Which named universe is live (`v1_us_usd`) |

Universes are editable lists of `{symbol, exchange, currency, primaryExchange}`.
v1 is US large caps (USD); v2 is EU/Xetra (EUR, `primaryExchange: IBIS`).

## Running it

Build order is **backtest → paper → live**. Validate the strategy before any
real order exists.

### 1. Backtest (validate the strategy)

```bash
# From cached CSV bars in data/cache/<SYMBOL>.csv:
python -m src.backtest

# Offline demo without a gateway (synthetic data — NOT real prices):
python -m src.backtest --synthetic
```

Outputs CAGR, max drawdown, Sharpe, win rate, #trades, avg hold, an equity-curve
CSV (`data/backtest_equity.csv`) and a trades CSV. The backtest fills a signal
from day *t*'s close at day *t+1*'s open, so there is no look-ahead. It runs in a
single currency (the active universe's currency).

> To backtest on real data, first populate `data/cache/` — e.g. run the agent
> against the paper gateway once (it caches every fetch), or drop in CSVs with
> columns `date,open,high,low,close,volume`.

### 2. Paper run (one cycle)

With IB Gateway running on the **paper** port (4002) and `LIVE_TRADING=false`:

```bash
python -m src.agent --once
```

This connects, reconciles positions, cleans up orphan orders, evaluates signals,
runs risk checks, places **paper** orders, and writes the full audit trail to
SQLite.

### 3. Paper run (scheduled, daily)

```bash
python -m src.agent --schedule
```

Runs every weekday at `scheduler.run_time` (Europe/Berlin). Keep IB Gateway
logged in.

## Going live (the one-flag switch)

Once you are satisfied with paper results:

1. In `.env`, set:
   ```ini
   LIVE_TRADING=true
   ```
   (Ensure `IB_LIVE_PORT=4001` and that IB Gateway is logged in with **live**
   credentials on that port.)
2. Start the agent. It will **require an explicit confirmation**:
   ```
   Type 'GO LIVE' exactly to proceed (anything else aborts):
   ```
   Only the exact string `GO LIVE` proceeds; anything else aborts.

That is the entire switch: **one flag + one confirmation**. The repository
default stays paper. The port is derived automatically from the flag, so paper
and live never get crossed.

## State & audit trail

Everything is persisted to SQLite (`DB_PATH`, default
`data/trading_agent.sqlite`, gitignored). Tables:

- `signals` — every evaluation (signal, reason, price, indicator snapshot)
- `orders` — every order with type, qty, limit, status, mode
- `fills` — executions with price/commission
- `positions` — current open positions (incl. trailing high)
- `closed_positions` — realized trades with P&L
- `equity_curve` — equity over time
- `events` — connects, kill-switch trips, orphan cleanups, errors, …

For an autonomous real-money system the audit trail is how you reconstruct *why*
every position exists — every decision is recorded.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

- `test_indicators.py` — SMA/RSI/ATR vs known values (incl. the canonical Wilder
  RSI reference series).
- `test_strategy.py` — entry/exit conditions on synthetic bars, including edge
  cases (RSI exactly at the threshold, trend break on the entry day).
- `test_risk.py` — sizing/rounding, max-positions cap, one-per-symbol, cooldown,
  kill-switch, cash buffer, buying-power.
```
